"""
WorkbenchSession - the tuning session backend.

Owns a loaded InstrumentIR, applies parameter changes from any caller
(OSC surface, MIDI mapper, or the WS3 optimizer), debounce-renders
through the full harness (tools/render/render_preset.py), plays the
result, logs every change as a (parameters, verdict) trajectory, and
writes tuned values back to the preset YAML with a meta.version bump.
"""

import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from symcrash_python_ap.primitives import Q15_ONE

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT / "src"), str(REPO_ROOT / "tools" / "render")):
    if p not in sys.path:
        sys.path.insert(0, p)

from render_preset import render_preset_note  # noqa: E402
from symcrash_python_ap.instrument_ir import load_instrument  # noqa: E402
from symcrash_python_ap.vm.harmonic_bank import clear_basis_atlas  # noqa: E402

from .params import build_registry  # noqa: E402
from .patchboard import PatchboardController  # noqa: E402

FS = 48000

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - playback is best-effort
    sd = None


class WorkbenchSession:
    """One preset, one tuning session."""

    def __init__(self, preset_path, debounce_ms: float = 180.0,
                 autoplay: bool = True, log_dir: Optional[Path] = None):
        self.preset_path = Path(preset_path)
        self.ir = load_instrument(str(self.preset_path))
        self._source_fingerprint = self._fingerprint(
            self.preset_path.read_bytes())
        self.registry = build_registry()
        self.changed: dict = {}
        self.patchboard = PatchboardController(self)

        # Performance state
        self.note = 60
        self.velocity = 0.85
        # 0.8 s keeps the edit->ear loop ~3 s (render is ~4-5x realtime,
        # pure-Python block loop); raise via the DUR knob for long tails
        self.duration = 0.8
        self.autoplay = autoplay
        self.debounce = debounce_ms / 1000.0
        # True while a Phase C streaming voice is live: suppresses the
        # debounce re-render (the stream itself is the audition path).
        self.streaming = False
        # Called with the session after a successful load_preset() so
        # transports can push_state / notify the surface.
        self.on_preset_loaded: Optional[Callable] = None
        # Called after any structural edit (including rejected edits, whose
        # cords need to snap back), SAVE, RELOAD, or source replacement.
        self.on_state_changed: Optional[Callable] = None

        # Trajectory log (calibration data for the metric committee)
        self._log_dir = Path(log_dir) if log_dir else REPO_ROOT / "_workbench_logs"
        self._log_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._log_path = self._log_dir / f"{self.preset_path.stem}-{stamp}.jsonl"

        self._dirty = threading.Event()
        self._stop = threading.Event()
        self._last_change = time.monotonic()
        self._render_lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        self._log("session_start", preset=str(self.preset_path))

    # ------------------------------------------------------------- handling

    def handle_message(self, address: str, *args) -> bool:
        """Apply a complete OSC message, preserving multi-argument payloads.

        Ordinary knobs remain scalar and flow through :meth:`handle`.
        Patchbay widgets send ``input, output...`` and must arrive intact.
        """
        if self.patchboard.handles(address):
            ok = self.patchboard.handle(address, args)
            if ok and address != "/editor/resync":
                self._log("structure_set", address=address,
                          args=[str(arg) for arg in args])
                self._touch(address, list(args))
            # Re-send even on rejection so an illegal cord snaps back to the
            # last accepted structure and the diagnostic becomes visible.
            self._notify_state_changed()
            return ok
        return self.handle(address, args[0] if args else None)

    def handle(self, address: str, value) -> bool:
        """Apply one control message. Returns True if it changed anything."""
        # String-valued messages: route before the float coercion below.
        if address == "/preset/load":
            return self.load_preset(value) if value else False
        if address == "/preset/list":
            # The reply needs a feedback channel: the OSC transports
            # (osc_server.serve / StreamingOSCBridge) answer this one.
            return True

        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                return False

        if address == "/play":
            if value is None or value >= 0.5:
                threading.Thread(target=self.render_and_play,
                                 daemon=True).start()
            return True
        if address == "/note":
            self.note = int(round(value))
            self._touch("note", self.note)
            return True
        if address == "/velocity":
            self.velocity = max(0.0, min(1.0, value))
            self._touch("velocity", self.velocity)
            return True
        if address == "/duration":
            self.duration = max(0.1, min(8.0, value))
            self._touch("duration", self.duration)
            return True
        if address == "/autoplay":
            self.autoplay = value >= 0.5
            return True
        if address == "/save":
            if value is None or value >= 0.5:
                self.save()
            return True
        if address == "/reload":
            if value is None or value >= 0.5:
                self.reload()
            return True

        param = self.registry.get(address)
        if param is None:
            print(f"[workbench] unknown address: {address}")
            return False
        if not param.set(self.ir, value):
            print(f"[workbench] {address}: preset has no such target")
            return False
        was_dirty = bool(self.changed) or self.patchboard.dirty.any
        self.changed[address] = value
        self._log("set", address=address, value=value)
        self._touch(address, value)
        if not was_dirty or address in {"/fm/algorithm", "/fm/feedback"}:
            self._notify_state_changed()
        return True

    def _touch(self, what, value):
        self._last_change = time.monotonic()
        self._dirty.set()

    # ------------------------------------------------------------- rendering

    def _run(self):
        while not self._stop.is_set():
            if not self._dirty.wait(0.1):
                continue
            while (time.monotonic() - self._last_change) < self.debounce:
                time.sleep(0.02)
            self._dirty.clear()
            if self.autoplay and not self.streaming:
                self.render_and_play()

    def render(self) -> np.ndarray:
        from .system_axes import SystemAxisRuntime

        with self._render_lock:
            clear_basis_atlas()
            t0 = time.monotonic()
            system_axes = SystemAxisRuntime(self.ir)
            samples = render_preset_note(
                self.ir, note=self.note, duration_sec=self.duration,
                velocity_q15=int(self.velocity * Q15_ONE),
                frame_transform=lambda frame: system_axes.apply(
                    frame, self.ir))
            dt = time.monotonic() - t0
        samples = np.asarray(samples, dtype=np.int16)
        self._log("render", note=self.note, velocity=self.velocity,
                  duration=self.duration, render_sec=round(dt, 3))
        self._write_wav(samples)
        return samples

    def stream(self, note: Optional[int] = None,
               velocity: Optional[float] = None):
        """
        Create a Phase C streaming voice bound to this session: a
        sustained note rendered block-by-block via vm_step_block with
        per-block parameter latch and atlas-mode axis sweeps.
        """
        from .streaming import StreamingVoice
        return StreamingVoice(self, note=note, velocity=velocity)

    def render_and_play(self):
        try:
            samples = self.render()
        except Exception as e:  # keep the OSC loop alive on render bugs
            print(f"[workbench] render failed: {e}")
            self._log("render_error", error=str(e))
            return
        if sd is not None:
            try:
                sd.stop()
                sd.play(samples, FS)
            except Exception as e:
                print(f"[workbench] playback failed: {e}")
        print(f"[workbench] note={self.note} vel={self.velocity:.2f} "
              f"({len(samples) / FS:.2f}s)")

    def _write_wav(self, samples: np.ndarray):
        out = REPO_ROOT / "audio_out" / "_workbench_last.wav"
        out.parent.mkdir(exist_ok=True)
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(FS)
            w.writeframes(samples.tobytes())

    # ------------------------------------------------------------- persistence

    def save(self) -> bool:
        """Write changed params back to the preset YAML, bump meta.version."""
        if not self.changed and not self.patchboard.dirty.any:
            print("[workbench] nothing to save")
            return False
        from ruamel.yaml import YAML

        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.width = 4096
        # read_bytes: read_text() would translate CRLF away (repo is CRLF)
        source_bytes = self.preset_path.read_bytes()
        if self._fingerprint(source_bytes) != self._source_fingerprint:
            self.patchboard.status = (
                "SAVE rejected: source changed on disk; RELOAD before saving")
            self._notify_state_changed()
            return False
        text = source_bytes.decode("utf-8")
        crlf = "\r\n" in text
        doc = yaml.load(text)
        root = doc["instrument_ir_v0"]

        applied, failed = [], []
        for address, value in self.changed.items():
            param = self.registry[address]
            try:
                ok = param.yaml_apply(root, self.ir, value)
            except Exception as e:
                print(f"[workbench] yaml apply failed for {address}: {e}")
                ok = False
            (applied if ok else failed).append(address)

        try:
            applied.extend(self.patchboard.apply_yaml(root))
        except Exception as e:
            self.patchboard.status = f"SAVE failed: {e}"
            self._log("save_error", error=str(e))
            print(f"[workbench] structural save failed: {e}")
            self._notify_state_changed()
            return False

        if failed:
            self.patchboard.status = (
                f"SAVE rejected: {len(failed)} field(s) could not be written")
            self._log("save_error", failed=failed)
            print(f"[workbench] save rejected; not written: {failed}")
            self._notify_state_changed()
            return False

        old_version = int(root["meta"].get("version", 0))
        root["meta"]["version"] = old_version + 1

        buf = io.StringIO()
        yaml.dump(doc, buf)
        out = buf.getvalue()
        if crlf:
            out = out.replace("\r\n", "\n").replace("\n", "\r\n")
        try:
            self._atomic_write_source(out)
        except Exception as e:
            self.patchboard.status = f"SAVE write failed: {e}"
            self._log("save_error", error=str(e))
            print(f"[workbench] save write failed: {e}")
            self._notify_state_changed()
            return False
        self.ir.meta.version = old_version + 1
        self._source_fingerprint = self._fingerprint(out.encode("utf-8"))

        self._log("save", version=old_version + 1, applied=applied,
                  failed=failed)
        print(f"[workbench] saved {self.preset_path.name} "
              f"v{old_version} -> v{old_version + 1} "
              f"({len(applied)} fields{', ' + str(len(failed)) + ' failed' if failed else ''})")
        self.changed.clear()
        self.patchboard.reset(
            f"Saved {self.preset_path.name} v{old_version + 1}")
        self._notify_state_changed()
        return True

    def reload(self):
        """Re-read the preset from disk (after a manual YAML edit)."""
        self.ir = load_instrument(str(self.preset_path))
        self._source_fingerprint = self._fingerprint(
            self.preset_path.read_bytes())
        self.changed.clear()
        self.patchboard.reset(f"Reloaded {self.preset_path.name}")
        self._log("reload")
        print(f"[workbench] reloaded {self.preset_path.name}")
        self._touch("reload", 1)
        self._notify_state_changed()

    def validate_source(self, source: str):
        """Validate complete YAML text through the real InstrumentIR loader."""
        if not isinstance(source, str) or not source.strip():
            return False, "Source is empty", None
        scratch = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", suffix=".yaml",
                    newline="", delete=False) as file:
                scratch = Path(file.name)
                file.write(source)
            ir = load_instrument(str(scratch))
            valid, error = ir.validate()
            if not valid:
                return False, error, None
            return True, "YAML and InstrumentIR are valid", ir
        except Exception as e:
            return False, str(e), None
        finally:
            if scratch is not None:
                try:
                    scratch.unlink(missing_ok=True)
                except OSError:
                    pass

    def apply_source(self, source: str):
        """Atomically replace the preset from the embedded source editor.

        The round-trip YAML parser bumps ``meta.version`` while retaining
        comments. The complete result is validated before the backing file or
        live IR is touched.
        """
        from ruamel.yaml import YAML

        if self.changed or self.patchboard.dirty.any:
            message = ("Source apply rejected: SAVE or RELOAD pending knob/"
                       "patchboard edits first")
            self.patchboard.status = message
            return False, message

        try:
            yaml = YAML()
            yaml.preserve_quotes = True
            yaml.width = 4096
            doc = yaml.load(source)
            root = doc["instrument_ir_v0"]
            meta = root["meta"]
            incoming_version = int(meta.get("version", 0))
            current_bytes = self.preset_path.read_bytes()
            if self._fingerprint(current_bytes) != self._source_fingerprint:
                message = ("Source apply rejected: file changed on disk; "
                           "RELOAD and REFRESH before applying")
                self.patchboard.status = message
                return False, message
            current = current_bytes.decode("utf-8")
            disk_doc = yaml.load(current)
            disk_version = int(
                disk_doc["instrument_ir_v0"]["meta"].get("version", 0))
            if incoming_version != disk_version:
                message = ("Source apply rejected: editor version "
                           f"{incoming_version} is stale; disk is "
                           f"{disk_version}. REFRESH before applying")
                self.patchboard.status = message
                return False, message
            meta["version"] = disk_version + 1
            buffer = io.StringIO()
            yaml.dump(doc, buffer)
            rendered = buffer.getvalue()
        except Exception as e:
            message = f"YAML parse failed: {e}"
            self.patchboard.status = message
            return False, message

        valid, message, ir = self.validate_source(rendered)
        if not valid:
            self.patchboard.status = f"Source rejected: {message}"
            return False, self.patchboard.status

        if "\r\n" in current:
            rendered = rendered.replace("\r\n", "\n").replace("\n", "\r\n")
        try:
            self._atomic_write_source(rendered)
        except Exception as e:
            message = f"Source write failed: {e}"
            self.patchboard.status = message
            return False, message

        self.ir = ir
        self._source_fingerprint = self._fingerprint(rendered.encode("utf-8"))
        self.changed.clear()
        self.patchboard.reset(
            f"Applied source: {self.preset_path.name} v{disk_version + 1}")
        self._log("source_apply", version=disk_version + 1)
        self._touch("source_apply", 1)
        self._notify_state_changed()
        return True, self.patchboard.status

    def _atomic_write_source(self, text: str):
        """Durably replace the backing YAML without exposing a partial file."""
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", suffix=".yaml.tmp",
                    prefix=f".{self.preset_path.stem}-",
                    dir=self.preset_path.parent, newline="", delete=False) as file:
                temp_path = Path(file.name)
                file.write(text)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_path, self.preset_path)
        except Exception:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    # ------------------------------------------------------------- library

    def list_presets(self):
        """The loadable preset catalog (see library.scan_preset_catalog)."""
        from .library import scan_preset_catalog
        return scan_preset_catalog()

    def load_preset(self, token) -> bool:
        """
        Swap this session to another preset (/preset/load). `token` is an
        id-or-filename: a filename stem, meta id, display name, capsule
        id from the library index, a YAML path, or a PresetEntry.

        Unsaved knob edits on the old preset are dropped, the JSONL
        trajectory context restarts in a fresh per-preset log file, and
        on_preset_loaded fires so transports can push_state (knob snap).
        """
        from .library import PresetEntry, resolve_preset

        entry = token if isinstance(token, PresetEntry) else \
            resolve_preset(str(token))
        if entry is None:
            print(f"[workbench] unknown preset: {token}")
            return False
        try:
            ir = load_instrument(str(entry.path))
        except Exception as e:
            print(f"[workbench] failed to load {entry.path.name}: {e}")
            self._log("preset_load_error", preset=str(entry.path),
                      error=str(e))
            return False

        self._log("preset_unload", preset=str(self.preset_path),
                  discarded_edits=sorted(self.changed))
        self.preset_path = entry.path
        self.ir = ir
        self._source_fingerprint = self._fingerprint(entry.path.read_bytes())
        self.changed.clear()
        self.patchboard.reset(f"Loaded {entry.path.name}")
        # New trajectory context: one JSONL per (preset, load)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._log_path = self._log_dir / f"{entry.path.stem}-{stamp}.jsonl"
        self._log("session_start", preset=str(entry.path),
                  via="preset_load")
        print(f"[workbench] loaded {entry.path.name} "
              f"({self.ir.meta.name} v{self.ir.meta.version})")
        if self.on_preset_loaded is not None:
            try:
                self.on_preset_loaded(self)
            except Exception as e:  # feedback must not kill the session
                print(f"[workbench] preset-loaded callback failed: {e}")
        self._touch("preset", 1)
        return True

    # ------------------------------------------------------------- feedback

    def push_state(self, send: Callable[[str, object], None]):
        """Send current values to the control surface (widget sync)."""
        for address, param in self.registry.items():
            try:
                v = param.get(self.ir)
            except Exception:
                v = None
            if v is not None:
                send(address, float(v))
        send("/note", float(self.note))
        send("/velocity", float(self.velocity))
        send("/duration", float(self.duration))
        send("/autoplay", 1.0 if self.autoplay else 0.0)
        # Snap the preset picker to the loaded preset (string payload)
        send("/preset/load", self.preset_path.stem)
        self.patchboard.push_state(send)
        send("/editor/dirty",
             1.0 if self.changed or self.patchboard.dirty.any else 0.0)
        send("/editor/preset/path", str(self.preset_path))
        send("/editor/version", float(self.ir.meta.version))

    def _notify_state_changed(self):
        if self.on_state_changed is None:
            return
        try:
            self.on_state_changed(self)
        except Exception as e:  # feedback must not kill the session
            print(f"[workbench] state callback failed: {e}")

    # ------------------------------------------------------------- plumbing

    @staticmethod
    def _fingerprint(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _log(self, event: str, **fields):
        entry = {"t": datetime.now().isoformat(timespec="milliseconds"),
                 "event": event, **fields}
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def close(self):
        self._stop.set()
        self._worker.join(timeout=1.0)
        self._log("session_end")
