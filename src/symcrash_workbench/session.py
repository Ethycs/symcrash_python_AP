"""
WorkbenchSession - the tuning session backend.

Owns a loaded InstrumentIR, applies parameter changes from any caller
(OSC surface, MIDI mapper, or the WS3 optimizer), debounce-renders
through the full harness (tools/render/render_preset.py), plays the
result, logs every change as a (parameters, verdict) trajectory, and
writes tuned values back to the preset YAML with a meta.version bump.
"""

import io
import json
import sys
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
        self.registry = build_registry()
        self.changed: dict = {}

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
        self.changed[address] = value
        self._log("set", address=address, value=value)
        self._touch(address, value)
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
        with self._render_lock:
            clear_basis_atlas()
            t0 = time.monotonic()
            samples = render_preset_note(
                self.ir, note=self.note, duration_sec=self.duration,
                velocity_q15=int(self.velocity * Q15_ONE))
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
        if not self.changed:
            print("[workbench] nothing to save")
            return False
        from ruamel.yaml import YAML

        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.width = 4096
        # read_bytes: read_text() would translate CRLF away (repo is CRLF)
        text = self.preset_path.read_bytes().decode("utf-8")
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

        old_version = int(root["meta"].get("version", 0))
        root["meta"]["version"] = old_version + 1
        self.ir.meta.version = old_version + 1

        buf = io.StringIO()
        yaml.dump(doc, buf)
        out = buf.getvalue()
        if crlf:
            out = out.replace("\r\n", "\n").replace("\n", "\r\n")
        with open(self.preset_path, "w", encoding="utf-8", newline="") as f:
            f.write(out)

        self._log("save", version=old_version + 1, applied=applied,
                  failed=failed)
        print(f"[workbench] saved {self.preset_path.name} "
              f"v{old_version} -> v{old_version + 1} "
              f"({len(applied)} fields{', ' + str(len(failed)) + ' failed' if failed else ''})")
        if failed:
            print(f"[workbench]   not written: {failed}")
        self.changed.clear()
        return True

    def reload(self):
        """Re-read the preset from disk (after a manual YAML edit)."""
        self.ir = load_instrument(str(self.preset_path))
        self.changed.clear()
        self._log("reload")
        print(f"[workbench] reloaded {self.preset_path.name}")
        self._touch("reload", 1)

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
        self.changed.clear()
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

    # ------------------------------------------------------------- plumbing

    def _log(self, event: str, **fields):
        entry = {"t": datetime.now().isoformat(timespec="milliseconds"),
                 "event": event, **fields}
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def close(self):
        self._stop.set()
        self._worker.join(timeout=1.0)
        self._log("session_end")
