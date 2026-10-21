"""
Phase C streaming loop (WS4c): sustained note rendered block-by-block
through vm_step_block with per-block parameter latch, plus atlas mode
(live single-axis sweeps while the note sustains).

Architecture
------------
StreamingVoice is bound to a WorkbenchSession and renders one gate-held
note 64 samples at a time. At every block boundary it:

  1. advances any active axis sweeps (atlas mode) directly via the
     param registry (no per-block JSONL spam), then
  2. drains a transport-agnostic injection queue, applying each message
     through WorkbenchSession.handle() (so live tweaks are trajectory-
     logged and tracked in session.changed for SAVE), then
  3. rebuilds the VoiceFrame from the live IR -- the latch -- and steps
     the VM.

A parameter injected mid-stream is therefore audible on the very next
block (1 block = 1.33 ms @ 48 kHz / 64), far inside the 50 ms
knob->ear budget. The slow debounce re-render in WorkbenchSession is
suppressed while a stream is live (session.streaming flag).

Transports are thin shims over the same two entry points
(inject / start_sweep):

  * StreamingOSCBridge -- activates when python-osc is importable.
  * tests / the WS3 refiner -- call inject() directly in-process.
  * MIDI CC mapper -- intentionally absent: no MIDI library in the env
    (see OSC_AVAILABLE-style pattern to add one later).

CLI (acceptance measurements + demo WAV):

    pixi run python -m symcrash_workbench.streaming string_pad_v0
    pixi run python -m symcrash_workbench.streaming string_pad_v0 --serve

The first form runs the headless acceptance report (latency, 10 s
realtime stability, sweep continuity) and writes a sustained-note demo
with a live axis sweep to audio_out/workbench/.
"""

import argparse
import copy as _copylib
import math
import queue
import sys
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from symcrash_python_ap.primitives import Q15_HALF, Q15_ONE, Q16_ONE

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT / "tools" / "render")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from symcrash_python_ap.frames import VoiceFrame  # noqa: E402
from symcrash_python_ap.instrument_ir import (  # noqa: E402
    extract_unison_params,
)
from symcrash_python_ap.voice.frame_builder import (  # noqa: E402
    GATE_HELD as _GATE_HELD,
    GOLDEN_Q32 as _GOLDEN_Q32,
    IRFrameBuilder,
    NoteClock,
    equal_power_pan,
    unison_positions,
)

from .library import resolve_preset  # noqa: E402
from .system_axes import SystemAxisRuntime  # noqa: E402
from symcrash_python_ap.vm.executor import vm_step_block  # noqa: E402
from symcrash_python_ap.vm.harmonic_bank import (  # noqa: E402
    LFOTarget, clear_basis_atlas,
)
from symcrash_python_ap.vm.state import OSC_MODE_FM6, VMState  # noqa: E402

FS = 48000
BLOCK_SIZE = 64
BLOCK_SEC = BLOCK_SIZE / FS
BLOCKS_PER_SEC = FS // BLOCK_SIZE
BUDGET_US = BLOCK_SEC * 1e6  # 1333.3 us realtime budget per block

# _GATE_HELD / _GOLDEN_Q32 are imported from frame_builder (single home).

# Fraction of the block budget the unison bank may consume before the
# note_on-time voice-count clamp engages (headroom for frame build, GC,
# device writes).
_CLAMP_HEADROOM = 0.75

try:
    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import ThreadingOSCUDPServer
    OSC_AVAILABLE = True
except Exception:  # pragma: no cover - python-osc is in the env today
    OSC_AVAILABLE = False

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - playback is best-effort
    sd = None


# ===================================================================== sweeps

@dataclass
class AxisSweep:
    """Linear single-parameter sweep, latched once per block (atlas mode)."""

    address: str
    start: float
    end: float
    duration_sec: float
    loop: str = "none"  # none | wrap | pingpong
    begin_block: int = 0
    active: bool = True

    def value_at(self, block_index: int) -> float:
        t = ((block_index - self.begin_block) * BLOCK_SEC
             / max(self.duration_sec, BLOCK_SEC))
        if self.loop == "wrap":
            t = t % 1.0
        elif self.loop == "pingpong":
            cyc = t % 2.0
            t = cyc if cyc <= 1.0 else 2.0 - cyc
        elif t >= 1.0:
            t = 1.0
            self.active = False
        return self.start + (self.end - self.start) * t


@dataclass
class LatchRecord:
    """One injected message that latched: when and how fast."""

    block_index: int       # block whose render reflects the value
    address: str
    value: float
    wall_latency_s: float  # inject() call -> latch at block boundary


@dataclass
class _UnisonCopy:
    """One unison voice copy: its own VMState + fixed stereo position."""

    state: "VMState"
    t: float           # -1..+1 position (detune + pan), fixed at note_on
    phase_q32: int     # golden-angle Q32 start phase (also noise-seed XOR)


# ====================================================================== voice

class StreamingVoice:
    """
    One sustained, gate-held note streaming through vm_step_block with
    the full harness semantics (LFOs, filter envelope, FM6, emphasis
    operators, velocity) re-latched from the live IR every block.

    Drive it either pull-style (render_block / render_seconds) or
    realtime (play_realtime). Inject parameter changes from any thread
    via inject(); run axis sweeps via start_sweep().
    """

    def __init__(self, session, note: Optional[int] = None,
                 velocity: Optional[float] = None):
        self.session = session
        if note is not None:
            session.note = int(note)
        if velocity is not None:
            session.velocity = max(0.0, min(1.0, float(velocity)))

        self.note = session.note
        self.state: Optional[VMState] = None
        self.block_index = 0
        self.gate = False
        self.active = False
        self._stop_event = threading.Event()

        # Unison bank (None = legacy mono path, bit-exact).
        self.copies: Optional[List[_UnisonCopy]] = None
        self._unison: Optional[dict] = None
        self._unison_gain = 1.0     # causal peak guard (only ever decreases)
        # Clamp the unison voice count to the measured realtime budget at
        # note_on. Tests that need the authored count set this to False.
        self.autoclamp = True

        self._queue: "queue.SimpleQueue" = queue.SimpleQueue()
        self._sweeps: List[AxisSweep] = []
        self._sweep_lock = threading.Lock()
        self.latch_log: deque = deque(maxlen=256)
        self._pending_preset: Optional[str] = None
        self._autoplay_at: Optional[float] = None

        self._note_off_block = 0
        self._release_from_q15 = 0
        self._release_blocks = 1
        self._last_gain_q15 = 0

        # Controller-owned L1 axes use their frozen MacroRuntime slew laws;
        # the live IR remains the target/source of truth for OSC and SAVE.
        self.system_axes = SystemAxisRuntime(session.ir)

    # ------------------------------------------------------------- gate

    def note_on(self, note: Optional[int] = None,
                velocity: Optional[float] = None):
        """(Re)trigger the sustained note; resets VM state and phase."""
        if note is not None:
            self.session.note = int(note)
        if velocity is not None:
            self.session.velocity = max(0.0, min(1.0, float(velocity)))
        self.note = self.session.note
        clear_basis_atlas()
        self._stop_event.clear()
        self.session.streaming = True
        self.block_index = 0
        self.gate = True
        self.active = True
        self._note_off_block = 0
        self._release_from_q15 = 0
        self._last_gain_q15 = 0
        self._unison_gain = 1.0
        self._autoplay_at = None
        self.system_axes.reset_from_ir(self.session.ir)
        self._unison = extract_unison_params(self.session.ir)
        if self._unison:
            self.state = None
            self._init_unison_bank()
        else:
            self.copies = None
            self.state = VMState.create(voice_id=0)
            self.state.reset_for_note_on(note=self.note)
        self.session._log("stream_note_on", note=self.note,
                          velocity=self.session.velocity,
                          unison_voices=len(self.copies) if self.copies else 1)

    def note_off(self):
        """Release the gate; the voice renders its release tail then stops."""
        if not self.gate:
            return
        self.gate = False
        self._note_off_block = self.block_index
        self._release_from_q15 = self._last_gain_q15
        self.session._log("stream_note_off", block=self.block_index)

    def stop(self):
        """Hard-stop the voice and re-enable the session debounce render."""
        self._stop_event.set()
        self.gate = False
        self.active = False
        self.session.streaming = False

    # ------------------------------------------------------------- input

    def inject(self, address: str, value) -> None:
        """
        Queue one control message (any thread). It latches at the next
        block boundary and is applied through WorkbenchSession.handle(),
        so it is trajectory-logged and tracked for SAVE.
        """
        self._queue.put((address, value, time.perf_counter()))

    def start_sweep(self, address: str, start: float, end: float,
                    duration_sec: float, loop: str = "none") -> AxisSweep:
        """
        Atlas mode: sweep one registry parameter linearly while the note
        sustains, latched once per block. loop: none | wrap | pingpong.
        """
        if address not in self.session.registry:
            raise KeyError(f"unknown parameter address: {address}")
        sweep = AxisSweep(address, float(start), float(end),
                          float(duration_sec), loop,
                          begin_block=self.block_index)
        with self._sweep_lock:
            self._sweeps.append(sweep)
        self.session._log("stream_sweep", address=address, start=start,
                          end=end, duration_sec=duration_sec, loop=loop)
        return sweep

    def clear_sweeps(self):
        with self._sweep_lock:
            self._sweeps.clear()

    # ------------------------------------------------------------- render

    @property
    def stereo(self) -> bool:
        """True when the unison bank is active (blocks are (64, 2))."""
        return self.copies is not None

    @property
    def stopped(self) -> bool:
        """True after an explicit service stop (distinct from note silence)."""
        return self._stop_event.is_set()

    def render_block(self) -> np.ndarray:
        """
        Latch pending input, render one 64-sample block.

        Mono presets return a 1-D int16 array (bit-exact legacy path);
        unison presets return an int16 (BLOCK_SIZE, 2) = (L, R) block.
        """
        if self.active:
            self._apply_sweeps()
        self._drain_queue()
        if self._pending_preset is not None:
            token, self._pending_preset = self._pending_preset, None
            if self.active:
                swapped = self._swap_preset(token)
                if swapped is not None:
                    return swapped
            else:
                self.session.load_preset(token)
                self.system_axes.reset_from_ir(self.session.ir)
        # A silent live service still drains the input queue above, allowing
        # the next keyboard or PLAY event to reactivate it at a block edge.
        if not self.active:
            return self._silence()
        frame = self._build_frame()
        out = self._step_bank(frame)
        self.block_index += 1
        if not self.gate and self.block_index >= (
                self._note_off_block + self._release_blocks):
            self.active = False
            self.session.streaming = False
        return out

    def render_seconds(self, seconds: float) -> np.ndarray:
        """Render up to `seconds` of audio (stops early if voice ends)."""
        blocks = []
        for _ in range(int(round(seconds * BLOCKS_PER_SEC))):
            if not self.active:
                break
            blocks.append(self.render_block())
        if not blocks:
            return np.zeros(0, dtype=np.int16)
        # A /preset/load swap can change the channel count mid-render:
        # upmix mono blocks so the concatenation stays consistent.
        if len({b.ndim for b in blocks}) > 1:
            blocks = [b if b.ndim == 2 else np.stack([b, b], axis=1)
                      for b in blocks]
        return np.concatenate(blocks)

    def play_realtime(self, seconds: Optional[float] = None,
                      keep_open: bool = False) -> bool:
        """
        Stream to the default audio device (best-effort; returns False
        when sounddevice is unavailable). Blocks until the voice ends (or an
        explicit stop when ``keep_open`` is true), `seconds` elapse, or
        KeyboardInterrupt.

        The device stream is opened stereo; mono blocks are upmixed for
        playback only, so a live /preset/load swap between mono and
        unison presets never needs to reopen the device. If the render
        loop sustains an over-budget block time, one unison copy is
        dropped at a time (with a log line) instead of glitching.
        """
        if sd is None:
            print("[stream] sounddevice unavailable; cannot play realtime")
            return False
        deadline = None if seconds is None else time.monotonic() + seconds
        recent = deque(maxlen=48)
        # 8 blocks of device buffer (10.7 ms) absorbs the rare OS/GC
        # stall (observed max ~8 ms) while keeping knob->ear ~12 ms.
        with sd.OutputStream(samplerate=FS, channels=2, dtype="int16",
                             blocksize=BLOCK_SIZE * 8) as out:
            try:
                while not self.stopped and (keep_open or self.active):
                    t0 = time.perf_counter()
                    block = self.render_block()
                    recent.append((time.perf_counter() - t0) * 1e6)
                    if block.ndim == 1:
                        block = np.stack([block, block], axis=1)
                    out.write(block)
                    if (self.copies and len(self.copies) > 1
                            and len(recent) == recent.maxlen
                            and sum(recent) / len(recent) > BUDGET_US):
                        dropped = self.copies.pop()
                        recent.clear()
                        print(f"[stream] over budget: dropped unison copy "
                              f"t={dropped.t:+.2f} -> "
                              f"{len(self.copies)} voices")
                        self.session._log("unison_drop",
                                          voices=len(self.copies))
                    if deadline is not None and time.monotonic() > deadline:
                        break
            except KeyboardInterrupt:
                pass
        return True

    # ------------------------------------------------------------- internals

    def _silence(self) -> np.ndarray:
        if self.copies is not None:
            return np.zeros((BLOCK_SIZE, 2), dtype=np.int16)
        return np.zeros(BLOCK_SIZE, dtype=np.int16)

    def _init_unison_bank(self):
        """
        Build the N-copy unison bank (mirrors render_preset._render_voice
        semantics: symmetric cents detune, golden-angle Q32 start phases,
        per-copy noise-seed XOR, equal-power pan, 1/sqrt(N) mix).

        When autoclamp is on, the per-copy block cost is measured on a
        scratch VMState first and the voice count is clamped so the bank
        fits the realtime budget (graceful degradation, no glitching).
        """
        n_authored = int(self._unison["voices"])
        n = n_authored
        if self.autoclamp:
            cost_us = self._measure_copy_cost_us()
            ceiling = max(1, int((BUDGET_US * _CLAMP_HEADROOM) //
                                 max(cost_us, 1e-3)))
            if n > ceiling:
                print(f"[stream] unison: clamped {n} -> {ceiling} voices "
                      f"(measured {cost_us:.0f} us/copy, budget "
                      f"{BUDGET_US:.0f} us/block)")
                self.session._log("unison_clamp", authored=n, voices=ceiling,
                                  copy_cost_us=round(cost_us, 1))
                n = ceiling
        self.copies = []
        for t, phase_q32 in unison_positions(n):
            st = VMState.create(voice_id=0)
            st.reset_for_note_on(note=self.note)
            if phase_q32:
                st.mode_b_state.phase_offset_q32 = phase_q32
            self.copies.append(_UnisonCopy(state=st, t=t, phase_q32=phase_q32))

    def _measure_copy_cost_us(self, warmup: int = 3, timed: int = 8) -> float:
        """
        Measure one voice copy's vm_step_block cost (us) on a throwaway
        VMState. Stateful LFO modulators are swapped for deep copies for
        the duration so the live IR's LFO phase is untouched; the basis
        atlas and JIT kernels get warmed as a side benefit.
        """
        ir = self.session.ir
        saved_mods = getattr(ir, "modulators", None)
        if saved_mods:
            ir.modulators = _copylib.deepcopy(saved_mods)
        try:
            st = VMState.create(voice_id=0)
            st.reset_for_note_on(note=self.note)
            for _ in range(warmup):
                st, _ = vm_step_block(st, self._build_frame())
            costs = []
            for _ in range(timed):
                frame = self._build_frame()
                t0 = time.perf_counter()
                st, _ = vm_step_block(st, frame)
                costs.append(time.perf_counter() - t0)
            costs.sort()
            return costs[len(costs) // 2] * 1e6  # median: jitter-robust
        finally:
            if saved_mods:
                ir.modulators = saved_mods

    def _step_bank(self, frame: VoiceFrame) -> np.ndarray:
        """Step the VM: mono legacy path, or the N-copy unison stereo mix."""
        if self.copies is None:
            self.state, out = vm_step_block(self.state, frame)
            # vm_step_block reuses its output buffer: copy or blocks alias
            return out.copy()

        # Unison: spread/width re-read from the live IR each block (knob
        # edits latch); the copy count is fixed until the next note_on.
        live = extract_unison_params(self.session.ir) or self._unison
        spread = live["spread_cents"]
        width = live["width"]
        noise_on = getattr(frame, "noise_level_q15", 0) > 0
        base_cents = frame.cents
        base_seed = getattr(frame, "noise_seed_u32", 0)
        n = len(self.copies)
        mix_l = np.zeros(BLOCK_SIZE, dtype=np.float64)
        mix_r = np.zeros(BLOCK_SIZE, dtype=np.float64)
        for c in self.copies:
            f = _copylib.copy(frame)  # shallow: per-copy scalar overrides
            f.cents = base_cents + round(spread * c.t)
            if noise_on and c.phase_q32:
                f.noise_seed_u32 = (base_seed ^ c.phase_q32) & 0xFFFFFFFF
            c.state, out = vm_step_block(c.state, f)
            pan_l, pan_r = equal_power_pan(c.t, width)  # equal-power pan
            o = out.astype(np.float64)
            mix_l += o * pan_l
            mix_r += o * pan_r
        # Decorrelated copies sum in power -> 1/sqrt(N); the peak guard is
        # causal (gain ratchets down only) so it never pumps per block.
        norm = self._unison_gain / math.sqrt(n)
        peak = max(float(np.abs(mix_l).max()),
                   float(np.abs(mix_r).max())) * norm
        if peak > 32000.0:
            self._unison_gain *= 32000.0 / peak
            norm = self._unison_gain / math.sqrt(n)
        left = np.clip(np.rint(mix_l * norm), -32768, 32767).astype(np.int16)
        right = np.clip(np.rint(mix_r * norm), -32768, 32767).astype(np.int16)
        return np.stack([left, right], axis=1)

    def _swap_preset(self, token) -> Optional[np.ndarray]:
        """
        Live /preset/load at a block boundary: fade the old voice out over
        one block (clean note end), swap the session IR, retrigger the
        note on the new preset (fresh attack from silence -> no click),
        and keep the stream running. Returns the fade-out block, or None
        when the token doesn't resolve (stream continues untouched).
        """
        entry = resolve_preset(str(token))
        if entry is None:
            print(f"[stream] /preset/load: unknown preset {token!r}")
            self.session._log("preset_load_unknown", token=str(token))
            return None
        out = self._step_bank(self._build_frame()).astype(np.float64)
        ramp = np.linspace(1.0, 0.0, BLOCK_SIZE)
        out *= ramp[:, None] if out.ndim == 2 else ramp
        fade = np.rint(out).astype(np.int16)
        self.clear_sweeps()
        if self.session.load_preset(entry):
            self.note_on(self.note, self.session.velocity)
        return fade

    def _apply_sweeps(self):
        with self._sweep_lock:
            sweeps = list(self._sweeps)
        done = []
        for sweep in sweeps:
            value = sweep.value_at(self.block_index)
            param = self.session.registry.get(sweep.address)
            if param is not None:
                param.set(self.session.ir, value)
            if not sweep.active:
                done.append(sweep)
        if done:
            with self._sweep_lock:
                for sweep in done:
                    if sweep in self._sweeps:
                        self._sweeps.remove(sweep)

    def _drain_queue(self):
        play_requested = False
        while True:
            try:
                address, value, t_in = self._queue.get_nowait()
            except queue.Empty:
                break
            if address == "/stream/keyboard":
                try:
                    note, gate = value
                    note = max(0, min(127, int(note)))
                    gate = float(gate)
                except (TypeError, ValueError):
                    continue
                if gate >= 0.5:
                    self.note_on(note, self.session.velocity)
                elif note == self.note:
                    # A stale release from a previously traversed key must
                    # not silence the newer monophonic note.
                    self.note_off()
                self.latch_log.append(LatchRecord(
                    self.block_index, address, gate,
                    time.perf_counter() - t_in))
                continue
            if address == "/play":
                # PLAY is a live note retrigger.  Do not route it through
                # WorkbenchSession.handle(), whose implementation starts the
                # legacy offline renderer and clears the shared basis atlas.
                try:
                    pressed = value is None or float(value) >= 0.5
                except (TypeError, ValueError):
                    pressed = False
                if pressed:
                    play_requested = True
                    self.latch_log.append(LatchRecord(
                        self.block_index, address, 1.0,
                        time.perf_counter() - t_in))
                continue
            if address == "/preset/load":
                # Swapping the IR under a mid-block frame build is not
                # safe from the OSC thread: defer to the block boundary
                # (render_block handles it right after this drain).
                if value is not None:
                    self._pending_preset = str(value)
                continue
            if address == "/editor/source/apply":
                ok, _ = self.session.apply_source(str(value))
                if ok:
                    # Source replacement can change graph, unison bank, and
                    # system state together; rebuild the voice coherently.
                    play_requested = True
                self.latch_log.append(LatchRecord(
                    self.block_index, address, 1.0 if ok else 0.0,
                    time.perf_counter() - t_in))
                continue
            if address.startswith("/editor/"):
                payload = value if isinstance(value, tuple) else (value,)
                ok = self.session.handle_message(address, *payload)
            else:
                ok = self.session.handle(address, value)
            if ok:
                if address == "/note":
                    self.note = self.session.note
                if address == "/autoplay":
                    if not self.session.autoplay:
                        self._autoplay_at = None
                elif (self.session.autoplay
                      and address not in {"/save", "/preset/list"}):
                    # Preserve the offline workbench's debounce semantics:
                    # continuous knob traffic remains smoothly live, then
                    # the completed edit retriggers the attack once.
                    self._autoplay_at = (time.monotonic()
                                         + self.session.debounce)
                self.latch_log.append(LatchRecord(
                    self.block_index, address,
                    (float(value) if value is not None
                     and not isinstance(value, tuple) else 1.0),
                    time.perf_counter() - t_in))

        if play_requested:
            self.note_on(self.session.note, self.session.velocity)
        elif (self._autoplay_at is not None
              and time.monotonic() >= self._autoplay_at):
            self.note_on(self.session.note, self.session.velocity)

    def _build_frame(self) -> VoiceFrame:
        """Rebuild the VoiceFrame from the live IR: the per-block latch.

        All frame construction lives in voice/frame_builder.IRFrameBuilder
        (relatch mode: every build re-extracts from the live IR so knob
        edits latch at block boundaries and the IR's stateful LFO
        modulators advance in place).
        """
        builder = IRFrameBuilder(
            self.session.ir, self.note,
            int(self.session.velocity * Q15_ONE),
            relatch_per_block=True,
        )
        if self.gate:
            clock = NoteClock(self.block_index, gate=True,
                              sustain_until_block=_GATE_HELD)
        else:
            clock = NoteClock(self.block_index, gate=False,
                              sustain_until_block=self._note_off_block,
                              release_from_q15=self._release_from_q15)
        frame = builder.build(clock)
        self.system_axes.apply(frame, self.session.ir)
        self._release_blocks = builder.release_blocks
        self._last_gain_q15 = builder.last_amp_gain_q15
        return frame


# ================================================================ OSC bridge

class StreamingOSCBridge:
    """
    Thin OSC shim over StreamingVoice (active when python-osc exists).

    In (default udp/7400):
    /stream/note_on [note] [velocity]   (re)trigger the sustained note
    /stream/note_off                    release
    /stream/sweep  addr lo hi dur [loop]  start an axis sweep
    /stream/sweep_stop                  clear sweeps
    /preset/list                        reply with the preset catalog
    /preset/load  <id-or-filename>      live-swap the session's preset
    anything else                       -> voice.inject(address, value)

    Out (feedback, default udp/7500 = Open Stage Control's --osc-port):
    /preset/item  stem id name capsules   one per catalog entry
    /preset/count n                        end of /preset/list reply
    /preset/loaded stem                    after a successful load
    <param address> value                  push_state widget sync
    """

    def __init__(self, voice: StreamingVoice,
                 listen_host: str = "0.0.0.0", listen_port: int = 7400,
                 feedback_host: Optional[str] = "127.0.0.1",
                 feedback_port: Optional[int] = 7500):
        if not OSC_AVAILABLE:
            raise RuntimeError("python-osc not installed")
        self.voice = voice
        self.client = None
        if feedback_host and feedback_port:
            from pythonosc.udp_client import SimpleUDPClient
            self.client = SimpleUDPClient(feedback_host, feedback_port)
            voice.session.on_preset_loaded = self._on_preset_loaded
            voice.session.on_state_changed = self._on_state_changed
        dispatcher = Dispatcher()
        dispatcher.set_default_handler(self._route)
        self.server = ThreadingOSCUDPServer((listen_host, listen_port),
                                            dispatcher)
        self._thread = threading.Thread(target=self.server.serve_forever,
                                        daemon=True)

    @property
    def port(self) -> int:
        return self.server.server_address[1]

    def start(self):
        self._thread.start()
        if self.client is not None:
            # Mirror the loaded preset onto the surface at connect time
            self.voice.session.push_state(self.client.send_message)
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    def _on_preset_loaded(self, session):
        if self.client is not None:
            session.push_state(self.client.send_message)
            self.client.send_message("/preset/loaded",
                                     session.preset_path.stem)

    def _on_state_changed(self, session):
        if self.client is not None:
            session.push_state(self.client.send_message)

    def _route(self, address, *args):
        v = self.voice
        try:
            if address == "/stream/note_on":
                v.note_on(int(args[0]) if len(args) > 0 else None,
                          float(args[1]) if len(args) > 1 else None)
            elif address == "/stream/note_off":
                v.note_off()
            elif address == "/stream/keyboard" and len(args) >= 2:
                # The Open Stage Control keyboard sends
                # /stream/keyboard <midi-note> <on|off>.  Queue it so VM
                # reset/release occurs at the next audio block boundary.
                v.inject(address, (int(args[0]), float(args[1])))
            elif address == "/stream/sweep" and len(args) >= 4:
                loop = str(args[4]) if len(args) > 4 else "none"
                v.start_sweep(str(args[0]), float(args[1]), float(args[2]),
                              float(args[3]), loop=loop)
            elif address == "/stream/sweep_stop":
                v.clear_sweeps()
            elif address == "/preset/list":
                if self.client is not None:
                    from .library import send_preset_list
                    send_preset_list(self.client.send_message)
            elif address == "/preset/load":
                token = str(args[0]) if args else ""
                if v.active:
                    v.inject(address, token)   # safe swap at block boundary
                elif token:
                    v.session.load_preset(token)
            elif address.startswith("/editor/"):
                # Patchbay widgets send input + zero or more outputs. Keep
                # the entire message together until the audio block boundary.
                v.inject(address, tuple(args))
            else:
                v.inject(address, args[0] if args else None)
        except Exception as e:  # keep the OSC thread alive on bad input
            print(f"[stream] OSC {address}: {e}")


# ============================================================== measurements

def _write_wav(path: Path, samples: np.ndarray):
    samples = np.asarray(samples, dtype=np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2 if samples.ndim == 2 else 1)
        w.setsampwidth(2)
        w.setframerate(FS)
        # (n, 2) C-order tobytes() is already L/R interleaved
        w.writeframes(samples.tobytes())


def boundary_step_metrics(samples: np.ndarray) -> dict:
    """
    Click detector: compare sample steps across block boundaries with
    steps inside blocks. A per-block discontinuity shows up as boundary
    steps far exceeding intra-block steps.
    """
    x = np.asarray(samples, dtype=np.int32)
    if len(x) < 2 * BLOCK_SIZE:
        return {"boundary_max": 0, "intra_max": 0, "ratio": 0.0}
    d = np.abs(np.diff(x))
    boundary_idx = np.arange(BLOCK_SIZE - 1, len(d), BLOCK_SIZE)
    mask = np.zeros(len(d), dtype=bool)
    mask[boundary_idx] = True
    boundary_max = int(d[mask].max()) if mask.any() else 0
    intra_max = int(d[~mask].max()) if (~mask).any() else 0
    return {
        "boundary_max": boundary_max,
        "intra_max": intra_max,
        "ratio": boundary_max / max(1, intra_max),
    }


def measure_latch_latency(make_session, note: int = 60,
                          address: str = "/amp/sustain", value: float = 0.25,
                          inject_block: int = 400,
                          total_blocks: int = 460) -> dict:
    """
    Headless knob->audio latency: render a control stream and an
    injected stream from identical fresh sessions; the first block whose
    audio differs is where the change reached the ear.
    """
    runs = {}
    for label in ("control", "injected"):
        session = make_session()
        try:
            voice = StreamingVoice(session)
            voice.note_on(note)
            blocks = []
            wall = None
            for i in range(total_blocks):
                if label == "injected" and i == inject_block:
                    voice.inject(address, value)
                blocks.append(voice.render_block())
            if label == "injected" and voice.latch_log:
                wall = voice.latch_log[-1].wall_latency_s
            voice.stop()
            runs[label] = (blocks, wall)
        finally:
            session.close()

    control, _ = runs["control"]
    injected, wall = runs["injected"]
    first_diff = None
    for i, (a, c) in enumerate(zip(injected, control)):
        if not np.array_equal(a, c):
            first_diff = i
            break
    latency_blocks = (first_diff - inject_block
                      if first_diff is not None else None)
    return {
        "inject_block": inject_block,
        "first_diff_block": first_diff,
        "latency_blocks": latency_blocks,
        "latency_ms": (latency_blocks + 1) * BLOCK_SEC * 1e3
        if latency_blocks is not None else None,  # +1: block must play out
        "wall_latch_s": wall,
    }


def measure_stream_stability(session, note: int = 60, seconds: float = 10.0,
                             warmup_blocks: int = 100) -> dict:
    """Sustained streaming timing: per-block wall time vs the RT budget."""
    voice = StreamingVoice(session)
    voice.note_on(note)
    for _ in range(warmup_blocks):
        voice.render_block()
    n = int(seconds * BLOCKS_PER_SEC)
    times = np.empty(n, dtype=np.float64)
    for i in range(n):
        t0 = time.perf_counter()
        voice.render_block()
        times[i] = time.perf_counter() - t0
    voice.stop()
    times_us = times * 1e6
    return {
        "blocks": n,
        "seconds": seconds,
        "mean_us": float(times_us.mean()),
        "p95_us": float(np.percentile(times_us, 95)),
        "max_us": float(times_us.max()),
        "budget_us": BUDGET_US,
        "mean_pct_budget": float(times_us.mean() / BUDGET_US * 100),
        "realtime_ok": bool(times_us.mean() < BUDGET_US),
    }


def render_sweep_demo(session, axis: str = "/axis/stable/warp",
                      note: int = 60,
                      out_dir: Optional[Path] = None) -> dict:
    """
    Atlas-mode demo: sustain a note, sweep one axis 0 -> 1 -> 0 live
    (pingpong, 3 s per leg), release. Writes the WAV for audition and
    returns continuity metrics (sweep vs static sustain).
    """
    voice = StreamingVoice(session)
    voice.note_on(note)
    intro = voice.render_seconds(1.0)
    static = voice.render_seconds(1.0)          # static reference window
    voice.start_sweep(axis, 0.0, 1.0, 3.0, loop="pingpong")
    swept = voice.render_seconds(6.0)           # one full up/down cycle
    voice.clear_sweeps()
    # park the axis back at its resting value
    param = session.registry.get(axis)
    if param is not None:
        param.set(session.ir, 0.0)
    outro = voice.render_seconds(0.5)
    voice.note_off()
    tail = []
    while voice.active:
        tail.append(voice.render_block())
    tail = np.concatenate(tail) if tail else np.zeros(0, dtype=np.int16)

    audio = np.concatenate([intro, static, swept, outro, tail])
    out_dir = out_dir or (REPO_ROOT / "audio_out" / "workbench")
    slug = axis.strip("/").replace("/", "_")
    wav_path = out_dir / f"{session.preset_path.stem}_stream_{slug}.wav"
    _write_wav(wav_path, audio)

    static_m = boundary_step_metrics(static)
    sweep_m = boundary_step_metrics(swept)
    changed = int(np.abs(swept[:len(static)].astype(np.int32)
                         - static.astype(np.int32)).max())
    return {
        "wav": str(wav_path),
        "axis": axis,
        "static": static_m,
        "sweep": sweep_m,
        "sweep_audible_lsb": changed,
        "duration_sec": len(audio) / FS,
    }


# ======================================================================= CLI

def _resolve_preset(arg: str) -> Path:
    p = Path(arg)
    if p.exists():
        return p
    candidate = REPO_ROOT / "presets" / "instruments" / f"{arg}.yaml"
    if candidate.exists():
        return candidate
    sys.exit(f"preset not found: {arg} (tried {candidate})")


def main():
    from .session import WorkbenchSession

    ap = argparse.ArgumentParser(
        prog="symcrash_workbench.streaming",
        description="Phase C streaming loop: acceptance report, demo WAV, "
                    "or live OSC-driven streaming")
    ap.add_argument("preset", help="preset name or path")
    ap.add_argument("--note", type=int, default=60)
    ap.add_argument("--axis", default="/axis/stable/warp",
                    help="axis address for the sweep demo")
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="stability run length (default 10 s)")
    ap.add_argument("--serve", action="store_true",
                    help="live mode: OSC control + realtime playback")
    ap.add_argument("--listen-port", type=int, default=7400)
    args = ap.parse_args()

    preset_path = _resolve_preset(args.preset)

    if args.serve:
        session = WorkbenchSession(preset_path, autoplay=False)
        voice = StreamingVoice(session)
        if OSC_AVAILABLE:
            bridge = StreamingOSCBridge(
                voice, listen_port=args.listen_port).start()
            print(f"[stream] OSC listening on udp/{bridge.port} "
                  f"(/stream/note_on, /stream/sweep, knob addresses)")
        else:
            print("[stream] python-osc not installed; in-process only")
        voice.note_on(args.note)
        print(f"[stream] sustaining note {args.note}; Ctrl+C to stop")
        try:
            if not voice.play_realtime(keep_open=True):
                while not voice.stopped:  # headless OSC service: free-run
                    voice.render_block()
        finally:
            voice.stop()
            session.close()
        return

    print("=" * 72)
    print(f"Workbench streaming acceptance report - {preset_path.name}")
    print(f"Budget: {BUDGET_US:.1f} us/block ({BLOCK_SIZE} samples @ {FS} Hz)")
    print("=" * 72)

    def make_session():
        return WorkbenchSession(preset_path, autoplay=False)

    print("\n[1/3] Knob->audio latency (param injected mid-sustain)")
    lat = measure_latch_latency(make_session, note=args.note)
    print(f"  injected at block {lat['inject_block']}, first audible diff "
          f"at block {lat['first_diff_block']}")
    print(f"  latency: {lat['latency_blocks']} block(s) -> "
          f"{lat['latency_ms']:.2f} ms knob->ear "
          f"(budget 50 ms) "
          f"[{'OK' if lat['latency_ms'] and lat['latency_ms'] < 50 else 'FAIL'}]")
    if lat["wall_latch_s"] is not None:
        print(f"  wall-clock inject->latch: {lat['wall_latch_s'] * 1e3:.3f} ms")

    print(f"\n[2/3] Sustained streaming stability ({args.seconds:.0f} s)")
    session = make_session()
    try:
        stab = measure_stream_stability(session, note=args.note,
                                        seconds=args.seconds)
    finally:
        session.close()
    print(f"  mean {stab['mean_us']:.1f} us  p95 {stab['p95_us']:.1f} us  "
          f"max {stab['max_us']:.1f} us  "
          f"({stab['mean_pct_budget']:.1f}% of budget) "
          f"[{'OK' if stab['realtime_ok'] else 'OVER'}]")

    print(f"\n[3/3] Atlas sweep continuity + demo WAV ({args.axis})")
    session = make_session()
    try:
        demo = render_sweep_demo(session, axis=args.axis, note=args.note)
    finally:
        session.close()
    print(f"  static sustain: boundary step max {demo['static']['boundary_max']}"
          f" LSB, intra-block max {demo['static']['intra_max']} LSB")
    print(f"  live sweep:     boundary step max {demo['sweep']['boundary_max']}"
          f" LSB, intra-block max {demo['sweep']['intra_max']} LSB "
          f"(ratio {demo['sweep']['ratio']:.2f})")
    print(f"  sweep audibly different from static: "
          f"{demo['sweep_audible_lsb']} LSB max")
    print(f"  demo WAV ({demo['duration_sec']:.1f} s): {demo['wav']}")
    print(f"\n  OSC adapter: {'available (python-osc)' if OSC_AVAILABLE else 'NOT available'}")
    print(f"  MIDI mapper: skipped (no MIDI library in environment)")


if __name__ == "__main__":
    main()
