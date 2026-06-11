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

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT / "tools" / "render")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from render_preset import (  # noqa: E402
    _EnvelopeState, _filtered_partials, extract_waveform,
    get_emit_template, ms_to_blocks,
)
from symcrash_python_ap.envelope_targets import EnvelopeTarget  # noqa: E402
from symcrash_python_ap.frames import VoiceFrame  # noqa: E402
from symcrash_python_ap.instrument_ir import (  # noqa: E402
    extract_filter_params, extract_oeis_params, extract_partials_q15,
    extract_phase_spread_q15,
)
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

# "never" sentinel for the mod-envelope sustain hold while the gate is on
_GATE_HELD = 1 << 30

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

        self._queue: "queue.SimpleQueue" = queue.SimpleQueue()
        self._sweeps: List[AxisSweep] = []
        self._sweep_lock = threading.Lock()
        self.latch_log: deque = deque(maxlen=256)

        self._note_off_block = 0
        self._release_from_q15 = 0
        self._release_blocks = 1
        self._last_gain_q15 = 0

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
        self.session.streaming = True
        self.state = VMState.create(voice_id=0)
        self.state.reset_for_note_on(note=self.note)
        self.block_index = 0
        self.gate = True
        self.active = True
        self._note_off_block = 0
        self._release_from_q15 = 0
        self._last_gain_q15 = 0
        self.session._log("stream_note_on", note=self.note,
                          velocity=self.session.velocity)

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

    def render_block(self) -> np.ndarray:
        """Latch pending input, render one 64-sample block (int16 copy)."""
        if not self.active:
            return np.zeros(BLOCK_SIZE, dtype=np.int16)
        self._apply_sweeps()
        self._drain_queue()
        frame = self._build_frame()
        self.state, out = vm_step_block(self.state, frame)
        self.block_index += 1
        if not self.gate and self.block_index >= (
                self._note_off_block + self._release_blocks):
            self.active = False
            self.session.streaming = False
        # vm_step_block reuses its output buffer: copy or blocks alias
        return out.copy()

    def render_seconds(self, seconds: float) -> np.ndarray:
        """Render up to `seconds` of audio (stops early if voice ends)."""
        blocks = []
        for _ in range(int(round(seconds * BLOCKS_PER_SEC))):
            if not self.active:
                break
            blocks.append(self.render_block())
        if not blocks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(blocks)

    def play_realtime(self, seconds: Optional[float] = None) -> bool:
        """
        Stream to the default audio device (best-effort; returns False
        when sounddevice is unavailable). Blocks until the voice ends,
        `seconds` elapse, or KeyboardInterrupt.
        """
        if sd is None:
            print("[stream] sounddevice unavailable; cannot play realtime")
            return False
        deadline = None if seconds is None else time.monotonic() + seconds
        # 8 blocks of device buffer (10.7 ms) absorbs the rare OS/GC
        # stall (observed max ~8 ms) while keeping knob->ear ~12 ms.
        with sd.OutputStream(samplerate=FS, channels=1, dtype="int16",
                             blocksize=BLOCK_SIZE * 8) as out:
            try:
                while self.active:
                    out.write(self.render_block().reshape(-1, 1))
                    if deadline is not None and time.monotonic() > deadline:
                        break
            except KeyboardInterrupt:
                pass
        return True

    # ------------------------------------------------------------- internals

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
        while True:
            try:
                address, value, t_in = self._queue.get_nowait()
            except queue.Empty:
                break
            if address == "/play":
                # An offline debounce render mid-stream would clear the
                # basis atlas under the streaming voice; the stream IS
                # the audition, so /play is a no-op here.
                continue
            ok = self.session.handle(address, value)
            if ok:
                if address == "/note":
                    self.note = self.session.note
                self.latch_log.append(LatchRecord(
                    self.block_index, address,
                    float(value) if value is not None else 0.0,
                    time.perf_counter() - t_in))

    def _build_frame(self) -> VoiceFrame:
        """Rebuild the VoiceFrame from the live IR: the per-block latch."""
        ir = self.session.ir
        note = self.note
        b = self.block_index
        velocity_q15 = int(self.session.velocity * 32767)

        # --- Amp ADSR, gate-driven (sustain holds until note_off) ---
        a_blocks, d_blocks, r_blocks = 1, 1, 1
        s_q15 = 32767
        vel_sens_q15 = 32767
        for node in ir.layers.L0_graph.nodes:
            if node.type == "Amp" and node.params.adsr:
                adsr = node.params.adsr
                ks = float(getattr(node.params, "key_scale_rate", 0.0))
                mult = 2.0 ** (-ks * (note - 60) / 24.0) if ks else 1.0
                a_blocks = ms_to_blocks(adsr.a_ms * mult)
                d_blocks = ms_to_blocks(adsr.d_ms * mult)
                r_blocks = ms_to_blocks(adsr.r_ms * mult)
                s_q15 = int(adsr.s * 32767)
                vel_sens_q15 = int(
                    getattr(node.params, "velocity_sens", 1.0) * 32767)
                break
        self._release_blocks = r_blocks
        vel_scale_q15 = (32767 - vel_sens_q15
                         + ((vel_sens_q15 * velocity_q15) >> 15))

        if self.gate:
            if b < a_blocks:
                gain_q15 = int(32767 * b / a_blocks)
                phase = "ATTACK"
            elif b < a_blocks + d_blocks:
                t = (b - a_blocks) / d_blocks
                gain_q15 = int(32767 - (32767 - s_q15) * t)
                phase = "DECAY"
            else:
                gain_q15 = s_q15
                phase = "SUSTAIN"
            env_sustain_blocks = _GATE_HELD
        else:
            t = (b - self._note_off_block) / max(1, r_blocks)
            gain_q15 = int(self._release_from_q15 * max(0.0, 1.0 - t))
            phase = "RELEASE"
            env_sustain_blocks = self._note_off_block
        gain_q15 = max(0, min(32767, gain_q15))
        self._last_gain_q15 = gain_q15

        # --- Emphasis operators by envelope phase (simple-form mapping,
        #     same as the offline harness) ---
        phase_to_template = {"ATTACK": "T_ATTACK", "DECAY": "T_STABLE",
                             "SUSTAIN": "T_STABLE", "RELEASE": "T_RECOVERY"}
        templates = ir.layers.L3_cmmm.emit_templates
        ops_enable: list = []
        families: dict = {}
        for cand in (phase_to_template.get(phase, "T_STABLE"),
                     "T_STABLE", "T_IDLE"):
            if cand in templates:
                tmpl = get_emit_template(ir, cand)
                ops_enable = tmpl["ops_enable"]
                families = tmpl["families"]
                break

        # --- Mod-envelope bindings (rebuilt per block: knob edits latch) ---
        env_states = [_EnvelopeState(bind, note)
                      for bind in (getattr(ir, "envelopes", None) or [])]
        env_targets_q15 = None
        if env_states:
            env_targets_q15 = {}
            for es in env_states:
                value = es.value_q15(b, env_sustain_blocks)
                if es.velocity_sens_q15:
                    scale = (32767 - es.velocity_sens_q15
                             + ((es.velocity_sens_q15 * velocity_q15) >> 15))
                    value = (value * scale) >> 15
                for target_id in es.target_ids:
                    env_targets_q15[target_id] = value

        # --- LFO modulators (stateful objects on the IR) ---
        total_cents = 0
        gain_scale_q15 = 32767
        pwm_width = 16384
        pwm_enabled = False
        for node in ir.layers.L0_graph.nodes:
            if node.type == "Osc":
                if getattr(node.params, "pwm_width_q15", None) not in (None, 16384):
                    pwm_width = node.params.pwm_width_q15
                if getattr(node.params, "pw", 0.5) != 0.5:
                    pwm_width = int(node.params.pw * 32767)
                pwm_enabled = bool(getattr(node.params, "pwm_enable", False))
                break
        lfos = list(getattr(ir, "modulators", None) or [])
        for lfo in lfos:
            lfo.advance_block()
            lfo_val = lfo.get_lfo_value()
            for routing in lfo.routings:
                scaled = (lfo_val * routing.amount_q15 + 16384) >> 15
                if routing.target == LFOTarget.PITCH:
                    total_cents += scaled
                elif routing.target == LFOTarget.GAIN:
                    gain_scale_q15 = max(0, min(32767, 32767 + scaled))
                elif routing.target == LFOTarget.PWM_WIDTH:
                    pwm_width = max(1638, min(31130, pwm_width + scaled))

        # --- Spectrum sources (re-extracted per block: knob edits latch) ---
        harmonic_amps = extract_partials_q15(ir) or []
        phase_spread_q15 = extract_phase_spread_q15(ir)
        oeis_params = extract_oeis_params(ir)
        if not harmonic_amps and any(
            r.target == LFOTarget.PWM_WIDTH for lfo in lfos
            for r in lfo.routings
        ):
            pwm_enabled = True

        block_amps = harmonic_amps
        filter_params = extract_filter_params(ir) if harmonic_amps else None
        if filter_params and harmonic_amps:
            env_v = 32767
            filter_envs = [e for e in env_states if
                           int(EnvelopeTarget.FILTER_CUTOFF) in e.target_ids]
            if filter_envs:
                env_v = max(e.value_q15(b, env_sustain_blocks)
                            for e in filter_envs)
            sweep_oct = 4.0 * filter_params["env_amount"] * (env_v / 32767.0)
            eff_cutoff = filter_params["cutoff_hz"] * (2.0 ** sweep_oct)
            f0 = 440.0 * 2 ** ((note - 69) / 12)
            block_amps = _filtered_partials(
                harmonic_amps, f0, eff_cutoff, filter_params["resonance"])

        final_gain = ((((gain_q15 * gain_scale_q15 + 16384) >> 15)
                       * vel_scale_q15 + 16384) >> 15)

        # --- FM6 path ---
        for node in ir.layers.L0_graph.nodes:
            if node.type == "FM6":
                post_hz = getattr(node.params, "post_filter_hz", None)
                return VoiceFrame(
                    note=note,
                    cents=int(total_cents),
                    coeff_osc_mode=OSC_MODE_FM6,
                    gain_q15=final_gain,
                    beta_target_q15=0,
                    fm_algorithm_id=getattr(node.params, "algorithm", 1),
                    fm_op_ratios=getattr(node.params, "op_ratios",
                                         [65536] * 6),
                    fm_op_levels=getattr(node.params, "op_levels",
                                         [32767, 0, 0, 0, 0, 0]),
                    fm_feedback_depth=getattr(node.params, "feedback", 0),
                    fm_post_filter_enabled=bool(post_hz) or None,
                    fm_post_filter_cutoff_hz=post_hz,
                    env_targets_q15=env_targets_q15,
                    ops_enable=ops_enable,
                    families=families,
                )

        # --- Mode B path ---
        return VoiceFrame(
            note=note,
            cents=int(total_cents),
            coeff_osc_mode=1,
            H_request=len(harmonic_amps) or 16,
            gain_q15=final_gain,
            beta_target_q15=0,
            partial_amps_q15=block_amps,
            partial_phase_spread_q15=phase_spread_q15,
            ops_enable=ops_enable,
            families=families,
            waveform_type=extract_waveform(ir),
            oeis_sequence_id=oeis_params["sequence_id"] if oeis_params else -1,
            oeis_rolloff_db=oeis_params["rolloff_db"] if oeis_params else -3.0,
            pwm_width_q15=pwm_width,
            pwm_enabled=pwm_enabled,
            env_targets_q15=env_targets_q15,
        )


# ================================================================ OSC bridge

class StreamingOSCBridge:
    """
    Thin OSC shim over StreamingVoice (active when python-osc exists).

    /stream/note_on [note] [velocity]   (re)trigger the sustained note
    /stream/note_off                    release
    /stream/sweep  addr lo hi dur [loop]  start an axis sweep
    /stream/sweep_stop                  clear sweeps
    anything else                       -> voice.inject(address, value)
    """

    def __init__(self, voice: StreamingVoice,
                 listen_host: str = "0.0.0.0", listen_port: int = 7400):
        if not OSC_AVAILABLE:
            raise RuntimeError("python-osc not installed")
        self.voice = voice
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
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    def _route(self, address, *args):
        v = self.voice
        try:
            if address == "/stream/note_on":
                v.note_on(int(args[0]) if len(args) > 0 else None,
                          float(args[1]) if len(args) > 1 else None)
            elif address == "/stream/note_off":
                v.note_off()
            elif address == "/stream/sweep" and len(args) >= 4:
                loop = str(args[4]) if len(args) > 4 else "none"
                v.start_sweep(str(args[0]), float(args[1]), float(args[2]),
                              float(args[3]), loop=loop)
            elif address == "/stream/sweep_stop":
                v.clear_sweeps()
            else:
                v.inject(address, args[0] if args else None)
        except Exception as e:  # keep the OSC thread alive on bad input
            print(f"[stream] OSC {address}: {e}")


# ============================================================== measurements

def _write_wav(path: Path, samples: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes(np.asarray(samples, dtype=np.int16).tobytes())


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
            if not voice.play_realtime():
                while voice.active:  # headless fallback: free-run
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
