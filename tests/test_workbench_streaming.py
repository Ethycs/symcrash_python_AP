"""
Workbench Phase C streaming tests (WS4c).

Covers the per-block latch semantics (knob -> audio within <= 2 blocks),
sustained-streaming realtime stability, atlas-mode axis sweep
continuity (no inter-block clicks), gate/release behavior, and the thin
OSC adapter. Audio rendered per test is kept under ~1.5 s.
"""

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from symcrash_workbench.session import WorkbenchSession  # noqa: E402
from symcrash_workbench.streaming import (  # noqa: E402
    BLOCK_SIZE, BLOCKS_PER_SEC, BUDGET_US, OSC_AVAILABLE,
    StreamingVoice, boundary_step_metrics,
)

PRESET = REPO_ROOT / "presets" / "instruments" / "string_pad_v0.yaml"

# string_pad_v0: attack 160 ms + decay 290 ms => sustain from block ~338
SUSTAIN_BLOCK = 360


@pytest.fixture
def make_session(tmp_path):
    """Factory for isolated sessions on a scratch copy of the preset."""
    scratch = tmp_path / "scratch_string_pad.yaml"
    shutil.copy(PRESET, scratch)
    created = []

    def factory():
        s = WorkbenchSession(scratch, autoplay=False, log_dir=tmp_path)
        created.append(s)
        return s

    yield factory
    for s in created:
        s.close()


def _render_blocks(voice, n):
    return [voice.render_block() for _ in range(n)]


# ===================================================================== latch

def test_latch_within_two_blocks(make_session):
    """A param injected mid-sustain reaches the audio within <= 2 blocks."""
    inject_at = SUSTAIN_BLOCK + 40
    total = inject_at + 30

    control_voice = StreamingVoice(make_session())
    control_voice.note_on(60)
    control = _render_blocks(control_voice, total)
    control_voice.stop()

    voice = StreamingVoice(make_session())
    voice.note_on(60)
    injected = []
    for i in range(total):
        if i == inject_at:
            voice.inject("/amp/sustain", 0.25)
        injected.append(voice.render_block())
    voice.stop()

    first_diff = next(
        (i for i, (a, c) in enumerate(zip(injected, control))
         if not np.array_equal(a, c)), None)
    assert first_diff is not None, "injected param never reached the audio"
    assert first_diff >= inject_at, "audio changed before the injection"
    latency_blocks = first_diff - inject_at
    assert latency_blocks <= 2, (
        f"latch took {latency_blocks} blocks "
        f"({latency_blocks * 64 / 48:.2f} ms); per-block latch expects <= 2")
    # Well under the 50 ms knob->ear budget
    assert (latency_blocks + 1) * BLOCK_SIZE / 48000 * 1e3 < 50.0


def test_latch_last_value_wins_and_changed_tracked(make_session):
    """Multiple injections of one address in a block: last wins; SAVE sees it."""
    session = make_session()
    voice = StreamingVoice(session)
    voice.note_on(60)
    voice.inject("/amp/sustain", 0.10)
    voice.inject("/amp/sustain", 0.62)
    voice.render_block()
    assert abs(session.registry["/amp/sustain"].get(session.ir) - 0.62) < 1e-6
    # Routed through session.handle => tracked for YAML write-back
    assert abs(session.changed["/amp/sustain"] - 0.62) < 1e-6
    voice.stop()


def test_latch_wall_latency_recorded(make_session):
    """inject() -> latch wall time is recorded and is sub-50 ms."""
    voice = StreamingVoice(make_session())
    voice.note_on(60)
    voice.render_block()
    voice.inject("/amp/decay", 150.0)
    voice.render_block()
    assert len(voice.latch_log) == 1
    rec = voice.latch_log[0]
    assert rec.address == "/amp/decay"
    assert rec.wall_latency_s < 0.050


def test_play_message_retriggers_without_offline_render(make_session,
                                                        monkeypatch):
    """PLAY restarts the live attack without touching the offline renderer."""
    session = make_session()
    voice = StreamingVoice(session)
    voice.note_on(60)
    _render_blocks(voice, 12)
    assert voice.block_index == 12

    def fail_offline_render():
        raise AssertionError("PLAY entered the offline render path")

    monkeypatch.setattr(session, "render_and_play", fail_offline_render)
    voice.inject("/play", 1.0)
    voice.render_block()

    # note_on resets the clock, then the requested attack renders block 0.
    assert voice.block_index == 1
    assert session.streaming is True
    assert voice.latch_log[-1].address == "/play"
    voice.stop()
    assert session.streaming is False


def test_autoplay_retriggers_once_after_edit_debounce(make_session):
    """AUTO re-articulates after an edit rather than on every knob packet."""
    session = make_session()
    session.debounce = 0.01
    voice = StreamingVoice(session)
    voice.note_on(60)
    _render_blocks(voice, 12)

    voice.inject("/autoplay", 1.0)
    voice.render_block()
    assert session.autoplay is True
    assert voice.block_index == 13

    voice.inject("/amp/sustain", 0.25)
    voice.render_block()
    assert voice.block_index == 14  # edit is live; retrigger is debounced

    time.sleep(0.015)
    voice.render_block()
    assert voice.block_index == 1
    assert voice.latch_log[-1].address == "/amp/sustain"

    # It is a one-shot debounce action, not a retrigger every block.
    voice.render_block()
    assert voice.block_index == 2


def test_autoplay_off_leaves_live_stream_clock_running(make_session):
    """With AUTO off, edits latch without resetting the current note."""
    session = make_session()
    session.debounce = 0.0
    voice = StreamingVoice(session)
    voice.note_on(60)
    _render_blocks(voice, 8)

    voice.inject("/amp/sustain", 0.25)
    voice.render_block()

    assert session.autoplay is False
    assert voice.block_index == 9


def test_keyboard_note_events_latch_and_stale_release_is_ignored(make_session):
    """The surface keyboard controls the monophonic live gate safely."""
    voice = StreamingVoice(make_session())
    voice.note_on(60)
    _render_blocks(voice, 8)

    voice.inject("/stream/keyboard", (67, 1.0))
    voice.render_block()
    assert voice.note == 67
    assert voice.gate is True
    assert voice.block_index == 1

    # Traversing from one key to another can deliver the prior key's
    # release after the new note-on.  It must not release the new note.
    voice.inject("/stream/keyboard", (60, 0.0))
    voice.render_block()
    assert voice.note == 67
    assert voice.gate is True

    voice.inject("/stream/keyboard", (67, 0.0))
    voice.render_block()
    assert voice.gate is False
    assert voice.latch_log[-1].address == "/stream/keyboard"


def test_silent_service_accepts_next_keyboard_note(make_session):
    """Finishing a release must not strand the live OSC keyboard."""
    voice = StreamingVoice(make_session())
    voice.note_on(60)
    _render_blocks(voice, 8)
    voice.note_off()
    for _ in range(1000):
        voice.render_block()
        if not voice.active:
            break
    assert voice.active is False
    assert voice.stopped is False

    voice.inject("/stream/keyboard", (72, 1.0))
    audio = voice.render_block()

    assert voice.active is True
    assert voice.gate is True
    assert voice.note == 72
    assert audio.shape[0] == BLOCK_SIZE


# =========================================================== system manifold

def test_system_axis_registry_and_yaml_roundtrip(make_session):
    """The 11 controller-owned axes have saveable /system/* routes."""
    from ruamel.yaml import YAML

    session = make_session()
    expected = {
        "energy", "motion", "focus", "brightness", "density", "risk",
        "space", "material", "unity", "dream", "slip",
    }
    addresses = {a.removeprefix("/system/") for a in session.registry
                 if a.startswith("/system/")}
    assert addresses == expected
    assert session.ir.control is None

    param = session.registry["/system/dream"]
    assert param.get(session.ir) == 0.0
    assert param.set(session.ir, 0.625)
    assert session.ir.control is not None
    assert param.get(session.ir) == 0.625

    yaml = YAML()
    doc = yaml.load(session.preset_path.read_bytes().decode("utf-8"))
    root = doc["instrument_ir_v0"]
    assert param.yaml_apply(root, session.ir, 0.625)
    assert root["character"]["dream"] == 0.625


def test_system_axis_latches_with_macro_slew_and_changes_audio(make_session):
    """A hidden system axis reaches sound at a block boundary via its slew."""
    from symcrash_python_ap.primitives import SystemMacro

    inject_at = SUSTAIN_BLOCK + 20
    total = inject_at + 80

    control_voice = StreamingVoice(make_session())
    control_voice.note_on(60)
    control = _render_blocks(control_voice, total)
    control_voice.stop()

    session = make_session()
    voice = StreamingVoice(session)
    voice.note_on(60)
    injected = []
    for i in range(total):
        if i == inject_at:
            voice.inject("/system/dream", 1.0)
        injected.append(voice.render_block())

    dream_id = SystemMacro.DREAM
    current = voice.system_axes.runtime.get_system_macro(dream_id)
    target = voice.system_axes.runtime.macro_target_q15[dream_id]
    voice.stop()

    first_diff = next(
        (i for i, (a, b) in enumerate(zip(injected, control))
         if not np.array_equal(a, b)), None)
    assert first_diff is not None and first_diff >= inject_at
    assert 0 < current < target == 32767
    assert session.changed["/system/dream"] == 1.0


def test_offline_workbench_render_applies_system_projection(make_session,
                                                            monkeypatch):
    """Legacy --offline audition uses the same system-axis projection."""
    from symcrash_python_ap.frames import VoiceFrame

    seen = {}

    def fake_render(ir, **kwargs):
        transform = kwargs["frame_transform"]
        frame = transform(VoiceFrame(ops_enable=[], families={}))
        seen["ops"] = frame.ops_enable
        return np.zeros(BLOCK_SIZE, dtype=np.int16)

    monkeypatch.setattr("symcrash_workbench.session.render_preset_note",
                        fake_render)
    session = make_session()
    session._write_wav = lambda samples: None
    assert session.handle("/system/dream", 1.0)
    session.render()
    assert {"BandShear", "SwapBands", "RotateNodes"} <= set(seen["ops"])


# ====================================================================== gate

def test_note_off_release_then_inactive(make_session):
    """After note_off the voice renders its release tail and goes inactive."""
    voice = StreamingVoice(make_session())
    voice.note_on(60)
    _render_blocks(voice, 80)
    voice.note_off()
    tail = []
    for _ in range(2000):  # release is 320 ms => 240 blocks
        if not voice.active:
            break
        tail.append(voice.render_block())
    assert not voice.active, "voice never finished its release"
    assert 100 < len(tail) < 600
    # Tail decays to (near) silence
    assert int(np.abs(tail[-1].astype(np.int32)).max()) < 256


# ================================================================= streaming

def test_streaming_smoke_realtime_budget(make_session):
    """~1 s sustained stream: finite, audible, mean block time in budget."""
    voice = StreamingVoice(make_session())
    voice.note_on(60)
    for _ in range(100):  # warmup (numba kernels load from cache)
        voice.render_block()
    n = BLOCKS_PER_SEC  # 1 second
    times = np.empty(n)
    peak = 0
    for i in range(n):
        t0 = time.perf_counter()
        block = voice.render_block()
        times[i] = time.perf_counter() - t0
        peak = max(peak, int(np.abs(block.astype(np.int32)).max()))
    voice.stop()
    times_us = times * 1e6
    assert peak > 1000, "sustained stream is silent"
    assert times_us.mean() < BUDGET_US, (
        f"mean block time {times_us.mean():.0f} us over the "
        f"{BUDGET_US:.0f} us realtime budget")
    assert np.percentile(times_us, 95) < 2 * BUDGET_US


# ================================================================ atlas mode

def test_axis_sweep_continuity(make_session):
    """A live axis sweep latched per block must not add boundary clicks."""
    voice = StreamingVoice(make_session())
    voice.note_on(60)
    _render_blocks(voice, SUSTAIN_BLOCK)  # reach sustain

    static = np.concatenate(_render_blocks(voice, 200))
    voice.start_sweep("/axis/stable/warp", 0.0, 1.0, 200 / BLOCKS_PER_SEC)
    swept = np.concatenate(_render_blocks(voice, 200))
    voice.stop()

    # Sweep is audible
    diff = np.abs(swept.astype(np.int32) - static.astype(np.int32)).max()
    assert diff > 100, "axis sweep had no audible effect"

    static_m = boundary_step_metrics(static)
    sweep_m = boundary_step_metrics(swept)
    # Click check: block-boundary steps stay within the signal's own
    # intra-block step scale (a per-block discontinuity would spike them)
    bound = max(static_m["intra_max"], sweep_m["intra_max"]) * 1.5 + 64
    assert sweep_m["boundary_max"] <= bound, (
        f"sweep boundary step {sweep_m['boundary_max']} LSB vs "
        f"intra-block max {sweep_m['intra_max']} LSB: per-block click")


def test_sweep_endpoint_clamps_and_retires(make_session):
    """A non-looping sweep parks at its end value and deactivates."""
    session = make_session()
    voice = StreamingVoice(session)
    voice.note_on(60)
    voice.render_block()
    sweep = voice.start_sweep("/axis/stable/glue", 0.0, 0.8,
                              10 / BLOCKS_PER_SEC)
    _render_blocks(voice, 20)
    assert not sweep.active
    assert not voice._sweeps
    v = session.registry["/axis/stable/glue"].get(session.ir)
    assert abs(v - 0.8) < 1e-3
    voice.stop()


def test_pingpong_sweep_returns_to_start(make_session):
    """Pingpong sweep: value returns near the start after a full cycle."""
    session = make_session()
    voice = StreamingVoice(session)
    voice.note_on(60)
    voice.render_block()
    voice.start_sweep("/axis/stable/warp", 0.0, 1.0,
                      20 / BLOCKS_PER_SEC, loop="pingpong")
    _render_blocks(voice, 20)  # peak of the triangle
    near_peak = session.registry["/axis/stable/warp"].get(session.ir)
    _render_blocks(voice, 20)  # back down
    near_start = session.registry["/axis/stable/warp"].get(session.ir)
    voice.stop()
    assert near_peak > 0.9
    assert near_start < 0.1


# ================================================================ OSC bridge

@pytest.mark.skipif(not OSC_AVAILABLE, reason="python-osc not installed")
def test_osc_bridge_routes_to_latch(make_session):
    """OSC knob message -> inject queue -> latched on the next block."""
    from pythonosc.udp_client import SimpleUDPClient
    from symcrash_workbench.streaming import StreamingOSCBridge

    session = make_session()
    voice = StreamingVoice(session)
    voice.note_on(60)
    bridge = StreamingOSCBridge(voice, listen_host="127.0.0.1",
                                listen_port=0).start()
    try:
        client = SimpleUDPClient("127.0.0.1", bridge.port)
        client.send_message("/amp/sustain", 0.33)
        deadline = time.monotonic() + 2.0
        latched = False
        while time.monotonic() < deadline:
            voice.render_block()
            v = session.registry["/amp/sustain"].get(session.ir)
            if abs(v - 0.33) < 1e-6:
                latched = True
                break
        assert latched, "OSC message never latched into the stream"
        assert voice.latch_log[-1].address == "/amp/sustain"

        # The controller-owned manifold uses the same OSC -> queue ->
        # block-latch path as ordinary patch parameters.
        client.send_message("/system/dream", 0.75)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            voice.render_block()
            if abs(session.registry["/system/dream"].get(session.ir)
                   - 0.75) < 1e-6:
                break
        assert abs(session.ir.control.dream - 0.75) < 1e-6
        assert voice.latch_log[-1].address == "/system/dream"

        client.send_message("/autoplay", 1.0)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not session.autoplay:
            voice.render_block()
        assert session.autoplay is True

        client.send_message("/play", 1.0)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            voice.render_block()
            if voice.latch_log[-1].address == "/play":
                break
        assert voice.latch_log[-1].address == "/play"
        assert voice.gate is True

        client.send_message("/stream/keyboard", [69, 1.0])
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            voice.render_block()
            if voice.note == 69:
                break
        assert voice.note == 69
        assert voice.gate is True

        client.send_message("/stream/keyboard", [69, 0.0])
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and voice.gate:
            voice.render_block()
        assert voice.gate is False

        # Native patchbays send input plus zero or more outputs. The bridge
        # must retain the full OSC argument list until the block boundary.
        client.send_message("/editor/graph", ["node1", "audio"])
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            voice.render_block()
            if not any(edge.src == session.ir.layers.L0_graph.nodes[0].id
                       for edge in session.ir.layers.L0_graph.edges):
                break
        assert not any(edge.src == session.ir.layers.L0_graph.nodes[0].id
                       for edge in session.ir.layers.L0_graph.edges)
    finally:
        bridge.stop()
        voice.stop()
