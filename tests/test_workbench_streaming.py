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


def test_play_message_ignored_mid_stream(make_session):
    """/play must not trigger the offline render path during streaming."""
    session = make_session()
    voice = StreamingVoice(session)
    voice.note_on(60)
    voice.inject("/play", 1.0)
    voice.render_block()  # would race clear_basis_atlas if not ignored
    assert session.streaming is True
    assert len(voice.latch_log) == 0
    voice.stop()
    assert session.streaming is False


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
    finally:
        bridge.stop()
        voice.stop()
