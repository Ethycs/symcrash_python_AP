"""Pico2 -> Cockpit -> ControlFrame -> live voice/audio integration.

Inputs are sampled at block boundaries. The bridge takes a CockpitEngine,
not a VoiceEngine; the serial callback never advances musical time.
"""
from copy import deepcopy

import numpy as np
import pytest

from symcrash_python_ap.cockpit_engine import CockpitEngine
from symcrash_python_ap.comm.pico2_synth_bridge import Pico2SynthBridge
from symcrash_python_ap.control_registry import ControlID
from symcrash_python_ap.envelope_scheduler import EnvelopePhase
from symcrash_python_ap.library.preset_compiler import PresetCompiler
from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.voiceframe_emitter import VoiceFrameEmitter


@pytest.fixture
def live():
    cockpit = CockpitEngine()
    bridge = Pico2SynthBridge(cockpit, virtual_mode=True)
    preset = PresetCompiler().compile({
        "preset": {"preset_id": "control-integration"},
        "env": {"amp": {"attack_ms": 1, "decay_ms": 1, "sustain": 0.75, "release_ms": 40}},
    })
    emitter = VoiceFrameEmitter(preset)
    return cockpit, bridge, emitter


def tick(live, count=1):
    cockpit, _, emitter = live
    for _ in range(count):
        emitter.consume_control_frame(cockpit.tick())
        frame = emitter.emit_frame()
    return frame


@pytest.mark.parametrize("pot,control,expected", [
    (0, ControlID.ATK_BLOCKS, 500), (1, ControlID.DCY_BLOCKS, 1000),
    (2, ControlID.SUS_Q15, 32767), (3, ControlID.REL_BLOCKS, 1500),
    (4, ControlID.CHAOS, 32767), (5, ControlID.TEXTURE, 32767),
    (6, ControlID.RESERVED_7, 0), (7, ControlID.MORPH, 32767),
    (8, ControlID.DETUNE, 2399), (9, ControlID.MASTER_GAIN, 32767),
])
def test_hardware_pots_route_through_cockpit(live, pot, control, expected):
    cockpit, bridge, _ = live
    before = cockpit.get_control_current(control)
    bridge.receiver.inject_macro(pot, 32767)
    assert cockpit.get_control_target(control) == expected
    assert cockpit.get_control_current(control) == before
    tick(live, 600)
    assert abs(cockpit.get_control_current(control) - expected) <= 8


@pytest.mark.parametrize("pot,control,expected", [
    (0, ControlID.ATK_BLOCKS, 1), (1, ControlID.DCY_BLOCKS, 1),
    (3, ControlID.REL_BLOCKS, 1), (8, ControlID.DETUNE, -2400),
])
def test_pot_minimum_mapping(live, pot, control, expected):
    cockpit, bridge, _ = live
    bridge.receiver.inject_macro(pot, 0)
    assert cockpit.get_control_target(control) == expected


def test_attack_and_decay_controls_drive_envelope_without_retrigger(live):
    cockpit, bridge, emitter = live
    original = deepcopy(emitter.preset.amp_env)
    bridge.receiver.inject_macro(0, 32767)
    bridge.receiver.inject_macro(1, 32767)
    tick(live, 100)
    cockpit.midi_note_on(60, 127)
    gains = [tick(live).gain_q15 for _ in range(12)]
    assert gains[0] == 0 and 0 < gains[-1] < 1000
    assert gains == sorted(gains)
    env = emitter.voice_state.amp_envelope
    assert env.phase == EnvelopePhase.ATTACK and env.phase_block == 12
    assert env._attack_blocks == 500 and env._decay_blocks == 1000
    assert emitter.preset.amp_env == original


def test_sustain_and_release_controls_change_audible_gain(live):
    cockpit, bridge, emitter = live
    bridge.receiver.inject_macro(2, 8192)
    bridge.receiver.inject_macro(3, 32767)
    tick(live, 100)
    cockpit.midi_note_on(60, 127)
    frame = tick(live, 12)
    assert 8180 <= frame.gain_q15 <= 8192
    cockpit.midi_note_off(60)
    frames = [tick(live) for _ in range(100)]
    assert frames[-1].is_active and frames[-1].gain_q15 > 7000
    assert frames[-1].gain_q15 < frames[0].gain_q15
    assert emitter.voice_state.amp_envelope._release_blocks == 1500


def test_pitch_snapshots_replace_each_component_and_replay_idempotently(live):
    cockpit, bridge, emitter = live
    cockpit.midi_note_on(69, 127)
    bridge.receiver.inject_macro(8, 24576)
    cockpit.set_control(ControlID.PITCH_BEND, 100)
    control_frame = cockpit.tick()
    emitter.consume_control_frame(control_frame)
    control_frame.events.clear()  # Replay only controls, not the note trigger.
    emitter.consume_control_frame(control_frame)
    assert emitter.emit_frame().cents == 1300
    cockpit.set_control(ControlID.PITCH_BEND, -50)
    assert tick(live).cents == 1150
    bridge.receiver.inject_macro(8, 16384)
    assert tick(live).cents == -50
    cockpit.set_control(ControlID.PITCH_BEND, 0)
    assert tick(live).cents == 0


def test_manual_detune_override_restores_latched_controls(live):
    cockpit, _, emitter = live
    cockpit.midi_note_on(69, 127)
    cockpit.set_control(ControlID.DETUNE, 1200)
    cockpit.set_control(ControlID.PITCH_BEND, 100)
    tick(live)
    emitter.set_detune(-1200)
    assert emitter.emit_frame().cents == -1100
    emitter.set_detune(None)
    assert emitter.emit_frame().cents == 1300


def test_morph_can_return_to_zero_over_nonzero_preset(live):
    cockpit, bridge, emitter = live
    emitter.preset.latch.tree_blend_q15 = 12000
    cockpit.midi_note_on(60, 127)
    assert tick(live).tree_blend_q15 == 12000
    bridge.receiver.inject_macro(7, 32767)
    assert tick(live, 200).tree_blend_q15 == 32767
    bridge.receiver.inject_macro(7, 0)
    assert tick(live, 200).tree_blend_q15 == 0


def test_gain_control_mutes_rendered_audio(live):
    cockpit, bridge, _ = live
    stage = UnifiedOutputStage()
    stage.voice_on(0, 60)
    cockpit.midi_note_on(60, 127)
    audible = []
    for _ in range(30):
        left, right = stage.render_block(tick(live))
        audible.extend(left)
        assert np.max(np.abs(np.asarray(left, dtype=int) - right)) <= 1
    assert np.max(np.abs(audible)) > 100
    bridge.receiver.inject_macro(9, 0)
    for _ in range(160):
        left, _ = stage.render_block(tick(live))
    assert np.max(np.abs(left)) < 10


def test_virtual_bridge_and_display(live):
    _, bridge, _ = live
    assert bridge.port is None
    assert bridge.start()
    bridge.receiver.inject_macro(3, 32767)
    tick(live, 100)
    display = bridge.get_control_display()
    assert set(display) == {"ATK", "DCY", "SUS", "REL", "CUTOFF", "RESO", "HARM", "ETA", "DETUNE", "GAIN"}
    assert display["REL"] == "2000ms"
    bridge.stop()
