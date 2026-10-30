"""ControlFrame command delivery, preset switching, and audible API parity."""
from copy import deepcopy
from pathlib import Path

import pytest

from symcrash_python_ap.cockpit_engine import CockpitEngine
from symcrash_python_ap.control_frame import (
    CaptureMark, ControlEvent, ControlFrame, EventType, GlobalControls,
)
from symcrash_python_ap.library.preset_compiler import PresetCompiler
from symcrash_python_ap.synth_master import SynthConfig, SynthMaster
from symcrash_python_ap.voice import VoiceEngine
from symcrash_python_ap.voice.voice_engine import EngineMode, VoicePhase


def preset(name='first', harmonics=None):
    result = PresetCompiler().compile({
        'preset': {'preset_id': name},
        'env': {'amp': {'attack_ms': 1, 'decay_ms': 1, 'sustain': .8, 'release_ms': 4}},
    })
    if harmonics is not None:
        result.harmonic_amps_q15 = harmonics
    return result


def test_load_changes_following_notes_without_rewriting_sounding_voices():
    engine = VoiceEngine(max_voices=3)
    first, second = preset(), preset('second', [32767])
    engine.register_control_preset(11, first, 0xABCDEF12)
    engine.register_control_preset(22, second)
    frame = ControlFrame()
    frame.add_preset_load(11, 0xABCDEF12)
    frame.add_note_on(60, 32767)
    frame.add_preset_load(22)
    frame.add_note_on(64, 32767)
    output = engine.step_block(frame)
    assert frame.events == []
    assert engine.voices[0].compiled_preset is first
    assert engine.voices[1].compiled_preset is second
    assert [i.frame.partial_amps_q15 for i in output.render_batch] == [
        first.harmonic_amps_q15, second.harmonic_amps_q15]
    assert engine.selected_preset_id == 22
    assert all(v.engine_mode == EngineMode.SIMPLE for v in engine.voices.values())
    next_frame = ControlFrame()
    next_frame.add_note_on(60, 20000)  # Same note now retriggers the selected sound.
    engine.consume_control_frame(next_frame)
    assert len(engine.voices) == 2
    assert engine.voices[0].compiled_preset is second


@pytest.mark.parametrize('slot_state', ['unallocated', 'idle', 'release', 'explicit'])
def test_selected_preset_applies_to_every_allocation_path(slot_state):
    engine = VoiceEngine(max_voices=1)
    old, new = preset(), preset('new')
    if slot_state in ('idle', 'release'):
        engine.allocate_voice_from_preset(old, 0)
    if slot_state == 'release':
        engine.note_on_simple(0, 60)
        engine.note_off_simple(0)
    engine.select_compiled_preset(new)
    frame = ControlFrame()
    frame.add_note_on(65, 32767)
    if slot_state == 'explicit':
        frame.events[-1].voice_id = 0
    output = engine.step_block(frame)
    assert engine.voices[0].compiled_preset is new
    assert output.render_batch[0].frame.note == 65
    assert output.render_batch[0].frame.is_starting


@pytest.mark.parametrize('command,error', [
    (ControlEvent(EventType.PRESET_LOAD, a=99), LookupError),
    (ControlEvent(EventType.PRESET_LOAD, a=1, b=123), ValueError),
    (ControlEvent(EventType.PRESET_LOAD, a=-1), ValueError),
    (ControlEvent(EventType.MODE_SET, a=6), ValueError),
    (ControlEvent(EventType.MODE_SET, a=2, b=1), ValueError),
    (ControlEvent(EventType.CAPTURE_MARK, a=2**32), ValueError),
    (ControlEvent(EventType.CAPTURE_MARK, b=1), ValueError),
    (ControlEvent(255), ValueError),
])
def test_invalid_commands_leave_events_controls_and_voices_untouched(command, error):
    engine = VoiceEngine()
    engine.register_control_preset(1, preset(), 456)
    frame = ControlFrame(mode_L=5, globals=GlobalControls(master_gain_q15=0))
    frame.add_note_on(60, 32767)
    frame.add_capture_mark(5)
    frame.events.append(command)
    before = deepcopy(frame)
    with pytest.raises(error):
        engine.consume_control_frame(frame)
    assert frame == before
    assert engine.master_gain_q15 == 32767 and engine.control_mode_L == 3
    assert not engine.voices and not engine.drain_capture_marks()


def test_unregistered_preset_can_be_registered_then_same_frame_retried_once():
    engine = VoiceEngine()
    frame = ControlFrame()
    frame.add_preset_load(0xFFFFFFFF, 0xFFFFFFFF)
    frame.add_note_on(69, 32767)
    with pytest.raises(LookupError):
        engine.consume_control_frame(frame)
    engine.register_control_preset(0xFFFFFFFF, preset(), 0xFFFFFFFF)
    assert engine.step_block(frame).render_batch[0].frame.is_starting
    assert not engine.step_block(frame).render_batch[0].frame.is_starting


@pytest.mark.parametrize('value', [-1, 2**32, 1.5])
def test_registration_requires_wire_ids_and_hashes(value):
    engine = VoiceEngine()
    with pytest.raises(ValueError):
        engine.register_control_preset(value, preset())
    with pytest.raises(ValueError):
        engine.register_control_preset(1, preset(), value)


def test_mode_and_capture_metadata_survive_replay_without_changing_dsp_resolution():
    frame = ControlFrame(block_index=0xFFFFFFFF, mode_L=1)
    frame.add_note_on(60, 32767)
    frame.add_mode_set(5)
    frame.add_capture_mark(42, channel=7, voice_id=0)
    frame.add_capture_mark(43)
    outputs = []
    for _ in range(2):
        engine = VoiceEngine()
        expected_tower = engine.default_program.attack_regime
        output = engine.step_block(deepcopy(frame))
        outputs.append(output)
        assert engine.control_mode_L == 5
        assert engine.voices[0].regime == expected_tower
        assert engine.voices[0].tower_level != 5
        assert output.capture_marks == [CaptureMark(0xFFFFFFFF, 42, 7, 0),
                                        CaptureMark(0xFFFFFFFF, 43)]
        assert engine.drain_capture_marks() == []
        assert engine.step_block().capture_marks == []
        assert engine.control_mode_L == 5
    assert outputs[0].render_batch == outputs[1].render_batch
    assert outputs[0].capture_marks == outputs[1].capture_marks
    baseline = ControlFrame()
    baseline.add_note_on(60, 32767)
    assert VoiceEngine().step_block(baseline).render_batch == outputs[0].render_batch


def test_capture_can_be_drained_before_stepping_and_header_updates_mode():
    engine = VoiceEngine()
    frame = ControlFrame(block_index=73, mode_L=0)
    frame.add_capture_mark(99)
    engine.consume_control_frame(frame)
    assert engine.block_count == 0 and engine.control_mode_L == 0
    assert engine.drain_capture_marks() == [CaptureMark(73, 99)]
    engine.consume_control_frame(frame)
    assert engine.drain_capture_marks() == []
    engine.consume_control_frame(ControlFrame(mode_L=2))
    assert engine.control_mode_L == 2


@pytest.mark.parametrize('mode', [-1, 6, 1.5])
def test_invalid_mode_header_is_rejected_before_notes(mode):
    frame = ControlFrame(mode_L=mode)
    frame.add_note_on(60, 32767)
    engine = VoiceEngine()
    with pytest.raises(ValueError):
        engine.step_block(frame)
    assert frame.events and not engine.voices and engine.block_count == 0


def test_cockpit_emits_commands_in_order_and_latches_mode_across_ticks():
    cockpit = CockpitEngine()
    cockpit.load_preset(3, 4)
    cockpit.set_mode(1)
    cockpit.midi_note_on(65, 127)
    cockpit.capture_mark(2)
    cockpit.set_mode(5)
    assert cockpit.mode_L == 3  # Commands take effect at the next block.
    frame = cockpit.tick()
    assert [e.type for e in frame.events] == [EventType.PRESET_LOAD, EventType.MODE_SET,
        EventType.NOTE_ON, EventType.CAPTURE_MARK, EventType.MODE_SET]
    assert frame.mode_L == 5 and frame.block_index == 0
    engine = VoiceEngine()
    engine.register_control_preset(3, preset(), 4)
    output = engine.step_block(frame)
    assert output.capture_marks == [CaptureMark(0, 2)]
    assert engine.control_mode_L == 5
    next_frame = cockpit.tick()
    assert next_frame.mode_L == 5 and not next_frame.events
    assert engine.step_block(next_frame).capture_marks == []
    assert engine.control_mode_L == 5


@pytest.mark.parametrize('action,value', [('load_preset', -1), ('capture_mark', 2**32),
                                         ('set_mode', 6)])
def test_cockpit_rejects_invalid_command_before_queueing(action, value):
    cockpit = CockpitEngine()
    with pytest.raises(ValueError):
        getattr(cockpit, action)(value)
    assert not cockpit.tick().events


@pytest.mark.parametrize('yaml_name', ['saw_lead_v0.yaml', 'fm_bell_v0.yaml'])
def test_cockpit_and_direct_notes_render_identical_loaded_presets(yaml_name):
    path = Path(__file__).resolve().parents[2] / 'presets' / 'instruments' / yaml_name
    a, b = SynthMaster(SynthConfig(n_voices=1)), SynthMaster(SynthConfig(n_voices=1))
    assert a.load_preset_from_yaml(str(path)) and b.load_preset_from_yaml(str(path))
    a.note_on(69, 100)
    b.cockpit.midi_note_on(69, 100)
    samples = []
    for _ in range(20):
        left, right, _ = a.render_block()
        other_l, other_r, _ = b.render_block()
        assert (left, right) == (other_l, other_r)
        samples.extend(left)
    assert any(samples)
    a.note_off(note=69)
    b.cockpit.midi_note_off(69)
    for _ in range(20):
        assert a.render_block()[:2] == b.render_block()[:2]


def test_synthmaster_commands_and_direct_notes_share_selected_sound_and_release():
    synth = SynthMaster(SynthConfig(n_voices=1))
    sound = preset()
    synth.register_control_preset(7, sound, 8)
    synth.cockpit.load_preset(7, 8)
    synth.cockpit.capture_mark(9)
    synth.cockpit.midi_note_on(60, 127)
    synth.render_block()
    assert synth.last_capture_marks == [CaptureMark(0, 9)]
    assert synth._get_compiled_preset() is sound
    assert synth.get_preset_name() == sound.preset_id
    assert synth.get_active_voice_count() == 1
    assert synth.note_on(60, 80) == 0  # Retrigger uses the same canonical slot.
    synth.render_block()
    assert synth.last_capture_marks == []
    synth.note_off(voice_id=0)
    for _ in range(8):
        synth.render_block()
    assert synth.get_active_voice_count() == 0
    assert synth.voice_engine.voices[0].phase == VoicePhase.FREE
    assert synth.note_on(65) == 0
    assert synth.get_active_voice_count() == 1
    assert synth.voice_engine.voices[0].compiled_preset is sound
    replacement = preset('replacement')
    synth.load_compiled_preset(replacement)
    assert synth.get_preset_name() == 'replacement'
    assert synth.note_on(65) == 0
    assert synth.voice_engine.voices[0].compiled_preset is replacement


def test_direct_default_notes_preserve_pitch_and_zero_velocity_means_release():
    synth = SynthMaster(SynthConfig(n_voices=1))
    assert synth.note_on(72) == 0
    assert synth.voice_engine.voices[0].note_u8 == 72
    assert synth.get_active_voice_count() == 1
    assert synth.note_on(72, 0) is None
    assert synth.voice_engine.voices[0].phase == VoicePhase.RELEASE


def test_compatibility_apply_delegates_commands_and_does_not_advance_time():
    synth = SynthMaster(SynthConfig(n_voices=1))
    frame = ControlFrame(mode_L=4)
    frame.add_note_on(62, 32767)
    frame.add_capture_mark(1)
    synth._apply_control_frame(frame)
    assert synth.voice_engine.block_count == 0
    assert synth.voice_engine.control_mode_L == 4
    assert synth.voice_engine.voices[0].note_u8 == 62
    assert not frame.events
    assert synth.voice_engine.drain_capture_marks() == [CaptureMark(0, 1)]


def test_single_emitter_preserves_capture_address_and_adopts_selected_preset():
    from symcrash_python_ap.voiceframe_emitter import VoiceFrameEmitter
    emitter = VoiceFrameEmitter(preset(), voice_id=7)
    new = preset('new')
    emitter._engine.register_control_preset(1, new)
    frame = ControlFrame(block_index=50)
    frame.add_preset_load(1)
    frame.add_capture_mark(2, voice_id=7)
    frame.add_capture_mark(3)
    frame.add_capture_mark(4, voice_id=8)
    emitter.consume_control_frame(frame)
    emitter.note_on(69)
    assert emitter.voice_state.compiled_preset is new
    assert emitter.emit_frame().note == 69
    assert emitter._engine.drain_capture_marks() == [CaptureMark(50, 2, voice_id=7), CaptureMark(50, 3)]
    assert len(frame.events) == 1 and frame.events[0].voice_id == 8
