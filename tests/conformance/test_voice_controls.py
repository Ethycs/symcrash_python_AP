"""ControlFrame v0 contracts at the canonical engine and compatibility APIs."""
from copy import deepcopy

import pytest

from symcrash_python_ap.cockpit_engine import CockpitEngine
from symcrash_python_ap.control_frame import ControlFrame, ControlItem, VoiceControlPatch, GlobalControls
from symcrash_python_ap.control_registry import ControlID as C, ControlRegistry, ValueType
from symcrash_python_ap.library.preset_compiler import PresetCompiler, BackboneOp
from symcrash_python_ap.voice import VoiceEngine
from symcrash_python_ap.voice.voice_engine import VoicePhase
from symcrash_python_ap.voiceframe_emitter import MultiVoiceEmitter, VoiceFrameEmitter
from symcrash_python_ap.vm.decode import decode_frame


def preset():
    result = PresetCompiler().compile({
        'preset': {'preset_id': 'controls'},
        'env': {'amp': {'attack_ms': 1, 'decay_ms': 1, 'sustain': .75, 'release_ms': 40}},
    })
    result.backbone = [BackboneOp(op=name, strength_q15=16384) for name in
                       ('Tilt', 'IntraMix', 'FoldPressure', 'FineClamp')]
    return result


def patch(voice_id=0, **values):
    registry = ControlRegistry()
    return VoiceControlPatch(voice_id, [ControlItem(int(C[name]), registry.get_entry(C[name]).value_type, value)
                                       for name, value in values.items()])


def engine(voices=2):
    result = VoiceEngine(max_voices=voices)
    p = preset()
    for vid in range(voices):
        result.allocate_voice_from_preset(p, vid)
    return result


def note_frame(note=69, channel=0, voice_id=None):
    frame = ControlFrame()
    frame.add_note_on(note, 32767, channel)
    if voice_id is not None:
        frame.events[-1].voice_id = voice_id
    return frame


def test_canonical_api_keeps_preallocated_simple_preset_and_consumes_events_once():
    e = engine()
    cf = note_frame()
    first = e.step_block(cf).render_batch[0].frame
    second = e.step_block(cf).render_batch[0].frame
    assert first.is_starting and not second.is_starting
    # One-block attack reaches the peak immediately, then decays to sustain.
    assert second.gain_q15 < first.gain_q15
    assert first.f0_hz == 440.0
    assert cf.events == []
    assert e.voices[0].compiled_preset is not None


def test_voice_zero_is_not_a_broadcast_and_controls_are_absolute():
    e = engine()
    e.consume_control_frame(note_frame(69, voice_id=0))
    e.consume_control_frame(note_frame(69, voice_id=1))
    cf = ControlFrame(voice_patches=[patch(0, DETUNE=1200, PITCH_BEND=100, DYNAMICS=0, MORPH=0)])
    for _ in range(2):
        e.consume_control_frame(cf)
    frames = [item.frame for item in e.step_block().render_batch]
    assert frames[0].cents == 1300 and frames[1].cents == 0
    assert frames[0].gain_q15 == 0
    assert frames[0].tree_blend_q15 == 0
    assert e.voices[0].controls is not e.voices[1].controls


def test_polyphonic_controls_match_single_voice_output():
    single, multi = VoiceFrameEmitter(preset()), MultiVoiceEmitter(preset(), max_voices=2)
    cf = note_frame()
    cf.voice_patches = [patch(0, DETUNE=-100, PITCH_BEND=50, ATK_BLOCKS=4,
                             DCY_BLOCKS=5, SUS_Q15=12345, REL_BLOCKS=20, MORPH=2000,
                             TIMBRE=8192, TEXTURE=0, CHAOS=16384, DYNAMICS=24576)]
    cf.globals.master_gain_q15 = 16000
    single.consume_control_frame(deepcopy(cf))
    multi.consume_control_frame(deepcopy(cf))
    for _ in range(16):
        assert single.emit_frame() == multi.emit_frames()[0]
    single.note_off()
    multi.note_off(69)
    for _ in range(10):
        assert single.emit_frame() == multi.emit_frames()[0]


@pytest.mark.parametrize('name,op_id', [('TIMBRE', 1), ('TEXTURE', 3), ('CHAOS', 20)])
@pytest.mark.parametrize('value', [0, 8192, 32767])
def test_family_scales_multiply_authored_strength_and_leave_other_families(name, op_id, value):
    e = engine(1)
    original = deepcopy(e.voices[0].compiled_preset)
    cf = note_frame()
    cf.voice_patches = [patch(**{name: value})]
    frame = e.step_block(cf).render_batch[0].frame
    decoded = {op.op_id: op.strength_q15 for op in decode_frame(frame).backbone_instrs}
    expected = int(16384 * value / 32767)
    assert abs(decoded[op_id] - expected) <= 1
    for other in {1, 3, 20, 2} - {op_id}:
        assert decoded[other] == 16384
    assert e.voices[0].compiled_preset == original


def test_sparse_globals_latch_and_reset_clears_them():
    multi = MultiVoiceEmitter(preset(), max_voices=2)
    cf = note_frame()
    cf.globals = GlobalControls(master_gain_q15=0, master_pan_q15=15000, tempo_q8_8=90 << 8)
    multi.consume_control_frame(cf)
    multi.consume_control_frame(ControlFrame())
    multi.emit_frames()
    frame = multi.emit_frames()[0]
    assert frame.gain_q15 == 0 and frame.pan_q15 == 15000
    assert multi._engine.step_block().param_frame.gate_q15 == 0
    assert multi._engine.tempo_q8_8 == 90 << 8
    multi.reset()
    assert multi._engine.master_gain_q15 == 32767
    assert multi._engine.master_pan_q15 == 0


def test_note_channels_explicit_slots_and_off_before_on():
    e = engine()
    cf = note_frame(60, channel=1, voice_id=1)
    cf.add_note_on(60, 32767, channel=2)
    e.consume_control_frame(cf)
    assert e.voices[1].channel == 1 and e.voices[0].channel == 2
    cf = note_frame(60, channel=1, voice_id=1)
    cf.add_note_off(60, channel=1)  # Must release the old note before retriggering.
    e.consume_control_frame(cf)
    assert all(v.phase == VoicePhase.ATTACK for v in e.voices.values())
    cf = ControlFrame()
    cf.add_note_off(60, channel=2)
    e.consume_control_frame(cf)
    assert e.voices[0].phase == VoicePhase.RELEASE
    assert e.voices[1].phase == VoicePhase.ATTACK


def test_repeated_note_does_not_orphan_a_voice_and_stolen_note_off_is_safe():
    multi = MultiVoiceEmitter(preset(), max_voices=1)
    assert multi.note_on(60) == multi.note_on(60) == 0
    multi.note_off(60)
    assert multi.note_on(62) == 0
    multi.note_off(60)  # The old note must not release the reused slot.
    assert multi._engine.voices[0].phase == VoicePhase.ATTACK
    multi.note_off(62)
    assert multi._engine.voices[0].phase == VoicePhase.RELEASE


def test_cockpit_pool_patches_include_later_targets_and_idle_slots():
    cockpit = CockpitEngine()
    cockpit.set_control(C.DETUNE, 1200)
    first = cockpit.tick(voice_ids=[0, 1])
    assert [p.voice_id for p in first.voice_patches] == [0, 1]
    e = VoiceEngine(max_voices=3)
    e.consume_control_frame(first)  # Latch before allocation.
    later = cockpit.tick(voice_ids=[0, 1, 2])
    assert [p.voice_id for p in later.voice_patches] == [2]
    e.consume_control_frame(later)
    for vid in range(3):
        e.allocate_voice_from_preset(preset(), vid)
        e.note_on_simple(vid, 69)
    assert [i.frame.f0_hz for i in e.step_block().render_batch] == [880.] * 3


def test_type_mismatch_is_rejected_reserved_ids_ignored_and_values_clamped():
    e = engine(1)
    cf = ControlFrame(voice_patches=[patch(DETUNE=9000, PITCH_BEND=-9000, ATK_BLOCKS=0)])
    e.consume_control_frame(cf)
    controls = e.voices[0].controls
    assert controls.pitch_offset_cents == 2200
    assert controls.values[C.ATK_BLOCKS] == 1
    e.consume_control_frame(ControlFrame(voice_patches=[VoiceControlPatch(0, [ControlItem(7, ValueType.Q15U, 5)])]))
    with pytest.raises(ValueError, match='Wrong value type'):
        e.consume_control_frame(ControlFrame(voice_patches=[VoiceControlPatch(0, [ControlItem(C.DETUNE, ValueType.Q15U, 5)])]))


def test_eta_override_clear_restores_preset_ramp():
    emitter = VoiceFrameEmitter(preset())
    emitter.note_on(69)
    emitter.set_eta(1000)
    assert emitter.emit_frame().beta_target_q15 == 1000
    emitter.set_eta(None)
    frame = emitter.emit_frame()
    assert frame.beta_target_q15 == emitter.voice_state.beta_current_q15
    assert emitter.voice_state.beta_target_q15 == emitter.preset.material.beta_target_q15


def test_single_emitter_external_id_is_respected():
    emitter = VoiceFrameEmitter(preset(), voice_id=7)
    cf = note_frame(69, voice_id=7)
    cf.voice_patches = [patch(0, DETUNE=-1200), patch(7, PITCH_BEND=100)]
    emitter.consume_control_frame(cf)
    frame = emitter.emit_frame()
    assert frame.voice_id == 7 and frame.cents == 100


def test_cmmm_canonical_frames_preserve_pitch_and_accept_live_adsr():
    e = VoiceEngine(max_voices=1)
    vid = e.allocate_voice()
    program = deepcopy(e.voices[vid].program)
    cf = note_frame(60)
    cf.voice_patches = [patch(DETUNE=1200, ATK_BLOCKS=1, DCY_BLOCKS=1, SUS_Q15=10000)]
    first = e.step_block(cf).render_batch[0].frame
    second = e.step_block().render_batch[0].frame
    assert first.note == 60 and first.cents == 1200
    assert first.f0_hz == pytest.approx(523.251130601)
    assert first.is_starting and not second.is_starting
    assert second.gain_q15 == 10000
    assert first.ops_enable
    assert e.voices[vid].program == program


def test_cmmm_retrigger_marks_start_and_early_release_never_jumps_up():
    e = VoiceEngine(max_voices=1)
    e.allocate_voice()
    cf = note_frame()
    cf.voice_patches = [patch(ATK_BLOCKS=100, REL_BLOCKS=10)]
    first = e.step_block(cf).render_batch[0].frame
    off = ControlFrame()
    off.add_note_off(69)
    released = e.step_block(off).render_batch[0].frame
    assert 0 < released.gain_q15 < first.gain_q15
    retriggered = e.step_block(note_frame()).render_batch[0].frame
    assert retriggered.is_starting
    assert not e.step_block().render_batch[0].frame.is_starting


def test_cmmm_snapshot_preserves_controls_and_release_start_for_replay():
    import json
    from symcrash_python_ap.voice.voice_engine import VoiceStateWithProgram

    a = VoiceEngine(max_voices=1)
    a.allocate_voice()
    cf = note_frame(channel=3)
    cf.voice_patches = [patch(DETUNE=100, PITCH_BEND=50, ATK_BLOCKS=100, REL_BLOCKS=20)]
    a.step_block(cf)
    off = ControlFrame()
    off.add_note_off(69, channel=3)
    a.consume_control_frame(off)
    snapshot = json.loads(json.dumps(a.voices[0].snapshot()))
    b = VoiceEngine(max_voices=1)
    b.voices[0] = VoiceStateWithProgram.restore(snapshot)
    for _ in range(10):
        assert a.step_block().render_batch == b.step_block().render_batch
    # New sparse patches must build on the restored latches.
    b.consume_control_frame(ControlFrame(voice_patches=[patch(PITCH_BEND=0)]))
    assert b.voices[0].controls.pitch_offset_cents == 100


def test_explicit_default_overrides_preset_and_midi_velocity_zero_releases():
    cockpit = CockpitEngine()
    e = engine(1)
    e.voices[0].compiled_preset.latch.tree_blend_q15 = 12000
    cockpit.midi_note_on(69, 127)
    cockpit.set_control(C.MORPH, 0)
    assert e.step_block(cockpit.tick()).render_batch[0].frame.tree_blend_q15 == 0
    cockpit.midi_note_on(69, 0)
    e.consume_control_frame(cockpit.tick())
    assert e.voices[0].phase == VoicePhase.RELEASE


def test_synthmaster_pan_and_gain_reach_rendered_audio():
    from symcrash_python_ap.synth_master import SynthMaster, SynthConfig

    synth = SynthMaster(SynthConfig(n_voices=1))
    synth.voice_engine.allocate_voice_from_preset(preset(), 0)
    cf = note_frame()
    cf.globals.master_pan_q15 = -32768
    left, right, _ = synth._render_block_voice_engine(cf)
    audio = list(left)
    for _ in range(12):
        left, right, _ = synth._render_block_voice_engine()
        audio.extend(left)
        assert not any(right)
    assert max(abs(x) for x in audio) > 100
    mute = ControlFrame(globals=GlobalControls(master_gain_q15=0))
    synth._render_block_voice_engine(mute)
    for _ in range(80):
        left, right, _ = synth._render_block_voice_engine()
    assert max(abs(x) for x in left + right) < 10


@pytest.mark.parametrize('control', ['TIMBRE', 'TEXTURE', 'CHAOS'])
def test_family_control_changes_rendered_audio(control):
    from symcrash_python_ap.vm.executor import vm_step_block
    from symcrash_python_ap.vm.state import VMState

    def render(value):
        e = engine(1)
        vm = VMState.create(voice_id=0)
        vm.reset_for_note_on(note=69)
        cf = note_frame()
        cf.voice_patches = [patch(**{control: value})]
        e.consume_control_frame(cf)
        samples = []
        for _ in range(30):
            frame = e.step_voice_simple(0)
            vm, output = vm_step_block(vm, frame)
            samples.extend(output)
        return samples

    assert render(0) != render(32767)
