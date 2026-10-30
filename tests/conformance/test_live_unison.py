"""Authored unison must create independently clocked, replayable live notes."""
from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction
import json
import subprocess
import sys

import numpy as np
import pytest

from symcrash_python_ap.control_frame import ControlFrame
from symcrash_python_ap.instrument_ir import load_instrument
from symcrash_python_ap.library.native_carrier import NativeCarrierConfig
from symcrash_python_ap.library.preset_compiler import compiled_preset_from_ir
from symcrash_python_ap.synth_master import SynthMaster, SynthConfig
from symcrash_python_ap.voice.voice_engine import VoiceEngine, VoiceStateWithProgram, VoicePhase
from symcrash_python_ap.voice.unison import members, power_gain, group_slots
from symcrash_python_ap.vm import VMState, vm_step_block, hotpath_jit
from symcrash_python_ap.frames import VoiceFramePacketV1
from tests.conformance.test_native_preset_transport import source, routes
from tests.conformance.test_vm_snapshot_replay import execution_path
from tests.conformance.test_scheduled_vm import schedule


def authored(count=3, width=1.0, mode=1):
    if mode == 3:
        result = load_instrument('presets/instruments/dx_marimba_v0.yaml')
        p = next(n.params for n in result.layers.L0_graph.nodes if n.type == 'FM6')
        p.unison_voices, p.unison_width, p.unison_spread = count, width, 25
        return result
    return source('saw', unison_voices=count, unison_width=width, unison_spread=25, osc_mode=mode)


def owner(preset=None, capacity=8):
    engine = VoiceEngine(max_voices=capacity)
    engine.select_compiled_preset(preset or compiled_preset_from_ir(authored()))
    return engine


def note(engine, note=69, channel=0, voice=0xffff, off=False):
    frame = ControlFrame()
    if off:
        frame.add_note_off(note, channel)
    else:
        frame.add_note_on(note, 24000, channel)
    frame.events[-1].voice_id = voice
    engine.consume_control_frame(frame)


def active(engine):
    return [v for v in engine.voices.values() if v.phase not in (VoicePhase.FREE, VoicePhase.RELEASE)]


@pytest.mark.parametrize('count', range(1, 9))
def test_member_positions_gain_and_authored_transport(count):
    direct, capsule = routes(authored(count))
    assert direct.native_carrier == capsule.native_carrier
    config = direct.native_carrier
    plans = list(members(1, list(range(count)), config))
    assert power_gain(count) == round(32767 / np.sqrt(count))
    for index, plan in enumerate(plans):
        position = Fraction(0) if count == 1 else Fraction(2 * index - count + 1, count - 1)
        assert plan.detune_cents == round(position * 25)
        assert plan.pan_q15 == round(position * 32767)
        assert plan.phase_q32 == index * 0x9e3779b9 % 2**32


@pytest.mark.parametrize('capacity', [1, 2, 3, 8])
def test_pool_limits_normalize_actual_count_and_do_not_steal_held_notes(capacity):
    engine = owner(capacity=capacity)
    note(engine)
    voices = active(engine)
    assert len(voices) == min(capacity, 3)
    assert all(v.unison.count == len(voices) for v in voices)
    assert all(v.unison.gain_q15 == power_gain(len(voices)) for v in voices)
    before = {v.voice_id: v.snapshot() for v in voices}
    note(engine, 72)
    assert len(active(engine)) == min(capacity, 6)
    assert all(engine.voices[vid].snapshot() == value for vid, value in before.items())


def test_retrigger_release_channel_and_explicit_group_addressing():
    engine = owner(capacity=8)
    note(engine, channel=0)
    first = [v.voice_id for v in active(engine)]
    note(engine, channel=1)
    assert len(active(engine)) == 6
    note(engine, channel=0)
    assert [v.voice_id for v in active(engine) if v.channel == 0] == first
    assert len(active(engine)) == 6
    note(engine, channel=0, voice=first[1], off=True)
    assert all(engine.voices[v].phase == VoicePhase.RELEASE for v in first)
    assert len(active(engine)) == 3
    note(engine, channel=1, off=True)
    assert not active(engine)
    note(engine, 74, voice=0)
    assert len(active(engine)) == 3 and engine.voices[0].note_u8 == 74


def test_preset_change_releases_surplus_and_reuses_group_without_orphans():
    engine = owner()
    note(engine)
    engine.select_compiled_preset(compiled_preset_from_ir(authored(1)))
    note(engine)
    assert len(active(engine)) == 1
    assert active(engine)[0].unison is None
    assert len([v for v in engine.voices.values() if v.phase == VoicePhase.RELEASE]) == 2
    engine.select_compiled_preset(compiled_preset_from_ir(authored(5)))
    note(engine)
    assert len(active(engine)) == 5
    note(engine, off=True)
    assert not active(engine)


@pytest.mark.parametrize('mode', [0, 1, 3])
@pytest.mark.parametrize('width', [0.0, 1.0])
def test_actual_stereo_native_capsule_audio_and_width(mode, width, execution_path):
    hosts = [SynthMaster(SynthConfig(n_voices=8)) for _ in range(2)]
    for host, preset in zip(hosts, routes(authored(width=width, mode=mode))):
        host.disable('cockpit')
        host.load_compiled_preset(preset)
        host.note_on(69, 100)
        assert host.get_active_voice_count() == 3
    traces = []
    for block in range(30):
        if block == 20:
            for host in hosts:
                host.note_off(voice_id=0)
        audio = [np.array(host.render_block()[:2]) for host in hosts]
        np.testing.assert_array_equal(*audio)
        traces.append(audio[0])
    audio = np.concatenate(traces, axis=1)
    assert np.count_nonzero(audio)
    assert bool(np.any(audio[0] != audio[1])) == bool(width)
    assert not active(hosts[0].voice_engine)


@pytest.mark.parametrize('mode', [0, 1, 3])
def test_initial_phase_applies_once_and_clocks_advance_independently(mode, execution_path):
    engine = owner(compiled_preset_from_ir(authored(mode=mode)))
    note(engine)
    for voice in active(engine):
        vm = VMState.create(voice.voice_id)
        frames = [engine.step_voice_simple(voice.voice_id) for _ in range(2)]
        origin = voice.unison.phase_q32
        assert frames[0].osc_start_phase_q32 == origin and frames[1].osc_start_phase_q32 is None
        for block, frame in enumerate(frames):
            vm_step_block(vm, frame)
            from symcrash_python_ap.fixed import midi_to_phase_inc_fixed
            increment = midi_to_phase_inc_fixed(frame.note, frame.cents)
            if mode == 1:
                assert vm.mode_b_state.phase == (origin + (block + 1) * 64 * increment) % 2**32
            elif mode == 0:
                assert vm.ccs_state.phase == (origin + (block + 1) * 64 * increment) % 2**32
            else:
                for op in vm.fm6_state.operators:
                    assert op.phase == ((origin * op.ratio_q16 >> 16) + (block + 1) * 64 * (increment * op.ratio_q16 >> 16)) % 2**32


def test_native_start_phase_cannot_be_silently_exported_as_frozen_wire():
    engine = owner()
    note(engine)
    frame = engine.step_voice_simple(1)
    with pytest.raises(ValueError, match='start phase'):
        VoiceFramePacketV1.from_voice_frame(frame)


def test_old_native_record_and_voice_snapshot_migrate_to_single_voice():
    engine = owner(compiled_preset_from_ir(authored(1)))
    note(engine)
    saved = json.loads(json.dumps(engine.voices[0].snapshot()))
    assert saved['snapshot_version'] == 6
    saved['snapshot_version'] = 5
    del saved['unison']
    carrier = saved['compiled_preset']['fields']['native_carrier']['fields']
    carrier['version'] = 1
    del carrier['classic_modulation']
    for key in ('unison_voices', 'unison_spread_cents', 'unison_width_q15'):
        del carrier[key]
    before = deepcopy(saved)
    restored = VoiceStateWithProgram.restore(saved)
    assert restored.unison is None and restored.compiled_preset.native_carrier.unison_voices == 1
    assert saved == before and restored.snapshot()['snapshot_version'] == 6


@pytest.mark.parametrize('version', [True, 1.0])
def test_legacy_carrier_version_must_be_integer(version):
    data = NativeCarrierConfig().to_dict()
    data['version'] = version
    for key in ('unison_voices', 'unison_spread_cents', 'unison_width_q15'):
        del data[key]
    with pytest.raises(ValueError):
        NativeCarrierConfig.from_dict(data)


def test_group_snapshot_future_frames_audio_and_retrigger(execution_path):
    engine = owner()
    note(engine)
    vms = {v.voice_id: VMState.create(v.voice_id) for v in active(engine)}
    for _ in range(5):
        for vid, vm in vms.items():
            vm_step_block(vm, engine.step_voice_simple(vid))
    replay = owner(engine.selected_preset)
    replay.voices = {vid: VoiceStateWithProgram.restore(json.loads(json.dumps(v.snapshot())))
                     for vid, v in engine.voices.items()}
    copied = {vid: VMState.restore(json.loads(json.dumps(vm.snapshot()))) for vid, vm in vms.items()}
    for block in range(15):
        if block in (3, 8):
            for current in (engine, replay):
                note(current, off=block == 8)
        for vid in vms:
            a, b = engine.step_voice_simple(vid), replay.step_voice_simple(vid)
            assert asdict(a) == asdict(b)
            np.testing.assert_array_equal(vm_step_block(vms[vid], a)[1], vm_step_block(copied[vid], b)[1])


def test_explicit_leader_survives_dict_order_and_preallocated_unselected_preset():
    engine = VoiceEngine(max_voices=8)
    preset = compiled_preset_from_ir(authored())
    for vid in (0, 1, 5):
        engine.allocate_voice_from_preset(preset, vid)
    note(engine, voice=5)
    assert engine._find_voice_by_note(69) == 5
    assert group_slots(engine, 5) == [0, 1, 5]
    assert all(v.compiled_preset == preset and v.unison.leader_id == 5 for v in active(engine))
    note(engine)
    assert engine._find_voice_by_note(69) == 5 and len(active(engine)) == 3


@pytest.mark.parametrize('field,value', [('unison_voices', 0), ('unison_voices', 9),
    ('unison_voices', True), ('unison_voices', 2.5), ('unison_spread', -1),
    ('unison_spread', 101), ('unison_width', float('nan')), ('unison_width', -0.1),
    ('unison_width', 1.1)])
def test_invalid_authored_unison_fails_direct_and_capsule(field, value):
    from symcrash_python_ap.library.compiler import CapsuleCompiler
    ir = authored()
    setattr(ir.layers.L0_graph.nodes[0].params, field, value)
    with pytest.raises(ValueError):
        compiled_preset_from_ir(ir)
    assert not CapsuleCompiler().compile_from_ir(ir).success


@pytest.mark.parametrize('mode', [0, 1, 3])
def test_one_available_copy_is_audio_identical_to_single_voice(mode, execution_path):
    hosts = [SynthMaster(SynthConfig(n_voices=1)) for _ in range(2)]
    for host, count in zip(hosts, (3, 1)):
        host.disable('cockpit')
        host.load_compiled_preset(compiled_preset_from_ir(authored(count, mode=mode)))
        host.note_on(69, 100)
    for block in range(20):
        if block == 13:
            for host in hosts:
                host.note_off(69)
        np.testing.assert_array_equal(np.array(hosts[0].render_block()[:2]), np.array(hosts[1].render_block()[:2]))


def test_release_completion_tail_steal_and_group_id_isolation():
    ir = authored()
    ir.layers.L0_graph.nodes[1].params.adsr.r_ms = 3
    engine = owner(compiled_preset_from_ir(ir), capacity=4)
    note(engine)
    note(engine, 70)
    held = active(engine)[-1].voice_id
    held_state = engine.voices[held].snapshot()
    note(engine, off=True)
    note(engine, 74)
    assert len(active(engine)) == 4
    assert engine.voices[held].snapshot() == held_state
    note(engine, off=True)  # Old note must not release stolen slots.
    assert len(active(engine)) == 4
    note(engine, 74, off=True)
    for _ in range(10):
        engine.step_block()
    assert len(active(engine)) == 1 and active(engine)[0].voice_id == held
    assert all(v.phase == VoicePhase.FREE for vid, v in engine.voices.items() if vid != held)


def test_scheduled_copy_hold_preserves_private_voice_and_vm_clocks(execution_path):
    from symcrash_python_ap.vm.harmonic_bank import LFOState
    ir = authored()
    ir.modulators = [LFOState(enabled=True, phase_inc=17)]
    host = SynthMaster(SynthConfig(n_voices=8))
    host.disable('cockpit')
    host.load_compiled_preset(compiled_preset_from_ir(ir))
    host.note_on(69)
    host.render_block()
    voice_before = host.voice_engine.voices[0].snapshot()
    vm_before = host._voice_slots[0].vm_state.snapshot()
    host.render_block(schedule=schedule(1))
    assert host.voice_engine.voices[0].snapshot() == voice_before
    assert host._voice_slots[0].vm_state.snapshot() == vm_before
    assert host.voice_engine.voices[1].lfos[0].elapsed_samples == 128
    assert host.voice_engine.voices[0].lfos[0].elapsed_samples == 64


@pytest.mark.parametrize('mode', [0, 1, 3])
def test_member_vm_binary_checkpoint_replays_actual_future_audio(mode, execution_path):
    from symcrash_python_ap.voice.voice_state_codec import vmstate_to_bytes, vmstate_from_bytes
    engine = owner(compiled_preset_from_ir(authored(mode=mode)))
    note(engine)
    vm = VMState.create(1)
    for _ in range(4):
        vm_step_block(vm, engine.step_voice_simple(1))
    restored = vmstate_from_bytes(vmstate_to_bytes(vm))
    for _ in range(8):
        frame = engine.step_voice_simple(1)
        np.testing.assert_array_equal(vm_step_block(vm, frame)[1], vm_step_block(restored, frame)[1])


def test_fresh_process_group_replay(tmp_path, execution_path):
    engine = owner()
    note(engine)
    vms = {v.voice_id: VMState.create(v.voice_id) for v in active(engine)}
    for _ in range(3):
        for vid, vm in vms.items():
            vm_step_block(vm, engine.step_voice_simple(vid))
    path = tmp_path / 'unison.json'
    path.write_text(json.dumps({'voices': {vid: v.snapshot() for vid, v in engine.voices.items()},
                                'vms': {vid: vm.snapshot() for vid, vm in vms.items()}}))
    expected = [[vm_step_block(vm, engine.step_voice_simple(vid))[1].tolist()
                 for vid, vm in vms.items()] for _ in range(5)]
    code = '''import json, sys
from symcrash_python_ap.vm import VMState, vm_step_block, hotpath_jit
from symcrash_python_ap.voice.voice_engine import VoiceEngine, VoiceStateWithProgram
hotpath_jit.set_jit_enabled(sys.argv[2] == '1')
data=json.load(open(sys.argv[1]))
engine=VoiceEngine()
engine.voices={int(k):VoiceStateWithProgram.restore(v) for k,v in data['voices'].items()}
vms={int(k):VMState.restore(v) for k,v in data['vms'].items()}
print(json.dumps([[vm_step_block(vm,engine.step_voice_simple(vid))[1].tolist() for vid,vm in vms.items()] for _ in range(5)]))
'''
    result = subprocess.run([sys.executable, '-c', code, str(path), str(int(hotpath_jit.jit_enabled()))],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == expected


@pytest.mark.parametrize('field,value', [('group_id', 0), ('leader_id', -1), ('index', 3),
    ('count', 9), ('gain_q15', 32767), ('phase_q32', 2**32), ('pan_q15', 32768),
    ('detune_cents', 101)])
def test_invalid_member_snapshot_fails_before_adoption(field, value):
    engine = owner()
    note(engine)
    saved = json.loads(json.dumps(engine.voices[1].snapshot()))
    saved['unison']['fields'][field] = value
    before = deepcopy(saved)
    with pytest.raises(ValueError):
        VoiceStateWithProgram.restore(saved)
    assert saved == before


def test_current_snapshot_requires_member_field_even_for_single_voice():
    engine = owner(compiled_preset_from_ir(authored(1)))
    note(engine)
    saved = engine.voices[0].snapshot()
    del saved['unison']
    with pytest.raises(ValueError, match='Incomplete'):
        VoiceStateWithProgram.restore(saved)


@pytest.mark.parametrize('phase', [-1, 2**32, True, 1.5])
def test_invalid_native_phase_fails_decode(phase):
    from symcrash_python_ap.frames import VoiceFrame
    from symcrash_python_ap.vm.decode import decode_frame
    with pytest.raises(ValueError, match='unsigned Q32'):
        decode_frame(VoiceFrame(osc_start_phase_q32=phase))


@pytest.mark.parametrize('mode', [2, 4, 5])
def test_native_phase_rejects_unsupported_resource_oscillators(mode):
    from symcrash_python_ap.frames import VoiceFrame
    from symcrash_python_ap.vm.decode import decode_frame
    with pytest.raises(ValueError, match='supports Mode A'):
        decode_frame(VoiceFrame(coeff_osc_mode=mode, osc_start_phase_q32=1))


def test_physical_slot_controls_preserve_member_detune_and_preset():
    from tests.conformance.test_voice_controls import patch
    engine = owner()
    before = deepcopy(engine.selected_preset)
    engine.consume_control_frame(ControlFrame(voice_patches=[patch(0, DETUNE=100)]))
    note(engine)
    frames = [engine.step_voice_simple(vid) for vid in range(3)]
    assert [frame.cents for frame in frames] == [75, 0, 25]
    assert engine.selected_preset == before
    engine.consume_control_frame(ControlFrame(voice_patches=[patch(0, DETUNE=0)]))
    assert engine.step_voice_simple(0).cents == -25


def test_direct_and_control_frame_note_paths_have_identical_unison_audio(execution_path):
    hosts = [SynthMaster(SynthConfig(n_voices=8)) for _ in range(2)]
    for host in hosts:
        host.disable('cockpit')
        host.load_compiled_preset(compiled_preset_from_ir(authored()))
    hosts[0].note_on(69, 100)
    cf = ControlFrame()
    cf.add_note_on(69, 100 * 32767 // 127)
    hosts[1].enable('cockpit')
    hosts[1].cockpit.tick = lambda **kwargs: cf
    for block in range(22):
        if block == 15:
            hosts[0].note_off(voice_id=0)
            cf.add_note_off(69)
            cf.events[-1].voice_id = 0
        np.testing.assert_array_equal(np.array(hosts[0].render_block()[:2]),
                                      np.array(hosts[1].render_block()[:2]))


def test_fm6_morph_unison_uses_ratio_scaled_initial_phases(execution_path):
    from symcrash_python_ap.instrument_ir import FM6MorphConfig
    from symcrash_python_ap.fixed import midi_to_phase_inc_fixed
    ir = authored(mode=3)
    ir.layers.L1_lift.fm6_morph = FM6MorphConfig(topo_a=0, topo_b=3, blend_q15=12000)
    engines = [owner(p) for p in routes(ir)]
    for engine in engines:
        note(engine)
    states = [VMState.create(1), VMState.create(1)]
    for block in range(6):
        frames = [engine.step_voice_simple(1) for engine in engines]
        assert frames[0] == frames[1]
        audio = [vm_step_block(state, frame)[1].copy() for state, frame in zip(states, frames)]
        np.testing.assert_array_equal(*audio)
        origin = engines[0].voices[1].unison.phase_q32
        increment = midi_to_phase_inc_fixed(frames[0].note, frames[0].cents)
        for op in states[0].fm6_morph_state.operators:
            assert op.phase == ((origin * op.ratio_q16 >> 16) + (block + 1) * 64 * (increment * op.ratio_q16 >> 16)) % 2**32


def test_polyphonic_emitter_direct_notes_expand_and_retrigger_same_group():
    from symcrash_python_ap.voiceframe_emitter import MultiVoiceEmitter
    emitter = MultiVoiceEmitter(compiled_preset_from_ir(authored()), max_voices=8)
    assert emitter.note_on(69, 24000, cents=12) == 0
    assert emitter.active_voice_count == 3
    assert [frame.cents for frame in emitter.emit_frames()] == [-13, 12, 37]
    assert emitter.note_on(69, 24000, cents=12) == 0 and emitter.active_voice_count == 3
    emitter.note_off(69)
    assert not active(emitter._engine)


def test_single_emitter_direct_and_control_paths_share_one_copy_fallback():
    from symcrash_python_ap.voiceframe_emitter import VoiceFrameEmitter
    emitters = [VoiceFrameEmitter(compiled_preset_from_ir(authored()), voice_id=7) for _ in range(2)]
    emitters[0].note_on(69, 24000)
    cf = ControlFrame()
    cf.add_note_on(69, 24000)
    emitters[1].consume_control_frame(cf)
    for _ in range(8):
        assert emitters[0].emit_frame() == emitters[1].emit_frame()
    assert emitters[0].voice_state.unison.count == 1
