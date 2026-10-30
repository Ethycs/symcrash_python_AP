"""Authored phase, shared clocks, note sync and replay must have one owner."""
from copy import deepcopy
from dataclasses import asdict
import json
import subprocess
import sys

import numpy as np
import pytest

from symcrash_python_ap import instrument_ir as ir
from symcrash_python_ap.library.compiler import CapsuleCompiler
from symcrash_python_ap.library.preset_compiler import compiled_preset_from_ir, compiled_preset_from_capsule
from symcrash_python_ap.vm import VMState, vm_step_block, hotpath_jit as hot
from symcrash_python_ap.vm.harmonic_bank import LFOState, LFORouting, LFOSyncMode, LFOTarget, LFOWaveform
from symcrash_python_ap.voice.lfo_clocks import initialize_lfo_clocks, advance_lfo_clocks, gate_lfo_clocks
from symcrash_python_ap.voice.voice_engine import VoiceEngine, VoiceStateWithProgram, VoicePhase
from symcrash_python_ap.voice.frame_builder import IRFrameBuilder, NoteClock
from tests.conformance.test_mode_b_phase_modulation import authored_fm
from tests.conformance.test_native_preset_transport import routes, owner, capsule_for
from tests.conformance.test_ir_regime_compilation import schedule
from tests.conformance.test_vm_snapshot_replay import execution_path


def clock(phase=0x40000000, increment=17, slot=0, sync=LFOSyncMode.FREE, **overrides):
    state = LFOState(enabled=True, phase=phase, phase_inc=increment, arc_slot=slot,
        sync_mode=sync, depth_q15=32767, waveform=LFOWaveform.SINE,
        routings=[LFORouting(LFOTarget.PHASE, 32767)])
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def authored(lfos):
    source = authored_fm()
    source.modulators = lfos
    return source


@pytest.mark.parametrize('sync', list(LFOSyncMode))
def test_allocation_preserves_authored_phase_until_its_sync_policy_resets_it(sync):
    source = authored([clock(sync=sync)])
    before = deepcopy(source)
    preset = compiled_preset_from_ir(source)
    engine = VoiceEngine()
    engine.allocate_voice_from_preset(preset, voice_id=0)
    state = engine.voices[0]
    assert state.arc_phase_q16[0] == state.lfos[0].phase == 0x40000000
    engine.note_on_simple(0, note=69)
    expected = 0x40000000 if sync == LFOSyncMode.FREE else 0
    assert state.arc_phase_q16[0] == state.lfos[0].phase == expected
    engine.step_voice_simple(0)
    assert state.arc_phase_q16[0] == state.lfos[0].phase == expected + 64 * 17
    assert source == before and preset.lfos[0].phase == 0x40000000


@pytest.mark.parametrize('sync', list(LFOSyncMode))
@pytest.mark.parametrize('legato', [False, True])
def test_gate_policy_resets_clock_and_envelope_together(sync, legato):
    lfos = [clock(sync=sync, elapsed_samples=123, current_depth_scale=12345)]
    phases = [0] * 8
    initialize_lfo_clocks(lfos, phases)
    gate_lfo_clocks(lfos, phases, is_legato=legato)
    reset = sync == LFOSyncMode.GATE_RESET or (sync == LFOSyncMode.LEGATO_RESET and not legato)
    assert phases[0] == lfos[0].phase == (0 if reset else 0x40000000)
    assert lfos[0].elapsed_samples == (0 if reset else 123)
    assert lfos[0].current_depth_scale == (0 if reset else 12345)


@pytest.mark.parametrize('simple', [False, True])
@pytest.mark.parametrize('sync', list(LFOSyncMode))
def test_actual_note_overlap_is_legato_but_release_then_note_is_not(simple, sync):
    engine = VoiceEngine()
    if simple:
        engine.allocate_voice_from_preset(compiled_preset_from_ir(authored([clock(sync=sync)])), voice_id=0)
        trigger = lambda: engine.note_on_simple(0, note=69)
        release = lambda: engine.note_off_simple(0)
    else:
        engine.allocate_voice(voice_id=0)
        engine.voices[0].lfos = [clock(sync=sync)]
        trigger = lambda: engine.note_on(0)
        release = lambda: engine.note_off(0)
    trigger()
    state = engine.voices[0]
    for _ in range(4):
        state.tick_lfos(state.lfos)
    before = state.arc_phase_q16[0]
    trigger()
    assert state.arc_phase_q16[0] == (0 if sync == LFOSyncMode.GATE_RESET else before)
    state.tick_lfos(state.lfos)
    before = state.arc_phase_q16[0]
    release()
    trigger()
    assert state.arc_phase_q16[0] == (before if sync == LFOSyncMode.FREE else 0)


@pytest.mark.parametrize('count', [1, 2, 4, 9])
def test_compatible_shared_slot_advances_once_and_wraps_q32(count):
    lfos = [clock(phase=0xfffffff0, increment=3, delay_samples=i * 100) for i in range(count)]
    phases = [0] * 8
    initialize_lfo_clocks(lfos, phases)
    advance_lfo_clocks(lfos, phases)
    assert phases[0] == (0xfffffff0 + 3 * 64) % 2**32
    assert all(lfo.phase == phases[0] and lfo.elapsed_samples == 64 for lfo in lfos)
    assert phases[1:] == [0] * 7
    assert lfos[0].current_depth_scale > 0
    if count > 1:
        assert lfos[1].current_depth_scale == 0  # Envelopes remain per LFO.


def test_shared_disabled_member_keeps_envelope_frozen_and_observes_shared_phase():
    active, disabled = clock(), clock(enabled=False)
    phases = [0] * 8
    initialize_lfo_clocks([active, disabled], phases)
    advance_lfo_clocks([active, disabled], phases)
    assert active.phase == disabled.phase == phases[0]
    assert (active.elapsed_samples, disabled.elapsed_samples) == (64, 0)
    disabled.enabled = True
    advance_lfo_clocks([active, disabled], phases)
    assert (active.elapsed_samples, disabled.elapsed_samples) == (128, 64)


def test_all_disabled_clock_and_envelopes_freeze():
    lfos = [clock(enabled=False), clock(enabled=False)]
    before = deepcopy(lfos)
    phases = [0] * 8
    initialize_lfo_clocks(lfos, phases)
    original = phases.copy()
    advance_lfo_clocks(lfos, phases)
    assert lfos == before and phases == original


@pytest.mark.parametrize('field,value', [('phase', 7), ('phase_inc', 31), ('sync_mode', LFOSyncMode.GATE_RESET)])
def test_conflicting_shared_clock_is_rejected_by_both_compilers(field, value):
    a, b = clock(), clock()
    setattr(b, field, value)
    source = authored([a, b])
    before = deepcopy(source)
    with pytest.raises(ValueError, match='sharing'):
        compiled_preset_from_ir(source)
    result = CapsuleCompiler().compile_from_ir(source)
    assert not result.success and any('sharing' in error for error in result.errors)
    assert source == before


def test_capsule_rejects_conflicting_raw_clock_records():
    capsule = capsule_for(authored([clock(), clock()]))
    capsule.sound_extras['lfos'][1]['raw_state']['phase_inc'] += 1
    with pytest.raises(ValueError, match='sharing'):
        compiled_preset_from_capsule(capsule)


@pytest.mark.parametrize('field,value', [('arc_slot', -1), ('arc_slot', 8), ('arc_slot', True),
    ('phase', -1), ('phase', 2**32), ('phase_inc', 1.5), ('phase_inc', 2**32),
    ('enabled', 1), ('sync_mode', True), ('delay_samples', -1)])
def test_invalid_clock_plan_cannot_partly_initialize_or_advance(field, value):
    lfos = [clock(slot=0), clock(slot=1)]
    setattr(lfos[1], field, value)
    original = deepcopy(lfos)
    phases = [123] * 8
    for operation in (initialize_lfo_clocks, advance_lfo_clocks, gate_lfo_clocks):
        with pytest.raises(ValueError):
            operation(lfos, phases)
        assert lfos == original and phases == [123] * 8


@pytest.mark.parametrize('phases', [[0] * 7, [0] * 9, [False] * 8, [-1] * 8, [2**32] * 8])
def test_invalid_owner_vector_fails_before_clock_mutation(phases):
    lfos = [clock()]
    before = deepcopy(lfos)
    for operation in (initialize_lfo_clocks, advance_lfo_clocks, gate_lfo_clocks):
        with pytest.raises(ValueError):
            operation(lfos, phases)
        assert lfos == before


def load_lfos(definitions):
    return ir.load_instrument_from_dict({'instrument_ir_v0': {'meta': {'id': 'clocks'},
        'graph': {'osc': {'type': 'Osc', 'waveform': 'sine'}},
        'modulators': {'lfos': definitions}}})


def test_loader_reserves_explicit_slots_before_assigning_implicit_ones():
    loaded = load_lfos([{}, {'arc_slot': 0}, {}, {'arc_slot': 4}])
    assert [lfo.arc_slot for lfo in loaded.modulators] == [1, 0, 2, 4]


def test_loader_supports_eight_independent_clocks_and_explicit_compatible_sharing():
    assert [lfo.arc_slot for lfo in load_lfos([{}] * 8).modulators] == list(range(8))
    with pytest.raises(ValueError, match='No free'):
        load_lfos([{}] * 9)
    assert len(load_lfos([{'arc_slot': 3}] * 9).modulators) == 9


@pytest.mark.parametrize('slot', [-1, 8, True, [], 1.5])
def test_loader_rejects_invalid_explicit_slots(slot):
    with pytest.raises(ValueError, match='arc_slot'):
        load_lfos([{'arc_slot': slot}])


@pytest.mark.parametrize('sync', list(LFOSyncMode))
def test_offline_and_live_start_from_same_authored_or_gate_reset_phase(sync):
    source = authored([clock(sync=sync), clock(sync=sync)])
    engine = owner(compiled_preset_from_ir(source))
    builder = IRFrameBuilder(source, 69, 24000)
    for block in range(8):
        live = engine.step_voice_simple(0)
        offline = builder.build(NoteClock(block, sustain_until_block=100))
        assert live.fm_modulator_q15 == offline.fm_modulator_q15
        assert [x.phase for x in engine.voices[0].lfos] == [x.phase for x in builder._lfos]


@pytest.mark.parametrize('shared', [False, True])
def test_native_capsule_audio_and_clock_replay_are_identical(shared, execution_path):
    lfos = [clock(increment=10391233)]
    if shared:
        lfos.append(clock(increment=10391233, ramp_samples=90))
    source = authored(lfos)
    before = deepcopy(source)
    engines = [owner(p) for p in routes(source)]
    vms = [VMState.create(0) for _ in range(2)]
    nonzero = 0
    for block in range(28):
        if block == 10:
            engines[1].voices[0] = VoiceStateWithProgram.restore(json.loads(json.dumps(engines[0].voices[0].snapshot())))
            vms[1] = VMState.restore(json.loads(json.dumps(vms[0].snapshot())))
        if block == 18:
            for engine in engines:
                engine.note_off_simple(0)
        frames = [engine.step_voice_simple(0) for engine in engines]
        assert asdict(frames[0]) == asdict(frames[1])
        audio = [vm_step_block(vm, frame)[1].copy() for vm, frame in zip(vms, frames)]
        np.testing.assert_array_equal(*audio)
        nonzero += np.count_nonzero(audio[0])
    assert nonzero and source == before


def test_legacy_checkpoint_with_stale_internal_phase_resumes_from_owned_arc_clock():
    engine = owner(compiled_preset_from_ir(authored([clock()])))
    for _ in range(5):
        engine.step_voice_simple(0)
    saved = json.loads(json.dumps(engine.voices[0].snapshot()))
    saved['lfos'][0]['fields']['phase'] = 0  # Older exporter kept an unused stale mirror.
    restored = owner(compiled_preset_from_ir(authored([clock()])))
    restored.voices[0] = VoiceStateWithProgram.restore(saved)
    for _ in range(4):
        assert asdict(engine.step_voice_simple(0)) == asdict(restored.step_voice_simple(0))
    assert restored.voices[0].lfos[0].phase == restored.voices[0].arc_phase_q16[0]


@pytest.mark.parametrize('value', [-1, 2**32])
@pytest.mark.parametrize('slot', [0, 7])
@pytest.mark.parametrize('has_lfo', [False, True])
def test_restore_rejects_invalid_owned_phases_even_in_unused_slots(value, slot, has_lfo):
    engine = owner(compiled_preset_from_ir(authored([clock()] if has_lfo else [])))
    saved = json.loads(json.dumps(engine.voices[0].snapshot()))
    saved['arc_phase_q16'][slot] = value
    before = deepcopy(saved)
    with pytest.raises(ValueError, match='unsigned Q32'):
        VoiceStateWithProgram.restore(saved)
    assert saved == before


@pytest.mark.parametrize('location,field,value', [
    ('runtime', 'phase_inc', 18), ('runtime', 'sync_mode', LFOSyncMode.GATE_RESET),
    ('preset', 'phase_inc', 18), ('preset', 'sync_mode', LFOSyncMode.GATE_RESET),
    ('preset', 'phase', 0)])
def test_restore_rejects_conflicting_shared_clock_definitions(location, field, value):
    from symcrash_python_ap.voice.snapshot_codec import encode
    engine = owner(compiled_preset_from_ir(authored([clock(), clock()])))
    saved = json.loads(json.dumps(engine.voices[0].snapshot()))
    lfos = (saved['lfos'] if location == 'runtime'
            else saved['compiled_preset']['fields']['lfos'])
    lfos[1]['fields'][field] = encode(value)
    before = deepcopy(saved)
    with pytest.raises(ValueError, match='sharing'):
        VoiceStateWithProgram.restore(saved)
    assert saved == before


def test_scheduler_hold_and_other_voice_do_not_advance_shared_clock():
    engine = owner(compiled_preset_from_ir(authored([clock(), clock()])))
    engine.allocate_voice_from_preset(engine.voices[0].compiled_preset, voice_id=1)
    engine.note_on_simple(1)
    before = engine.voices[1].snapshot()
    engine.step_voice_simple(0)
    assert engine.voices[1].snapshot() == before
    saved = engine.voices[0].snapshot()
    engine.step_block(schedule=schedule())
    assert engine.voices[0].snapshot() == saved


def test_fresh_process_restores_shared_clock_audio(tmp_path, execution_path):
    engine = owner(compiled_preset_from_ir(authored([clock(), clock()])))
    vm = VMState.create(0)
    for _ in range(5):
        vm_step_block(vm, engine.step_voice_simple(0))
    path = tmp_path / 'clock.json'
    path.write_text(json.dumps({'voice': engine.voices[0].snapshot(), 'vm': vm.snapshot()}))
    expected = [vm_step_block(vm, engine.step_voice_simple(0))[1].tolist() for _ in range(8)]
    code = '''import json, sys
from symcrash_python_ap.voice.voice_engine import VoiceEngine, VoiceStateWithProgram
from symcrash_python_ap.vm import VMState, vm_step_block, hotpath_jit
hotpath_jit.set_jit_enabled(sys.argv[2] == '1')
data=json.load(open(sys.argv[1]))
engine=VoiceEngine()
engine.voices[0]=VoiceStateWithProgram.restore(data['voice'])
vm=VMState.restore(data['vm'])
print(json.dumps([vm_step_block(vm,engine.step_voice_simple(0))[1].tolist() for _ in range(8)]))
'''
    result = subprocess.run([sys.executable, '-c', code, str(path), str(int(hot.jit_enabled()))],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == expected


def test_shared_clock_stereo_matches_two_independent_identical_clocks(execution_path):
    from symcrash_python_ap.synth_master import SynthMaster, SynthConfig
    shared = authored([clock(increment=10391233), clock(increment=10391233)])
    independent = deepcopy(shared)
    independent.modulators[1].arc_slot = 1
    presets = [*routes(shared), compiled_preset_from_ir(independent)]
    synths = [SynthMaster(SynthConfig(n_voices=2)) for _ in presets]
    for synth, preset in zip(synths, presets):
        synth.disable('cockpit')
        synth.load_compiled_preset(preset)
        synth.note_on(69, 100)
    nonzero = 0
    for block in range(24):
        if block == 16:
            for synth in synths:
                synth.note_off(69)
        audio = [np.array(synth.render_block()[:2]) for synth in synths]
        np.testing.assert_array_equal(audio[0], audio[1])
        np.testing.assert_array_equal(audio[0], audio[2])
        nonzero += np.count_nonzero(audio[0])
    assert nonzero
