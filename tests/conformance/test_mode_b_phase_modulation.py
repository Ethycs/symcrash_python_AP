"""WSE-STD-210 phase units and real modulator delivery, beyond field presence."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import subprocess
import sys

import numpy as np
import pytest

from symcrash_python_ap import instrument_ir as ir
from symcrash_python_ap.frames import VoiceFrame, VoiceFramePacketV0b, VoiceFramePacketV1
from symcrash_python_ap.vm import VMState, vm_step_block
from symcrash_python_ap.vm import harmonic_bank as bank, hotpath_jit as hot
from symcrash_python_ap.vm.decode import decode_frame
from symcrash_python_ap.vm.phase_modulation import modulation_lanes_q15
from symcrash_python_ap.voice.phase_modulation import phase_lfo_lanes
from symcrash_python_ap.voice.voice_engine import VoiceStateWithProgram
from symcrash_python_ap.voice import vm_binary_codec
from symcrash_python_ap.voice.frame_builder import IRFrameBuilder, NoteClock
from tests.conformance.test_native_preset_transport import source, routes, owner
from tests.conformance.test_vm_snapshot_replay import execution_path


def generate(kind, state, basis):
    out = np.zeros(bank.NUM_LEAVES * bank.VM_K, dtype=np.int16)
    if kind == 'scalar':
        bank.mode_b_generate_coeffs(state, basis, out)
    elif kind == 'vector':
        bank.mode_b_generate_coeffs_vectorized(state, basis, out)
    else:
        bank.mode_b_generate_coeffs_hw(state.phase, state.phase_inc,
            state.weights.a_h_q15, state.weights.b_h_q15, state.weights.dc_q15,
            basis.Usin_h, basis.Ucos_h, basis.DC_frame, out,
            state.fm_mod_buffer, state.fm_depth_q15)
    return out


@pytest.mark.parametrize('kind', ['scalar', 'vector', 'hw'])
@pytest.mark.parametrize('depth,samples', [
    (0, [-32768, -1, 1, 32767]), (1, [-32768, -1, 1, 32767]),
    (16384, [16384, -16384, -32768, 32767]),
    (32767, [-32768, -1, 1, 32767]), (12345, [7, -13, 22017, -30117]),
])
def test_all_generators_use_normative_full_turn_signed_phase(kind, depth, samples, monkeypatch):
    # Observe the phase reaching the oscillator, independently of its audio math.
    monkeypatch.setattr(hot, '_JIT_ENABLED', False)
    state = bank.ModeBState(phase=0xfedcba98, H_active=3, H_bandlimit=3,
                           fm_depth_q15=depth, fm_mod_buffer=np.array(samples, dtype=np.int16))
    basis = bank.HarmonicBasisFrames(np.zeros((64, 4), dtype=np.int16),
        np.zeros((16, 64, 4), dtype=np.int16), np.zeros((16, 64, 4), dtype=np.int16), 16, 60, 261.6)
    observed = []
    monkeypatch.setattr(bank, 'sin_cos_q15', lambda phase: (observed.append(int(phase)) or (0, 32767)))
    generate(kind, state, basis)
    phases = [(state.phase + sample * depth * 2 ** (32 - 15 - 15)) % 2**32 for sample in samples]
    if kind == 'scalar':
        expected = [(phase * h) % 2**32 for phase in phases for h in range(1, 4)]
    else:
        expected = [(phase * h) % 2**32 for h in range(1, 17 if kind == 'hw' else 4) for phase in phases]
    assert observed == expected


@pytest.mark.parametrize('kind', ['scalar', 'vector', 'hw'])
@pytest.mark.parametrize('sample,depth,turn', [(16384, 16384, 0x40000000),
    (-16384, 16384, -0x40000000), (-32768, 16384, -0x80000000), (1, 1, 4)])
def test_actual_coefficients_equal_known_constant_phase_shift(kind, sample, depth, turn):
    state = bank.ModeBState(phase=0x12345678, H_active=1, fm_depth_q15=depth,
                           fm_mod_buffer=np.full(bank.VM_K, sample, dtype=np.int16))
    state.weights.set_sine()
    expected = deepcopy(state)
    expected.phase = (expected.phase + turn) % 2**32
    expected.fm_depth_q15 = 0
    expected.fm_mod_buffer = None
    basis = bank.get_basis_for_note(69)
    np.testing.assert_array_equal(generate(kind, state, basis), generate(kind, expected, basis))


@pytest.mark.parametrize('samples', [[], [4], [-32768, 32767, 5], list(range(64))])
def test_l1_buffer_selection_and_short_buffer_zero_padding(samples):
    expected = (samples + [0] * bank.VM_K)[:bank.VM_K]
    assert modulation_lanes_q15(samples, bank.VM_K).tolist() == expected
    state = bank.ModeBState(H_active=1, fm_depth_q15=16384, fm_mod_buffer=samples)
    state.weights.set_sine()
    basis = bank.get_basis_for_note(69)
    complete = deepcopy(state)
    complete.fm_mod_buffer = expected
    for kind in ('scalar', 'vector', 'hw'):
        np.testing.assert_array_equal(generate(kind, state, basis), generate(kind, complete, basis))


@pytest.mark.parametrize('value', [[32768], [-32769], [1.5], [True], [1, True], [[1]], 'bad'])
def test_malformed_modulator_rejected_at_frame_boundary(value):
    with pytest.raises(ValueError, match='modulator'):
        decode_frame(VoiceFrame(fm_source=2, fm_depth_q15=123, fm_modulator_q15=value))


@pytest.mark.parametrize('field,value', [('fm_source', -1), ('fm_source', 3),
    ('fm_source', True), ('fm_source', 0.0), ('fm_depth_q15', -1),
    ('fm_depth_q15', 32768), ('fm_depth_q15', False), ('fm_depth_q15', 0.0)])
def test_malformed_depth_and_source_rejected(field, value):
    with pytest.raises(ValueError, match='FM'):
        decode_frame(VoiceFrame(**{field: value}))


def lfo(**values):
    result = bank.LFOState(enabled=True, waveform=bank.LFOWaveform.SQUARE,
        depth_q15=32767, phase=0, phase_inc=0x40000000,
        routings=[bank.LFORouting(bank.LFOTarget.PHASE, 32767)])
    for name, value in values.items():
        setattr(result, name, value)
    return result


def qmul(a, b):
    # Independent exact binary arithmetic; float is exact for these small products.
    return round(a * b / 32768)


def test_lfo_samples_first_k_times_and_does_not_advance_owner():
    osc = lfo()
    original = deepcopy(osc)
    full = qmul(qmul(qmul(qmul(32767, 32767), 32767), 32767), 32767)
    assert phase_lfo_lanes([osc]) == [full, full, -full, -full]
    assert osc == original
    arc = [0x80000000] + [0] * 7
    assert phase_lfo_lanes([osc], arc) == [-full, -full, full, full]
    assert arc[0] == 0x80000000 and osc == original


@pytest.mark.parametrize('delay,ramp', [(0, 0), (2, 0), (0, 3), (2, 3), (64, 20)])
def test_delay_ramp_boundaries_use_current_sample_time(delay, ramp):
    osc = lfo(phase_inc=0, delay_samples=delay, ramp_samples=ramp)
    expected = []
    for k in range(bank.VM_K):
        env = 0 if k < delay else (min(32767, (k - delay) * 32767 // ramp) if ramp else 32767)
        value = qmul(qmul(qmul(qmul(32767, env), 32767), 32767), 32767)
        expected.append(value)
    assert phase_lfo_lanes([osc]) == expected


def test_explicit_routes_sum_before_clipping_and_disabled_route_blocks_fallback():
    positive = lfo(phase_inc=0)
    negative = lfo(phase=0x80000000, phase_inc=0)
    assert phase_lfo_lanes([positive, positive, negative]) == phase_lfo_lanes([negative, positive, positive])
    assert phase_lfo_lanes([positive, positive]) == [32767] * bank.VM_K
    disabled = lfo(enabled=False)
    other = lfo(routings=[bank.LFORouting(bank.LFOTarget.PITCH, 100)])
    assert phase_lfo_lanes([disabled, other]) == [0] * bank.VM_K
    assert any(phase_lfo_lanes([other]))
    assert phase_lfo_lanes([]) == [0] * bank.VM_K
    assert phase_lfo_lanes([positive], depth_scale_q15=0) == [0] * bank.VM_K


def authored_fm():
    authored = source('sine', fm_enable=True, fm_source=2, fm_depth_q15=7000)
    osc = lfo(phase_inc=10391233, waveform=bank.LFOWaveform.SINE,
              delay_samples=70, ramp_samples=150, sync_mode=bank.LFOSyncMode.GATE_RESET)
    authored.modulators = [osc]
    return authored


def test_native_capsule_lfo_lanes_and_real_audio_match(execution_path):
    authored = authored_fm()
    original = deepcopy(authored)
    a, b = (owner(p) for p in routes(authored))
    vms = [VMState.create(i) for i in range(3)]
    differs = nonzero = 0
    for block in range(30):
        if block == 20:
            a.note_off_simple(0)
            b.note_off_simple(0)
        frames = [engine.step_voice_simple(0) for engine in (a, b)]
        assert asdict(frames[0]) == asdict(frames[1])
        outputs = [vm_step_block(vm, frame)[1].copy() for vm, frame in zip(vms, frames)]
        np.testing.assert_array_equal(*outputs)
        reference = vm_step_block(vms[2], replace(frames[0], fm_source=0))[1]
        differs += np.count_nonzero(reference != outputs[0])
        nonzero += np.count_nonzero(outputs[0])
        np.testing.assert_array_equal(vms[0].mode_b_state.fm_mod_buffer, frames[0].fm_modulator_q15)
    assert nonzero and differs > 100
    assert authored == original
    assert a.voices[0].lfos[0].elapsed_samples == 30 * 64


def test_offline_and_live_lfo_lane_timing_match_without_double_advancement():
    authored = authored_fm()
    engine = owner(routes(authored)[0])
    builder = IRFrameBuilder(authored, note=69, velocity_q15=24000)
    for block in range(10):
        offline = builder.build(NoteClock(block, sustain_until_block=100))
        live = engine.step_voice_simple(0)
        assert offline.fm_modulator_q15 == live.fm_modulator_q15
        assert builder._lfos[0].elapsed_samples == engine.voices[0].lfos[0].elapsed_samples == (block + 1) * 64


def test_source_off_and_zero_depth_keep_unmodulated_audio(execution_path):
    baseline, off, zero, missing = [VMState.create(0) for _ in range(4)]
    for _ in range(12):
        f = VoiceFrame(note=69, coeff_osc_mode=1, waveform_type=3)
        output = vm_step_block(baseline, f)[1].copy()
        for vm, frame in [(off, replace(f, fm_source=0, fm_depth_q15=32767)),
                          (zero, replace(f, fm_source=2, fm_modulator_q15=[32767] * 4)),
                          (missing, replace(f, fm_source=2, fm_depth_q15=32767))]:
            np.testing.assert_array_equal(vm_step_block(vm, frame)[1], output)
            assert vm.mode_b_state.fm_depth_q15 == 0 and vm.mode_b_state.fm_mod_buffer is None


def test_feedback_uses_previous_block_and_retrigger_clears_it(execution_path):
    state = VMState.create(0)
    frame = VoiceFrame(note=69, coeff_osc_mode=1, fm_depth_q15=11000, fm_source=1)
    previous = state.prev_fm_mod_buffer.copy()
    assert not np.any(previous)
    for _ in range(6):
        vm_step_block(state, frame)
        np.testing.assert_array_equal(state.mode_b_state.fm_mod_buffer, previous)
        previous = state.prev_fm_mod_buffer.copy()
    assert np.any(previous)
    state.reset_for_note_on(69)
    vm_step_block(state, frame)
    assert not np.any(state.mode_b_state.fm_mod_buffer)
    vm_step_block(state, replace(frame, fm_source=0))
    assert state.mode_b_state.fm_mod_buffer is None


def test_feedback_reenabled_after_bypass_uses_latest_operator_block(monkeypatch, execution_path):
    original = bank.extract_modulator_signal
    captured = []
    def capture(coefficients, method=0):
        result = original(coefficients, method)
        captured.append(result.copy())
        return result
    monkeypatch.setattr(bank, 'extract_modulator_signal', capture)
    state = VMState.create(0)
    frame = VoiceFrame(note=69, coeff_osc_mode=1, fm_depth_q15=11000, fm_source=1)
    for src in [1, 0, 0, 2, 0]:
        vm_step_block(state, replace(frame, fm_source=src))
    assert len(captured) == 5
    latest = captured[-1]
    vm_step_block(state, frame)
    np.testing.assert_array_equal(state.mode_b_state.fm_mod_buffer[:bank.VM_K], latest)
    assert state.prev_fm_mod_buffer.shape == (8,)
    assert not np.any(state.prev_fm_mod_buffer[bank.VM_K:])


@pytest.mark.parametrize('binary', [False, True])
@pytest.mark.parametrize('fm_source', [1, 2])
def test_voice_and_vm_checkpoints_replay_lfo_pm_future_audio(binary, fm_source, execution_path):
    authored = authored_fm()
    authored.layers.L0_graph.nodes[0].params.fm_source = fm_source
    preset = routes(authored)[0]
    engine = owner(preset)
    vm = VMState.create(0)
    for _ in range(7):
        vm_step_block(vm, engine.step_voice_simple(0))
    saved = json.loads(json.dumps(engine.voices[0].snapshot()))
    restored = owner(preset)
    restored.voices[0] = VoiceStateWithProgram.restore(saved)
    other = (vm_binary_codec.decode(vm_binary_codec.encode(vm)) if binary
             else VMState.restore(json.loads(json.dumps(vm.snapshot()))))
    for block in range(18):
        if block == 8:
            engine.note_off_simple(0)
            restored.note_off_simple(0)
        f, g = engine.step_voice_simple(0), restored.step_voice_simple(0)
        assert asdict(f) == asdict(g)
        np.testing.assert_array_equal(vm_step_block(vm, f)[1], vm_step_block(other, g)[1])


@pytest.mark.parametrize('packet', [VoiceFramePacketV0b, VoiceFramePacketV1])
def test_frozen_packets_reject_unrepresentable_native_modulator(packet):
    with pytest.raises(ValueError, match='modulator'):
        packet.from_voice_frame(VoiceFrame(fm_modulator_q15=[0] * bank.VM_K))


def test_legacy_four_lane_feedback_json_migrates_to_binary_layout():
    saved = VMState.create(0).snapshot()
    saved['prev_fm_mod_buffer'] = [-32768, -3, 7, 32767]
    before = deepcopy(saved)
    restored = VMState.restore(saved)
    assert saved == before
    assert restored.prev_fm_mod_buffer.tolist() == saved['prev_fm_mod_buffer'] + [0] * 4
    copy = vm_binary_codec.decode(vm_binary_codec.encode(restored))
    np.testing.assert_array_equal(copy.prev_fm_mod_buffer, restored.prev_fm_mod_buffer)


@pytest.mark.parametrize('bad', [[1] * 3, [1] * 9, [True] * 4, [1.5] * 4, [32768] * 4])
def test_feedback_checkpoint_rejects_invalid_shapes_and_values(bad):
    saved = VMState.create(0).snapshot()
    saved['prev_fm_mod_buffer'] = bad
    with pytest.raises(ValueError, match='feedback'):
        VMState.restore(saved)


@pytest.mark.parametrize('depth,expected', [(0, 0), (.5, 16384), (1, 32767)])
def test_documented_normalized_fm_depth_loads(depth, expected):
    authored = ir.load_instrument_from_dict({'instrument_ir_v0': {
        'meta': {'id': 'fm-normalized'}, 'graph': {'osc': {'type': 'Osc',
        'waveform': 'sine', 'fm_enable': True, 'fm_source': 2, 'fm_depth': depth}}}})
    assert authored.layers.L0_graph.nodes[0].params.fm_depth_q15 == expected


@pytest.mark.parametrize('depth', [-.1, 1.1, float('nan'), True])
def test_invalid_normalized_fm_depth_fails(depth):
    with pytest.raises(ValueError, match='fm_depth'):
        ir.load_instrument_from_dict({'instrument_ir_v0': {
            'meta': {'id': 'fm-bad'}, 'graph': {'osc': {'type': 'Osc', 'fm_depth': depth}}}})


def test_phase_target_and_exact_depth_survive_yaml_save_load(tmp_path):
    authored = authored_fm()
    path = tmp_path / 'fm.yaml'
    ir.save_instrument(authored, path)
    loaded = ir.load_instrument(path)
    assert authored == loaded
    assert loaded.modulators[0].routings[0].target == bank.LFOTarget.PHASE


def test_mode_a_cannot_silently_ignore_authored_mode_b_fm():
    with pytest.raises(ValueError, match='Mode B'):
        routes(source('sine', osc_mode=0, fm_enable=True, fm_source=1, fm_depth_q15=123))


def test_per_voice_clocks_scheduler_hold_and_retrigger():
    from tests.conformance.test_ir_regime_compilation import schedule
    engine = owner(routes(authored_fm())[0])
    engine.allocate_voice_from_preset(engine.voices[0].compiled_preset, voice_id=1)
    engine.note_on_simple(1, note=69, velocity_q15=24000)
    initial = engine.voices[1].snapshot()
    first = engine.step_voice_simple(0).fm_modulator_q15
    for _ in range(6):
        engine.step_voice_simple(0)
    assert engine.voices[1].snapshot() == initial
    saved = engine.voices[0].snapshot()
    engine.step_block(schedule=schedule())
    assert engine.voices[0].snapshot() == saved
    engine.note_on_simple(0, note=69, velocity_q15=24000)
    assert engine.step_voice_simple(0).fm_modulator_q15 == first


def test_stereo_host_direct_and_capsule_match_with_audible_lfo_pm(execution_path):
    from symcrash_python_ap.synth_master import SynthMaster, SynthConfig
    a, b = routes(authored_fm())
    plain = deepcopy(a)
    plain.native_carrier = replace(plain.native_carrier, fm_source=0)
    synths = [SynthMaster(SynthConfig(n_voices=2)) for _ in range(3)]
    for synth, preset in zip(synths, [a, b, plain]):
        synth.disable('cockpit')
        synth.load_compiled_preset(preset)
        synth.note_on(69, 100)
    different = nonzero = 0
    for block in range(30):
        if block == 20:
            for synth in synths:
                synth.note_off(69)
        outputs = [np.array(synth.render_block()[:2]) for synth in synths]
        np.testing.assert_array_equal(outputs[0], outputs[1])
        different += np.count_nonzero(outputs[0] != outputs[2])
        nonzero += np.count_nonzero(outputs[0])
    assert nonzero and different > 100


def test_fresh_process_checkpoint_continues_lfo_modulation(tmp_path, execution_path):
    engine = owner(routes(authored_fm())[0])
    vm = VMState.create(0)
    for _ in range(9):
        vm_step_block(vm, engine.step_voice_simple(0))
    path = tmp_path / 'checkpoint.json'
    path.write_text(json.dumps({'voice': engine.voices[0].snapshot(), 'vm': vm.snapshot()}))
    expected = [vm_step_block(vm, engine.step_voice_simple(0))[1].tolist() for _ in range(12)]
    code = '''import json, sys
from symcrash_python_ap.voice.voice_engine import VoiceEngine, VoiceStateWithProgram
from symcrash_python_ap.vm import VMState, vm_step_block, hotpath_jit
hotpath_jit.set_jit_enabled(sys.argv[2] == '1')
data = json.load(open(sys.argv[1]))
engine = VoiceEngine()
engine.voices[0] = VoiceStateWithProgram.restore(data['voice'])
vm = VMState.restore(data['vm'])
print(json.dumps([vm_step_block(vm, engine.step_voice_simple(0))[1].tolist() for _ in range(12)]))
'''
    result = subprocess.run([sys.executable, '-c', code, str(path), str(int(hot.jit_enabled()))],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == expected


def test_routing_negative_amount_and_exact_raw_depth_precedence():
    a = lfo(phase_inc=0)
    b = lfo(phase_inc=0, routings=[bank.LFORouting(bank.LFOTarget.PHASE, -32767)])
    assert phase_lfo_lanes([a, b]) == [0] * bank.VM_K
    authored = ir.load_instrument_from_dict({'instrument_ir_v0': {'meta': {'id': 'pm'},
        'graph': {'osc': {'type': 'Osc', 'fm_depth': .8, 'fm_depth_q15': 0}},
        'modulators': {'lfos': [{'targets': [{'type': 'fm_phase', 'amount': .5}]}]}}})
    assert authored.layers.L0_graph.nodes[0].params.fm_depth_q15 == 0
    assert authored.modulators[0].routings[0].target == bank.LFOTarget.PHASE
