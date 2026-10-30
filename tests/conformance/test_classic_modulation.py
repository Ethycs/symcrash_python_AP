"""Audible, portable and independently calculated classic modulation laws."""
from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction
import json
import subprocess
import sys

import numpy as np
import pytest

from symcrash_python_ap import instrument_ir as ir
from symcrash_python_ap.frames import VoiceFrame, VoiceFramePacketV1
from symcrash_python_ap.library.classic_modulation import ClassicModulation
from symcrash_python_ap.library.native_carrier import NativeCarrierConfig
from symcrash_python_ap.library.preset_compiler import compiled_preset_from_ir, compiled_preset_from_capsule
from symcrash_python_ap.library.compiler import CapsuleCompiler
from symcrash_python_ap.voice.frame_builder import IRFrameBuilder, NoteClock
from symcrash_python_ap.voice.voice_engine import VoiceEngine, VoiceStateWithProgram
from symcrash_python_ap.vm import VMState, vm_step_block, harmonic_bank as bank
from symcrash_python_ap.vm.decode import decode_frame
from tests.conformance.test_native_preset_transport import source, routes, owner, capsule_for
from tests.conformance.test_vm_snapshot_replay import execution_path


def authored(family='all', legacy=False, **params):
    result = source('saw', **params)
    target = result.layers.L0_graph.nodes[0].params if legacy else result.layers.L1_lift
    values = dict(vibrato=ir.VibratoParams(7, .5, .3, 0, 0),
                  body_warble=ir.BodyWarbleParams(2, .5, 4),
                  attack_texture=ir.AttackTextureParams(60, .5, [3, 5]))
    for name, value in values.items():
        if family in (name, 'all'):
            setattr(target, name, value)
    return result


@pytest.mark.parametrize('family', ['vibrato', 'body_warble', 'attack_texture', 'all'])
@pytest.mark.parametrize('legacy', [False, True])
def test_families_survive_capsule_and_change_live_and_offline_audio(family, legacy, execution_path):
    original = authored(family, legacy)
    before = deepcopy(original)
    direct, capsule = routes(original)
    assert direct.native_carrier == capsule.native_carrier
    engines = [owner(p) for p in (direct, capsule, compiled_preset_from_ir(source()))]
    builders = [IRFrameBuilder(s, 69, 24000) for s in (original, source())]
    states = [VMState.create(0) for _ in range(5)]
    changed = [0, 0]
    for block in range(40):
        if block == 28:
            for engine in engines:
                engine.note_off_simple(0)
        frames = [e.step_voice_simple(0) for e in engines]
        assert frames[0] == frames[1]
        frames += [builder.build(NoteClock(block, gate=block < 28, sustain_until_block=28)) for builder in builders]
        assert frames[0].harmonic_gains_q15 == frames[3].harmonic_gains_q15
        assert frames[0].harmonic_gain_bound_q15 == frames[3].harmonic_gain_bound_q15
        audio = [vm_step_block(state, frame)[1].copy() for state, frame in zip(states, frames)]
        np.testing.assert_array_equal(audio[0], audio[1])
        changed[0] += np.count_nonzero(audio[0] != audio[2])
        changed[1] += np.count_nonzero(audio[3] != audio[4])
    assert min(changed) > 100 and original == before


@pytest.mark.parametrize('family,zero', [('vibrato', ir.VibratoParams()),
    ('body_warble', ir.BodyWarbleParams()), ('attack_texture', ir.AttackTextureParams())])
def test_explicit_l1_zero_overrides_legacy_family_and_is_bit_exact_bypass(family, zero, execution_path):
    original = authored(family, legacy=True)
    setattr(original.layers.L1_lift, family, zero)
    presets = [*routes(original), compiled_preset_from_ir(source())]
    assert all(p.native_carrier.classic_modulation is None for p in presets)
    engines, vms = [owner(p) for p in presets], [VMState.create(0) for _ in presets]
    for _ in range(12):
        frames = [engine.step_voice_simple(0) for engine in engines]
        assert frames[0] == frames[1] == frames[2]
        audio = [vm_step_block(vm, frame)[1].copy() for vm, frame in zip(vms, frames)]
        np.testing.assert_array_equal(audio[0], audio[1])
        np.testing.assert_array_equal(audio[0], audio[2])


def test_independent_quarter_cycle_delay_ramp_attack_overlap_and_saturation():
    # One quarter turn per block, delay one block, ramp two blocks.
    config = ClassicModulation(vibrato_inc_q32=2**30, amp_depth_q15=16384,
        timbre_depth_q15=8192, delay_samples=64, ramp_samples=128,
        warble_inc_q32=2**30, warble_depth_q15=16384, warble_harmonic=4,
        attack_samples=256, attack_amount_q15=16384, attack_harmonics=(4, 5))
    sine = [0, 32767, 0, -32767, 0, 32767]
    for block in range(6):
        frame = VoiceFrame(gain_q15=24000)
        config.apply(frame, block)
        env = max(0, min(32768, (block - 1) * 16384))
        v = round(Fraction(sine[block] * env, 32768))
        amp = round(Fraction(v * 16384, 32768))
        assert frame.gain_q15 == min(32767, round(Fraction(24000 * (32768 + amp), 32768)))
        timbre = round(Fraction(v * 8192, 32768))
        warble = round(Fraction(sine[block] * 16384, 32768))
        attack = max(0, (4 - block) * 4096)
        gains = [32768] * 32
        gains[1] -= timbre
        gains[3] += timbre + warble + attack
        gains[4] += attack
        assert frame.harmonic_gains_q15 == gains
        assert frame.harmonic_gain_bound_q15 == 73728  # constant even after attack
    negative = replace(config, timbre_depth_q15=32767, warble_depth_q15=32767)
    frame = VoiceFrame()
    negative.apply(frame, 7)
    assert frame.harmonic_gains_q15[3] == 0  # overlaps clamp after summation


def spectral_reference(state, basis, gains, bound):
    h_count = min(state.H_active, state.H_bandlimit, basis.H_max)
    total = sum(max(abs(int(state.weights.a_h_q15[h])), abs(int(state.weights.b_h_q15[h]))) for h in range(h_count))
    total = -(-(total * bound) // 32768)
    shift = 0
    while total // 2**shift > 32767:
        shift += 1
    rounded = lambda x: round(Fraction(int(x), 32768))
    result = []
    for leaf in range(64):
        for lane in range(bank.VM_K):
            phase = state.phase
            if state.fm_mod_buffer is not None and lane < len(state.fm_mod_buffer):
                phase += int(state.fm_mod_buffer[lane]) * state.fm_depth_q15 * 4
            value = rounded(state.weights.dc_q15 * int(basis.DC_frame[leaf, lane]))
            for h in range(h_count):
                sine, cosine = bank.sin_cos_q15((phase * (h + 1)) % 2**32)
                a = rounded((int(state.weights.a_h_q15[h]) // 2**shift) * gains[h])
                b = rounded((int(state.weights.b_h_q15[h]) // 2**shift) * gains[h])
                ws, wc = rounded(a * cosine - b * sine), rounded(a * sine + b * cosine)
                value += rounded(ws * int(basis.Usin_h[h, leaf, lane]))
                value += rounded(wc * int(basis.Ucos_h[h, leaf, lane]))
            result.append(max(-32768, min(32767, value)))
    return np.asarray(result, dtype=np.int16)


@pytest.mark.parametrize('count', [0, 1, 4, 16, 32])
def test_signed_spectral_weights_match_independent_fraction_reference(count, execution_path):
    rng = np.random.default_rng(310 + count)
    state = bank.ModeBState(H_active=count, H_bandlimit=32, phase=3788812303,
        fm_depth_q15=24577, fm_mod_buffer=np.array([-32768, 16384, 7, 32767], dtype=np.int16))
    state.weights.a_h_q15[:] = rng.integers(-32768, 32768, 32, dtype=np.int16)
    state.weights.b_h_q15[:] = rng.integers(-32768, 32768, 32, dtype=np.int16)
    state.weights.dc_q15 = -15500
    before = deepcopy(state.weights)
    gains = [int(v) for v in rng.integers(0, 120001, 32)]
    basis = bank.get_basis_for_note(69)
    expected = spectral_reference(state, basis, gains, 120000)
    for generate in (bank.mode_b_generate_coeffs, bank.mode_b_generate_coeffs_vectorized):
        actual = np.zeros_like(expected)
        generate(state, basis, actual, gains, 120000)
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(state.weights.a_h_q15, before.a_h_q15)
    np.testing.assert_array_equal(state.weights.b_h_q15, before.b_h_q15)
    assert state.weights.dc_q15 == before.dc_q15


@pytest.mark.parametrize('kind', ['saw', 'triangle', 'custom', 'pwm', 'oeis'])
def test_actual_vm_preserves_base_weights_and_phase_with_no_accumulated_modulation(kind, execution_path):
    original = authored('all')
    p = original.layers.L0_graph.nodes[0].params
    if kind == 'custom':
        p.waveform, p.harmonics, p.phase_spread = 'partials', ir.HarmonicProfile([.7, .2, .4, .1, .1, .2]), .713
    elif kind == 'pwm':
        p.pwm_enable, p.pwm_width_q15 = True, 8192
    elif kind == 'oeis':
        p.oeis_sequence_id = 1
    else:
        p.waveform = kind
    engine, active, plain = owner(compiled_preset_from_ir(original)), VMState.create(0), VMState.create(0)
    changed = 0
    for block in range(12):
        frame = engine.step_voice_simple(0)
        clean = replace(frame, harmonic_gains_q15=None, harmonic_gain_bound_q15=32768)
        a, b = vm_step_block(active, frame)[1].copy(), vm_step_block(plain, clean)[1].copy()
        np.testing.assert_array_equal(active.mode_b_state.weights.a_h_q15, plain.mode_b_state.weights.a_h_q15)
        np.testing.assert_array_equal(active.mode_b_state.weights.b_h_q15, plain.mode_b_state.weights.b_h_q15)
        assert active.mode_b_state.phase == plain.mode_b_state.phase
        changed += np.count_nonzero(a != b)
    assert changed > 0


def test_json_and_binary_checkpoint_replay_release_and_retrigger(execution_path):
    from symcrash_python_ap.voice.voice_state_codec import vmstate_to_bytes, vmstate_from_bytes
    first = owner(compiled_preset_from_ir(authored()))
    vm = VMState.create(0)
    for _ in range(9):
        vm_step_block(vm, first.step_voice_simple(0))
    second = VoiceEngine()
    second.voices[0] = VoiceStateWithProgram.restore(json.loads(json.dumps(first.voices[0].snapshot())))
    replay = VMState.restore(json.loads(json.dumps(vm.snapshot())))
    binary = vmstate_from_bytes(vmstate_to_bytes(vm))
    for block in range(25):
        if block == 7:
            first.note_off_simple(0)
            second.note_off_simple(0)
        if block == 16:
            first.note_on_simple(0, 72, 24000)
            second.note_on_simple(0, 72, 24000)
        a, b = first.step_voice_simple(0), second.step_voice_simple(0)
        assert a == b
        expected = vm_step_block(vm, a)[1].copy()
        np.testing.assert_array_equal(expected, vm_step_block(replay, b)[1])
        np.testing.assert_array_equal(expected, vm_step_block(binary, b)[1])
        if block == 16:
            assert a.harmonic_gains_q15[2] == 49152  # attack restarted


@pytest.mark.parametrize('mode', [0, 1, 3])
def test_amplitude_vibrato_works_for_every_native_mode(mode, execution_path):
    from tests.conformance.test_live_unison import authored as unison_authored
    original = unison_authored(count=1, mode=mode)
    original.layers.L1_lift.vibrato = ir.VibratoParams(187.5, .5, 0, 0, 0)
    preset, capsule = routes(original)
    original.layers.L1_lift.vibrato = None
    engines = [owner(p) for p in (preset, capsule, compiled_preset_from_ir(original))]
    vms = [VMState.create(0) for _ in engines]
    changed = 0
    for _ in range(16):
        frames = [e.step_voice_simple(0) for e in engines]
        assert frames[0] == frames[1] and frames[0].harmonic_gains_q15 is None
        audio = [vm_step_block(vm, f)[1].copy() for vm, f in zip(vms, frames)]
        np.testing.assert_array_equal(audio[0], audio[1])
        changed += np.count_nonzero(audio[0] != audio[2])
    assert changed


@pytest.mark.parametrize('family', ['vibrato', 'body_warble', 'attack_texture'])
def test_spectral_modulation_rejects_explicit_mode_a_before_playback(family):
    original = authored(family, osc_mode=0)
    with pytest.raises(ValueError, match='Mode B'):
        compiled_preset_from_ir(original)
    assert not CapsuleCompiler().compile_from_ir(original).success
    with pytest.raises(ValueError, match='Mode B'):
        IRFrameBuilder(original, 69)


@pytest.mark.parametrize('family,field,value', [
    ('vibrato', 'rate_hz', float('nan')), ('vibrato', 'rate_hz', 376),
    ('vibrato', 'depth_amp', 1.1), ('vibrato', 'depth_timbre', True),
    ('vibrato', 'delay_ms', -1), ('vibrato', 'ramp_ms', float('inf')),
    ('body_warble', 'target_harmonic', 0), ('body_warble', 'target_harmonic', 33),
    ('body_warble', 'depth', -1), ('attack_texture', 'harmonics', [3, 3]),
    ('attack_texture', 'harmonics', [True]), ('attack_texture', 'harmonics', [33]),
    ('attack_texture', 'bright_ms', -1), ('attack_texture', 'bright_amount', 1.1)])
def test_bad_authoring_fails_direct_and_capsule(family, field, value):
    original = authored()
    setattr(getattr(original.layers.L1_lift, family), field, value)
    with pytest.raises(ValueError):
        compiled_preset_from_ir(original)
    assert not CapsuleCompiler().compile_from_ir(original).success


@pytest.mark.parametrize('parser,data', [(ir._parse_vibrato_params, {'depth_cents': 5}),
    (ir._parse_body_warble_params, {'randomness': .1}),
    (ir._parse_attack_texture_params, {'noise_burst_ms': 10})])
def test_unsupported_documentation_example_fields_are_not_silently_ignored(parser, data):
    with pytest.raises(ValueError, match='Unknown'):
        parser(data)


@pytest.mark.parametrize('gains,bound', [([32768] * 31, 32768), ([32769] * 32, 32768),
    ([True] * 32, 32768), ([1.0] * 32, 32768), ([-1] * 32, 32768),
    ([0] * 32, 131073), (None, 40000), ([0] * 32, True)])
def test_invalid_raw_frames_fail_before_vm_mutation(gains, bound):
    vm = VMState.create(0)
    before = vm.snapshot()
    with pytest.raises(ValueError):
        vm_step_block(vm, VoiceFrame(coeff_osc_mode=1, harmonic_gains_q15=gains, harmonic_gain_bound_q15=bound))
    assert vm.snapshot() == before


def test_native_spectral_fields_require_mode_b_and_cannot_be_dropped_on_wire():
    for mode in (0, 2, 3, 4, 5):
        with pytest.raises(ValueError, match='Mode B'):
            decode_frame(VoiceFrame(coeff_osc_mode=mode, harmonic_gains_q15=[32768] * 32))
    frame = owner(compiled_preset_from_ir(authored())).step_voice_simple(0)
    with pytest.raises(ValueError, match='harmonic modulation'):
        VoiceFramePacketV1.from_voice_frame(frame)


@pytest.mark.parametrize('version', [1, 2])
def test_old_carrier_versions_migrate_detached_to_modulation_bypass(version):
    data = NativeCarrierConfig().to_dict()
    del data['classic_modulation']
    data['version'] = version
    if version == 1:
        for field in ('unison_voices', 'unison_spread_cents', 'unison_width_q15'):
            del data[field]
    before = deepcopy(data)
    restored = NativeCarrierConfig.from_dict(data)
    assert restored.version == 3 and restored.classic_modulation is None
    assert data == before


@pytest.mark.parametrize('field,value', [('amp_depth_q15', True), ('timbre_depth_q15', -1),
    ('warble_harmonic', 33), ('attack_harmonics', [2, 2]), ('delay_samples', -1),
    ('vibrato_inc_q32', 2**32), ('extra', 1)])
def test_malformed_capsule_modulation_is_rejected(field, value):
    capsule = capsule_for(authored())
    capsule.sound_extras['native_carrier']['classic_modulation'][field] = value
    with pytest.raises(ValueError):
        compiled_preset_from_capsule(capsule)


def test_spectral_configuration_mode_is_checked_at_capsule_and_snapshot_load():
    capsule = capsule_for(authored())
    capsule.sound_extras['osc_mode'] = 0
    with pytest.raises(ValueError, match='Mode B'):
        compiled_preset_from_capsule(capsule)
    saved = owner(compiled_preset_from_ir(authored())).voices[0].snapshot()
    saved['compiled_preset']['fields']['osc_mode'] = 3
    with pytest.raises(ValueError, match='Mode B'):
        VoiceStateWithProgram.restore(saved)


def test_unison_final_stereo_and_scheduled_private_clocks(execution_path):
    from symcrash_python_ap.synth_master import SynthMaster, SynthConfig
    from tests.conformance.test_scheduled_vm import schedule
    original = authored(unison_voices=3)
    hosts = [SynthMaster(SynthConfig(n_voices=3)) for _ in range(2)]
    for host, preset in zip(hosts, routes(original)):
        host.disable('cockpit')
        host.load_compiled_preset(preset)
        host.note_on(69, 100)
    nonzero = 0
    for block in range(20):
        before = [host.voice_engine.voices[0].snapshot() for host in hosts] if block == 8 else []
        before_vm = [host._voice_slots[0].vm_state.snapshot() for host in hosts] if block == 8 else []
        if block == 15:
            for host in hosts:
                host.note_off(69)
        policy = schedule(1, 2) if block == 8 else None
        samples = [host.render_block(schedule=policy)[:2] for host in hosts]
        assert samples[0] == samples[1]
        nonzero += sum(x != 0 for channel in samples[0] for x in channel)
        if block == 8:
            for host, saved, vm_saved in zip(hosts, before, before_vm):
                assert host.voice_engine.voices[0].snapshot() == saved
                assert host._voice_slots[0].vm_state.snapshot() == vm_saved
                assert host.voice_engine.voices[1].block_count == saved['block_count'] + 1
    assert nonzero > 500


def test_fresh_process_restore_reproduces_future_nonzero_audio(tmp_path, execution_path):
    from symcrash_python_ap.vm import hotpath_jit
    engine, vm = owner(compiled_preset_from_ir(authored())), VMState.create(0)
    for _ in range(8):
        vm_step_block(vm, engine.step_voice_simple(0))
    path = tmp_path / 'classic.json'
    path.write_text(json.dumps({'voice': engine.voices[0].snapshot(), 'vm': vm.snapshot()}))
    expected = [vm_step_block(vm, engine.step_voice_simple(0))[1].tolist() for _ in range(7)]
    code = '''import json, sys
from symcrash_python_ap.voice.voice_engine import VoiceEngine, VoiceStateWithProgram
from symcrash_python_ap.vm import VMState, vm_step_block, hotpath_jit
hotpath_jit.set_jit_enabled(sys.argv[2] == '1')
data = json.load(open(sys.argv[1]))
engine = VoiceEngine()
engine.voices[0] = VoiceStateWithProgram.restore(data['voice'])
vm = VMState.restore(data['vm'])
print(json.dumps([vm_step_block(vm, engine.step_voice_simple(0))[1].tolist() for _ in range(7)]))
'''
    result = subprocess.run([sys.executable, '-c', code, str(path), str(int(hotpath_jit.jit_enabled()))],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == expected and np.count_nonzero(expected)


@pytest.mark.parametrize('family', ['vibrato', 'body_warble', 'attack_texture'])
def test_empty_l1_mapping_explicitly_overrides_legacy_values_on_disk(family, tmp_path):
    original = authored(family, legacy=True)
    original.layers.L1_lift = ir._parse_l1_lift({family: {}})
    path = tmp_path / 'preset.yaml'
    ir.save_instrument(original, path)
    assert compiled_preset_from_ir(ir.load_instrument(path)).native_carrier.classic_modulation is None


@pytest.mark.parametrize('family', ['vibrato', 'body_warble', 'attack_texture'])
def test_time_varying_gains_change_audio_with_identical_carrier_path_and_headroom(family, execution_path):
    original = authored(family)
    if family == 'vibrato':
        original.layers.L1_lift.vibrato.depth_amp = 0  # isolate spectral motion
    engine = owner(compiled_preset_from_ir(original))
    vms = [VMState.create(0), VMState.create(0)]
    changed = 0
    for _ in range(36):
        frame = engine.step_voice_simple(0)
        neutral = replace(frame, harmonic_gains_q15=[32768] * 32)
        assert frame.harmonic_gain_bound_q15 == neutral.harmonic_gain_bound_q15
        audio = [vm_step_block(vm, f)[1].copy() for vm, f in zip(vms, (frame, neutral))]
        changed += np.count_nonzero(audio[0] != audio[1])
    assert changed > 100
