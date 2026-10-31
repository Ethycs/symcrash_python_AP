"""Complete Mode B commands, phase continuity and independent spectrum laws."""
from copy import deepcopy
from dataclasses import replace
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm import VMState, vm_step_block, harmonic_bank as bank, hotpath_jit
from symcrash_python_ap.vm.decode import decode_frame
from symcrash_python_ap.vm.carrier_spectrum import _weights
from symcrash_python_ap.voice.voice_state_codec import vmstate_to_bytes, vmstate_from_bytes
from symcrash_python_ap.voice.frame_builder import IRFrameBuilder, NoteClock
from symcrash_python_ap.instrument_ir import HarmonicProfile
from tests.conformance.test_native_preset_transport import source, routes, owner
from tests.conformance.test_mode_b_arithmetic import reference
from tests.conformance.test_vm_snapshot_replay import execution_path


def frame(**kw):
    return VoiceFrame(**dict(dict(note=69, coeff_osc_mode=1, waveform_type=3,
                                  gain_q15=24000, beta_target_q15=0), **kw))


@pytest.fixture(autouse=True, scope='module')
def prepared_mode_b_resources():
    from symcrash_python_ap.vm.basis_resource import prepare_basis_resources
    prepare_basis_resources()


def expected_weights(f):
    """Scalar authoring oracle; no production selector/cache/weight methods."""
    params = decode_frame(f).coeff_osc_params
    cap = min(32, 9 * 2**31 // (10 * params.phase_inc))
    count = min(f.H_request, cap)
    a, b = [0] * 32, [0] * 32
    if f.oeis_sequence_id >= 0:
        # Independent literal memberships for the selectors used in this suite.
        sets = {0: {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31},
                1: {1, 2, 3, 5, 8, 13, 21}, 6: set(range(1, 33, 2))}
        active = sorted(h for h in sets[f.oeis_sequence_id] if h <= count)
        for h in active:
            a[h-1] = int(32767 * 10**(f.oeis_rolloff_db * math.log2(h) / 20))
    elif f.partial_amps_q15:
        count = min(count, len(f.partial_amps_q15))
        step = f.partial_phase_spread_q15 * 0x9e3779b9 // 32768
        for i in range(count):
            value = f.partial_amps_q15[i]
            if f.partial_phase_spread_q15:
                from fractions import Fraction
                sine, cosine = bank.sin_cos_q15(((i + 1) * step) % 2**32)
                a[i], b[i] = round(Fraction(value * cosine, 32768)), round(Fraction(value * sine, 32768))
            else:
                a[i] = value
    elif f.pwm_enabled or f.waveform_type == 2:
        for h in range(1, count + 1):
            a[h-1] = int(2 * 32767 / (math.pi * h) * math.sin(math.pi * h * f.pwm_width_q15 / 32768))
    elif f.waveform_type == 0:
        count = min(count, 1)
        if count:
            a[0] = 32767
    else:
        for h in range(1, count + 1):
            if f.waveform_type in (3, 4):
                sign = (1 if h % 2 else -1) * (1 if f.waveform_type == 3 else -1)
                a[h-1] = int(sign * (2 / math.pi) * (1 / h) * 32767)
            elif h % 2 and f.waveform_type == 1:
                a[h-1] = int((1 / h) * 32767)
            elif h % 2 and f.waveform_type == 5:
                a[h-1] = int((-1)**((h - 1) // 2) * (8 / math.pi**2) / h**2 * 32767)
    return cap, count, a, b


def check_weights(vm, f):
    cap, count, a, b = expected_weights(f)
    actual = vm.mode_b_state
    assert (actual.H_bandlimit, actual.H_active, actual.weights.H_active) == (cap, count, count)
    np.testing.assert_array_equal(actual.weights.a_h_q15, a)
    np.testing.assert_array_equal(actual.weights.b_h_q15, b)
    assert actual.weights.dc_q15 == 0


SEQUENCE = [frame(waveform_type=0), frame(oeis_sequence_id=0), frame(oeis_sequence_id=1),
    frame(oeis_sequence_id=1, oeis_rolloff_db=-6),
    frame(partial_amps_q15=[20000, -9000, 7000, 1500], partial_phase_spread_q15=14000),
    frame(waveform_type=0), frame(pwm_enabled=True, pwm_width_q15=8192),
    frame(waveform_type=0), frame(waveform_type=2, pwm_width_q15=12000),
    frame(waveform_type=2, pwm_width_q15=27000), frame(waveform_type=1),
    frame(waveform_type=5), frame(waveform_type=4), frame(H_request=2),
    frame(H_request=32), frame(H_request=0), frame(H_request=255)]


def test_every_selector_removal_and_count_change_refreshes_without_restarting_phase(execution_path):
    vm = VMState.create(0)
    start = 0
    for f in SEQUENCE * 2:
        inc = decode_frame(f).coeff_osc_params.phase_inc
        output = vm_step_block(vm, f)[1]
        check_weights(vm, f)
        assert vm.mode_b_state.phase == (start + 64 * inc) % 2**32
        if f.H_request == 0:
            assert not np.any(output)
        start = vm.mode_b_state.phase


@pytest.mark.parametrize('count', [0, 1, 8, 16, 32, 255])
@pytest.mark.parametrize('kind', ['saw', 'custom', 'pwm', 'oeis'])
def test_request_and_pitch_cap_are_enforced_for_all_spectra(count, kind, execution_path):
    kwargs = dict(H_request=count, note=103, cents=147)
    if kind == 'custom':
        kwargs['partial_amps_q15'] = [10000] * 32
    elif kind == 'pwm':
        kwargs.update(pwm_enabled=True, pwm_width_q15=9000)
    elif kind == 'oeis':
        kwargs.update(oeis_sequence_id=1)
    f = frame(**kwargs)
    vm = VMState.create(0)
    vm_step_block(vm, f)
    check_weights(vm, f)
    assert vm.mode_b_state.H_active * decode_frame(f).coeff_osc_params.phase_inc <= 9 * 2**31 // 10


@pytest.mark.parametrize('note', [0, 20, 21, 69, 108, 109, 127])
def test_source_anchor_basis_matches_independent_sample_phases(note, execution_path):
    from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit
    f = frame(note=note, cents=0, waveform_type=0)
    params = decode_frame(f).coeff_osc_params
    vm_step_block(VMState.create(0), f)
    basis = bank.get_basis_for_pitch(note, params.phase_inc, 1)
    for lane in range(bank.VM_K):
        samples = [bank.sin_cos_q15(((lane + n) * params.phase_inc) % 2**32) for n in range(64)]
        np.testing.assert_array_equal(basis.Usin_h[0, :, lane], fwpt_jit([s for s, _ in samples], headroom_bits=2))
        np.testing.assert_array_equal(basis.Ucos_h[0, :, lane], fwpt_jit([c for _, c in samples], headroom_bits=2))
    assert not basis.Usin_h.flags.writeable


@pytest.mark.parametrize('note,cents', [(20, -99), (21, 37), (69, 25), (109, 80), (127, -50)])
def test_detuned_basis_uses_independent_anchor_oracle_and_exact_phase(note, cents, execution_path):
    # WSE-STD-210 section 5 specifies fixed pitch anchors and fixed-point
    # interpolation. It does not require a separately transformed bank at
    # every target pitch. Retain exact comparisons for both contracts:
    # original FWPT coefficients at source anchors (above), and the independent
    # Fraction interpolation oracle between anchors. Phase is never quantized.
    from tests.conformance.test_basis_resource import reference_basis
    f = frame(note=note, cents=cents, waveform_type=0)
    params = decode_frame(f).coeff_osc_params
    state = VMState.create(0)
    vm_step_block(state, f)
    assert state.mode_b_state.phase == (64 * params.phase_inc) % 2**32
    actual = bank.get_basis_for_pitch(note, params.phase_inc, 1)
    expected = reference_basis(note, params.phase_inc, 1)
    np.testing.assert_array_equal(actual.Ucos_h, expected.Ucos_h)
    np.testing.assert_array_equal(actual.Usin_h, expected.Usin_h)
    assert not actual.Ucos_h.flags.writeable and not actual.Usin_h.flags.writeable


def test_legato_and_cents_sweep_rebuilds_caps_and_restores_removed_high_partials(execution_path):
    vm = VMState.create(0)
    previous = 0
    for note, cents in [(100, 0), (100, 600), (100, -700), (48, 0), (127, 0), (24, 0)]:
        f = frame(note=note, cents=cents, H_request=32, legato_mode=True)
        vm_step_block(vm, f)
        check_weights(vm, f)
        assert vm.mode_b_state.phase == (previous + 64 * decode_frame(f).coeff_osc_params.phase_inc) % 2**32
        previous = vm.mode_b_state.phase
    assert vm.mode_b_state.H_active == 32 and vm.mode_b_state.weights.a_h_q15[31] != 0


@pytest.mark.parametrize('oeis,partials,pwm', [(1, [1000]*32, True), (-1, [1000]*32, True), (1, [], True)])
def test_selection_priority_cannot_be_overwritten_by_pwm_later_in_the_block(oeis, partials, pwm, execution_path):
    vm = VMState.create(0)
    f = frame(oeis_sequence_id=oeis, partial_amps_q15=partials, pwm_enabled=pwm)
    for width in (0, 8192, 20000, 32767):
        f.pwm_width_q15 = width
        vm_step_block(vm, f)
        check_weights(vm, f)
        assert not vm.mode_b_state.pwm_enabled


def test_full_vm_trace_matches_independent_spectra_and_fraction_coefficients(monkeypatch, execution_path):
    def render():
        vm = VMState.create(0)
        outputs = []
        for f in SEQUENCE:
            outputs.append(vm_step_block(vm, f)[1].copy())
            check_weights(vm, f)
        return np.stack(outputs)
    actual = render()
    calls = []
    def independent(state, basis, output):
        calls.append(state.phase)
        output[:] = reference(state, basis)
    monkeypatch.setattr(bank, 'mode_b_generate_coeffs_vectorized', independent)
    np.testing.assert_array_equal(actual, render())
    assert len(calls) == len(SEQUENCE) and np.count_nonzero(actual) > 500


def test_mutable_input_and_shared_cache_never_alias_another_voice(execution_path):
    f = frame(partial_amps_q15=[20000, 1000, -6000])
    a, b = VMState.create(0), VMState.create(1)
    vm_step_block(a, f)
    vm_step_block(b, f)
    old = b.mode_b_state.weights.a_h_q15.copy()
    f.partial_amps_q15[1] = 11000
    vm_step_block(a, f)
    assert a.mode_b_state.weights.a_h_q15[1] == 11000
    np.testing.assert_array_equal(b.mode_b_state.weights.a_h_q15, old)
    a.mode_b_state.weights.a_h_q15.fill(0)
    vm_step_block(a, f)
    check_weights(a, f)


def test_cold_cache_json_binary_and_fresh_process_replay_across_spectrum_changes(tmp_path, execution_path):
    vm = VMState.create(0)
    for f in SEQUENCE[:5]:
        vm_step_block(vm, f)
    snapshot = json.loads(json.dumps(vm.snapshot()))
    path = tmp_path / 'carrier.json'
    path.write_text(json.dumps(snapshot))
    json_vm, binary_vm = VMState.restore(snapshot), vmstate_from_bytes(vmstate_to_bytes(vm))
    _weights.cache_clear()
    bank.get_basis_for_pitch.cache_clear()
    audio = []
    for f in SEQUENCE[5:]:
        expected = vm_step_block(vm, f)[1].copy()
        np.testing.assert_array_equal(expected, vm_step_block(json_vm, f)[1])
        np.testing.assert_array_equal(expected, vm_step_block(binary_vm, f)[1])
        audio.append(expected.tolist())
    code = '''import json, sys
from symcrash_python_ap.vm import VMState, vm_step_block, hotpath_jit
from tests.conformance.test_carrier_spectrum import SEQUENCE
hotpath_jit.set_jit_enabled(sys.argv[2] == '1')
vm = VMState.restore(json.load(open(sys.argv[1])))
print(json.dumps([vm_step_block(vm, f)[1].tolist() for f in SEQUENCE[5:]]))
'''
    result = subprocess.run([sys.executable, '-c', code, str(path), str(int(hotpath_jit.jit_enabled()))],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == audio


def test_native_capsule_and_offline_authoring_retain_all_requested_custom_partials(execution_path):
    authored = source('partials', harmonics=HarmonicProfile([.2] * 32))
    engines = [owner(p) for p in routes(authored)]
    builder = IRFrameBuilder(authored, 69, 24000)
    for block in range(5):
        frames = [e.step_voice_simple(0) for e in engines] + [builder.build(NoteClock(block))]
        assert frames[0] == frames[1]
        assert all(f.H_request == 32 for f in frames)
        for f in frames:
            vm = VMState.create(0)
            vm_step_block(vm, f)
            check_weights(vm, f)
            assert vm.mode_b_state.H_active == 32


def test_oeis_priority_is_independent_of_unused_authored_custom_count(execution_path):
    original = source('partials', harmonics=HarmonicProfile([.2] * 3), oeis_sequence_id=1)
    without = deepcopy(original)
    without.layers.L0_graph.nodes[0].params.harmonics = None
    engines = [owner(p) for s in (original, without) for p in routes(s)]
    builders = [IRFrameBuilder(s, 69, 24000) for s in (original, without)]
    vms = [VMState.create(0) for _ in range(6)]
    for block in range(8):
        frames = [e.step_voice_simple(0) for e in engines] + [b.build(NoteClock(block)) for b in builders]
        assert all(f.H_request == 16 for f in frames)
        audio = [vm_step_block(vm, f)[1].copy() for vm, f in zip(vms, frames)]
        for result in audio[1:4]:
            np.testing.assert_array_equal(audio[0], result)
        np.testing.assert_array_equal(audio[4], audio[5])


@pytest.mark.parametrize('field,value', [('H_request', -1), ('H_request', 256), ('H_request', True),
    ('H_request', 16.0), ('waveform_type', 6), ('oeis_sequence_id', 8),
    ('oeis_rolloff_db', float('nan')), ('oeis_rolloff_db', 1), ('pwm_enabled', 1),
    ('pwm_width_q15', -1), ('pwm_width_q15', 32768), ('partial_phase_spread_q15', 32768),
    ('partial_amps_q15', [32768]), ('partial_amps_q15', [1.0]), ('partial_amps_q15', [True])])
def test_invalid_native_spectrum_is_rejected_before_any_vm_state_change(field, value):
    f, vm = frame(), VMState.create(0)
    setattr(f, field, value)
    before = vm.snapshot()
    with pytest.raises(ValueError):
        vm_step_block(vm, f)
    assert vm.snapshot() == before
