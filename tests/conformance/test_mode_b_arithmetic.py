"""Independent integer arithmetic for every Mode B coefficient generator."""
from fractions import Fraction

import numpy as np
import pytest

from symcrash_python_ap.vm import harmonic_bank as bank
from tests.conformance.test_mode_b_phase_modulation import generate
from tests.conformance.test_vm_snapshot_replay import execution_path


def rounded(product):
    return round(Fraction(int(product), 2**15))


def reference(state, basis):
    h_count = min(state.H_active, state.H_bandlimit, basis.H_max)
    total = sum(max(abs(int(state.weights.a_h_q15[h])), abs(int(state.weights.b_h_q15[h])))
                for h in range(h_count))
    shift = 0
    while total // (2**shift) > 32767:
        shift += 1
    result = []
    for leaf in range(64):
        for lane in range(bank.VM_K):
            phase = state.phase
            if state.fm_mod_buffer is not None and lane < len(state.fm_mod_buffer):
                phase += int(state.fm_mod_buffer[lane]) * state.fm_depth_q15 * 4
            accumulator = rounded(state.weights.dc_q15 * int(basis.DC_frame[leaf, lane]))
            for h in range(h_count):
                sine, cosine = bank.sin_cos_q15((phase * (h + 1)) % 2**32)
                a = int(state.weights.a_h_q15[h]) // 2**shift
                b = int(state.weights.b_h_q15[h]) // 2**shift
                sin_weight = rounded(a * cosine - b * sine)
                cos_weight = rounded(a * sine + b * cosine)
                accumulator += rounded(sin_weight * int(basis.Usin_h[h, leaf, lane]))
                accumulator += rounded(cos_weight * int(basis.Ucos_h[h, leaf, lane]))
            result.append(max(-32768, min(32767, accumulator)))
    return np.array(result, dtype=np.int16)


@pytest.mark.parametrize('shift', [1, 2, 7, 15, 31])
def test_vector_reduction_matches_exact_fraction_at_signed_halfway_boundaries(shift):
    values = np.array([q * (1 << shift) + (1 << (shift - 1)) + delta
                       for q in range(-17, 18) for delta in [-1, 0, 1]], dtype=np.int64)
    expected = [round(Fraction(int(value), 1 << shift)) for value in values]
    np.testing.assert_array_equal(bank._rshift_rne_array(values, shift), expected)


@pytest.mark.parametrize('harmonics', [0, 1, 2, 7, 16, 32])
@pytest.mark.parametrize('phase,modulated', [(0, False), (0x40000000, True), (3788812303, True)])
def test_all_generators_match_independent_wide_accumulator(harmonics, phase, modulated, execution_path):
    rng = np.random.default_rng(210 + harmonics)
    state = bank.ModeBState(phase=phase, H_active=harmonics, H_bandlimit=32,
        fm_depth_q15=32767 if modulated else 0,
        fm_mod_buffer=np.array([-32768, 16384, -7, 32767], dtype=np.int16) if modulated else None)
    state.weights.a_h_q15[:harmonics] = rng.integers(-32768, 32768, size=harmonics, dtype=np.int16)
    state.weights.b_h_q15[:harmonics] = rng.integers(-32768, 32768, size=harmonics, dtype=np.int16)
    state.weights.dc_q15 = -23211
    basis = bank.get_basis_for_note(69)
    expected = reference(state, basis)
    for kind in ('scalar', 'vector') + (('hw',) if harmonics <= bank.HW_H_MAX else ()):
        np.testing.assert_array_equal(generate(kind, state, basis), expected, err_msg=kind)


def synthetic_basis():
    # Legal Q15 basis values chosen to exercise arithmetic, independently of a
    # particular frozen transform atlas. Every lane has the same boundary case.
    return bank.HarmonicBasisFrames(np.full((64, 4), 32767, dtype=np.int16),
        np.zeros((16, 64, 4), dtype=np.int16), np.zeros((16, 64, 4), dtype=np.int16),
        16, 69, 440.0)


@pytest.mark.parametrize('sign', [-1, 1])
def test_harmonic_cancellation_occurs_before_final_saturation(sign, execution_path):
    state = bank.ModeBState(H_active=2)
    state.weights.a_h_q15[:2] = [16000, -16000]
    state.weights.dc_q15 = 30000 * sign
    basis = synthetic_basis()
    basis.Usin_h[:2] = 32767 * sign
    expected = reference(state, basis)
    assert int(expected[0]) == rounded(state.weights.dc_q15 * 32767)
    assert abs(int(expected[0])) > 29000
    for kind in ('scalar', 'vector', 'hw'):
        np.testing.assert_array_equal(generate(kind, state, basis), expected, err_msg=kind)


@pytest.mark.parametrize('dc', [-32768, -1, 1, 32767])
def test_zero_harmonics_retains_dc(dc, execution_path):
    state = bank.ModeBState(H_active=0)
    state.weights.dc_q15 = dc
    basis = synthetic_basis()
    basis.DC_frame[0, :] = [-32768, -16384, 16384, 32767]
    for kind in ('scalar', 'vector', 'hw'):
        np.testing.assert_array_equal(generate(kind, state, basis), reference(state, basis))


def test_recorded_halfway_fixture_matches_exact_rounding(execution_path):
    # Reproduce the deterministic pre-fix probe without importing generated data.
    rng = np.random.default_rng(210)
    for harmonics in [1, 2, 7, 16]:
        for _ in range(4):
            state = bank.ModeBState(phase=int(rng.integers(0, 2**32)), H_active=harmonics)
            state.weights.set_custom([int(x) for x in rng.integers(1, 20001, size=harmonics)], 18345)
    assert state.phase == 3788812303
    basis = bank.get_basis_for_note(69)
    expected = reference(state, basis)
    assert (int(expected[71]), int(expected[114])) == (54, -26)
    for kind in ('scalar', 'vector', 'hw'):
        np.testing.assert_array_equal(generate(kind, state, basis), expected)


def test_subtractive_golden_audio_matches_independent_coefficient_generator(monkeypatch, execution_path):
    from symcrash_python_ap.vm import VMState, vm_step_block
    from tests.artifacts.golden_trace_subtractive_v0 import generate_golden_sequence

    def render():
        state = VMState.create(0)
        state.reset_for_note_on(note=69, cents=0)
        return np.stack([vm_step_block(state, frame)[1].copy()
                         for frame in generate_golden_sequence()])

    actual = render()
    calls = []

    def exact_generator(state, basis, output):
        calls.append(state.phase)
        output[:] = reference(state, basis)

    monkeypatch.setattr(bank, 'mode_b_generate_coeffs_vectorized', exact_generator)
    np.testing.assert_array_equal(actual, render())
    assert calls and np.count_nonzero(actual)
