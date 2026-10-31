"""Tail acceleration against arbitrary-precision and independent scalar oracles."""
from dataclasses import replace
from math import isqrt

import numpy as np
import pytest

from symcrash_python_ap.primitives import TailMode
from symcrash_python_ap.tables.coefficient_tail import (
    CoefficientTailPolicy, DEFAULT_TAIL_POLICY, MAX_ENERGY_BUDGET,
)
from symcrash_python_ap.vm import coefficient_tail as tail, hotpath_jit as hot
from symcrash_python_ap.vm import coefficient_tail_jit as jit
from symcrash_python_ap.vm.decode import DecodedPlan
from symcrash_python_ap.vm.state import VMState
from tests.conformance.test_coefficient_tail import reference_tail


@pytest.fixture(autouse=True)
def enabled():
    old = hot.jit_enabled()
    if not hot.NUMBA_AVAILABLE:
        pytest.skip('Numba unavailable')
    hot.set_jit_enabled(True)
    jit.prepare()
    try:
        yield
    finally:
        hot.set_jit_enabled(old)


def test_ratio_retains_all_69_bits_of_the_reference_numerator():
    rng = np.random.default_rng(7001)
    pairs = [(current - offset, current)
             for current in (2, 3, 32768, 2**30, 2**32, 2**38, 2**39)
             for offset in (1, 2, 32767, current // 2) if 0 < offset < current]
    for _ in range(10000):
        current = int(rng.integers(2, 2**39 + 1))
        pairs.append((int(rng.integers(1, current)), current))
    assert max((budget << 30).bit_length() for budget, _ in pairs) == 69
    for budget, current in pairs:
        assert jit._gain_square(budget, current) == (budget << 30) // current
        # Both shifted dividend/remainder are bounded without discarding bits.
        assert (budget << 15) < 2**54
        assert (((budget << 15) % current) << 15) < 2**54


def test_integer_root_at_every_possible_gain_boundary():
    for root in range(32768):
        for square in (max(0, root * root - 1), root * root,
                       min(2**30 - 1, (root + 1) ** 2 - 1)):
            assert jit._gain_root(square) == isqrt(square)


def test_range_proofs_cover_every_policy_and_layout():
    energy = 512 * 32768**2
    assert energy == MAX_ENERGY_BUDGET == 2**39
    assert energy * 32768 <= 2**54  # fine/coarse ratio
    assert energy * 3600 < 2**51  # largest normalization weight
    assert 65535 * 32767 < 2**31  # conservative signed interpolation product
    assert 22 * 8 * 2 == 352  # largest int16 band copy, unchanged layout


@pytest.mark.parametrize('slots', [2, 4, 8])
def test_energy_caps_match_reference_through_rails_and_rounding_overshoot(slots):
    rng = np.random.default_rng(7010 + slots)
    sources = [np.zeros(64 * slots, dtype=np.int16),
               np.full(64 * slots, -32768, dtype=np.int16),
               np.resize(np.arange(-3, 4, dtype=np.int16), 64 * slots)]
    sources.extend(rng.integers(-32768, 32768, 64 * slots, dtype=np.int16)
                   for _ in range(40))
    for source in sources:
        power = sum(int(v)**2 for v in source)
        budgets = (0, 1, 2, 17, max(0, power - 1), power // 3,
                   power, MAX_ENERGY_BUDGET)
        for budget in budgets:
            expected, actual = source.copy(), source.copy()
            tail.limit_energy(expected, budget)
            jit._limit(actual, 0, len(actual), budget)
            np.testing.assert_array_equal(actual, expected)
            assert sum(int(v)**2 for v in actual) <= budget
    # Small budgets explicitly exercise rounded-energy overshoot correction.
    for budget in range(29):
        source = np.arange(-3, 4, dtype=np.int16)
        expected, actual = source.copy(), source.copy()
        tail.limit_energy(expected, budget)
        jit._limit(actual, 0, len(actual), budget)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('slots', [2, 4, 8])
@pytest.mark.parametrize('mode', list(TailMode))
@pytest.mark.parametrize('curve', ['flat', 'pink', 'bright'])
@pytest.mark.parametrize('traction', [0, 1, 16384, 32766, 32767])
def test_whole_tail_matches_independent_scalar_with_slew_and_pairs(slots, mode, curve, traction):
    rng = np.random.default_rng(7020 + slots)
    state = VMState()
    state.coeff_state = rng.integers(-32768, 32768, 64 * slots, dtype=np.int16)
    state.tail_previous_coeffs = np.full(64 * slots, -20000, dtype=np.int16)
    expected = VMState()
    expected.coeff_state = state.coeff_state.copy()
    expected.tail_previous_coeffs = state.tail_previous_coeffs.copy()
    policy = CoefficientTailPolicy(normalize_curve=curve, leaf_energy_max=2**27,
        fine_energy_max=2**29, global_energy_max=2**32,
        fineclamp_cap_ratio_q15=123, slew_max_delta=29,
        sheaf_pairs=((0, 2), (59, 60)))
    cfg = DecodedPlan(tail_mode=mode, tail_policy=policy, traction_q15=traction)
    tail.apply_coefficient_tail(state, cfg)
    reference_tail(expected, cfg)
    np.testing.assert_array_equal(state.coeff_state, expected.coeff_state)
    np.testing.assert_array_equal(state.tail_previous_coeffs, expected.tail_previous_coeffs)


@pytest.mark.parametrize('slots', [2, 4, 8])
def test_prepared_signatures_and_default_caps_do_not_depend_on_layout(slots):
    signatures = (list(jit.hard_kernel.signatures), list(jit.normalize_kernel.signatures))
    for traction in (0, np.int64(1), 16384, 32767):
        for curve in ('flat', 'pink', 'bright'):
            policy = replace(DEFAULT_TAIL_POLICY, normalize_curve=curve)
            for method in ('hard_project', 'normalize_scale'):
                source = np.full((64, slots), -32768, dtype=np.int16)
                expected, actual = source.copy(), source.copy()
                hot.set_jit_enabled(False)
                getattr(tail, method)(expected, policy, traction)
                hot.set_jit_enabled(True)
                getattr(tail, method)(actual, policy, traction)
                np.testing.assert_array_equal(actual, expected)
    assert (jit.hard_kernel.signatures, jit.normalize_kernel.signatures) == signatures


@pytest.mark.parametrize('case', ['readonly', 'strided', 'int32', 'float', 'flat',
    'wrong_slots', 'wrong_leaves', 'endian', 'fractional_traction', 'negative_traction',
    'huge_traction', 'unknown_policy', 'huge_policy', 'disabled'])
def test_unadmitted_inputs_fall_back_before_any_write(case):
    values = np.full((64, 4), 1234, dtype=np.int16)
    policy, traction = DEFAULT_TAIL_POLICY, 32767
    if case == 'readonly':
        values.setflags(write=False)
    elif case == 'strided':
        values = np.full((64, 8), 1234, dtype=np.int16)[:, ::2]
    elif case in ('int32', 'float'):
        values = values.astype(np.int32 if case == 'int32' else np.float64)
    elif case == 'flat':
        values = values.reshape(-1)
    elif case == 'wrong_slots':
        values = np.full((64, 3), 1234, dtype=np.int16)
    elif case == 'wrong_leaves':
        values = values[:63]
    elif case == 'endian':
        values = values.astype('>i2')
    elif case == 'fractional_traction':
        traction = 0.5
    elif case == 'negative_traction':
        traction = -1
    elif case == 'huge_traction':
        traction = 1 << 100
    elif case == 'unknown_policy':
        policy = object()
    elif case == 'huge_policy':
        policy = replace(DEFAULT_TAIL_POLICY)
        object.__setattr__(policy, 'global_energy_max', 1 << 100)
    elif case == 'disabled':
        hot.set_jit_enabled(False)
    before = values.copy()
    assert jit.try_hard_project(values, policy, traction) is False
    assert jit.try_normalize_scale(values, policy, traction) is False
    np.testing.assert_array_equal(values, before)


def test_policy_cache_is_bounded_without_reencoding_weights():
    for ratio in range(160):
        policy = replace(DEFAULT_TAIL_POLICY, fineclamp_cap_ratio_q15=ratio)
        weights = jit._weights(policy)
        assert weights is tail.CURVE_ENERGY_WEIGHTS[policy.normalize_curve]
    assert jit._weights.cache_info().currsize <= 128
