"""Exact host acceleration of the existing CCS energy projections.

Stored coefficients remain int16, with the existing K=2/4/8 layout and Q15
rounding. Energy is at most 512 * 32768**2 = 2**39. Computing budget * 2**30
directly could require 69 bits; two radix-2**15 divisions compute the same
quotient with intermediates below 2**54. No floating-point sqrt is used.

Hard projection needs no coefficient scratch. Normalization copies only an
over-budget band, as the reference does: at most 22*K int16 values (352 bytes
at K=8). The bounded policy cache retains references to the existing immutable
six-weight tuples, not a new table representation or hardware allocation.
"""
from functools import lru_cache
import operator

import numpy as np

from symcrash_python_ap.primitives import Q15_ONE, Q15_SCALE
from symcrash_python_ap.tables.coefficient_tail import (
    CoefficientTailPolicy, CURVE_ENERGY_WEIGHTS, DEFAULT_TAIL_POLICY,
    FINE_LEAF_START, LEAF_BANDS,
)
from symcrash_python_ap.vm.hotpath_jit import njit, jit_enabled, _rshift_rne

_prepared = False


@njit(cache=True)
def _energy(values, start, stop):
    result = np.int64(0)
    for index in range(start, stop):
        value = np.int64(values[index])
        result += value * value
    return result


@njit(cache=True)
def _gain_square(budget, current):
    """floor(budget * 2**30 / current), for 0 < budget < current <= 2**39."""
    high, remainder = divmod(budget << 15, current)
    return (high << 15) + ((remainder << 15) // current)


@njit(cache=True)
def _gain_root(square):
    """Integer square root on 0..2**30-1, in exactly 15 bounded steps."""
    low, high = 0, Q15_ONE
    for _ in range(15):
        middle = (low + high + 1) // 2
        if middle * middle <= square:
            low = middle
        else:
            high = middle - 1
    return low


@njit(cache=True)
def _scaled_energy(values, start, stop, gain):
    result = np.int64(0)
    for index in range(start, stop):
        value = _rshift_rne(np.int64(values[index]) * gain, 15)
        result += value * value
    return result


@njit(cache=True)
def _limit(values, start, stop, budget):
    current = _energy(values, start, stop)
    if current <= budget:
        return
    if budget == 0:
        values[start:stop] = 0
        return
    gain = _gain_root(_gain_square(budget, current))
    if _scaled_energy(values, start, stop, gain) > budget:
        low, high = 0, gain - 1
        for _ in range(15):
            if low >= high:
                break
            middle = (low + high + 1) // 2
            if _scaled_energy(values, start, stop, middle) <= budget:
                low = middle
            else:
                high = middle - 1
        gain = low
    for index in range(start, stop):
        values[index] = _rshift_rne(np.int64(values[index]) * gain, 15)


@njit(cache=True)
def _fine(values, slots, ratio, fine_cap):
    split = FINE_LEAF_START * slots
    budget = min(fine_cap, (_energy(values, 0, split) * ratio) >> 15)
    _limit(values, split, len(values), budget)


@njit(cache=True)
def hard_kernel(values, slots, traction, cap_ratio, leaf_cap, fine_cap, global_cap):
    for leaf in range(64):
        _limit(values, leaf * slots, (leaf + 1) * slots, leaf_cap)
    ratio = Q15_SCALE - _rshift_rne((Q15_SCALE - cap_ratio) * traction, 15)
    if traction == Q15_ONE:
        ratio = cap_ratio
    _fine(values, slots, ratio, fine_cap)
    _limit(values, 0, len(values), global_cap)
    _fine(values, slots, ratio, fine_cap)


@njit(cache=True)
def normalize_kernel(values, slots, traction, global_cap, weights):
    if traction == 0:
        return
    total_weight = sum(weights)
    for band_index in range(len(LEAF_BANDS)):
        first, last = LEAF_BANDS[band_index]
        start, stop = first * slots, last * slots
        budget = global_cap * weights[band_index] // total_weight
        if _energy(values, start, stop) <= budget:
            continue
        projected = values[start:stop].copy()
        _limit(projected, 0, len(projected), budget)
        if traction == Q15_ONE:
            values[start:stop] = projected
        else:
            for index in range(stop - start):
                before = np.int64(values[start + index])
                delta = np.int64(projected[index]) - before
                values[start + index] = before + _rshift_rne(delta * traction, 15)


@lru_cache(maxsize=128)
def _weights(policy):
    # Frozen policies are normally validated at construction. Repeat once per
    # distinct policy before entering compiled fixed-width arithmetic.
    policy.__post_init__()
    return CURVE_ENERGY_WEIGHTS[policy.normalize_curve]


def _admit(coeffs, policy, traction):
    if (not jit_enabled() or type(coeffs) is not np.ndarray
            or coeffs.dtype != np.dtype(np.int16) or coeffs.ndim != 2
            or coeffs.shape[0] != 64 or coeffs.shape[1] not in (2, 4, 8)
            or not coeffs.flags.c_contiguous or not coeffs.flags.writeable
            or type(policy) is not CoefficientTailPolicy):
        return None
    try:
        traction = operator.index(traction)
        if not 0 <= traction <= Q15_ONE:
            return None
        weights = _weights(policy)
    except (TypeError, ValueError, OverflowError):
        return None
    return traction, weights


def try_hard_project(coeffs, policy, traction):
    admitted = _admit(coeffs, policy, traction)
    if admitted is None:
        return False
    traction, _ = admitted
    hard_kernel(coeffs.reshape(-1), coeffs.shape[1], traction,
                policy.fineclamp_cap_ratio_q15, policy.leaf_energy_max,
                policy.fine_energy_max, policy.global_energy_max)
    return True


def try_normalize_scale(coeffs, policy, traction):
    admitted = _admit(coeffs, policy, traction)
    if admitted is None:
        return False
    traction, weights = admitted
    normalize_kernel(coeffs.reshape(-1), coeffs.shape[1], traction,
                     policy.global_energy_max, weights)
    return True


def prepare():
    """Prepare the two public signatures before audio service starts."""
    global _prepared
    if not _prepared and jit_enabled():
        coeffs = np.zeros((64, 4), dtype=np.int16)
        try_hard_project(coeffs, DEFAULT_TAIL_POLICY, Q15_ONE)
        try_normalize_scale(coeffs, DEFAULT_TAIL_POLICY, Q15_ONE)
        _prepared = True
