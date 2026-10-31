"""Deterministic CCS tail operating on leaf-major temporal slots.

Hard caps apply to every coefficient, including nodes omitted by the musical
operator mask. Traction controls the authored projection, never hard safety.
See the conformance report for this profile's intentionally bounded scope.
"""

from math import isqrt

import numpy as np

from ..primitives import Q15_HALF, Q15_ONE, Q15_SCALE, RSHIFT_RNE
from ..tables.coefficient_tail import (
    CURVE_ENERGY_WEIGHTS, DEFAULT_TAIL_POLICY, FINE_LEAF_START, LEAF_BANDS,
)
from ..tables.tail_modes import get_tail_ops


def energy(values: np.ndarray) -> int:
    wide = values.astype(np.int64, copy=False)
    # Integer dot product performs the same int64 sum without a squared array.
    return int(np.vdot(wide, wide))


def scale_rne(values: np.ndarray, gain: int) -> np.ndarray:
    """Signed Q15 multiply with ties to even and widened intermediates."""
    product = values.astype(np.int64) * gain
    quotient = product >> 15
    remainder = product & (Q15_SCALE - 1)
    return quotient + ((remainder > Q15_HALF) |
                       ((remainder == Q15_HALF) & ((quotient & 1) != 0)))


def limit_energy(values: np.ndarray, budget: int) -> None:
    """Attenuate in place; integer rounding must not exceed the energy cap.

    Usually one sqrt and one scale suffice. In a rounding overshoot, a fixed
    maximum of 15 binary-search steps finds a safe Q15 gain. Rounded sample
    magnitudes are monotone in gain, so no data-dependent convergence loop is
    needed. Already legal input is bit-exact identity.
    """
    current = energy(values)
    if current <= budget:
        return
    if budget == 0:
        values.fill(0)
        return
    gain = isqrt((budget << 30) // current)
    scaled = scale_rne(values, gain)
    if energy(scaled) > budget:
        lo, hi = 0, gain - 1
        for _ in range(15):
            if lo >= hi:
                break
            mid = (lo + hi + 1) // 2
            if energy(scale_rne(values, mid)) <= budget:
                lo = mid
            else:
                hi = mid - 1
        scaled = scale_rne(values, lo)
    values[:] = scaled


def _fine_limit(coeffs, policy, traction):
    # At zero traction allow a 1:1 fine/coarse energy ratio; at full traction
    # apply the authored ratio exactly. Hard absolute limits still apply.
    ratio = Q15_SCALE - RSHIFT_RNE(
        (Q15_SCALE - policy.fineclamp_cap_ratio_q15) * traction, 15)
    if traction == Q15_ONE:
        ratio = policy.fineclamp_cap_ratio_q15
    coarse = coeffs[:FINE_LEAF_START]
    fine = coeffs[FINE_LEAF_START:]
    limit_energy(fine, min(policy.fine_energy_max, (energy(coarse) * ratio) >> 15))


def hard_project(coeffs, policy, traction):
    from .coefficient_tail_jit import try_hard_project
    if try_hard_project(coeffs, policy, traction):
        return
    wide = coeffs.astype(np.int64)
    leaf_energies = np.sum(wide * wide, axis=1)
    for index in np.flatnonzero(leaf_energies > policy.leaf_energy_max):
        limit_energy(coeffs[index], policy.leaf_energy_max)
    _fine_limit(coeffs, policy, traction)
    limit_energy(coeffs, policy.global_energy_max)
    # Global rounding can slightly change the fine/coarse ratio.
    _fine_limit(coeffs, policy, traction)


def normalize_scale(coeffs, policy, traction):
    from .coefficient_tail_jit import try_normalize_scale
    if try_normalize_scale(coeffs, policy, traction):
        return
    if traction == 0:
        return
    weights = CURVE_ENERGY_WEIGHTS[policy.normalize_curve]
    total_weight = sum(weights)
    for (start, end), weight in zip(LEAF_BANDS, weights):
        band = coeffs[start:end]
        budget = policy.global_energy_max * weight // total_weight
        if energy(band) <= budget:
            continue
        projected = band.copy()
        limit_energy(projected, budget)
        if traction == Q15_ONE:
            band[:] = projected
        else:
            band[:] = band.astype(np.int64) + scale_rne(
                projected.astype(np.int64) - band, traction)


def sheaf_project(coeffs, policy, traction):
    # No section duplicates/constraints are stored in the default CCS frame.
    # An unconstrained vector is already compatible. Never average unrelated
    # temporal slots as the legacy node-boundary projection did.
    for left, right in policy.sheaf_pairs:
        a = coeffs[left].astype(np.int64)
        b = coeffs[right].astype(np.int64)
        average = scale_rne(a + b, Q15_HALF)
        if traction == Q15_ONE:
            coeffs[left] = average
            coeffs[right] = average
        else:
            coeffs[left] = a + scale_rne(average - a, traction)
            coeffs[right] = b + scale_rne(average - b, traction)


def apply_coefficient_tail(state, cfg):
    policy = cfg.tail_policy or DEFAULT_TAIL_POLICY
    coeffs = state.coeff_state.reshape(64, -1)
    traction = cfg.traction_q15
    ops = get_tail_ops(cfg.tail_mode)
    # Frozen-packet author order cannot move a projection out of the tail.
    # Scheduler §7: FineClamp -> Normalize -> optional Glue -> Sheaf -> Emergency.
    requested = {entry.op_id for entry in cfg.wire_tail_ops}
    ops = [op_id for op_id in (2, 11, 12, 23, 19) if op_id in set(ops) | requested]
    for op_id in ops:
        if op_id == 2:
            hard_project(coeffs, policy, traction)
        elif op_id == 11:
            normalize_scale(coeffs, policy, traction)
        elif op_id == 12:
            from .lanes import dispatch_coeff_op
            entry = next(e for e in cfg.wire_tail_ops if e.op_id == 12)
            dispatch_coeff_op(state, entry, cfg.mask_u32)
        elif op_id == 23:
            sheaf_project(coeffs, policy, traction)
        elif op_id == 19:
            coeffs[FINE_LEAF_START:] = 0

    if policy.slew_max_delta is not None:
        previous = state.tail_previous_coeffs.astype(np.int64)
        delta = state.coeff_state.astype(np.int64) - previous
        state.coeff_state[:] = previous + np.clip(
            delta, -policy.slew_max_delta, policy.slew_max_delta)
    # Slew history and projections may violate a newly tightened cap. Safety
    # takes precedence when no vector can satisfy both the cap and the slew.
    hard_project(coeffs, policy, traction)
    if 19 in ops:
        coeffs[FINE_LEAF_START:] = 0
    state.tail_previous_coeffs[:] = state.coeff_state
