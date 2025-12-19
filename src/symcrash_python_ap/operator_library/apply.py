"""
Operator Application (Bit-Accurate)
===================================

Functions for applying operators to coefficient vectors.
"""

import numpy as np
from typing import Dict, List

from . import (
    Operator, OpType,
    COEFF_LEN, BAND_SLICES,
    Q15_SCALE, Q15_ONE,
    to_q15, from_q15, q15_mul
)


# =============================================================================
# OPERATOR APPLICATION
# =============================================================================

def apply_diag(c: np.ndarray, op: Operator) -> np.ndarray:
    """Apply diagonal operator."""
    return q15_mul(c, op.diag)


def apply_permute(c: np.ndarray, op: Operator) -> np.ndarray:
    """Apply permutation operator."""
    return c[op.perm]


def apply_lowrank(c: np.ndarray, op: Operator) -> np.ndarray:
    """Apply low-rank operator: c + sum U_k (V_k . c)."""
    result = c.copy().astype(np.int32)

    rank = op.U.shape[0]
    for k in range(rank):
        # V_k . c (dot product)
        dot = np.sum(op.V[k].astype(np.int32) * c.astype(np.int32)) >> 15
        dot = np.clip(dot, -Q15_SCALE, Q15_ONE)

        # U_k * dot
        contribution = (op.U[k].astype(np.int32) * dot) >> 15
        result += contribution

    return np.clip(result, -Q15_SCALE, Q15_ONE).astype(np.int16)


def apply_banded_intra(c: np.ndarray, op: Operator) -> np.ndarray:
    """Apply banded intra-band mixing."""
    result = np.zeros_like(c, dtype=np.int32)

    for band_id, (start, end) in enumerate(BAND_SLICES):
        kernel_info = op.bands_kernel.get(band_id)
        if kernel_info is None:
            result[start:end] = c[start:end]
            continue

        kernel = kernel_info["kernel"]  # [a, b, a] in q15

        for i in range(start, end):
            # Reflection boundary handling
            i_left = max(start, i - 1)
            i_right = min(end - 1, i + 1)

            # Tridiagonal: a*c[i-1] + b*c[i] + a*c[i+1]
            val = (kernel[0].astype(np.int32) * c[i_left] +
                   kernel[1].astype(np.int32) * c[i] +
                   kernel[2].astype(np.int32) * c[i_right]) >> 15
            result[i] = val

    return np.clip(result, -Q15_SCALE, Q15_ONE).astype(np.int16)


def apply_inject(c: np.ndarray, op: Operator, amplitude: float = 1.0) -> np.ndarray:
    """Apply injection operator."""
    amp_q15 = to_q15(np.array([amplitude]))[0]
    inject_scaled = q15_mul(op.inject_vec, np.full_like(op.inject_vec, amp_q15))

    result = c.astype(np.int32) + inject_scaled.astype(np.int32)
    return np.clip(result, -Q15_SCALE, Q15_ONE).astype(np.int16)


def apply_proj_energy_cap(c: np.ndarray, op: Operator) -> np.ndarray:
    """Apply energy capping projection."""
    targets = from_q15(op.proj_params["targets_q15"])
    epsilon = op.proj_params["epsilon"]

    result = c.copy().astype(np.float64) / Q15_SCALE

    for band_id, (start, end) in enumerate(BAND_SLICES):
        band = result[start:end]
        energy = np.sum(band**2)
        target_energy = np.sum(targets[start:end]**2)

        if energy > target_energy + epsilon:
            scale = np.sqrt(target_energy / (energy + epsilon))
            result[start:end] *= scale

    return to_q15(result)


def apply_operator(c: np.ndarray, op: Operator, **kwargs) -> np.ndarray:
    """Apply any operator to coefficient vector."""

    if op.op_type == OpType.DIAG:
        return apply_diag(c, op)

    elif op.op_type == OpType.PERMUTE:
        return apply_permute(c, op)

    elif op.op_type == OpType.LOWRANK:
        return apply_lowrank(c, op)

    elif op.op_type == OpType.BANDED:
        # Check if it's intra-band mixing
        if op.bands_kernel and "kernel" in list(op.bands_kernel.values())[0]:
            return apply_banded_intra(c, op)
        else:
            # Cross-band diffuse (simplified: just return c for now)
            return c

    elif op.op_type == OpType.INJECT:
        amp = kwargs.get("amplitude", 1.0)
        return apply_inject(c, op, amplitude=amp)

    elif op.op_type == OpType.PROJ:
        proj_type = op.proj_params.get("type", "energy_cap")
        if proj_type == "energy_cap":
            return apply_proj_energy_cap(c, op)
        elif proj_type == "normalize":
            return apply_proj_energy_cap(c, op)  # Same logic
        else:
            return c

    elif op.op_type == OpType.MIX:
        # MIX requires library reference (handle in orchestrator)
        return c

    return c


# =============================================================================
# OPERATOR WORD COMPOSITION
# =============================================================================

def compose_word(c: np.ndarray, op_ids: List[int], library: Dict[int, Operator]) -> np.ndarray:
    """Apply sequence of operators (operator word)."""
    result = c.copy()

    for op_id in op_ids:
        if op_id in library:
            result = apply_operator(result, library[op_id])

    return result
