"""
Operator Cost Table (Canonical Table v0)
========================================

Per CMMM-2 spec, operator costs use the formula:
    ticks = base + (ticks_per_active_node * K) + (ticks_per_total_node * L)

Where:
- K = active nodes (popcount of node mask)
- L = total nodes (32 for COEFF_LEN=256, NODE_LEN=8)

Cost categories:
- DIAG ops: scale with K (cheap to budget down)
- BANDED ops: scale with K (local kernels)
- PERMUTE ops: scale with L (global touch, expensive)
- LOWRANK ops: scale with L (dense global)
- INJECT ops: mostly fixed cost
- PROJ ops: scale with L (global measurement)

Tick unit: 1 tick = 1 Q15 multiply-add on 1 coefficient
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


# Constants
L_TOTAL_NODES = 32  # COEFF_LEN(256) / NODE_LEN(8)
NODE_LEN = 8


@dataclass(frozen=True)
class OperatorCost:
    """
    Cost model for a single operator.

    Attributes:
        op_id: Operator ID (0-23)
        ticks_base: Fixed overhead
        ticks_per_active_node: Cost per active node K
        ticks_per_total_node: Cost per total node L (global ops)
        nonlinear: True if op uses nonlinear math (sqrt, etc.)
        is_tail: True if op is a tail safety operator
        drop_priority: 0=never drop, 255=drop first
    """
    op_id: int
    ticks_base: int
    ticks_per_active_node: int
    ticks_per_total_node: int
    nonlinear: bool
    is_tail: bool
    drop_priority: int


# =============================================================================
# CANONICAL OPERATOR COST TABLE (v0)
# =============================================================================
# Based on CMMM-2 spec analysis of operator types

OPERATOR_COSTS: Dict[int, OperatorCost] = {
    # DIAG ops (scale with K)
    0:  OperatorCost(0,  0,   0, 0, False, False,   0),   # Identity (no-op)
    1:  OperatorCost(1,  8,   1, 0, False, False,  80),   # Tilt
    15: OperatorCost(15, 8,   1, 0, False, False, 160),   # PressureRamp
    16: OperatorCost(16, 8,   1, 0, False, False,  40),   # GateMask
    20: OperatorCost(20, 8,   1, 0, False, False, 160),   # FoldPressure
    22: OperatorCost(22, 8,   1, 0, False, False, 180),   # SwallowtailMask

    # BANDED ops (scale with K, local kernels)
    3:  OperatorCost(3,  16,  3, 0, False, False, 110),   # IntraMix
    4:  OperatorCost(4,  16,  2, 0, False, False, 120),   # DiffuseUp
    5:  OperatorCost(5,  16,  2, 0, False, False, 120),   # DiffuseDown

    # PERMUTE ops (scale with L, global touch)
    6:  OperatorCost(6,  32,  0, 1, False, False, 200),   # BandShear
    9:  OperatorCost(9,  32,  0, 1, False, False, 220),   # SwapBands
    10: OperatorCost(10, 32,  0, 1, False, False, 190),   # RotateNodes
    17: OperatorCost(17, 32,  0, 1, False, False, 210),   # QuasiRotate

    # LOWRANK ops (scale with L, dense global)
    7:  OperatorCost(7,  64,  0, 4, False, False, 230),   # LowRankMorph

    # INJECT ops (events, mostly fixed)
    8:  OperatorCost(8,  16,  1, 0, False, False, 240),   # GlobalPulse
    13: OperatorCost(13, 8,   1, 0, False, False, 245),   # Impulse
    14: OperatorCost(14, 16,  1, 0, False, False, 250),   # NoiseBand
    21: OperatorCost(21, 16,  1, 0, False, False, 248),   # CuspInjection

    # PROJ ops (scale with L, tail candidates)
    2:  OperatorCost(2,  64,  0, 2, False, True,    0),   # FineClamp (TAIL)
    11: OperatorCost(11, 96,  0, 3, True,  True,    0),   # NormalizeScale (TAIL, nonlinear)
    12: OperatorCost(12, 48,  0, 1, False, False,  60),   # GluePair
    19: OperatorCost(19, 64,  0, 2, False, True,    0),   # EmergencySimplify (TAIL)
    23: OperatorCost(23, 96,  0, 3, False, True,   10),   # SheafProject (TAIL)

    # MIX ops (overhead only, add component costs separately)
    18: OperatorCost(18, 16,  0, 0, False, False, 140),   # MorphMix
}


def predict_ticks(
    op_ids: List[int],
    K_active_nodes: int,
    L_total_nodes: int = L_TOTAL_NODES
) -> Tuple[int, bool]:
    """
    Predict total tick cost for a list of operators.

    Args:
        op_ids: List of operator IDs to execute
        K_active_nodes: Number of active nodes
        L_total_nodes: Total nodes (default 32)

    Returns:
        Tuple of (total_ticks, uses_nonlinear)
    """
    ticks = 0
    uses_nonlinear = False

    for op_id in op_ids:
        if op_id not in OPERATOR_COSTS:
            continue  # Unknown op, skip

        cost = OPERATOR_COSTS[op_id]
        ticks += (
            cost.ticks_base +
            cost.ticks_per_active_node * K_active_nodes +
            cost.ticks_per_total_node * L_total_nodes
        )
        uses_nonlinear |= cost.nonlinear

    return ticks, uses_nonlinear


def get_operator_cost(op_id: int) -> OperatorCost:
    """Get cost entry for an operator, or Identity cost if unknown."""
    return OPERATOR_COSTS.get(op_id, OPERATOR_COSTS[0])


def is_tail_operator(op_id: int) -> bool:
    """Check if an operator is a tail safety operator."""
    cost = OPERATOR_COSTS.get(op_id)
    return cost.is_tail if cost else False


def is_global_operator(op_id: int) -> bool:
    """Check if an operator scales with L (global touch)."""
    cost = OPERATOR_COSTS.get(op_id)
    return cost.ticks_per_total_node > 0 if cost else False


# =============================================================================
# OPERATOR CATEGORIES (for degradation logic)
# =============================================================================

# Operators by category for degradation ordering
INJECT_OPS = {8, 13, 14, 21}      # Drop first in degradation
PERMUTE_OPS = {6, 9, 10, 17}      # Drop second
LOWRANK_OPS = {7}                  # Drop third
TAIL_OPS = {2, 11, 19, 23}         # Never drop (unless tail mode change)
DIAG_OPS = {0, 1, 15, 16, 20, 22}  # Keep longest (cheap)
BANDED_OPS = {3, 4, 5}             # Keep with DIAG


def categorize_operators(op_ids: List[int]) -> Dict[str, List[int]]:
    """
    Categorize operators for degradation ordering.

    Returns dict with keys: 'inject', 'permute', 'lowrank', 'tail', 'backbone'
    """
    result = {
        'inject': [],
        'permute': [],
        'lowrank': [],
        'tail': [],
        'backbone': [],
    }

    for op_id in op_ids:
        if op_id in INJECT_OPS:
            result['inject'].append(op_id)
        elif op_id in PERMUTE_OPS:
            result['permute'].append(op_id)
        elif op_id in LOWRANK_OPS:
            result['lowrank'].append(op_id)
        elif op_id in TAIL_OPS:
            result['tail'].append(op_id)
        else:
            result['backbone'].append(op_id)

    return result
