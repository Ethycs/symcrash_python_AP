"""
Operator Cost Table (Canonical Table v0)
========================================

Single source of truth for operator costs. `budgets/operator_costs.py`
re-exports from here.

Cost model (WSE-STD-DEGRADE-v0 section on cost prediction):
    ticks = base + (ticks_per_active_node * K) + (ticks_per_total_node * L)

Where:
- K = active nodes (popcount of node mask)
- L = total nodes (32 for COEFF_LEN=256, NODE_LEN=8)

Tick unit: 1 tick = 1 Q15 multiply-add on 1 COEFFICIENT (not per node).
Per-node terms therefore carry a factor of NODE_LEN=8:
- DIAG: ~1 mul per coeff -> 8 ticks per active node
- BANDED (tridiag): ~3 mul per coeff -> 24 ticks per active node
- PERMUTE: copy/reindex -> 8 ticks per total node
- LOWRANK: dense global (rank=2) -> 32 ticks per total node

NOTE (2026-07-02): CMMM-SPEC.md section 3.3 lists the same table with
per-NODE tick units (values /8). These per-coefficient values are the
active, budget-ladder-calibrated ones; an errata note in CMMM-SPEC
records the unit discrepancy. Do not change values without retuning
budgets/regime_budgets.py.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import IntEnum


# Constants
COEFF_LEN = 256
NODE_LEN = 8
L_TOTAL_NODES = COEFF_LEN // NODE_LEN  # 32


class OpCategory(IntEnum):
    """Operator category for cost scaling."""
    DIAG = 0       # Diagonal scaling (K-scaling)
    BANDED = 1     # Banded/local (K-scaling)
    PERMUTE = 2    # Permutation (L-scaling)
    LOWRANK = 3    # Low-rank global (L-scaling)
    PROJ = 4       # Projection/normalize (L-scaling)
    INJECT = 5     # Injection (K-scaling)
    MIX = 6        # Operator mixing (inherits)


@dataclass(frozen=True)
class OperatorCost:
    """
    Resource cost for a single operator.

    All costs in integer tick units (1 tick = 1 Q15 mul-add on 1 coeff).
    """
    op_id: int
    op_name: str
    category: OpCategory

    # Cost model: ticks = base + K*per_active + L*per_total
    ticks_base: int
    ticks_per_active_node: int   # Scales with K (active nodes)
    ticks_per_total_node: int    # Scales with L (total nodes = 32)

    # Flags
    nonlinear: bool              # Uses sqrt, tanh, etc.
    is_tail: bool                # Must always run if possible

    # Degradation priority (0 = never drop, 255 = drop first)
    drop_priority: int

    def compute_ticks(self, K: int, L: int = L_TOTAL_NODES) -> int:
        """Compute total ticks for given K active nodes."""
        return self.ticks_base + (self.ticks_per_active_node * K) + (self.ticks_per_total_node * L)


# Version for compatibility checking
COST_TABLE_VERSION = "v0.1.0"


def _op(op_id, name, cat, base, per_k, per_l, nonlinear, is_tail, drop):
    return OperatorCost(op_id, name, cat, base, per_k, per_l, nonlinear, is_tail, drop)


# =============================================================================
# CANONICAL OPERATOR COST TABLE (v0)
# =============================================================================

OPERATOR_COSTS: Dict[int, OperatorCost] = {
    # DIAG ops (K-scaling)
    0:  _op(0,  "Identity",          OpCategory.DIAG,    0,   0,  0, False, False,   0),
    1:  _op(1,  "Tilt",              OpCategory.DIAG,    8,   8,  0, False, False,  80),
    15: _op(15, "PressureRamp",      OpCategory.DIAG,    8,   8,  0, False, False, 160),
    16: _op(16, "GateMask",          OpCategory.DIAG,    8,   8,  0, False, False,  40),
    20: _op(20, "FoldPressure",      OpCategory.DIAG,    8,   8,  0, False, False, 160),
    22: _op(22, "SwallowtailMask",   OpCategory.DIAG,    8,   8,  0, False, False, 180),

    # BANDED ops (K-scaling, local kernels)
    3:  _op(3,  "IntraMix",          OpCategory.BANDED, 16,  24,  0, False, False, 110),
    4:  _op(4,  "DiffuseUp",         OpCategory.BANDED, 16,  16,  0, False, False, 120),
    5:  _op(5,  "DiffuseDown",       OpCategory.BANDED, 16,  16,  0, False, False, 120),

    # PERMUTE ops (L-scaling, global touch)
    6:  _op(6,  "BandShear",         OpCategory.PERMUTE, 32,  0,  8, False, False, 200),
    9:  _op(9,  "SwapBands",         OpCategory.PERMUTE, 32,  0,  8, False, False, 220),
    10: _op(10, "RotateNodes",       OpCategory.PERMUTE, 32,  0,  8, False, False, 190),
    17: _op(17, "QuasiRotate",       OpCategory.PERMUTE, 32,  0,  8, False, False, 210),

    # LOWRANK ops (L-scaling, dense global; rank=2 starter)
    7:  _op(7,  "LowRankMorph",      OpCategory.LOWRANK, 64,  0, 32, False, False, 230),

    # INJECT ops (K-scaling, sparse writes; drop first)
    8:  _op(8,  "GlobalPulse",       OpCategory.INJECT, 16,   8,  0, False, False, 240),
    13: _op(13, "Impulse",           OpCategory.INJECT,  8,   8,  0, False, False, 245),
    14: _op(14, "NoiseBand",         OpCategory.INJECT, 16,   8,  0, False, False, 250),
    21: _op(21, "CuspInjection",     OpCategory.INJECT, 16,   8,  0, False, False, 248),

    # PROJ ops (L-scaling, tail candidates)
    2:  _op(2,  "FineClamp",         OpCategory.PROJ,   64,   0, 16, False, True,    0),
    11: _op(11, "NormalizeScale",    OpCategory.PROJ,   96,   0, 24, True,  True,    0),
    12: _op(12, "GluePair",          OpCategory.PROJ,   48,   0,  8, False, False,  60),
    19: _op(19, "EmergencySimplify", OpCategory.PROJ,   64,   0, 16, False, True,    0),
    23: _op(23, "SheafProject",      OpCategory.PROJ,   96,   0, 24, False, True,   10),

    # MIX ops (overhead only; add component costs separately)
    18: _op(18, "MorphMix",          OpCategory.MIX,    16,   0,  0, False, False, 140),
}


# =============================================================================
# ACCESSOR FUNCTIONS
# =============================================================================

def get_operator_cost(op_id: int) -> OperatorCost:
    """
    Get cost for operator by ID.

    Returns a conservative default for unknown operators.
    """
    if op_id in OPERATOR_COSTS:
        return OPERATOR_COSTS[op_id]

    # Conservative default for unknown operators
    return OperatorCost(
        op_id=op_id,
        op_name=f"Unknown_{op_id}",
        category=OpCategory.DIAG,
        ticks_base=100,
        ticks_per_active_node=10,
        ticks_per_total_node=0,
        nonlinear=False,
        is_tail=False,
        drop_priority=255,  # Drop first
    )


def predict_ticks(
    op_ids: List[int],
    K_active_nodes: int,
    L_total_nodes: int = L_TOTAL_NODES
) -> Tuple[int, bool]:
    """
    Predict total tick cost for a list of operators.

    Returns:
        Tuple of (total_ticks, uses_nonlinear)
    """
    ticks = 0
    uses_nonlinear = False

    for op_id in op_ids:
        if op_id not in OPERATOR_COSTS:
            continue  # Unknown op, skip
        cost = OPERATOR_COSTS[op_id]
        ticks += cost.compute_ticks(K_active_nodes, L_total_nodes)
        uses_nonlinear |= cost.nonlinear

    return ticks, uses_nonlinear


def get_tail_operators() -> List[int]:
    """Get list of operator IDs marked as tail."""
    return [op_id for op_id, cost in OPERATOR_COSTS.items() if cost.is_tail]


def get_nonlinear_operators() -> List[int]:
    """Get list of operator IDs that use nonlinear operations."""
    return [op_id for op_id, cost in OPERATOR_COSTS.items() if cost.nonlinear]


def get_operators_by_category(category: OpCategory) -> List[int]:
    """Get all operator IDs in a category."""
    return [op_id for op_id, cost in OPERATOR_COSTS.items() if cost.category == category]


def is_tail_operator(op_id: int) -> bool:
    """Check if an operator is a tail safety operator."""
    cost = OPERATOR_COSTS.get(op_id)
    return cost.is_tail if cost else False


def is_global_op(op_id: int) -> bool:
    """Check if operator scales with L (total nodes) rather than K."""
    return get_operator_cost(op_id).ticks_per_total_node > 0


# Back-compat alias (tables-side name)
is_global_operator = is_global_op


# =============================================================================
# OPERATOR CATEGORIES (for degradation logic)
# =============================================================================

# Operators by category for degradation ordering
INJECT_OPS = frozenset(get_operators_by_category(OpCategory.INJECT))
PERMUTE_OPS = frozenset(get_operators_by_category(OpCategory.PERMUTE))
LOWRANK_OPS = frozenset(get_operators_by_category(OpCategory.LOWRANK))
DIAG_OPS = frozenset(get_operators_by_category(OpCategory.DIAG))
BANDED_OPS = frozenset(get_operators_by_category(OpCategory.BANDED))

# Tail-capable op ids. Renamed from the former module-level TAIL_OPS set to
# avoid colliding with tables/tail_modes.TAIL_OPS (Dict[TailMode, List[int]]).
TAIL_CAPABLE_OPS = frozenset(get_tail_operators())


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
        elif op_id in TAIL_CAPABLE_OPS:
            result['tail'].append(op_id)
        else:
            result['backbone'].append(op_id)

    return result


__all__ = [
    'COEFF_LEN', 'NODE_LEN', 'L_TOTAL_NODES',
    'OpCategory', 'OperatorCost',
    'COST_TABLE_VERSION', 'OPERATOR_COSTS',
    'get_operator_cost', 'get_tail_operators', 'get_nonlinear_operators',
    'get_operators_by_category', 'is_global_op', 'is_global_operator',
    'is_tail_operator', 'predict_ticks', 'categorize_operators',
    'INJECT_OPS', 'PERMUTE_OPS', 'LOWRANK_OPS', 'DIAG_OPS', 'BANDED_OPS',
    'TAIL_CAPABLE_OPS',
]
