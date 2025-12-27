"""
Operator Cost Table (WSE-STD-031)
=================================

Defines resource costs for all operators in integer tick units.
Costs distinguish K-scaling (active nodes) vs L-scaling (total nodes).

Cost model:
    ticks = base + (ticks_per_active_node × K) + (ticks_per_total_node × L)

Where:
    K = active nodes (can be reduced by degradation)
    L = total nodes (constant 32 for COEFF_LEN=256, NODE_LEN=8)

Tick unit: 1 tick ≈ 1 Q15 multiply-add on 1 coefficient.

Scaling categories:
    - DIAG/BANDED: Scale with K (cheap to budget down)
    - PERMUTE/LOWRANK: Scale with L (global touch, expensive)
    - PROJ: Mostly L-scaling (global measurement)
    - INJECT: Scale with K (sparse writes)
"""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import IntEnum


# =============================================================================
# CONSTANTS (from operator_library)
# =============================================================================

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


# =============================================================================
# OPERATOR COST DATACLASS
# =============================================================================

@dataclass(frozen=True)
class OperatorCost:
    """
    Resource cost for a single operator.

    All costs in integer tick units.
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


# =============================================================================
# OPERATOR COST TABLE
# =============================================================================

# Version for compatibility checking
COST_TABLE_VERSION = "v0.1.0"

# Cost constants derived from operator complexity
# NODE_LEN = 8 coeffs per node
# DIAG: ~1 mul per coeff → NODE_LEN ticks/node
# BANDED (tridiag): ~3 mul per coeff → 3*NODE_LEN ticks/node
# PERMUTE: copy/reindex → NODE_LEN ticks per TOTAL node
# LOWRANK: dense global → 4*NODE_LEN per total node (rank=2 starter)

OPERATOR_COSTS: Dict[int, OperatorCost] = {

    # -------------------------------------------------------------------------
    # DIAG Operators (K-scaling)
    # -------------------------------------------------------------------------

    0: OperatorCost(
        op_id=0,
        op_name="Identity",
        category=OpCategory.DIAG,
        ticks_base=0,
        ticks_per_active_node=0,
        ticks_per_total_node=0,
        nonlinear=False,
        is_tail=False,
        drop_priority=0,  # Never drop (no-op anyway)
    ),

    1: OperatorCost(
        op_id=1,
        op_name="Tilt",
        category=OpCategory.DIAG,
        ticks_base=8,
        ticks_per_active_node=8,  # NODE_LEN muls
        ticks_per_total_node=0,
        nonlinear=False,
        is_tail=False,
        drop_priority=80,  # Cheap color, moderate priority
    ),

    15: OperatorCost(
        op_id=15,
        op_name="PressureRamp",
        category=OpCategory.DIAG,
        ticks_base=8,
        ticks_per_active_node=8,
        ticks_per_total_node=0,
        nonlinear=False,
        is_tail=False,
        drop_priority=160,  # Catastrophe - artistic, not essential
    ),

    16: OperatorCost(
        op_id=16,
        op_name="GateMask",
        category=OpCategory.DIAG,
        ticks_base=8,
        ticks_per_active_node=8,
        ticks_per_total_node=0,
        nonlinear=False,
        is_tail=False,
        drop_priority=40,  # Structure - important
    ),

    20: OperatorCost(
        op_id=20,
        op_name="FoldPressure",
        category=OpCategory.DIAG,
        ticks_base=8,
        ticks_per_active_node=8,
        ticks_per_total_node=0,
        nonlinear=False,
        is_tail=False,
        drop_priority=160,
    ),

    22: OperatorCost(
        op_id=22,
        op_name="SwallowtailMask",
        category=OpCategory.DIAG,
        ticks_base=8,
        ticks_per_active_node=8,
        ticks_per_total_node=0,
        nonlinear=False,
        is_tail=False,
        drop_priority=180,
    ),

    # -------------------------------------------------------------------------
    # BANDED Operators (K-scaling, local kernels)
    # -------------------------------------------------------------------------

    3: OperatorCost(
        op_id=3,
        op_name="IntraMix",
        category=OpCategory.BANDED,
        ticks_base=16,
        ticks_per_active_node=24,  # 3 kernel taps × NODE_LEN
        ticks_per_total_node=0,
        nonlinear=False,
        is_tail=False,
        drop_priority=110,  # Local coherence
    ),

    4: OperatorCost(
        op_id=4,
        op_name="DiffuseUp",
        category=OpCategory.BANDED,
        ticks_base=16,
        ticks_per_active_node=16,  # 2 ops × NODE_LEN
        ticks_per_total_node=0,
        nonlinear=False,
        is_tail=False,
        drop_priority=120,  # Transport
    ),

    5: OperatorCost(
        op_id=5,
        op_name="DiffuseDown",
        category=OpCategory.BANDED,
        ticks_base=16,
        ticks_per_active_node=16,
        ticks_per_total_node=0,
        nonlinear=False,
        is_tail=False,
        drop_priority=120,
    ),

    # -------------------------------------------------------------------------
    # PERMUTE Operators (L-scaling, global touch)
    # -------------------------------------------------------------------------

    6: OperatorCost(
        op_id=6,
        op_name="BandShear",
        category=OpCategory.PERMUTE,
        ticks_base=32,
        ticks_per_active_node=0,
        ticks_per_total_node=8,  # Must copy all coeffs
        nonlinear=False,
        is_tail=False,
        drop_priority=200,  # Global - drop early
    ),

    9: OperatorCost(
        op_id=9,
        op_name="SwapBands",
        category=OpCategory.PERMUTE,
        ticks_base=32,
        ticks_per_active_node=0,
        ticks_per_total_node=8,
        nonlinear=False,
        is_tail=False,
        drop_priority=220,  # Dramatic - drop early
    ),

    10: OperatorCost(
        op_id=10,
        op_name="RotateNodes",
        category=OpCategory.PERMUTE,
        ticks_base=32,
        ticks_per_active_node=0,
        ticks_per_total_node=8,
        nonlinear=False,
        is_tail=False,
        drop_priority=190,
    ),

    17: OperatorCost(
        op_id=17,
        op_name="QuasiRotate",
        category=OpCategory.PERMUTE,
        ticks_base=32,
        ticks_per_active_node=0,
        ticks_per_total_node=8,
        nonlinear=False,
        is_tail=False,
        drop_priority=210,
    ),

    # -------------------------------------------------------------------------
    # LOWRANK Operators (L-scaling, global dense)
    # -------------------------------------------------------------------------

    7: OperatorCost(
        op_id=7,
        op_name="LowRankMorph",
        category=OpCategory.LOWRANK,
        ticks_base=64,
        ticks_per_active_node=0,
        ticks_per_total_node=32,  # rank=2: 2×2×NODE_LEN
        nonlinear=False,
        is_tail=False,
        drop_priority=230,  # Expensive - drop early
    ),

    # -------------------------------------------------------------------------
    # INJECT Operators (K-scaling, sparse writes)
    # -------------------------------------------------------------------------

    8: OperatorCost(
        op_id=8,
        op_name="GlobalPulse",
        category=OpCategory.INJECT,
        ticks_base=16,
        ticks_per_active_node=8,
        ticks_per_total_node=0,
        nonlinear=False,
        is_tail=False,
        drop_priority=240,  # Events - drop first
    ),

    13: OperatorCost(
        op_id=13,
        op_name="Impulse",
        category=OpCategory.INJECT,
        ticks_base=8,
        ticks_per_active_node=8,
        ticks_per_total_node=0,
        nonlinear=False,
        is_tail=False,
        drop_priority=245,
    ),

    14: OperatorCost(
        op_id=14,
        op_name="NoiseBand",
        category=OpCategory.INJECT,
        ticks_base=16,
        ticks_per_active_node=8,
        ticks_per_total_node=0,
        nonlinear=False,
        is_tail=False,
        drop_priority=250,  # PRNG-ish
    ),

    21: OperatorCost(
        op_id=21,
        op_name="CuspInjection",
        category=OpCategory.INJECT,
        ticks_base=16,
        ticks_per_active_node=8,
        ticks_per_total_node=0,
        nonlinear=False,
        is_tail=False,
        drop_priority=248,
    ),

    # -------------------------------------------------------------------------
    # PROJ Operators (L-scaling, global measurement, TAIL candidates)
    # -------------------------------------------------------------------------

    2: OperatorCost(
        op_id=2,
        op_name="FineClamp",
        category=OpCategory.PROJ,
        ticks_base=64,
        ticks_per_active_node=0,
        ticks_per_total_node=16,  # Band energy caps
        nonlinear=False,
        is_tail=True,  # MANDATORY TAIL
        drop_priority=0,  # Never drop
    ),

    11: OperatorCost(
        op_id=11,
        op_name="NormalizeScale",
        category=OpCategory.PROJ,
        ticks_base=96,
        ticks_per_active_node=0,
        ticks_per_total_node=24,  # sqrt-ish
        nonlinear=True,  # Uses sqrt
        is_tail=True,  # TAIL
        drop_priority=0,  # Never drop (but can downgrade tail mode)
    ),

    12: OperatorCost(
        op_id=12,
        op_name="GluePair",
        category=OpCategory.PROJ,
        ticks_base=48,
        ticks_per_active_node=0,
        ticks_per_total_node=8,  # Only two bands
        nonlinear=False,
        is_tail=False,
        drop_priority=60,  # Tail-adjacent
    ),

    19: OperatorCost(
        op_id=19,
        op_name="EmergencySimplify",
        category=OpCategory.PROJ,
        ticks_base=64,
        ticks_per_active_node=0,
        ticks_per_total_node=16,
        nonlinear=False,
        is_tail=True,  # EMERGENCY TAIL
        drop_priority=0,  # Never drop
    ),

    23: OperatorCost(
        op_id=23,
        op_name="SheafProject",
        category=OpCategory.PROJ,
        ticks_base=96,
        ticks_per_active_node=0,
        ticks_per_total_node=24,
        nonlinear=False,
        is_tail=True,  # TAIL (optional in MIN mode)
        drop_priority=10,  # Very high priority (but can downgrade)
    ),

    # -------------------------------------------------------------------------
    # MIX Operators (inherits from mixed ops)
    # -------------------------------------------------------------------------

    18: OperatorCost(
        op_id=18,
        op_name="MorphMix",
        category=OpCategory.MIX,
        ticks_base=16,  # Overhead only; add cost(A) + cost(B) separately
        ticks_per_active_node=0,
        ticks_per_total_node=0,
        nonlinear=False,
        is_tail=False,
        drop_priority=140,
    ),
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


def get_tail_operators() -> list[int]:
    """Get list of operator IDs marked as tail."""
    return [op_id for op_id, cost in OPERATOR_COSTS.items() if cost.is_tail]


def get_nonlinear_operators() -> list[int]:
    """Get list of operator IDs that use nonlinear operations."""
    return [op_id for op_id, cost in OPERATOR_COSTS.items() if cost.nonlinear]


def get_operators_by_category(category: OpCategory) -> list[int]:
    """Get all operator IDs in a category."""
    return [op_id for op_id, cost in OPERATOR_COSTS.items() if cost.category == category]


def is_global_op(op_id: int) -> bool:
    """Check if operator scales with L (total nodes) rather than K."""
    cost = get_operator_cost(op_id)
    return cost.ticks_per_total_node > 0


__all__ = [
    'COEFF_LEN', 'NODE_LEN', 'L_TOTAL_NODES',
    'OpCategory', 'OperatorCost',
    'COST_TABLE_VERSION', 'OPERATOR_COSTS',
    'get_operator_cost', 'get_tail_operators', 'get_nonlinear_operators',
    'get_operators_by_category', 'is_global_op',
]
