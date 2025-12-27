"""
VM Tables (WSE-STD-VM-HYBRID v0)
================================

Immutable, versioned tables for the hybrid VM.
Must match Python + FPGA implementations.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple, Optional

# =============================================================================
# CONSTANTS
# =============================================================================

COEFF_LEN = 256  # Number of coefficients (can be 64 or 256)
NODE_LEN = 8     # Coefficients per node
L_TOTAL_NODES = 32  # Total nodes (always 32)
BLOCK_SAMPLES = 64  # Samples per block

TABLE_VERSION = "0.1.0"


# =============================================================================
# LANE ENUM
# =============================================================================

class Lane(IntEnum):
    """Fixed execution lanes in the hybrid VM."""
    CARRIER = 1
    INJECT = 2
    DIAG = 3
    BANDED = 4
    PERMUTE = 5
    LOWRANK = 6
    PROJ = 7      # Projection (used in backbone, not tail lane)
    TAIL = 8      # Tail-only lane
    IDWPT = 9
    MIX_GAIN = 10


# =============================================================================
# OPERATOR LANE TABLE
# =============================================================================

@dataclass(frozen=True)
class OperatorLane:
    """Lane assignment and properties for an operator."""
    op_id: int
    op_name: str
    lane: Lane
    k_scaling: bool      # True = K-scaling, False = L-scaling
    nonlinear: bool
    is_tail: bool
    drop_priority: int   # 0 = never drop, higher = drop first


# Full operator lane assignment table (WSE-STD-VM-HYBRID Section 5)
OPERATOR_LANES: Dict[int, OperatorLane] = {
    0: OperatorLane(0, "Identity", Lane.DIAG, False, False, False, 0),
    1: OperatorLane(1, "Tilt", Lane.DIAG, True, False, False, 80),
    2: OperatorLane(2, "FineClamp", Lane.PROJ, False, False, True, 0),
    3: OperatorLane(3, "IntraMix", Lane.BANDED, True, False, False, 110),
    4: OperatorLane(4, "DiffuseUp", Lane.BANDED, True, False, False, 120),
    5: OperatorLane(5, "DiffuseDown", Lane.BANDED, True, False, False, 120),
    6: OperatorLane(6, "BandShear", Lane.PERMUTE, False, False, False, 140),
    7: OperatorLane(7, "LowRankMorph", Lane.LOWRANK, False, False, False, 200),
    8: OperatorLane(8, "GlobalPulse", Lane.INJECT, True, False, False, 220),
    9: OperatorLane(9, "SwapBands", Lane.PERMUTE, False, False, False, 150),
    10: OperatorLane(10, "RotateNodes", Lane.PERMUTE, False, False, False, 130),
    11: OperatorLane(11, "NormalizeScale", Lane.PROJ, False, True, True, 0),
    12: OperatorLane(12, "GluePair", Lane.PROJ, False, False, False, 60),
    13: OperatorLane(13, "Impulse", Lane.INJECT, True, False, False, 230),
    14: OperatorLane(14, "NoiseBand", Lane.INJECT, True, False, False, 240),
    15: OperatorLane(15, "PressureRamp", Lane.DIAG, True, False, False, 160),
    16: OperatorLane(16, "GateMask", Lane.DIAG, True, False, False, 40),
    17: OperatorLane(17, "QuasiRotate", Lane.PERMUTE, False, False, False, 180),
    18: OperatorLane(18, "MorphMix", Lane.DIAG, False, False, False, 100),  # MIX uses DIAG lane
    19: OperatorLane(19, "EmergencySimplify", Lane.PROJ, False, False, True, 0),
    20: OperatorLane(20, "FoldPressure", Lane.DIAG, True, False, False, 160),
    21: OperatorLane(21, "CuspInjection", Lane.INJECT, True, False, False, 248),
    22: OperatorLane(22, "SwallowtailMask", Lane.DIAG, True, False, False, 180),
    23: OperatorLane(23, "SheafProject", Lane.PROJ, False, False, True, 10),
}


def get_operator_lane(op_id: int) -> OperatorLane:
    """Get lane assignment for an operator ID."""
    if op_id in OPERATOR_LANES:
        return OPERATOR_LANES[op_id]
    # Default for unknown operators
    return OperatorLane(op_id, f"Unknown_{op_id}", Lane.DIAG, False, False, False, 255)


# =============================================================================
# TAIL MODE TABLE
# =============================================================================

class TailMode(IntEnum):
    """Tail mode selection."""
    FULL = 0       # [2, 11, 23] - FineClamp, NormalizeScale, SheafProject
    MID = 1        # [2, 11] - FineClamp, NormalizeScale
    MIN = 2        # [2] - FineClamp only
    EMERGENCY = 3  # [19] - EmergencySimplify only


TAIL_OPS: Dict[TailMode, List[int]] = {
    TailMode.FULL: [2, 11, 23],
    TailMode.MID: [2, 11],
    TailMode.MIN: [2],
    TailMode.EMERGENCY: [19],
}


def get_tail_ops(tail_mode: TailMode) -> List[int]:
    """Get operator IDs for a tail mode."""
    return TAIL_OPS.get(tail_mode, [2])


# =============================================================================
# PATTERN TABLE (K-LADDER MASKS)
# =============================================================================

# K-ladder values (decreasing)
K_LADDER = [32, 24, 16, 12, 8, 4, 1]

# Node priority order (low-bands first)
NODE_PRIORITY = [
    0, 1, 2, 3, 4, 5, 6, 7,       # Band 0-1 (coarse)
    8, 9, 10, 11, 12, 13, 14, 15,  # Band 2-3 (mid)
    16, 17, 18, 19, 20, 21, 22, 23,  # Band 4 (fine)
    24, 25, 26, 27, 28, 29, 30, 31,  # Band 5 (ultrafine)
]


@dataclass(frozen=True)
class PatternEntry:
    """Pattern table entry for K-ladder."""
    pattern_id: int
    k_value: int
    mask_u32: int


# Pre-computed pattern table
PATTERN_TABLE: Dict[int, PatternEntry] = {}


def _build_pattern_table():
    """Build pattern table from K-ladder."""
    global PATTERN_TABLE
    for idx, k in enumerate(K_LADDER):
        mask = 0
        for i in range(k):
            node = NODE_PRIORITY[i]
            mask |= (1 << node)
        PATTERN_TABLE[idx] = PatternEntry(idx, k, mask)


_build_pattern_table()


def get_pattern_by_id(pattern_id: int) -> PatternEntry:
    """Get pattern entry by ID."""
    return PATTERN_TABLE.get(pattern_id, PATTERN_TABLE[0])


def get_pattern_for_k(k: int) -> PatternEntry:
    """Get pattern entry for a given K value."""
    for entry in PATTERN_TABLE.values():
        if entry.k_value == k:
            return entry
    # Return closest K
    for entry in PATTERN_TABLE.values():
        if entry.k_value <= k:
            return entry
    return PATTERN_TABLE[len(PATTERN_TABLE) - 1]


def popcount32(x: int) -> int:
    """Count set bits in a 32-bit integer."""
    x = x & 0xFFFFFFFF
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    return x & 0x3F


# =============================================================================
# TABLES CONTAINER
# =============================================================================

@dataclass
class Tables:
    """
    Container for all VM tables.

    All tables are versioned and immutable.
    """
    version: str = TABLE_VERSION

    def get_operator_lane(self, op_id: int) -> OperatorLane:
        """Get lane assignment for an operator."""
        return get_operator_lane(op_id)

    def get_tail_ops(self, tail_mode: TailMode) -> List[int]:
        """Get operator IDs for a tail mode."""
        return get_tail_ops(tail_mode)

    def get_pattern(self, pattern_id: int) -> PatternEntry:
        """Get pattern entry by ID."""
        return get_pattern_by_id(pattern_id)

    def get_pattern_for_k(self, k: int) -> PatternEntry:
        """Get pattern entry for a given K value."""
        return get_pattern_for_k(k)


# Default tables instance
DEFAULT_TABLES = Tables()


__all__ = [
    # Constants
    'COEFF_LEN', 'NODE_LEN', 'L_TOTAL_NODES', 'BLOCK_SAMPLES', 'TABLE_VERSION',

    # Enums
    'Lane', 'TailMode',

    # Lane table
    'OperatorLane', 'OPERATOR_LANES', 'get_operator_lane',

    # Tail table
    'TAIL_OPS', 'get_tail_ops',

    # Pattern table
    'K_LADDER', 'NODE_PRIORITY', 'PatternEntry', 'PATTERN_TABLE',
    'get_pattern_by_id', 'get_pattern_for_k', 'popcount32',

    # Tables container
    'Tables', 'DEFAULT_TABLES',
]
