"""
VM Tables (WSE-STD-VM-HYBRID v0)
================================

Immutable, versioned tables for the hybrid VM.
Must match Python + FPGA implementations.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple, Optional

# Import canonical TailMode and K ladder from primitives (single source of truth)
from ..primitives import TailMode, K_LADDER_MASKS, get_k_mask, BLOCK_SIZE

# Import TAIL_OPS from canonical source (tables/tail_modes.py)
from ..tables.tail_modes import TAIL_OPS as _TAIL_OPS_CANONICAL, get_tail_ops as _get_tail_ops_canonical

# =============================================================================
# CONSTANTS
# =============================================================================

# Polyphase configuration
# VM_K = number of polyphase slots (4 or 8)
# Both use 1-sample phase offsets and interleaved reconstruction:
#   - K=4: 4 IDWPTs, 256 coefficients, out[k::4] = idwpt(slot_k)[0::4]
#   - K=8: 8 IDWPTs, 512 coefficients, out[k::8] = idwpt(slot_k)[0::8]
VM_K = 4  # Change to 8 for higher quality, 4 for lower memory
COEFF_LEN = 64 * VM_K  # 256 for K=4, 512 for K=8
NODE_LEN = 8     # Coefficients per node (for operator compatibility)
L_TOTAL_NODES = 32  # Total nodes for operator system (legacy view)
BLOCK_SAMPLES = BLOCK_SIZE  # Samples per block (from primitives)

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
# TAIL MODE TABLE (re-exported from tables/tail_modes.py)
# =============================================================================
# TailMode enum imported from primitives.py (single source of truth)
# TAIL_OPS imported from tables/tail_modes.py (canonical source)

# Re-export for backward compatibility
TAIL_OPS = _TAIL_OPS_CANONICAL


def get_tail_ops(tail_mode: TailMode) -> List[int]:
    """Get operator IDs for a tail mode."""
    return _get_tail_ops_canonical(tail_mode)


# =============================================================================
# PATTERN TABLE (K-LADDER MASKS)
# =============================================================================

# K-ladder values (decreasing) - WSE-STD-DEGRADE-v0 §3 K_RUNG_LIST
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


# Pre-computed pattern table, derived from the single data source
# primitives.K_LADDER_MASKS (WSE-STD-DEGRADE-v0 §3.1 frozen mapping).
PATTERN_TABLE: Dict[int, PatternEntry] = {
    pattern_id: PatternEntry(pattern_id, bin(mask).count('1'), mask)
    for pattern_id, mask in K_LADDER_MASKS.items()
}


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
