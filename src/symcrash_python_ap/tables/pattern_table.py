"""
Pattern Table (Canonical Table v0)
==================================

The PatternTable maps pattern_id to leaf_mask_u32, determining which
nodes are active for coefficient processing.

Node model (from CMMM-2):
- COEFF_LEN = 256
- NODE_LEN = 8
- Total nodes = 32 (fits in uint32 bitmask)
- Node n maps to coefficients [n*8, n*8+8)

Pattern categories:
- FULL patterns: All 32 nodes active
- BAND patterns: Specific bands enabled
- SPARSE patterns: Subset of nodes for degradation
- DEGRADED patterns: K-staircase (32->24->16->12->8->4->1)

Band layout (from BAND_SLICES):
- Band 0: nodes 0     (1 node)   - Fundamental
- Band 1: nodes 1-2   (2 nodes)  - Body
- Band 2: nodes 3-6   (4 nodes)  - Mid
- Band 3: nodes 7-12  (6 nodes)  - Presence
- Band 4: nodes 13-20 (8 nodes)  - Brilliance
- Band 5: nodes 21-31 (11 nodes) - Air
"""

from dataclasses import dataclass
from typing import Dict, List


# Constants
COEFF_LEN = 256
NODE_LEN = 8
NUM_NODES = COEFF_LEN // NODE_LEN  # 32

# Band to node mapping
BAND_NODE_RANGES = [
    (0, 1),    # Band 0: 1 node
    (1, 3),    # Band 1: 2 nodes
    (3, 7),    # Band 2: 4 nodes
    (7, 13),   # Band 3: 6 nodes
    (13, 21),  # Band 4: 8 nodes
    (21, 32),  # Band 5: 11 nodes
]


def popcount32(x: int) -> int:
    """Count set bits in 32-bit mask."""
    return (x & 0xFFFFFFFF).bit_count()


def nodes_to_mask(nodes: List[int]) -> int:
    """Convert list of node indices to bitmask."""
    mask = 0
    for n in nodes:
        if 0 <= n < 32:
            mask |= (1 << n)
    return mask


def band_to_mask(band_id: int) -> int:
    """Get mask for a single band."""
    if band_id < 0 or band_id >= len(BAND_NODE_RANGES):
        return 0
    start, end = BAND_NODE_RANGES[band_id]
    mask = 0
    for n in range(start, end):
        mask |= (1 << n)
    return mask


def bands_to_mask(band_ids: List[int]) -> int:
    """Get mask for multiple bands."""
    mask = 0
    for band_id in band_ids:
        mask |= band_to_mask(band_id)
    return mask


def first_k_nodes_mask(k: int) -> int:
    """Get mask for first K nodes (in band priority order)."""
    if k <= 0:
        return 0
    if k >= 32:
        return 0xFFFFFFFF

    # Priority: lower bands first (fundamental → air)
    mask = 0
    count = 0
    for band_id in range(len(BAND_NODE_RANGES)):
        start, end = BAND_NODE_RANGES[band_id]
        for n in range(start, end):
            mask |= (1 << n)
            count += 1
            if count >= k:
                return mask
    return mask


# =============================================================================
# CANONICAL PATTERN TABLE (v0)
# =============================================================================

@dataclass(frozen=True)
class PatternEntry:
    """Pattern table entry."""
    pattern_id: int
    name: str
    mask: int
    k_count: int  # Derived from popcount


# Pattern IDs
PATTERN_FULL = 0
PATTERN_K24 = 1
PATTERN_K16 = 2
PATTERN_K12 = 3
PATTERN_K8 = 4
PATTERN_K4 = 5
PATTERN_K1 = 6
PATTERN_BANDS_0_1_2 = 10  # Fundamental + body + mid
PATTERN_BANDS_0_1 = 11    # Fundamental + body
PATTERN_BAND_0 = 12       # Fundamental only


# Build canonical patterns. K-staircase masks come from the single data
# source primitives.K_LADDER_MASKS (WSE-STD-DEGRADE-v0 §3.1 frozen mapping;
# identical to first_k_nodes_mask(k) because band node ranges are contiguous).
from ..primitives import K_LADDER_MASKS as _K_LADDER_MASKS

_FULL_MASK = _K_LADDER_MASKS[PATTERN_FULL]
_K24_MASK = _K_LADDER_MASKS[PATTERN_K24]
_K16_MASK = _K_LADDER_MASKS[PATTERN_K16]
_K12_MASK = _K_LADDER_MASKS[PATTERN_K12]
_K8_MASK = _K_LADDER_MASKS[PATTERN_K8]
_K4_MASK = _K_LADDER_MASKS[PATTERN_K4]
_K1_MASK = _K_LADDER_MASKS[PATTERN_K1]
_BANDS_0_1_2_MASK = bands_to_mask([0, 1, 2])
_BANDS_0_1_MASK = bands_to_mask([0, 1])
_BAND_0_MASK = band_to_mask(0)


PATTERN_TABLE: Dict[int, PatternEntry] = {
    PATTERN_FULL: PatternEntry(PATTERN_FULL, "FULL", _FULL_MASK, 32),
    PATTERN_K24: PatternEntry(PATTERN_K24, "K24", _K24_MASK, 24),
    PATTERN_K16: PatternEntry(PATTERN_K16, "K16", _K16_MASK, 16),
    PATTERN_K12: PatternEntry(PATTERN_K12, "K12", _K12_MASK, 12),
    PATTERN_K8: PatternEntry(PATTERN_K8, "K8", _K8_MASK, 8),
    PATTERN_K4: PatternEntry(PATTERN_K4, "K4", _K4_MASK, 4),
    PATTERN_K1: PatternEntry(PATTERN_K1, "K1", _K1_MASK, 1),
    PATTERN_BANDS_0_1_2: PatternEntry(PATTERN_BANDS_0_1_2, "BANDS_0_1_2", _BANDS_0_1_2_MASK, 7),
    PATTERN_BANDS_0_1: PatternEntry(PATTERN_BANDS_0_1, "BANDS_0_1", _BANDS_0_1_MASK, 3),
    PATTERN_BAND_0: PatternEntry(PATTERN_BAND_0, "BAND_0", _BAND_0_MASK, 1),
}


def get_node_mask(pattern_id: int) -> int:
    """Get node mask for a pattern ID. Returns FULL mask if unknown."""
    entry = PATTERN_TABLE.get(pattern_id)
    return entry.mask if entry else _FULL_MASK


def get_k_count(pattern_id: int) -> int:
    """Get active node count for a pattern ID."""
    entry = PATTERN_TABLE.get(pattern_id)
    return entry.k_count if entry else 32


# =============================================================================
# K DEGRADATION STAIRCASE
# =============================================================================

# Canonical K values for degradation (in order)
K_STAIRCASE = [32, 24, 16, 12, 8, 4, 1]

# Map K value to pattern ID
K_TO_PATTERN = {
    32: PATTERN_FULL,
    24: PATTERN_K24,
    16: PATTERN_K16,
    12: PATTERN_K12,
    8: PATTERN_K8,
    4: PATTERN_K4,
    1: PATTERN_K1,
}


def next_lower_k(k: int) -> int:
    """Get next lower K value in degradation staircase."""
    for step in K_STAIRCASE:
        if step < k:
            return step
    return 1


def pattern_for_k(k: int) -> int:
    """Get pattern ID for a given K value (or next lower)."""
    for step in K_STAIRCASE:
        if k >= step:
            return K_TO_PATTERN[step]
    return PATTERN_K1
