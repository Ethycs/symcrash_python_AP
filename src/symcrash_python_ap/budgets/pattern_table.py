"""
Pattern Table for Node Masks (WSE-STD-020 §3.3)
===============================================

Defines canonical node mask patterns for the K degradation ladder.
Pattern IDs map to leaf_mask_u32 values.

Node layout (32 nodes total, low-bands-first priority):
    Band 0: nodes 0      (1 node,  8 coeffs)
    Band 1: nodes 1-2    (2 nodes, 16 coeffs)
    Band 2: nodes 3-6    (4 nodes, 32 coeffs)
    Band 3: nodes 7-12   (6 nodes, 48 coeffs)
    Band 4: nodes 13-20  (8 nodes, 64 coeffs)
    Band 5: nodes 21-31  (11 nodes, 88 coeffs)

K ladder (canonical degradation steps):
    K=32 → K=24 → K=16 → K=12 → K=8 → K=4 → K=1

Priority ordering: Keep low-frequency nodes first (bands 0→5).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

# =============================================================================
# CONSTANTS
# =============================================================================

COEFF_LEN = 256
NODE_LEN = 8
L_TOTAL_NODES = 32  # COEFF_LEN // NODE_LEN

# Band structure (from operator_library)
BAND_SLICES = [
    (0, 8),      # Band 0: 8 coeffs, 1 node
    (8, 24),     # Band 1: 16 coeffs, 2 nodes
    (24, 56),    # Band 2: 32 coeffs, 4 nodes
    (56, 104),   # Band 3: 48 coeffs, 6 nodes
    (104, 168),  # Band 4: 64 coeffs, 8 nodes
    (168, 256),  # Band 5: 88 coeffs, 11 nodes
]

# Nodes per band
NODES_PER_BAND = [(end - start) // NODE_LEN for start, end in BAND_SLICES]
# = [1, 2, 4, 6, 8, 11]

# K ladder steps (canonical degradation)
K_LADDER = (32, 24, 16, 12, 8, 4, 1)


# =============================================================================
# NODE PRIORITY ORDERING
# =============================================================================

def build_node_priority() -> List[int]:
    """
    Build canonical node priority ordering.

    Returns node indices in band order (low→high frequency),
    and within each band from left to right.

    This is the "keep low bands first" ordering for degradation.
    """
    priority = []
    node_idx = 0
    for band_id, (start, end) in enumerate(BAND_SLICES):
        num_nodes = (end - start) // NODE_LEN
        for _ in range(num_nodes):
            priority.append(node_idx)
            node_idx += 1
    return priority


# Static priority ordering
NODE_PRIORITY = build_node_priority()
# = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
# (happens to be sequential because bands are contiguous)


# =============================================================================
# BITMASK UTILITIES
# =============================================================================

def popcount32(x: int) -> int:
    """Count set bits in a 32-bit value (FPGA-friendly)."""
    return (x & 0xFFFFFFFF).bit_count()


def build_mask_for_k(k: int) -> int:
    """
    Build a node mask with exactly K active nodes.

    Uses NODE_PRIORITY to select which nodes to keep (low bands first).

    Args:
        k: Number of active nodes (1-32)

    Returns:
        32-bit mask with K bits set
    """
    if k <= 0:
        return 0
    if k >= L_TOTAL_NODES:
        return (1 << L_TOTAL_NODES) - 1  # All ones

    mask = 0
    for i in range(min(k, len(NODE_PRIORITY))):
        node = NODE_PRIORITY[i]
        mask |= (1 << node)
    return mask


def shrink_mask_to_k(mask: int, k_target: int) -> int:
    """
    Shrink an existing mask to have at most K active nodes.

    Keeps nodes according to NODE_PRIORITY order.

    Args:
        mask: Current node mask
        k_target: Target number of active nodes

    Returns:
        New mask with at most k_target bits set
    """
    if k_target <= 0:
        return 0

    current_k = popcount32(mask)
    if current_k <= k_target:
        return mask  # Already at or below target

    # Build new mask keeping only first k_target nodes by priority
    out = 0
    count = 0
    for node in NODE_PRIORITY:
        if (mask >> node) & 1:
            out |= (1 << node)
            count += 1
            if count >= k_target:
                break
    return out


def mask_to_node_list(mask: int) -> List[int]:
    """Convert bitmask to list of active node indices."""
    nodes = []
    for i in range(L_TOTAL_NODES):
        if (mask >> i) & 1:
            nodes.append(i)
    return nodes


def node_list_to_mask(nodes: List[int]) -> int:
    """Convert list of node indices to bitmask."""
    mask = 0
    for node in nodes:
        if 0 <= node < L_TOTAL_NODES:
            mask |= (1 << node)
    return mask


# =============================================================================
# PATTERN TABLE ENTRIES
# =============================================================================

@dataclass(frozen=True)
class PatternEntry:
    """Pattern table entry for node mask configuration."""
    pattern_id: int
    name: str
    leaf_mask_u32: int
    k_value: int  # popcount of mask
    description: str


# Pre-computed patterns for K ladder
PATTERN_TABLE: Dict[int, PatternEntry] = {}

def _build_pattern_table():
    """Build the canonical pattern table for K ladder."""
    global PATTERN_TABLE

    pattern_id = 0

    # Full mask (K=32)
    PATTERN_TABLE[pattern_id] = PatternEntry(
        pattern_id=pattern_id,
        name="FULL_32",
        leaf_mask_u32=build_mask_for_k(32),
        k_value=32,
        description="All 32 nodes active",
    )
    pattern_id += 1

    # K ladder patterns
    for k in [24, 16, 12, 8, 4, 1]:
        PATTERN_TABLE[pattern_id] = PatternEntry(
            pattern_id=pattern_id,
            name=f"K_{k}",
            leaf_mask_u32=build_mask_for_k(k),
            k_value=k,
            description=f"Low-bands-first, {k} nodes",
        )
        pattern_id += 1

    # Band-specific patterns (for brightness control)
    # Bands 0-2 only (low frequency)
    low_freq_mask = 0
    for band_id in range(3):
        start, end = BAND_SLICES[band_id]
        for coeff_idx in range(start, end, NODE_LEN):
            node = coeff_idx // NODE_LEN
            low_freq_mask |= (1 << node)

    PATTERN_TABLE[pattern_id] = PatternEntry(
        pattern_id=pattern_id,
        name="LOW_FREQ",
        leaf_mask_u32=low_freq_mask,
        k_value=popcount32(low_freq_mask),
        description="Bands 0-2 only (7 nodes)",
    )
    pattern_id += 1

    # Bands 3-5 only (high frequency)
    high_freq_mask = 0
    for band_id in range(3, 6):
        start, end = BAND_SLICES[band_id]
        for coeff_idx in range(start, end, NODE_LEN):
            node = coeff_idx // NODE_LEN
            high_freq_mask |= (1 << node)

    PATTERN_TABLE[pattern_id] = PatternEntry(
        pattern_id=pattern_id,
        name="HIGH_FREQ",
        leaf_mask_u32=high_freq_mask,
        k_value=popcount32(high_freq_mask),
        description="Bands 3-5 only (25 nodes)",
    )

_build_pattern_table()


# =============================================================================
# PATTERN LOOKUP
# =============================================================================

def get_pattern_by_id(pattern_id: int) -> PatternEntry:
    """Get pattern entry by ID."""
    if pattern_id in PATTERN_TABLE:
        return PATTERN_TABLE[pattern_id]
    # Default to full mask
    return PATTERN_TABLE[0]


def get_pattern_for_k(k: int) -> PatternEntry:
    """
    Get the best pattern for a target K value.

    Returns the pattern with closest K <= target.
    """
    best = None
    for entry in PATTERN_TABLE.values():
        if entry.k_value <= k:
            if best is None or entry.k_value > best.k_value:
                best = entry
    return best if best else PATTERN_TABLE[0]


def get_next_lower_k(current_k: int) -> int:
    """
    Get the next lower K value in the degradation ladder.

    Args:
        current_k: Current K value

    Returns:
        Next lower K from ladder, or 1 if at minimum
    """
    for k in K_LADDER:
        if k < current_k:
            return k
    return 1


# =============================================================================
# COEFFICIENT MASK UTILITIES
# =============================================================================

def node_mask_to_coeff_mask(node_mask: int) -> List[bool]:
    """
    Expand node mask to coefficient-level mask.

    Args:
        node_mask: 32-bit node mask

    Returns:
        List of 256 booleans for coefficient activity
    """
    coeff_mask = [False] * COEFF_LEN
    for node in range(L_TOTAL_NODES):
        if (node_mask >> node) & 1:
            start = node * NODE_LEN
            for i in range(NODE_LEN):
                coeff_mask[start + i] = True
    return coeff_mask


def get_active_coeff_indices(node_mask: int) -> List[int]:
    """Get list of active coefficient indices from node mask."""
    indices = []
    for node in range(L_TOTAL_NODES):
        if (node_mask >> node) & 1:
            start = node * NODE_LEN
            indices.extend(range(start, start + NODE_LEN))
    return indices


__all__ = [
    'COEFF_LEN', 'NODE_LEN', 'L_TOTAL_NODES',
    'BAND_SLICES', 'NODES_PER_BAND', 'K_LADDER',
    'NODE_PRIORITY', 'build_node_priority',
    'popcount32', 'build_mask_for_k', 'shrink_mask_to_k',
    'mask_to_node_list', 'node_list_to_mask',
    'PatternEntry', 'PATTERN_TABLE',
    'get_pattern_by_id', 'get_pattern_for_k', 'get_next_lower_k',
    'node_mask_to_coeff_mask', 'get_active_coeff_indices',
]
