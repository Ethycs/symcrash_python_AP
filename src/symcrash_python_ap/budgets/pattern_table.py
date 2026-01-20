"""
Pattern Table for Node Masks (WSE-STD-020 §3.3)
===============================================

Re-exports from tables/pattern_table.py (canonical source).

Provides additional budget-specific patterns and utilities for degradation
scenarios including extended patterns for artistic control.

Node layout (32 nodes total, low-bands-first priority):
    Band 0: nodes 0      (1 node,  8 coeffs)
    Band 1: nodes 1-2    (2 nodes, 16 coeffs)
    Band 2: nodes 3-6    (4 nodes, 32 coeffs)
    Band 3: nodes 7-12   (6 nodes, 48 coeffs)
    Band 4: nodes 13-20  (8 nodes, 64 coeffs)
    Band 5: nodes 21-31  (11 nodes, 88 coeffs)

K ladder (canonical degradation steps):
    K=32 → K=24 → K=16 → K=12 → K=8 → K=4 → K=1
"""

from typing import Dict, List

# Re-export canonical definitions from tables/pattern_table.py
from ..tables.pattern_table import (
    # Constants
    COEFF_LEN, NODE_LEN, NUM_NODES,
    BAND_NODE_RANGES,
    K_STAIRCASE, K_TO_PATTERN,
    # Pattern table entries
    PatternEntry, PATTERN_TABLE,
    # Pattern lookup functions
    get_node_mask, get_k_count,
    pattern_for_k, next_lower_k,
    # Mask utilities
    popcount32, nodes_to_mask, band_to_mask, bands_to_mask, first_k_nodes_mask,
    # Pattern ID constants
    PATTERN_FULL, PATTERN_K24, PATTERN_K16, PATTERN_K12, PATTERN_K8, PATTERN_K4, PATTERN_K1,
    PATTERN_BANDS_0_1_2, PATTERN_BANDS_0_1, PATTERN_BAND_0,
)


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Alias for backward compatibility
L_TOTAL_NODES = NUM_NODES  # 32

# K ladder as tuple (tables uses list)
K_LADDER = tuple(K_STAIRCASE)

# Band structure as coefficient ranges (for backward compatibility)
# BAND_NODE_RANGES is node-based: [(0, 1), (1, 3), ...]
# BAND_SLICES is coefficient-based: [(0, 8), (8, 24), ...]
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


# =============================================================================
# NODE PRIORITY ORDERING
# =============================================================================

def build_node_priority() -> List[int]:
    """
    Build canonical node priority ordering.

    Returns node indices in band order (low→high frequency).
    This is the "keep low bands first" ordering for degradation.
    """
    return list(range(L_TOTAL_NODES))


# Static priority ordering (sequential for contiguous bands)
NODE_PRIORITY = build_node_priority()


# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# =============================================================================

def build_mask_for_k(k: int) -> int:
    """Build a node mask with exactly K active nodes (low bands first)."""
    return first_k_nodes_mask(k)


def shrink_mask_to_k(mask: int, k_target: int) -> int:
    """Shrink an existing mask to have at most K active nodes."""
    if k_target <= 0:
        return 0

    current_k = popcount32(mask)
    if current_k <= k_target:
        return mask

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
    return nodes_to_mask(nodes)


def get_pattern_by_id(pattern_id: int) -> PatternEntry:
    """Get pattern entry by ID."""
    return PATTERN_TABLE.get(pattern_id, PATTERN_TABLE[PATTERN_FULL])


def get_pattern_for_k(k: int) -> PatternEntry:
    """Get the best pattern for a target K value."""
    pattern_id = pattern_for_k(k)
    return PATTERN_TABLE[pattern_id]


def get_next_lower_k(current_k: int) -> int:
    """Get the next lower K value in the degradation ladder."""
    return next_lower_k(current_k)


def node_mask_to_coeff_mask(node_mask: int) -> List[bool]:
    """Expand node mask to coefficient-level mask."""
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
    # Re-exported from tables/pattern_table.py
    'COEFF_LEN', 'NODE_LEN', 'L_TOTAL_NODES', 'NUM_NODES',
    'K_LADDER', 'K_STAIRCASE', 'K_TO_PATTERN',
    'PatternEntry', 'PATTERN_TABLE',
    'popcount32',
    'PATTERN_FULL', 'PATTERN_K24', 'PATTERN_K16', 'PATTERN_K12',
    'PATTERN_K8', 'PATTERN_K4', 'PATTERN_K1',
    'PATTERN_BANDS_0_1_2', 'PATTERN_BANDS_0_1', 'PATTERN_BAND_0',
    # Backward compatibility
    'BAND_SLICES', 'BAND_NODE_RANGES', 'NODES_PER_BAND',
    'NODE_PRIORITY', 'build_node_priority',
    'build_mask_for_k', 'shrink_mask_to_k',
    'mask_to_node_list', 'node_list_to_mask',
    'get_pattern_by_id', 'get_pattern_for_k', 'get_next_lower_k',
    'node_mask_to_coeff_mask', 'get_active_coeff_indices',
    # Mask utilities
    'get_node_mask', 'get_k_count', 'pattern_for_k', 'next_lower_k',
    'nodes_to_mask', 'band_to_mask', 'bands_to_mask', 'first_k_nodes_mask',
]
