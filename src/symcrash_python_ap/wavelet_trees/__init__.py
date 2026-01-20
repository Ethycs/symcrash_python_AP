"""
Wavelet Packet Trees
====================

Individual tree definitions for CDF 5/3 wavelet packet synthesis.
Each tree_id has its own module.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple

from ..primitives import BLOCK_SIZE, NUM_LEAVES, COEFF_LEN

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================

MAX_DEPTH = 6            # J = 6
LEAF_COUNT = NUM_LEAVES  # L = 64 leaves per tree (D=6 master tree)
NODE_SIZE = COEFF_LEN // LEAF_COUNT  # Coefficients per node (512 / 64 = 8)
NODE_COUNT = LEAF_COUNT  # 64 nodes (one per leaf)


# =============================================================================
# TREE NODE REPRESENTATION
# =============================================================================

@dataclass
class PacketNode:
    """A node in the wavelet packet tree."""
    depth: int
    index: int  # k at this depth
    is_leaf: bool = False
    is_complex: bool = False

    @property
    def freq_interval(self) -> Tuple[float, float]:
        """Normalized frequency interval [low, high)."""
        low = self.index / (2 ** self.depth)
        high = (self.index + 1) / (2 ** self.depth)
        return (low, high)

    @property
    def freq_key(self) -> Tuple[int, int, int]:
        """
        Integer comparison key for frequency ordering.
        Returns (scaled_freq, depth, index) for deterministic sorting.
        """
        scaled = self.index << (MAX_DEPTH - self.depth)
        return (scaled, self.depth, self.index)

    def __lt__(self, other: 'PacketNode') -> bool:
        return self.freq_key < other.freq_key

    def __le__(self, other: 'PacketNode') -> bool:
        return self.freq_key <= other.freq_key

    def __eq__(self, other: 'PacketNode') -> bool:
        return self.freq_key == other.freq_key


@dataclass
class TreeROM:
    """ROM tables for a single tree (frozen for FPGA)."""
    tree_id: int
    name: str

    # Split mask: split_mask[d][k] = True if node (d,k) is split
    split_mask: Dict[int, Dict[int, bool]] = field(default_factory=dict)

    # Leaf list in frequency order: leaf_list[l] = (depth, k)
    leaf_list: List[Tuple[int, int]] = field(default_factory=list)

    # Complex leaf mask (default): complex_mask[l] = True if complex
    complex_mask: List[bool] = field(default_factory=list)

    # Convenience arrays
    leaf_depth: List[int] = field(default_factory=list)
    leaf_k: List[int] = field(default_factory=list)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def enumerate_leaves(split_mask: Dict[int, Dict[int, bool]]) -> List[PacketNode]:
    """Enumerate all leaves from a split mask."""
    leaves = []

    def recurse(d: int, k: int):
        if d >= MAX_DEPTH:
            leaves.append(PacketNode(depth=d, index=k, is_leaf=True))
            return

        if split_mask.get(d, {}).get(k, False):
            recurse(d + 1, 2 * k)
            recurse(d + 1, 2 * k + 1)
        else:
            leaves.append(PacketNode(depth=d, index=k, is_leaf=True))

    recurse(0, 0)
    return leaves


# =============================================================================
# TREE IMPORTS
# =============================================================================

from .tree_0_balanced import build_tree_0_balanced
from .tree_1_harmonic import build_tree_1_harmonic
from .tree_2_texture import build_tree_2_texture
from .tree_3_transient import build_tree_3_transient
from .tree_4_drone import build_tree_4_drone

# =============================================================================
# BASIS IMPORTS
# =============================================================================

from .equipment_basis import (
    EquipmentBasis,
    EquipmentSynthesizer,
    build_60hz_basis,
    build_50hz_basis,
    build_tube_basis,
    build_tape_basis,
    build_vinyl_basis,
    build_digital_basis,
    build_studio_chain_basis,
    make_clean_signal,
    make_tube_warmth,
    make_hum_60hz,
    make_tape_saturation,
    make_vinyl_warmth,
    MAINS_60HZ,
    MAINS_50HZ,
    SAMPLE_RATE,
    NYQUIST,
)

from .scale_synthesis import (
    Scale,
    ScaleSynthesizer,
    SCALES,
    JUST_RATIOS,
    compute_scale_frequencies,
    inject_scale_note,
    inject_chord,
    make_transpose_permutation,
    make_inversion_permutation,
    make_octave_fold_permutation,
)

__all__ = [
    # Constants
    'BLOCK_SIZE', 'MAX_DEPTH', 'LEAF_COUNT', 'COEFF_LEN', 'NODE_SIZE', 'NODE_COUNT',
    # Tree types
    'PacketNode', 'TreeROM', 'enumerate_leaves',
    # Tree builders
    'build_tree_0_balanced',
    'build_tree_1_harmonic',
    'build_tree_2_texture',
    'build_tree_3_transient',
    'build_tree_4_drone',
    # Equipment basis
    'EquipmentBasis', 'EquipmentSynthesizer',
    'build_60hz_basis', 'build_50hz_basis', 'build_tube_basis',
    'build_tape_basis', 'build_vinyl_basis', 'build_digital_basis',
    'build_studio_chain_basis',
    'make_clean_signal', 'make_tube_warmth', 'make_hum_60hz',
    'make_tape_saturation', 'make_vinyl_warmth',
    'MAINS_60HZ', 'MAINS_50HZ', 'SAMPLE_RATE', 'NYQUIST',
    # Scale synthesis
    'Scale', 'ScaleSynthesizer', 'SCALES', 'JUST_RATIOS',
    'compute_scale_frequencies', 'inject_scale_note', 'inject_chord',
    'make_transpose_permutation', 'make_inversion_permutation',
    'make_octave_fold_permutation',
]
