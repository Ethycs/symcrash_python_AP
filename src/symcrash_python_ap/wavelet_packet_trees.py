"""
Wavelet Packet Tree Specification (CDF 5/3)
===========================================

Full implementation of:
- CDF 5/3 lifting (bit-accurate)
- B = 64 samples, J = 6 levels, 32 leaves per tree
- 5 tree types (tree_id 0..4)
- Frequency-ordered leaf mapping
- Complex leaf atom transform support
- ROM table generation for FPGA

Frozen spec: 1 leaf ⇄ 1 coefficient node (8 coeffs)
"""

from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
import numpy as np

import sys
sys.path.insert(0, '/home/claude/synth_pipeline')

from fixed import (
    sat16, mul_q15_sat, shift_round, shift_round_sat16,
    Q15_ONE, Q15_SHIFT, Q15_HALF
)


# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================

BLOCK_SIZE = 64          # B = 64 samples
MAX_DEPTH = 6            # J = 6
LEAF_COUNT = 32          # L = 32 leaves per tree
COEFF_LEN = 256          # Total coefficient vector length
NODE_SIZE = 8            # Coefficients per node
NODE_COUNT = 32          # COEFF_LEN / NODE_SIZE


# =============================================================================
# CDF 5/3 LIFTING COEFFICIENTS (bit-accurate)
# =============================================================================

# CDF 5/3 lifting:
# Predict: d[n] = x[2n+1] - (x[2n] + x[2n+2]) / 2
# Update:  s[n] = x[2n] + (d[n-1] + d[n]) / 4

# In Q1.15:
CDF53_PREDICT_HALF = Q15_HALF      # 0.5 for predict averaging
CDF53_UPDATE_QUARTER = Q15_HALF >> 1  # 0.25 for update


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
        # k1 << d2 vs k2 << d1 comparison
        # Use max_depth for scaling
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
    
    # Leaf list in frequency order: leaf_list[ℓ] = (depth, k)
    leaf_list: List[Tuple[int, int]] = field(default_factory=list)
    
    # Complex leaf mask (default): complex_mask[ℓ] = True if complex
    complex_mask: List[bool] = field(default_factory=list)
    
    # Convenience arrays
    leaf_depth: List[int] = field(default_factory=list)
    leaf_k: List[int] = field(default_factory=list)


# =============================================================================
# TREE DEFINITIONS
# =============================================================================

def build_tree_0_balanced() -> TreeROM:
    """
    Tree 0 — Balanced Default (uniform resolution)
    All leaves at depth 5. Simple, predictable.
    """
    rom = TreeROM(tree_id=0, name="balanced")
    
    # Initialize split mask
    for d in range(MAX_DEPTH):
        rom.split_mask[d] = {}
        for k in range(2 ** d):
            # Split all nodes at d < 5
            rom.split_mask[d][k] = (d < 5)
    
    # Enumerate leaves (all at depth 5)
    leaves = []
    for k in range(32):
        leaves.append(PacketNode(depth=5, index=k, is_leaf=True))
    
    # Sort by frequency
    leaves.sort()
    
    # Store
    rom.leaf_list = [(n.depth, n.index) for n in leaves]
    rom.leaf_depth = [n.depth for n in leaves]
    rom.leaf_k = [n.index for n in leaves]
    
    # Complex mask: top half (N_top = 16)
    rom.complex_mask = [i >= 16 for i in range(32)]
    
    return rom


def build_tree_1_harmonic() -> TreeROM:
    """
    Tree 1 — Harmonic Anchor (wider lows, detailed highs)
    Cleaner pitch reading, less over-resolved lows.
    """
    rom = TreeROM(tree_id=1, name="harmonic")
    
    # Initialize all as not-split
    for d in range(MAX_DEPTH):
        rom.split_mask[d] = {k: False for k in range(2 ** d)}
    
    # Step 1: Split to depth 3 everywhere
    for d in range(3):
        for k in range(2 ** d):
            rom.split_mask[d][k] = True
    
    # Step 2: Low 2 regions (k=0,1 at d=3) split to depth 4 only
    for k in [0, 1]:
        rom.split_mask[3][k] = True
        # Children at d=4 are leaves (don't split)
    
    # Step 3: Remaining 6 regions (k=2..7 at d=3) split to depth 5
    for k in range(2, 8):
        rom.split_mask[3][k] = True
        # Their children at d=4
        for k4 in range(k * 2, k * 2 + 2):
            rom.split_mask[4][k4] = True
            # Children at d=5 are leaves by default
    
    # Step 4: Split 4 highest-freq depth-5 leaves to depth 6
    # Highest freq at d=5 are k=28,29,30,31
    for k5 in [28, 29, 30, 31]:
        rom.split_mask[5][k5] = True
    
    # Enumerate leaves
    leaves = _enumerate_leaves(rom.split_mask)
    leaves.sort()
    
    rom.leaf_list = [(n.depth, n.index) for n in leaves]
    rom.leaf_depth = [n.depth for n in leaves]
    rom.leaf_k = [n.index for n in leaves]
    
    # Complex mask: N_top = 12
    rom.complex_mask = [i >= 20 for i in range(32)]
    
    return rom


def build_tree_2_texture() -> TreeROM:
    """
    Tree 2 — Texture Microscope (high-band resolution)
    For texture/detail voices, noise-band, shatter.
    """
    rom = TreeROM(tree_id=2, name="texture")
    
    # Initialize
    for d in range(MAX_DEPTH):
        rom.split_mask[d] = {k: False for k in range(2 ** d)}
    
    # Step 1: Split to depth 4 everywhere
    for d in range(4):
        for k in range(2 ** d):
            rom.split_mask[d][k] = True
    
    # Step 2: Lower 4 depth-4 nodes are leaves, upper 12 split to depth 5
    for k in range(16):
        if k >= 4:
            rom.split_mask[4][k] = True
    
    # Step 3: Split 4 highest-freq depth-5 leaves to depth 6
    # After step 2, depth-5 nodes exist for k=8..31 (from parents k=4..15)
    # Highest freq: k=28,29,30,31 at d=5
    for k5 in [28, 29, 30, 31]:
        rom.split_mask[5][k5] = True
    
    leaves = _enumerate_leaves(rom.split_mask)
    leaves.sort()
    
    rom.leaf_list = [(n.depth, n.index) for n in leaves]
    rom.leaf_depth = [n.depth for n in leaves]
    rom.leaf_k = [n.index for n in leaves]
    
    # Complex mask: N_top = 20 (aggressive)
    rom.complex_mask = [i >= 12 for i in range(32)]
    
    return rom


def build_tree_3_transient() -> TreeROM:
    """
    Tree 3 — Transient / Impact (mid-band emphasis)
    Attacks bloom without brittle top fizz.
    """
    rom = TreeROM(tree_id=3, name="transient")
    
    # Initialize
    for d in range(MAX_DEPTH):
        rom.split_mask[d] = {k: False for k in range(2 ** d)}
    
    # Step 1: Split to depth 4 everywhere
    for d in range(4):
        for k in range(2 ** d):
            rom.split_mask[d][k] = True
    
    # Step 2: Middle 8 depth-4 nodes (k=4..11) split to depth 5
    for k in range(4, 12):
        rom.split_mask[4][k] = True
    
    # Step 3: Split 8 central depth-5 leaves to depth 6
    # Depth-5 nodes from parents k=4..11 are k=8..23
    # Central 8: k=12..19
    for k5 in range(12, 20):
        rom.split_mask[5][k5] = True
    
    leaves = _enumerate_leaves(rom.split_mask)
    leaves.sort()
    
    rom.leaf_list = [(n.depth, n.index) for n in leaves]
    rom.leaf_depth = [n.depth for n in leaves]
    rom.leaf_k = [n.index for n in leaves]
    
    # Complex mask: N_top = 12
    rom.complex_mask = [i >= 20 for i in range(32)]
    
    return rom


def build_tree_4_drone() -> TreeROM:
    """
    Tree 4 — Field / Drone (low-band detail, calm top)
    Slow violence, body motion.
    """
    rom = TreeROM(tree_id=4, name="drone")
    
    # Initialize
    for d in range(MAX_DEPTH):
        rom.split_mask[d] = {k: False for k in range(2 ** d)}
    
    # Step 1: Split to depth 4 everywhere
    for d in range(4):
        for k in range(2 ** d):
            rom.split_mask[d][k] = True
    
    # Step 2: Low 8 depth-4 nodes (k=0..7) split to depth 5
    for k in range(8):
        rom.split_mask[4][k] = True
    
    # Step 3: Split 8 depth-5 leaves in low region to depth 6
    # Depth-5 nodes from parents k=0..7 are k=0..15
    # Split all 8 lowest: k=0..7
    for k5 in range(8):
        rom.split_mask[5][k5] = True
    
    leaves = _enumerate_leaves(rom.split_mask)
    leaves.sort()
    
    rom.leaf_list = [(n.depth, n.index) for n in leaves]
    rom.leaf_depth = [n.depth for n in leaves]
    rom.leaf_k = [n.index for n in leaves]
    
    # Complex mask: N_top = 8 (calm top)
    rom.complex_mask = [i >= 24 for i in range(32)]
    
    return rom


def _enumerate_leaves(split_mask: Dict[int, Dict[int, bool]]) -> List[PacketNode]:
    """Enumerate all leaves from a split mask."""
    leaves = []
    
    def recurse(d: int, k: int):
        if d >= MAX_DEPTH:
            # Max depth reached, this is a leaf
            leaves.append(PacketNode(depth=d, index=k, is_leaf=True))
            return
        
        if split_mask.get(d, {}).get(k, False):
            # Node is split, recurse to children
            recurse(d + 1, 2 * k)
            recurse(d + 1, 2 * k + 1)
        else:
            # Node is a leaf
            leaves.append(PacketNode(depth=d, index=k, is_leaf=True))
    
    recurse(0, 0)
    return leaves


# =============================================================================
# CDF 5/3 LIFTING (BIT-ACCURATE)
# =============================================================================

class CDF53Lifting:
    """
    CDF 5/3 wavelet lifting implementation.
    
    Forward: signal -> (approx, detail)
    Inverse: (approx, detail) -> signal
    """
    
    @staticmethod
    def forward_step(x: List[int]) -> Tuple[List[int], List[int]]:
        """
        Forward CDF 5/3 lifting step.
        
        x[0..2n-1] -> s[0..n-1] (approx), d[0..n-1] (detail)
        """
        n = len(x) // 2
        if n == 0:
            return x, []
        
        s = [0] * n  # Approximation (even samples after update)
        d = [0] * n  # Detail (odd samples after predict)
        
        # Step 1: Predict (compute detail)
        # d[i] = x[2i+1] - (x[2i] + x[2i+2]) / 2
        for i in range(n):
            left = x[2 * i]
            right = x[2 * i + 2] if 2 * i + 2 < len(x) else x[2 * i]  # Symmetric extension
            
            # (left + right) / 2 with rounding
            avg = shift_round(left + right, 1)
            d[i] = sat16(x[2 * i + 1] - avg)
        
        # Step 2: Update (compute approximation)
        # s[i] = x[2i] + (d[i-1] + d[i]) / 4
        for i in range(n):
            d_left = d[i - 1] if i > 0 else d[0]  # Symmetric extension
            d_right = d[i]
            
            # (d_left + d_right) / 4 with rounding
            update = shift_round(d_left + d_right, 2)
            s[i] = sat16(x[2 * i] + update)
        
        return s, d
    
    @staticmethod
    def inverse_step(s: List[int], d: List[int]) -> List[int]:
        """
        Inverse CDF 5/3 lifting step.
        
        s[0..n-1], d[0..m-1] -> x[0..n+m-1]
        Handles case where s and d may have different lengths.
        """
        n_s = len(s)
        n_d = len(d)
        
        if n_s == 0:
            return []
        if n_d == 0:
            return s
        
        # Ensure same length for interleaving
        n = max(n_s, n_d)
        
        # Pad if necessary
        s_padded = s + [s[-1]] * (n - n_s) if n_s < n else s
        d_padded = d + [0] * (n - n_d) if n_d < n else d
        
        x = [0] * (2 * n)
        
        # Step 1: Undo update (recover even samples)
        # x[2i] = s[i] - (d[i-1] + d[i]) / 4
        for i in range(n):
            d_left = d_padded[i - 1] if i > 0 else d_padded[0]
            d_right = d_padded[i] if i < len(d_padded) else 0
            
            update = shift_round(d_left + d_right, 2)
            x[2 * i] = sat16(s_padded[i] - update)
        
        # Step 2: Undo predict (recover odd samples)
        # x[2i+1] = d[i] + (x[2i] + x[2i+2]) / 2
        for i in range(n):
            left = x[2 * i]
            right = x[2 * i + 2] if 2 * i + 2 < 2 * n else x[2 * i]
            
            avg = shift_round(left + right, 1)
            d_val = d_padded[i] if i < len(d_padded) else 0
            x[2 * i + 1] = sat16(d_val + avg)
        
        return x


# =============================================================================
# COMPLEX LEAF ATOM TRANSFORM
# =============================================================================

class ComplexAtomTransform:
    """
    8x8 "complex atom" transform for selected leaves.
    
    This is a simple orthogonal transform that adds phase information
    to the 8-coefficient node, making it behave like a complex-valued band.
    """
    
    # Precomputed 8x8 DCT-like matrix for complex leaf transform
    # This is a real-valued approximation that creates phase spread
    
    @staticmethod
    def _build_transform_matrix() -> np.ndarray:
        """Build 8x8 transform matrix (orthogonal)."""
        n = NODE_SIZE
        M = np.zeros((n, n))
        
        for k in range(n):
            for i in range(n):
                if k == 0:
                    M[k, i] = 1.0 / np.sqrt(n)
                else:
                    M[k, i] = np.sqrt(2.0 / n) * np.cos(np.pi * k * (2*i + 1) / (2*n))
        
        return M
    
    _TRANSFORM_Q15 = None
    _INVERSE_Q15 = None
    
    @classmethod
    def _ensure_matrices(cls):
        """Lazy initialization of transform matrices."""
        if cls._TRANSFORM_Q15 is None:
            M = cls._build_transform_matrix()
            cls._TRANSFORM_Q15 = (M * Q15_ONE).astype(np.int16)
            cls._INVERSE_Q15 = (M.T * Q15_ONE).astype(np.int16)
    
    @classmethod
    def forward(cls, node: List[int]) -> List[int]:
        """Forward complex atom transform."""
        cls._ensure_matrices()
        
        result = [0] * NODE_SIZE
        for k in range(NODE_SIZE):
            acc = 0
            for i in range(NODE_SIZE):
                acc += int(cls._TRANSFORM_Q15[k, i]) * node[i]
            result[k] = shift_round_sat16(acc, Q15_SHIFT)
        
        return result
    
    @classmethod
    def inverse(cls, node: List[int]) -> List[int]:
        """Inverse complex atom transform."""
        cls._ensure_matrices()
        
        result = [0] * NODE_SIZE
        for i in range(NODE_SIZE):
            acc = 0
            for k in range(NODE_SIZE):
                acc += int(cls._INVERSE_Q15[i, k]) * node[k]
            result[i] = shift_round_sat16(acc, Q15_SHIFT)
        
        return result


# =============================================================================
# WAVELET PACKET SYNTHESIS
# =============================================================================

class PacketSynthesizer:
    """
    Wavelet packet inverse synthesis using CDF 5/3.
    
    Takes coefficient nodes, applies tree structure, produces audio block.
    """
    
    def __init__(self, tree_rom: TreeROM):
        self.rom = tree_rom
        self.lifting = CDF53Lifting()
        
        # Precompute leaf -> sample count mapping
        # At depth d, a leaf covers 64/2^d samples (in time domain after full synthesis)
        # But at the coefficient level, each leaf has 8 coefficients
        self._build_synthesis_schedule()
    
    def _build_synthesis_schedule(self):
        """
        Build the schedule for inverse packet synthesis.
        
        We work bottom-up: for each non-leaf node that was split,
        we need to combine its children.
        """
        # Find all nodes that need combining (i.e., were split)
        self.combine_schedule = []  # List of (depth, k) to process
        
        for d in range(MAX_DEPTH - 1, -1, -1):
            for k in range(2 ** d):
                if self.rom.split_mask.get(d, {}).get(k, False):
                    self.combine_schedule.append((d, k))
    
    def synthesize(self, coeff_nodes: List[List[int]], 
                   complex_override: Optional[List[bool]] = None) -> List[int]:
        """
        Wavelet packet synthesis using frequency-band modulation.
        
        Each leaf represents a frequency band. The coefficients encode
        the amplitude in that band. We synthesize by generating carriers
        at the appropriate sample-rate frequencies.
        
        For a 48kHz sample rate with 64-sample blocks:
        - Nyquist = 24kHz
        - Leaf 0 (depth 5, k=0) covers 0-750 Hz
        - Leaf 1 covers 750-1500 Hz
        - etc.
        
        The carrier frequency is the center of each leaf's band.
        """
        # Use override or default complex mask
        complex_mask = complex_override if complex_override else self.rom.complex_mask
        
        # Step 1: Apply inverse complex atom transform where needed
        processed_nodes = []
        for leaf_idx in range(LEAF_COUNT):
            node = list(coeff_nodes[leaf_idx])
            if complex_mask[leaf_idx]:
                node = ComplexAtomTransform.inverse(node)
            processed_nodes.append(node)
        
        # Step 2: Synthesize each frequency band
        # Key insight: we need to generate carriers at SAMPLE RATE frequencies
        # not per-block frequencies
        
        output = [0] * BLOCK_SIZE
        
        # Sample rate and Nyquist (assuming 48kHz)
        sample_rate = 48000.0
        nyquist = sample_rate / 2
        
        for leaf_idx, (depth, k) in enumerate(self.rom.leaf_list):
            node = processed_nodes[leaf_idx]
            
            # Skip if node is silent
            node_energy = sum(abs(c) for c in node)
            if node_energy < 10:
                continue
            
            # Compute center frequency in Hz
            # At depth d, leaf k covers [k/2^d, (k+1)/2^d] of Nyquist
            freq_low_norm = k / (2 ** depth)
            freq_high_norm = (k + 1) / (2 ** depth)
            freq_center_norm = (freq_low_norm + freq_high_norm) / 2
            
            # Center frequency in Hz
            freq_hz = freq_center_norm * nyquist
            
            # Angular frequency for sample-rate synthesis
            omega = 2 * np.pi * freq_hz / sample_rate
            
            # Compute amplitude and phase from node coefficients
            # Use even/odd decomposition for amplitude/phase encoding
            c_even = sum(node[i] for i in range(0, NODE_SIZE, 2))
            c_odd = sum(node[i] for i in range(1, NODE_SIZE, 2))
            
            # RMS-like amplitude
            rms_amplitude = int(np.sqrt(sum(c*c for c in node) / NODE_SIZE))
            
            # Phase from even/odd balance
            phase_offset = np.arctan2(c_odd, c_even) if (c_even != 0 or c_odd != 0) else leaf_idx * np.pi / 8
            
            # Skip silent nodes
            if rms_amplitude < 10:
                continue
            
            for i in range(BLOCK_SIZE):
                # Generate sample-rate carrier
                carrier = np.sin(omega * i + phase_offset)
                
                # Modulated sample 
                sample = int(rms_amplitude * carrier)
                output[i] = sat16(output[i] + sample)
        
        return output
    
    def synthesize_full(self, coeff_nodes: List[List[int]],
                        complex_override: Optional[List[bool]] = None) -> List[int]:
        """
        Full wavelet packet synthesis with proper CDF 5/3 inverse.
        
        This is more complex but bit-accurate.
        """
        complex_mask = complex_override if complex_override else self.rom.complex_mask
        
        # Apply complex transform
        processed_nodes = []
        for leaf_idx in range(LEAF_COUNT):
            node = list(coeff_nodes[leaf_idx])
            if complex_mask[leaf_idx]:
                node = ComplexAtomTransform.inverse(node)
            processed_nodes.append(node)
        
        # Map leaves into tree structure
        # tree_data[d][k] = coefficient data at node (d, k)
        tree_data: Dict[int, Dict[int, List[int]]] = {d: {} for d in range(MAX_DEPTH + 1)}
        
        # Place processed leaves
        for leaf_idx, (d, k) in enumerate(self.rom.leaf_list):
            tree_data[d][k] = processed_nodes[leaf_idx]
        
        # Bottom-up synthesis
        for d, k in self.combine_schedule:
            left_k = 2 * k
            right_k = 2 * k + 1
            
            left_data = tree_data.get(d + 1, {}).get(left_k, [])
            right_data = tree_data.get(d + 1, {}).get(right_k, [])
            
            if left_data and right_data:
                # Pad to same length
                max_len = max(len(left_data), len(right_data))
                left_padded = left_data + [0] * (max_len - len(left_data))
                right_padded = right_data + [0] * (max_len - len(right_data))
                
                # Inverse lifting
                combined = self.lifting.inverse_step(left_padded, right_padded)
                tree_data[d][k] = combined
            elif left_data:
                tree_data[d][k] = left_data
            elif right_data:
                tree_data[d][k] = right_data
        
        # Get root
        result = tree_data.get(0, {}).get(0, [0] * BLOCK_SIZE)
        
        # Ensure correct length
        if len(result) < BLOCK_SIZE:
            result = result + [0] * (BLOCK_SIZE - len(result))
        
        return result[:BLOCK_SIZE]


# =============================================================================
# PACKET TREE SET
# =============================================================================

class PacketTreeSet:
    """
    Complete set of wavelet packet trees.
    
    Manages tree ROMs and synthesis for all tree_id values.
    """
    
    def __init__(self):
        # Build all trees
        self.trees: Dict[int, TreeROM] = {
            0: build_tree_0_balanced(),
            1: build_tree_1_harmonic(),
            2: build_tree_2_texture(),
            3: build_tree_3_transient(),
            4: build_tree_4_drone(),
        }
        
        # Create synthesizers
        self.synthesizers: Dict[int, PacketSynthesizer] = {
            tid: PacketSynthesizer(rom) for tid, rom in self.trees.items()
        }
        
        # Crossfade state for tree transitions
        self.xfade_blocks = 32  # Blocks for crossfade
    
    def get_tree(self, tree_id: int) -> TreeROM:
        """Get tree ROM by ID."""
        return self.trees.get(tree_id, self.trees[0])
    
    def unpack_coeffs(self, tree_id: int, coeff_q15: List[int]) -> List[List[int]]:
        """
        Unpack coefficient vector into nodes ordered by tree's leaf list.
        
        coeff_q15[256] -> 32 nodes of 8 coefficients each
        """
        # coeff_q15 is already 256 coefficients = 32 nodes of 8
        # leaf_list[ℓ] maps to node ℓ directly (per spec)
        nodes = []
        for leaf_idx in range(LEAF_COUNT):
            start = leaf_idx * NODE_SIZE
            end = start + NODE_SIZE
            
            if end <= len(coeff_q15):
                nodes.append(list(coeff_q15[start:end]))
            else:
                nodes.append([0] * NODE_SIZE)
        
        return nodes
    
    def synthesize(self, tree_id: int, coeff_q15: List[int],
                   complex_override: Optional[List[bool]] = None) -> List[int]:
        """
        Full synthesis: coefficients -> audio block.
        """
        nodes = self.unpack_coeffs(tree_id, coeff_q15)
        synth = self.synthesizers.get(tree_id, self.synthesizers[0])
        return synth.synthesize(nodes, complex_override)
    
    def synthesize_crossfade(self, old_tree_id: int, new_tree_id: int,
                             coeff_q15: List[int], alpha_q15: int) -> List[int]:
        """
        Crossfade synthesis between two trees.
        
        alpha_q15: 0 = all old, Q15_ONE = all new
        """
        x_old = self.synthesize(old_tree_id, coeff_q15)
        x_new = self.synthesize(new_tree_id, coeff_q15)
        
        alpha_inv = Q15_ONE - alpha_q15
        
        result = []
        for i in range(BLOCK_SIZE):
            mixed = (mul_q15_sat(x_old[i], alpha_inv) + 
                     mul_q15_sat(x_new[i], alpha_q15))
            result.append(sat16(mixed))
        
        return result


# =============================================================================
# ROM TABLE EXPORT
# =============================================================================

def export_rom_tables() -> Dict:
    """Export all tree ROM tables as Python/JSON-serializable dict."""
    tree_set = PacketTreeSet()
    
    export = {
        "constants": {
            "BLOCK_SIZE": BLOCK_SIZE,
            "MAX_DEPTH": MAX_DEPTH,
            "LEAF_COUNT": LEAF_COUNT,
            "COEFF_LEN": COEFF_LEN,
            "NODE_SIZE": NODE_SIZE,
        },
        "trees": {}
    }
    
    for tree_id, rom in tree_set.trees.items():
        # Flatten split mask for export
        split_flat = {}
        for d in range(MAX_DEPTH):
            for k in range(2 ** d):
                key = f"{d}_{k}"
                split_flat[key] = rom.split_mask.get(d, {}).get(k, False)
        
        export["trees"][tree_id] = {
            "name": rom.name,
            "split_mask": split_flat,
            "leaf_list": rom.leaf_list,
            "leaf_depth": rom.leaf_depth,
            "leaf_k": rom.leaf_k,
            "complex_mask": rom.complex_mask,
        }
    
    return export


def export_verilog_rom(tree_id: int) -> str:
    """Export tree ROM as Verilog case statements."""
    tree_set = PacketTreeSet()
    rom = tree_set.get_tree(tree_id)
    
    lines = [
        f"// Tree {tree_id}: {rom.name}",
        f"// Generated ROM tables",
        "",
        f"// Leaf list: leaf_idx -> (depth, k)",
        f"function [7:0] get_leaf_depth_{tree_id};",
        f"  input [4:0] leaf_idx;",
        f"  case (leaf_idx)",
    ]
    
    for i, (d, k) in enumerate(rom.leaf_list):
        lines.append(f"    5'd{i}: get_leaf_depth_{tree_id} = 8'd{d};")
    
    lines.extend([
        f"    default: get_leaf_depth_{tree_id} = 8'd5;",
        f"  endcase",
        f"endfunction",
        "",
        f"function [7:0] get_leaf_k_{tree_id};",
        f"  input [4:0] leaf_idx;",
        f"  case (leaf_idx)",
    ])
    
    for i, (d, k) in enumerate(rom.leaf_list):
        lines.append(f"    5'd{i}: get_leaf_k_{tree_id} = 8'd{k};")
    
    lines.extend([
        f"    default: get_leaf_k_{tree_id} = 8'd0;",
        f"  endcase",
        f"endfunction",
        "",
        f"// Complex mask: leaf_idx -> is_complex",
        f"function get_complex_mask_{tree_id};",
        f"  input [4:0] leaf_idx;",
        f"  case (leaf_idx)",
    ])
    
    for i, is_complex in enumerate(rom.complex_mask):
        lines.append(f"    5'd{i}: get_complex_mask_{tree_id} = 1'b{1 if is_complex else 0};")
    
    lines.extend([
        f"    default: get_complex_mask_{tree_id} = 1'b0;",
        f"  endcase",
        f"endfunction",
    ])
    
    return "\n".join(lines)


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_trees():
    """Verify all trees have exactly 32 leaves."""
    tree_set = PacketTreeSet()
    
    print("=" * 60)
    print("WAVELET PACKET TREE VERIFICATION")
    print("=" * 60)
    
    for tree_id, rom in tree_set.trees.items():
        n_leaves = len(rom.leaf_list)
        n_complex = sum(rom.complex_mask)
        
        depths = {}
        for d, k in rom.leaf_list:
            depths[d] = depths.get(d, 0) + 1
        
        print(f"\nTree {tree_id} ({rom.name}):")
        print(f"  Total leaves: {n_leaves} {'✓' if n_leaves == 32 else '✗'}")
        print(f"  Complex leaves: {n_complex}")
        print(f"  Depth distribution: {dict(sorted(depths.items()))}")
        
        # Verify frequency ordering
        is_sorted = all(
            PacketNode(rom.leaf_list[i][0], rom.leaf_list[i][1]) <= 
            PacketNode(rom.leaf_list[i+1][0], rom.leaf_list[i+1][1])
            for i in range(len(rom.leaf_list) - 1)
        )
        print(f"  Frequency ordered: {'✓' if is_sorted else '✗'}")


def test_synthesis():
    """Test synthesis pipeline."""
    print("\n" + "=" * 60)
    print("SYNTHESIS TEST")
    print("=" * 60)
    
    tree_set = PacketTreeSet()
    
    # Create test coefficients (simple impulse in coarse band)
    test_coeffs = [0] * COEFF_LEN
    test_coeffs[0] = 16384  # Impulse in first coefficient
    test_coeffs[8] = 8192   # Some energy in second node
    
    for tree_id in range(5):
        audio = tree_set.synthesize(tree_id, test_coeffs)
        
        peak = max(abs(x) for x in audio)
        energy = sum(x * x for x in audio) / BLOCK_SIZE
        
        print(f"Tree {tree_id}: peak={peak}, energy={energy:.0f}")


if __name__ == "__main__":
    verify_trees()
    test_synthesis()
    
    # Export ROM tables
    print("\n" + "=" * 60)
    print("ROM EXPORT")
    print("=" * 60)
    
    rom_data = export_rom_tables()
    print(f"Exported {len(rom_data['trees'])} trees")
    
    # Show sample Verilog
    print("\nSample Verilog for Tree 0:")
    print(export_verilog_rom(0)[:500] + "...")
