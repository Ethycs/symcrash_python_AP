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

from typing import List, Dict, Tuple, Optional
import numpy as np

from .fixed import (
    sat16, mul_q15_sat, shift_round, shift_round_sat16,
    Q15_ONE, Q15_SHIFT, Q15_HALF
)

# Import tree definitions and shared types from wavelet_trees package
from .wavelet_trees import (
    BLOCK_SIZE, MAX_DEPTH, LEAF_COUNT, COEFF_LEN, NODE_SIZE, NODE_COUNT,
    PacketNode, TreeROM, enumerate_leaves,
    build_tree_0_balanced,
    build_tree_1_harmonic,
    build_tree_2_texture,
    build_tree_3_transient,
    build_tree_4_drone,
)


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
                   complex_override: Optional[List[bool]] = None,
                   phase_state: Optional[List[float]] = None) -> List[int]:
        """
        Wavelet packet synthesis using frequency-encoded coefficient modulation.

        Each leaf represents a frequency band. The 8 coefficients within each
        node encode amplitudes at 8 different frequencies spread across the band.
        The coefficient INDEX encodes WHERE within the band the frequency is,
        enabling proper harmonic synthesis.

        For a 48kHz sample rate with 64-sample blocks:
        - Nyquist = 24kHz
        - Leaf 0 covers 0-750 Hz (8 sub-bands of ~93.75 Hz each)
        - Leaf 1 covers 750-1500 Hz
        - etc.

        The Baby Fourier / Harmonic Target Injector places energy in specific
        coefficient positions to encode actual harmonic frequencies, not just
        band center frequencies.

        Args:
            coeff_nodes: 32 nodes of 8 coefficients each
            complex_override: Optional complex leaf mask override
            phase_state: Optional phase accumulator state (256 values) for
                        continuous phase across blocks

        Returns:
            64-sample audio block
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

        # Step 2: Synthesize with frequency-encoded coefficients
        output = [0] * BLOCK_SIZE

        # Sample rate and Nyquist (assuming 48kHz)
        sample_rate = 48000.0
        nyquist = sample_rate / 2

        # Use uniform leaf distribution for frequency mapping
        # (matches harmonic_injector.py leaf mapping)
        band_width = nyquist / LEAF_COUNT

        for leaf_idx in range(LEAF_COUNT):
            node = processed_nodes[leaf_idx]

            # Skip if node is silent
            node_energy = sum(abs(c) for c in node)
            if node_energy < 10:
                continue

            # Leaf frequency range
            freq_low_hz = leaf_idx * band_width
            freq_high_hz = (leaf_idx + 1) * band_width
            freq_span = freq_high_hz - freq_low_hz

            # Generate each sub-frequency within the band
            for coeff_idx in range(NODE_SIZE):
                coeff_val = node[coeff_idx]
                if abs(coeff_val) < 10:
                    continue

                # Coefficient index encodes position within band
                # coeff 0 = low end of band, coeff NODE_SIZE-1 = high end
                sub_pos = (coeff_idx + 0.5) / NODE_SIZE
                sub_freq_hz = freq_low_hz + sub_pos * freq_span

                # Skip DC and very low frequencies
                if sub_freq_hz < 20:
                    continue

                # Angular frequency (radians per sample)
                omega = 2 * np.pi * sub_freq_hz / sample_rate

                # Get phase from state if available, otherwise use position-based phase
                phase_idx = leaf_idx * NODE_SIZE + coeff_idx
                if phase_state is not None:
                    phase = phase_state[phase_idx]
                else:
                    # Use coefficient position for initial phase spread
                    # Spread coeffs evenly across 2*pi and leaves across pi
                    phase = coeff_idx * (2 * np.pi / NODE_SIZE) + leaf_idx * (np.pi / LEAF_COUNT)

                # Generate samples for this sub-frequency
                for i in range(BLOCK_SIZE):
                    carrier = np.sin(omega * i + phase)
                    sample = int(coeff_val * carrier)
                    output[i] = sat16(output[i] + sample)

                # Update phase state for next block (advance by block length)
                # This is critical for phase continuity across blocks!
                if phase_state is not None:
                    # Advance phase by omega * BLOCK_SIZE
                    new_phase = phase + omega * BLOCK_SIZE
                    # Wrap to [-pi, pi] to prevent float drift
                    new_phase = np.fmod(new_phase + np.pi, 2 * np.pi) - np.pi
                    phase_state[phase_idx] = new_phase

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

        coeff_q15[256] -> 64 nodes of 4 coefficients each
        """
        # coeff_q15 is 256 coefficients = 64 nodes of 4 (with NODE_SIZE=4)
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
                   complex_override: Optional[List[bool]] = None,
                   phase_state: Optional[List[float]] = None) -> List[int]:
        """
        Full synthesis: coefficients -> audio block.

        Args:
            tree_id: Which tree structure to use
            coeff_q15: 256 Q15 coefficients
            complex_override: Optional complex leaf mask
            phase_state: Optional 256-element phase accumulator array.
                        If provided, phases are updated in-place for continuity
                        across blocks. Critical for pitched audio.

        Returns:
            64 Q15 audio samples
        """
        nodes = self.unpack_coeffs(tree_id, coeff_q15)
        synth = self.synthesizers.get(tree_id, self.synthesizers[0])
        return synth.synthesize(nodes, complex_override, phase_state)
    
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


