"""
FM6 Morph Engine - Continuous Topology Morphing
================================================

Per WSE-STD-FM-TOPOLOGY-v0 Section 8:
- Blends between two FM archetypes (A and B)
- Interpolates edge weights, carrier levels, and feedback
- CMMM can drive blend parameter for smooth transitions

This engine extends fm6_engine.py with topology morphing capability.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass, field

from .fm_archetypes import (
    ARCHETYPES, get_archetype, dx7_to_archetype,
    compute_blended_matrix, compute_blended_carriers, blend_feedback_op,
    ARCH_CHAIN, Q15_ONE
)
from .fm6_state import FM6State, FMOperator, Q32_MASK
from .tables import VM_K, COEFF_LEN

# Import sine table from SineCore
try:
    from ..wavelet.sine_core import sin_cos_q15
except ImportError:
    import math

    def sin_cos_q15(phase_q32: int) -> Tuple[int, int]:
        """Compute sin/cos from Q32 phase."""
        phase_rad = (phase_q32 / (1 << 32)) * 2 * math.pi
        sin_val = int(math.sin(phase_rad) * 32767)
        cos_val = int(math.cos(phase_rad) * 32767)
        return max(-32768, min(32767, sin_val)), max(-32768, min(32767, cos_val))


# Import from primitives (single source of truth)
from ..primitives import NUM_LEAVES


def _sat16(x: int) -> int:
    """Saturate to int16 range."""
    if x > 32767:
        return 32767
    elif x < -32768:
        return -32768
    return x


def _rshift_rne(val: int, shift: int) -> int:
    """Right shift with round-to-nearest-even."""
    if shift <= 0:
        return val
    half = 1 << (shift - 1)
    result = (val + half) >> shift
    if (val & ((1 << shift) - 1)) == half:
        result = result & ~1
    return result


def _rshift_rne_array(arr: np.ndarray, shift: int) -> np.ndarray:
    """Right shift with round-to-nearest-even (array version)."""
    if shift <= 0:
        return arr
    half = 1 << (shift - 1)
    return (arr + half) >> shift


@dataclass
class FM6MorphState:
    """
    Runtime state for morphing FM6 synthesis.

    Extends FM6State with topology morphing parameters.
    """
    # Base FM6 state
    operators: list = field(default_factory=lambda: [FMOperator(op_id=i) for i in range(6)])
    op_outputs: np.ndarray = field(default_factory=lambda: np.zeros((6, VM_K), dtype=np.int32))
    feedback_buffer: np.ndarray = field(default_factory=lambda: np.zeros(VM_K, dtype=np.int16))
    master_level_q15: int = Q15_ONE

    # Morphing state
    topo_a_id: int = ARCH_CHAIN      # Source archetype (0-7)
    topo_b_id: int = ARCH_CHAIN      # Target archetype (0-7)
    blend_q15: int = 0               # Blend amount (0=A, 32767=B)
    dx7_compat_id: int = 0           # DX7 compatibility (1-32, 0=disabled)

    # Cached blended matrix (recomputed when blend changes)
    _cached_blend: int = -1
    _cached_matrix: np.ndarray = None
    _cached_carrier_mask: int = 0
    _cached_carrier_fades: np.ndarray = None

    def __post_init__(self):
        """Initialize numpy arrays if needed."""
        if not isinstance(self.op_outputs, np.ndarray):
            self.op_outputs = np.zeros((6, VM_K), dtype=np.int32)
        if not isinstance(self.feedback_buffer, np.ndarray):
            self.feedback_buffer = np.zeros(VM_K, dtype=np.int16)

    def set_morph(self, topo_a: int, topo_b: int, blend: int):
        """Set morphing parameters."""
        self.topo_a_id = max(0, min(7, topo_a))
        self.topo_b_id = max(0, min(7, topo_b))
        self.blend_q15 = max(0, min(Q15_ONE, blend))
        # Invalidate cache
        self._cached_blend = -1

    def set_dx7_compat(self, alg_id: int):
        """Set DX7 compatibility mode."""
        if 1 <= alg_id <= 32:
            self.dx7_compat_id = alg_id
            # Map to archetype
            self.topo_a_id = dx7_to_archetype(alg_id)
            self.blend_q15 = 0  # Pure A topology
            self._cached_blend = -1

    def get_blended_matrix(self) -> np.ndarray:
        """Get or compute blended modulation matrix."""
        if self._cached_blend != self.blend_q15 or self._cached_matrix is None:
            self._cached_matrix = compute_blended_matrix(
                self.topo_a_id, self.topo_b_id, self.blend_q15
            )
            self._cached_carrier_mask, self._cached_carrier_fades = compute_blended_carriers(
                self.topo_a_id, self.topo_b_id, self.blend_q15
            )
            self._cached_blend = self.blend_q15
        return self._cached_matrix

    def get_carrier_info(self) -> Tuple[int, np.ndarray]:
        """Get blended carrier mask and fade levels."""
        if self._cached_blend != self.blend_q15:
            self.get_blended_matrix()  # Updates cache
        return self._cached_carrier_mask, self._cached_carrier_fades

    def reset(self):
        """Reset phase and outputs."""
        for op in self.operators:
            op.phase = 0
        self.op_outputs.fill(0)
        self.feedback_buffer.fill(0)


def fm6_generate_block_morphed(state: FM6MorphState, base_phase_inc: int) -> np.ndarray:
    """
    Generate one block of FM audio with topology morphing.

    Args:
        state: FM6MorphState with operators and morph parameters
        base_phase_inc: Q32 phase increment for fundamental frequency

    Returns:
        output_k: [K] lane-rate output samples (int16)
    """
    ops = state.operators

    # Get blended modulation matrix
    mod_matrix = state.get_blended_matrix()
    carrier_mask, carrier_fades = state.get_carrier_info()

    # Get feedback operators from both archetypes
    arch_a = get_archetype(state.topo_a_id)
    arch_b = get_archetype(state.topo_b_id)
    fb_op_a = arch_a.feedback_op
    fb_op_b = arch_b.feedback_op

    # Compute topological order for blended topology
    # For simplicity, use standard order [5,4,3,2,1,0] (modulators before carriers)
    eval_order = [5, 4, 3, 2, 1, 0]

    # Clear operator outputs
    state.op_outputs.fill(0)

    # Evaluate in topological order
    for op_idx in eval_order:
        op = ops[op_idx]

        # Compute operator phase increment
        op_phase_inc = (base_phase_inc * op.ratio_q16) >> 16

        # Sum modulation from all modulators using blended weights
        mod_sum = np.zeros(VM_K, dtype=np.int32)
        for src_idx in range(6):
            weight = mod_matrix[src_idx, op_idx]
            if weight > 0:
                # Apply weighted modulation
                mod_contrib = (state.op_outputs[src_idx].astype(np.int32) * weight) >> 15
                mod_sum += mod_contrib

        # Apply blended feedback
        if op_idx == fb_op_a or op_idx == fb_op_b:
            # Blend feedback from both topologies
            blended_fb = blend_feedback_op(
                fb_op_a, fb_op_b, state.blend_q15, state.op_outputs
            )

            # Determine feedback depth
            fb_depth = op.feedback_q15
            if arch_a.feedback_enhanced and state.blend_q15 < Q15_ONE // 2:
                fb_depth = min(fb_depth * 2, Q15_ONE)
            elif arch_b.feedback_enhanced and state.blend_q15 >= Q15_ONE // 2:
                fb_depth = min(fb_depth * 2, Q15_ONE)

            # Apply scaled feedback from previous block (z^-1 delay)
            if fb_depth > 0:
                fb_scaled = (state.feedback_buffer.astype(np.int32) * fb_depth) >> 15
                mod_sum += fb_scaled

        # Generate operator output for each k-slot
        for k in range(VM_K):
            # Advance phase
            op.phase = (op.phase + op_phase_inc) & Q32_MASK

            # Apply modulation to phase
            mod_val = int(mod_sum[k])
            mod_phase = (mod_val << 16) & Q32_MASK
            effective_phase = (op.phase + mod_phase) & Q32_MASK

            # Compute sine
            sin_val, _ = sin_cos_q15(effective_phase)

            # Apply envelope and level
            output = _rshift_rne(sin_val * op.level_q15, 15)
            output = _rshift_rne(output * op.env_value_q15, 15)

            state.op_outputs[op_idx, k] = _sat16(output)

    # Store blended feedback for next block
    state.feedback_buffer[:] = blend_feedback_op(
        fb_op_a, fb_op_b, state.blend_q15, state.op_outputs
    ).astype(np.int16)

    # Sum carrier outputs with fade levels
    output_k = np.zeros(VM_K, dtype=np.int32)
    num_active_carriers = 0

    for c in range(6):
        if carrier_mask & (1 << c):
            fade = carrier_fades[c]
            if fade > 0:
                # Apply fade to carrier output
                carrier_contrib = (state.op_outputs[c].astype(np.int32) * fade) >> 15
                output_k += carrier_contrib
                num_active_carriers += 1

    # Apply master level
    output_k = _rshift_rne_array(output_k * state.master_level_q15, 15)

    # Normalize by carrier count
    if num_active_carriers > 1:
        output_k = output_k // num_active_carriers

    return np.clip(output_k, -32768, 32767).astype(np.int16)


def fm6_morph_to_coefficients(output_k: np.ndarray) -> np.ndarray:
    """
    Convert lane-rate FM output to CCS coefficient format.

    Same as fm6_to_coefficients but for morph engine.
    """
    coeffs = np.zeros(COEFF_LEN, dtype=np.int16)

    for k in range(VM_K):
        for leaf in range(NUM_LEAVES):
            coeffs[leaf * VM_K + k] = output_k[k]

    return coeffs


def fm6_morph_generate_coeffs(state: FM6MorphState, output: np.ndarray,
                               base_phase_inc: int):
    """
    Generate morphed FM6 output directly into coefficient buffer.

    Args:
        state: FM6MorphState with operators and morph parameters
        output: [512] output coefficient array (modified in place)
        base_phase_inc: Q32 phase increment for fundamental
    """
    fm_output = fm6_generate_block_morphed(state, base_phase_inc)
    coeffs = fm6_morph_to_coefficients(fm_output)
    np.copyto(output[:COEFF_LEN], coeffs)


def fm6_morph_generate_audio(state: FM6MorphState, base_phase_inc: int,
                              num_blocks: int = 1) -> np.ndarray:
    """
    Generate morphed FM6 audio directly.

    Args:
        state: FM6MorphState with operators and morph parameters
        base_phase_inc: Q32 phase increment for fundamental
        num_blocks: Number of blocks to generate

    Returns:
        audio: [num_blocks * K] audio samples (int16)
    """
    audio_blocks = []

    for _ in range(num_blocks):
        block = fm6_generate_block_morphed(state, base_phase_inc)
        audio_blocks.append(block)

    return np.concatenate(audio_blocks)


def create_fm6_morph_state(topo_a: int = 0, topo_b: int = 0,
                           blend: int = 0) -> FM6MorphState:
    """
    Create FM6MorphState with specified morph parameters.

    Args:
        topo_a: Source archetype ID (0-7)
        topo_b: Target archetype ID (0-7)
        blend: Blend amount (0=A, 32767=B)

    Returns:
        FM6MorphState instance
    """
    state = FM6MorphState()
    state.set_morph(topo_a, topo_b, blend)

    # Initialize operator envelopes to full (for testing)
    for op in state.operators:
        op.env_value_q15 = Q15_ONE

    return state


__all__ = [
    'FM6MorphState',
    'fm6_generate_block_morphed',
    'fm6_morph_to_coefficients',
    'fm6_morph_generate_coeffs',
    'fm6_morph_generate_audio',
    'create_fm6_morph_state',
]
