"""
FM6 Engine - 6-Operator FM Synthesis
=====================================

Per WSE-STD-FM-TOPOLOGY-v0:
- Evaluates operators in topological order
- Applies modulation matrix (modulators before carriers)
- Sums carrier outputs
- Converts to coefficient format (512 values)

All arithmetic in Q15/Q32 fixed-point per WSE-STD-001.
"""

import numpy as np
from typing import Tuple

from .fm_algorithms import get_algorithm, FMAlgorithm
from .fm6_state import FM6State, Q15_ONE, Q16_ONE, Q32_MASK
from .tables import VM_K, COEFF_LEN

# Import sine table from SineCore
try:
    from ..wavelet.sine_core import sin_cos_q15
except ImportError:
    # Fallback implementation if sine_core not available
    import math

    def sin_cos_q15(phase_q32: int) -> Tuple[int, int]:
        """Compute sin/cos from Q32 phase."""
        phase_rad = (phase_q32 / (1 << 32)) * 2 * math.pi
        sin_val = int(math.sin(phase_rad) * 32767)
        cos_val = int(math.cos(phase_rad) * 32767)
        return max(-32768, min(32767, sin_val)), max(-32768, min(32767, cos_val))


# Constants - import from primitives (single source of truth)
from ..primitives import NUM_LEAVES
# COEFF_LEN imported from tables.py (64 leaves × VM_K k-slots)


def _sat16(x: int) -> int:
    """Saturate to int16 range."""
    if x > 32767:
        return 32767
    elif x < -32768:
        return -32768
    return x


def _rshift_rne(val: int, shift: int) -> int:
    """Right shift with round-to-nearest-even (scalar version)."""
    if shift <= 0:
        return val
    half = 1 << (shift - 1)
    result = (val + half) >> shift
    # Round to even on tie
    if (val & ((1 << shift) - 1)) == half:
        result = result & ~1
    return result


def _rshift_rne_array(arr: np.ndarray, shift: int) -> np.ndarray:
    """Right shift with round-to-nearest-even (array version)."""
    if shift <= 0:
        return arr
    half = 1 << (shift - 1)
    result = (arr + half) >> shift
    # Round to even on tie (simplified for arrays - skip for performance)
    return result


def fm6_generate_block(state: FM6State, base_phase_inc: int) -> np.ndarray:
    """
    Generate one block of FM audio at lane rate.

    Args:
        state: FM6State with operators and algorithm
        base_phase_inc: Q32 phase increment for fundamental frequency

    Returns:
        output_k: [K] lane-rate output samples (int16)
    """
    alg = get_algorithm(state.algorithm_id)
    ops = state.operators

    # Clear operator outputs
    state.op_outputs.fill(0)

    # Evaluate in topological order
    for op_idx in alg.eval_order:
        op = ops[op_idx]

        # Compute operator phase increment (includes ratio and detune)
        op_phase_inc = op.compute_phase_inc(base_phase_inc)

        # Sum modulation from all modulators
        mod_sum = np.zeros(VM_K, dtype=np.int32)
        for mod_idx in alg.get_modulators(op_idx):
            mod_sum += state.op_outputs[mod_idx]

        # Apply self-feedback (if this is the feedback operator)
        if alg.feedback_op == op_idx and op.feedback_q15 > 0:
            # Use previous block's output (z⁻¹ delay)
            feedback_scaled = (state.feedback_buffer * op.feedback_q15) >> 15
            mod_sum += feedback_scaled

        # Generate operator output for each k-slot
        for k in range(VM_K):
            # Advance phase
            op.phase = (op.phase + op_phase_inc) & Q32_MASK

            # Apply modulation to phase
            # Scale modulation: mod_sum is Q15, need to convert to phase offset
            # FM modulation index: mod_phase = modulator_output << 16 (approx)
            # Convert to Python int to avoid numpy overflow
            mod_val = int(mod_sum[k])
            mod_phase = (mod_val << 16) & Q32_MASK
            effective_phase = (op.phase + mod_phase) & Q32_MASK

            # Compute sine
            sin_val, _ = sin_cos_q15(effective_phase)

            # Apply envelope and level
            # output = sin * level * envelope (all Q15)
            output = _rshift_rne(sin_val * op.level_q15, 15)
            output = _rshift_rne(output * op.env_value_q15, 15)

            state.op_outputs[op_idx, k] = _sat16(output)

    # Store feedback for next block (z⁻¹ delay)
    if alg.feedback_op >= 0:
        state.feedback_buffer[:] = state.op_outputs[alg.feedback_op]

    # Sum carrier outputs
    output_k = np.zeros(VM_K, dtype=np.int32)
    for c in alg.get_carriers():
        output_k += state.op_outputs[c]

    # Apply master level and saturate
    output_k = _rshift_rne_array(output_k * state.master_level_q15, 15)

    # Count carriers for normalization
    num_carriers = len(alg.get_carriers())
    if num_carriers > 1:
        # Scale down to prevent clipping when summing carriers
        output_k = output_k // num_carriers

    return np.clip(output_k, -32768, 32767).astype(np.int16)


def fm6_generate_block_full(state: FM6State, base_phase_inc: int) -> np.ndarray:
    """
    Generate one block of FM audio at SAMPLE rate (64 samples).

    This generates FM audio directly without the lane-rate intermediary,
    producing a full 64-sample block suitable for direct output.

    Args:
        state: FM6State with operators and algorithm
        base_phase_inc: Q32 phase increment for fundamental frequency

    Returns:
        output: [64] audio samples (int16)
    """
    alg = get_algorithm(state.algorithm_id)
    ops = state.operators
    num_samples = 64

    # Temporary storage for operator outputs at sample rate
    op_outputs_sample = np.zeros((6, num_samples), dtype=np.int32)

    # Evaluate in topological order
    for op_idx in alg.eval_order:
        op = ops[op_idx]

        # Compute operator phase increment (includes ratio and detune)
        op_phase_inc = op.compute_phase_inc(base_phase_inc)

        # Generate operator output for each sample
        for s in range(num_samples):
            # Sum modulation from all modulators (sample-by-sample)
            mod_sum = 0
            for mod_idx in alg.get_modulators(op_idx):
                mod_sum += op_outputs_sample[mod_idx, s]

            # Apply self-feedback (use previous sample in block, or prev block for s=0)
            if alg.feedback_op == op_idx and op.feedback_q15 > 0:
                if s == 0:
                    # Use last sample from previous block
                    fb_val = int(state.feedback_buffer[VM_K - 1]) if len(state.feedback_buffer) > 0 else 0
                else:
                    # Use previous sample in this block
                    fb_val = int(op_outputs_sample[op_idx, s - 1])
                feedback_scaled = (fb_val * op.feedback_q15) >> 15
                mod_sum += feedback_scaled

            # Advance phase
            op.phase = (op.phase + op_phase_inc) & Q32_MASK

            # Apply modulation to phase
            mod_phase = (int(mod_sum) << 16) & Q32_MASK
            effective_phase = (op.phase + mod_phase) & Q32_MASK

            # Compute sine
            sin_val, _ = sin_cos_q15(effective_phase)

            # Apply envelope and level
            output = _rshift_rne(sin_val * op.level_q15, 15)
            output = _rshift_rne(output * op.env_value_q15, 15)

            op_outputs_sample[op_idx, s] = _sat16(output)

    # Store last few samples for next block's feedback
    # Use downsampled version for compatibility with existing feedback buffer
    for k in range(VM_K):
        state.feedback_buffer[k] = op_outputs_sample[alg.feedback_op, (k + 1) * (64 // VM_K) - 1] if alg.feedback_op >= 0 else 0

    # Sum carrier outputs
    output = np.zeros(num_samples, dtype=np.int32)
    carriers = alg.get_carriers()
    for c in carriers:
        output += op_outputs_sample[c]

    # Apply master level
    output = _rshift_rne_array(output * state.master_level_q15, 15)

    # Normalize by carrier count
    if len(carriers) > 1:
        output = output // len(carriers)

    return np.clip(output, -32768, 32767).astype(np.int16)


def fm6_to_coefficients(output_k: np.ndarray) -> np.ndarray:
    """
    Convert lane-rate FM output to CCS coefficient format.

    DEPRECATED: This function produces stair-stepped output because it
    replicates K values across all 64 leaves. Use fm6_generate_coeffs_native()
    instead for coefficient-native FM synthesis.

    Args:
        output_k: [K] lane-rate FM output samples

    Returns:
        coeffs: [512] coefficient array for CCS pipeline
    """
    coeffs = np.zeros(COEFF_LEN, dtype=np.int16)

    # Replicate each k-slot value across 64 leaves
    # Layout: coeffs[leaf * K + k] for each leaf, each k-slot
    for k in range(VM_K):
        for leaf in range(NUM_LEAVES):
            coeffs[leaf * VM_K + k] = output_k[k]

    return coeffs


def fm6_generate_coeffs(state: FM6State, output: np.ndarray, base_phase_inc: int):
    """
    Generate FM6 coefficients via FWPT gateway (sample-rate native FM).

    This is the FIRST-CLASS FM6 path:
    1. Generate 64 audio samples at sample rate (true DX7-style FM)
    2. Transform via FWPT to 64 wavelet coefficients
    3. Replicate across K time slots -> 512 CCS coefficients

    Sample-rate FM provides authentic DX7-style character:
    - Proper sample-by-sample feedback (not lane-rate)
    - Full modulation bandwidth (not decimated)
    - Classic FM timbre everyone expects

    The FWPT gateway converts to coefficient domain for CCS operator
    processing (TILT, COMPRESS, etc.) and IDWPT reconstruction.

    Cost analysis (per block):
    - FM synthesis: 64 samples × 6 ops × ~7 ops = ~2,688 ops
    - FWPT transform: 6 levels × 126 steps × 4 ops = ~504 ops
    - Replication: 512 writes = ~512 ops
    - Total: ~3,704 ops (vs ~22,400 for coefficient-native)
    - Savings: 6× faster than basis-frame approach

    Args:
        state: FM6State with operators and algorithm
        output: coefficient buffer [COEFF_LEN] to fill (int16 or int32)
        base_phase_inc: Q32 phase increment for fundamental frequency
    """
    from ..wavelet.fast_lifting import fwpt_jit

    # Step 1: Generate 64 audio samples at SAMPLE RATE
    # This gives authentic DX7-style FM with proper feedback
    audio_block = fm6_generate_block_full(state, base_phase_inc)

    # Step 2: Transform to coefficients via FWPT (analysis transform)
    # headroom_bits=1 prevents overflow during lifting steps
    coeffs_64 = fwpt_jit(audio_block.tolist(), headroom_bits=1)

    # Step 3: Replicate across K time slots for polyphase reconstruction
    # Layout: output[leaf * K + k] for each leaf and k-slot
    # All k-slots get same coefficient (FWPT captures full block spectrum)
    for k in range(VM_K):
        for leaf in range(NUM_LEAVES):
            output[leaf * VM_K + k] = int(np.clip(coeffs_64[leaf], -32768, 32767))


def fm6_generate_audio(state: FM6State, base_phase_inc: int, num_blocks: int = 1) -> np.ndarray:
    """
    Generate FM6 audio directly (bypassing coefficient path).

    Useful for testing and direct FM synthesis without wavelet processing.

    Args:
        state: FM6State with operators and algorithm
        base_phase_inc: Q32 phase increment for fundamental
        num_blocks: Number of blocks to generate

    Returns:
        audio: [num_blocks * K] audio samples (int16)
    """
    audio_blocks = []

    for _ in range(num_blocks):
        block = fm6_generate_block(state, base_phase_inc)
        audio_blocks.append(block)

    return np.concatenate(audio_blocks)


def fwpt_analyze_block(audio_block: np.ndarray, headroom_bits: int = 1) -> np.ndarray:
    """
    FWPT adapter: Convert FM6 audio block to CCS coefficients.

    This is the "on-demand adapter" for when you need to re-enter coefficient
    domain for effects processing. Only call this when coefficient-domain
    operators are actually needed.

    Pipeline: FM6 audio (64 samples) -> FWPT -> 64 coefficients

    Args:
        audio_block: int16 array of 64 audio samples
        headroom_bits: bits to shift down for headroom (default 1)

    Returns:
        coeffs: int16 array of 64 wavelet coefficients

    Cost: ~504 ops (6 levels × 126 steps × ~4 ops per step)
    """
    from ..wavelet.fast_lifting import fwpt_jit

    # Convert to list for fwpt_jit
    coeffs_list = fwpt_jit(audio_block.tolist(), headroom_bits=headroom_bits)

    return np.array(coeffs_list, dtype=np.int16)


def replicate_coeffs_to_polyphase(coeffs_64: np.ndarray, output: np.ndarray):
    """
    Replicate 64 coefficients across K time slots for CCS polyphase.

    This replicates each coefficient value across all K k-slots, which is
    appropriate for FM6 since the audio block represents a full time window.

    Args:
        coeffs_64: int16 array of 64 wavelet coefficients
        output: preallocated int32/int16 array of COEFF_LEN (64 * VM_K)

    Layout: output[leaf * K + k] = coeffs_64[leaf] for all k
    """
    for k in range(VM_K):
        for leaf in range(NUM_LEAVES):
            output[leaf * VM_K + k] = int(coeffs_64[leaf])


# =============================================================================
# Preset Configurations
# =============================================================================
#
# DEPRECATED: These configure_* functions are legacy convenience functions.
# New code should use instrument_ir_v0 YAML presets with FM6 nodes in L0_graph.
#
# Example YAML (preferred):
#   layers:
#     L0_graph:
#       nodes:
#         - id: "fm6"
#           type: "FM6"
#           params:
#             algorithm: 22
#             feedback: 4
#             op_ratios: [65536, 65602, ...]
#             op_levels: [32767, 24576, ...]
#
# The CapsuleCompiler extracts FM6 parameters from L0 nodes and populates
# VoiceParamID.FM6_* defaults. This integrates with the standard voice engine
# pipeline (VoiceFrameEmitter -> VoiceFrame -> Executor).
#
# These functions remain for backwards compatibility with render scripts.
# =============================================================================

def configure_electric_piano(state: FM6State):
    """
    Configure FM6 state for E.Piano sound (Algorithm 5).

    Classic DX7 E.Piano uses:
    - Algorithm 5 (dual stack)
    - Operators with specific ratios and levels
    """
    state.set_algorithm(5)

    # E.Piano ratios (simplified)
    state.operators[0].ratio_q16 = Q16_ONE           # Op1: 1.0
    state.operators[1].ratio_q16 = Q16_ONE * 14      # Op2: 14.0
    state.operators[2].ratio_q16 = Q16_ONE           # Op3: 1.0
    state.operators[3].ratio_q16 = Q16_ONE           # Op4: 1.0
    state.operators[4].ratio_q16 = Q16_ONE           # Op5: 1.0
    state.operators[5].ratio_q16 = Q16_ONE           # Op6: 1.0

    # E.Piano levels
    state.operators[0].level_q15 = Q15_ONE           # Carrier full
    state.operators[1].level_q15 = Q15_ONE // 4      # Modulator medium
    state.operators[2].level_q15 = Q15_ONE // 2
    state.operators[3].level_q15 = Q15_ONE // 2
    state.operators[4].level_q15 = Q15_ONE // 2
    state.operators[5].level_q15 = Q15_ONE // 8      # Feedback op low


def configure_brass(state: FM6State):
    """
    Configure FM6 state for Brass sound (Algorithm 22).

    Classic DX7 Brass uses Algorithm 22 (split to 3 carriers):
    - Op1 unmodulated anchor at 1:1
    - Op2/3/4 modulated carriers with slight detune for ensemble thickness
    - Op5 modulator at 1:1 creates 2nd harmonic interaction
    - Op6 modulator at 2:1 with feedback adds brass edge (3rd+ harmonics)
    """
    state.set_algorithm(22)

    # Brass ratios - slight carrier detune for thickness, 2:1 mod for edge
    state.operators[0].ratio_q16 = Q16_ONE           # Op1: 1.0 (anchor)
    state.operators[1].ratio_q16 = Q16_ONE + 66      # Op2: 1.001 (+1.7 cents)
    state.operators[2].ratio_q16 = Q16_ONE - 66      # Op3: 0.999 (-1.7 cents)
    state.operators[3].ratio_q16 = Q16_ONE + 132     # Op4: 1.002 (+3.4 cents)
    state.operators[4].ratio_q16 = Q16_ONE           # Op5: 1.0 (modulator)
    state.operators[5].ratio_q16 = Q16_ONE * 2       # Op6: 2.0 (modulator + FB)

    # Brass levels - balanced carriers, higher mod index for bite
    state.operators[0].level_q15 = Q15_ONE               # Op1: Full anchor
    state.operators[1].level_q15 = (Q15_ONE * 3) // 4    # Op2: 0.75 carrier
    state.operators[2].level_q15 = (Q15_ONE * 3) // 4    # Op3: 0.75 carrier
    state.operators[3].level_q15 = (Q15_ONE * 5) // 8    # Op4: 0.625 carrier
    state.operators[4].level_q15 = (Q15_ONE * 5) // 8    # Op5: 0.625 mod depth
    state.operators[5].level_q15 = Q15_ONE // 2          # Op6: 0.5 FB mod

    # Set feedback for brass edge
    state.operators[5].feedback_q15 = Q15_ONE // 8       # Light feedback


def configure_bell(state: FM6State):
    """
    Configure FM6 state for Bell sound (Algorithm 1).

    Classic DX7 Bell uses:
    - Algorithm 1 (6-op chain)
    - Non-integer ratios for inharmonic spectrum
    """
    state.set_algorithm(1)

    # Bell ratios (inharmonic for metallic timbre)
    state.operators[0].ratio_q16 = Q16_ONE           # Op1: 1.0 (carrier)
    state.operators[1].ratio_q16 = (Q16_ONE * 14) // 10  # Op2: 1.4
    state.operators[2].ratio_q16 = (Q16_ONE * 20) // 10  # Op3: 2.0
    state.operators[3].ratio_q16 = (Q16_ONE * 31) // 10  # Op4: 3.1
    state.operators[4].ratio_q16 = (Q16_ONE * 44) // 10  # Op5: 4.4
    state.operators[5].ratio_q16 = (Q16_ONE * 57) // 10  # Op6: 5.7

    # Bell levels (decay through chain)
    state.operators[0].level_q15 = Q15_ONE
    state.operators[1].level_q15 = Q15_ONE * 3 // 4
    state.operators[2].level_q15 = Q15_ONE // 2
    state.operators[3].level_q15 = Q15_ONE // 3
    state.operators[4].level_q15 = Q15_ONE // 4
    state.operators[5].level_q15 = Q15_ONE // 5


def configure_organ(state: FM6State):
    """
    Configure FM6 state for Organ sound (Algorithm 32).

    Algorithm 32 is pure additive (all carriers).
    """
    state.set_algorithm(32)

    # Organ ratios (harmonic series)
    state.operators[0].ratio_q16 = Q16_ONE           # Fundamental
    state.operators[1].ratio_q16 = Q16_ONE * 2       # 2nd harmonic
    state.operators[2].ratio_q16 = Q16_ONE * 3       # 3rd harmonic
    state.operators[3].ratio_q16 = Q16_ONE * 4       # 4th harmonic
    state.operators[4].ratio_q16 = Q16_ONE * 5       # 5th harmonic
    state.operators[5].ratio_q16 = Q16_ONE * 6       # 6th harmonic

    # Organ levels (all equal for fundamental organ)
    for op in state.operators:
        op.level_q15 = Q15_ONE // 2


def configure_synth_brass(state: FM6State):
    """
    Configure FM6 state for aggressive Synth Brass (Algorithm 16).

    Algorithm 16: 5 modulators all feeding into single carrier.
    Creates sawtooth-like spectrum with harmonic series modulation.
    """
    state.set_algorithm(16)

    # Harmonic series for sawtooth-like spectrum
    state.operators[0].ratio_q16 = Q16_ONE           # Op1: 1.0 (carrier)
    state.operators[1].ratio_q16 = Q16_ONE * 2       # Op2: 2.0
    state.operators[2].ratio_q16 = Q16_ONE * 3       # Op3: 3.0
    state.operators[3].ratio_q16 = Q16_ONE * 4       # Op4: 4.0
    state.operators[4].ratio_q16 = Q16_ONE * 5       # Op5: 5.0
    state.operators[5].ratio_q16 = Q16_ONE * 2       # Op6: 2.0 (w/feedback)

    # 1/n rolloff for harmonic levels, boosted feedback mod
    state.operators[0].level_q15 = Q15_ONE               # Carrier full
    state.operators[1].level_q15 = Q15_ONE // 2          # 1/2
    state.operators[2].level_q15 = Q15_ONE // 3          # 1/3
    state.operators[3].level_q15 = Q15_ONE // 4          # 1/4
    state.operators[4].level_q15 = Q15_ONE // 5          # 1/5
    state.operators[5].level_q15 = (Q15_ONE * 5) // 8    # 0.625 (feedback buzz)

    # Higher feedback for aggressive buzz
    state.operators[5].feedback_q15 = Q15_ONE // 6


def configure_jump_brass(state: FM6State):
    """
    Configure FM6 state for "Jump" style 80s brass (Algorithm 19/22).

    Warm, harmonic unison brass with heavy carrier detune.
    - Lower modulator ratios (1:1 instead of 1.5/2.0) for warmth
    - Reduced mod depth for less harsh harmonics
    - Strong fundamental from full-level carriers
    """
    state.set_algorithm(19)

    # Heavy detune for thickness, but 1:1 modulators for warmth
    # Q16: 65536 = 1.0x, each 38 units ≈ 1 cent
    state.operators[0].ratio_q16 = Q16_ONE               # Op1: 1.0 (anchor)
    state.operators[1].ratio_q16 = Q16_ONE + 570         # Op2: +15 cents
    state.operators[2].ratio_q16 = Q16_ONE - 570         # Op3: -15 cents
    state.operators[3].ratio_q16 = Q16_ONE + 950         # Op4: +25 cents
    state.operators[4].ratio_q16 = Q16_ONE               # Op5: 1.0 (warm mod - 2nd harmonic)
    state.operators[5].ratio_q16 = Q16_ONE               # Op6: 1.0 (warm mod - adds body)

    # Full carriers, REDUCED mod depth for warmth over brightness
    state.operators[0].level_q15 = Q15_ONE               # Full anchor
    state.operators[1].level_q15 = Q15_ONE               # Full +detune
    state.operators[2].level_q15 = Q15_ONE               # Full -detune
    state.operators[3].level_q15 = (Q15_ONE * 7) // 8    # 0.875 (wide detune)
    state.operators[4].level_q15 = Q15_ONE // 3          # 0.33 mod (subtle warmth)
    state.operators[5].level_q15 = Q15_ONE // 4          # 0.25 feedback (gentle)

    state.operators[5].feedback_q15 = Q15_ONE // 16      # Very light feedback


__all__ = [
    # Core synthesis
    'fm6_generate_block',
    'fm6_generate_block_full',
    'fm6_generate_audio',
    # FWPT adapter (on-demand coefficient conversion)
    'fwpt_analyze_block',
    'replicate_coeffs_to_polyphase',
    # Legacy coefficient path (deprecated)
    'fm6_to_coefficients',
    'fm6_generate_coeffs',
    # Preset configurations
    'configure_electric_piano',
    'configure_brass',
    'configure_bell',
    'configure_organ',
    'configure_synth_brass',
    'configure_jump_brass',
]
