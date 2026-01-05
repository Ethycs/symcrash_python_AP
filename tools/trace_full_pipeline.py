"""Trace full CCS pipeline to understand where spurs come from."""

import numpy as np
import sys
sys.path.insert(0, 'src')

# Clear cache
import symcrash_python_ap.vm.coeff_lut_streaming as lut_mod
lut_mod._NOTE_LUT_CACHE.clear()

from symcrash_python_ap.vm.coeff_lut_streaming import (
    get_lut_for_note, CCSState, ccs_generate_coeffs_polyphase,
    DEFAULT_K, DEFAULT_P, BLOCK_SIZE, NUM_LEAVES, VM_K
)
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.wavelet.sine_core import sine_q15

SAMPLE_RATE = 48000
f0 = 440.0
K = 8

print("=== FULL PIPELINE TRACE ===")
print()

# Get LUT
lut = get_lut_for_note(wave_id=0, midi_note=69)
print(f"LUT: P={lut.P}, K={lut.K}, shape={lut.data.shape}")

# Setup CCS state
ccs_state = CCSState()
ccs_state.note = 69
ccs_state.wave_id = 0
ccs_state.phase_inc = int((f0 / SAMPLE_RATE) * (1 << 32))
ccs_state.phase = 0

print(f"Phase inc: {ccs_state.phase_inc}")
print(f"Phase inc per block: {ccs_state.phase_inc * BLOCK_SIZE}")
print()

# Ground truth generation
TARGET_RMS_Q15 = 16384
Q15_ONE = 32767

def generate_ground_truth(n_samples):
    """Generate ground truth samples starting at phase=0."""
    output = np.empty(n_samples, dtype=np.int16)
    phase = 0
    phase_inc = int((f0 / SAMPLE_RATE) * (1 << 32))
    for n in range(n_samples):
        sample = sine_q15(phase)
        sample = (sample * TARGET_RMS_Q15) // Q15_ONE
        output[n] = np.clip(sample, -32768, 32767)
        phase = (phase + phase_inc) & 0xFFFFFFFF
    return output

# Generate several blocks of ground truth
n_blocks = 5
ground_truth = generate_ground_truth(n_blocks * BLOCK_SIZE)

print(f"=== COMPARING BLOCK BY BLOCK ===")
print()

for block_idx in range(n_blocks):
    print(f"--- Block {block_idx} ---")

    # Get CCS output for this block
    carrier_512 = np.zeros(512, dtype=np.int16)
    phase_before = ccs_state.phase
    ccs_generate_coeffs_polyphase(ccs_state, carrier_512)

    # Run K IDWPTs
    idwpt_outputs = []
    for k in range(K):
        coeffs_k = [carrier_512[l * K + k] for l in range(NUM_LEAVES)]
        audio_k = idwpt_jit(coeffs_k, headroom_bits=1)
        idwpt_outputs.append(audio_k)

    # Interleave (correct rule)
    ccs_audio = np.zeros(BLOCK_SIZE, dtype=np.int16)
    for k in range(K):
        y_k = idwpt_outputs[k]
        for m in range(BLOCK_SIZE // K):
            ccs_audio[k + m * K] = y_k[m * K]

    # Ground truth for this block
    gt_block = ground_truth[block_idx * BLOCK_SIZE : (block_idx + 1) * BLOCK_SIZE]

    # Error analysis
    error = np.abs(ccs_audio.astype(np.int32) - gt_block.astype(np.int32))
    max_err = error.max()
    mean_err = error.mean()

    phase_idx = (phase_before >> 24) & 0xFF  # For P=256

    print(f"  Phase: {phase_before}, phase_idx: {phase_idx}")
    print(f"  CCS samples 0-7: {list(ccs_audio[:8])}")
    print(f"  GT samples 0-7:  {list(gt_block[:8])}")
    print(f"  Max error: {max_err}, Mean error: {mean_err:.1f}")
    print()

    # Advance state
    ccs_state.advance_block()

print("=== KEY OBSERVATION ===")
print()
print("Each block, the CCS output should match ground truth samples")
print("starting at the equivalent sample position in the continuous waveform.")
print()
print("If there's error, it means the LUT coefficients or interleave")
print("don't correctly reconstruct the expected waveform samples.")
