"""Test if per-k phase queries fix the error accumulation."""

import numpy as np
import sys
sys.path.insert(0, 'src')

# Clear cache
import symcrash_python_ap.vm.coeff_lut_streaming as lut_mod
lut_mod._NOTE_LUT_CACHE.clear()

from symcrash_python_ap.vm.coeff_lut_streaming import (
    get_lut_for_note, CCSState, BLOCK_SIZE, NUM_LEAVES, VM_K
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.wavelet.sine_core import sine_q15

SAMPLE_RATE = 48000
f0 = 440.0
K = 8

print("=== TESTING PER-K PHASE QUERIES ===")
print()

lut = get_lut_for_note(wave_id=0, midi_note=69)
print(f"LUT: P={lut.P}, K={lut.K}")

phase_inc = int((f0 / SAMPLE_RATE) * (1 << 32))
print(f"Phase inc per sample: {phase_inc}")
print()

# Ground truth
TARGET_RMS_Q15 = 16384
Q15_ONE = 32767

def generate_ground_truth(n_samples, start_phase=0):
    output = np.empty(n_samples, dtype=np.int16)
    phase = start_phase
    for n in range(n_samples):
        sample = sine_q15(phase)
        sample = (sample * TARGET_RMS_Q15) // Q15_ONE
        output[n] = np.clip(sample, -32768, 32767)
        phase = (phase + phase_inc) & 0xFFFFFFFF
    return output

# Test several blocks
n_blocks = 5
phase_32 = 0

for block_idx in range(n_blocks):
    print(f"--- Block {block_idx} ---")

    # METHOD 1: Current method (all k at same phase)
    subband_output_1 = np.zeros((NUM_LEAVES, K), dtype=np.int16)
    lut.get_subband_samples(phase_32, subband_output_1)

    # Run K IDWPTs and interleave
    ccs_audio_1 = np.zeros(BLOCK_SIZE, dtype=np.int16)
    for k in range(K):
        coeffs_k = [subband_output_1[l, k] for l in range(NUM_LEAVES)]
        audio_k = idwpt_jit(coeffs_k, headroom_bits=1)
        for m in range(BLOCK_SIZE // K):
            ccs_audio_1[k + m * K] = audio_k[m * K]

    # METHOD 2: Per-k phase queries
    subband_output_2 = np.zeros((NUM_LEAVES, K), dtype=np.int16)
    for k in range(K):
        # Query LUT at phase_32 + k*phase_inc for this k slot
        phase_k = (phase_32 + k * phase_inc) & 0xFFFFFFFF
        temp = np.zeros((NUM_LEAVES, K), dtype=np.int16)
        lut.get_subband_samples(phase_k, temp)
        # Use only the k column from the queried result (or k=0?)
        # Actually, we need to think about this more carefully...
        # The LUT[l, k, p] was built with block starting at phase_p + k*phase_inc
        # If we query at phase_k = phase_32 + k*phase_inc, we get coefficients
        # for blocks starting at phase_k + k'*phase_inc for each k'
        # We want the k'=0 slot, which corresponds to block at phase_k
        subband_output_2[:, k] = temp[:, 0]

    ccs_audio_2 = np.zeros(BLOCK_SIZE, dtype=np.int16)
    for k in range(K):
        coeffs_k = [subband_output_2[l, k] for l in range(NUM_LEAVES)]
        audio_k = idwpt_jit(coeffs_k, headroom_bits=1)
        for m in range(BLOCK_SIZE // K):
            ccs_audio_2[k + m * K] = audio_k[m * K]

    # Ground truth
    gt_block = generate_ground_truth(BLOCK_SIZE, phase_32)

    # Errors
    error_1 = np.abs(ccs_audio_1.astype(np.int32) - gt_block.astype(np.int32))
    error_2 = np.abs(ccs_audio_2.astype(np.int32) - gt_block.astype(np.int32))

    print(f"  Phase: {phase_32}")
    print(f"  Method 1 (same phase): max_err={error_1.max()}, mean_err={error_1.mean():.1f}")
    print(f"  Method 2 (per-k phase): max_err={error_2.max()}, mean_err={error_2.mean():.1f}")
    print()

    # Advance phase
    phase_32 = (phase_32 + phase_inc * BLOCK_SIZE) & 0xFFFFFFFF

print("=== INTERPRETATION ===")
print()
print("If Method 2 has lower error, the fix is to query LUT at")
print("phase + k*phase_inc for each k slot, using the k=0 column.")
