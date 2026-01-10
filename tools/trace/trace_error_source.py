"""Trace the source of the 2 LSB error in CCS."""

import numpy as np
import sys
sys.path.insert(0, 'src')

# Clear cache
import symcrash_python_ap.vm.coeff_lut_streaming as lut_mod
lut_mod._NOTE_LUT_CACHE.clear()

from symcrash_python_ap.vm.coeff_lut_streaming import (
    get_lut_for_note, BLOCK_SIZE, NUM_LEAVES, DEFAULT_K
)
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.wavelet.sine_core import sine_q15

SAMPLE_RATE = 48000
f0 = 440.0
K = 8

print("=== TRACING ERROR SOURCE ===")
print()

lut = get_lut_for_note(wave_id=0, midi_note=69)
phase_inc = int((f0 / SAMPLE_RATE) * (1 << 32))

TARGET_RMS_Q15 = 16384
Q15_ONE = 32767

def generate_block(start_phase, k_offset=0):
    block = np.empty(BLOCK_SIZE, dtype=np.int16)
    phase = (start_phase + k_offset * phase_inc) & 0xFFFFFFFF
    for n in range(BLOCK_SIZE):
        sample = sine_q15(phase)
        sample = (sample * TARGET_RMS_Q15) // Q15_ONE
        block[n] = np.clip(sample, -32768, 32767)
        phase = (phase + phase_inc) & 0xFFFFFFFF
    return block

# Test at a single block where we saw errors
# From earlier trace, block 3 (phase=3264175040) had mean error 2.2, max 7
test_phase = 3264175040
print(f"Testing at phase {test_phase}")
print()

# Method 1: Direct FWPT/IDWPT (should be perfect roundtrip)
print("=== METHOD 1: Direct FWPT/IDWPT ===")
direct_output = np.zeros(BLOCK_SIZE, dtype=np.int16)
for k in range(K):
    block = generate_block(test_phase, k_offset=k)
    coeffs = fwpt_jit(block.tolist(), headroom_bits=1)
    recon = idwpt_jit(coeffs, headroom_bits=1)
    for m in range(BLOCK_SIZE // K):
        direct_output[k + m * K] = recon[m * K]

gt_block = generate_block(test_phase, k_offset=0)
error_direct = np.abs(direct_output.astype(np.int32) - gt_block.astype(np.int32))
print(f"Direct FWPT/IDWPT error: max={error_direct.max()}, mean={error_direct.mean():.1f}")
print()

# Method 2: Using LUT (current method)
print("=== METHOD 2: LUT-based ===")
subband_output = np.zeros((NUM_LEAVES, K), dtype=np.int16)
lut.get_subband_samples(test_phase, subband_output)

lut_output = np.zeros(BLOCK_SIZE, dtype=np.int16)
for k in range(K):
    coeffs_k = [subband_output[l, k] for l in range(NUM_LEAVES)]
    audio_k = idwpt_jit(coeffs_k, headroom_bits=1)
    for m in range(BLOCK_SIZE // K):
        lut_output[k + m * K] = audio_k[m * K]

error_lut = np.abs(lut_output.astype(np.int32) - gt_block.astype(np.int32))
print(f"LUT-based error: max={error_lut.max()}, mean={error_lut.mean():.1f}")
print()

# Method 3: Coefficient comparison
print("=== METHOD 3: Coefficient Comparison ===")
print("Comparing LUT coefficients vs direct FWPT coefficients")
print()

total_coeff_error = 0
max_coeff_error = 0
for k in range(K):
    # Direct FWPT coefficients
    block = generate_block(test_phase, k_offset=k)
    direct_coeffs = fwpt_jit(block.tolist(), headroom_bits=1)

    # LUT coefficients
    lut_coeffs = [subband_output[l, k] for l in range(NUM_LEAVES)]

    # Compare
    for l in range(NUM_LEAVES):
        err = abs(direct_coeffs[l] - int(lut_coeffs[l]))
        total_coeff_error += err
        max_coeff_error = max(max_coeff_error, err)

    print(f"k={k}: max_coeff_err={max([abs(direct_coeffs[l] - int(lut_coeffs[l])) for l in range(NUM_LEAVES)])}")

print()
print(f"Total coefficient error across all k: {total_coeff_error}")
print(f"Max coefficient error: {max_coeff_error}")
print()

# If coefficient error is 0, the error must be in IDWPT or interleave
print("=== IDWPT ROUNDTRIP ERROR ===")
for k in range(K):
    block = generate_block(test_phase, k_offset=k)
    coeffs = fwpt_jit(block.tolist(), headroom_bits=1)
    recon = idwpt_jit(coeffs, headroom_bits=1)
    recon_np = np.array(recon, dtype=np.int16)
    error_rt = np.abs(recon_np.astype(np.int32) - block.astype(np.int32))
    print(f"k={k}: FWPT->IDWPT roundtrip error: max={error_rt.max()}, mean={error_rt.mean():.2f}")
