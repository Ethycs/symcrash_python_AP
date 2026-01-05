"""Compare LUT coefficients vs direct FWPT at the exact same phase."""

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

print("=== COMPARING LUT vs DIRECT FWPT ===")
print()

lut = get_lut_for_note(wave_id=0, midi_note=69)
print(f"LUT: P={lut.P}, K={lut.K}")

phase_inc = int((f0 / SAMPLE_RATE) * (1 << 32))
print(f"Phase inc: {phase_inc}")
print()

TARGET_RMS_Q15 = 16384
Q15_ONE = 32767

def generate_block(start_phase, k_offset=0):
    """Generate 64-sample block starting at given phase + k_offset samples."""
    block = np.empty(BLOCK_SIZE, dtype=np.int16)
    phase = (start_phase + k_offset * phase_inc) & 0xFFFFFFFF
    for n in range(BLOCK_SIZE):
        sample = sine_q15(phase)
        sample = (sample * TARGET_RMS_Q15) // Q15_ONE
        block[n] = np.clip(sample, -32768, 32767)
        phase = (phase + phase_inc) & 0xFFFFFFFF
    return block

# Test at several runtime phases
test_phases = [0, 1 << 30, 1 << 31, 3 << 30]

for phase_32 in test_phases:
    print(f"=== Runtime phase: {phase_32} ===")

    # Get LUT interpolated coefficients
    subband_output = np.zeros((NUM_LEAVES, K), dtype=np.int16)
    lut.get_subband_samples(phase_32, subband_output)

    # For each k, compare LUT[l, k] vs direct FWPT on block at phase + k*phase_inc
    print("Comparing leaf 3 (high energy) for each k:")
    for k in range(K):
        # LUT value
        lut_val = subband_output[3, k]

        # Direct FWPT at this phase + k samples
        block = generate_block(phase_32, k_offset=k)
        direct_coeffs = fwpt_jit(block.tolist(), headroom_bits=1)
        direct_val = direct_coeffs[3]

        error = abs(int(lut_val) - direct_val)
        print(f"  k={k}: LUT={lut_val:7d}, Direct={direct_val:7d}, Error={error:5d}")

    # Now test reconstruction
    print()
    print("Reconstruction comparison:")

    # Method A: Using LUT coefficients
    idwpt_outputs_lut = []
    for k in range(K):
        coeffs_k = [subband_output[l, k] for l in range(NUM_LEAVES)]
        audio_k = idwpt_jit(coeffs_k, headroom_bits=1)
        idwpt_outputs_lut.append(audio_k)

    output_lut = np.zeros(BLOCK_SIZE, dtype=np.int16)
    for k in range(K):
        for m in range(BLOCK_SIZE // K):
            output_lut[k + m * K] = idwpt_outputs_lut[k][m * K]

    # Method B: Direct FWPT/IDWPT (like test_polyphase_stitching)
    idwpt_outputs_direct = []
    for k in range(K):
        block = generate_block(phase_32, k_offset=k)
        coeffs = fwpt_jit(block.tolist(), headroom_bits=1)
        audio_k = idwpt_jit(coeffs, headroom_bits=1)
        idwpt_outputs_direct.append(audio_k)

    output_direct = np.zeros(BLOCK_SIZE, dtype=np.int16)
    for k in range(K):
        for m in range(BLOCK_SIZE // K):
            output_direct[k + m * K] = idwpt_outputs_direct[k][m * K]

    # Ground truth
    gt_block = generate_block(phase_32, k_offset=0)

    error_lut = np.abs(output_lut.astype(np.int32) - gt_block.astype(np.int32))
    error_direct = np.abs(output_direct.astype(np.int32) - gt_block.astype(np.int32))

    print(f"  LUT method:    max_err={error_lut.max():5d}, mean_err={error_lut.mean():.1f}")
    print(f"  Direct method: max_err={error_direct.max():5d}, mean_err={error_direct.mean():.1f}")
    print()
