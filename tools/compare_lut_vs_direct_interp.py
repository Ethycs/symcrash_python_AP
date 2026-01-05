"""Compare LUT interpolated coefficients vs direct FWPT at non-aligned phases."""

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

print("=== COMPARING LUT INTERPOLATION vs DIRECT FWPT ===")
print()

lut = get_lut_for_note(wave_id=0, midi_note=69)
P = lut.P
shift = 24  # For P=256

print(f"LUT: P={P}, K={lut.K}")
print(f"Bin size (in phase): {1 << shift}")
print()

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

# Test at phases within bin 0 (0 to (1<<shift)-1)
# These require interpolation between bin 0 and bin 1
print("=== Testing phases within bin 0 (interpolation cases) ===")
print()

bin_size = 1 << shift
test_fracs = [0.0, 0.25, 0.5, 0.75, 0.99]

for frac in test_fracs:
    phase_32 = int(frac * bin_size)

    print(f"Phase: {phase_32} (frac={frac:.2f} into bin 0)")

    # Get LUT interpolated coefficients
    subband_output = np.zeros((NUM_LEAVES, K), dtype=np.int16)
    lut.get_subband_samples(phase_32, subband_output)

    # Direct FWPT at this exact phase for k=0
    block = generate_block(phase_32, k_offset=0)
    direct_coeffs = fwpt_jit(block.tolist(), headroom_bits=1)

    # Compare leaf 3
    lut_val = subband_output[3, 0]
    direct_val = direct_coeffs[3]
    error = abs(int(lut_val) - direct_val)

    print(f"  Leaf 3, k=0: LUT={lut_val:7d}, Direct={direct_val:7d}, Error={error:4d}")

    # Full reconstruction comparison
    idwpt_outputs_lut = []
    for k in range(K):
        coeffs_k = [subband_output[l, k] for l in range(NUM_LEAVES)]
        audio_k = idwpt_jit(coeffs_k, headroom_bits=1)
        idwpt_outputs_lut.append(audio_k)

    output_lut = np.zeros(BLOCK_SIZE, dtype=np.int16)
    for k in range(K):
        for m in range(BLOCK_SIZE // K):
            output_lut[k + m * K] = idwpt_outputs_lut[k][m * K]

    # Ground truth for this phase
    gt_block = generate_block(phase_32, k_offset=0)
    error_recon = np.abs(output_lut.astype(np.int32) - gt_block.astype(np.int32))
    print(f"  Reconstruction: max_err={error_recon.max():4d}, mean_err={error_recon.mean():.1f}")
    print()

# Now test what happens at actual runtime phase values
print("=== Testing at actual runtime phase sequence ===")
print()

phase_32 = 0
for block in range(10):
    phase_idx = (phase_32 >> shift) & 0xFF
    frac = (phase_32 & (bin_size - 1)) / bin_size

    # LUT
    subband_output = np.zeros((NUM_LEAVES, K), dtype=np.int16)
    lut.get_subband_samples(phase_32, subband_output)

    # Direct
    block_data = generate_block(phase_32, k_offset=0)
    direct_coeffs = fwpt_jit(block_data.tolist(), headroom_bits=1)

    lut_val = subband_output[3, 0]
    direct_val = direct_coeffs[3]
    error = abs(int(lut_val) - direct_val)

    # Reconstruction
    idwpt_outputs_lut = []
    for k in range(K):
        coeffs_k = [subband_output[l, k] for l in range(NUM_LEAVES)]
        audio_k = idwpt_jit(coeffs_k, headroom_bits=1)
        idwpt_outputs_lut.append(audio_k)

    output_lut = np.zeros(BLOCK_SIZE, dtype=np.int16)
    for k in range(K):
        for m in range(BLOCK_SIZE // K):
            output_lut[k + m * K] = idwpt_outputs_lut[k][m * K]

    gt_block = generate_block(phase_32, k_offset=0)
    error_recon = np.abs(output_lut.astype(np.int32) - gt_block.astype(np.int32))

    print(f"Block {block}: phase={phase_32:12d}, idx={phase_idx:3d}, frac={frac:.3f}")
    print(f"  Coeff err={error:4d}, Recon max_err={error_recon.max():4d}, mean={error_recon.mean():.1f}")

    # Advance phase
    phase_32 = (phase_32 + phase_inc * BLOCK_SIZE) & 0xFFFFFFFF
