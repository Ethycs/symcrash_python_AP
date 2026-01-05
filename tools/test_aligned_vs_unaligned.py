"""Compare error at bin-aligned vs non-aligned phases."""

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

print("=== ALIGNED VS NON-ALIGNED PHASE ERROR ===")
print()

lut = get_lut_for_note(wave_id=0, midi_note=69)
P = lut.P  # Use actual P from LUT
import math
shift = 32 - int(math.log2(P))
bin_size = 1 << shift
phase_inc = int((f0 / SAMPLE_RATE) * (1 << 32))
print(f"LUT P={P}, shift={shift}, bin_size={bin_size}")

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

def test_at_phase(phase_32, label):
    """Test CCS vs direct at a given phase."""
    # Get LUT coefficients
    subband_output = np.zeros((NUM_LEAVES, K), dtype=np.int16)
    lut.get_subband_samples(phase_32, subband_output)

    # Compute output and coefficient errors
    total_coeff_err = 0
    max_coeff_err = 0
    total_output_err = 0
    max_output_err = 0

    gt_block = generate_block(phase_32, k_offset=0)
    lut_output = np.zeros(BLOCK_SIZE, dtype=np.int16)

    for k in range(K):
        # Direct FWPT
        block = generate_block(phase_32, k_offset=k)
        direct_coeffs = fwpt_jit(block.tolist(), headroom_bits=1)

        # LUT IDWPT
        lut_coeffs = [subband_output[l, k] for l in range(NUM_LEAVES)]
        audio_k = idwpt_jit(lut_coeffs, headroom_bits=1)

        for m in range(BLOCK_SIZE // K):
            lut_output[k + m * K] = audio_k[m * K]

        # Coefficient error
        for l in range(NUM_LEAVES):
            err = abs(direct_coeffs[l] - int(lut_coeffs[l]))
            total_coeff_err += err
            max_coeff_err = max(max_coeff_err, err)

    # Output error
    output_err = np.abs(lut_output.astype(np.int32) - gt_block.astype(np.int32))
    max_output_err = output_err.max()
    mean_output_err = output_err.mean()

    frac = (phase_32 & (bin_size - 1)) / bin_size
    phase_idx = (phase_32 >> shift) & 0xFF

    print(f"{label}:")
    print(f"  phase={phase_32}, idx={phase_idx}, frac={frac:.3f}")
    print(f"  Coefficient error: total={total_coeff_err}, max={max_coeff_err}")
    print(f"  Output error: max={max_output_err}, mean={mean_output_err:.1f}")
    print()

# Test at bin-aligned phases (frac=0)
print("=== BIN-ALIGNED PHASES (frac=0) ===")
for bin_idx in range(0, 256, 32):
    phase_aligned = bin_idx * bin_size
    test_at_phase(phase_aligned, f"Bin {bin_idx}")

# Test at mid-bin phases (frac=0.5)
print("=== MID-BIN PHASES (frac=0.5) ===")
for bin_idx in range(0, 256, 32):
    phase_mid = bin_idx * bin_size + bin_size // 2
    test_at_phase(phase_mid, f"Bin {bin_idx}+0.5")

# Test at the actual runtime phase sequence
print("=== RUNTIME PHASE SEQUENCE ===")
phase_32 = 0
for block in range(10):
    frac = (phase_32 & (bin_size - 1)) / bin_size
    phase_idx = (phase_32 >> shift) & 0xFF
    test_at_phase(phase_32, f"Block {block}")
    phase_32 = (phase_32 + phase_inc * BLOCK_SIZE) & 0xFFFFFFFF
