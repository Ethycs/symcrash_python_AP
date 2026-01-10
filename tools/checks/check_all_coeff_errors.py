"""Check interpolation error across all leaves and k values."""

import numpy as np
import sys
sys.path.insert(0, 'src')

# Clear cache
import symcrash_python_ap.vm.coeff_lut_streaming as lut_mod
lut_mod._NOTE_LUT_CACHE.clear()

from symcrash_python_ap.vm.coeff_lut_streaming import (
    get_lut_for_note, BLOCK_SIZE, NUM_LEAVES, DEFAULT_K
)
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit
from symcrash_python_ap.wavelet.sine_core import sine_q15

SAMPLE_RATE = 48000
f0 = 440.0
K = 8

print("=== CHECKING ALL COEFFICIENT ERRORS ===")
print()

lut = get_lut_for_note(wave_id=0, midi_note=69)
P = lut.P
import math
shift = 32 - int(math.log2(P))
bin_size = 1 << shift
phase_inc = int((f0 / SAMPLE_RATE) * (1 << 32))

print(f"LUT P={P}, K={K}")
print()

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

# Test at mid-bin (frac=0.5)
test_phase = bin_size // 2

# Get LUT interpolated coefficients
subband_output = np.zeros((NUM_LEAVES, K), dtype=np.int16)
lut.get_subband_samples(test_phase, subband_output)

# Compare to direct FWPT for each k
print("Coefficient error distribution (LUT interpolated vs Direct FWPT):")
print()

total_error = 0
errors_per_leaf = np.zeros(NUM_LEAVES)
errors_per_k = np.zeros(K)
error_histogram = {}

for k in range(K):
    block = generate_block(test_phase, k_offset=k)
    direct_coeffs = fwpt_jit(block.tolist(), headroom_bits=1)

    for l in range(NUM_LEAVES):
        lut_val = int(subband_output[l, k])
        direct_val = direct_coeffs[l]
        err = abs(lut_val - direct_val)
        total_error += err
        errors_per_leaf[l] += err
        errors_per_k[k] += err
        error_histogram[err] = error_histogram.get(err, 0) + 1

print(f"Total error: {total_error}")
print(f"Average error per coefficient: {total_error / (NUM_LEAVES * K):.2f}")
print()

print("Error histogram:")
for err in sorted(error_histogram.keys()):
    count = error_histogram[err]
    pct = 100 * count / (NUM_LEAVES * K)
    print(f"  Error {err}: {count} coefficients ({pct:.1f}%)")
print()

print("Error by k-slot:")
for k in range(K):
    print(f"  k={k}: total={int(errors_per_k[k])}, avg={errors_per_k[k]/NUM_LEAVES:.2f}")
print()

print("Top 10 leaves by error:")
top_leaves = np.argsort(-errors_per_leaf)[:10]
for l in top_leaves:
    print(f"  leaf {l}: total_error={int(errors_per_leaf[l])}")

# Check if the error is from rounding vs actual mismatch
print()
print("=== ROUNDING VS ACTUAL MISMATCH ===")
print()

# The LUT interpolation formula: val = v0 + ((v1 - v0) * frac + half) >> frac_bits
# The frac for mid-bin should be exactly 0.5, so:
# val = v0 + (v1 - v0) / 2 = (v0 + v1) / 2

phase_idx = (test_phase >> shift) & (P - 1)
frac = test_phase & ((1 << shift) - 1)
print(f"test_phase={test_phase}, phase_idx={phase_idx}, frac={frac}, frac_ratio={frac/(1<<shift):.4f}")

# Check for k=0, leaf=3 in detail
l, k = 3, 0
v0 = int(lut.data[l, k, phase_idx])
v1 = int(lut.data[l, k, phase_idx + 1])
half = 1 << (shift - 1)
frac_64 = np.int64(frac)
interp_val = v0 + ((np.int64(v1 - v0) * frac_64 + half) >> shift)

block = generate_block(test_phase, k_offset=k)
direct = fwpt_jit(block.tolist(), headroom_bits=1)[l]

print(f"Leaf {l}, k={k}:")
print(f"  v0={v0}, v1={v1}")
print(f"  Interpolated: {interp_val}")
print(f"  Direct FWPT: {direct}")
print(f"  LUT output: {subband_output[l, k]}")
print(f"  Error: {abs(int(subband_output[l, k]) - direct)}")
