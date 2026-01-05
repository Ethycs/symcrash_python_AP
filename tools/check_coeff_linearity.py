"""Check if wavelet coefficients are linear functions of phase."""

import numpy as np
import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit
from symcrash_python_ap.wavelet.sine_core import sine_q15

SAMPLE_RATE = 48000
BLOCK_SIZE = 64
f0 = 440.0

print("=== CHECKING COEFFICIENT LINEARITY vs PHASE ===")
print()

phase_inc = int((f0 / SAMPLE_RATE) * (1 << 32))

TARGET_RMS_Q15 = 16384
Q15_ONE = 32767

def generate_block(start_phase):
    block = np.empty(BLOCK_SIZE, dtype=np.int16)
    phase = start_phase & 0xFFFFFFFF
    for n in range(BLOCK_SIZE):
        sample = sine_q15(phase)
        sample = (sample * TARGET_RMS_Q15) // Q15_ONE
        block[n] = np.clip(sample, -32768, 32767)
        phase = (phase + phase_inc) & 0xFFFFFFFF
    return block

# Check leaf 3 coefficient as a function of phase within one bin
P = 256
shift = 24
bin_size = 1 << shift

print(f"Bin size: {bin_size} phase units")
print()

# Sample coefficient at many points within bin 0 and bin 1
phases_in_bin = np.linspace(0, bin_size * 2, 101)

coeffs_leaf3 = []
for phase_32 in phases_in_bin:
    block = generate_block(int(phase_32))
    coeffs = fwpt_jit(block.tolist(), headroom_bits=1)
    coeffs_leaf3.append(coeffs[3])

coeffs_leaf3 = np.array(coeffs_leaf3)

print("Coefficient values at leaf 3 across phases:")
print(f"  Phase 0 (bin 0 start): {coeffs_leaf3[0]}")
print(f"  Phase {bin_size//2} (bin 0 mid): {coeffs_leaf3[25]}")
print(f"  Phase {bin_size} (bin 1 start): {coeffs_leaf3[50]}")
print(f"  Phase {bin_size + bin_size//2} (bin 1 mid): {coeffs_leaf3[75]}")
print(f"  Phase {bin_size * 2} (bin 2 start): {coeffs_leaf3[100]}")
print()

# Check linearity: if linear, mid = (start + end) / 2
bin0_start = coeffs_leaf3[0]
bin1_start = coeffs_leaf3[50]
bin0_mid_actual = coeffs_leaf3[25]
bin0_mid_linear = (bin0_start + bin1_start) / 2

print("Linearity check for bin 0:")
print(f"  Linear interpolation at mid: {bin0_mid_linear:.1f}")
print(f"  Actual coefficient at mid: {bin0_mid_actual}")
print(f"  Error: {abs(bin0_mid_linear - bin0_mid_actual):.1f}")
print()

# Check multiple points
print("Detailed linearity check across bin 0:")
print("frac     linear_pred   actual   error")
print("-" * 45)

for i, frac in enumerate([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    idx = int(frac * 50)  # Map frac to index in phases_in_bin
    actual = coeffs_leaf3[idx]
    linear = bin0_start + (bin1_start - bin0_start) * frac
    error = abs(linear - actual)
    print(f"{frac:.1f}      {linear:8.1f}     {actual:6d}    {error:6.1f}")

print()
print("=== KEY INSIGHT ===")
print()
print("If error is large (>100), coefficient is NOT a linear function of phase.")
print("This means linear interpolation between bin edges introduces systematic error.")
