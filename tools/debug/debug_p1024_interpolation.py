"""Debug why P=1024 still has interpolation errors."""

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

print("=== DEBUGGING P=1024 INTERPOLATION ===")
print()

lut = get_lut_for_note(wave_id=0, midi_note=69)
P = lut.P
import math
shift = 32 - int(math.log2(P))
bin_size = 1 << shift
phase_inc = int((f0 / SAMPLE_RATE) * (1 << 32))

print(f"LUT P={P}, shift={shift}, bin_size={bin_size}")
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
test_phase = bin_size // 2  # Mid-point of bin 0
print(f"Test phase: {test_phase} (frac=0.5 into bin 0)")
print()

# What the LUT has at bin edges
print("=== LUT VALUES AT BIN EDGES ===")
lut_bin0 = lut.data[:, 0, 0]  # Leaf coefficients at bin 0, k=0
lut_bin1 = lut.data[:, 0, 1]  # Leaf coefficients at bin 1, k=0
print(f"LUT bin 0, leaf 3: {lut_bin0[3]}")
print(f"LUT bin 1, leaf 3: {lut_bin1[3]}")
linear_interp = (int(lut_bin0[3]) + int(lut_bin1[3])) // 2
print(f"Linear interpolation at frac=0.5: {linear_interp}")
print()

# What direct FWPT gives at mid-bin
print("=== DIRECT FWPT AT MID-BIN ===")
block_mid = generate_block(test_phase, k_offset=0)
direct_coeffs = fwpt_jit(block_mid.tolist(), headroom_bits=1)
print(f"Direct FWPT leaf 3: {direct_coeffs[3]}")
print(f"Error vs linear interp: {abs(direct_coeffs[3] - linear_interp)}")
print()

# What about bin edges?
print("=== DIRECT FWPT AT BIN EDGES ===")
block_bin0 = generate_block(0, k_offset=0)
block_bin1 = generate_block(bin_size, k_offset=0)
direct_bin0 = fwpt_jit(block_bin0.tolist(), headroom_bits=1)
direct_bin1 = fwpt_jit(block_bin1.tolist(), headroom_bits=1)
print(f"Direct FWPT at phase 0, leaf 3: {direct_bin0[3]}")
print(f"Direct FWPT at phase {bin_size}, leaf 3: {direct_bin1[3]}")
print()

# Compare to LUT
print("=== LUT vs DIRECT AT BIN EDGES ===")
print(f"LUT bin 0 leaf 3: {lut_bin0[3]}, Direct: {direct_bin0[3]}, Error: {abs(int(lut_bin0[3]) - direct_bin0[3])}")
print(f"LUT bin 1 leaf 3: {lut_bin1[3]}, Direct: {direct_bin1[3]}, Error: {abs(int(lut_bin1[3]) - direct_bin1[3])}")
print()

# The issue: LUT is built at phase centers, not edges!
# Let me check what phase the LUT was actually built at
print("=== LUT BUILD PHASE CHECK ===")
# For P=1024, shift=22, bin_size=4194304
# LUT bin p was built at phase_p = p << shift = p * bin_size
# So LUT bin 0 is at phase 0, bin 1 at phase 4194304
print(f"LUT bin 0 should be built at phase: 0")
print(f"LUT bin 1 should be built at phase: {bin_size}")

# Verify by comparing LUT values to direct FWPT at those phases
print()
print("=== VERIFICATION ===")
for p in range(5):
    phase_p = p * bin_size
    block = generate_block(phase_p, k_offset=0)
    direct = fwpt_jit(block.tolist(), headroom_bits=1)
    lut_val = lut.data[3, 0, p]
    err = abs(int(lut_val) - direct[3])
    print(f"Bin {p} at phase {phase_p}: LUT={lut_val}, Direct={direct[3]}, Error={err}")
