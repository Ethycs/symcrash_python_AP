"""Step by step debug of interpolation."""

import numpy as np
import sys
sys.path.insert(0, 'src')

# Clear cache
import symcrash_python_ap.vm.coeff_lut_streaming as lut_mod
lut_mod._NOTE_LUT_CACHE.clear()

from symcrash_python_ap.vm.coeff_lut_streaming import get_lut_for_note

print("=== STEP BY STEP INTERPOLATION DEBUG ===")
print()

lut = get_lut_for_note(wave_id=0, midi_note=69)
P = lut.P  # 256
shift = 24
bin_size = 1 << shift  # 16777216

# Test phase at frac=0.75 into bin 0
phase_32 = int(0.75 * bin_size)  # 12582912
print(f"phase_32 = {phase_32}")
print()

# Step 1: Compute indices
phase_idx = (phase_32 >> shift) & 0xFF
phase_idx_next = (phase_idx + 1) & 0xFF
print(f"phase_idx = {phase_idx}")
print(f"phase_idx_next = {phase_idx_next}")
print()

# Step 2: Compute frac
frac_bits = shift
frac = phase_32 & ((1 << frac_bits) - 1)
print(f"frac_bits = {frac_bits}")
print(f"frac mask = {(1 << frac_bits) - 1}")
print(f"frac = {frac}")
print(f"frac as ratio = {frac / (1 << frac_bits):.4f}")
print()

# Step 3: Get LUT values
# Check a single coefficient: leaf 3, k=0
v0 = int(lut.data[3, 0, phase_idx])
v1 = int(lut.data[3, 0, phase_idx_next])
print(f"v0 = LUT[3, 0, {phase_idx}] = {v0}")
print(f"v1 = LUT[3, 0, {phase_idx_next}] = {v1}")
print()

# Step 4: Compute interpolation
diff = v1 - v0
half = 1 << (frac_bits - 1)
print(f"diff = v1 - v0 = {diff}")
print(f"half = 2^{frac_bits-1} = {half}")
print()

# The interpolation formula
# val = v0 + ((diff * frac + half) >> frac_bits)
mult = diff * frac
print(f"diff * frac = {mult}")
added = mult + half
print(f"diff * frac + half = {added}")
shifted = added >> frac_bits
print(f"(diff * frac + half) >> {frac_bits} = {shifted}")
val = v0 + shifted
print(f"v0 + shifted = {val}")
print()

# Now let's trace what numpy does
print("=== NUMPY IMPLEMENTATION ===")
print()

# This is what get_subband_samples does
v0_np = lut.data[:, :, phase_idx].astype(np.int32)
v1_np = lut.data[:, :, phase_idx_next].astype(np.int32)
diff_np = v1_np - v0_np
val_np = v0_np + ((diff_np * frac + half) >> frac_bits)

print(f"v0_np[3,0] = {v0_np[3, 0]}")
print(f"v1_np[3,0] = {v1_np[3, 0]}")
print(f"diff_np[3,0] = {diff_np[3, 0]}")
print(f"diff_np[3,0] * frac = {diff_np[3, 0] * frac}")
print(f"(diff_np[3,0] * frac + half) = {diff_np[3, 0] * frac + half}")
print(f"val_np[3,0] = {val_np[3, 0]}")
print()

# Call the actual function
output = np.zeros((64, 8), dtype=np.int16)
lut.get_subband_samples(phase_32, output)
print(f"get_subband_samples output[3,0] = {output[3, 0]}")
print()

# Check if there's overflow in the multiplication
print("=== OVERFLOW CHECK ===")
max_diff = np.max(np.abs(lut.data[:, :, 0].astype(np.int32) - lut.data[:, :, 1].astype(np.int32)))
print(f"Max |diff| between adjacent bins: {max_diff}")
print(f"frac at 0.75: {frac}")
print(f"Max product |diff * frac|: {max_diff * frac}")
print(f"Max int32: {2**31 - 1}")
print(f"Overflow? {max_diff * frac > 2**31 - 1}")
