"""Debug the interpolation math."""

import numpy as np
import sys
sys.path.insert(0, 'src')

# Clear cache
import symcrash_python_ap.vm.coeff_lut_streaming as lut_mod
lut_mod._NOTE_LUT_CACHE.clear()

from symcrash_python_ap.vm.coeff_lut_streaming import get_lut_for_note
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit
from symcrash_python_ap.wavelet.sine_core import sine_q15

SAMPLE_RATE = 48000
BLOCK_SIZE = 64
f0 = 440.0

print("=== DEBUGGING INTERPOLATION ===")
print()

lut = get_lut_for_note(wave_id=0, midi_note=69)
P = lut.P
shift = 24
mask = 0xFF
bin_size = 1 << shift

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

print(f"LUT P={P}, shift={shift}, bin_size={bin_size}")
print()

# Test case: phase at frac=0.75 into bin 0
phase_32 = int(0.75 * bin_size)
print(f"Test phase: {phase_32} (frac=0.75 into bin 0)")
print()

# Manual interpolation calculation
phase_idx = (phase_32 >> shift) & mask
phase_idx_next = (phase_idx + 1) & mask
frac_bits = shift
frac = phase_32 & ((1 << frac_bits) - 1)

print(f"phase_idx = {phase_idx}")
print(f"phase_idx_next = {phase_idx_next}")
print(f"frac_bits = {frac_bits}")
print(f"frac = {frac} (should be ~{int(0.75 * bin_size)}, actual frac ratio = {frac / bin_size:.4f})")
print()

# Get LUT values at these indices
v0_l3_k0 = int(lut.data[3, 0, phase_idx])
v1_l3_k0 = int(lut.data[3, 0, phase_idx_next])
print(f"LUT[3, 0, {phase_idx}] = {v0_l3_k0}")
print(f"LUT[3, 0, {phase_idx_next}] = {v1_l3_k0}")
print()

# Manual lerp
diff = v1_l3_k0 - v0_l3_k0
half = 1 << (frac_bits - 1)
val_manual = v0_l3_k0 + ((diff * frac + half) >> frac_bits)
print(f"Manual lerp: {v0_l3_k0} + ({diff} * {frac} + {half}) >> {frac_bits} = {val_manual}")
print()

# What get_subband_samples returns
subband_output = np.zeros((64, 8), dtype=np.int16)
lut.get_subband_samples(phase_32, subband_output)
lut_result = subband_output[3, 0]
print(f"get_subband_samples result: {lut_result}")
print()

# What direct FWPT gives
block = generate_block(phase_32, k_offset=0)
direct_coeffs = fwpt_jit(block.tolist(), headroom_bits=1)
direct_val = direct_coeffs[3]
print(f"Direct FWPT at phase {phase_32}: {direct_val}")
print()

# The problem: is LUT[3,0,0] the coefficient at phase=0?
print("=== CHECKING LUT VALUES AT BIN EDGES ===")
print()

# Generate direct FWPT at phase = 0 (bin 0 edge)
block_p0 = generate_block(0, k_offset=0)
direct_p0 = fwpt_jit(block_p0.tolist(), headroom_bits=1)[3]
print(f"Direct FWPT at phase 0: {direct_p0}")
print(f"LUT[3, 0, 0]: {lut.data[3, 0, 0]}")

# Generate direct FWPT at phase = bin_size (bin 1 edge)
block_p1 = generate_block(bin_size, k_offset=0)
direct_p1 = fwpt_jit(block_p1.tolist(), headroom_bits=1)[3]
print(f"Direct FWPT at phase {bin_size}: {direct_p1}")
print(f"LUT[3, 0, 1]: {lut.data[3, 0, 1]}")

# Generate direct FWPT at phase = 0.75 * bin_size
block_p075 = generate_block(phase_32, k_offset=0)
direct_p075 = fwpt_jit(block_p075.tolist(), headroom_bits=1)[3]
print(f"Direct FWPT at phase {phase_32}: {direct_p075}")

# What would linear interpolation give?
expected_linear = direct_p0 + (direct_p1 - direct_p0) * 0.75
print(f"Expected linear interpolation: {expected_linear:.1f}")
print()

# The LUT interpolation result
print(f"LUT interpolation result: {lut_result}")
print(f"Direct FWPT result: {direct_p075}")
print(f"Error: {abs(int(lut_result) - direct_p075)}")
print()

# Check if K dimension is the problem
print("=== CHECKING ALL K VALUES ===")
for k in range(8):
    lut_v0 = lut.data[3, k, 0]
    lut_v1 = lut.data[3, k, 1]
    interp = lut_v0 + ((lut_v1 - lut_v0) * frac + half) >> frac_bits

    block_direct = generate_block(phase_32, k_offset=k)
    direct_k = fwpt_jit(block_direct.tolist(), headroom_bits=1)[3]

    lut_interp = subband_output[3, k]
    error = abs(int(lut_interp) - direct_k)

    print(f"k={k}: LUT[0]={lut_v0:7d}, LUT[1]={lut_v1:7d}, interp={lut_interp:7d}, direct={direct_k:7d}, err={error:4d}")
