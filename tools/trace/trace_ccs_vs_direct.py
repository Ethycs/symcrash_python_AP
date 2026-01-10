"""Compare CCS LUT coefficients vs direct FWPT on ground truth."""

import numpy as np
import sys
sys.path.insert(0, 'src')

# Clear cache
import symcrash_python_ap.vm.coeff_lut_streaming as lut_mod
lut_mod._NOTE_LUT_CACHE.clear()

from symcrash_python_ap.vm.coeff_lut_streaming import (
    get_lut_for_note, CCSState, ccs_generate_coeffs_polyphase,
    DEFAULT_K, DEFAULT_P, BLOCK_SIZE, NUM_LEAVES, VM_K
)
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.wavelet.sine_core import sine_q15

SAMPLE_RATE = 48000
f0 = 440.0
K = 8

print("=== COMPARING CCS LUT vs DIRECT FWPT ===")
print()

# Get LUT
lut = get_lut_for_note(wave_id=0, midi_note=69)
print(f"LUT: P={lut.P}, K={lut.K}, shape={lut.data.shape}")

# Setup CCS state
ccs_state = CCSState()
ccs_state.note = 69
ccs_state.wave_id = 0
ccs_state.phase_inc = int((f0 / SAMPLE_RATE) * (1 << 32))
ccs_state.phase = 0

print(f"Phase inc: {ccs_state.phase_inc}")
print()

# Generate ground truth for comparison
# The ground truth should match what the LUT was built from
TARGET_RMS_Q15 = 16384
Q15_ONE = 32767

def generate_block_direct(start_phase, k):
    """Generate a 64-sample block starting at start_phase + k*phase_inc."""
    block = np.empty(BLOCK_SIZE, dtype=np.int16)
    phase = (start_phase + k * ccs_state.phase_inc) & 0xFFFFFFFF
    for n in range(BLOCK_SIZE):
        sample = sine_q15(phase)
        sample = (sample * TARGET_RMS_Q15) // Q15_ONE
        block[n] = np.clip(sample, -32768, 32767)
        phase = (phase + ccs_state.phase_inc) & 0xFFFFFFFF
    return block

# For block 0, phase=0
# LUT should have phase_idx = (0 >> 24) & 0xFF = 0
# But LUT was built with phase_p at CENTER of bin: (p << shift) + (1 << (shift-1))
# For P=256, shift=24, so phase_p = (0 << 24) + (1 << 23) = 8388608

print("=== BLOCK 0 COMPARISON ===")
print()

# What CCS gives us
carrier_512 = np.zeros(512, dtype=np.int16)
ccs_generate_coeffs_polyphase(ccs_state, carrier_512)

# Reshape to (64, K)
ccs_coeffs = carrier_512.reshape(64, K)

print("CCS coefficients (from LUT interpolation):")
print(f"  Phase = {ccs_state.phase}")
print(f"  phase_idx = {(ccs_state.phase >> 24) & 0xFF}")
print(f"  Leaf 3, k=0: {ccs_coeffs[3, 0]}")
print(f"  Leaf 3, k=7: {ccs_coeffs[3, 7]}")

# What direct FWPT gives us for the same phase
# LUT was built with phase_p = (p << 24) + (1 << 23) for bin p
# When runtime phase=0, we interpolate between p=0 and p=1
# p=0 was built at phase_p = (1 << 23) = 8388608
# p=1 was built at phase_p = (1 << 24) + (1 << 23) = 25165824

print()
print("Direct FWPT at runtime phase=0:")
direct_coeffs_k0 = fwpt_jit(generate_block_direct(0, 0).tolist(), headroom_bits=1)
direct_coeffs_k7 = fwpt_jit(generate_block_direct(0, 7).tolist(), headroom_bits=1)
print(f"  Leaf 3, k=0: {direct_coeffs_k0[3]}")
print(f"  Leaf 3, k=7: {direct_coeffs_k7[3]}")

print()
print("Direct FWPT at LUT phase_p for bin 0 (center = 8388608):")
phase_bin_center = (1 << 23)  # Center of bin 0 for P=256
direct_coeffs_k0_lut = fwpt_jit(generate_block_direct(phase_bin_center, 0).tolist(), headroom_bits=1)
direct_coeffs_k7_lut = fwpt_jit(generate_block_direct(phase_bin_center, 7).tolist(), headroom_bits=1)
print(f"  Leaf 3, k=0: {direct_coeffs_k0_lut[3]}")
print(f"  Leaf 3, k=7: {direct_coeffs_k7_lut[3]}")

print()
print("LUT data at p=0:")
print(f"  Leaf 3, k=0: {lut.data[3, 0, 0]}")
print(f"  Leaf 3, k=7: {lut.data[3, 7, 0]}")

# The issue: LUT is built at phase centers, but runtime uses phase=0
# which is at the EDGE of bin 0, not the center!
print()
print("=== KEY INSIGHT ===")
print()
print("LUT bin 0 is built at phase 8388608 (center of [0, 16777216))")
print("Runtime phase=0 is at the EDGE, not center")
print("This creates a systematic offset of half a phase bin!")
print()

# Check what the error is
print("=== ERROR ANALYSIS ===")
print()
error_k0 = abs(int(ccs_coeffs[3, 0]) - direct_coeffs_k0[3])
error_k7 = abs(int(ccs_coeffs[3, 7]) - direct_coeffs_k7[3])
print(f"Error at k=0: {error_k0}")
print(f"Error at k=7: {error_k7}")
