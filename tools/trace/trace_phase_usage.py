"""Trace how phase is used in the CCS pipeline."""

import numpy as np
import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.vm.coeff_lut_streaming import (
    get_lut_for_note, CCSState, ccs_generate_coeffs_polyphase,
    DEFAULT_K, DEFAULT_P, BLOCK_SIZE, NUM_LEAVES, VM_K
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit

print("=== TRACING PHASE USAGE ===")
print()

# Setup
lut = get_lut_for_note(wave_id=0, midi_note=69)  # 440 Hz
print(f"LUT: K={lut.K}, P={lut.P}, shape={lut.data.shape}")
print()

# Simulate a few blocks
ccs_state = CCSState()
ccs_state.note = 69
ccs_state.wave_id = 0
# Phase increment for 440 Hz
ccs_state.phase_inc = int((440.0 / 48000) * (1 << 32))
print(f"Phase increment: {ccs_state.phase_inc}")
print(f"Phase increment per block: {ccs_state.phase_inc * BLOCK_SIZE}")
print(f"Phase steps per block: {(ccs_state.phase_inc * BLOCK_SIZE) >> 26:.4f}")
print()

print("=== BLOCK BY BLOCK PHASE TRACE ===")
print("block   phase_32          phase_idx (P=64)   k=0 coeff[3]  k=7 coeff[3]")
print("-" * 80)

for block in range(10):
    # This is what happens in executor.py
    phase_before = ccs_state.phase
    phase_idx = (phase_before >> 26) & 0x3F

    # Get coefficients
    carrier_512 = np.zeros(512, dtype=np.int16)
    ccs_generate_coeffs_polyphase(ccs_state, carrier_512)

    # Check leaf 3 (high energy)
    coeff_l3_k0 = carrier_512[3 * VM_K + 0]
    coeff_l3_k7 = carrier_512[3 * VM_K + 7]

    print(f"{block:5d}   {phase_before:12d}   {phase_idx:8d}           {coeff_l3_k0:8d}    {coeff_l3_k7:8d}")

    # Advance phase (happens at end of block in executor)
    ccs_state.advance_block()

print()
print("=== KEY INSIGHT ===")
print()
print("The K dimension in carrier_512 comes from LUT data[:, k, phase_idx]")
print("where k=0..7 are DIFFERENT BLOCK ALIGNMENTS at the SAME oscillator phase.")
print()
print("But the polyphase interleave in lane_ccs_render expects:")
print("  coeff[:, k] to represent the SAME waveform at different SUB-SAMPLE phases")
print()
print("The mismatch:")
print("  - LUT k dimension: block shifted by k samples (spatial offset)")
print("  - Interleave k dimension: sub-phase for sample n where n % K == k (temporal offset)")
print()

# Now let's check what the K values actually represent
print("=== WHAT K VALUES REPRESENT IN LUT ===")
print()
print("For phase_idx=0, comparing coeff[leaf=3, k, phase=0]:")
for k in range(lut.K):
    val = lut.data[3, k, 0]
    print(f"  k={k}: {val:8d}")

print()
print("These are coefficients from FWPT of blocks starting at sample k, k+1, k+2...")
print("They should smoothly transition (they do, diff ~300-400)")
print()
print("But when we use them for polyphase interleave:")
print("  - output[0] uses y_k0[0], which reconstructs x[start+0]")
print("  - output[1] uses y_k1[0], which reconstructs x[start+1]")
print("  - ...")
print("This IS what we want! But only if phase advances correctly.")
print()
print("THE REAL PROBLEM:")
print("Phase is updated ONCE per 64-sample block.")
print("So k=0..7 all see the SAME phase_idx.")
print("But they represent different sub-sample offsets (0..7 samples).")
print()
print("Each k should ideally see phase advanced by k samples:")
print("  k=0: phase")
print("  k=1: phase + 1*phase_inc")
print("  k=2: phase + 2*phase_inc")
print("  ...")
