"""Compare basis frame generation to runtime coefficient generation."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from symcrash_python_ap.vm.harmonic_bank import (
    generate_basis_frames, VM_K, NUM_LEAVES, BLOCK_SIZE, PHASE_MASK, DEFAULT_H_MAX
)
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.wavelet.sine_core import sin_cos_q15, freq_to_phase_inc
from symcrash_python_ap.fixed import Q15_ONE, RSHIFT_RNE, SAT16

note = 60
freq_hz = 261.63
phase_inc = freq_to_phase_inc(freq_hz, 48000)
print(f"Phase inc per sample: {phase_inc}")

# Generate basis frames
basis = generate_basis_frames(note, H_max=32)

# Manually generate a sine wave for block 0 and block 1
# and compare to Mode B output

def generate_reference_sine_block(start_phase, phase_inc, h=1):
    """Generate reference sine block directly."""
    block = []
    phase = start_phase
    for n in range(BLOCK_SIZE):
        h_phase = (phase * h) & PHASE_MASK
        sin_val, _ = sin_cos_q15(h_phase)
        block.append(sin_val)
        phase = (phase + phase_inc) & PHASE_MASK
    return block

def generate_modeb_block(start_phase, basis, h=1):
    """Generate block using Mode B method (matching mode_b_generate_coeffs)."""
    # For single harmonic h with a_h=1, b_h=0:
    # w_sin = a_h * cos(h*phi) - b_h * sin(h*phi) = cos(h*phi)
    # w_cos = a_h * sin(h*phi) + b_h * cos(h*phi) = sin(h*phi)
    #
    # coeff = w_sin * Usin_h + w_cos * Ucos_h

    coeffs = np.zeros((NUM_LEAVES, VM_K), dtype=np.int16)

    for k in range(VM_K):
        # Runtime uses same phase for all k-slots
        phi = start_phase
        h_phase = (phi * h) & PHASE_MASK
        sin_phi, cos_phi = sin_cos_q15(h_phase)

        # Weights (a_h=1, b_h=0)
        a_h = Q15_ONE
        b_h = 0
        w_sin = RSHIFT_RNE(a_h * cos_phi - b_h * sin_phi, 15)
        w_cos = RSHIFT_RNE(a_h * sin_phi + b_h * cos_phi, 15)

        for leaf in range(NUM_LEAVES):
            sin_contrib = RSHIFT_RNE(w_sin * int(basis.Usin_h[h - 1, leaf, k]), 15)
            cos_contrib = RSHIFT_RNE(w_cos * int(basis.Ucos_h[h - 1, leaf, k]), 15)
            coeffs[leaf, k] = SAT16(sin_contrib + cos_contrib)

    # Use k=0 for IDWPT
    audio = idwpt_jit(coeffs[:, 0].tolist(), headroom_bits=1)
    return audio

# Test block 0 (phase = 0)
print("\n=== Block 0 (phase=0) ===")
ref_block0 = generate_reference_sine_block(0, phase_inc, h=1)
modeb_block0 = generate_modeb_block(0, basis, h=1)

print(f"Reference first 8: {ref_block0[:8]}")
print(f"Mode B first 8:    {list(modeb_block0[:8])}")
print(f"Max diff: {max(abs(r - m) for r, m in zip(ref_block0, modeb_block0))}")

# Test block 1 (phase = 64 * phase_inc)
block1_phase = (BLOCK_SIZE * phase_inc) & PHASE_MASK
print(f"\n=== Block 1 (phase={block1_phase}) ===")
ref_block1 = generate_reference_sine_block(block1_phase, phase_inc, h=1)
modeb_block1 = generate_modeb_block(block1_phase, basis, h=1)

print(f"Reference first 8: {ref_block1[:8]}")
print(f"Mode B first 8:    {list(modeb_block1[:8])}")
print(f"Max diff: {max(abs(r - m) for r, m in zip(ref_block1, modeb_block1))}")

# Check boundary between block 0 and block 1
print(f"\n=== Boundary check ===")
print(f"Block 0 last:  ref={ref_block0[-1]}, modeb={modeb_block0[-1]}")
print(f"Block 1 first: ref={ref_block1[0]}, modeb={modeb_block1[0]}")
print(f"Reference diff at boundary: {abs(ref_block1[0] - ref_block0[-1])}")
print(f"Mode B diff at boundary:    {abs(modeb_block1[0] - modeb_block0[-1])}")

# Now test with higher harmonic
print("\n=== Testing h=31 (high harmonic) ===")
ref_h31_b0 = generate_reference_sine_block(0, phase_inc, h=31)
modeb_h31_b0 = generate_modeb_block(0, basis, h=31)
print(f"Max diff for h=31, block 0: {max(abs(r - m) for r, m in zip(ref_h31_b0, modeb_h31_b0))}")

ref_h31_b1 = generate_reference_sine_block(block1_phase, phase_inc, h=31)
modeb_h31_b1 = generate_modeb_block(block1_phase, basis, h=31)
print(f"Max diff for h=31, block 1: {max(abs(r - m) for r, m in zip(ref_h31_b1, modeb_h31_b1))}")

print(f"\nBlock 0 last:  ref={ref_h31_b0[-1]}, modeb={modeb_h31_b0[-1]}")
print(f"Block 1 first: ref={ref_h31_b1[0]}, modeb={modeb_h31_b1[0]}")
print(f"Reference diff at boundary: {abs(ref_h31_b1[0] - ref_h31_b0[-1])}")
print(f"Mode B diff at boundary:    {abs(modeb_h31_b1[0] - modeb_h31_b0[-1])}")

# Check what the basis frames actually encode
print("\n=== Basis frame analysis ===")
# For k=0, the basis should reconstruct sin(h*theta) where theta goes 0 to 63*phase_inc
# Let's verify by IDWPT of the basis

for h in [1, 7, 15, 31]:
    print(f"\n--- h={h} ---")
    # IDWPT of Usin_h for k=0
    usin_audio = idwpt_jit(basis.Usin_h[h-1, :, 0].tolist(), headroom_bits=1)
    # Compare to reference
    ref_sin = []
    phase = 0  # k=0 starts at phase 0
    for n in range(BLOCK_SIZE):
        h_phase = (phase * h) & PHASE_MASK
        s, _ = sin_cos_q15(h_phase)
        ref_sin.append(s)
        phase = (phase + phase_inc) & PHASE_MASK

    max_diff = max(abs(int(r) - int(u)) for r, u in zip(ref_sin, usin_audio))
    print(f"  Usin_h[k=0] vs reference: max_diff={max_diff}")
    print(f"  First 4 ref:   {ref_sin[:4]}")
    print(f"  First 4 basis: {list(usin_audio[:4])}")
