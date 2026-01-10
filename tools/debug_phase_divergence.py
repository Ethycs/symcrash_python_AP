"""Debug: Why do some phases cause Mode B divergence?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sin_cos_q15, PHASE_MASK
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.vm.harmonic_bank import generate_basis_frames, NUM_LEAVES, BLOCK_SIZE
from symcrash_python_ap.fixed import Q15_ONE, RSHIFT_RNE, SAT16

SAMPLE_RATE = 48000

print("=" * 70)
print("PHASE DIVERGENCE ANALYSIS")
print("=" * 70)

note = 69  # A4 = 440 Hz
freq_hz = 440.0
phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)
basis = generate_basis_frames(note, H_max=8)

h = 5

def compare_at_phase(start_phase, label):
    """Compare Mode B vs Direct at given starting phase."""
    # Generate target waveform starting at this phase
    target = []
    phase = start_phase
    for n in range(BLOCK_SIZE):
        h_phase = (phase * h) & PHASE_MASK
        sin_val, _ = sin_cos_q15(h_phase)
        target.append(sin_val)
        phase = (phase + phase_inc) & PHASE_MASK

    # Direct: FWPT of target
    direct_coeffs = fwpt_jit(target, headroom_bits=1)

    # Mode B: modulate basis
    h_phase = (start_phase * h) & PHASE_MASK
    sin_phi, cos_phi = sin_cos_q15(h_phase)
    w_sin = RSHIFT_RNE(Q15_ONE * cos_phi, 15)
    w_cos = RSHIFT_RNE(Q15_ONE * sin_phi, 15)

    mode_b_coeffs = []
    for leaf in range(NUM_LEAVES):
        sin_basis = int(basis.Usin_h[h-1, leaf, 0])
        cos_basis = int(basis.Ucos_h[h-1, leaf, 0])
        val = SAT16(RSHIFT_RNE(w_sin * sin_basis, 15) + RSHIFT_RNE(w_cos * cos_basis, 15))
        mode_b_coeffs.append(val)

    # Compare
    err = np.array(direct_coeffs, dtype=np.int32) - np.array(mode_b_coeffs, dtype=np.int32)
    rms = np.sqrt(np.mean(err**2))
    max_err = np.max(np.abs(err))

    print(f"{label:>25} phase={start_phase:>12} h*phase>>32={((start_phase * h) >> 32) & 0xF:>3} coeff_err_RMS={rms:>8.1f} Max={max_err:>6}")
    return rms

print("\n--- Block-by-block phase evolution ---")
phase = 0
block_phase_inc = BLOCK_SIZE * phase_inc
for block in range(10):
    compare_at_phase(phase, f"Block {block}")
    phase = (phase + block_phase_inc) & PHASE_MASK

print("\n--- Clean fractions of 2^32 ---")
for frac in [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]:
    phase = int(frac * (1 << 32)) & PHASE_MASK
    compare_at_phase(phase, f"Phase {frac:.3f}")

print("\n--- The phase from block 1 specifically ---")
phase_block_1 = (block_phase_inc) & PHASE_MASK
print(f"Phase at block 1: {phase_block_1}")
print(f"As fraction of 2^32: {phase_block_1 / (1 << 32):.6f}")

# Check: does the k=0 basis actually match what we need?
print("\n--- Basis Frame Check ---")
print(f"K_STRIDE = 1 (used in generate_basis_frames)")
print(f"k=0 basis was generated starting at phase = 0")
print(f"")
print("But wait - the basis for k=0 encodes sin(h*theta) where theta")
print("goes from 0 to 63*phase_inc. The FWPT of that is what Usin stores.")
print("")
print("When runtime phase is NOT zero, the target waveform is sin(h*(theta+phi))")
print("where theta still goes from 0 to 63*phase_inc, but shifted by phi.")
print("")
print("The trigonometric identity says:")
print("  sin(h*(theta+phi)) = sin(h*theta)cos(h*phi) + cos(h*theta)sin(h*phi)")
print("")
print("But the MODE B formula uses:")
print("  coeff = w_sin*Usin + w_cos*Ucos")
print("")
print("where Usin = FWPT(sin(h*theta)), Ucos = FWPT(cos(h*theta))")
print("")
print("For this to work, we need:")
print("  FWPT(sin(h*(theta+phi))) = FWPT(sin(h*theta))cos(h*phi) + FWPT(cos(h*theta))sin(h*phi)")
print("")
print("This is TRUE because FWPT is LINEAR!")
print("But the PROBLEM is we're comparing to IDWPT of the result...")

print("\n--- The Real Issue ---")
# Let's trace through what Mode B actually produces vs Direct
start_phase = phase_block_1

# Target
target = []
phase = start_phase
for n in range(BLOCK_SIZE):
    h_phase = (phase * h) & PHASE_MASK
    sin_val, _ = sin_cos_q15(h_phase)
    target.append(sin_val)
    phase = (phase + phase_inc) & PHASE_MASK

# Direct FWPT
direct_coeffs = np.array(fwpt_jit(target, headroom_bits=1), dtype=np.int16)

# Mode B coeffs
h_phase = (start_phase * h) & PHASE_MASK
sin_phi, cos_phi = sin_cos_q15(h_phase)
w_sin = RSHIFT_RNE(Q15_ONE * cos_phi, 15)
w_cos = RSHIFT_RNE(Q15_ONE * sin_phi, 15)

mode_b_coeffs = []
for leaf in range(NUM_LEAVES):
    sin_basis = int(basis.Usin_h[h-1, leaf, 0])
    cos_basis = int(basis.Ucos_h[h-1, leaf, 0])
    val = SAT16(RSHIFT_RNE(w_sin * sin_basis, 15) + RSHIFT_RNE(w_cos * cos_basis, 15))
    mode_b_coeffs.append(val)
mode_b_coeffs = np.array(mode_b_coeffs, dtype=np.int16)

# Show leaf-by-leaf comparison
print(f"\nBlock 1 (phase={start_phase}):")
print(f"sin_phi={sin_phi}, cos_phi={cos_phi}")
print(f"\n{'Leaf':>5} {'Direct':>8} {'ModeB':>8} {'Usin':>8} {'Ucos':>8} {'Diff':>8}")
print("-" * 50)
for leaf in range(8):
    d = direct_coeffs[leaf]
    m = mode_b_coeffs[leaf]
    us = basis.Usin_h[h-1, leaf, 0]
    uc = basis.Ucos_h[h-1, leaf, 0]
    print(f"{leaf:>5} {d:>8} {m:>8} {us:>8} {uc:>8} {d-m:>8}")

# The issue must be in how the test_coeff_generation_match.py was computing things
print("\n\nWait - let me check test_coeff_generation_match.py logic...")
