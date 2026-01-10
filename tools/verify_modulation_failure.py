"""Verify: Mode B modulation fails because roundtrip error varies with phase.

The Mode B formula:
    coeff = w_sin*Usin + w_cos*Ucos

Assumes that modulating the basis frames with sin(hphi), cos(hphi) produces
the same result as FWPT of the actual target waveform.

This ONLY works if FWPT→IDWPT is lossless. For H5, it's not.
"""
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
print("MODE B MODULATION FAILURE ANALYSIS")
print("=" * 70)

note = 69  # A4 = 440 Hz
freq_hz = 440.0
phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)
basis = generate_basis_frames(note, H_max=8)

# Test for H1 and H5
for h in [1, 5]:
    print(f"\n{'='*70}")
    print(f"H{h} ANALYSIS")
    print(f"{'='*70}")

    print(f"\n--- Phase 0 (basis was generated at this phase) ---")
    # At phase=0, the basis should work perfectly
    start_phase = 0

    # Generate target waveform
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
    print(f"At phase=0: sin(h*phi)={sin_phi}, cos(h*phi)={cos_phi}")
    print(f"Mode B coeff error: RMS={np.sqrt(np.mean(err**2)):.1f}, Max={np.max(np.abs(err))}")

    print(f"\n--- Phase = quarter cycle (phi = 2^30) ---")
    start_phase = 1 << 30  # Quarter cycle

    # Generate target waveform
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
    print(f"At phase=2^30: sin(h*phi)={sin_phi}, cos(h*phi)={cos_phi}")
    print(f"Mode B coeff error: RMS={np.sqrt(np.mean(err**2)):.1f}, Max={np.max(np.abs(err))}")

    # What's really happening: the basis encodes the ERROR from roundtrip
    print(f"\n--- Roundtrip Analysis ---")

    # Basis was generated from FWPT of sin(h*theta) starting at phase 0
    # When we IDWPT the basis, we get back an approximation of sin(h*theta)
    basis_sin_recon = idwpt_jit(basis.Usin_h[h-1, :, 0].tolist(), headroom_bits=1)
    basis_cos_recon = idwpt_jit(basis.Ucos_h[h-1, :, 0].tolist(), headroom_bits=1)

    # Generate original sin/cos at phase 0
    original_sin = []
    original_cos = []
    phase = 0
    for n in range(BLOCK_SIZE):
        h_phase = (phase * h) & PHASE_MASK
        sin_val, cos_val = sin_cos_q15(h_phase)
        original_sin.append(sin_val)
        original_cos.append(cos_val)
        phase = (phase + phase_inc) & PHASE_MASK

    # Roundtrip error in basis
    sin_err = np.array(original_sin, dtype=np.int32) - np.array(basis_sin_recon, dtype=np.int32)
    cos_err = np.array(original_cos, dtype=np.int32) - np.array(basis_cos_recon, dtype=np.int32)

    print(f"Basis sin roundtrip error: RMS={np.sqrt(np.mean(sin_err**2)):.0f}")
    print(f"Basis cos roundtrip error: RMS={np.sqrt(np.mean(cos_err**2)):.0f}")

print("\n" + "=" * 70)
print("ROOT CAUSE IDENTIFIED")
print("=" * 70)
print("""
The Mode B modulation formula assumes:
    sin(h*(θ+phi)) = sin(hθ)cos(hphi) + cos(hθ)sin(hphi)

But the basis frames Usin, Ucos are NOT sin(hθ), cos(hθ).
They are FWPT(sin(hθ)), FWPT(cos(hθ)).

When IDWPT(Usin) ≈ sin(hθ) (lossless), the modulation works.
When IDWPT(Usin) ≠ sin(hθ) (lossy), the modulation breaks.

For H5, roundtrip error is 38.8%, so modulation fails badly.

The 50 Hz artifact is the ERROR signal being modulated across blocks.
""")
