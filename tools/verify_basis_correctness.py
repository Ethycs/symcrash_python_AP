"""Verify basis frame correctness - do they reconstruct the expected signal?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sin_cos_q15, PHASE_MASK
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.vm.harmonic_bank import VM_K, K_STRIDE, NUM_LEAVES, BLOCK_SIZE
from symcrash_python_ap.fixed import Q15_ONE, RSHIFT_RNE, SAT16

SAMPLE_RATE = 48000

print("=" * 70)
print("BASIS FRAME CORRECTNESS VERIFICATION")
print(f"VM_K={VM_K}, K_STRIDE={K_STRIDE}")
print("=" * 70)

note = 69
freq_hz = 440.0
phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)
h = 5

# What we WANT: Generate H5 time-domain signal, then IDWPT its coefficients
print("\n--- Ground Truth: Direct FWPT of H5 signal ---")

# Generate H5 signal starting at phase 0
h5_signal = []
phase = 0
for n in range(BLOCK_SIZE):
    h_phase = (phase * h) & PHASE_MASK
    sin_val, _ = sin_cos_q15(h_phase)
    h5_signal.append(sin_val)
    phase = (phase + phase_inc) & PHASE_MASK

h5_coeffs_direct = np.array(fwpt_jit(h5_signal, headroom_bits=1), dtype=np.int16)
print(f"Direct H5 coeffs (leaf 0-7): {h5_coeffs_direct[:8].tolist()}")

# Reconstruct and check
h5_recon_direct = np.array(idwpt_jit(h5_coeffs_direct.tolist(), headroom_bits=1), dtype=np.int16)
h5_signal_arr = np.array(h5_signal, dtype=np.int16)
err_direct = h5_signal_arr.astype(np.int32) - h5_recon_direct.astype(np.int32)
print(f"Direct roundtrip RMS: {np.sqrt(np.mean(err_direct**2)):.1f}")

# What Mode B DOES: Use sin/cos basis frames with modulation
print("\n--- Mode B: Basis frames + modulation ---")

# Generate sin and cos basis for k=0 (starting at phase 0)
k = 0
k_phase_offset = (k * K_STRIDE * phase_inc) & PHASE_MASK

sin_basis_signal = []
cos_basis_signal = []
phase = k_phase_offset
for n in range(BLOCK_SIZE):
    h_phase = (phase * h) & PHASE_MASK
    sin_val, cos_val = sin_cos_q15(h_phase)
    sin_basis_signal.append(sin_val)
    cos_basis_signal.append(cos_val)
    phase = (phase + phase_inc) & PHASE_MASK

Usin = np.array(fwpt_jit(sin_basis_signal, headroom_bits=1), dtype=np.int16)
Ucos = np.array(fwpt_jit(cos_basis_signal, headroom_bits=1), dtype=np.int16)

print(f"Usin (leaf 0-7): {Usin[:8].tolist()}")
print(f"Ucos (leaf 0-7): {Ucos[:8].tolist()}")

# At phase=0, sin(h*phi)=0, cos(h*phi)=1
# So the output should be: a_h * (cos(0)*Usin + sin(0)*Ucos) = a_h * Usin
# But we want sin(h*theta), so we need Usin only

# For pure sine starting at phase 0:
#   x(t) = sin(h*theta) where theta starts at 0
# Mode B computes: w_sin * Usin + w_cos * Ucos
#   where w_sin = a_h * cos(h*phi) and w_cos = a_h * sin(h*phi)
#
# At runtime phase phi=0:
#   sin(h*phi) = 0, cos(h*phi) = 1
#   w_sin = a_h * 1 = a_h
#   w_cos = a_h * 0 = 0
#   result = a_h * Usin
#
# But Usin = FWPT(sin(h*theta)) where theta starts at 0
# So result should equal FWPT(a_h * sin(h*theta)) = a_h * h5_coeffs_direct

# Let's verify
a_h = Q15_ONE
phi = 0
h_phase = (phi * h) & PHASE_MASK
sin_phi, cos_phi = sin_cos_q15(h_phase)

print(f"\nAt phi=0: sin(h*phi)={sin_phi}, cos(h*phi)={cos_phi}")

w_sin = RSHIFT_RNE(a_h * cos_phi, 15)
w_cos = RSHIFT_RNE(a_h * sin_phi, 15)
print(f"w_sin={w_sin}, w_cos={w_cos}")

# Compute Mode B coefficients
modeb_coeffs = np.zeros(NUM_LEAVES, dtype=np.int16)
for leaf in range(NUM_LEAVES):
    sin_contrib = RSHIFT_RNE(w_sin * int(Usin[leaf]), 15)
    cos_contrib = RSHIFT_RNE(w_cos * int(Ucos[leaf]), 15)
    modeb_coeffs[leaf] = SAT16(sin_contrib + cos_contrib)

print(f"\nMode B coeffs (leaf 0-7): {modeb_coeffs[:8].tolist()}")
print(f"Direct coeffs (leaf 0-7): {h5_coeffs_direct[:8].tolist()}")

# Compare
coeff_diff = h5_coeffs_direct.astype(np.int32) - modeb_coeffs.astype(np.int32)
print(f"\nCoefficient difference (Mode B vs Direct):")
print(f"  RMS: {np.sqrt(np.mean(coeff_diff**2)):.1f}")
print(f"  Max: {np.max(np.abs(coeff_diff))}")

# Reconstruct Mode B and compare to expected signal
modeb_recon = np.array(idwpt_jit(modeb_coeffs.tolist(), headroom_bits=1), dtype=np.int16)
err_modeb = h5_signal_arr.astype(np.int32) - modeb_recon.astype(np.int32)
print(f"\nMode B reconstruction vs expected signal:")
print(f"  RMS: {np.sqrt(np.mean(err_modeb**2)):.1f}")
print(f"  Max: {np.max(np.abs(err_modeb))}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
if np.max(np.abs(coeff_diff)) < 10:
    print("Mode B coefficients MATCH direct FWPT (within rounding)")
    print("The basis frames are correct.")
else:
    print("Mode B coefficients DO NOT MATCH direct FWPT")
    print("There may be an issue with basis frame generation or modulation.")
