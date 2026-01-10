"""Debug the theory behind interleaved polyphase reconstruction."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sin_cos_q15, PHASE_MASK
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.vm.harmonic_bank import VM_K, K_STRIDE, NUM_LEAVES, BLOCK_SIZE
from symcrash_python_ap.fixed import Q15_ONE

SAMPLE_RATE = 48000

print("=" * 70)
print("INTERLEAVED RECONSTRUCTION THEORY")
print(f"VM_K={VM_K}, K_STRIDE={K_STRIDE}")
print("=" * 70)

note = 69
freq_hz = 440.0
phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)
h = 5  # H5 = 2200 Hz

# Generate expected signal
print("\n--- Generate expected H5 signal ---")
expected = []
phase = 0
for n in range(BLOCK_SIZE):
    h_phase = (phase * h) & PHASE_MASK
    sin_val, _ = sin_cos_q15(h_phase)
    expected.append(sin_val)
    phase = (phase + phase_inc) & PHASE_MASK

expected = np.array(expected, dtype=np.int16)
print(f"Expected signal: first 8 = {expected[:8].tolist()}")
print(f"Expected signal: samples [0,4,8,12] = {expected[0::4][:4].tolist()}")

# Direct FWPT -> IDWPT roundtrip (should be near-perfect)
print("\n--- Direct FWPT -> IDWPT roundtrip ---")
coeffs_direct = np.array(fwpt_jit(expected.tolist(), headroom_bits=1), dtype=np.int16)
recon_direct = np.array(idwpt_jit(coeffs_direct.tolist(), headroom_bits=1), dtype=np.int16)
err_direct = expected.astype(np.int32) - recon_direct.astype(np.int32)
print(f"Direct roundtrip RMS error: {np.sqrt(np.mean(err_direct**2)):.1f}")
print(f"Direct roundtrip max error: {np.max(np.abs(err_direct))}")

# Polyphase approach: what SHOULD happen
print("\n--- Polyphase Interleave (K_STRIDE=1) ---")
print("With K_STRIDE=1, each k-slot represents 1-sample offset:")
print("  k=0: samples at phase offset 0")
print("  k=1: samples at phase offset 1*phase_inc")
print("  k=2: samples at phase offset 2*phase_inc")
print("  k=3: samples at phase offset 3*phase_inc")

# Generate each k-slot's signal (offset by k samples)
k_signals = []
for k in range(VM_K):
    k_signal = []
    phase = (k * phase_inc) & PHASE_MASK  # Offset by k samples
    for n in range(BLOCK_SIZE):
        h_phase = (phase * h) & PHASE_MASK
        sin_val, _ = sin_cos_q15(h_phase)
        k_signal.append(sin_val)
        phase = (phase + phase_inc) & PHASE_MASK
    k_signals.append(np.array(k_signal, dtype=np.int16))

print(f"\nk=0 signal samples [0,4,8,12]: {k_signals[0][0::4][:4].tolist()}")
print(f"k=1 signal samples [0,4,8,12]: {k_signals[1][0::4][:4].tolist()}")

# FWPT each k-slot
k_coeffs = []
for k in range(VM_K):
    coeffs = np.array(fwpt_jit(k_signals[k].tolist(), headroom_bits=1), dtype=np.int16)
    k_coeffs.append(coeffs)

# IDWPT each k-slot and interleave
k_recons = []
for k in range(VM_K):
    recon = np.array(idwpt_jit(k_coeffs[k].tolist(), headroom_bits=1), dtype=np.int16)
    k_recons.append(recon)

# Interleave: out[k::K] = recon_k[0::K]
output_interleaved = np.zeros(BLOCK_SIZE, dtype=np.int16)
for k in range(VM_K):
    output_interleaved[k::VM_K] = k_recons[k][0::VM_K]

print(f"\nInterleaved output samples [0,1,2,3]: {output_interleaved[:4].tolist()}")
print(f"Expected samples [0,1,2,3]: {expected[:4].tolist()}")

err_interleaved = expected.astype(np.int32) - output_interleaved.astype(np.int32)
print(f"\nInterleaved RMS error: {np.sqrt(np.mean(err_interleaved**2)):.1f}")
print(f"Interleaved max error: {np.max(np.abs(err_interleaved))}")

# Now the key question: what does Mode B generate?
print("\n--- Mode B Coefficient Generation ---")
print("Mode B uses sin/cos phase modulation on basis frames.")
print("The question: does the coefficient evolution match what")
print("the polyphase interleave expects?")

# What Mode B generates for one block
from symcrash_python_ap.vm.harmonic_bank import (
    generate_basis_frames, ModeBState, mode_b_generate_coeffs
)
from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_block

basis = generate_basis_frames(note, H_max=8)

mb_state = ModeBState()
mb_state.reset(note=note, phase_inc=phase_inc)
mb_state.weights.a_h_q15.fill(0)
mb_state.weights.a_h_q15[h-1] = Q15_ONE  # H5 only
mb_state.H_active = h

coeffs_modeb = np.zeros(NUM_LEAVES * VM_K, dtype=np.int16)
mode_b_generate_coeffs(mb_state, basis, coeffs_modeb)
coeffs_modeb_2d = coeffs_modeb.reshape(NUM_LEAVES, VM_K)

# Compare Mode B coefficients with polyphase coefficients
print("\nCoefficient comparison (leaf 0):")
print(f"  Polyphase k-slot coeffs: {[k_coeffs[k][0] for k in range(VM_K)]}")
print(f"  Mode B k-slot coeffs:    {coeffs_modeb_2d[0, :].tolist()}")

# IDWPT the Mode B coefficients
output_modeb = idwpt_polyphase_block(coeffs_modeb_2d, headroom_bits=1)

err_modeb = expected.astype(np.int32) - output_modeb.astype(np.int32)
print(f"\nMode B RMS error: {np.sqrt(np.mean(err_modeb**2)):.1f}")
print(f"Mode B max error: {np.max(np.abs(err_modeb))}")

print("\n--- Summary ---")
print(f"Direct roundtrip:       RMS = {np.sqrt(np.mean(err_direct**2)):.1f}")
print(f"Polyphase interleave:   RMS = {np.sqrt(np.mean(err_interleaved**2)):.1f}")
print(f"Mode B synthesis:       RMS = {np.sqrt(np.mean(err_modeb**2)):.1f}")
