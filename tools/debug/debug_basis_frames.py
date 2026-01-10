"""Debug basis frames for H5 - are they encoding the 50 Hz pattern?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq

from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sin_cos_q15, PHASE_MASK
from symcrash_python_ap.vm.harmonic_bank import generate_basis_frames, VM_K, K_STRIDE, NUM_LEAVES, BLOCK_SIZE

SAMPLE_RATE = 48000

print("=" * 70)
print("BASIS FRAMES ANALYSIS")
print(f"VM_K={VM_K}, K_STRIDE={K_STRIDE}")
print("=" * 70)

note = 69  # A4 = 440 Hz
freq_hz = 440.0
phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

basis = generate_basis_frames(note, H_max=8)

print(f"\nBasis shapes:")
print(f"  DC_frame: {basis.DC_frame.shape}")
print(f"  Ucos_h: {basis.Ucos_h.shape}")
print(f"  Usin_h: {basis.Usin_h.shape}")

# The basis frames are STATIC - they don't change per block
# But when we modulate them with runtime phase, we get time-varying coefficients

# Let's trace what happens when we use the basis
h = 5
print(f"\n--- H5 Basis Frame Analysis ---")
print(f"Usin_h[4] (H5 sin basis) for k=0:")
print(f"  First 8 leaves: {basis.Usin_h[4, :8, 0].tolist()}")
print(f"  Range: [{np.min(basis.Usin_h[4, :, 0])}, {np.max(basis.Usin_h[4, :, 0])}]")

print(f"\nUcos_h[4] (H5 cos basis) for k=0:")
print(f"  First 8 leaves: {basis.Ucos_h[4, :8, 0].tolist()}")
print(f"  Range: [{np.min(basis.Ucos_h[4, :, 0])}, {np.max(basis.Ucos_h[4, :, 0])}]")

# Now simulate what Mode B does: modulate basis with runtime phase
print(f"\n--- Simulating Mode B coefficient generation ---")
from symcrash_python_ap.fixed import RSHIFT_RNE, SAT16, Q15_ONE

def generate_coeffs_for_phase(base_phase, h):
    """Generate coefficients for H=h at given phase."""
    coeffs = np.zeros((NUM_LEAVES, VM_K), dtype=np.int16)

    for k in range(VM_K):
        phi = base_phase  # Same phase for all k (current implementation)
        h_phase = (phi * h) & PHASE_MASK
        sin_phi, cos_phi = sin_cos_q15(h_phase)

        # a_h * sin(h*theta + h*phi) using basis
        # w_sin = a_h * cos(h*phi), w_cos = a_h * sin(h*phi)
        a_h = Q15_ONE
        w_sin = RSHIFT_RNE(a_h * cos_phi, 15)
        w_cos = RSHIFT_RNE(a_h * sin_phi, 15)

        for leaf in range(NUM_LEAVES):
            sin_basis = int(basis.Usin_h[h-1, leaf, k])
            cos_basis = int(basis.Ucos_h[h-1, leaf, k])
            sin_contrib = RSHIFT_RNE(w_sin * sin_basis, 15)
            cos_contrib = RSHIFT_RNE(w_cos * cos_basis, 15)
            coeffs[leaf, k] = SAT16(sin_contrib + cos_contrib)

    return coeffs

# Generate coefficients for 100 consecutive blocks
all_coeffs = []
phase = 0
for block_idx in range(100):
    coeffs = generate_coeffs_for_phase(phase, h)
    all_coeffs.append(coeffs[:, 0].copy())  # Just k=0 slot
    phase = (phase + BLOCK_SIZE * phase_inc) & PHASE_MASK

all_coeffs = np.array(all_coeffs)  # [100 blocks, 64 leaves]

# Analyze leaf 0's evolution
leaf0 = all_coeffs[:, 0].astype(np.float64)
n = len(leaf0)
hann = np.hanning(n)
spectrum = np.abs(rfft(leaf0 * hann)) * (2.0 / np.mean(hann))
freqs = rfftfreq(n, BLOCK_SIZE / SAMPLE_RATE)

print(f"\nLeaf 0 coefficient spectrum (Mode B, k=0):")
idx_50 = int(50 * n * BLOCK_SIZE / SAMPLE_RATE)
print(f"  50 Hz component: {spectrum[idx_50]:.0f}")
print(f"  DC component: {spectrum[0]:.0f}")

# Top peaks
sorted_idx = np.argsort(spectrum[1:])[::-1] + 1
print(f"\nTop 5 peaks:")
for i in range(5):
    idx = sorted_idx[i]
    print(f"  {freqs[idx]:.1f} Hz: {spectrum[idx]:.0f}")

# The key insight: the sin_phi/cos_phi modulation IS the source
print(f"\n--- Phase Modulation Analysis ---")
print(f"Phase increment per block: {BLOCK_SIZE * phase_inc}")
print(f"H5 effective phase inc: {BLOCK_SIZE * phase_inc * h}")
print(f"H5 cycles per block: {(BLOCK_SIZE * phase_inc * h) / (2**32):.4f}")
print(f"Fractional part: {((BLOCK_SIZE * phase_inc * h) % (2**32)) / (2**32):.4f}")

# The modulation weights sin_phi, cos_phi are rotating at H5's rate
# When projected onto the basis, this creates the amplitude modulation
print(f"\n--- Weight Evolution ---")
phase = 0
weights_sin = []
weights_cos = []
for block in range(100):
    h_phase = (phase * h) & PHASE_MASK
    sin_phi, cos_phi = sin_cos_q15(h_phase)
    weights_sin.append(sin_phi)
    weights_cos.append(cos_phi)
    phase = (phase + BLOCK_SIZE * phase_inc) & PHASE_MASK

# Analyze weight evolution
weights_sin = np.array(weights_sin, dtype=np.float64)
spectrum_w = np.abs(rfft(weights_sin * hann)) * (2.0 / np.mean(hann))

print(f"sin(h*phi) weight spectrum:")
print(f"  50 Hz: {spectrum_w[idx_50]:.0f}")
sorted_idx_w = np.argsort(spectrum_w[1:])[::-1] + 1
for i in range(3):
    idx = sorted_idx_w[i]
    print(f"  {freqs[idx]:.1f} Hz: {spectrum_w[idx]:.0f}")
