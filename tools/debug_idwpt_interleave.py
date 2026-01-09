"""
Debug IDWPT interleave reconstruction.

Trace through exactly how shifted-block reconstruction should work.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq

from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.wavelet.sine_core import sin_cos_q15, freq_to_phase_inc, PHASE_MASK

BLOCK_SIZE = 64
SAMPLE_RATE = 48000
VM_K = 8  # Mode B uses K=8


def generate_reference_sine(freq_hz, num_samples, start_sample=0):
    """Generate reference sine wave samples."""
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)
    samples = []
    phase = (start_sample * phase_inc) & PHASE_MASK
    for _ in range(num_samples):
        sin_val, _ = sin_cos_q15(phase)
        samples.append(sin_val)
        phase = (phase + phase_inc) & PHASE_MASK
    return samples


def test_shifted_block_roundtrip():
    """Test FWPT/IDWPT roundtrip with shifted blocks."""
    print("=" * 60)
    print("TEST: Shifted-Block FWPT/IDWPT Roundtrip")
    print("=" * 60)

    freq_hz = 440.0
    print(f"Generating 440 Hz sine")

    # Generate enough samples to fill multiple blocks
    total_samples = 128  # A bit more than one block

    # Method: Shifted-block FWPT
    # Block k uses samples [k : k + 64]
    # This creates 8 overlapping 64-sample blocks

    K = 8  # Number of k-slots

    # Generate 64+K-1 samples (enough for K shifted 64-sample blocks)
    samples = generate_reference_sine(freq_hz, 64 + K - 1)
    print(f"Generated {len(samples)} samples")
    print(f"First 10 samples: {samples[:10]}")

    # FWPT each shifted block
    coeffs_2d = []  # [K, 64]
    for k in range(K):
        block = samples[k:k+64]
        coeffs = fwpt_jit(block)
        coeffs_2d.append(coeffs)

    coeffs_2d = np.array(coeffs_2d)  # Shape: [K, 64]
    print(f"\nFWPT coefficients shape: {coeffs_2d.shape}")

    # Now reconstruct using polyphase interleave
    # The claim is: to get sample n, use y_k[m] where k = n mod K, m = n // K

    # Method A: Use k-slot k, take sample at index (n // K) where n mod K == k
    # This means: out[k + m*K] = y_k[m] for m = 0, 1, ..., 64//K - 1
    # i.e., out[k::K] = y_k[0:64//K]

    out_A = np.zeros(64, dtype=np.int16)
    for k in range(K):
        coeffs_k = coeffs_2d[k].tolist()
        y_k = idwpt_jit(coeffs_k)
        # Take first 64//K samples from y_k, place at positions k, k+K, k+2K, ...
        out_A[k::K] = y_k[:64//K]

    # Method B: Current implementation - out[k::K] = y_k[0::K]
    out_B = np.zeros(64, dtype=np.int16)
    for k in range(K):
        coeffs_k = coeffs_2d[k].tolist()
        y_k = idwpt_jit(coeffs_k)
        out_B[k::K] = y_k[0::K]

    # Reference: just the first 64 samples
    ref = np.array(samples[:64], dtype=np.int16)

    print(f"\nFirst 16 samples comparison:")
    print(f"{'n':>3} | {'Ref':>8} | {'Method A':>8} | {'Method B':>8}")
    print("-" * 40)
    for n in range(16):
        print(f"{n:3d} | {ref[n]:8d} | {out_A[n]:8d} | {out_B[n]:8d}")

    # Correlation
    corr_A = np.corrcoef(ref.astype(float), out_A.astype(float))[0, 1]
    corr_B = np.corrcoef(ref.astype(float), out_B.astype(float))[0, 1]
    print(f"\nCorrelation with reference:")
    print(f"  Method A: {corr_A:.6f}")
    print(f"  Method B: {corr_B:.6f}")

    # FFT of reconstructed
    print(f"\n=== FFT Analysis ===")
    spectrum_ref = np.abs(rfft(ref.astype(float)))
    spectrum_A = np.abs(rfft(out_A.astype(float)))
    spectrum_B = np.abs(rfft(out_B.astype(float)))
    freqs = rfftfreq(64, 1/SAMPLE_RATE)

    # Find peaks
    def find_peak(spectrum):
        peak_idx = np.argmax(spectrum[1:]) + 1
        return freqs[peak_idx]

    print(f"Reference peak: {find_peak(spectrum_ref):.1f} Hz")
    print(f"Method A peak:  {find_peak(spectrum_A):.1f} Hz")
    print(f"Method B peak:  {find_peak(spectrum_B):.1f} Hz")


def test_mode_b_alignment():
    """Test that Mode B basis generation aligns with shifted-block model."""
    print("\n" + "=" * 60)
    print("TEST: Mode B Basis Generation Alignment")
    print("=" * 60)

    from symcrash_python_ap.vm.harmonic_bank import generate_basis_frames

    freq_hz = 440.0
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

    # Generate basis frames for A4
    basis = generate_basis_frames(anchor_note=69, H_max=8)

    print(f"Basis anchor: {basis.anchor_note} ({basis.anchor_freq_hz:.2f} Hz)")

    # The basis Usin_h[h-1, leaf, k] should reconstruct to sin(h * (theta + k*phase_inc))
    # For H=1 (fundamental), this should give sin(theta + k*phase_inc)

    # If we weight Usin_h[0] with a_1=1 (pure sine), the output coefficients
    # should, when passed through IDWPT, give a 440 Hz sine starting at phase 0

    # Let's manually compute what the output should be:
    # At block start, phase = 0
    # coeffs[:, k] = a_1 * Usin_h[0, :, k] (since we're using sin_phi=0, cos_phi=1 at phase=0)
    # Wait, that's not right. Let me re-read the mode_b_generate_coeffs math.

    # mode_b_generate_coeffs does:
    #   h_phase = (phi * h) & PHASE_MASK
    #   sin_phi, cos_phi = sin_cos_q15(h_phase)
    #   w_sin = a_h*cos(hphi) - b_h*sin(hphi)
    #   w_cos = a_h*sin(hphi) + b_h*cos(hphi)
    #   coeff = w_sin * Usin + w_cos * Ucos

    # For pure sine (a_1=1, b_1=0) at phase=0:
    #   h_phase = 0
    #   sin_phi = 0, cos_phi = 32767
    #   w_sin = a_1 * 32767 / 32768 = 1
    #   w_cos = a_1 * 0 = 0
    #   coeff = 1 * Usin

    # So at phase=0, we get just the Usin basis frame
    # The Usin basis was computed from sin(h*theta) where theta ranges over the block
    # For k-slot k, the block starts at phase = k * phase_inc

    # Let's verify by computing expected IDWPT output for k=0
    coeffs_k0 = basis.Usin_h[0, :, 0]  # H1 sine, all leaves, k=0
    y_k0 = idwpt_jit(coeffs_k0.tolist())

    # This should reconstruct sin(phase) for phase = 0, phase_inc, 2*phase_inc, ...
    expected_k0 = generate_reference_sine(freq_hz, 64, start_sample=0)

    print(f"\nk=0 reconstruction vs reference:")
    print(f"{'n':>3} | {'IDWPT':>8} | {'Ref':>8} | {'Diff':>8}")
    print("-" * 40)
    for n in range(16):
        diff = y_k0[n] - expected_k0[n]
        print(f"{n:3d} | {y_k0[n]:8d} | {expected_k0[n]:8d} | {diff:8d}")

    corr = np.corrcoef(np.array(y_k0).astype(float), np.array(expected_k0).astype(float))[0, 1]
    print(f"\nCorrelation: {corr:.6f}")

    # FFT to check frequency
    spectrum = np.abs(rfft(np.array(y_k0).astype(float)))
    freqs = rfftfreq(64, 1/SAMPLE_RATE)
    peak_idx = np.argmax(spectrum[1:]) + 1
    print(f"IDWPT peak frequency: {freqs[peak_idx]:.1f} Hz (expected: {freq_hz:.1f} Hz)")


def test_simple_idwpt():
    """Simple test: FWPT a sine, IDWPT back, check frequency."""
    print("\n" + "=" * 60)
    print("TEST: Simple FWPT/IDWPT Roundtrip")
    print("=" * 60)

    freq_hz = 440.0
    samples = generate_reference_sine(freq_hz, 64)

    print(f"Input: 440 Hz sine, 64 samples")
    print(f"First 10 samples: {samples[:10]}")

    # FWPT
    coeffs = fwpt_jit(samples)
    print(f"\nFWPT coefficients (non-zero):")
    for i, c in enumerate(coeffs):
        if abs(c) > 100:
            print(f"  leaf {i}: {c}")

    # IDWPT
    reconstructed = idwpt_jit(coeffs)

    # Compare
    ref = np.array(samples, dtype=np.int16)
    rec = np.array(reconstructed, dtype=np.int16)

    print(f"\nReconstruction comparison:")
    print(f"{'n':>3} | {'Original':>8} | {'Reconstructed':>13}")
    print("-" * 35)
    for n in range(10):
        print(f"{n:3d} | {ref[n]:8d} | {rec[n]:13d}")

    corr = np.corrcoef(ref.astype(float), rec.astype(float))[0, 1]
    print(f"\nCorrelation: {corr:.6f}")

    # FFT
    spectrum = np.abs(rfft(rec.astype(float)))
    freqs = rfftfreq(64, 1/SAMPLE_RATE)
    peak_idx = np.argmax(spectrum[1:]) + 1
    print(f"Reconstructed peak frequency: {freqs[peak_idx]:.1f} Hz")


if __name__ == "__main__":
    test_simple_idwpt()
    test_mode_b_alignment()
    test_shifted_block_roundtrip()
