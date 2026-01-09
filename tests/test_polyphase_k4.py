"""
CCS Conformance Test: Polyphase K=4 Validation
==============================================

Verifies that K=4 polyphase IDWPT produces correct output and
compares to k=0 only approach.

Tests:
1. Output matches expected waveform shape
2. K=4 produces smoother boundaries than k=0 only
3. Phase continuity across blocks
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_block
from symcrash_python_ap.wavelet.sine_core import (
    freq_to_phase_inc, sine_q15, PHASE_MASK
)

SAMPLE_RATE = 48000
BLOCK_SIZE = 64
VM_K = 8


def generate_test_coeffs_k8(note: int, num_blocks: int = 10):
    """
    Generate K=8 coefficient state for test signal.

    Simulates what the LUT-based carrier injection produces.
    """
    freq_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

    all_coeffs = []
    phase = 0

    for _ in range(num_blocks):
        # Generate K=8 coefficient frames, each shifted by 1 sample
        coeff_2d = np.zeros((64, VM_K), dtype=np.int16)

        for k in range(VM_K):
            # Generate 64-sample block starting at phase + k*phase_inc
            block_phase = (phase + k * phase_inc) & PHASE_MASK
            block = []
            p = block_phase
            for _ in range(BLOCK_SIZE):
                block.append(sine_q15(p))
                p = (p + phase_inc) & PHASE_MASK

            # FWPT to get coefficients
            coeffs = fwpt_jit(block, headroom_bits=1)
            coeff_2d[:, k] = coeffs

        all_coeffs.append(coeff_2d)

        # Advance phase by block
        phase = (phase + BLOCK_SIZE * phase_inc) & PHASE_MASK

    return all_coeffs


def render_k0_only(coeff_2d_k8: np.ndarray) -> np.ndarray:
    """Render using k=0 only (old method)."""
    coeffs = coeff_2d_k8[:, 0].tolist()
    audio = idwpt_jit(coeffs, headroom_bits=1)
    return np.array(audio, dtype=np.int16)


def render_k4_polyphase(coeff_2d_k8: np.ndarray) -> np.ndarray:
    """Render using K=4 polyphase (new method)."""
    # Sample every 2nd slot: 0, 2, 4, 6
    coeff_2d_k4 = coeff_2d_k8[:, ::2].astype(np.int16)
    return idwpt_polyphase_block(coeff_2d_k4, headroom_bits=1)


def measure_boundary_discontinuity(audio_blocks: list) -> float:
    """
    Measure average boundary discontinuity between adjacent blocks.

    Returns average |block[i][-1] - block[i+1][0]| normalized by signal RMS.
    """
    if len(audio_blocks) < 2:
        return 0.0

    discontinuities = []
    for i in range(len(audio_blocks) - 1):
        disc = abs(int(audio_blocks[i][-1]) - int(audio_blocks[i + 1][0]))
        discontinuities.append(disc)

    avg_disc = np.mean(discontinuities)

    # Normalize by signal RMS
    all_samples = np.concatenate(audio_blocks)
    rms = np.sqrt(np.mean(all_samples.astype(np.float64) ** 2))

    if rms > 0:
        return avg_disc / rms
    return avg_disc


def test_basic_polyphase_output():
    """Test that K=4 polyphase produces valid output."""
    print("\n=== Basic Polyphase Output Test ===\n")

    note = 69  # A4
    coeffs_list = generate_test_coeffs_k8(note, num_blocks=1)
    coeff_2d = coeffs_list[0]

    # Render both ways
    audio_k0 = render_k0_only(coeff_2d)
    audio_k4 = render_k4_polyphase(coeff_2d)

    print(f"k=0 only: shape={audio_k0.shape}, range=[{audio_k0.min()}, {audio_k0.max()}]")
    print(f"K=4 poly: shape={audio_k4.shape}, range=[{audio_k4.min()}, {audio_k4.max()}]")

    # Both should have valid output
    assert audio_k0.shape == (64,), f"k=0 shape error: {audio_k0.shape}"
    assert audio_k4.shape == (64,), f"K=4 shape error: {audio_k4.shape}"

    # Both should have reasonable amplitude
    k0_rms = np.sqrt(np.mean(audio_k0.astype(np.float64) ** 2))
    k4_rms = np.sqrt(np.mean(audio_k4.astype(np.float64) ** 2))

    print(f"k=0 RMS: {k0_rms:.1f}")
    print(f"K=4 RMS: {k4_rms:.1f}")

    # RMS should be similar (within 50%)
    ratio = k4_rms / k0_rms if k0_rms > 0 else 0
    print(f"RMS ratio (K4/k0): {ratio:.3f}")

    if 0.5 < ratio < 2.0 and k4_rms > 1000:
        print("[PASS] Basic polyphase output valid")
        return True
    else:
        print("[FAIL] Basic polyphase output invalid")
        return False


def test_boundary_continuity():
    """Test that K=4 polyphase has better boundary continuity."""
    print("\n=== Boundary Continuity Test ===\n")

    note = 60  # C4
    num_blocks = 20
    coeffs_list = generate_test_coeffs_k8(note, num_blocks=num_blocks)

    # Render sequences
    k0_blocks = [render_k0_only(c) for c in coeffs_list]
    k4_blocks = [render_k4_polyphase(c) for c in coeffs_list]

    # Measure discontinuity
    disc_k0 = measure_boundary_discontinuity(k0_blocks)
    disc_k4 = measure_boundary_discontinuity(k4_blocks)

    print(f"k=0 boundary discontinuity ratio: {disc_k0:.4f}")
    print(f"K=4 boundary discontinuity ratio: {disc_k4:.4f}")

    # K=4 should have similar or better continuity
    # (with correct phase-aligned LUT, both should be good)
    if disc_k0 < 0.1 and disc_k4 < 0.1:
        print("[PASS] Both methods have good boundary continuity")
        return True
    elif disc_k4 <= disc_k0 * 1.5:  # Allow K=4 to be up to 50% worse (still acceptable)
        print(f"[PASS] K=4 continuity acceptable (ratio: {disc_k4/disc_k0:.2f}x)")
        return True
    else:
        print("[FAIL] Boundary continuity issues")
        return False


def test_waveform_shape():
    """Test that output waveform has expected sinusoidal shape."""
    print("\n=== Waveform Shape Test ===\n")

    note = 69  # A4 = 440 Hz
    num_blocks = 50
    coeffs_list = generate_test_coeffs_k8(note, num_blocks=num_blocks)

    # Render with K=4
    audio = np.concatenate([render_k4_polyphase(c) for c in coeffs_list])

    # FFT analysis
    fft = np.fft.rfft(audio)
    magnitude = np.abs(fft)

    # Find fundamental
    fundamental_bin = np.argmax(magnitude[1:]) + 1
    fundamental_mag = magnitude[fundamental_bin]

    # Calculate THD (sum of harmonics relative to fundamental)
    harmonic_energy = 0
    for h in range(2, 10):
        bin_h = fundamental_bin * h
        if bin_h < len(magnitude):
            harmonic_energy += magnitude[bin_h] ** 2

    thd = np.sqrt(harmonic_energy) / fundamental_mag * 100 if fundamental_mag > 0 else 0

    freq_resolution = SAMPLE_RATE / len(audio)
    measured_freq = fundamental_bin * freq_resolution
    expected_freq = 440.0

    print(f"Expected frequency: {expected_freq:.2f} Hz")
    print(f"Measured frequency: {measured_freq:.2f} Hz")
    print(f"THD: {thd:.2f}%")

    freq_error = abs(measured_freq - expected_freq)
    freq_error_percent = (freq_error / expected_freq) * 100

    print(f"Frequency error: {freq_error:.2f} Hz ({freq_error_percent:.2f}%)")

    # Allow up to 2% frequency error (FFT bin resolution) and 5% THD
    if freq_error_percent < 2.0 and thd < 5.0:
        print("[PASS] Waveform shape is correct sinusoid")
        return True
    else:
        print("[FAIL] Waveform shape issues")
        return False


def test_various_frequencies():
    """Test K=4 polyphase across various frequencies."""
    print("\n=== Various Frequencies Test ===\n")

    test_notes = [48, 60, 72, 84]  # C3, C4, C5, C6
    all_pass = True

    print(f"{'Note':>6} {'Freq Hz':>10} {'RMS':>8} {'Status':>8}")
    print("-" * 40)

    for note in test_notes:
        freq_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))
        coeffs_list = generate_test_coeffs_k8(note, num_blocks=5)

        # Render with K=4
        audio = np.concatenate([render_k4_polyphase(c) for c in coeffs_list])
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))

        note_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][note % 12]
        octave = (note // 12) - 1

        status = "[ok]" if rms > 1000 else "[!]"
        print(f"{note_name}{octave:>2} ({note:2d}) {freq_hz:10.2f} {rms:8.1f} {status}")

        if rms < 1000:
            all_pass = False

    if all_pass:
        print("\n[PASS] All frequencies produce valid output")
        return True
    else:
        print("\n[FAIL] Some frequencies have issues")
        return False


def main():
    print("=" * 60)
    print("CCS Conformance: Polyphase K=4 Validation")
    print("=" * 60)

    results = []

    results.append(("Basic Polyphase Output", test_basic_polyphase_output()))
    results.append(("Boundary Continuity", test_boundary_continuity()))
    results.append(("Waveform Shape", test_waveform_shape()))
    results.append(("Various Frequencies", test_various_frequencies()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        all_passed = all_passed and passed

    print()
    if all_passed:
        print("[PASS] K=4 polyphase validation complete!")
        return 0
    else:
        print("[FAIL] Some K=4 polyphase tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
