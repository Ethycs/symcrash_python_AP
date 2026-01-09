"""
Debug Mode B frequency mapping.

Tests whether Mode B generates the correct fundamental frequency
by directly checking basis frame math and IDWPT output.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq

from symcrash_python_ap.vm.harmonic_bank import (
    generate_basis_frames, mode_b_generate_coeffs, ModeBState, HarmonicWeights,
    BLOCK_SIZE, SAMPLE_RATE, VM_K, NUM_LEAVES
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, PHASE_MASK

def test_pure_sine_440hz():
    """Test pure sine wave at 440 Hz."""
    print("=" * 60)
    print("TEST: Pure Sine at 440 Hz")
    print("=" * 60)

    # Target: A4 = 440 Hz, MIDI note 69
    note = 69
    freq_hz = 440.0
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

    print(f"Target frequency: {freq_hz} Hz")
    print(f"MIDI note: {note}")
    print(f"Phase inc: {phase_inc} (Q32)")

    # Convert phase_inc back to frequency to verify
    freq_verify = (phase_inc / (2**32)) * SAMPLE_RATE
    print(f"Phase inc -> frequency: {freq_verify:.2f} Hz")

    # Generate basis frames for note 69
    basis = generate_basis_frames(anchor_note=note, H_max=8)
    print(f"\nBasis frames generated for anchor note {note} ({basis.anchor_freq_hz:.2f} Hz)")

    # Create Mode B state
    mb_state = ModeBState()
    mb_state.reset(note=note, phase_inc=phase_inc)
    mb_state.weights.set_sine()  # Pure sine: H1 only
    mb_state.H_active = 1

    print(f"H_active: {mb_state.H_active}")
    print(f"Harmonic weights (a_h): {mb_state.weights.a_h_q15[:4].tolist()}")

    # Generate several blocks and IDWPT reconstruct
    all_samples = []

    for block_idx in range(20):
        # Generate coefficients
        carrier_512 = np.zeros(512, dtype=np.int16)
        mode_b_generate_coeffs(mb_state, basis, carrier_512)

        # IDWPT reconstruct each k-slot
        block_samples = []
        for k in range(VM_K):
            # Extract k-slot coefficients (every 8th value starting at k)
            coeffs_64 = carrier_512[k::VM_K].tolist()
            samples = idwpt_jit(coeffs_64)
            block_samples.extend(samples)

        all_samples.extend(block_samples[:BLOCK_SIZE])

        # Advance phase
        mb_state.advance_block()

    samples = np.array(all_samples, dtype=np.float64)

    # FFT analysis
    print(f"\n=== FFT Analysis ===")
    spectrum = np.abs(rfft(samples))
    freqs = rfftfreq(len(samples), 1/SAMPLE_RATE)

    # Find peaks
    threshold = np.max(spectrum) * 0.1
    peaks = []
    for i in range(1, len(spectrum)-1):
        if spectrum[i] > threshold and spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
            peaks.append((freqs[i], spectrum[i]))

    peaks.sort(key=lambda x: -x[1])
    print("Top frequency components:")
    for freq, amp in peaks[:5]:
        print(f"  {freq:.1f} Hz (amplitude: {amp:.0f})")

    # Check if 440 Hz is present
    idx_440 = int(440 * len(samples) / SAMPLE_RATE)
    amp_440 = spectrum[idx_440] if idx_440 < len(spectrum) else 0
    print(f"\nAmplitude at exactly 440 Hz bin: {amp_440:.0f}")

    # Check fundamental
    if peaks:
        fund_freq = peaks[0][0]
        print(f"\n*** Detected fundamental: {fund_freq:.1f} Hz ***")
        if abs(fund_freq - 440) < 10:
            print("SUCCESS: Fundamental matches 440 Hz!")
        else:
            print(f"ERROR: Expected 440 Hz, got {fund_freq:.1f} Hz")
            print(f"       Ratio: {fund_freq / 440:.3f}")


def test_basis_frame_structure():
    """Examine the basis frame structure to understand leaf mapping."""
    print("\n" + "=" * 60)
    print("TEST: Basis Frame Structure Analysis")
    print("=" * 60)

    note = 69
    basis = generate_basis_frames(anchor_note=note, H_max=8)

    print(f"Anchor note: {note}")
    print(f"Anchor freq: {basis.anchor_freq_hz:.2f} Hz")
    print(f"H_max: {basis.H_max}")

    # Check which leaves have energy for H1 sine
    print(f"\n=== H1 Sine (Usin_h[0]) Energy by Leaf ===")
    print("Leaf | Energy | k=0 | k=1 | k=2 | k=3")
    print("-" * 50)

    for leaf in range(64):
        row = basis.Usin_h[0, leaf, :]  # H1 sine, all k-slots
        energy = np.sum(row.astype(np.float64)**2)
        if energy > 1e6:  # Only show significant leaves
            print(f"{leaf:4d} | {energy:8.0f} | {row[0]:6d} | {row[1]:6d} | {row[2]:6d} | {row[3]:6d}")

    # Expected: 440 Hz should map to leaf ~1 (375-750 Hz band at 48kHz/64 leaves)
    # Leaf bandwidth = 24000 / 64 = 375 Hz
    # 440 Hz is in band 1 (375-750 Hz)
    expected_leaf = int(440 / (SAMPLE_RATE / 2 / NUM_LEAVES))
    print(f"\nExpected leaf for 440 Hz: ~{expected_leaf} (leaf BW = {SAMPLE_RATE/2/NUM_LEAVES:.1f} Hz)")


def test_idwpt_directly():
    """Test IDWPT directly with known coefficients."""
    print("\n" + "=" * 60)
    print("TEST: Direct IDWPT Verification")
    print("=" * 60)

    # Create coefficients that SHOULD produce 440 Hz
    # At 48kHz, 440 Hz = leaf 1 (375-750 Hz band)

    # Method 1: Put energy only in leaf 1
    coeffs = [0] * 64
    coeffs[1] = 16384  # Put amplitude in leaf 1 only

    samples = idwpt_jit(coeffs)
    samples_arr = np.array(samples, dtype=np.float64)

    # FFT
    spectrum = np.abs(rfft(samples_arr))
    freqs = rfftfreq(len(samples_arr), 1/SAMPLE_RATE)

    peak_idx = np.argmax(spectrum[1:]) + 1  # Skip DC
    peak_freq = freqs[peak_idx]

    print(f"Leaf 1 only -> Peak frequency: {peak_freq:.1f} Hz")

    # What frequency should leaf 1 contain?
    # Wavelet leaves are NOT direct frequency bins!
    # The frequency content depends on the wavelet decomposition structure

    # Method 2: Try different leaves
    print("\nLeaf scan (which leaf produces what frequency):")
    for test_leaf in [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 32]:
        coeffs = [0] * 64
        coeffs[test_leaf] = 16384
        samples = idwpt_jit(coeffs)
        samples_arr = np.array(samples, dtype=np.float64)

        spectrum = np.abs(rfft(samples_arr))
        peak_idx = np.argmax(spectrum[1:]) + 1
        peak_freq = freqs[peak_idx] if peak_idx < len(freqs) else 0

        print(f"  Leaf {test_leaf:2d} -> {peak_freq:.1f} Hz")


def test_fwpt_440hz():
    """Generate 440 Hz sine, FWPT it, see which leaves get energy."""
    print("\n" + "=" * 60)
    print("TEST: FWPT of 440 Hz Sine")
    print("=" * 60)

    from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit
    from symcrash_python_ap.wavelet.sine_core import sin_cos_q15

    freq_hz = 440.0
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

    # Generate one block of 440 Hz sine
    samples = []
    phase = 0
    for _ in range(BLOCK_SIZE):
        sin_val, _ = sin_cos_q15(phase)
        samples.append(sin_val)
        phase = (phase + phase_inc) & PHASE_MASK

    # FWPT
    coeffs = fwpt_jit(samples)

    print(f"Generated {BLOCK_SIZE} samples of {freq_hz} Hz sine")
    print(f"Phase inc: {phase_inc}")

    # Find which leaves have energy
    print("\n=== FWPT Coefficient Energy by Leaf ===")
    print("Leaf | Coeff Value | Energy")
    print("-" * 40)

    for leaf, coeff in enumerate(coeffs):
        if abs(coeff) > 100:  # Show significant leaves
            print(f"{leaf:4d} | {coeff:12d} | {coeff**2:12d}")

    # The leaves with energy tell us where 440 Hz maps in wavelet domain
    max_leaf = np.argmax(np.abs(coeffs))
    print(f"\nStrongest leaf: {max_leaf} (coeff = {coeffs[max_leaf]})")


if __name__ == "__main__":
    test_basis_frame_structure()
    test_fwpt_440hz()
    test_idwpt_directly()
    test_pure_sine_440hz()
