"""
CCS Conformance Test: Frequency Validation
==========================================

Verifies that Mode A and Mode B produce correct pitch for each MIDI note.

Test approach:
1. For each MIDI note 48-72:
   - Generate 1 second of audio (or enough cycles for accurate FFT)
   - FFT to find fundamental frequency
   - Compare to expected frequency
   - Error should be < 1 cent

1 cent = 1/100 semitone = (2^(1/1200) - 1) ≈ 0.0578% frequency error
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from symcrash_python_ap.fixed import midi_to_phase_inc_fixed, midi_to_freq_hz
from symcrash_python_ap.wavelet.sine_core import (
    freq_to_phase_inc, midi_to_freq, PHASE_MASK, sine_q15
)
from symcrash_python_ap.vm.harmonic_bank import (
    ModeBState, HarmonicWeights, get_basis_for_note,
    mode_b_generate_coeffs, mode_b_advance_phase, BLOCK_SIZE, VM_K
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit

SAMPLE_RATE = 48000


def freq_to_cents_error(measured_hz: float, expected_hz: float) -> float:
    """Convert frequency error to cents."""
    if expected_hz <= 0 or measured_hz <= 0:
        return float('inf')
    ratio = measured_hz / expected_hz
    return 1200 * np.log2(ratio)


def measure_fundamental_fft(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """
    Measure fundamental frequency using FFT with parabolic interpolation.

    Returns frequency in Hz.
    """
    # Window for cleaner FFT
    window = np.hanning(len(samples))
    windowed = samples * window

    # FFT
    fft = np.fft.rfft(windowed)
    magnitude = np.abs(fft)

    # Find peak (skip DC bin)
    peak_bin = np.argmax(magnitude[1:]) + 1

    # Parabolic interpolation for sub-bin precision
    if peak_bin > 0 and peak_bin < len(magnitude) - 1:
        alpha = magnitude[peak_bin - 1]
        beta = magnitude[peak_bin]
        gamma = magnitude[peak_bin + 1]

        if 2 * beta - alpha - gamma != 0:
            p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
            refined_bin = peak_bin + p
        else:
            refined_bin = peak_bin
    else:
        refined_bin = peak_bin

    # Convert bin to frequency
    freq_resolution = sample_rate / len(samples)
    return refined_bin * freq_resolution


def generate_direct_sine(note: int, num_samples: int) -> np.ndarray:
    """Generate pure sine wave using direct SineCore (Mode A reference)."""
    freq_hz = midi_to_freq(note)
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

    samples = np.zeros(num_samples, dtype=np.int16)
    phase = 0

    for i in range(num_samples):
        samples[i] = sine_q15(phase)
        phase = (phase + phase_inc) & PHASE_MASK

    return samples


def generate_mode_b_audio(note: int, num_samples: int) -> np.ndarray:
    """Generate audio using Mode B harmonic bank."""
    freq_hz = midi_to_freq(note)
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

    # Initialize Mode B state
    state = ModeBState()
    state.reset(note, phase_inc)
    state.weights.set_sine()  # Pure sine for frequency test
    state.H_active = 1

    # Get basis frames for this note
    basis = get_basis_for_note(note)

    # Generate blocks
    num_blocks = (num_samples + BLOCK_SIZE - 1) // BLOCK_SIZE
    all_samples = []

    coeffs = np.zeros(512, dtype=np.int16)

    for _ in range(num_blocks):
        # Generate coefficients
        mode_b_generate_coeffs(state, basis, coeffs)

        # IDWPT for k=0 only (basic test)
        coeff_slice = coeffs[0::VM_K]  # k=0 coefficients
        time_samples = idwpt_jit(list(coeff_slice))
        all_samples.extend(time_samples[:BLOCK_SIZE])

        # Advance phase
        mode_b_advance_phase(state)

    return np.array(all_samples[:num_samples], dtype=np.int16)


def test_pitch_table_accuracy():
    """Test that pitch table produces correct frequencies."""
    print("\n=== Pitch Table Accuracy Test ===\n")

    max_error_cents = 0
    errors = []

    for note in range(48, 73):
        expected_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))

        # Get phase increment from fixed-point LUT
        phase_inc = midi_to_phase_inc_fixed(note)

        # Convert back to frequency
        measured_hz = (phase_inc * SAMPLE_RATE) / (2**32)

        # Calculate error
        error_cents = freq_to_cents_error(measured_hz, expected_hz)
        errors.append((note, expected_hz, measured_hz, error_cents))
        max_error_cents = max(max_error_cents, abs(error_cents))

    # Print results
    print(f"{'Note':>4} {'Expected Hz':>12} {'Measured Hz':>12} {'Error (cents)':>14}")
    print("-" * 48)

    for note, expected, measured, error in errors:
        note_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][note % 12]
        octave = (note // 12) - 1
        status = "[ok]" if abs(error) < 1.0 else "[!]"
        print(f"{note_name}{octave:>2} ({note:3d}) {expected:12.4f} {measured:12.4f} {error:+14.6f} {status}")

    print(f"\nMax error: {max_error_cents:.6f} cents")

    if max_error_cents < 1.0:
        print("[PASS] All notes within 1 cent tolerance")
        return True
    else:
        print("[FAIL] Some notes exceed 1 cent tolerance")
        return False


def test_direct_sine_frequency():
    """Test that direct SineCore produces correct frequency."""
    print("\n=== Direct SineCore Frequency Test ===\n")

    # Test subset of notes
    test_notes = [48, 60, 69, 72]  # C3, C4, A4, C5
    num_samples = SAMPLE_RATE  # 1 second

    max_error_cents = 0

    print(f"{'Note':>4} {'Expected Hz':>12} {'Measured Hz':>12} {'Error (cents)':>14}")
    print("-" * 48)

    for note in test_notes:
        expected_hz = midi_to_freq(note)

        # Generate audio
        samples = generate_direct_sine(note, num_samples)

        # Measure frequency
        measured_hz = measure_fundamental_fft(samples.astype(np.float64), SAMPLE_RATE)

        # Calculate error
        error_cents = freq_to_cents_error(measured_hz, expected_hz)
        max_error_cents = max(max_error_cents, abs(error_cents))

        note_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][note % 12]
        octave = (note // 12) - 1
        status = "[ok]" if abs(error_cents) < 1.0 else "[!]"
        print(f"{note_name}{octave:>2} ({note:3d}) {expected_hz:12.4f} {measured_hz:12.4f} {error_cents:+14.6f} {status}")

    print(f"\nMax error: {max_error_cents:.6f} cents")

    if max_error_cents < 1.0:
        print("[PASS] Direct sine within 1 cent tolerance")
        return True
    else:
        print("[FAIL] Direct sine exceeds 1 cent tolerance")
        return False


def test_mode_b_frequency():
    """Test that Mode B harmonic bank produces correct frequency."""
    print("\n=== Mode B Frequency Test ===\n")

    # Test subset of notes
    test_notes = [48, 60, 69, 72]  # C3, C4, A4, C5
    num_samples = SAMPLE_RATE  # 1 second

    max_error_cents = 0

    print(f"{'Note':>4} {'Expected Hz':>12} {'Measured Hz':>12} {'Error (cents)':>14}")
    print("-" * 48)

    for note in test_notes:
        expected_hz = midi_to_freq(note)

        # Generate audio
        samples = generate_mode_b_audio(note, num_samples)

        # Measure frequency
        measured_hz = measure_fundamental_fft(samples.astype(np.float64), SAMPLE_RATE)

        # Calculate error
        error_cents = freq_to_cents_error(measured_hz, expected_hz)
        max_error_cents = max(max_error_cents, abs(error_cents))

        note_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][note % 12]
        octave = (note // 12) - 1
        status = "[ok]" if abs(error_cents) < 1.0 else "[!]"
        print(f"{note_name}{octave:>2} ({note:3d}) {expected_hz:12.4f} {measured_hz:12.4f} {error_cents:+14.6f} {status}")

    print(f"\nMax error: {max_error_cents:.6f} cents")

    if max_error_cents < 1.0:
        print("[PASS] Mode B within 1 cent tolerance")
        return True
    else:
        print("[FAIL] Mode B exceeds 1 cent tolerance")
        return False


def test_full_range_pitch_table():
    """Test pitch table across full MIDI note range."""
    print("\n=== Full Range Pitch Table Test (Notes 21-108) ===\n")

    max_error_cents = 0
    worst_note = 0

    for note in range(21, 109):  # A0 to C8
        expected_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))
        phase_inc = midi_to_phase_inc_fixed(note)
        measured_hz = (phase_inc * SAMPLE_RATE) / (2**32)
        error_cents = abs(freq_to_cents_error(measured_hz, expected_hz))

        if error_cents > max_error_cents:
            max_error_cents = error_cents
            worst_note = note

    note_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][worst_note % 12]
    octave = (worst_note // 12) - 1

    print(f"Notes tested: 21-108 (A0 to C8)")
    print(f"Worst case: {note_name}{octave} (note {worst_note}) with {max_error_cents:.6f} cents error")

    if max_error_cents < 1.0:
        print("[PASS] All notes within 1 cent tolerance")
        return True
    else:
        print("[FAIL] Some notes exceed 1 cent tolerance")
        return False


def test_cents_detuning():
    """Test that cents detuning is accurate."""
    print("\n=== Cents Detuning Test ===\n")

    base_note = 69  # A4 = 440 Hz
    max_error_cents = 0

    print(f"{'Detune':>8} {'Expected Hz':>12} {'Measured Hz':>12} {'Error (cents)':>14}")
    print("-" * 52)

    for detune_cents in [-100, -50, -10, 0, 10, 50, 100]:
        expected_hz = 440.0 * (2.0 ** (detune_cents / 1200.0))

        phase_inc = midi_to_phase_inc_fixed(base_note, detune_cents)
        measured_hz = (phase_inc * SAMPLE_RATE) / (2**32)

        error_cents = freq_to_cents_error(measured_hz, expected_hz)
        max_error_cents = max(max_error_cents, abs(error_cents))

        status = "[ok]" if abs(error_cents) < 1.0 else "[!]"
        print(f"{detune_cents:+8d} {expected_hz:12.4f} {measured_hz:12.4f} {error_cents:+14.6f} {status}")

    print(f"\nMax error: {max_error_cents:.6f} cents")

    if max_error_cents < 1.0:
        print("[PASS] Cents detuning within 1 cent tolerance")
        return True
    else:
        print("[FAIL] Cents detuning exceeds 1 cent tolerance")
        return False


def main():
    print("=" * 60)
    print("CCS Conformance: Frequency Validation")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Pitch Table Accuracy", test_pitch_table_accuracy()))
    results.append(("Full Range Pitch Table", test_full_range_pitch_table()))
    results.append(("Cents Detuning", test_cents_detuning()))
    results.append(("Direct SineCore Frequency", test_direct_sine_frequency()))
    results.append(("Mode B Frequency", test_mode_b_frequency()))

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
        print("[PASS] All frequency validation tests passed!")
        return 0
    else:
        print("[FAIL] Some frequency validation tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
