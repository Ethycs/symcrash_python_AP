"""
FFT Analysis of Synth Mode Demos
================================

Analyzes the spectral content of each generated synth mode demo
to verify they have the expected frequency characteristics.
"""

import sys
import numpy as np
import wave
from pathlib import Path
from typing import Tuple, List

# Constants
FS = 48000
BLOCK_SIZE = 64

OUTPUT_DIR = Path(__file__).parent / "audio_out" / "synth_modes"


def load_wav(filepath: Path) -> np.ndarray:
    """Load a WAV file and return samples as float32."""
    with wave.open(str(filepath), 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()

        raw = wf.readframes(n_frames)

        if sampwidth == 2:
            samples = np.frombuffer(raw, dtype=np.int16)
        elif sampwidth == 4:
            samples = np.frombuffer(raw, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        # If stereo, take left channel
        if n_channels == 2:
            samples = samples[::2]

        return samples.astype(np.float32) / 32768.0


def analyze_fft(samples: np.ndarray, fft_size: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT magnitude spectrum.

    Returns:
        freqs: Frequency bins (Hz)
        magnitudes: Magnitude in dB
    """
    # Take a section from the middle (avoid attack/release)
    mid = len(samples) // 2
    start = max(0, mid - fft_size // 2)
    segment = samples[start:start + fft_size]

    # Apply Hann window
    window = np.hanning(len(segment))
    windowed = segment * window

    # Compute FFT
    fft_result = np.fft.rfft(windowed)
    magnitudes = np.abs(fft_result)

    # Convert to dB (with floor to avoid log(0))
    magnitudes_db = 20 * np.log10(magnitudes + 1e-10)

    # Frequency bins
    freqs = np.fft.rfftfreq(len(segment), 1.0 / FS)

    return freqs, magnitudes_db


def find_peaks(freqs: np.ndarray, magnitudes_db: np.ndarray,
               threshold_db: float = -40, min_distance_hz: float = 50) -> List[Tuple[float, float]]:
    """
    Find peaks in the spectrum.

    Returns:
        List of (frequency, magnitude_db) tuples for detected peaks
    """
    peaks = []
    max_mag = np.max(magnitudes_db)
    threshold = max_mag + threshold_db  # Relative to max

    # Simple peak detection
    for i in range(1, len(magnitudes_db) - 1):
        if magnitudes_db[i] > threshold:
            if magnitudes_db[i] > magnitudes_db[i-1] and magnitudes_db[i] > magnitudes_db[i+1]:
                freq = freqs[i]
                # Check minimum distance from existing peaks
                too_close = False
                for existing_freq, _ in peaks:
                    if abs(freq - existing_freq) < min_distance_hz:
                        too_close = True
                        break
                if not too_close:
                    peaks.append((freq, magnitudes_db[i]))

    # Sort by magnitude (strongest first)
    peaks.sort(key=lambda x: -x[1])

    return peaks[:20]  # Return top 20 peaks


def note_to_freq(note: int) -> float:
    """Convert MIDI note to frequency."""
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def freq_to_note_name(freq: float) -> str:
    """Convert frequency to approximate note name."""
    if freq <= 0:
        return "?"
    note_num = 69 + 12 * np.log2(freq / 440.0)
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_idx = int(round(note_num)) % 12
    octave = int(round(note_num)) // 12 - 1
    return f"{note_names[note_idx]}{octave}"


def analyze_demo(name: str, expected_fundamental: float = None,
                 expected_harmonics: int = None, has_fm: bool = False) -> dict:
    """
    Analyze a single demo file.

    Args:
        name: Filename without extension
        expected_fundamental: Expected fundamental frequency (Hz)
        expected_harmonics: Expected number of clear harmonics
        has_fm: Whether FM sidebands are expected

    Returns:
        Analysis results dictionary
    """
    filepath = OUTPUT_DIR / f"{name}.wav"

    if not filepath.exists():
        return {"error": f"File not found: {filepath}"}

    samples = load_wav(filepath)
    freqs, magnitudes = analyze_fft(samples)
    peaks = find_peaks(freqs, magnitudes)

    result = {
        "name": name,
        "duration_sec": len(samples) / FS,
        "rms_db": 20 * np.log10(np.sqrt(np.mean(samples**2)) + 1e-10),
        "peak_db": 20 * np.log10(np.max(np.abs(samples)) + 1e-10),
        "top_peaks": peaks[:10],
        "fundamental_detected": peaks[0][0] if peaks else None,
    }

    # Check if fundamental matches expected
    if expected_fundamental and peaks:
        detected = peaks[0][0]
        error_cents = 1200 * np.log2((detected + 1e-10) / expected_fundamental)
        result["fundamental_error_cents"] = error_cents
        result["fundamental_match"] = abs(error_cents) < 50  # Within 50 cents

    # Count harmonics
    if peaks:
        fundamental = peaks[0][0]
        harmonic_count = 0
        for i in range(2, 20):
            expected_h = fundamental * i
            # Check if any peak is close to this harmonic
            for freq, mag in peaks:
                if abs(freq - expected_h) < 30:  # Within 30 Hz
                    harmonic_count += 1
                    break
        result["harmonic_count"] = harmonic_count

    # FM analysis: check for non-harmonic sidebands
    if has_fm and peaks:
        fundamental = peaks[0][0]
        non_harmonic = 0
        for freq, mag in peaks[1:]:
            # Check if this is a harmonic
            ratio = freq / fundamental
            nearest_harmonic = round(ratio)
            if nearest_harmonic > 0:
                deviation = abs(ratio - nearest_harmonic)
                if deviation > 0.1:  # More than 10% off harmonic
                    non_harmonic += 1
        result["non_harmonic_peaks"] = non_harmonic
        result["has_fm_sidebands"] = non_harmonic >= 2

    return result


def main():
    print("=" * 70)
    print("FFT ANALYSIS OF SYNTH MODE DEMOS")
    print("=" * 70)
    print(f"\nLooking in: {OUTPUT_DIR}")
    print()

    # Expected values for each demo
    demos = [
        ("01_mode_a_wavetable", note_to_freq(60), 16, False),   # C4, 16 harmonics
        ("02_mode_b_harmonic", note_to_freq(60), 16, False),    # C4, 16 harmonics
        ("03_mode_b_fm_feedback", note_to_freq(60), 8, True),   # C4, 8 harmonics + FM
        ("04_mode_b_drone", note_to_freq(48), 8, False),        # C3, 8 harmonics
        ("05_sample_coeff_native", 440.0, 1, False),            # 440 Hz sine, 1 harmonic
        ("06_sample_pitch_shift", 220.0, None, False),          # 220 Hz saw (varying)
        ("07_mode_b_melody", None, 12, False),                  # Varying notes
        ("08_fm_bell", note_to_freq(72), 6, True),              # C5, 6 harmonics + FM
    ]

    all_results = []

    for name, expected_fund, expected_harmonics, has_fm in demos:
        print("-" * 70)
        print(f"Analyzing: {name}")
        print("-" * 70)

        result = analyze_demo(name, expected_fund, expected_harmonics, has_fm)
        all_results.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue

        print(f"  Duration: {result['duration_sec']:.2f}s")
        print(f"  RMS Level: {result['rms_db']:.1f} dB")
        print(f"  Peak Level: {result['peak_db']:.1f} dB")
        print()

        if result.get("fundamental_detected"):
            fund = result["fundamental_detected"]
            note = freq_to_note_name(fund)
            print(f"  Fundamental: {fund:.1f} Hz ({note})")

            if expected_fund:
                expected_note = freq_to_note_name(expected_fund)
                error = result.get("fundamental_error_cents", 0)
                status = "OK" if result.get("fundamental_match") else "MISMATCH"
                print(f"  Expected: {expected_fund:.1f} Hz ({expected_note})")
                print(f"  Error: {error:+.1f} cents [{status}]")

        if "harmonic_count" in result:
            print(f"  Detected harmonics: {result['harmonic_count']}")
            if expected_harmonics:
                print(f"  Expected harmonics: ~{expected_harmonics}")

        if has_fm:
            if result.get("has_fm_sidebands"):
                print(f"  FM sidebands: YES ({result.get('non_harmonic_peaks', 0)} non-harmonic peaks)")
            else:
                print(f"  FM sidebands: minimal/none detected")

        print()
        print("  Top 5 spectral peaks:")
        for i, (freq, mag) in enumerate(result.get("top_peaks", [])[:5]):
            note = freq_to_note_name(freq)
            print(f"    {i+1}. {freq:8.1f} Hz ({note:4s}) @ {mag:6.1f} dB")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    issues = []
    for result in all_results:
        if "error" in result:
            issues.append(f"{result.get('name', 'unknown')}: File not found")
        elif result.get("fundamental_match") == False:
            issues.append(f"{result['name']}: Fundamental mismatch")
        elif result.get("rms_db", -100) < -40:
            issues.append(f"{result['name']}: Very low level ({result['rms_db']:.1f} dB)")

    if issues:
        print("\nPotential issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nAll demos appear to have correct spectral content!")

    print()
    print("Expected characteristics:")
    print("  - Mode A/B: Clear harmonic series from fundamental")
    print("  - FM modes: Additional non-harmonic sidebands")
    print("  - Drone: Lower fundamental (C3 = 130.8 Hz)")
    print("  - Sample: 440 Hz sine fundamental")
    print("  - Bell: Higher fundamental (C5 = 523.3 Hz) + FM sidebands")


if __name__ == "__main__":
    main()
