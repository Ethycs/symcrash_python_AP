#!/usr/bin/env python3
"""
Analyze spectrum of WAV files to find combing artifacts.
"""

import sys
from pathlib import Path
import wave
import numpy as np
from scipy.fft import rfft, rfftfreq

def analyze_wav(filepath, label=None):
    """Analyze a WAV file and print spectrum info."""
    if label is None:
        label = Path(filepath).name

    print("\n" + "="*60)
    print("Analyzing: {}".format(label))
    print("="*60)

    # Read WAV
    with wave.open(str(filepath), 'r') as f:
        nframes = f.getnframes()
        rate = f.getframerate()
        data = f.readframes(nframes)
        samples = np.frombuffer(data, dtype=np.int16)

    print("Samples: {}, Rate: {} Hz, Duration: {:.2f}s".format(
        len(samples), rate, len(samples)/rate))

    # Basic stats
    rms = np.sqrt(np.mean(samples.astype(np.float64)**2))
    print("RMS: {:.1f}, Max: {}".format(rms, np.max(np.abs(samples))))

    # FFT
    spectrum = np.abs(rfft(samples.astype(np.float64)))
    freqs = rfftfreq(len(samples), 1/rate)

    # Find fundamental
    # Skip DC and very low frequencies
    start_idx = int(50 * len(freqs) / (rate/2))  # Start at 50 Hz
    fundamental_idx = start_idx + np.argmax(spectrum[start_idx:start_idx+1000])
    fundamental_freq = freqs[fundamental_idx]
    fundamental_mag = spectrum[fundamental_idx]

    print("\nFundamental: {:.1f} Hz (magnitude: {:.0f})".format(
        fundamental_freq, fundamental_mag))

    # Find top 10 peaks
    peak_indices = np.argsort(spectrum)[-15:][::-1]
    print("\nTop 15 frequency components:")
    for i, idx in enumerate(peak_indices):
        freq = freqs[idx]
        mag = spectrum[idx]
        rel_db = 20 * np.log10(mag / fundamental_mag + 1e-10)
        harmonic = freq / fundamental_freq if fundamental_freq > 0 else 0
        print("  {:2d}. {:7.1f} Hz  mag={:10.0f}  {:+6.1f} dB  ({:.2f}x fundamental)".format(
            i+1, freq, mag, rel_db, harmonic))

    # Check for comb pattern (evenly spaced peaks)
    print("\nChecking for comb artifacts...")

    # Look for peaks that are NOT harmonics of fundamental
    non_harmonic_peaks = []
    for idx in peak_indices[:15]:
        freq = freqs[idx]
        if freq < 50:
            continue
        harmonic_ratio = freq / fundamental_freq
        # Check if it's close to an integer harmonic
        nearest_harmonic = round(harmonic_ratio)
        if nearest_harmonic > 0:
            deviation = abs(harmonic_ratio - nearest_harmonic) / nearest_harmonic
            if deviation > 0.02:  # More than 2% off from harmonic
                non_harmonic_peaks.append((freq, spectrum[idx], harmonic_ratio))

    if non_harmonic_peaks:
        print("  WARNING: Found non-harmonic peaks (potential combing):")
        for freq, mag, ratio in non_harmonic_peaks:
            rel_db = 20 * np.log10(mag / fundamental_mag + 1e-10)
            print("    {:.1f} Hz ({:+.1f} dB) - ratio: {:.3f}".format(freq, rel_db, ratio))
    else:
        print("  OK: All major peaks are harmonic")

    # Check harmonic decay pattern
    print("\nHarmonic analysis:")
    for h in range(1, 10):
        target_freq = fundamental_freq * h
        # Find closest bin
        target_idx = int(target_freq * len(freqs) / (rate/2))
        if target_idx < len(spectrum):
            # Look in a small window around target
            window = 5
            start = max(0, target_idx - window)
            end = min(len(spectrum), target_idx + window)
            local_max = np.max(spectrum[start:end])
            local_max_idx = start + np.argmax(spectrum[start:end])
            actual_freq = freqs[local_max_idx]
            rel_db = 20 * np.log10(local_max / fundamental_mag + 1e-10)
            marker = "" if h == 1 else ""
            if rel_db > -40:  # Only show significant harmonics
                print("  H{}: {:.1f} Hz  {:+6.1f} dB{}".format(
                    h, actual_freq, rel_db, marker))

    return spectrum, freqs, fundamental_freq


def main():
    base = Path(__file__).parent.parent

    # Analyze available test files
    test_files = [
        ("test_carrier_only.wav", "Carrier Only (v1, new frame each block)"),
        ("test_carrier_only_v2.wav", "Carrier Only (v2, same frame)"),
        ("test_wet_bypass.wav", "Wet Bypass (100% wet, anticlick off)"),
    ]

    for filename, label in test_files:
        filepath = base / filename
        if filepath.exists():
            try:
                analyze_wav(filepath, label)
            except Exception as e:
                print("Error analyzing {}: {}".format(filename, e))
        else:
            print("\nSkipping {} - file not found".format(filename))

    # Also check the original clean test if it exists
    audio_out = base / "audio_out"
    if audio_out.exists():
        clean_tests = list(audio_out.glob("test_pipeline_sine*.wav"))
        if clean_tests:
            # Analyze most recent
            latest = sorted(clean_tests)[-1]
            analyze_wav(latest, "Original test_pipeline_sine (known clean)")


if __name__ == "__main__":
    main()
