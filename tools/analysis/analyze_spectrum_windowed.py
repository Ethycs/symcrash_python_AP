#!/usr/bin/env python3
"""
Analyze spectrum with proper windowing to avoid spectral leakage.

Spectral leakage happens when the FFT window doesn't align with signal period.
Using a window function (Hann, Blackman) reduces leakage.
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.lib.fft_utils import rfft_mag, rfft_mag_windowed
from tools.lib.wav_utils import read_wav_mono


def analyze_wav(filepath, label=None, use_window=True):
    """Analyze a WAV file with proper windowing."""
    if label is None:
        label = Path(filepath).name

    print("\n" + "=" * 60)
    print("Analyzing: {} (windowed={})".format(label, use_window))
    print("=" * 60)

    # Read WAV
    rate, samples = read_wav_mono(filepath)

    print("Samples: {}, Rate: {} Hz".format(len(samples), rate))

    # Apply window to reduce spectral leakage
    if use_window:
        spectrum, freqs, _window_gain = rfft_mag_windowed(
            samples, rate, window_name="blackman"
        )
    else:
        spectrum, freqs = rfft_mag(samples, rate)

    # Find fundamental
    start_idx = int(50 * len(freqs) / (rate/2))
    fundamental_idx = start_idx + np.argmax(spectrum[start_idx:start_idx+1000])
    fundamental_freq = freqs[fundamental_idx]
    fundamental_mag = spectrum[fundamental_idx]

    print("Fundamental: {:.2f} Hz (magnitude: {:.0f})".format(
        fundamental_freq, fundamental_mag))

    # Find peaks above -60 dB
    threshold = fundamental_mag * 0.001  # -60 dB
    peak_indices = []
    for i in range(10, len(spectrum) - 10):
        if spectrum[i] > threshold:
            if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
                peak_indices.append(i)

    # Sort by magnitude
    peak_indices.sort(key=lambda i: spectrum[i], reverse=True)

    print("\nTop peaks above -60 dB:")
    for i, idx in enumerate(peak_indices[:10]):
        freq = freqs[idx]
        mag = spectrum[idx]
        rel_db = 20 * np.log10(mag / fundamental_mag + 1e-10)
        harmonic = freq / fundamental_freq if fundamental_freq > 0 else 0
        print("  {:2d}. {:7.2f} Hz  {:+6.1f} dB  ({:.2f}x fundamental)".format(
            i+1, freq, rel_db, harmonic))

    # Check for non-harmonic content (with window, should be minimal)
    non_harmonic = []
    for idx in peak_indices[:20]:
        freq = freqs[idx]
        if freq < 50:
            continue
        harmonic_ratio = freq / fundamental_freq
        nearest_harmonic = round(harmonic_ratio)
        if nearest_harmonic > 0:
            deviation = abs(harmonic_ratio - nearest_harmonic) / nearest_harmonic
            rel_db = 20 * np.log10(spectrum[idx] / fundamental_mag + 1e-10)
            # With windowing, any deviation > 0.5% is suspicious
            if deviation > 0.005 and rel_db > -80:
                non_harmonic.append((freq, rel_db, harmonic_ratio, deviation))

    if non_harmonic:
        print("\nNON-HARMONIC CONTENT (deviation > 0.5%):")
        for freq, rel_db, ratio, dev in non_harmonic:
            print("  {:.2f} Hz ({:+.1f} dB) ratio={:.4f} deviation={:.2f}%".format(
                freq, rel_db, ratio, dev * 100))
    else:
        print("\nPURE: No non-harmonic content detected")

    # Measure noise floor
    # Noise floor = average of bins far from harmonics
    noise_bins = []
    for i in range(len(spectrum)):
        freq = freqs[i]
        if freq < 50 or freq > 20000:
            continue
        # Check if far from any harmonic
        for h in range(1, 20):
            if abs(freq - fundamental_freq * h) < 10:
                break
        else:
            noise_bins.append(spectrum[i])

    if noise_bins:
        noise_floor = np.mean(noise_bins)
        noise_db = 20 * np.log10(noise_floor / fundamental_mag + 1e-10)
        print("\nNoise floor: {:+.1f} dB (average of non-harmonic bins)".format(noise_db))


def main():
    base = Path(__file__).parent.parent

    test_files = [
        ("compare_orig.wav", "Original style (f0_hz=261.63)"),
        ("compare_new.wav", "New style (no f0_hz)"),
    ]

    for filename, label in test_files:
        filepath = base / filename
        if filepath.exists():
            # Compare with and without window
            analyze_wav(filepath, label + " WITHOUT window", use_window=False)
            analyze_wav(filepath, label + " WITH Blackman window", use_window=True)

    # Also analyze original clean test
    audio_out = base / "audio_out"
    if audio_out.exists():
        clean_tests = list(audio_out.glob("test_pipeline_sine*.wav"))
        if clean_tests:
            latest = sorted(clean_tests)[-1]
            analyze_wav(latest, "Original test_pipeline_sine (windowed)", use_window=True)


if __name__ == "__main__":
    main()
