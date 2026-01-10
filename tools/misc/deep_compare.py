#!/usr/bin/env python3
"""
Deep comparison between original clean test and new carrier test.
"""

import sys
from pathlib import Path
import wave
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import windows

def read_wav(filepath):
    with wave.open(str(filepath), 'r') as f:
        rate = f.getframerate()
        nch = f.getnchannels()
        data = f.readframes(f.getnframes())
        samples = np.frombuffer(data, dtype=np.int16)
        if nch == 2:
            samples = samples[::2]  # Take left channel
    return samples, rate

def analyze_signal(samples, label, rate=48000):
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)

    print("Samples: {}  Rate: {} Hz  Duration: {:.2f}s".format(
        len(samples), rate, len(samples) / rate))

    rms = np.sqrt(np.mean(samples.astype(np.float64)**2))
    print("RMS: {:.1f}  Max: {}  Min: {}".format(rms, np.max(samples), np.min(samples)))

    # First 20 samples
    print("\nFirst 20 samples:")
    print("  ", samples[:20].tolist())

    # Check block boundaries for discontinuities
    print("\nBlock boundary analysis (64-sample blocks):")
    discontinuities = []
    for i in range(64, min(len(samples), 6400), 64):
        # Calculate expected derivative from previous samples
        prev_deriv = samples[i-1] - samples[i-2]
        actual_jump = samples[i] - samples[i-1]
        # If jump is much larger than expected derivative, it's a discontinuity
        if abs(actual_jump - prev_deriv) > 50:
            discontinuities.append((i, actual_jump, prev_deriv))

    if discontinuities:
        print("  Found {} boundary discontinuities:".format(len(discontinuities)))
        for i, jump, expected in discontinuities[:5]:
            print("    Sample {}: jump={} expected={}".format(i, jump, expected))
    else:
        print("  No significant discontinuities at block boundaries")

    # FFT with window
    window = windows.blackman(len(samples))
    windowed = samples.astype(np.float64) * window
    spectrum = np.abs(rfft(windowed))
    freqs = rfftfreq(len(samples), 1/rate)

    # Find fundamental
    start_idx = int(50 * len(freqs) / (rate/2))
    fund_idx = start_idx + np.argmax(spectrum[start_idx:start_idx+2000])
    fund_freq = freqs[fund_idx]
    fund_mag = spectrum[fund_idx]

    print("\nSpectral analysis (Blackman windowed):")
    print("  Fundamental: {:.2f} Hz".format(fund_freq))

    # Look for content not at harmonics
    print("  Checking for non-harmonic content...")
    non_harm_energy = 0
    harm_energy = fund_mag ** 2

    for i in range(len(spectrum)):
        freq = freqs[i]
        if freq < 20:
            continue
        mag = spectrum[i]
        if mag < fund_mag * 0.0001:  # Below -80 dB
            continue

        # Check if near a harmonic
        is_harmonic = False
        for h in range(1, 30):
            if abs(freq - fund_freq * h) < 5:
                is_harmonic = True
                harm_energy += mag ** 2
                break

        if not is_harmonic and mag > fund_mag * 0.001:  # Above -60 dB
            rel_db = 20 * np.log10(mag / fund_mag)
            print("    {:.1f} Hz: {:+.1f} dB".format(freq, rel_db))
            non_harm_energy += mag ** 2

    if non_harm_energy > 0:
        snr = 10 * np.log10(harm_energy / non_harm_energy)
        print("  Harmonic SNR: {:.1f} dB".format(snr))
    else:
        print("  PURE: No non-harmonic content above -60 dB")

    return spectrum, freqs


def main():
    base = Path(__file__).parent.parent

    # Original clean test
    orig_path = base / "audio_out" / "test_pipeline_sine-20260102-094932.wav"
    if orig_path.exists():
        orig, rate = read_wav(orig_path)
        analyze_signal(orig, "ORIGINAL test_pipeline_sine (known clean)", rate)

    # New compare test
    new_path = base / "compare_orig.wav"
    if new_path.exists():
        new, rate = read_wav(new_path)
        analyze_signal(new, "NEW compare_orig.wav", rate)

    # Direct sample comparison if both exist and same frequency
    if orig_path.exists() and new_path.exists():
        print("\n" + "=" * 70)
        print("DIRECT COMPARISON")
        print("=" * 70)

        # They're at different frequencies (130 Hz vs 261 Hz)
        # So check waveform shape instead

        # Normalize and compare one cycle
        orig_norm = orig.astype(np.float64) / np.max(np.abs(orig))
        new_norm = new.astype(np.float64) / np.max(np.abs(new))

        # Find period of each
        orig_period = int(48000 / 130.75)  # ~367 samples
        new_period = int(48000 / 261.63)   # ~183 samples

        print("Original period: {} samples ({:.2f} Hz)".format(orig_period, 48000/orig_period))
        print("New period: {} samples ({:.2f} Hz)".format(new_period, 48000/new_period))

        # Check cycle-to-cycle consistency in new signal
        print("\nCycle consistency check (new signal):")
        cycles = []
        for i in range(10):
            start = i * new_period
            end = start + new_period
            if end < len(new):
                cycles.append(new[start:end])

        if len(cycles) >= 2:
            cycle_diffs = []
            for i in range(1, len(cycles)):
                diff = np.abs(cycles[i].astype(np.int32) - cycles[0].astype(np.int32))
                cycle_diffs.append(np.max(diff))
            print("  Max difference between cycles: {}".format(max(cycle_diffs)))
            print("  Cycle differences: {}".format(cycle_diffs[:5]))


if __name__ == "__main__":
    main()
