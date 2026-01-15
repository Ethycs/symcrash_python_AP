#!/usr/bin/env python3
"""Analyze FM6 brass waveform - generate snapshots at different time points."""

import numpy as np
from pathlib import Path

# Load the rendered WAV file
import wave
import struct

wav_path = Path(__file__).parent.parent / "audio_out" / "presets" / "fm_brass_v0_note60.wav"

with wave.open(str(wav_path), 'r') as wav:
    n_frames = wav.getnframes()
    sample_rate = wav.getframerate()
    raw_data = wav.readframes(n_frames)
    samples = np.array(struct.unpack(f'<{n_frames}h', raw_data), dtype=np.int16)

print("FM6 Brass Waveform Analysis")
print("=" * 60)
print(f"File: {wav_path.name}")
print(f"Duration: {len(samples) / sample_rate:.2f}s ({len(samples)} samples)")
print(f"Sample rate: {sample_rate} Hz")
print(f"Peak: {samples.max()} / {samples.min()}")
print(f"RMS: {np.sqrt(np.mean(samples.astype(float)**2)):.1f}")

# Analyze different time regions
regions = [
    ("Attack (0-50ms)", 0, int(0.050 * sample_rate)),
    ("Early (50-150ms)", int(0.050 * sample_rate), int(0.150 * sample_rate)),
    ("Sustain (300-500ms)", int(0.300 * sample_rate), int(0.500 * sample_rate)),
    ("Late (800-1000ms)", int(0.800 * sample_rate), int(1.000 * sample_rate)),
]

print("\n" + "=" * 60)
print("Regional Analysis")
print("=" * 60)

for name, start, end in regions:
    if end > len(samples):
        end = len(samples)
    if start >= len(samples):
        continue

    region = samples[start:end]

    # Basic stats
    rms = np.sqrt(np.mean(region.astype(float)**2))
    peak = max(abs(region.max()), abs(region.min()))

    # Zero crossings (indicator of frequency content)
    zero_crossings = np.sum(np.diff(np.sign(region.astype(float))) != 0)
    zc_rate = zero_crossings / (len(region) / sample_rate)  # per second

    # Estimate fundamental from zero crossings
    est_freq = zc_rate / 2

    # Check for amplitude modulation (electric shaver symptom)
    # Split into small windows and check variance of peaks
    window_size = 64
    n_windows = len(region) // window_size
    if n_windows > 0:
        window_peaks = []
        for i in range(n_windows):
            w = region[i*window_size:(i+1)*window_size]
            window_peaks.append(max(abs(w.max()), abs(w.min())))
        peak_variance = np.std(window_peaks) / (np.mean(window_peaks) + 1)
    else:
        peak_variance = 0

    print(f"\n{name}:")
    print(f"  RMS: {rms:.1f}")
    print(f"  Peak: {peak}")
    print(f"  Zero crossings: {zero_crossings} ({zc_rate:.0f}/sec)")
    print(f"  Est. frequency: {est_freq:.1f} Hz (expected ~262 Hz for C4)")
    print(f"  Amplitude variance: {peak_variance:.3f}", end="")
    if peak_variance > 0.3:
        print(" [HIGH - possible AM/beating]")
    else:
        print(" [OK]")

# Print actual waveform samples at key points
print("\n" + "=" * 60)
print("Waveform Snapshots (first 32 samples of each region)")
print("=" * 60)

for name, start, end in regions:
    if start >= len(samples):
        continue

    snapshot = samples[start:start+32]
    print(f"\n{name} (samples {start}-{start+31}):")

    # ASCII visualization
    max_val = 32767
    width = 40
    for i, s in enumerate(snapshot):
        bar_pos = int((s / max_val + 1) / 2 * width)
        bar = [' '] * width
        bar[width//2] = '|'
        if 0 <= bar_pos < width:
            bar[bar_pos] = '*'
        print(f"  {start+i:5d}: {s:7d} {''.join(bar)}")

# Check for the "electric shaver" pattern - rapid amplitude modulation
print("\n" + "=" * 60)
print("Electric Shaver Detection")
print("=" * 60)

# Look at first 500ms where the problem is reported
attack_region = samples[:int(0.5 * sample_rate)]

# Compute envelope via Hilbert transform approximation
# Simple: absolute value smoothed
abs_signal = np.abs(attack_region.astype(float))
# Moving average with 64-sample window
kernel_size = 64
envelope = np.convolve(abs_signal, np.ones(kernel_size)/kernel_size, mode='valid')

# Check for oscillation in envelope (AM)
if len(envelope) > 200:
    envelope_diff = np.diff(envelope)
    sign_changes = np.sum(np.diff(np.sign(envelope_diff)) != 0)
    am_rate = sign_changes / (len(envelope) / sample_rate)

    print(f"Envelope oscillation rate: {am_rate:.1f} Hz")
    if am_rate > 50:
        print("[DETECTED] Rapid envelope modulation - likely cause of 'electric shaver' sound")
        print("This is caused by beating between detuned carriers")
    else:
        print("[OK] Envelope is smooth")

# Frequency analysis via FFT
print("\n" + "=" * 60)
print("Frequency Content (first 100ms)")
print("=" * 60)

fft_region = samples[:int(0.1 * sample_rate)]
fft_result = np.fft.rfft(fft_region)
freqs = np.fft.rfftfreq(len(fft_region), 1/sample_rate)
magnitudes = np.abs(fft_result)

# Find top 10 peaks
peak_indices = np.argsort(magnitudes)[-10:][::-1]
print("\nTop frequency components:")
for idx in peak_indices:
    freq = freqs[idx]
    mag = magnitudes[idx]
    print(f"  {freq:7.1f} Hz: {mag:10.0f}")
