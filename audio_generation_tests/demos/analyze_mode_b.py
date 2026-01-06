"""Analyze Mode B output for jaggies/discontinuities."""

import sys
import numpy as np
from pathlib import Path
import wave

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Load the audio file
wav_path = Path(__file__).parent.parent / "audio_out" / "mode_b_test" / "mode_b_test_a440.wav"

with wave.open(str(wav_path), 'rb') as wf:
    n_channels = wf.getnchannels()
    sample_width = wf.getsampwidth()
    framerate = wf.getframerate()
    n_frames = wf.getnframes()

    raw_data = wf.readframes(n_frames)
    samples = np.frombuffer(raw_data, dtype=np.int16)

    # Get left channel only
    if n_channels == 2:
        samples = samples[::2]

print(f"Loaded: {wav_path}")
print(f"Sample rate: {framerate} Hz")
print(f"Duration: {len(samples) / framerate:.2f}s")
print(f"Samples: {len(samples)}")
print()

# Analyze for discontinuities
# Look at sample-to-sample differences
diffs = np.diff(samples.astype(np.int32))
abs_diffs = np.abs(diffs)

print("Sample-to-sample difference analysis:")
print(f"  Max diff: {np.max(abs_diffs)}")
print(f"  Mean diff: {np.mean(abs_diffs):.1f}")
print(f"  Std diff: {np.std(abs_diffs):.1f}")

# Find large jumps (potential jaggies)
# A "jaggy" would be a discontinuity larger than typical waveform
threshold = np.mean(abs_diffs) + 4 * np.std(abs_diffs)
large_jumps = np.where(abs_diffs > threshold)[0]
print(f"  Threshold for 'large': {threshold:.0f}")
print(f"  Large jumps (>{threshold:.0f}): {len(large_jumps)}")

if len(large_jumps) > 0:
    print(f"\n  First 20 large jump positions:")
    for i, pos in enumerate(large_jumps[:20]):
        print(f"    Sample {pos}: {samples[pos]} -> {samples[pos+1]} (diff={diffs[pos]})")

# Check for block-boundary discontinuities (every 64 samples)
BLOCK_SIZE = 64
block_boundaries = list(range(BLOCK_SIZE, len(samples), BLOCK_SIZE))
boundary_diffs = [abs(int(samples[b]) - int(samples[b-1])) for b in block_boundaries if b < len(samples)]

print(f"\nBlock boundary analysis (every {BLOCK_SIZE} samples):")
print(f"  Boundaries checked: {len(boundary_diffs)}")
print(f"  Max boundary diff: {max(boundary_diffs) if boundary_diffs else 0}")
print(f"  Mean boundary diff: {np.mean(boundary_diffs):.1f}")

# Compare boundary diffs to non-boundary diffs
non_boundary_indices = [i for i in range(len(diffs)) if (i+1) % BLOCK_SIZE != 0]
non_boundary_diffs = abs_diffs[non_boundary_indices]
boundary_indices = [i for i in range(len(diffs)) if (i+1) % BLOCK_SIZE == 0]
boundary_only_diffs = abs_diffs[boundary_indices]

print(f"\n  Boundary vs non-boundary comparison:")
print(f"    Non-boundary mean diff: {np.mean(non_boundary_diffs):.1f}")
print(f"    Boundary mean diff: {np.mean(boundary_only_diffs):.1f}")
print(f"    Ratio: {np.mean(boundary_only_diffs) / np.mean(non_boundary_diffs):.2f}x")

# Spectral analysis - check for aliasing
from numpy.fft import fft, fftfreq

# Take a chunk from the middle (after warmup)
start = framerate // 2  # 0.5s in
chunk = samples[start:start + 4096].astype(np.float64)

# Window and FFT
window = np.hanning(len(chunk))
spectrum = np.abs(fft(chunk * window))
freqs = fftfreq(len(chunk), 1/framerate)

# Find positive frequencies
pos_mask = freqs > 0
pos_freqs = freqs[pos_mask]
pos_spectrum = spectrum[pos_mask]

# Normalize to dB
max_val = np.max(pos_spectrum)
spectrum_db = 20 * np.log10(pos_spectrum / max_val + 1e-10)

# Find peaks
peak_indices = []
for i in range(1, len(spectrum_db) - 1):
    if spectrum_db[i] > spectrum_db[i-1] and spectrum_db[i] > spectrum_db[i+1] and spectrum_db[i] > -60:
        peak_indices.append(i)

print(f"\nSpectral analysis (4096 samples from 0.5s):")
print(f"  Fundamental expected: 440 Hz")
print(f"  Top 10 peaks:")
sorted_peaks = sorted(peak_indices, key=lambda i: spectrum_db[i], reverse=True)[:10]
for i, idx in enumerate(sorted_peaks):
    print(f"    {i+1}. {pos_freqs[idx]:.1f} Hz: {spectrum_db[idx]:.1f} dB")

# Check for intermodulation/aliasing artifacts
print(f"\nChecking for non-harmonic content (aliasing indicators):")
fundamental = 440.0
harmonics = [fundamental * (h+1) for h in range(20)]
for idx in sorted_peaks:
    freq = pos_freqs[idx]
    # Check if this is a harmonic
    is_harmonic = any(abs(freq - h) < 10 for h in harmonics)
    if not is_harmonic and spectrum_db[idx] > -40:
        print(f"  NON-HARMONIC: {freq:.1f} Hz @ {spectrum_db[idx]:.1f} dB")
