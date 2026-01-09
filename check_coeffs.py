"""Check square vs triangle time domain."""
import sys
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path

def analyze_boundaries(filepath, name):
    fs, data = wav.read(filepath)
    if len(data.shape) > 1:
        samples = data[:, 0]
    else:
        samples = data

    print(f"\n=== {name} ===")

    block_size = 64
    diffs_at_boundary = []
    diffs_within_block = []

    for i in range(1000, len(samples)-1):
        diff = abs(int(samples[i+1]) - int(samples[i]))
        if i % block_size == (block_size - 1):
            diffs_at_boundary.append(diff)
        elif i % block_size == 32:
            diffs_within_block.append(diff)

    print(f"  Avg diff at block boundaries: {np.mean(diffs_at_boundary):.1f}")
    print(f"  Avg diff within blocks: {np.mean(diffs_within_block):.1f}")
    ratio = np.mean(diffs_at_boundary)/np.mean(diffs_within_block)
    print(f"  Boundary/Within ratio: {ratio:.2f}x")
    return samples

analyze_boundaries('audio_generation_tests/audio_out/waveforms/01_sine.wav', 'SINE')
analyze_boundaries('audio_generation_tests/audio_out/waveforms/02_square.wav', 'SQUARE')
analyze_boundaries('audio_generation_tests/audio_out/waveforms/05_triangle.wav', 'TRIANGLE')
analyze_boundaries('audio_generation_tests/audio_out/waveforms/04_sawtooth.wav', 'SAWTOOTH')

# Now load square for detailed analysis
fs, data = wav.read('audio_generation_tests/audio_out/waveforms/02_square.wav')
if len(data.shape) > 1:
    samples = data[:, 0]
else:
    samples = data

# Find period (C4 = 261.63 Hz, period = 183.5 samples at 48kHz)
period = int(48000 / 261.63)
print(f"Expected period: {period} samples")

# Look at one period in the middle of the file
start = 10000
one_period = samples[start:start+period*2]

print(f"\nTime domain values (2 periods starting at sample {start}):")
print(f"  Min: {one_period.min()}")
print(f"  Max: {one_period.max()}")

# Check flatness: standard deviation of samples in top/bottom regions
top_mask = one_period > 10000
bottom_mask = one_period < -10000

if np.sum(top_mask) > 0:
    top_std = np.std(one_period[top_mask])
    print(f"  Top region std dev: {top_std:.1f} (lower = flatter)")

if np.sum(bottom_mask) > 0:
    bottom_std = np.std(one_period[bottom_mask])
    print(f"  Bottom region std dev: {bottom_std:.1f} (lower = flatter)")

# Compare to ideal bandlimited square
print("\n--- Ideal bandlimited square wave for comparison ---")
t = np.arange(period*2) / 48000
ideal = np.zeros_like(t)
for h in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]:
    ideal += (1/h) * np.sin(2 * np.pi * 261.63 * h * t)
ideal = (ideal / np.max(np.abs(ideal)) * 16000).astype(np.int16)

ideal_top = ideal[ideal > 10000]
ideal_bottom = ideal[ideal < -10000]
if len(ideal_top) > 0:
    print(f"  Ideal top region std dev: {np.std(ideal_top):.1f}")
if len(ideal_bottom) > 0:
    print(f"  Ideal bottom region std dev: {np.std(ideal_bottom):.1f}")

# Check for block boundary artifacts
print("\n--- Block boundary analysis ---")
block_size = 64
# Look at difference between consecutive samples at block boundaries
diffs_at_boundary = []
diffs_within_block = []

for i in range(1000, len(samples)-1):
    diff = abs(int(samples[i+1]) - int(samples[i]))
    if i % block_size == (block_size - 1):  # Last sample of block
        diffs_at_boundary.append(diff)
    elif i % block_size == 32:  # Middle of block
        diffs_within_block.append(diff)

print(f"  Avg diff at block boundaries: {np.mean(diffs_at_boundary):.1f}")
print(f"  Avg diff within blocks: {np.mean(diffs_within_block):.1f}")
print(f"  Boundary/Within ratio: {np.mean(diffs_at_boundary)/np.mean(diffs_within_block):.2f}x")
