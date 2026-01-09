"""Debug phase resolution for coefficient LUT."""
import numpy as np

# C4 = 261.63 Hz at 48kHz
freq_hz = 261.63
sample_rate = 48000

# Samples per cycle
samples_per_cycle = sample_rate / freq_hz
print(f"Samples per cycle: {samples_per_cycle:.1f}")

# For P=256 bins, how many samples per bin?
P = 256
samples_per_bin = samples_per_cycle / P
print(f"With P={P}: {samples_per_bin:.2f} samples per bin")

# A 64-sample block spans how many phase bins?
block_size = 64
bins_per_block = block_size / samples_per_bin
print(f"A 64-sample block spans {bins_per_block:.1f} phase bins")

# For square wave, the transition happens at 50% of cycle
# How many bins into the cycle?
transition_bin = P // 2
print(f"Square wave transition at bin {transition_bin}")

# The issue: when we interpolate between bins, we're assuming
# smooth transitions. But square wave has a discontinuity.

print("\n=== The Problem ===")
print("Square wave c-vector: [..., +32767, +32767, -32767, -32767, ...]")
print("Coefficient LUT stores FWPT of each phase slice.")
print("At the transition point, adjacent bins have very different coefficients.")
print("Linear interpolation between them creates artifacts.")

print("\n=== Solutions ===")
print("1. Use higher P (e.g., 1024) so block always stays within one bin")
print(f"   With P=1024: {64 / (samples_per_cycle / 1024):.1f} bins per block")
print("2. Use bandlimited c-vector (precomputed harmonic synthesis)")
print("3. Use non-linear interpolation that handles discontinuities")
