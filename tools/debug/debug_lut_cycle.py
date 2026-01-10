"""Debug LUT cycle length and phase quantization."""

import numpy as np
import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.vm.coeff_lut_streaming import (
    get_lut_for_note, DEFAULT_K, DEFAULT_P, SAMPLE_RATE, BLOCK_SIZE
)

print("=== LUT PARAMETERS FOR 440 Hz (MIDI 69) ===")
lut = get_lut_for_note(wave_id=0, midi_note=69)

print(f"Target frequency: {lut.target_freq_hz} Hz")
print(f"K (polyphase slots): {lut.K}")
print(f"P (phase steps): {lut.P}")
print(f"LUT shape: {lut.data.shape}")

# Calculate cycle length
cycle_len = int(round(SAMPLE_RATE / lut.target_freq_hz))
print(f"\nCycle length: {cycle_len} samples")
print(f"Exact period: {SAMPLE_RATE / lut.target_freq_hz:.4f} samples")

# Phase increment per block
phase_inc_per_sample = lut.target_freq_hz / SAMPLE_RATE
phase_inc_per_block = phase_inc_per_sample * BLOCK_SIZE
print(f"\nPhase increment per sample: {phase_inc_per_sample:.6f} cycles")
print(f"Phase increment per block: {phase_inc_per_block:.6f} cycles")
print(f"Phase steps (P) per cycle: {lut.P}")
print(f"Phase step resolution: {1/lut.P:.6f} cycles per step")

# Check if P evenly divides the period
samples_per_phase_step = cycle_len / lut.P
print(f"\nSamples per phase step: {samples_per_phase_step:.4f}")
print(f"Is this an integer? {samples_per_phase_step == int(samples_per_phase_step)}")

# The problem: if cycle_len doesn't align with P, there will be phase discontinuities
# For 440 Hz: cycle_len = 109 samples
# P = 64, so each phase step covers 109/64 = 1.703 samples
# This non-integer relationship means phase interpolation is needed

print("\n=== PHASE STEP POSITIONS ===")
print("Phase step -> Sample position in cycle")
for p in range(min(lut.P, 10)):
    start = int(round(p * cycle_len / lut.P))
    exact_start = p * cycle_len / lut.P
    print(f"  p={p:2d}: sample {start:3d} (exact: {exact_start:.2f}, error: {start - exact_start:+.2f})")

print(f"  ...")
for p in range(lut.P - 3, lut.P):
    start = int(round(p * cycle_len / lut.P))
    exact_start = p * cycle_len / lut.P
    print(f"  p={p:2d}: sample {start:3d} (exact: {exact_start:.2f}, error: {start - exact_start:+.2f})")

# Check the spur frequency relationship
print("\n=== POTENTIAL SPUR SOURCES ===")

# Phase quantization creates sidebands at f0 ± fs/P
sideband_spacing = SAMPLE_RATE / lut.P
print(f"Phase quantization sideband spacing: {sideband_spacing:.1f} Hz")

# Block rate artifacts
block_rate = SAMPLE_RATE / BLOCK_SIZE
print(f"Block rate: {block_rate:.1f} Hz")

# K-related artifacts
k_rate = block_rate / lut.K
print(f"K-rate (block_rate / K): {k_rate:.2f} Hz")

# Cycle aliasing: if cycle_len doesn't divide evenly into P or K
cycle_freq = SAMPLE_RATE / cycle_len
print(f"Actual cycle frequency: {cycle_freq:.4f} Hz (vs target {lut.target_freq_hz} Hz)")
print(f"Frequency error: {cycle_freq - lut.target_freq_hz:.4f} Hz")

# Check the actual phase increment used at runtime
phase_inc_32 = int((lut.target_freq_hz / SAMPLE_RATE) * (1 << 32))
actual_freq = (phase_inc_32 / (1 << 32)) * SAMPLE_RATE
print(f"\nRuntime phase increment (32-bit): {phase_inc_32}")
print(f"Actual frequency from phase_inc: {actual_freq:.6f} Hz")
print(f"Phase increment error: {actual_freq - lut.target_freq_hz:.6f} Hz")

# The key issue: LUT was built for cycle_len samples, but runtime uses phase_inc
# which may not align perfectly
runtime_cycle_samples = SAMPLE_RATE / actual_freq
print(f"\nRuntime cycle length: {runtime_cycle_samples:.4f} samples")
print(f"LUT cycle length: {cycle_len} samples")
print(f"Mismatch: {runtime_cycle_samples - cycle_len:.4f} samples")
