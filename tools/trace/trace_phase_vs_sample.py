"""Trace how oscillator phase maps to sample position."""

import numpy as np
import sys
sys.path.insert(0, 'src')

SAMPLE_RATE = 48000
BLOCK_SIZE = 64
f0 = 440.0
P = 256

print("=== PHASE vs SAMPLE POSITION MAPPING ===")
print()

phase_inc = int((f0 / SAMPLE_RATE) * (1 << 32))
cycle_samples = SAMPLE_RATE / f0  # ~109.09 samples per cycle

print(f"Frequency: {f0} Hz")
print(f"Cycle length: {cycle_samples:.2f} samples")
print(f"Phase inc per sample: {phase_inc}")
print(f"Phase inc per block: {phase_inc * BLOCK_SIZE}")
print()

# Trace what happens over several cycles
print("=== SAMPLE POSITION TRACE ===")
print("block  sample_pos   phase_32        phase_frac   cycle_frac   error")
print("-" * 80)

phase = 0
for block in range(20):
    sample_pos = block * BLOCK_SIZE

    # Phase as fraction of cycle [0, 1)
    phase_frac = phase / (1 << 32)

    # Expected cycle fraction based on sample position
    cycle_frac = (sample_pos % cycle_samples) / cycle_samples

    # Error (in cycle fraction)
    error = phase_frac - cycle_frac
    # Wrap to [-0.5, 0.5)
    if error > 0.5:
        error -= 1.0
    elif error < -0.5:
        error += 1.0

    print(f"{block:5d}  {sample_pos:10d}   {phase:12d}   {phase_frac:10.6f}   {cycle_frac:10.6f}   {error:+.6f}")

    # Advance phase
    phase = (phase + phase_inc * BLOCK_SIZE) & 0xFFFFFFFF

print()
print("=== KEY INSIGHT ===")
print()
print("The LUT stores coefficients indexed by OSCILLATOR PHASE (phase_32 / 2^32).")
print("The ground truth test uses SAMPLE POSITION (block * 64 + k).")
print()
print("For a 440 Hz sine at 48kHz:")
print(f"  - One cycle = {cycle_samples:.2f} samples")
print(f"  - One block = {BLOCK_SIZE} samples")
print(f"  - Blocks per cycle = {cycle_samples / BLOCK_SIZE:.4f}")
print()
print("The phase accumulator advances by phase_inc_per_block per block.")
print("This is a MODULAR operation - it wraps at 2^32.")
print()
print("So for block N:")
print("  - Sample position = N * 64")
print("  - Phase = (N * phase_inc * 64) mod 2^32")
print()
print("These are NOT the same! Phase is modular (wraps per cycle),")
print("sample position is linear (continuous).")
print()
print("This is CORRECT for LUT indexing - we want phase-indexed coefficients.")
print("The question is: are the LUT coefficients correct for the phase?")
