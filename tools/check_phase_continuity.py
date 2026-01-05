"""Check if phase advances create continuous waveform."""

import numpy as np
import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.vm.coeff_lut_streaming import (
    get_lut_for_note, BLOCK_SIZE, DEFAULT_P
)

print("=== PHASE CONTINUITY CHECK ===")
print()

# For 440 Hz at 48000 Hz sample rate
f0 = 440.0
fs = 48000
cycle_len = fs / f0  # 109.09 samples per cycle

print(f"Target frequency: {f0} Hz")
print(f"Cycle length: {cycle_len:.2f} samples")
print()

# LUT parameters
lut = get_lut_for_note(wave_id=0, midi_note=69)
P = lut.P  # 64 phase steps

# The LUT maps phase_idx p to sample position: start(p) = round(p * cycle_len / P)
print(f"LUT P: {P}")
print(f"Samples per phase step: {cycle_len / P:.4f}")
print()

# Runtime phase increment
phase_inc = int((f0 / fs) * (1 << 32))
phase_inc_per_block = phase_inc * BLOCK_SIZE

print(f"Phase increment per sample: {phase_inc}")
print(f"Phase increment per block: {phase_inc_per_block}")
print()

# How many phase STEPS (P) does phase advance per block?
# phase_idx = (phase_32 >> 26) & 0x3F
# So phase_idx advances by: (phase_inc_per_block >> 26) mod 64
phase_idx_inc_per_block = (phase_inc_per_block >> 26) & 0x3F
print(f"Phase index increment per block: {phase_idx_inc_per_block}")
print()

# In samples, one phase step = cycle_len / P samples
samples_per_phase_step = cycle_len / P
samples_advanced_per_block_via_phase = phase_idx_inc_per_block * samples_per_phase_step

print(f"Samples advanced per block (via phase steps): {samples_advanced_per_block_via_phase:.2f}")
print(f"Samples per block (actual): {BLOCK_SIZE}")
print(f"Mismatch: {samples_advanced_per_block_via_phase - BLOCK_SIZE:.2f} samples")
print()

# Let's trace actual sample positions
print("=== SAMPLE POSITION TRACE ===")
print("block  phase_32        phase_idx  start(phase_idx)  expected_start  error")
print("-" * 80)

phase = 0
expected_start = 0
for block in range(10):
    phase_idx = (phase >> 26) & 0x3F
    # LUT start position
    lut_start = int(round(phase_idx * cycle_len / P)) % int(cycle_len)

    # Expected start (continuous playback)
    expected = int(expected_start) % int(cycle_len)

    error = (lut_start - expected + int(cycle_len/2)) % int(cycle_len) - int(cycle_len/2)

    print(f"{block:5d}  {phase:12d}  {phase_idx:9d}  {lut_start:16d}  {expected:14d}  {error:+6d}")

    # Advance
    phase = (phase + phase_inc_per_block) & 0xFFFFFFFF
    expected_start += BLOCK_SIZE

print()
print("=== INTERPRETATION ===")
print()
print(f"The phase advances by {phase_idx_inc_per_block} steps per block.")
print(f"Each step = {samples_per_phase_step:.2f} samples in LUT space.")
print(f"Total advance = {samples_advanced_per_block_via_phase:.2f} samples.")
print(f"But we need to advance by exactly {BLOCK_SIZE} samples.")
print()
print(f"Error per block: {samples_advanced_per_block_via_phase - BLOCK_SIZE:.2f} samples")
print(f"This error accumulates and creates periodic discontinuities!")
