#!/usr/bin/env python3
"""Debug DC - look at actual waveform."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES

frame = VoiceFrame(
    voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
    beta_target_q15=0,
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    is_active=True,
)

state = VMState.create(0)
state.reset_for_note_on(f0_hz=440.0)

# Collect samples from several blocks
all_samples = []
for i in range(20):
    state, out = vm_step_block(state, frame, DEFAULT_TABLES)
    all_samples.append(out.copy())

# Look at block boundaries
for i in range(5):
    block = all_samples[i]
    next_block = all_samples[i+1]
    jump = next_block[0] - block[-1]
    print(f"Block {i} end: {block[-1]:+6d}, Block {i+1} start: {next_block[0]:+6d}, jump: {jump:+6d}")

# Check if there's a pattern in block means
print("\nBlock means:")
for i, block in enumerate(all_samples[:10]):
    print(f"  Block {i}: mean={np.mean(block):+8.1f}, min={block.min():+6d}, max={block.max():+6d}")

# The phase at 440Hz: 64 samples at 48kHz = 64/48000 sec = 1.33ms
# 440Hz period = 1/440 = 2.27ms
# So each block is 1.33/2.27 = 0.59 of a cycle = 58.7% of a cycle
# This means the waveform doesn't complete an integer number of cycles per block
# The "DC" of each block depends on which phase we sample
print(f"\n440Hz at 48kHz: {48000/440:.2f} samples/cycle")
print(f"Block size 64 samples = {64*440/48000:.3f} cycles")
print("Non-integer cycles per block causes apparent DC to vary")

# The real question: over many complete cycles, what's the DC?
all_flat = np.concatenate(all_samples)
# Find a point where phase ~ 0 (zero crossing going up)
diffs = np.diff(all_flat)
zero_crossings = np.where((all_flat[:-1] < 0) & (all_flat[1:] >= 0) & (diffs > 0))[0]
if len(zero_crossings) > 10:
    # Measure from one zero crossing to another (multiple complete cycles)
    start = zero_crossings[0]
    end = zero_crossings[-1]
    complete_cycles = all_flat[start:end]
    print(f"\n{len(zero_crossings)-1} complete cycles from sample {start} to {end}")
    print(f"DC of complete cycles: {np.mean(complete_cycles):.2f} LSB")
