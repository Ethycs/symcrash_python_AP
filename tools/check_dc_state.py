#!/usr/bin/env python3
"""Check DC filter state over time."""
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

print("DC filter state over time:")
print("Block |   dc_x1   |   dc_y1   | Output mean | Output max | Output min")
print("-" * 75)

for block in range(20):
    state, out = vm_step_block(state, frame, DEFAULT_TABLES)
    mean = np.mean(out)
    print(f"{block:5d} | {state.dc_block_x1:9d} | {state.dc_block_y1:9d} | "
          f"{mean:11.1f} | {out.max():10d} | {out.min():10d}")

# Run more blocks to stabilize
for _ in range(1000):
    state, out = vm_step_block(state, frame, DEFAULT_TABLES)

print(f"\nAfter 1020 blocks:")
print(f"  dc_x1: {state.dc_block_x1}")
print(f"  dc_y1: {state.dc_block_y1}")
print(f"  Output mean: {np.mean(out):.2f}")
print(f"  Output max: {out.max()}, min: {out.min()}")
