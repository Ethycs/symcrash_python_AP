#!/usr/bin/env python3
"""Debug DC offset issue."""
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

# Track DC over time
dc_values = []
for i in range(1000):
    state, out = vm_step_block(state, frame, DEFAULT_TABLES)
    dc = np.mean(out)
    dc_values.append(dc)
    if i < 10 or i % 100 == 0:
        print(f"Block {i:4d}: DC={dc:+7.2f}, dc_x1={state.dc_block_x1:+8d}, dc_y1={state.dc_block_y1:+8d}")

print(f"\nFinal DC average: {np.mean(dc_values[-100:]):.2f} LSB")
print(f"DC filter state: x1={state.dc_block_x1}, y1={state.dc_block_y1}")

# Check the raw sine before DC filter
from symcrash_python_ap.wavelet.sine_core import sine_q15_vectorized
phases = np.arange(64, dtype=np.int32) * 39370  # ~440Hz phase inc
phases = (phases >> 16) & 0xFFFF  # To 16-bit for LUT
sines = sine_q15_vectorized(phases)
print(f"\nRaw sine mean: {np.mean(sines):.2f}")
print(f"Raw sine sum: {np.sum(sines)}")
