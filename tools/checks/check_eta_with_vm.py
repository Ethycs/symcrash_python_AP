#!/usr/bin/env python3
"""Check eta stability using full VM path."""
import sys
sys.path.insert(0, 'src')
import numpy as np

from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES

frame = VoiceFrame(
    voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
    beta_target_q15=16384,  # 50/50
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    is_active=True,
)

state = VMState.create(0)
state.reset_for_note_on(f0_hz=440.0)

print("Tracking eta_prev_q15 in VMState over blocks:")
for i in range(50):
    prev_eta = state.eta_prev_q15
    state, out = vm_step_block(state, frame, DEFAULT_TABLES)
    post_eta = state.eta_prev_q15

    if i < 10 or i >= 45:
        print(f"Block {i:3d}: eta_prev before={prev_eta:5d}, after={post_eta:5d}, blocks_since_start={state.blocks_since_start}")
    elif i == 10:
        print("...")
