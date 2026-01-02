#!/usr/bin/env python3
"""Check eta stability after warmup."""
import sys
sys.path.insert(0, 'src')
import numpy as np

from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.vm.decode import decode_frame
from symcrash_python_ap.vm.lanes import lane_carrier_render, lane_idwpt_render, lane_mix_gain
from symcrash_python_ap.vm.tables import DEFAULT_TABLES

frame = VoiceFrame(
    voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
    beta_target_q15=16384,  # 50/50
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    is_active=True,
)

state = VMState.create(0)
state.reset_for_note_on(f0_hz=440.0)

print("Tracking eta_prev and eta_curr over blocks:")
for i in range(50):
    cfg = decode_frame(frame, DEFAULT_TABLES)
    carrier = lane_carrier_render(state, cfg).copy()
    wet = lane_idwpt_render(state, cfg, carrier)
    _ = lane_mix_gain(cfg, carrier, wet)

    if i < 10 or i >= 45:
        print(f"Block {i:3d}: eta_prev={cfg.eta_prev_q15:5d}, eta_curr={cfg.eta_curr_q15:5d}, delta={cfg.eta_curr_q15 - cfg.eta_prev_q15:5d}")
    elif i == 10:
        print("...")
