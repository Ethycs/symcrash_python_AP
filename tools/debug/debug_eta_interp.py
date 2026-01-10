#!/usr/bin/env python3
"""Debug eta interpolation in vm_step_block."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.vm.decode import decode_frame
from symcrash_python_ap.vm.tables import DEFAULT_TABLES

# Create test frame with eta=16384 (50/50)
frame = VoiceFrame(
    voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
    beta_target_q15=16384,  # 50/50
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    is_active=True,
)

state = VMState.create(0)
state.reset_for_note_on(f0_hz=440.0)

# Decode multiple frames to see eta interpolation
print("Checking eta interpolation across blocks:")
print("=" * 60)

for block in range(10):
    cfg = decode_frame(frame, DEFAULT_TABLES,
                       prev_eta_q15=getattr(state, '_prev_eta_q15', 0),
                       prev_gain_q15=32767)

    print(f"Block {block}: eta_prev={cfg.eta_prev_q15:5d}, eta_curr={cfg.eta_curr_q15:5d}, "
          f"delta={cfg.eta_curr_q15 - cfg.eta_prev_q15:+6d}")

    # Store for next iteration
    state._prev_eta_q15 = cfg.eta_curr_q15

# Check what beta_target_q15 maps to eta
print("\n" + "=" * 60)
print("Checking beta_target_q15 to eta mapping:")

from symcrash_python_ap.vm.decode import decode_frame

for beta in [0, 8192, 16384, 24576, 32767]:
    frame_test = VoiceFrame(
        voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
        beta_target_q15=beta,
        partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
        is_active=True,
    )
    cfg = decode_frame(frame_test, DEFAULT_TABLES)
    print(f"  beta_target={beta:5d} -> eta_curr={cfg.eta_curr_q15:5d}")
