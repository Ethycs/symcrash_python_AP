#!/usr/bin/env python3
"""Check if clipping is causing artifacts."""
import sys
sys.path.insert(0, 'src')
import numpy as np

from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.decode import decode_frame
from symcrash_python_ap.vm.lanes import (
    lane_carrier_render, lane_idwpt_render, lane_mix_gain,
    _sincos_batch_q15, _ALPHA_Q15
)
from symcrash_python_ap.vm.tables import DEFAULT_TABLES

# Warmup
frame = VoiceFrame(
    voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
    beta_target_q15=16384,
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    is_active=True,
)
state = VMState.create(0)
state.reset_for_note_on(f0_hz=440.0)

for _ in range(50):
    state, _ = vm_step_block(state, frame, DEFAULT_TABLES)

# Track clipping over many blocks
NUM_BLOCKS = 500
clipping_count = 0
clip_samples = []
max_pre_clip = 0
total_samples = 0

for i in range(NUM_BLOCKS):
    cfg = decode_frame(frame, DEFAULT_TABLES, prev_eta_q15=state.eta_prev_q15, prev_gain_q15=state.gain_prev_q15)
    carrier = lane_carrier_render(state, cfg).copy()
    wet = lane_idwpt_render(state, cfg, carrier).copy()

    # Do mix manually to see pre-clipping values
    eta_vals = np.full(64, cfg.eta_curr_q15, dtype=np.int32)  # stable eta
    phases = ((eta_vals.astype(np.int64) * 0x40000000) >> 15).astype(np.uint32)
    dry_weights, wet_weights = _sincos_batch_q15(phases)

    carrier_i32 = carrier.astype(np.int32)
    wet_i32 = wet.astype(np.int32)
    mixed = (carrier_i32 * dry_weights + wet_i32 * wet_weights) >> 15

    # Apply gain
    gain_vals = np.full(64, cfg.gain_curr_q15, dtype=np.int32)
    samples = (mixed * gain_vals) >> 15

    # Check for values that would clip
    pre_clip = samples.copy()
    max_pre_clip = max(max_pre_clip, np.max(np.abs(pre_clip)))

    clips = (samples > 32767) | (samples < -32768)
    if np.any(clips):
        clipping_count += np.sum(clips)
        clip_samples.append(samples[clips].tolist())

    total_samples += 64

    # Update state
    state.eta_prev_q15 = cfg.eta_curr_q15
    state.gain_prev_q15 = cfg.gain_curr_q15

print(f"Total samples: {total_samples}")
print(f"Max pre-clip value: {max_pre_clip}")
print(f"Clipping count: {clipping_count} ({100*clipping_count/total_samples:.2f}%)")

if clip_samples:
    all_clips = [v for s in clip_samples for v in s]
    print(f"Clip values range: {min(all_clips)} to {max(all_clips)}")

# Now check what carrier amplitude is before mixing
print(f"\nCarrier amplitude: min={carrier.min()}, max={carrier.max()}")
print(f"Wet amplitude: min={wet.min()}, max={wet.max()}")
print(f"After equal-power mix (carrier≈wet): amplitude should be sqrt(2)x = {carrier.max() * 1.414:.0f}")
