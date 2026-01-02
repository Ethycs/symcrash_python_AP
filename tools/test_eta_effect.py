#!/usr/bin/env python3
"""Test if ETA actually changes the mix."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES

def get_rms(eta_q15):
    """Get RMS of output at a given eta."""
    frame = VoiceFrame(
        voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
        beta_target_q15=eta_q15,
        partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
        is_active=True,
    )
    state = VMState.create(0)
    state.reset_for_note_on(f0_hz=440.0)

    samples = []
    for _ in range(100):
        state, out = vm_step_block(state, frame, DEFAULT_TABLES)
        samples.extend(out.tolist())

    return np.sqrt(np.mean(np.array(samples, dtype=np.float64)**2))

print("Testing ETA effect on output:")
for eta in [0, 8192, 16384, 24576, 32767]:
    rms = get_rms(eta)
    print(f"  eta={eta:5d}: RMS={rms:.1f}")

# More detailed test - check actual carrier vs wet weights
print("\n" + "=" * 60)
print("Testing crossfade weights directly:")

from symcrash_python_ap.vm.lanes import _sincos_batch_q15
import numpy as np

for eta_q15 in [0, 8192, 16384, 24576, 32767]:
    phases = np.array([((eta_q15 * 0x40000000) >> 15)], dtype=np.uint32)
    cos_vals, sin_vals = _sincos_batch_q15(phases)

    # cos = dry weight, sin = wet weight
    dry_weight = cos_vals[0]
    wet_weight = sin_vals[0]
    power = (dry_weight**2 + wet_weight**2) >> 15

    print(f"  eta={eta_q15:5d}: dry={dry_weight:5d}, wet={wet_weight:5d}, power={power}")

# Test: what does lane_mix_gain actually do?
print("\n" + "=" * 60)
print("Testing lane_mix_gain directly:")

from symcrash_python_ap.vm.lanes import lane_mix_gain
from symcrash_python_ap.vm.decode import decode_frame, DecodedPlan

# Create carrier and wet buffers
carrier = np.array([10000] * 64, dtype=np.int16)
wet = np.array([5000] * 64, dtype=np.int16)

# Test with different eta values
for eta in [0, 16384, 32767]:
    cfg = DecodedPlan()
    cfg.eta_prev_q15 = eta
    cfg.eta_curr_q15 = eta
    cfg.gain_prev_q15 = 32767
    cfg.gain_curr_q15 = 32767

    output = lane_mix_gain(cfg, carrier, wet)
    print(f"  eta={eta:5d}: carrier=10000, wet=5000 -> output[0]={output[0]}")
