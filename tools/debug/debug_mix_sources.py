#!/usr/bin/env python3
"""Debug what's being mixed in the 50/50 case."""
import sys
sys.path.insert(0, 'src')
import numpy as np

from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.vm.decode import decode_frame
from symcrash_python_ap.vm.lanes import lane_carrier_render, lane_idwpt_render
from symcrash_python_ap.vm.tables import DEFAULT_TABLES

# Create test frame
frame = VoiceFrame(
    voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
    beta_target_q15=16384,  # 50/50
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    is_active=True,
)

state = VMState.create(0)
state.reset_for_note_on(f0_hz=440.0)

# Decode to get plan
cfg = decode_frame(frame, DEFAULT_TABLES)

print("Configuration:")
print(f"  idwpt_enable: {cfg.idwpt_enable}")
print(f"  eta_curr_q15: {cfg.eta_curr_q15}")
print(f"  eta_prev_q15: {cfg.eta_prev_q15}")
print(f"  coeff_state sum: {np.sum(np.abs(state.coeff_state))}")
print(f"  mask_u32: {cfg.mask_u32:#x}")

# Render carrier
carrier = lane_carrier_render(state, cfg).copy()
print(f"\nCarrier first 10: {carrier[:10].tolist()}")

# Render wet (IDWPT) - test with effect_mode_jit directly
from symcrash_python_ap.wavelet.fast_lifting import effect_mode_jit

carrier_list = carrier.tolist()

# Test passthrough (no mask)
wet_passthrough = effect_mode_jit(carrier_list, None)
wet_passthrough_arr = np.array(wet_passthrough, dtype=np.int16)
print(f"\nWet (passthrough, mask=None) first 10: {wet_passthrough_arr[:10].tolist()}")

# Test with unity mask
wet_unity = effect_mode_jit(carrier_list, [32767]*64)
wet_unity_arr = np.array(wet_unity, dtype=np.int16)
print(f"Wet (unity mask) first 10: {wet_unity_arr[:10].tolist()}")

# Now use the actual lane function
wet = lane_idwpt_render(state, cfg, carrier).copy()
print(f"Wet (lane function) first 10: {wet[:10].tolist()}")

# Compare
diff = carrier.astype(np.int32) - wet.astype(np.int32)
print(f"\nCarrier - Wet diff:")
print(f"  max: {np.max(np.abs(diff))}")
print(f"  rms: {np.sqrt(np.mean(diff**2)):.2f}")
print(f"  first 10: {diff[:10].tolist()}")

# FFT of the difference
if np.any(diff != 0):
    fft_diff = np.abs(np.fft.rfft(diff.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(diff), 1/48000)

    # Find peaks
    peak_indices = np.argsort(fft_diff)[-5:][::-1]
    print(f"\nDifference spectrum peaks:")
    for idx in peak_indices:
        if fft_diff[idx] > 0.1:
            print(f"  {freqs[idx]:.1f} Hz: {fft_diff[idx]:.1f}")
