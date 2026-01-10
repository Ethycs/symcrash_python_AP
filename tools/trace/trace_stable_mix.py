#!/usr/bin/env python3
"""Trace what happens during stable 50/50 mix."""
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

# Create and warm up state
frame = VoiceFrame(
    voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
    beta_target_q15=16384,  # 50/50
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    is_active=True,
)
state = VMState.create(0)
state.reset_for_note_on(f0_hz=440.0)

# Warmup
for _ in range(50):
    state, _ = vm_step_block(state, frame, DEFAULT_TABLES)

print(f"After warmup: eta_prev={state.eta_prev_q15}")

# Now trace one block in detail
cfg = decode_frame(frame, DEFAULT_TABLES, prev_eta_q15=state.eta_prev_q15, prev_gain_q15=state.gain_prev_q15)

print(f"\nConfiguration:")
print(f"  eta_prev_q15: {cfg.eta_prev_q15}")
print(f"  eta_curr_q15: {cfg.eta_curr_q15}")
print(f"  eta_delta: {cfg.eta_curr_q15 - cfg.eta_prev_q15}")
print(f"  gain_prev_q15: {cfg.gain_prev_q15}")
print(f"  gain_curr_q15: {cfg.gain_curr_q15}")

# Carrier render
carrier = lane_carrier_render(state, cfg).copy()
print(f"\nCarrier: min={carrier.min()}, max={carrier.max()}, mean={carrier.mean():.1f}")

# Wet render
wet = lane_idwpt_render(state, cfg, carrier).copy()
print(f"Wet: min={wet.min()}, max={wet.max()}, mean={wet.mean():.1f}")

# Check difference between carrier and wet
diff = carrier.astype(np.int32) - wet.astype(np.int32)
print(f"Diff (carrier-wet): max_abs={np.max(np.abs(diff))}, mean={diff.mean():.2f}, rms={np.sqrt(np.mean(diff**2)):.2f}")

# Check crossfade weights
eta_delta = cfg.eta_curr_q15 - cfg.eta_prev_q15
eta_vals = cfg.eta_prev_q15 + ((_ALPHA_Q15 * eta_delta) >> 15)
phases = ((eta_vals.astype(np.int64) * 0x40000000) >> 15).astype(np.uint32)
dry_weights, wet_weights = _sincos_batch_q15(phases)

print(f"\nCrossfade weights:")
print(f"  eta_vals: min={eta_vals.min()}, max={eta_vals.max()}")
print(f"  dry_weights: min={dry_weights.min()}, max={dry_weights.max()}, std={dry_weights.std():.2f}")
print(f"  wet_weights: min={wet_weights.min()}, max={wet_weights.max()}, std={wet_weights.std():.2f}")

# Verify equal-power property sample by sample
power_sum = (dry_weights.astype(np.int64)**2 + wet_weights.astype(np.int64)**2) >> 15
print(f"  equal-power check (should be 32767): min={power_sum.min()}, max={power_sum.max()}")

# Do the mix manually and check
carrier_i32 = carrier.astype(np.int32)
wet_i32 = wet.astype(np.int32)
mixed = (carrier_i32 * dry_weights + wet_i32 * wet_weights) >> 15

print(f"\nMixed: min={mixed.min()}, max={mixed.max()}, mean={mixed.mean():.1f}")

# Apply gain
gain_delta = cfg.gain_curr_q15 - cfg.gain_prev_q15
gain_vals = cfg.gain_prev_q15 + ((_ALPHA_Q15 * gain_delta) >> 15)
samples = (mixed * gain_vals) >> 15
print(f"After gain: min={samples.min()}, max={samples.max()}")

# Now compare: what if we just output carrier directly (no mix)?
carrier_with_gain = (carrier.astype(np.int32) * gain_vals) >> 15
print(f"Carrier with same gain: min={carrier_with_gain.min()}, max={carrier_with_gain.max()}")

# FFT comparison
fft_carrier = np.abs(np.fft.rfft(carrier.astype(np.float64)))
fft_wet = np.abs(np.fft.rfft(wet.astype(np.float64)))
fft_mixed = np.abs(np.fft.rfft(mixed.astype(np.float64)))
fft_diff = np.abs(np.fft.rfft(diff.astype(np.float64)))
freqs = np.fft.rfftfreq(64, 1/48000)

print("\nSpectrum comparison (at 0, 750, 1500, 2250, 3000 Hz):")
print("Freq | Carrier | Wet | Mixed | Diff")
for f in [0, 750, 1500, 2250, 3000]:
    idx = np.argmin(np.abs(freqs - f))
    print(f"{freqs[idx]:6.0f} | {fft_carrier[idx]:7.0f} | {fft_wet[idx]:7.0f} | {fft_mixed[idx]:7.0f} | {fft_diff[idx]:7.0f}")
