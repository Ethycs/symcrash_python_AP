#!/usr/bin/env python3
"""Check if carrier and wet have phase mismatch."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.vm.decode import decode_frame
from symcrash_python_ap.vm.lanes import lane_carrier_render, lane_idwpt_render
from symcrash_python_ap.vm.tables import DEFAULT_TABLES

# Create test frame with eta=16384 (50/50)
frame = VoiceFrame(
    voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
    beta_target_q15=16384,
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    is_active=True,
)

state = VMState.create(0)
state.reset_for_note_on(f0_hz=440.0)

# Skip a few blocks to stabilize
cfg = decode_frame(frame, DEFAULT_TABLES, prev_eta_q15=0)
state.eta_prev_q15 = cfg.eta_curr_q15

# Collect multiple blocks
all_carrier = []
all_wet = []

for block in range(100):
    cfg = decode_frame(frame, DEFAULT_TABLES, prev_eta_q15=state.eta_prev_q15)

    carrier = lane_carrier_render(state, cfg).copy()
    wet = lane_idwpt_render(state, cfg, carrier).copy()

    all_carrier.extend(carrier.tolist())
    all_wet.extend(wet.tolist())

    state.eta_prev_q15 = cfg.eta_curr_q15

carrier_arr = np.array(all_carrier, dtype=np.float64)
wet_arr = np.array(all_wet, dtype=np.float64)

print("Carrier vs Wet comparison:")
print(f"  Carrier RMS: {np.sqrt(np.mean(carrier_arr**2)):.1f}")
print(f"  Wet RMS: {np.sqrt(np.mean(wet_arr**2)):.1f}")

# Cross-correlation to find phase offset
corr = np.correlate(carrier_arr[:1000], wet_arr[:1000], mode='full')
lag_idx = np.argmax(corr) - 999
print(f"  Phase lag (samples): {lag_idx}")
print(f"  Phase lag (degrees at 440Hz): {lag_idx * 440 * 360 / 48000:.1f}")

# Compute difference signal
diff = carrier_arr - wet_arr
print(f"\nDifference signal:")
print(f"  Max: {np.max(np.abs(diff)):.1f}")
print(f"  RMS: {np.sqrt(np.mean(diff**2)):.1f}")

# FFT of difference
fft_diff = np.abs(np.fft.rfft(diff))
freqs = np.fft.rfftfreq(len(diff), 1/48000)

# Find fund of carrier for reference
fft_carrier = np.abs(np.fft.rfft(carrier_arr))
fund_idx = np.argmax(fft_carrier[10:]) + 10
fund_mag = fft_carrier[fund_idx]

print(f"\nDifference spectrum (relative to carrier fundamental):")
for freq in [440, 880, 1320, 1760, 2200, 750, 1500]:
    idx = np.argmin(np.abs(freqs - freq))
    db = 20 * np.log10(fft_diff[idx] / fund_mag + 1e-10)
    print(f"  {freq:4d} Hz: {db:+.1f} dB")

# The key question: is the difference signal periodic?
print(f"\nDifference signal first 20 values:")
print(f"  {diff[:20].astype(int).tolist()}")
