#!/usr/bin/env python3
"""Analyze where 2200Hz peak comes from in the mix."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES

def analyze(samples, label):
    fft = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1/48000)
    fund_idx = np.argmax(fft[10:]) + 10
    fund_mag = fft[fund_idx]
    fund_freq = freqs[fund_idx]

    print(f"\n{label}:")
    print(f"  Fundamental: {fund_freq:.1f} Hz at index {fund_idx}")

    # Check 2200Hz specifically
    idx_2200 = np.argmin(np.abs(freqs - 2200))
    db_2200 = 20 * np.log10(fft[idx_2200] / fund_mag + 1e-10)
    print(f"  2200 Hz: {db_2200:+.1f} dB")

    # Check 5x fundamental (should be ~2200 if fundamental is 440)
    idx_5x = np.argmin(np.abs(freqs - fund_freq * 5))
    db_5x = 20 * np.log10(fft[idx_5x] / fund_mag + 1e-10)
    print(f"  5x fund ({fund_freq*5:.0f} Hz): {db_5x:+.1f} dB")

    # What's the actual frequency at the 2200Hz peak?
    peak_region = fft[idx_2200-10:idx_2200+10]
    peak_freqs = freqs[idx_2200-10:idx_2200+10]
    local_max_idx = np.argmax(peak_region)
    actual_peak_freq = peak_freqs[local_max_idx]
    print(f"  Actual peak near 2200: {actual_peak_freq:.1f} Hz")

    return fund_freq

# Test pure carrier
print("=" * 60)
print("Test 1: Pure carrier (eta=0)")
frame1 = VoiceFrame(
    voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
    beta_target_q15=0,  # Pure carrier
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    is_active=True,
)
state1 = VMState.create(0)
state1.reset_for_note_on(f0_hz=440.0)
samples1 = []
for _ in range(500):
    state1, out = vm_step_block(state1, frame1, DEFAULT_TABLES)
    samples1.extend(out.tolist())
analyze(np.array(samples1), "Pure carrier")

# Test 50/50 mix
print("\n" + "=" * 60)
print("Test 2: 50/50 mix (eta=16384)")
frame2 = VoiceFrame(
    voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
    beta_target_q15=16384,  # 50/50
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    is_active=True,
)
state2 = VMState.create(0)
state2.reset_for_note_on(f0_hz=440.0)
samples2 = []
for _ in range(500):
    state2, out = vm_step_block(state2, frame2, DEFAULT_TABLES)
    samples2.extend(out.tolist())
analyze(np.array(samples2), "50/50 mix")

# Test: What happens if carrier and wet are exactly the same?
# If they're identical, the mix should be identical to either one
print("\n" + "=" * 60)
print("Test 3: Check carrier vs wet signal correlation")

# Reset and get both carrier and wet separately
from symcrash_python_ap.vm.lanes import lane_carrier_render, lane_idwpt_render
from symcrash_python_ap.vm.decode import decode_frame, CarrierRole

frame3 = VoiceFrame(
    voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
    beta_target_q15=16384,
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    is_active=True,
)
state3 = VMState.create(0)
state3.reset_for_note_on(f0_hz=440.0)

# Decode once to get plan
cfg = decode_frame(state3, frame3, DEFAULT_TABLES)

# Render carrier
carrier_buf = lane_carrier_render(state3, cfg)
print(f"Carrier first 5: {carrier_buf[:5].tolist()}")

# Render wet (IDWPT)
cfg.idwpt_enable = True
wet_buf = lane_idwpt_render(state3, cfg, carrier_buf)
print(f"Wet first 5: {wet_buf[:5].tolist()}")

# Compare
diff = np.abs(carrier_buf.astype(np.int32) - wet_buf.astype(np.int32))
print(f"Carrier vs Wet diff: max={diff.max()}, mean={diff.mean():.2f}")

# If there's a difference, that's the source of artifacts when mixing!
