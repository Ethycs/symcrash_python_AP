#!/usr/bin/env python3
"""Analyze multi-block mixing to find artifact source."""
import sys
sys.path.insert(0, 'src')
import numpy as np

from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.vm.decode import decode_frame
from symcrash_python_ap.vm.lanes import lane_carrier_render, lane_idwpt_render, lane_mix_gain
from symcrash_python_ap.vm.tables import DEFAULT_TABLES

# Warm up for eta stabilization
NUM_WARMUP = 32
NUM_BLOCKS = 128
BLOCK_SIZE = 64

def run_test(eta_value):
    """Run multi-block test at given eta value."""
    frame = VoiceFrame(
        voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
        beta_target_q15=eta_value,
        partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
        is_active=True,
    )

    state = VMState.create(0)
    state.reset_for_note_on(f0_hz=440.0)

    # Warm up (let eta stabilize)
    for _ in range(NUM_WARMUP):
        cfg = decode_frame(frame, DEFAULT_TABLES)
        carrier = lane_carrier_render(state, cfg).copy()
        wet = lane_idwpt_render(state, cfg, carrier)
        _ = lane_mix_gain(cfg, carrier, wet)

    # Collect audio
    carrier_all = []
    wet_all = []
    mixed_all = []

    for _ in range(NUM_BLOCKS):
        cfg = decode_frame(frame, DEFAULT_TABLES)
        carrier = lane_carrier_render(state, cfg).copy()
        wet = lane_idwpt_render(state, cfg, carrier).copy()
        mixed = lane_mix_gain(cfg, carrier, wet).copy()

        carrier_all.append(carrier)
        wet_all.append(wet)
        mixed_all.append(mixed)

    carrier_arr = np.concatenate(carrier_all)
    wet_arr = np.concatenate(wet_all)
    mixed_arr = np.concatenate(mixed_all)

    return carrier_arr, wet_arr, mixed_arr

def analyze_spectrum(arr, name, sr=48000):
    """Analyze spectrum and find peaks."""
    fft = np.abs(np.fft.rfft(arr.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(arr), 1/sr)

    # Find fundamental peak
    fund_idx = np.argmax(fft[1:]) + 1
    fund_mag = fft[fund_idx]

    # Find top 10 peaks
    peaks = np.argsort(fft)[-10:][::-1]

    print(f"\n{name} spectrum:")
    print(f"  Fundamental: {freqs[fund_idx]:.1f} Hz, mag={fund_mag:.0f}")
    print(f"  Top peaks (freq, dB rel):")
    for idx in peaks:
        if fft[idx] > fund_mag * 0.001:  # Above -60 dB
            db = 20 * np.log10(fft[idx] / fund_mag + 1e-10)
            print(f"    {freqs[idx]:8.1f} Hz: {db:6.1f} dB")

# Test at 50/50 mix
print("Testing at eta=16384 (50/50 mix)...")
carrier, wet, mixed = run_test(16384)

analyze_spectrum(carrier, "Carrier")
analyze_spectrum(wet, "Wet")
analyze_spectrum(mixed, "Mixed")

# Analyze the difference signal
diff = carrier.astype(np.int32) - wet.astype(np.int32)
print(f"\nCarrier - Wet difference:")
print(f"  max: {np.max(np.abs(diff))}")
print(f"  mean: {np.mean(diff):.2f}")
print(f"  rms: {np.sqrt(np.mean(diff**2)):.2f}")

# FFT of difference
fft_diff = np.abs(np.fft.rfft(diff.astype(np.float64)))
freqs = np.fft.rfftfreq(len(diff), 1/48000)
fft_mixed = np.abs(np.fft.rfft(mixed.astype(np.float64)))
fund_mag = np.max(fft_mixed)

print(f"\nDifference spectrum peaks (rel to mixed fundamental):")
peaks = np.argsort(fft_diff)[-10:][::-1]
for idx in peaks:
    if fft_diff[idx] > 10:
        db = 20 * np.log10(fft_diff[idx] / fund_mag + 1e-10)
        print(f"  {freqs[idx]:8.1f} Hz: {db:6.1f} dB (abs={fft_diff[idx]:.0f})")
