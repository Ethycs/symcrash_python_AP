#!/usr/bin/env python3
"""Hunt down the hum in the oscillator."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES

# Render a long sample for low-freq analysis
frame = VoiceFrame(
    voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
    beta_target_q15=0,  # Pure carrier
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    is_active=True,
)

state = VMState.create(0)
state.reset_for_note_on(f0_hz=440.0)

# Render 10 seconds worth
samples = []
for _ in range(7500):  # 7500 blocks * 64 samples = 480000 samples = 10 sec
    state, out = vm_step_block(state, frame, DEFAULT_TABLES)
    samples.extend(out.tolist())

samples = np.array(samples, dtype=np.float64)
print(f"Rendered {len(samples)} samples ({len(samples)/48000:.1f} sec)")

# Check DC offset
dc_offset = np.mean(samples)
print(f"\nDC offset: {dc_offset:.2f} LSB ({dc_offset/32767*100:.4f}%)")

# Check for low-frequency content (hum)
fft = np.abs(np.fft.rfft(samples))
freqs = np.fft.rfftfreq(len(samples), 1/48000)

# Find fundamental
fund_idx = np.argmax(fft[10:]) + 10  # Skip DC region
fund_mag = fft[fund_idx]
fund_freq = freqs[fund_idx]
print(f"Fundamental: {fund_freq:.1f} Hz")

# Check specific hum frequencies
print("\nLow frequency content (potential hum):")
for freq in [50, 60, 100, 120, 150, 180, 240]:
    idx = np.argmin(np.abs(freqs - freq))
    if idx < len(fft):
        db = 20 * np.log10(fft[idx] / fund_mag + 1e-10)
        print(f"  {freq:3d} Hz: {db:+6.1f} dB")

# Check block boundary frequency (48000/64 = 750 Hz)
print("\nBlock boundary artifacts:")
for freq in [750, 1500, 2250]:
    idx = np.argmin(np.abs(freqs - freq))
    db = 20 * np.log10(fft[idx] / fund_mag + 1e-10)
    print(f"  {freq:4d} Hz: {db:+6.1f} dB")

# Check for any suspicious peaks below 200 Hz (excluding fundamental vicinity)
print("\nPeaks below 200 Hz (excluding fund):")
low_freqs = freqs[freqs < 200]
low_fft = fft[:len(low_freqs)]
# Find peaks
for i in range(1, len(low_fft)-1):
    if low_fft[i] > low_fft[i-1] and low_fft[i] > low_fft[i+1]:
        if low_fft[i] > fund_mag * 0.0001:  # > -80dB
            db = 20 * np.log10(low_fft[i] / fund_mag + 1e-10)
            print(f"  {low_freqs[i]:.1f} Hz: {db:+.1f} dB")
