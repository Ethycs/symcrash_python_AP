#!/usr/bin/env python3
"""Debug wet path - what's being mixed?"""
import sys
sys.path.insert(0, 'src')
import numpy as np
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES

# Test: 50/50 mix (eta=16384)
frame = VoiceFrame(
    voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
    beta_target_q15=16384,  # 50/50
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    is_active=True,
)

state = VMState.create(0)
state.reset_for_note_on(f0_hz=440.0)

# Check coeff_state
print(f"Initial coeff_state sum: {np.sum(np.abs(state.coeff_state))}")
print(f"First 10 coeff_state values: {state.coeff_state[:10].tolist()}")

# Render a few blocks and check what's happening
samples = []
for i in range(5):
    state, out = vm_step_block(state, frame, DEFAULT_TABLES)
    samples.append(out.copy())
    if i == 0:
        print(f"\nAfter block 0:")
        print(f"  coeff_state sum: {np.sum(np.abs(state.coeff_state))}")
        print(f"  First 10 coeff_state: {state.coeff_state[:10].tolist()}")
        print(f"  Output first 10: {out[:10].tolist()}")

# Check FFT of first few blocks
all_samples = np.concatenate(samples)
fft = np.abs(np.fft.rfft(all_samples))
freqs = np.fft.rfftfreq(len(all_samples), 1/48000)
fund_idx = np.argmax(fft[5:]) + 5
fund_mag = fft[fund_idx]
print(f"\nFFT of first 5 blocks:")
print(f"  Fundamental: {freqs[fund_idx]:.1f} Hz")

# Check 2200Hz peak
idx_2200 = np.argmin(np.abs(freqs - 2200))
db_2200 = 20 * np.log10(fft[idx_2200] / fund_mag + 1e-10)
print(f"  2200 Hz: {db_2200:+.1f} dB")
