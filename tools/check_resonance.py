#!/usr/bin/env python3
"""Check for tinny resonance - compare carrier vs wet path."""
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
    print(f"  Fundamental: {fund_freq:.1f} Hz")

    # Check harmonics and high frequency content
    print(f"  Harmonics:")
    for h in [2, 3, 4, 5, 8, 16]:
        idx = np.argmin(np.abs(freqs - fund_freq * h))
        if idx < len(fft):
            db = 20 * np.log10(fft[idx] / fund_mag + 1e-10)
            print(f"    H{h} ({fund_freq*h:.0f}Hz): {db:+.1f} dB")

    # Check for resonance peaks in high freq region (2-10kHz)
    print(f"  High freq peaks (2-10kHz):")
    hf_start = np.searchsorted(freqs, 2000)
    hf_end = np.searchsorted(freqs, 10000)
    hf_fft = fft[hf_start:hf_end]
    hf_freqs = freqs[hf_start:hf_end]

    # Find top 5 peaks
    peak_indices = np.argsort(hf_fft)[-5:][::-1]
    for i in peak_indices:
        db = 20 * np.log10(hf_fft[i] / fund_mag + 1e-10)
        if db > -60:
            print(f"    {hf_freqs[i]:.0f} Hz: {db:+.1f} dB")

# Test 1: Pure carrier (eta=0)
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
analyze(np.array(samples1), "Pure carrier (eta=0)")

# Test 2: Full wet (eta=32767)
print("\n" + "=" * 60)
print("Test 2: Full wet path (eta=32767)")
frame2 = VoiceFrame(
    voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
    beta_target_q15=32767,  # Full wet
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    is_active=True,
)
state2 = VMState.create(0)
state2.reset_for_note_on(f0_hz=440.0)
samples2 = []
for _ in range(500):
    state2, out = vm_step_block(state2, frame2, DEFAULT_TABLES)
    samples2.extend(out.tolist())
analyze(np.array(samples2), "Full wet (eta=32767)")

# Test 3: 50/50 mix (eta=16384)
print("\n" + "=" * 60)
print("Test 3: 50/50 mix (eta=16384)")
frame3 = VoiceFrame(
    voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
    beta_target_q15=16384,  # 50/50
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    is_active=True,
)
state3 = VMState.create(0)
state3.reset_for_note_on(f0_hz=440.0)
samples3 = []
for _ in range(500):
    state3, out = vm_step_block(state3, frame3, DEFAULT_TABLES)
    samples3.extend(out.tolist())
analyze(np.array(samples3), "50/50 mix (eta=16384)")
