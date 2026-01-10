#!/usr/bin/env python3
"""Check resonance - with explicit warmup period skipped."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES

WARMUP_BLOCKS = 50
ANALYZE_BLOCKS = 500

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

    peak_indices = np.argsort(hf_fft)[-5:][::-1]
    for i in peak_indices:
        db = 20 * np.log10(hf_fft[i] / fund_mag + 1e-10)
        if db > -60:
            print(f"    {hf_freqs[i]:.0f} Hz: {db:+.1f} dB")

# Test 50/50 mix with warmup
print("=" * 60)
print(f"Test: 50/50 mix with {WARMUP_BLOCKS} blocks warmup")
frame = VoiceFrame(
    voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
    beta_target_q15=16384,  # 50/50
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    is_active=True,
)
state = VMState.create(0)
state.reset_for_note_on(f0_hz=440.0)

# Warmup - don't collect audio
print(f"Running {WARMUP_BLOCKS} warmup blocks...")
for _ in range(WARMUP_BLOCKS):
    state, _ = vm_step_block(state, frame, DEFAULT_TABLES)

print(f"After warmup: eta_prev={state.eta_prev_q15}, blocks_since_start={state.blocks_since_start}")

# Now collect stable audio
samples = []
for _ in range(ANALYZE_BLOCKS):
    state, out = vm_step_block(state, frame, DEFAULT_TABLES)
    samples.extend(out.tolist())

analyze(np.array(samples), f"50/50 mix (after {WARMUP_BLOCKS} warmup)")

# Also test pure carrier with warmup
print("\n" + "=" * 60)
print(f"Test: Pure carrier with {WARMUP_BLOCKS} blocks warmup")
frame2 = VoiceFrame(
    voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
    beta_target_q15=0,  # Pure carrier
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    is_active=True,
)
state2 = VMState.create(0)
state2.reset_for_note_on(f0_hz=440.0)

for _ in range(WARMUP_BLOCKS):
    state2, _ = vm_step_block(state2, frame2, DEFAULT_TABLES)

samples2 = []
for _ in range(ANALYZE_BLOCKS):
    state2, out = vm_step_block(state2, frame2, DEFAULT_TABLES)
    samples2.extend(out.tolist())

analyze(np.array(samples2), f"Pure carrier (after {WARMUP_BLOCKS} warmup)")
