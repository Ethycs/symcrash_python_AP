#!/usr/bin/env python3
"""Test pure wet path spectrum - does IDWPT generate harmonics?"""
import sys
sys.path.insert(0, 'src')
import numpy as np
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES

def analyze_spectrum(eta_q15, label):
    """Analyze spectrum at a given eta."""
    frame = VoiceFrame(
        voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
        beta_target_q15=eta_q15,
        partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],  # Pure sine
        is_active=True,
    )
    state = VMState.create(0)
    state.reset_for_note_on(f0_hz=440.0)

    # Skip to stabilize
    for _ in range(10):
        state, _ = vm_step_block(state, frame, DEFAULT_TABLES)

    samples = []
    for _ in range(500):
        state, out = vm_step_block(state, frame, DEFAULT_TABLES)
        samples.extend(out.tolist())

    samples = np.array(samples, dtype=np.float64)
    fft = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1/48000)

    fund_idx = np.argmax(fft[10:]) + 10
    fund_mag = fft[fund_idx]
    fund_freq = freqs[fund_idx]

    print(f"\n{label}:")
    print(f"  Fundamental: {fund_freq:.1f} Hz")

    # Check harmonics
    for h in [2, 3, 4, 5, 6, 7, 8]:
        idx = np.argmin(np.abs(freqs - fund_freq * h))
        db = 20 * np.log10(fft[idx] / fund_mag + 1e-10)
        print(f"  H{h} ({fund_freq*h:.0f} Hz): {db:+.1f} dB")

print("=" * 60)
print("Spectrum analysis: Pure carrier vs Pure wet")
print("=" * 60)

analyze_spectrum(0, "Pure carrier (eta=0)")
analyze_spectrum(32767, "Pure wet (eta=32767)")
