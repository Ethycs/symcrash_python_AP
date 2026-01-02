#!/usr/bin/env python3
"""Test spectrum at very low ETA values."""
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
        partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
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

    # Check key frequencies
    results = {}
    for freq in [1320, 2200]:
        idx = np.argmin(np.abs(freqs - freq))
        db = 20 * np.log10(fft[idx] / fund_mag + 1e-10)
        results[freq] = db

    print(f"eta={eta_q15:5d} ({eta_q15/32767*100:5.1f}%): 1320Hz={results[1320]:+.1f} dB, 2200Hz={results[2200]:+.1f} dB")
    return results

print("=" * 60)
print("Spectrum at various low ETA values")
print("=" * 60)

for eta in [0, 100, 328, 656, 1000, 3277, 6554, 16384, 32767]:
    analyze_spectrum(eta, f"eta={eta}")
