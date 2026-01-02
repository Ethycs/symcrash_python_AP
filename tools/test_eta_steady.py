#!/usr/bin/env python3
"""Test spectrum after ETA has stabilized (skip initial ramp)."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES

def analyze_spectrum(eta_q15, label, skip_blocks=10):
    """Analyze spectrum at a given eta, skipping initial blocks."""
    frame = VoiceFrame(
        voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
        beta_target_q15=eta_q15,
        partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
        is_active=True,
    )
    state = VMState.create(0)
    state.reset_for_note_on(f0_hz=440.0)

    # Skip initial blocks to let eta stabilize
    for _ in range(skip_blocks):
        state, _ = vm_step_block(state, frame, DEFAULT_TABLES)

    # Now collect samples
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

    print(f"\n{label} (eta={eta_q15}, skipped {skip_blocks} blocks):")
    print(f"  Fundamental: {fund_freq:.1f} Hz")

    # Check 2200Hz
    idx_2200 = np.argmin(np.abs(freqs - 2200))
    db_2200 = 20 * np.log10(fft[idx_2200] / fund_mag + 1e-10)
    print(f"  2200 Hz: {db_2200:+.1f} dB")

    # Check 1320Hz
    idx_1320 = np.argmin(np.abs(freqs - 1320))
    db_1320 = 20 * np.log10(fft[idx_1320] / fund_mag + 1e-10)
    print(f"  1320 Hz: {db_1320:+.1f} dB")

    return db_2200

print("=" * 60)
print("Spectrum analysis with stabilized ETA (skip initial blocks)")
print("=" * 60)

# Compare with different skip amounts
for skip in [0, 1, 5, 10, 50]:
    analyze_spectrum(16384, f"50/50 mix (skip={skip})", skip_blocks=skip)
