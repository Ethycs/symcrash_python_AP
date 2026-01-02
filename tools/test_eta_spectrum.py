#!/usr/bin/env python3
"""Test if ETA changes the spectrum."""
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

    print(f"\n{label} (eta={eta_q15}):")
    print(f"  Fundamental: {fund_freq:.1f} Hz, mag={fund_mag:.0f}")

    # Check 2200Hz
    idx_2200 = np.argmin(np.abs(freqs - 2200))
    db_2200 = 20 * np.log10(fft[idx_2200] / fund_mag + 1e-10)
    print(f"  2200 Hz: {db_2200:+.1f} dB (mag={fft[idx_2200]:.0f})")

    # Check H5 (5th harmonic)
    idx_h5 = np.argmin(np.abs(freqs - fund_freq * 5))
    db_h5 = 20 * np.log10(fft[idx_h5] / fund_mag + 1e-10)
    print(f"  H5 ({fund_freq*5:.0f} Hz): {db_h5:+.1f} dB")

    # Check top 5 non-fundamental peaks
    fft_masked = fft.copy()
    fft_masked[:20] = 0  # Remove DC
    fft_masked[fund_idx-10:fund_idx+10] = 0  # Remove fundamental

    print(f"  Top peaks (excl. fundamental):")
    for _ in range(3):
        idx = np.argmax(fft_masked)
        if fft_masked[idx] > 0:
            db = 20 * np.log10(fft_masked[idx] / fund_mag + 1e-10)
            print(f"    {freqs[idx]:.1f} Hz: {db:+.1f} dB")
            fft_masked[idx-5:idx+5] = 0

print("=" * 60)
print("Spectrum analysis at different ETA values")
print("=" * 60)

analyze_spectrum(0, "Pure carrier")
analyze_spectrum(16384, "50/50 mix")
analyze_spectrum(32767, "Full wet")
