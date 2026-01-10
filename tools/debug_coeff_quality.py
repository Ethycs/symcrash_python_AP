"""Debug coefficient quality - why can't FWPT represent H5?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sin_cos_q15, PHASE_MASK
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit

BLOCK_SIZE = 64
SAMPLE_RATE = 48000

print("=" * 70)
print("COEFFICIENT QUALITY ANALYSIS")
print("=" * 70)

def analyze_signal(signal, label):
    """Analyze FWPT coefficients and reconstruction."""
    coeffs = np.array(fwpt_jit(signal, headroom_bits=1), dtype=np.int16)
    recon = np.array(idwpt_jit(coeffs.tolist(), headroom_bits=1), dtype=np.int16)

    signal_arr = np.array(signal, dtype=np.int16)
    err = signal_arr.astype(np.int32) - recon.astype(np.int32)

    # Energy analysis
    signal_energy = np.sum(signal_arr.astype(np.float64)**2)
    coeff_energy = np.sum(coeffs.astype(np.float64)**2)
    recon_energy = np.sum(recon.astype(np.float64)**2)

    print(f"\n{label}:")
    print(f"  Signal energy: {signal_energy:.0f}")
    print(f"  Coeff energy:  {coeff_energy:.0f} (ratio: {coeff_energy/signal_energy:.3f})")
    print(f"  Recon energy:  {recon_energy:.0f} (ratio: {recon_energy/signal_energy:.3f})")
    print(f"  Roundtrip RMS: {np.sqrt(np.mean(err**2)):.1f}")
    print(f"  Roundtrip Max: {np.max(np.abs(err))}")

    # Check if any coefficients are saturated
    saturated = np.sum(np.abs(coeffs) >= 32767)
    print(f"  Saturated coeffs: {saturated}")

    return coeffs, recon, err

# Test different frequencies
freqs = [
    (440, "H1 (440 Hz, 0.59 cyc)"),
    (880, "H2 (880 Hz, 1.17 cyc)"),
    (1320, "H3 (1320 Hz, 1.76 cyc)"),
    (1760, "H4 (1760 Hz, 2.35 cyc)"),
    (2200, "H5 (2200 Hz, 2.93 cyc)"),
    (750*3, "3.0 cycles (2250 Hz)"),
    (2640, "H6 (2640 Hz, 3.52 cyc)"),
]

for freq, label in freqs:
    phase_inc = freq_to_phase_inc(freq, SAMPLE_RATE)
    signal = []
    phase = 0
    for n in range(BLOCK_SIZE):
        sin_val, _ = sin_cos_q15(phase)
        signal.append(sin_val)
        phase = (phase + phase_inc) & PHASE_MASK

    coeffs, recon, err = analyze_signal(signal, label)

# Deep dive on H5
print("\n" + "=" * 70)
print("H5 DEEP DIVE")
print("=" * 70)

freq = 2200
phase_inc = freq_to_phase_inc(freq, SAMPLE_RATE)
signal = []
phase = 0
for n in range(BLOCK_SIZE):
    sin_val, _ = sin_cos_q15(phase)
    signal.append(sin_val)
    phase = (phase + phase_inc) & PHASE_MASK

signal_arr = np.array(signal, dtype=np.int16)
coeffs = np.array(fwpt_jit(signal, headroom_bits=1), dtype=np.int16)
recon = np.array(idwpt_jit(coeffs.tolist(), headroom_bits=1), dtype=np.int16)
err = signal_arr.astype(np.int32) - recon.astype(np.int32)

print("\nSample comparison (first 16):")
print("  n  | Signal | Recon  | Error")
print("-" * 35)
for n in range(16):
    print(f"  {n:2d} | {signal_arr[n]:6d} | {recon[n]:6d} | {err[n]:6d}")

print("\nSample comparison (last 16):")
print("  n  | Signal | Recon  | Error")
print("-" * 35)
for n in range(48, 64):
    print(f"  {n:2d} | {signal_arr[n]:6d} | {recon[n]:6d} | {err[n]:6d}")

# Check boundary values
print(f"\nBoundary analysis:")
print(f"  signal[0] = {signal_arr[0]}, signal[63] = {signal_arr[63]}")
print(f"  recon[0]  = {recon[0]}, recon[63]  = {recon[63]}")
print(f"  For periodic continuation: signal[64] would be ~{signal_arr[0] + (signal_arr[1] - signal_arr[0])}")
