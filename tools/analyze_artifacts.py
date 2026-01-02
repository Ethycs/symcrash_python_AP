#!/usr/bin/env python3
"""Analyze the source of high-frequency artifacts in the wet path."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from symcrash_python_ap.wavelet.fast_lifting import effect_mode_jit

# Test at different frequencies to understand the pattern
test_freqs = [220, 440, 880, 1760]

print("Testing FWPT->IDWPT round-trip error spectrum:")
print("=" * 60)

for f0 in test_freqs:
    # Generate 1000 blocks of signal
    all_samples = []
    for block_idx in range(1000):
        carrier = [int(32767 * np.sin(2 * np.pi * f0 * (block_idx * 64 + i) / 48000))
                   for i in range(64)]
        # Apply IDWPT with no mask (passthrough)
        wet = effect_mode_jit(carrier, None)
        all_samples.extend(wet)

    samples = np.array(all_samples, dtype=np.float64)

    # FFT analysis
    fft = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1/48000)

    # Find fundamental
    fund_idx = np.argmin(np.abs(freqs - f0))
    fund_mag = fft[fund_idx]

    print(f"\nCarrier at {f0} Hz:")

    # Find top 5 non-fundamental peaks
    # Mask out fundamental region
    mask_width = 20
    fft_masked = fft.copy()
    fft_masked[max(0, fund_idx-mask_width):fund_idx+mask_width+1] = 0
    fft_masked[:10] = 0  # Remove DC region

    top_indices = np.argsort(fft_masked)[-5:][::-1]
    for idx in top_indices:
        if fft[idx] > fund_mag * 0.0001:  # > -80dB
            db = 20 * np.log10(fft[idx] / fund_mag + 1e-10)
            ratio = freqs[idx] / f0
            print(f"  {freqs[idx]:7.1f} Hz ({ratio:.2f}x f0): {db:+.1f} dB")

# Check if artifacts are due to block boundary effects
print("\n" + "=" * 60)
print("Checking block boundary artifacts (at 750Hz = 48000/64):")
block_freq = 48000 / 64
for f0 in [440]:
    all_samples = []
    for block_idx in range(1000):
        carrier = [int(32767 * np.sin(2 * np.pi * f0 * (block_idx * 64 + i) / 48000))
                   for i in range(64)]
        wet = effect_mode_jit(carrier, None)
        all_samples.extend(wet)

    samples = np.array(all_samples, dtype=np.float64)
    fft = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1/48000)
    fund_idx = np.argmin(np.abs(freqs - f0))
    fund_mag = fft[fund_idx]

    # Check block boundary frequency and harmonics
    for mult in [1, 2, 3, 4, 5]:
        check_freq = block_freq * mult
        idx = np.argmin(np.abs(freqs - check_freq))
        db = 20 * np.log10(fft[idx] / fund_mag + 1e-10)
        print(f"  {check_freq:6.0f} Hz (block x{mult}): {db:+.1f} dB")

    # Also check intermodulation: f0 ± block_freq
    for sign, name in [(1, "+"), (-1, "-")]:
        for mult in [1, 2, 3]:
            check_freq = f0 + sign * block_freq * mult
            if check_freq > 0:
                idx = np.argmin(np.abs(freqs - check_freq))
                db = 20 * np.log10(fft[idx] / fund_mag + 1e-10)
                print(f"  {check_freq:6.0f} Hz (f0 {name} block x{mult}): {db:+.1f} dB")
