#!/usr/bin/env python3
"""Check raw sine LUT output before any processing."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from symcrash_python_ap.wavelet.sine_core import sine_q15_vectorized

# Generate 440Hz sine using the same method as lane_carrier_render
sample_rate = 48000
f0 = 440.0
phase_inc = int(f0 * (2**32) / sample_rate)

print(f"Phase increment: {phase_inc} (= 0x{phase_inc:08x})")

# Generate phases for 100 samples
phases_32 = np.array([phase_inc * i for i in range(100)], dtype=np.int64) & 0xFFFFFFFF
phases_16 = ((phases_32 >> 16) & 0xFFFF).astype(np.int32)

# Lookup sines
sines = sine_q15_vectorized(phases_16)

print(f"\nRaw sine values first 10: {sines[:10].tolist()}")
print(f"Raw sine max: {sines.max()}, min: {sines.min()}")

# FFT to check purity
fft = np.abs(np.fft.rfft(sines.astype(np.float64)))
freqs = np.fft.rfftfreq(len(sines), 1/sample_rate)
fund_idx = np.argmax(fft[1:]) + 1
fund_mag = fft[fund_idx]

print(f"\nRaw sine spectrum:")
print(f"  Fundamental: {freqs[fund_idx]:.1f} Hz")
for h in [2, 3, 4, 5]:
    idx = np.argmin(np.abs(freqs - freqs[fund_idx] * h))
    if idx < len(fft):
        db = 20 * np.log10(fft[idx] / fund_mag + 1e-10)
        print(f"  H{h}: {db:+.1f} dB")

# Compare with expected
expected = (32767 * np.sin(2 * np.pi * 440 * np.arange(100) / 48000)).astype(np.int32)
print(f"\nExpected first 10: {expected[:10].tolist()}")
print(f"Difference first 10: {(sines[:10] - expected[:10]).tolist()}")
