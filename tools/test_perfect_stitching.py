"""Test A: Perfect stitching - does FWPT/IDWPT on consecutive blocks produce clean sine?"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from scipy.fft import rfft, rfftfreq
from scipy.signal import windows

from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit

SAMPLE_RATE = 48000
BLOCK_SIZE = 64
f0 = 440.0

print("=== TEST A: PERFECT STITCHING ===")
print("Generate sine -> FWPT each block -> IDWPT each block -> check for spurs")
print()

# Generate 1 second of pure sine
duration = 1.0
n_samples = int(duration * SAMPLE_RATE)
t = np.arange(n_samples) / SAMPLE_RATE
sine_input = (np.sin(2 * np.pi * f0 * t) * 16000).astype(np.int16)

print(f"Input: {f0} Hz sine, {n_samples} samples")

# Process block by block: FWPT -> IDWPT
output = np.zeros(n_samples, dtype=np.int16)
n_blocks = n_samples // BLOCK_SIZE

for b in range(n_blocks):
    start = b * BLOCK_SIZE
    end = start + BLOCK_SIZE
    block = sine_input[start:end]

    # Forward transform
    coeffs = fwpt_jit(block.tolist(), headroom_bits=1)

    # Inverse transform
    recon = idwpt_jit(coeffs, headroom_bits=1)

    output[start:end] = np.array(recon, dtype=np.int16)

# Check reconstruction error
max_error = np.max(np.abs(sine_input[:n_blocks*BLOCK_SIZE].astype(np.int32) -
                          output[:n_blocks*BLOCK_SIZE].astype(np.int32)))
print(f"Max reconstruction error: {max_error} LSB")

# FFT analysis of output
print()
print("=== FFT of reconstructed signal ===")
window = windows.blackmanharris(len(output))
windowed = output.astype(np.float64) * window
spectrum = np.abs(rfft(windowed))
freqs = rfftfreq(len(output), 1 / SAMPLE_RATE)
spectrum_db = 20 * np.log10(spectrum / spectrum.max() + 1e-10)

# Find peaks
peaks = []
for i in range(1, len(spectrum) - 1):
    if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
        if spectrum_db[i] > -80 and freqs[i] > 20:
            peaks.append((freqs[i], spectrum_db[i]))

peaks.sort(key=lambda x: -x[1])

print(f"Found {len(peaks)} peaks above -80 dB")
print("\nTop 20 peaks:")
for freq, db in peaks[:20]:
    # Check if harmonic
    is_harm = any(abs(freq - f0 * h) < 5 for h in range(1, 50))
    note = "HARMONIC" if is_harm else "SPUR"
    print(f"  {freq:8.1f} Hz: {db:+6.1f} dB  {note}")

# Check specifically for 750 Hz sidebands
print()
print("=== Checking for 750 Hz sidebands ===")
block_rate = SAMPLE_RATE / BLOCK_SIZE
for n in range(-5, 6):
    if n == 0:
        continue
    expected_spur = f0 + n * block_rate
    if expected_spur > 0 and expected_spur < SAMPLE_RATE / 2:
        # Find nearest peak
        idx = np.argmin(np.abs(freqs - expected_spur))
        db = spectrum_db[idx]
        print(f"  {f0:.0f} + {n}×{block_rate:.0f} = {expected_spur:.0f} Hz: {db:+.1f} dB")

if len([p for p in peaks if p[1] > -60 and not any(abs(p[0] - f0*h) < 5 for h in range(1,50))]) == 0:
    print()
    print("RESULT: Clean sine - no significant spurs from FWPT/IDWPT stitching")
else:
    print()
    print("RESULT: Spurs present - problem in FWPT/IDWPT or stitching")
