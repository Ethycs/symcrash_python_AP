"""Test polyphase IDWPT stitching with ground truth waveform."""

import numpy as np
import sys
sys.path.insert(0, 'src')

from scipy.fft import rfft, rfftfreq
from scipy.signal import windows

from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.wavelet.sine_core import sine_q15

SAMPLE_RATE = 48000
BLOCK_SIZE = 64
K = 8
f0 = 440.0

print("=== TEST: POLYPHASE STITCHING WITH GROUND TRUTH ===")
print()

# Generate ground truth sine wave
duration = 1.0
n_samples = int(duration * SAMPLE_RATE)
phase_inc = int((f0 / SAMPLE_RATE) * (1 << 32))

# Generate using exact same method as LUT generation
print("Generating ground truth sine...")
gt_sine = np.zeros(n_samples, dtype=np.int16)
phase = 0
for n in range(n_samples):
    gt_sine[n] = (sine_q15(phase) * 16384) // 32767  # Same normalization as LUT
    phase = (phase + phase_inc) & 0xFFFFFFFF

print(f"Ground truth: {n_samples} samples, RMS={np.sqrt(np.mean(gt_sine.astype(float)**2)):.1f}")

# Now process with polyphase IDWPT, using FWPT on ground truth
print()
print("Processing with polyphase FWPT/IDWPT...")
output = np.zeros(n_samples, dtype=np.int16)
n_blocks = n_samples // BLOCK_SIZE

for b in range(n_blocks):
    block_start = b * BLOCK_SIZE

    # For each k, take shifted block from ground truth, run FWPT, then IDWPT
    idwpt_outputs = []
    for k in range(K):
        # Extract shifted block (like LUT does)
        block = gt_sine[block_start + k : block_start + k + BLOCK_SIZE]
        if len(block) < BLOCK_SIZE:
            # Pad with zeros at end
            block = np.concatenate([block, np.zeros(BLOCK_SIZE - len(block), dtype=np.int16)])

        # FWPT -> IDWPT round trip
        coeffs = fwpt_jit(block.tolist(), headroom_bits=1)
        recon = idwpt_jit(coeffs, headroom_bits=1)
        idwpt_outputs.append(recon)

    # Polyphase interleave: out[k::K] = y_k[0::K]
    for k in range(K):
        for m in range(BLOCK_SIZE // K):
            output[block_start + k + m * K] = idwpt_outputs[k][m * K]

# Compare to ground truth
valid_len = n_blocks * BLOCK_SIZE
error = np.abs(output[:valid_len].astype(np.int32) - gt_sine[:valid_len].astype(np.int32))
print(f"Max error vs ground truth: {error.max()} LSB")
print(f"Mean error: {error.mean():.1f} LSB")

# FFT analysis
print()
print("=== FFT ANALYSIS ===")
window = windows.blackmanharris(valid_len)
windowed = output[:valid_len].astype(np.float64) * window
spectrum = np.abs(rfft(windowed))
freqs = rfftfreq(valid_len, 1 / SAMPLE_RATE)
spectrum_db = 20 * np.log10(spectrum / spectrum.max() + 1e-10)

# Find peaks
peaks = []
for i in range(1, len(spectrum) - 1):
    if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
        if spectrum_db[i] > -80 and freqs[i] > 20:
            peaks.append((freqs[i], spectrum_db[i]))

peaks.sort(key=lambda x: -x[1])

print(f"Found {len(peaks)} peaks above -80 dB")

# Check for 750 Hz sidebands
print()
print("750 Hz sideband check:")
block_rate = SAMPLE_RATE / BLOCK_SIZE
for n in range(-5, 6):
    if n == 0:
        continue
    expected = f0 + n * block_rate
    if 0 < expected < SAMPLE_RATE / 2:
        idx = np.argmin(np.abs(freqs - expected))
        print(f"  {f0:.0f} + {n}×750 = {expected:.0f} Hz: {spectrum_db[idx]:+.1f} dB")

# Show top peaks
print()
print("Top 10 peaks:")
for freq, db in peaks[:10]:
    is_harm = any(abs(freq - f0 * h) < 5 for h in range(1, 50))
    note = "HARMONIC" if is_harm else "SPUR"
    print(f"  {freq:8.1f} Hz: {db:+6.1f} dB  {note}")
