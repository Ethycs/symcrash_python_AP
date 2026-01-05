"""Verify ground truth FFT is clean."""

import numpy as np
import sys
sys.path.insert(0, 'src')

from scipy.fft import rfft, rfftfreq
from scipy.signal import windows

from symcrash_python_ap.wavelet.sine_core import sine_q15

SAMPLE_RATE = 48000
f0 = 440.0

print("=== VERIFYING GROUND TRUTH FFT ===")
print()

phase_inc = int((f0 / SAMPLE_RATE) * (1 << 32))
TARGET_RMS_Q15 = 16384
Q15_ONE = 32767

# Generate 1 second
duration = 1.0
n_samples = int(duration * SAMPLE_RATE)

ground_truth = np.empty(n_samples, dtype=np.int16)
phase = 0
for n in range(n_samples):
    sample = sine_q15(phase)
    sample = (sample * TARGET_RMS_Q15) // Q15_ONE
    ground_truth[n] = np.clip(sample, -32768, 32767)
    phase = (phase + phase_inc) & 0xFFFFFFFF

print(f"Ground truth: {n_samples} samples")
print(f"RMS: {np.sqrt(np.mean(ground_truth.astype(float)**2)):.1f}")
print(f"Peak: {np.abs(ground_truth).max()}")
print()

# FFT
window = windows.blackmanharris(n_samples)
windowed = ground_truth.astype(np.float64) * window
spectrum = np.abs(rfft(windowed))
freqs = rfftfreq(n_samples, 1 / SAMPLE_RATE)

fund_idx = np.argmax(spectrum)
spectrum_db = 20 * np.log10(spectrum / spectrum[fund_idx] + 1e-10)

print(f"Fundamental: {freqs[fund_idx]:.1f} Hz")
print()

# Check spur frequencies
print("=== GROUND TRUTH AT SPUR FREQUENCIES ===")
block_rate = SAMPLE_RATE / 64  # 750 Hz
test_freqs = [
    (310, "440-130 (n×750-440 for n=1)"),
    (1190, "440+750"),
    (6310, "n×750-440"),
    (12310, "n×750-440"),
    (18310, "n×750-440"),
    (7190, "n×750+440"),
]

for freq, label in test_freqs:
    idx = np.argmin(np.abs(freqs - freq))
    print(f"  {freq:>6d} Hz ({label}): {spectrum_db[idx]:+.1f} dB")

# Find all peaks
print()
print("=== TOP 20 PEAKS IN GROUND TRUTH ===")
peaks = []
for i in range(1, len(spectrum) - 1):
    if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
        if spectrum_db[i] > -150 and freqs[i] > 10:
            peaks.append((freqs[i], spectrum_db[i]))

peaks.sort(key=lambda x: -x[1])
for freq, db in peaks[:20]:
    is_harm = any(abs(freq - f0 * h) < 2 for h in range(1, 100))
    note = "HARMONIC" if is_harm else ""
    print(f"  {freq:8.1f} Hz: {db:+7.1f} dB  {note}")
