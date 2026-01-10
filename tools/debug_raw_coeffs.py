"""Debug raw coefficients for H5 - is the artifact baked in?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq

from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sin_cos_q15, PHASE_MASK
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit

BLOCK_SIZE = 64
SAMPLE_RATE = 48000

print("=" * 70)
print("RAW COEFFICIENT ANALYSIS FOR H5")
print("=" * 70)

freq_hz = 440.0
phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)
h = 5

# Generate 100 consecutive blocks of H5
print("\n--- Generating 100 consecutive blocks ---")
all_coeffs = []  # Store coefficients for each block
all_audio = []   # Store reconstructed audio

phase = 0
for block_idx in range(100):
    # Generate H5 signal for this block
    signal = []
    for n in range(BLOCK_SIZE):
        h_phase = (phase * h) & PHASE_MASK
        sin_val, _ = sin_cos_q15(h_phase)
        signal.append(sin_val)
        phase = (phase + phase_inc) & PHASE_MASK

    # FWPT to get coefficients
    coeffs = np.array(fwpt_jit(signal, headroom_bits=1), dtype=np.int16)
    all_coeffs.append(coeffs.copy())

    # IDWPT to get audio
    audio = np.array(idwpt_jit(coeffs.tolist(), headroom_bits=1), dtype=np.int16)
    all_audio.extend(audio.tolist())

all_coeffs = np.array(all_coeffs)  # [100 blocks, 64 coeffs]
all_audio = np.array(all_audio, dtype=np.float64)

print(f"Coefficients shape: {all_coeffs.shape}")
print(f"Audio samples: {len(all_audio)}")

# Analyze the COEFFICIENTS for 50 Hz pattern
print("\n--- Coefficient Time Series Analysis ---")
# Look at leaf 0's coefficient evolution over time
leaf0_coeffs = all_coeffs[:, 0].astype(np.float64)

n_c = len(leaf0_coeffs)
hann_c = np.hanning(n_c)
spectrum_c = np.abs(rfft(leaf0_coeffs * hann_c)) * (2.0 / np.mean(hann_c))
freqs_c = rfftfreq(n_c, BLOCK_SIZE / SAMPLE_RATE)  # Time between blocks

print(f"\nLeaf 0 coefficient evolution:")
print(f"  Sample rate: {SAMPLE_RATE / BLOCK_SIZE:.1f} Hz (block rate)")
print(f"  Frequency resolution: {freqs_c[1]:.2f} Hz")

# Find peaks in coefficient spectrum
idx_50_c = int(50 * n_c * BLOCK_SIZE / SAMPLE_RATE)
dc_amp = spectrum_c[0]
amp_50_c = spectrum_c[idx_50_c] if idx_50_c < len(spectrum_c) else 0

print(f"  DC component: {dc_amp:.0f}")
print(f"  50 Hz component: {amp_50_c:.0f}")
if dc_amp > 0:
    print(f"  50 Hz relative to DC: {20*np.log10(amp_50_c/dc_amp):.1f} dB")

# Show top 5 peaks in coefficient spectrum
sorted_idx = np.argsort(spectrum_c[1:])[::-1] + 1  # Skip DC
print(f"\nTop 5 peaks in leaf 0 coefficient spectrum:")
for i in range(5):
    idx = sorted_idx[i]
    freq = freqs_c[idx]
    amp = spectrum_c[idx]
    print(f"  {freq:.1f} Hz: {amp:.0f}")

# Analyze the AUDIO for 50 Hz pattern
print("\n--- Audio Time Series Analysis ---")
n_a = len(all_audio)
hann_a = np.hanning(n_a)
spectrum_a = np.abs(rfft(all_audio * hann_a)) * (2.0 / np.mean(hann_a))
freqs_a = rfftfreq(n_a, 1/SAMPLE_RATE)

idx_h5 = int(2200 * n_a / SAMPLE_RATE)
idx_50_a = int(50 * n_a / SAMPLE_RATE)
amp_h5 = spectrum_a[idx_h5]
amp_50_a = spectrum_a[idx_50_a]

print(f"Audio analysis:")
print(f"  H5 (2200 Hz): {amp_h5:.0f}")
print(f"  50 Hz: {amp_50_a:.0f}")
print(f"  50 Hz relative to H5: {20*np.log10(amp_50_a/amp_h5):.1f} dB")

# Key question: Is the 50 Hz in the coefficients or introduced by IDWPT?
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
If 50 Hz is strong in coefficient spectrum: artifact is from FWPT
If 50 Hz is only in audio spectrum: artifact is from IDWPT boundary
""")
