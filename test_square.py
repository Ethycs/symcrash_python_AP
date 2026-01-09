"""FFT analysis - isolate Mode B spur source."""
import numpy as np
import scipy.io.wavfile as wav
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from symcrash_python_ap.vm.harmonic_bank import (
    mode_b_generate_coeffs, get_basis_for_note, ModeBState, HarmonicWeights
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit

N = 8192

# =============================================================================
# TEST 0: Isolate Mode B coefficient generation
# =============================================================================
print("=" * 60)
print("TEST 0: Isolate Mode B coefficient generation")
print("=" * 60)

# Create Mode B state for sine wave
mb_state = ModeBState()
mb_state.reset(note=60, phase_inc=23406951)  # C4 phase_inc
mb_state.weights.set_sine()
mb_state.H_active = 1

# Get basis frames
basis = get_basis_for_note(60)

# Generate enough blocks for good FFT resolution
all_samples = []
for block in range(200):  # 200 blocks = 200*64/48000 = 0.267 sec
    # Generate coefficients
    coeffs = np.zeros(512, dtype=np.int16)
    mode_b_generate_coeffs(mb_state, basis, coeffs)

    # Reconstruct audio via IDWPT
    # Extract k=0 coefficients only (as lane_ccs_render does)
    # coeff_state layout: [leaf0_k0..k7, leaf1_k0..k7, ...]
    # We need coeff_2d[:, 0] = [leaf0_k0, leaf1_k0, ..., leaf63_k0]
    coeff_2d = coeffs.reshape(64, 8)
    k0_coeffs = coeff_2d[:, 0].tolist()
    audio = idwpt_jit(k0_coeffs, headroom_bits=1)
    all_samples.extend(audio)

    # Advance phase
    mb_state.advance_block()

samples = np.array(all_samples, dtype=np.int16)

# FFT
fft = np.abs(np.fft.rfft(samples[500:500+N])) / (N/2)
freqs = np.fft.rfftfreq(N, 1/48000)

print("Direct Mode B -> IDWPT FFT peaks:")
threshold = np.max(fft) * 0.001
for i in range(1, len(fft)-1):
    if fft[i] > threshold and fft[i] > fft[i-1] and fft[i] > fft[i+1]:
        print(f"  {freqs[i]:8.1f} Hz: {fft[i]:8.1f}")

idx_h1 = np.argmin(np.abs(freqs - 261.63))
idx_h2 = np.argmin(np.abs(freqs - 523.3))
print(f"\nFundamental (261 Hz): {fft[idx_h1]:.1f}")
print(f"h=2 (523 Hz): {fft[idx_h2]:.1f} (should be ~0)")
print(f"h=2/h=1 ratio: {fft[idx_h2]/fft[idx_h1]*100:.3f}%")

# Check sub-harmonic spurious content
print("\nSub-harmonic check (spurious):")
for f in [99.6, 134.8, 175.8, 351.6]:
    idx = np.argmin(np.abs(freqs - f))
    print(f"  {f:.1f} Hz: {fft[idx]:.1f} ({fft[idx]/fft[idx_h1]*100:.3f}%)")

# =============================================================================
# TEST 1: Demo sine file (01_sine.wav)
# =============================================================================
print("\n" + "=" * 60)
print("TEST 1: Demo sine file (01_sine.wav)")
print("=" * 60)

fs, data = wav.read('audio_generation_tests/audio_out/waveforms/01_sine.wav')
if len(data.shape) > 1:
    samples_sine = data[:, 0]
else:
    samples_sine = data

sine_fft = np.abs(np.fft.rfft(samples_sine[1000:1000+N])) / (N/2)
freqs = np.fft.rfftfreq(N, 1/fs)

# Find peaks
threshold = np.max(sine_fft) * 0.001
print("Sine FFT peaks (>0.1% of max):")
for i in range(1, len(sine_fft)-1):
    if sine_fft[i] > threshold and sine_fft[i] > sine_fft[i-1] and sine_fft[i] > sine_fft[i+1]:
        print(f"  {freqs[i]:8.1f} Hz: {sine_fft[i]:8.1f}")

idx_h2 = np.argmin(np.abs(freqs - 523.3))
print(f"\nSine h=2 (523 Hz): {sine_fft[idx_h2]:.1f} (should be ~0)")

# =============================================================================
# TEST 2: Square wave file (02_square.wav)
# =============================================================================
print("\n" + "=" * 60)
print("TEST 2: Square wave Mode B (02_square.wav)")
print("=" * 60)

fs, data = wav.read('audio_generation_tests/audio_out/waveforms/02_square.wav')

# Take mono channel
if len(data.shape) > 1:
    samples = data[:, 0]
else:
    samples = data

# Use a clean section (skip first 1000 samples for settling)
samples = samples[1000:1000+8192]

# FFT
N = len(samples)
fft = np.fft.rfft(samples)
freqs = np.fft.rfftfreq(N, 1/fs)
magnitudes = np.abs(fft) / (N/2)  # Normalize

# Find peaks
threshold = np.max(magnitudes) * 0.01  # 1% of max
peaks = []
for i in range(1, len(magnitudes)-1):
    if magnitudes[i] > threshold and magnitudes[i] > magnitudes[i-1] and magnitudes[i] > magnitudes[i+1]:
        peaks.append((freqs[i], magnitudes[i]))

print(f"Sample rate: {fs} Hz")
print(f"Fundamental: C4 = 261.63 Hz")
print(f"\nFFT Analysis - Significant peaks (>1% of max):")
print("-" * 50)

# Expected harmonics for square wave: 261.63, 784.9, 1308.15, 1831.41, ...
# (odd multiples: 1, 3, 5, 7, 9, ...)
expected_harmonics = [261.63 * h for h in range(1, 64, 2)]

for freq, mag in sorted(peaks, key=lambda x: x[1], reverse=True)[:20]:
    # Check if this is expected
    is_expected = any(abs(freq - eh) < 10 for eh in expected_harmonics)
    marker = "" if is_expected else " *** SPURIOUS ***"
    print(f"  {freq:8.1f} Hz: {mag:8.1f} (dB: {20*np.log10(mag+1e-10):6.1f}){marker}")

# Also check for even harmonics (should NOT be present in square wave)
print("\nEven harmonic check (should be near zero for square wave):")
for h in [2, 4, 6, 8, 10]:
    expected_freq = 261.63 * h
    # Find closest bin
    idx = np.argmin(np.abs(freqs - expected_freq))
    mag = magnitudes[idx]
    print(f"  h={h} ({expected_freq:.1f} Hz): {mag:.1f} (dB: {20*np.log10(mag+1e-10):.1f})")
