"""Test Mode A with H5 harmonic."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq

from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sine_q15, PHASE_MASK
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit
from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_block
from symcrash_python_ap.vm.coeff_lut_streaming import (
    StreamingCoeffLUT, VM_K, BLOCK_SIZE, NUM_LEAVES
)
from symcrash_python_ap.fixed import Q15_ONE

SAMPLE_RATE = 48000

print("=" * 70)
print("MODE A H5 TEST")
print(f"VM_K={VM_K}")
print("=" * 70)

# Create a custom H5 waveform generator
def h5_only(phase):
    """Generate pure H5 harmonic."""
    h = 5
    h_phase = (phase * h) & 0xFFFFFFFF
    return sine_q15(h_phase)

# Generate Mode A LUT for H5 at 440 Hz
print("\n--- Generating Mode A LUT for H5 (2200 Hz carrier) ---")
freq_hz = 440.0
phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

lut = StreamingCoeffLUT.generate_for_frequency(
    wave_id=0,
    name="h5_sine",
    target_freq_hz=freq_hz,
    generate_sample_fn=h5_only,
    K=VM_K,
    P=64,
    normalize=False,
)

print(f"LUT shape: {lut.data.shape}")

# Generate audio using Mode A
print("\n--- Rendering with Mode A ---")
num_blocks = 100
all_samples = []
phase = 0

for _ in range(num_blocks):
    # Get coefficients from LUT
    output = np.zeros((NUM_LEAVES, VM_K), dtype=np.int16)
    lut.get_subband_samples(phase, output)

    # IDWPT polyphase reconstruction
    audio = idwpt_polyphase_block(output, headroom_bits=1)
    all_samples.extend(audio.tolist())

    phase = (phase + BLOCK_SIZE * phase_inc) & 0xFFFFFFFF

samples = np.array(all_samples, dtype=np.float64)

# Analyze 50 Hz artifact
n = len(samples)
hann = np.hanning(n)
windowed = samples * hann
spectrum = np.abs(rfft(windowed)) * (2.0 / np.mean(hann))
freqs = rfftfreq(n, 1/SAMPLE_RATE)

# H5 fundamental is 2200 Hz
idx_h5 = int(2200 * n / SAMPLE_RATE)
amp_h5 = spectrum[idx_h5] if idx_h5 < len(spectrum) else 1

# 50 Hz artifact
idx_50 = int(50 * n / SAMPLE_RATE)
amp_50 = spectrum[idx_50]

rel_50 = 20 * np.log10(amp_50 / amp_h5) if amp_h5 > 0 else -100

print(f"Mode A H5-only:")
print(f"  2200 Hz amplitude: {amp_h5:.0f}")
print(f"  50 Hz artifact: {rel_50:.1f} dB relative")

# Now test with violin-like profile via Mode A
print("\n--- Testing Mode A with multi-harmonic (violin-like) ---")

def violin_sample(phase):
    """Generate violin-like waveform."""
    harmonics = [Q15_ONE, 24576, 16384, 8192, 3673, 2457, 1229, 614]
    total = sum(harmonics)
    val = 0
    for h, amp in enumerate(harmonics, 1):
        h_phase = (phase * h) & 0xFFFFFFFF
        val += (sine_q15(h_phase) * amp) >> 15
    # Normalize
    val = (val * Q15_ONE) // total
    return int(np.clip(val, -32768, 32767))

lut_violin = StreamingCoeffLUT.generate_for_frequency(
    wave_id=0,
    name="violin",
    target_freq_hz=freq_hz,
    generate_sample_fn=violin_sample,
    K=VM_K,
    P=64,
    normalize=False,
)

all_samples_violin = []
phase = 0

for _ in range(num_blocks):
    output = np.zeros((NUM_LEAVES, VM_K), dtype=np.int16)
    lut_violin.get_subband_samples(phase, output)
    audio = idwpt_polyphase_block(output, headroom_bits=1)
    all_samples_violin.extend(audio.tolist())
    phase = (phase + BLOCK_SIZE * phase_inc) & 0xFFFFFFFF

samples_violin = np.array(all_samples_violin, dtype=np.float64)

n = len(samples_violin)
hann = np.hanning(n)
windowed = samples_violin * hann
spectrum = np.abs(rfft(windowed)) * (2.0 / np.mean(hann))

idx_fund = int(440 * n / SAMPLE_RATE)
amp_fund = spectrum[idx_fund] if idx_fund < len(spectrum) else 1

idx_50 = int(50 * n / SAMPLE_RATE)
amp_50 = spectrum[idx_50]

rel_50_violin = 20 * np.log10(amp_50 / amp_fund) if amp_fund > 0 else -100

print(f"Mode A violin-like:")
print(f"  440 Hz amplitude: {amp_fund:.0f}")
print(f"  50 Hz artifact: {rel_50_violin:.1f} dB relative")
