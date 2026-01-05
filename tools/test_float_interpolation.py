"""Test if floating-point interpolation reduces the spurs."""

import numpy as np
import sys
sys.path.insert(0, 'src')

from scipy.fft import rfft, rfftfreq
from scipy.signal import windows

# Clear cache
import symcrash_python_ap.vm.coeff_lut_streaming as lut_mod
lut_mod._NOTE_LUT_CACHE.clear()

from symcrash_python_ap.vm.coeff_lut_streaming import (
    get_lut_for_note, BLOCK_SIZE, NUM_LEAVES, DEFAULT_K
)
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.wavelet.sine_core import sine_q15

SAMPLE_RATE = 48000
f0 = 440.0
K = 8

print("=== TESTING FLOATING POINT INTERPOLATION ===")
print()

lut = get_lut_for_note(wave_id=0, midi_note=69)
phase_inc = int((f0 / SAMPLE_RATE) * (1 << 32))

TARGET_RMS_Q15 = 16384
Q15_ONE = 32767

def generate_ground_truth(n_samples):
    output = np.empty(n_samples, dtype=np.int16)
    phase = 0
    for n in range(n_samples):
        sample = sine_q15(phase)
        sample = (sample * TARGET_RMS_Q15) // Q15_ONE
        output[n] = np.clip(sample, -32768, 32767)
        phase = (phase + phase_inc) & 0xFFFFFFFF
    return output

def get_subband_samples_float(lut, phase_32):
    """Floating-point interpolation for comparison."""
    P = lut.P
    shift = 24 if P == 256 else 26
    mask = P - 1

    phase_idx = (phase_32 >> shift) & mask
    phase_idx_next = (phase_idx + 1) & mask

    # Float fraction
    frac = (phase_32 & ((1 << shift) - 1)) / (1 << shift)

    v0 = lut.data[:, :, phase_idx].astype(np.float64)
    v1 = lut.data[:, :, phase_idx_next].astype(np.float64)

    val = v0 + (v1 - v0) * frac
    return np.round(val).astype(np.int16)

# Generate 1 second
duration = 1.0
n_samples = int(duration * SAMPLE_RATE)
n_blocks = n_samples // BLOCK_SIZE
valid_len = n_blocks * BLOCK_SIZE

ground_truth = generate_ground_truth(n_samples)
ccs_int64 = np.zeros(n_samples, dtype=np.int16)  # Current int64 interp
ccs_float = np.zeros(n_samples, dtype=np.int16)  # Float interp

# Generate both
phase_32 = 0
for block_idx in range(n_blocks):
    # Int64 interpolation (current)
    subband_int64 = np.zeros((NUM_LEAVES, K), dtype=np.int16)
    lut.get_subband_samples(phase_32, subband_int64)

    # Float interpolation
    subband_float = get_subband_samples_float(lut, phase_32)

    # Run IDWPTs for both
    block_start = block_idx * BLOCK_SIZE

    for k in range(K):
        coeffs_int64 = [subband_int64[l, k] for l in range(NUM_LEAVES)]
        coeffs_float = [subband_float[l, k] for l in range(NUM_LEAVES)]

        audio_int64 = idwpt_jit(coeffs_int64, headroom_bits=1)
        audio_float = idwpt_jit(coeffs_float, headroom_bits=1)

        for m in range(BLOCK_SIZE // K):
            ccs_int64[block_start + k + m * K] = audio_int64[m * K]
            ccs_float[block_start + k + m * K] = audio_float[m * K]

    phase_32 = (phase_32 + phase_inc * BLOCK_SIZE) & 0xFFFFFFFF

# Compare
error_int64 = ccs_int64[:valid_len].astype(np.int32) - ground_truth[:valid_len].astype(np.int32)
error_float = ccs_float[:valid_len].astype(np.int32) - ground_truth[:valid_len].astype(np.int32)

print(f"Int64 interp error: RMS={np.sqrt(np.mean(error_int64**2)):.2f}, max={np.abs(error_int64).max()}")
print(f"Float interp error: RMS={np.sqrt(np.mean(error_float**2)):.2f}, max={np.abs(error_float).max()}")
print()

# Check if they're different
diff = error_int64 - error_float
print(f"Difference between int64 and float: RMS={np.sqrt(np.mean(diff**2)):.3f}, max={np.abs(diff).max()}")
print()

# FFT of both
window = windows.blackmanharris(valid_len)
int64_spectrum = np.abs(rfft(ccs_int64[:valid_len].astype(np.float64) * window))
float_spectrum = np.abs(rfft(ccs_float[:valid_len].astype(np.float64) * window))
freqs = rfftfreq(valid_len, 1 / SAMPLE_RATE)

fund_idx = np.argmax(int64_spectrum)
int64_db = 20 * np.log10(int64_spectrum / int64_spectrum[fund_idx] + 1e-10)
float_db = 20 * np.log10(float_spectrum / float_spectrum[fund_idx] + 1e-10)

print("=== SPUR COMPARISON ===")
spur_freqs = [1190, 6310, 12310, 18310]
for freq in spur_freqs:
    idx = np.argmin(np.abs(freqs - freq))
    print(f"{freq} Hz: int64={int64_db[idx]:+.1f} dB, float={float_db[idx]:+.1f} dB")

# The spurs should be identical since the problem isn't interpolation anymore
