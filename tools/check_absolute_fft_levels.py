"""Check absolute FFT levels to understand spur magnitudes."""

import numpy as np
import sys
sys.path.insert(0, 'src')

from scipy.fft import rfft, rfftfreq
from scipy.signal import windows

# Clear cache
import symcrash_python_ap.vm.coeff_lut_streaming as lut_mod
lut_mod._NOTE_LUT_CACHE.clear()

from symcrash_python_ap.vm.coeff_lut_streaming import (
    get_lut_for_note, BLOCK_SIZE, NUM_LEAVES, VM_K
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.wavelet.sine_core import sine_q15

SAMPLE_RATE = 48000
f0 = 440.0
K = 8

print("=== ABSOLUTE FFT LEVEL ANALYSIS ===")
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

# Generate 1 second
duration = 1.0
n_samples = int(duration * SAMPLE_RATE)
n_blocks = n_samples // BLOCK_SIZE
valid_len = n_blocks * BLOCK_SIZE

ground_truth = generate_ground_truth(n_samples)
ccs_output = np.zeros(n_samples, dtype=np.int16)

# Generate CCS output
phase_32 = 0
for block_idx in range(n_blocks):
    subband_output = np.zeros((NUM_LEAVES, K), dtype=np.int16)
    lut.get_subband_samples(phase_32, subband_output)

    idwpt_outputs = []
    for k in range(K):
        coeffs_k = [subband_output[l, k] for l in range(NUM_LEAVES)]
        audio_k = idwpt_jit(coeffs_k, headroom_bits=1)
        idwpt_outputs.append(audio_k)

    block_start = block_idx * BLOCK_SIZE
    for k in range(K):
        for m in range(BLOCK_SIZE // K):
            ccs_output[block_start + k + m * K] = idwpt_outputs[k][m * K]

    phase_32 = (phase_32 + phase_inc * BLOCK_SIZE) & 0xFFFFFFFF

# Also check what the fft_analysis_ccs.py script produces
from symcrash_python_ap.vm.lanes import lane_ccs_render

# FFT without window (raw DFT)
print("=== RAW FFT (no window) ===")
gt_fft = rfft(ground_truth[:valid_len].astype(np.float64))
ccs_fft = rfft(ccs_output[:valid_len].astype(np.float64))
freqs = rfftfreq(valid_len, 1 / SAMPLE_RATE)

# Find fundamental
fund_idx = np.argmax(np.abs(gt_fft))
print(f"Fundamental index: {fund_idx}, freq: {freqs[fund_idx]:.1f} Hz")
print(f"GT magnitude at fundamental: {np.abs(gt_fft[fund_idx]):.0f}")
print(f"CCS magnitude at fundamental: {np.abs(ccs_fft[fund_idx]):.0f}")
print()

# Check at spur frequencies
print("=== ABSOLUTE MAGNITUDES AT SPUR FREQUENCIES ===")
block_rate = SAMPLE_RATE / BLOCK_SIZE
spur_freqs = [1190, 6310, 12310, 18310]

for freq in spur_freqs:
    idx = np.argmin(np.abs(freqs - freq))
    gt_mag = np.abs(gt_fft[idx])
    ccs_mag = np.abs(ccs_fft[idx])
    gt_db = 20 * np.log10(gt_mag / np.abs(gt_fft[fund_idx]) + 1e-10)
    ccs_db = 20 * np.log10(ccs_mag / np.abs(ccs_fft[fund_idx]) + 1e-10)
    print(f"{freq} Hz: GT={gt_mag:.0f} ({gt_db:+.1f} dB), CCS={ccs_mag:.0f} ({ccs_db:+.1f} dB)")

print()
print("=== ERROR SIGNAL ANALYSIS ===")
error = ccs_output[:valid_len].astype(np.int32) - ground_truth[:valid_len].astype(np.int32)
error_fft = rfft(error.astype(np.float64))

print(f"Error signal RMS: {np.sqrt(np.mean(error**2)):.2f} LSB")
print(f"Error signal max: {np.abs(error).max()} LSB")
print()

# The error signal should have the same spectrum as the spurs
print("Error FFT at spur frequencies:")
for freq in spur_freqs:
    idx = np.argmin(np.abs(freqs - freq))
    err_mag = np.abs(error_fft[idx])
    err_rel_db = 20 * np.log10(err_mag / np.abs(gt_fft[fund_idx]) + 1e-10)
    print(f"  {freq} Hz: mag={err_mag:.1f}, rel to GT fund={err_rel_db:+.1f} dB")

# Key question: does the error spectrum match the CCS spur spectrum?
print()
print("=== DOES ERROR SPECTRUM MATCH CCS SPURS? ===")
for freq in spur_freqs:
    idx = np.argmin(np.abs(freqs - freq))
    err_mag = np.abs(error_fft[idx])
    ccs_minus_gt_mag = np.abs(ccs_fft[idx]) - np.abs(gt_fft[idx])
    print(f"{freq} Hz: error_mag={err_mag:.1f}, ccs-gt_mag={ccs_minus_gt_mag:.1f}")
