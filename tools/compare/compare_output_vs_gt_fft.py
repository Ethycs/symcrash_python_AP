"""Compare CCS output FFT vs ground truth FFT."""

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

print("=== COMPARING CCS OUTPUT vs GROUND TRUTH FFT ===")
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

# FFT of both signals
window = windows.blackmanharris(valid_len)
gt_windowed = ground_truth[:valid_len].astype(np.float64) * window
ccs_windowed = ccs_output[:valid_len].astype(np.float64) * window

gt_spectrum = np.abs(rfft(gt_windowed))
ccs_spectrum = np.abs(rfft(ccs_windowed))
freqs = rfftfreq(valid_len, 1 / SAMPLE_RATE)

# Normalize to fundamental
gt_fund_idx = np.argmax(gt_spectrum)
ccs_fund_idx = np.argmax(ccs_spectrum)

gt_db = 20 * np.log10(gt_spectrum / gt_spectrum[gt_fund_idx] + 1e-10)
ccs_db = 20 * np.log10(ccs_spectrum / ccs_spectrum[ccs_fund_idx] + 1e-10)

print(f"Ground truth RMS: {np.sqrt(np.mean(ground_truth[:valid_len].astype(float)**2)):.1f}")
print(f"CCS output RMS: {np.sqrt(np.mean(ccs_output[:valid_len].astype(float)**2)):.1f}")
print(f"Fundamental frequency: GT={freqs[gt_fund_idx]:.1f} Hz, CCS={freqs[ccs_fund_idx]:.1f} Hz")
print()

# Compare at specific spur frequencies
print("=== COMPARISON AT SPUR FREQUENCIES ===")
block_rate = SAMPLE_RATE / BLOCK_SIZE
spur_freqs = [
    (f0 + block_rate, f"+750"),
    (f0 - block_rate, f"-750"),
    (f0 + 2*block_rate, f"+1500"),
    (f0 - 2*block_rate, f"-1500"),
    (6310, "6310"),
    (12310, "12310"),
    (18310, "18310"),
]

print(f"{'Freq':>10s}  {'GT (dB)':>10s}  {'CCS (dB)':>10s}  {'Diff':>8s}")
print("-" * 45)

for freq, label in spur_freqs:
    if 0 < freq < SAMPLE_RATE / 2:
        idx = np.argmin(np.abs(freqs - freq))
        print(f"{label:>10s}  {gt_db[idx]:>10.1f}  {ccs_db[idx]:>10.1f}  {ccs_db[idx] - gt_db[idx]:>+8.1f}")

print()
print("=== TOP DIFFERENCES (CCS vs GT) ===")
diff_db = ccs_db - gt_db
# Exclude fundamental region
fund_region = np.abs(freqs - f0) < 10

# Find biggest positive differences (CCS has more energy than GT)
diff_db_masked = diff_db.copy()
diff_db_masked[fund_region] = -np.inf

sorted_idx = np.argsort(-diff_db_masked)
print("Frequencies where CCS has more energy than GT:")
for i in sorted_idx[:20]:
    if diff_db[i] > 3:  # At least 3 dB more
        print(f"  {freqs[i]:8.1f} Hz: GT={gt_db[i]:+7.1f} dB, CCS={ccs_db[i]:+7.1f} dB, diff={diff_db[i]:+5.1f} dB")
