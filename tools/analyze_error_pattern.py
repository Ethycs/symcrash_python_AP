"""Analyze the pattern of per-block errors."""

import numpy as np
import sys
sys.path.insert(0, 'src')

from scipy.fft import rfft, rfftfreq

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

print("=== ANALYZING ERROR PATTERN ===")
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

ground_truth = generate_ground_truth(n_samples)
ccs_output = np.zeros(n_samples, dtype=np.int16)

# Generate CCS output and collect error per block
phase_32 = 0
block_errors = []
block_max_errors = []

for block_idx in range(n_blocks):
    # Get CCS coefficients
    subband_output = np.zeros((NUM_LEAVES, K), dtype=np.int16)
    lut.get_subband_samples(phase_32, subband_output)

    # Run K IDWPTs and interleave
    idwpt_outputs = []
    for k in range(K):
        coeffs_k = [subband_output[l, k] for l in range(NUM_LEAVES)]
        audio_k = idwpt_jit(coeffs_k, headroom_bits=1)
        idwpt_outputs.append(audio_k)

    # Interleave
    block_start = block_idx * BLOCK_SIZE
    for k in range(K):
        for m in range(BLOCK_SIZE // K):
            ccs_output[block_start + k + m * K] = idwpt_outputs[k][m * K]

    # Compute error
    gt_block = ground_truth[block_start : block_start + BLOCK_SIZE]
    ccs_block = ccs_output[block_start : block_start + BLOCK_SIZE]
    error = ccs_block.astype(np.int32) - gt_block.astype(np.int32)
    block_errors.append(error.mean())
    block_max_errors.append(np.abs(error).max())

    # Advance phase
    phase_32 = (phase_32 + phase_inc * BLOCK_SIZE) & 0xFFFFFFFF

block_errors = np.array(block_errors)
block_max_errors = np.array(block_max_errors)

print(f"Analyzed {n_blocks} blocks")
print(f"Mean error per block: min={block_errors.min():.2f}, max={block_errors.max():.2f}, mean={block_errors.mean():.2f}")
print(f"Max error per block: min={block_max_errors.min()}, max={block_max_errors.max()}, mean={block_max_errors.mean():.1f}")
print()

# FFT of error signal
error_signal = ccs_output[:n_blocks * BLOCK_SIZE].astype(np.int32) - ground_truth[:n_blocks * BLOCK_SIZE].astype(np.int32)
print(f"Error signal: min={error_signal.min()}, max={error_signal.max()}, rms={np.sqrt(np.mean(error_signal**2)):.1f}")
print()

# FFT of error signal
print("=== FFT OF ERROR SIGNAL ===")
spectrum = np.abs(rfft(error_signal.astype(np.float64)))
freqs = rfftfreq(len(error_signal), 1 / SAMPLE_RATE)
spectrum_db = 20 * np.log10(spectrum / spectrum.max() + 1e-10)

# Find peaks in error spectrum
peaks = []
for i in range(1, len(spectrum) - 1):
    if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
        if spectrum_db[i] > -40:
            peaks.append((freqs[i], spectrum_db[i]))

peaks.sort(key=lambda x: -x[1])

print(f"Top 20 error spectrum peaks:")
block_rate = SAMPLE_RATE / BLOCK_SIZE
for freq, db in peaks[:20]:
    # Check if it's at n*750 ± 440
    for n in range(-100, 100):
        if abs(freq - (n * block_rate + f0)) < 5:
            print(f"  {freq:8.1f} Hz: {db:+6.1f} dB  ({n}×750 + 440)")
            break
        if abs(freq - (n * block_rate - f0)) < 5:
            print(f"  {freq:8.1f} Hz: {db:+6.1f} dB  ({n}×750 - 440)")
            break
    else:
        print(f"  {freq:8.1f} Hz: {db:+6.1f} dB")

# FFT of per-block mean error (at block rate)
print()
print("=== FFT OF PER-BLOCK MEAN ERROR ===")
if len(block_errors) > 1:
    spectrum_block = np.abs(rfft(block_errors))
    freqs_block = rfftfreq(len(block_errors), BLOCK_SIZE / SAMPLE_RATE)
    spectrum_block_db = 20 * np.log10(spectrum_block / spectrum_block.max() + 1e-10)

    print(f"Top peaks in block-rate error spectrum:")
    block_peaks = []
    for i in range(1, len(spectrum_block) - 1):
        if spectrum_block[i] > spectrum_block[i-1] and spectrum_block[i] > spectrum_block[i+1]:
            if spectrum_block_db[i] > -40:
                block_peaks.append((freqs_block[i], spectrum_block_db[i]))
    block_peaks.sort(key=lambda x: -x[1])
    for freq, db in block_peaks[:10]:
        print(f"  {freq:8.1f} Hz: {db:+6.1f} dB")
