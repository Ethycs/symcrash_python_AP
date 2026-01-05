import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.vm.coeff_tables_fast import generate_sinusoid_block_fast
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
import numpy as np

# Generate 261.63 Hz sine (MIDI 60)
freq_hz = 261.63
audio_ref = generate_sinusoid_block_fast(freq_hz, phase_start=0, num_samples=64)

print(f'Reference audio (first 10 samples): {audio_ref[:10]}')
print(f'Reference audio (last 10 samples): {audio_ref[-10:]}')
print(f'Reference audio min/max: {audio_ref.min()}/{audio_ref.max()}')

# Run FWPT
coeffs = fwpt_jit(audio_ref.tolist(), headroom_bits=1)
coeffs_arr = np.array(coeffs, dtype=np.int16)

print(f'\nCoefficients (first 10): {coeffs_arr[:10]}')
print(f'Coefficients (last 10): {coeffs_arr[-10:]}')
print(f'Non-zero coeffs: {np.count_nonzero(coeffs_arr)}/64')

# Reconstruct
audio_recon = idwpt_jit(coeffs, headroom_bits=1)
audio_recon_arr = np.array(audio_recon, dtype=np.int16)

print(f'\nReconstructed audio (first 10): {audio_recon_arr[:10]}')
print(f'Reconstruction error: {np.mean(np.abs(audio_ref - audio_recon_arr)):.2f} samples')

# FFT
spectrum = np.fft.rfft(audio_recon_arr.astype(np.float32))
freqs = np.fft.rfftfreq(64, 1.0/48000)
peak_idx = np.argmax(np.abs(spectrum))
peak_freq = freqs[peak_idx]

print(f'\nFFT peak frequency: {peak_freq:.2f} Hz (expected {freq_hz:.2f} Hz)')
