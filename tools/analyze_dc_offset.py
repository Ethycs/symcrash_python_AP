#!/usr/bin/env python3
"""Analyze DC offset in IDWPT transform."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from symcrash_python_ap.wavelet.fast_lifting import effect_mode_jit

# Generate pure 440Hz sine
sr = 48000
samples = np.arange(64)
carrier = (np.sin(2 * np.pi * 440 * samples / sr) * 28000).astype(np.int16)
carrier_list = carrier.tolist()

print('Carrier stats:')
print(f'  sum: {np.sum(carrier)}')
print(f'  mean: {np.mean(carrier):.2f}')

# Passthrough with headroom_bits=1
wet = effect_mode_jit(carrier_list, None, headroom_bits=1)
wet_arr = np.array(wet, dtype=np.int16)

print(f'\nWet (passthrough) stats:')
print(f'  sum: {np.sum(wet_arr)}')
print(f'  mean: {np.mean(wet_arr):.2f}')

diff = wet_arr.astype(np.int32) - carrier.astype(np.int32)
print(f'\nDiff (wet - carrier) stats:')
print(f'  sum: {np.sum(diff)}')
print(f'  mean: {np.mean(diff):.2f}')
print(f'  max: {np.max(np.abs(diff))}')

# The sum of diffs is exactly what appears at DC in FFT
print(f'\n  DC offset expected in FFT: {np.sum(diff):.0f}')

# Now analyze mixing
# At 50/50 mix: mixed = 0.5*carrier + 0.5*wet
# = 0.5*carrier + 0.5*(carrier + error) = carrier + 0.5*error
mixed = (carrier.astype(np.int32) + wet_arr.astype(np.int32)) // 2
mixed_i16 = mixed.astype(np.int16)

# Spectrum of pure carrier
fft_carrier = np.abs(np.fft.rfft(carrier.astype(np.float64)))
fft_wet = np.abs(np.fft.rfft(wet_arr.astype(np.float64)))
fft_mixed = np.abs(np.fft.rfft(mixed.astype(np.float64)))
fft_diff = np.abs(np.fft.rfft(diff.astype(np.float64)))
freqs = np.fft.rfftfreq(64, 1/48000)

print('\nSpectrum comparison at 440Hz and harmonics:')
print('Freq (Hz) | Carrier | Wet | Mixed | Diff')
for f in [0, 440, 880, 1320, 1760, 2200]:
    idx = np.argmin(np.abs(freqs - f))
    actual_f = freqs[idx]
    print(f'{actual_f:8.1f} | {fft_carrier[idx]:7.1f} | {fft_wet[idx]:6.1f} | {fft_mixed[idx]:5.1f} | {fft_diff[idx]:5.1f}')
