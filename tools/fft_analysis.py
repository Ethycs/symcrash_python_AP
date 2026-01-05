"""FFT analysis of generated audio."""

import numpy as np
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else 'audio_out/ccs_sine_440hz-20260105-150929.wav'
rate, data = wavfile.read(filepath)
if len(data.shape) > 1:
    data = data[:, 0]  # Use left channel

# Skip first 0.1 seconds (latency period + gain ramp)
start = int(0.1 * rate)
data = data[start:]

print(f'Analyzing {len(data)} samples after skipping latency period')
print()

# FFT
fft = rfft(data.astype(np.float64))
freqs = rfftfreq(len(data), 1/rate)

# Find carrier
f0 = 440.0
fund_idx = np.argmin(np.abs(freqs - f0))
A_carrier = np.abs(fft[fund_idx])

print(f'Carrier (440 Hz): amplitude = {A_carrier:.0f}')
print()

# Check spur levels relative to carrier (dBc)
spur_freqs = [310, 750, 1190, 1500, 6310, 7190, 12310, 18310]
print('Spur levels (dBc = 20*log10(A_spur/A_carrier)):')
for freq in spur_freqs:
    idx = np.argmin(np.abs(freqs - freq))
    A_spur = np.abs(fft[idx])
    dBc = 20 * np.log10(A_spur / A_carrier + 1e-10)
    print(f'  {freq:5d} Hz: A={A_spur:8.0f}, dBc={dBc:+.1f}')

print()

# Also check noise floor (average of bins away from harmonics)
noise_bins = []
for i, f in enumerate(freqs):
    if f > 100 and f < 20000:
        if abs(f - 440) > 50 and abs(f - 880) > 50 and abs(f - 1320) > 50:
            if all(abs(f - sf) > 50 for sf in spur_freqs):
                noise_bins.append(np.abs(fft[i]))
noise_floor = np.median(noise_bins)
dBc_floor = 20 * np.log10(noise_floor / A_carrier + 1e-10)
print(f'Noise floor (median): dBc = {dBc_floor:.1f}')
