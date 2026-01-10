"""Analyze the old WAV file to understand what's different."""
import numpy as np
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq

SAMPLE_RATE = 48000
BLOCK_SIZE = 64
f0 = 440.0

# Load the oldest file (likely before fixes)
old_file = 'audio_out/ccs_sine_440hz-20260105-125220.wav'
new_file = 'audio_out/ccs_sine_440hz-20260105-144208.wav'

print("=== COMPARING OLD vs NEW WAV FILES ===")
print()

for label, fname in [("OLD (12:52)", old_file), ("NEW (14:42)", new_file)]:
    rate, data = wavfile.read(fname)
    # Take mono (left channel)
    if len(data.shape) > 1:
        mono = data[:, 0]
    else:
        mono = data

    n_samples = len(mono)
    n_blocks = n_samples // BLOCK_SIZE

    print(f"{label}: {fname}")
    print(f"  Samples: {n_samples}, Blocks: {n_blocks}")

    # Boundary discontinuity analysis
    delta_0 = []
    delta_1 = []
    for b in range(1, n_blocks):
        boundary = b * BLOCK_SIZE
        d0 = int(mono[boundary]) - int(mono[boundary - 1])
        delta_0.append(d0)
        if boundary >= 2:
            slope_before = int(mono[boundary - 1]) - int(mono[boundary - 2])
            slope_after = int(mono[boundary]) - int(mono[boundary - 1])
            d1 = slope_after - slope_before
            delta_1.append(d1)

    delta_0 = np.array(delta_0)
    delta_1 = np.array(delta_1)

    print(f"  Value jump RMS at boundaries: {np.sqrt(np.mean(delta_0**2)):.1f}")
    print(f"  Slope jump RMS at boundaries: {np.sqrt(np.mean(delta_1**2)):.1f}")

    # FFT for spur analysis
    fft = rfft(mono.astype(np.float64))
    freqs = rfftfreq(n_samples, 1 / SAMPLE_RATE)

    fund_idx = np.argmin(np.abs(freqs - f0))
    A_carrier = np.abs(fft[fund_idx])

    print(f"  Carrier amplitude: {A_carrier:.0f}")

    # Check key spur frequencies
    spur_freqs = [310, 1190, 6310, 12310]
    for freq in spur_freqs:
        idx = np.argmin(np.abs(freqs - freq))
        A_spur = np.abs(fft[idx])
        dBc = 20 * np.log10(A_spur / A_carrier + 1e-10)
        print(f"  Spur at {freq} Hz: {dBc:+.1f} dBc")

    print()
