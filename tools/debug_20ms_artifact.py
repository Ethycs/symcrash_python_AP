"""Debug 20ms periodic artifact."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq
import wave
import struct

# Check what could cause 20ms periodic artifacts
fs = 48000
block_size = 64
blocks_per_sec = fs / block_size

print(f'Block rate: {blocks_per_sec} Hz ({1000/blocks_per_sec:.2f} ms)')
print(f'20ms = {20/1000 * blocks_per_sec:.1f} blocks')
print(f'50 Hz = 1/0.02s')
print()

# Check if it's related to vibrato ramp
vibrato_ramp_blocks = int(27.0 / 1000.0 * blocks_per_sec)
print(f'Vibrato ramp: {vibrato_ramp_blocks} blocks = {vibrato_ramp_blocks * block_size / fs * 1000:.1f} ms')

# Check vibrato delay
vibrato_delay_blocks = int(40.0 / 1000.0 * blocks_per_sec)
print(f'Vibrato delay: {vibrato_delay_blocks} blocks = {vibrato_delay_blocks * block_size / fs * 1000:.1f} ms')

# Load and analyze the wav file for 50 Hz component
wav_path = Path(__file__).parent.parent / "audio_generation_tests/audio_out/presets/violin_ir_single_note.wav"
with wave.open(str(wav_path), 'r') as wf:
    n_frames = wf.getnframes()
    data = wf.readframes(n_frames)
    samples = np.array(struct.unpack(f'<{n_frames}h', data), dtype=np.float64)

# Analyze spectrum around 50 Hz
print(f'\nAnalyzing {len(samples)} samples...')

# Take a 1-second chunk from sustain portion
start = int(1.5 * fs)  # After attack
end = start + fs
chunk = samples[start:end]

spectrum = np.abs(rfft(chunk))
freqs = rfftfreq(len(chunk), 1/fs)

# Find peaks near 50 Hz
low_freq_mask = (freqs > 10) & (freqs < 100)
low_freqs = freqs[low_freq_mask]
low_spectrum = spectrum[low_freq_mask]

print(f'\nLow frequency peaks (10-100 Hz):')
# Find local maxima
for i in range(1, len(low_spectrum)-1):
    if low_spectrum[i] > low_spectrum[i-1] and low_spectrum[i] > low_spectrum[i+1]:
        if low_spectrum[i] > np.mean(low_spectrum) * 2:  # Only significant peaks
            print(f'  {low_freqs[i]:.1f} Hz: amplitude = {low_spectrum[i]:.0f}')

# Check autocorrelation for 20ms period
print(f'\nAutocorrelation analysis:')
lag_20ms = int(0.02 * fs)  # 960 samples
lag_10ms = int(0.01 * fs)  # 480 samples

# Compute autocorrelation at specific lags
for lag_ms in [10, 15, 20, 25, 30]:
    lag = int(lag_ms / 1000 * fs)
    if lag < len(chunk) - 1000:
        corr = np.corrcoef(chunk[:-lag], chunk[lag:])[0, 1]
        print(f'  Lag {lag_ms}ms: correlation = {corr:.4f}')
