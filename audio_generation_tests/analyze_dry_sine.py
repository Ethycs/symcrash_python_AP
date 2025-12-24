"""Analyze the dry sine output file."""
import numpy as np
import wave

filepath = '../audio_out/classic_waveforms/1_dry_sine.wav'
with wave.open(filepath, 'rb') as wf:
    n = wf.getnframes()
    data = np.frombuffer(wf.readframes(n), dtype=np.int16)

# Just left channel
left = data[::2]

# Analyze
deltas = np.diff(left.astype(np.float64))
d2 = np.diff(deltas)

print(f'DRY SINE (1_dry_sine.wav)')
print(f'  Samples: {len(left)}')
print(f'  Peak: {np.max(np.abs(left))}')
print(f'  Max delta: {np.max(np.abs(deltas)):.0f}')
print(f'  Max d2: {np.max(np.abs(d2)):.0f}')
print(f'  Spikes (d2 > 500): {np.sum(np.abs(d2) > 500)}')

# First 20 samples
print(f'  First 20 samples: {left[:20].tolist()}')
