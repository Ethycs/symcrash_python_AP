"""Analyze remaining spikes in the safety-processed file."""
import numpy as np
import wave

filepath = '../audio_out/safety_debug/1_with_safety.wav'
with wave.open(filepath, 'rb') as wf:
    n = wf.getnframes()
    data = np.frombuffer(wf.readframes(n), dtype=np.int16)

# Just left channel
left = data[::2]

# Analyze
deltas = np.diff(left.astype(np.float64))
d2 = np.diff(deltas)

# Find spikes
spikes = np.where(np.abs(d2) > 500)[0]
print(f'Spike count: {len(spikes)}')
print(f'Spike positions: {spikes.tolist()}')
print(f'Spike d2 values: {d2[spikes].tolist()}')

# Check if at voice start (first 8 samples)
print(f'\nFirst 20 samples: {left[:20].tolist()}')
print(f'First 10 deltas: {deltas[:10].tolist()}')
print(f'First 10 d2: {d2[:10].tolist()}')

# Check block positions
print('\nBlock analysis (block_size=64):')
block_size = 64
for s in spikes:
    block = s // block_size
    pos = s % block_size
    print(f'  Spike at {s}: block {block}, position {pos}')
