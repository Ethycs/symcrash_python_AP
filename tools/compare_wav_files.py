"""Compare WAV files for bit-exactness."""
import numpy as np
from scipy.io import wavfile
import sys

files = [
    'audio_out/ccs_sine_440hz-20260105-131857.wav',  # Before int64 fix
    'audio_out/ccs_sine_440hz-20260105-144208.wav',  # After int64 fix (latest)
]

data = []
for f in files:
    rate, d = wavfile.read(f)
    data.append(d)
    print(f"{f}: shape={d.shape}, dtype={d.dtype}")

# Check if identical
if np.array_equal(data[0], data[1]):
    print("RESULT: Files are BIT-EXACT IDENTICAL")
else:
    diff = data[1].astype(np.int32) - data[0].astype(np.int32)
    print(f"RESULT: Files DIFFER")
    print(f"  Max diff: {np.abs(diff).max()}")
    print(f"  Non-zero samples: {np.count_nonzero(diff)} / {diff.size}")
