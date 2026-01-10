"""Check pure sine for spikes."""
import numpy as np
import sys
sys.path.insert(0, 'src')
from symcrash_python_ap.wavelet.sine_core import sine_q15_vectorized

# Pure sine at 440Hz
phase_inc = int(440 * (2**32) / 48000)
num_samples = 545

phases_32 = np.arange(num_samples, dtype=np.int64) * phase_inc
phases_16 = ((phases_32 >> 16) & 0xFFFF).astype(np.int32)
signal = sine_q15_vectorized(phases_16)

# Check derivatives
d1 = np.diff(signal)
d2 = np.diff(d1)

print("Pure sine 440Hz:")
print(f"  Signal range: {signal.min()} to {signal.max()}")
print(f"  d1 range: {d1.min()} to {d1.max()}")
print(f"  d2 range: {d2.min()} to {d2.max()}")
print(f"  d2 std: {d2.std():.1f}")

# Find any outliers in d2
threshold = d2.std() * 3
outliers = np.where(np.abs(d2) > threshold)[0]
print(f"  Outliers (|d2| > {threshold:.0f}): {len(outliers)}")

if len(outliers) > 0:
    print(f"  Outlier positions: {outliers[:20]}")
    print(f"  Outlier d2 values: {d2[outliers[:20]]}")

    # Check if they cluster at specific cycle positions
    samples_per_cycle = int(48000 / 440)
    cycle_pos = (outliers % samples_per_cycle) / samples_per_cycle
    print(f"  Cycle positions: {cycle_pos[:10]}")
