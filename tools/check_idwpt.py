#!/usr/bin/env python3
"""Check IDWPT passthrough quality."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from symcrash_python_ap.wavelet.fast_lifting import effect_mode_jit

# Create a pure 440Hz sine
t = np.arange(64) / 48000
sine = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
sine_list = sine.tolist()

# Unity mask (all 32767)
unity_mask = [32767] * 64

# Process through IDWPT
output = effect_mode_jit(sine_list, unity_mask)
output = np.array(output)

# Compare input and output
diff = output - sine
print(f"Input range: [{sine.min()}, {sine.max()}]")
print(f"Output range: [{output.min()}, {output.max()}]")
print(f"Difference range: [{diff.min()}, {diff.max()}]")
print(f"RMS difference: {np.sqrt(np.mean(diff**2)):.2f}")
print(f"Max abs difference: {np.abs(diff).max()}")

# Check frequency content of difference
diff_fft = np.abs(np.fft.rfft(diff))
freqs = np.fft.rfftfreq(64, 1/48000)
print(f"\nDifference spectrum peaks:")
for i in np.argsort(diff_fft)[-5:][::-1]:
    if diff_fft[i] > 1:
        print(f"  {freqs[i]:.0f} Hz: {diff_fft[i]:.1f}")

# Also test with no mask (should be same as input)
output_no_mask = effect_mode_jit(sine_list, None)
output_no_mask = np.array(output_no_mask)
diff_no_mask = output_no_mask - sine
print(f"\nNo mask - max abs difference: {np.abs(diff_no_mask).max()}")
