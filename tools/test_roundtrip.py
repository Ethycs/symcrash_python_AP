#!/usr/bin/env python3
"""Test IDWPT round-trip error."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from symcrash_python_ap.wavelet.fast_lifting import effect_mode_jit

# A pure 440Hz sine
samples = 64
carrier = [int(32767 * np.sin(2 * np.pi * 440 * i / 48000)) for i in range(samples)]

# Passthrough: mask_q15=None
wet = effect_mode_jit(carrier, None)

# Calculate error
error = np.array(wet) - np.array(carrier)
print(f"Passthrough error: max={np.abs(error).max()}, rms={np.sqrt(np.mean(error**2)):.2f}")

# Unity mask: mask_q15=[32767]*64
wet_unity = effect_mode_jit(carrier, [32767]*64)
error_unity = np.array(wet_unity) - np.array(carrier)
print(f"Unity mask error: max={np.abs(error_unity).max()}, rms={np.sqrt(np.mean(error_unity**2)):.2f}")

# Compare wet paths
print(f"Wet passthrough vs unity diff: max={np.abs(np.array(wet) - np.array(wet_unity)).max()}")

# Let's see the actual differences
print("\nFirst 10 samples comparison:")
print("  i | carrier |   wet   | unity   | err_pass | err_unity")
print("-" * 60)
for i in range(10):
    print(f" {i:2d} | {carrier[i]:7d} | {wet[i]:7d} | {wet_unity[i]:7d} | {wet[i]-carrier[i]:8d} | {wet_unity[i]-carrier[i]:9d}")
