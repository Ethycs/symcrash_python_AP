#!/usr/bin/env python3
"""Clear Numba cache and verify lifting fix."""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Find and remove Numba cache files
print("Clearing Numba cache...")
count = 0
for root, dirs, files in os.walk("src"):
    if "__pycache__" in dirs:
        cache_path = os.path.join(root, "__pycache__")
        for f in os.listdir(cache_path):
            if "nbc" in f or "nbi" in f:  # Numba cache files
                full_path = os.path.join(cache_path, f)
                print("  Removing:", full_path)
                os.remove(full_path)
                count += 1

print("Removed {} cache files".format(count))

# Force reimport
print("\nReimporting fast_lifting...")
from symcrash_python_ap.wavelet.fast_lifting import _effect_mode_jit_core, NUMBA_AVAILABLE
print("Numba available:", NUMBA_AVAILABLE)

# Check if the fix is in place
import inspect
try:
    source = inspect.getsource(_effect_mode_jit_core.py_func)

    # Check for the fix
    if "+2) >> 2" in source and "matches forward" in source:
        print("FIX VERIFIED: Inverse update has +2 rounding")
    else:
        print("WARNING: Inverse update fix may not be applied")

    if "+1) >> 1" in source and "matches forward" in source:
        print("FIX VERIFIED: Inverse predict has +1 rounding")
    else:
        print("WARNING: Inverse predict fix may not be applied")

except Exception as e:
    print("Could not inspect source:", e)

# Test the roundtrip
print("\nTesting FWPT->IDWPT roundtrip...")
import numpy as np
from symcrash_python_ap.wavelet.fast_lifting import effect_mode_jit

# Create a simple test signal (sine wave)
t = np.arange(64) / 48000.0
carrier = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16).tolist()

# Roundtrip with identity mask (all ones)
mask = [32767] * 64
wet = effect_mode_jit(carrier, mask, headroom_bits=1)

# Compare
carrier_np = np.array(carrier, dtype=np.float32)
wet_np = np.array(wet, dtype=np.float32)
error = wet_np - carrier_np
error_rms = np.sqrt(np.mean(error**2))
carrier_rms = np.sqrt(np.mean(carrier_np**2))

print("Carrier RMS: {:.1f}".format(carrier_rms))
print("Wet RMS: {:.1f}".format(np.sqrt(np.mean(wet_np**2))))
print("Error RMS: {:.1f} ({:.2f}% of carrier)".format(error_rms, 100*error_rms/carrier_rms))
print("Max error: {}".format(int(np.max(np.abs(error)))))

# Show sample comparison
print("\nFirst 10 samples (carrier vs wet vs error):")
for i in range(10):
    print("  [{}] {} -> {} (err: {})".format(i, carrier[i], wet[i], wet[i] - carrier[i]))
