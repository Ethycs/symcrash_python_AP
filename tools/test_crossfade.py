#!/usr/bin/env python3
"""Test equal-power crossfade for noise issues."""

import numpy as np
import math

# Test the _sincos_batch_q15 function
from symcrash_python_ap.vm.lanes import _sincos_batch_q15, _init_qw_table

print("Equal-Power Crossfade Test")
print("=" * 60)

# Test boundary values
# eta=0 should give dry=1, wet=0 (cos(0)=1, sin(0)=0)
# eta=32767 should give dry=0, wet=1 (cos(90)=0, sin(90)=1)

eta_values = [0, 8192, 16384, 24576, 32767]
Q15_ONE = 32767

print("\nBoundary checks (eta -> cos, sin):")
print("-" * 60)

for eta in eta_values:
    # Map eta to phase (0 to 0x40000000)
    phase = (eta * 0x40000000) >> 15
    phases = np.array([phase], dtype=np.uint32)

    cos_val, sin_val = _sincos_batch_q15(phases)

    # Expected values
    theta = (eta / 32767.0) * (math.pi / 2)
    expected_cos = int(math.cos(theta) * 32767)
    expected_sin = int(math.sin(theta) * 32767)

    cos_err = abs(cos_val[0] - expected_cos)
    sin_err = abs(sin_val[0] - expected_sin)

    # Check power sum (should be ~32767^2 = ~1.07e9 for equal power)
    power = cos_val[0]**2 + sin_val[0]**2
    expected_power = 32767**2
    power_ratio = power / expected_power

    print(f"eta={eta:5d}: cos={cos_val[0]:6d} (exp={expected_cos:6d}, err={cos_err:3d}), "
          f"sin={sin_val[0]:6d} (exp={expected_sin:6d}, err={sin_err:3d}), "
          f"power={power_ratio:.3f}")

# Test full range
print("\n" + "=" * 60)
print("Full range test (64 eta values):")
print("-" * 60)

eta_range = np.linspace(0, 32767, 64).astype(np.int32)
phases = ((eta_range.astype(np.int64) * 0x40000000) >> 15).astype(np.uint32)
cos_vals, sin_vals = _sincos_batch_q15(phases)

# Check for anomalies
max_cos_err = 0
max_sin_err = 0
min_power_ratio = 1.0
max_power_ratio = 1.0

for i, eta in enumerate(eta_range):
    theta = (eta / 32767.0) * (math.pi / 2)
    expected_cos = int(math.cos(theta) * 32767)
    expected_sin = int(math.sin(theta) * 32767)

    cos_err = abs(cos_vals[i] - expected_cos)
    sin_err = abs(sin_vals[i] - expected_sin)
    max_cos_err = max(max_cos_err, cos_err)
    max_sin_err = max(max_sin_err, sin_err)

    power = cos_vals[i]**2 + sin_vals[i]**2
    power_ratio = power / (32767**2)
    min_power_ratio = min(min_power_ratio, power_ratio)
    max_power_ratio = max(max_power_ratio, power_ratio)

print(f"Max cos error: {max_cos_err} LSB")
print(f"Max sin error: {max_sin_err} LSB")
print(f"Power ratio range: {min_power_ratio:.3f} to {max_power_ratio:.3f}")

# Check for sign issues
negative_cos = np.sum(cos_vals < 0)
negative_sin = np.sum(sin_vals < 0)
print(f"Negative cos values: {negative_cos} (should be 0 for first quadrant)")
print(f"Negative sin values: {negative_sin} (should be 0 for first quadrant)")

# Check monotonicity
cos_monotonic = all(cos_vals[i] >= cos_vals[i+1] for i in range(len(cos_vals)-1))
sin_monotonic = all(sin_vals[i] <= sin_vals[i+1] for i in range(len(sin_vals)-1))
print(f"Cos monotonically decreasing: {cos_monotonic}")
print(f"Sin monotonically increasing: {sin_monotonic}")

# Show first and last few values
print("\nFirst 5 values:")
for i in range(5):
    print(f"  eta={eta_range[i]:5d}: cos={cos_vals[i]:6d}, sin={sin_vals[i]:6d}")

print("\nLast 5 values:")
for i in range(-5, 0):
    print(f"  eta={eta_range[i]:5d}: cos={cos_vals[i]:6d}, sin={sin_vals[i]:6d}")

# Check if eta=32767 gives proper values
print("\n" + "=" * 60)
print("Critical check: eta=32767 (full wet)")
eta = 32767
phase = (eta * 0x40000000) >> 15
phases = np.array([phase], dtype=np.uint32)
cos_val, sin_val = _sincos_batch_q15(phases)
print(f"cos={cos_val[0]} (expected ~0)")
print(f"sin={sin_val[0]} (expected ~32767)")

if cos_val[0] > 1000 or sin_val[0] < 30000:
    print("WARNING: Crossfade at eta=32767 looks wrong!")
