#!/usr/bin/env python3
"""Test sincos interpolation accuracy."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from symcrash_python_ap.vm.lanes import _sincos_batch_q15

# Test across the full eta range
eta_vals = np.arange(0, 32768, dtype=np.int32)
phases = ((eta_vals.astype(np.int64) * 0x40000000) >> 15).astype(np.uint32)

cos_vals, sin_vals = _sincos_batch_q15(phases)

# Compute expected values using floating point
eta_float = eta_vals / 32767.0  # 0 to 1
expected_cos = (np.cos(eta_float * np.pi / 2) * 32767).astype(np.int32)
expected_sin = (np.sin(eta_float * np.pi / 2) * 32767).astype(np.int32)

# Compute errors
cos_error = cos_vals - expected_cos
sin_error = sin_vals - expected_sin

print("Sine/Cosine interpolation accuracy test:")
print(f"  Sine error: max={np.max(np.abs(sin_error))}, rms={np.sqrt(np.mean(sin_error**2)):.2f}")
print(f"  Cosine error: max={np.max(np.abs(cos_error))}, rms={np.sqrt(np.mean(cos_error**2)):.2f}")

# Check equal-power property: cos^2 + sin^2 should = 1.0 (= 32767^2 in Q15*Q15 >> 15 = 32767)
power_sum = ((cos_vals.astype(np.int64) ** 2 + sin_vals.astype(np.int64) ** 2) >> 15).astype(np.int32)
expected_power = 32767
power_error = power_sum - expected_power

print(f"\nEqual-power check (should be ~32767 everywhere):")
print(f"  Power sum range: [{power_sum.min()}, {power_sum.max()}]")
print(f"  Power error: max={np.max(np.abs(power_error))}, mean={np.mean(power_error):.2f}")

# Check at key points
print(f"\nKey points:")
print(f"  eta=0: cos={cos_vals[0]} (exp=32767), sin={sin_vals[0]} (exp=0)")
print(f"  eta=16384: cos={cos_vals[16384]} (exp=23170), sin={sin_vals[16384]} (exp=23170)")
print(f"  eta=32767: cos={cos_vals[32767]} (exp=0), sin={sin_vals[32767]} (exp=32767)")
