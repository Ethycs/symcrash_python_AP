#!/usr/bin/env python3
"""Test sine_core 64K LUT fast path."""

from symcrash_python_ap.wavelet.sine_core import (
    sine_q15_fast, cosine_q15_fast,
    sine_q15_vectorized, cosine_q15_vectorized
)
import numpy as np

print("64K LUT Fast Path Tests")
print("=" * 40)

# Boundary values (16-bit phase: 0=0deg, 16384=90deg, 32768=180deg, 49152=270deg)
tests = [
    (0, "sin(0)", 0),
    (16384, "sin(90)", 32767),
    (32768, "sin(180)", 0),
    (49152, "sin(270)", -32767),
]
print("Scalar sine_q15_fast:")
passed = True
for phase, name, expected in tests:
    actual = sine_q15_fast(phase)
    ok = abs(actual - expected) <= 1
    print(f"  {name}: {actual} (expected {expected}) {'OK' if ok else 'FAIL'}")
    passed = passed and ok

# Cosine tests
cos_tests = [
    (0, "cos(0)", 32767),
    (16384, "cos(90)", 0),
    (32768, "cos(180)", -32767),
    (49152, "cos(270)", 0),
]
print("\nScalar cosine_q15_fast:")
for phase, name, expected in cos_tests:
    actual = cosine_q15_fast(phase)
    ok = abs(actual - expected) <= 1
    print(f"  {name}: {actual} (expected {expected}) {'OK' if ok else 'FAIL'}")
    passed = passed and ok

# Vectorized test
print("\nVectorized sine_q15_vectorized:")
phases = np.array([0, 16384, 32768, 49152], dtype=np.int32)
expected = np.array([0, 32767, 0, -32767])
actual = sine_q15_vectorized(phases)
max_error = np.abs(actual - expected).max()
print(f"  Max error: {max_error} LSB")
passed = passed and max_error <= 1

print("\nVectorized cosine_q15_vectorized:")
expected_cos = np.array([32767, 0, -32767, 0])
actual_cos = cosine_q15_vectorized(phases)
max_error_cos = np.abs(actual_cos - expected_cos).max()
print(f"  Max error: {max_error_cos} LSB")
passed = passed and max_error_cos <= 1

# Benchmark
import time
phases_large = np.arange(65536, dtype=np.int32)

# Warmup
for _ in range(10):
    sine_q15_vectorized(phases_large)

N = 1000
start = time.perf_counter()
for _ in range(N):
    sine_q15_vectorized(phases_large)
elapsed = time.perf_counter() - start
us_per_call = elapsed * 1e6 / N

print(f"\nBenchmark (65536 lookups):")
print(f"  sine_q15_vectorized: {us_per_call:.1f} us")

print("\n" + "=" * 40)
print(f"RESULT: {'PASS' if passed else 'FAIL'}")
