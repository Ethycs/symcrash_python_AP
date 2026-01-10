#!/usr/bin/env python3
"""Benchmark fused vs separate FWPT+IDWPT."""

import time
import numpy as np
from symcrash_python_ap.wavelet.fast_lifting import (
    effect_mode_fast, effect_mode_fused
)

# Test data
carrier = [int(16384 * np.sin(2 * np.pi * 440 * i / 48000)) for i in range(64)]
mask = [32767] * 64

# Warmup
for _ in range(100):
    effect_mode_fast(carrier, mask)
    effect_mode_fused(carrier, mask)

N = 1000

# Benchmark effect_mode_fast
start = time.perf_counter()
for _ in range(N):
    effect_mode_fast(carrier, mask)
elapsed = time.perf_counter() - start
fast_us = elapsed * 1e6 / N
print(f"effect_mode_fast:  {fast_us:.1f} us")

# Benchmark effect_mode_fused
start = time.perf_counter()
for _ in range(N):
    effect_mode_fused(carrier, mask)
elapsed = time.perf_counter() - start
fused_us = elapsed * 1e6 / N
print(f"effect_mode_fused: {fused_us:.1f} us")

print()
if fused_us < fast_us:
    print(f"Fused is {fast_us / fused_us:.2f}x faster")
else:
    print(f"Fast is {fused_us / fast_us:.2f}x faster")

# Verify correctness
result_fast = effect_mode_fast(carrier, mask)
result_fused = effect_mode_fused(carrier, mask)
max_diff = max(abs(a - b) for a, b in zip(result_fast, result_fused))
print(f"\nMax difference between implementations: {max_diff}")
