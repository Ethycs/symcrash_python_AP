#!/usr/bin/env python3
"""Deep profile of IDWPT to find exact bottlenecks."""

import time
import cProfile
import pstats
import io
import numpy as np
from symcrash_python_ap.wavelet.wavelet_packet_trees import PacketTreeSet

pts = PacketTreeSet()
carrier = [int(16384 * np.sin(2 * np.pi * 440 * i / 48000)) for i in range(64)]
mask = [32767] * 64

# Warmup
for _ in range(100):
    pts.effect_mode(0, carrier, mask)

# Profile 100 iterations
N = 100

print("Profiling effect_mode...")
pr = cProfile.Profile()
pr.enable()
for _ in range(N):
    pts.effect_mode(0, carrier, mask)
pr.disable()

# Print sorted by cumulative time
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(30)
print(s.getvalue())

# Also print by tottime (self time, excluding callees)
print("\n" + "="*60)
print("By SELF time (excluding callees):")
print("="*60)
s2 = io.StringIO()
ps2 = pstats.Stats(pr, stream=s2).sort_stats('tottime')
ps2.print_stats(30)
print(s2.getvalue())

# =============================================================================
# Test simplified inline lifting (no function call overhead)
# =============================================================================
print("\n" + "="*60)
print("Testing simplified inline lifting")
print("="*60)

def sat16_inline(x):
    """Inline saturation - faster for single values."""
    return max(-32768, min(32767, x))

def cdf53_forward_inline(x):
    """Simplified CDF 5/3 forward - no function calls."""
    n = len(x) // 2
    if n == 0:
        return x, []

    s = [0] * n
    d = [0] * n

    # Predict: d[i] = x[2i+1] - (x[2i] + x[2i+2]) / 2
    for i in range(n):
        left = x[2 * i]
        right = x[2 * i + 2] if 2 * i + 2 < len(x) else x[2 * i]
        avg = (left + right + 1) >> 1  # Round-to-nearest
        d[i] = sat16_inline(x[2 * i + 1] - avg)

    # Update: s[i] = x[2i] + (d[i-1] + d[i]) / 4
    for i in range(n):
        d_left = d[i - 1] if i > 0 else d[0]
        d_right = d[i]
        update = (d_left + d_right + 2) >> 2
        s[i] = sat16_inline(x[2 * i] + update)

    return s, d

def cdf53_inverse_inline(s, d):
    """Simplified CDF 5/3 inverse - no function calls."""
    n_s = len(s)
    n_d = len(d)

    if n_s == 0:
        return []
    if n_d == 0:
        return s

    n = max(n_s, n_d)

    # Pad if needed
    if n_s < n:
        s = s + [s[-1]] * (n - n_s)
    if n_d < n:
        d = d + [0] * (n - n_d)

    x = [0] * (2 * n)

    # Undo update: x[2i] = s[i] - (d[i-1] + d[i]) / 4
    for i in range(n):
        d_left = d[i - 1] if i > 0 else d[0]
        d_right = d[i] if i < len(d) else 0
        update = (d_left + d_right + 2) >> 2
        x[2 * i] = sat16_inline(s[i] - update)

    # Undo predict: x[2i+1] = d[i] + (x[2i] + x[2i+2]) / 2
    for i in range(n):
        left = x[2 * i]
        right = x[2 * i + 2] if 2 * i + 2 < 2 * n else x[2 * i]
        avg = (left + right + 1) >> 1
        d_val = d[i] if i < len(d) else 0
        x[2 * i + 1] = sat16_inline(d_val + avg)

    return x

# Test roundtrip
test = list(carrier)
s, d = cdf53_forward_inline(test)
reconstructed = cdf53_inverse_inline(s, d)
error = max(abs(test[i] - reconstructed[i]) for i in range(len(test)))
print(f"Roundtrip error (inline): {error}")

# Benchmark inline forward+inverse
N2 = 1000
start = time.perf_counter()
for _ in range(N2):
    s, d = cdf53_forward_inline(test)
    _ = cdf53_inverse_inline(s, d)
elapsed = time.perf_counter() - start
print(f"Single-level inline: {elapsed * 1e6 / N2:.1f} us")

# Full tree using inline
def fwpt_inline(x):
    """FWPT using inline lifting."""
    queue = [list(x)]
    for depth in range(6):
        next_queue = []
        for node in queue:
            if len(node) >= 2:
                s, d = cdf53_forward_inline(node)
                next_queue.append(s)
                next_queue.append(d)
            else:
                next_queue.append(node)
        queue = next_queue
    return queue

def idwpt_inline(leaves):
    """IDWPT using inline lifting."""
    queue = [list(l) for l in leaves]
    for depth in range(5, -1, -1):
        next_queue = []
        for i in range(0, len(queue), 2):
            s = queue[i] if i < len(queue) else [0]
            d = queue[i+1] if i+1 < len(queue) else []
            combined = cdf53_inverse_inline(s, d)
            next_queue.append(combined)
        queue = next_queue
    return queue[0] if queue else [0] * 64

# Warmup
for _ in range(10):
    leaves = fwpt_inline(carrier)
    _ = idwpt_inline(leaves)

# Benchmark full tree inline
start = time.perf_counter()
for _ in range(N2):
    leaves = fwpt_inline(carrier)
elapsed = time.perf_counter() - start
print(f"FWPT inline: {elapsed * 1e6 / N2:.1f} us")

start = time.perf_counter()
for _ in range(N2):
    result = idwpt_inline(leaves)
elapsed = time.perf_counter() - start
print(f"IDWPT inline: {elapsed * 1e6 / N2:.1f} us")

start = time.perf_counter()
for _ in range(N2):
    leaves = fwpt_inline(carrier)
    result = idwpt_inline(leaves)
elapsed = time.perf_counter() - start
total_inline = elapsed * 1e6 / N2
print(f"Full FWPT+IDWPT inline: {total_inline:.1f} us")

# Compare to original
original_time = 1200  # us from earlier benchmark
print(f"\nSpeedup: {original_time / total_inline:.1f}x")
print(f"Headroom: {1333 - total_inline:.1f} us ({(1333 - total_inline) / 1333 * 100:.0f}%)")

# =============================================================================
# Test the new fast_lifting module
# =============================================================================
print("\n" + "="*60)
print("Testing fast_lifting module")
print("="*60)

from symcrash_python_ap.wavelet.fast_lifting import (
    fwpt_balanced, idwpt_balanced, effect_mode_fast
)

# Warmup
for _ in range(10):
    leaves = fwpt_balanced(carrier)
    _ = idwpt_balanced(leaves)

# Benchmark
start = time.perf_counter()
for _ in range(N2):
    leaves = fwpt_balanced(carrier)
elapsed = time.perf_counter() - start
print(f"fast_lifting FWPT: {elapsed * 1e6 / N2:.1f} us")

start = time.perf_counter()
for _ in range(N2):
    result = idwpt_balanced(leaves)
elapsed = time.perf_counter() - start
print(f"fast_lifting IDWPT: {elapsed * 1e6 / N2:.1f} us")

start = time.perf_counter()
for _ in range(N2):
    result = effect_mode_fast(carrier, mask)
elapsed = time.perf_counter() - start
total_fast = elapsed * 1e6 / N2
print(f"fast_lifting effect_mode_fast: {total_fast:.1f} us")

print(f"\nSpeedup vs original: {original_time / total_fast:.1f}x")
print(f"Headroom: {1333 - total_fast:.1f} us ({(1333 - total_fast) / 1333 * 100:.0f}%)")
