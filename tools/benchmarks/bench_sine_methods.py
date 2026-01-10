#!/usr/bin/env python3
"""Benchmark different sine implementations for carrier render."""

import time
import numpy as np

# Test data - 64 samples, 8 harmonics worth of phases
np.random.seed(42)
phases_16bit = np.random.randint(0, 65536, size=(8, 64), dtype=np.int32)

# =============================================================================
# Method 1: Current parabola (half-wave symmetry)
# =============================================================================
def parabola_sine_vectorized(phase_norm):
    """Current implementation using half-wave symmetry."""
    is_negative = phase_norm >= 32768
    p = np.where(is_negative, phase_norm - 32768, phase_norm)
    is_falling = p >= 16384
    x = np.where(is_falling, 32768 - p, p)
    result = (4 * x * (16384 - x)) >> 14
    return np.where(is_negative, -result, result).astype(np.int32)

# =============================================================================
# Method 2: LUT-based (full 64K table, no interpolation)
# =============================================================================
# Pre-generate full sine table
_SINE_LUT_64K = (np.sin(np.arange(65536) * 2 * np.pi / 65536) * 32767).astype(np.int16)

def lut_sine_vectorized(phase_norm):
    """LUT lookup - O(1) per sample, no math."""
    return _SINE_LUT_64K[phase_norm].astype(np.int32)

# =============================================================================
# Method 3: Quarter-wave LUT with interpolation (like sine_core)
# =============================================================================
_QW_SIZE = 1024
_QW_TABLE = (np.sin(np.linspace(0, np.pi/2, _QW_SIZE + 1)) * 32767).astype(np.int32)

def qw_sine_vectorized(phase_norm):
    """Quarter-wave LUT with half-wave symmetry (no interpolation)."""
    # Half-wave symmetry
    is_negative = phase_norm >= 32768
    p = np.where(is_negative, phase_norm - 32768, phase_norm)

    # Quarter-wave symmetry
    is_falling = p >= 16384
    q = np.where(is_falling, 32768 - p, p)

    # Scale to table index (0-16383 -> 0-1024)
    indices = (q >> 4).astype(np.int32)  # 16384 / 16 = 1024
    indices = np.clip(indices, 0, _QW_SIZE)

    result = _QW_TABLE[indices]
    return np.where(is_negative, -result, result).astype(np.int32)

# =============================================================================
# Benchmark
# =============================================================================
N = 1000

# Warmup
for _ in range(100):
    for h in range(8):
        parabola_sine_vectorized(phases_16bit[h])
        lut_sine_vectorized(phases_16bit[h])
        qw_sine_vectorized(phases_16bit[h])

print("Sine Method Comparison (64 samples, 8 harmonics)")
print("=" * 60)

# Parabola
start = time.perf_counter()
for _ in range(N):
    for h in range(8):
        parabola_sine_vectorized(phases_16bit[h])
elapsed = time.perf_counter() - start
parabola_us = elapsed * 1e6 / N
print(f"Parabola (current):     {parabola_us:6.1f} us/block")

# Full LUT
start = time.perf_counter()
for _ in range(N):
    for h in range(8):
        lut_sine_vectorized(phases_16bit[h])
elapsed = time.perf_counter() - start
lut_us = elapsed * 1e6 / N
print(f"64K LUT (no interp):    {lut_us:6.1f} us/block")

# Quarter-wave LUT
start = time.perf_counter()
for _ in range(N):
    for h in range(8):
        qw_sine_vectorized(phases_16bit[h])
elapsed = time.perf_counter() - start
qw_us = elapsed * 1e6 / N
print(f"1K Quarter-wave LUT:    {qw_us:6.1f} us/block")

print()
print(f"LUT speedup vs parabola:   {parabola_us / lut_us:.2f}x")
print(f"QW LUT speedup vs parabola: {parabola_us / qw_us:.2f}x")

# =============================================================================
# Quality comparison
# =============================================================================
print()
print("=" * 60)
print("Quality Comparison (vs numpy.sin)")
print("=" * 60)

# Reference
ref_phases = np.arange(65536)
ref_sin = (np.sin(ref_phases * 2 * np.pi / 65536) * 32767).astype(np.int32)

# Parabola error
para_sin = parabola_sine_vectorized(ref_phases)
para_error = np.abs(para_sin - ref_sin)
print(f"Parabola:  max error = {para_error.max()} LSB, RMS = {np.sqrt((para_error**2).mean()):.1f} LSB")

# Full LUT error (should be 0)
lut_sin = lut_sine_vectorized(ref_phases)
lut_error = np.abs(lut_sin - ref_sin)
print(f"64K LUT:   max error = {lut_error.max()} LSB, RMS = {np.sqrt((lut_error**2).mean()):.1f} LSB")

# QW LUT error
qw_sin = qw_sine_vectorized(ref_phases)
qw_error = np.abs(qw_sin - ref_sin)
print(f"1K QW LUT: max error = {qw_error.max()} LSB, RMS = {np.sqrt((qw_error**2).mean()):.1f} LSB")

# =============================================================================
# Memory cost
# =============================================================================
print()
print("=" * 60)
print("Memory Cost")
print("=" * 60)
print(f"Parabola:  0 bytes (computed)")
print(f"64K LUT:   {_SINE_LUT_64K.nbytes:,} bytes ({_SINE_LUT_64K.nbytes / 1024:.0f} KB)")
print(f"1K QW LUT: {_QW_TABLE.nbytes:,} bytes ({_QW_TABLE.nbytes / 1024:.1f} KB)")
