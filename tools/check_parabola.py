#!/usr/bin/env python3
"""Check parabola sine accuracy."""

import numpy as np

# Check parabola at key points
def parabola_sine(p):
    if p < 16384:
        return (4 * p * (16384 - p)) >> 14
    elif p < 32768:
        x = 32768 - p
        return (4 * x * (16384 - x)) >> 14
    elif p < 49152:
        x = p - 32768
        return -((4 * x * (16384 - x)) >> 14)
    else:
        x = 65536 - p
        return -((4 * x * (16384 - x)) >> 14)

print("Phase -> Parabola vs Expected")
print("0 (0 deg):", parabola_sine(0), "expected 0")
print("16384 (90 deg):", parabola_sine(16384), "expected 32767")
print("32768 (180 deg):", parabola_sine(32768), "expected 0")
print("49152 (270 deg):", parabola_sine(49152), "expected -32767")

# The parabola peaks at x=8192 (45 deg in quarter), not at boundary
print()
print("Parabola at 8192 (mid-quarter):", parabola_sine(8192))
print("Expected sin(45)*32767 =", int(32767 * 0.707))

# The issue: parabola 4x(1-x) peaks at x=0.5, which maps to phase 8192
# But sine peaks at phase 16384
# So the parabola is a poor approximation for audio!

# Let's compute actual error
errors = []
for p in range(65536):
    expected = int(32767 * np.sin(p * 2 * np.pi / 65536))
    actual = parabola_sine(p)
    errors.append(abs(expected - actual))

print()
print(f"Max error: {max(errors)} LSB")
print(f"RMS error: {np.sqrt(np.mean([e**2 for e in errors])):.1f} LSB")
print(f"Max error at phase: {errors.index(max(errors))}")
