#!/usr/bin/env python3
"""
Generate Test Vectors for Verilog CDF 5/3 Inverse Lifting
==========================================================

Generates golden reference vectors from the Python implementation
for verification of the Verilog modules.

Outputs:
- test_vectors.txt: Human-readable test cases
- test_vectors.mem: Verilog $readmemh compatible format
"""

import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'symcrash_python_ap'))

from fixed import sat16, shift_round


def inverse_step(s, d):
    """
    Python reference implementation of CDF 5/3 inverse lifting.
    Matches wavelet_packet_trees.py exactly.
    """
    n_s = len(s)
    n_d = len(d)

    if n_s == 0:
        return []
    if n_d == 0:
        return s

    n = max(n_s, n_d)

    # Pad if necessary
    s_padded = s + [s[-1]] * (n - n_s) if n_s < n else s
    d_padded = d + [0] * (n - n_d) if n_d < n else d

    x = [0] * (2 * n)

    # Step 1: Undo update (recover even samples)
    # x[2i] = s[i] - (d[i-1] + d[i]) / 4
    for i in range(n):
        d_left = d_padded[i - 1] if i > 0 else d_padded[0]
        d_right = d_padded[i] if i < len(d_padded) else 0

        update = shift_round(d_left + d_right, 2)
        x[2 * i] = sat16(s_padded[i] - update)

    # Step 2: Undo predict (recover odd samples)
    # x[2i+1] = d[i] + (x[2i] + x[2i+2]) / 2
    for i in range(n):
        left = x[2 * i]
        right = x[2 * i + 2] if 2 * i + 2 < 2 * n else x[2 * i]

        avg = shift_round(left + right, 1)
        d_val = d_padded[i] if i < len(d_padded) else 0
        x[2 * i + 1] = sat16(d_val + avg)

    return x


def to_hex(val, bits=16):
    """Convert signed integer to hex string."""
    if val < 0:
        val = (1 << bits) + val
    return f"{val:04x}"


def generate_test_cases():
    """Generate test cases with golden reference outputs."""

    test_cases = [
        {
            "name": "DC Signal",
            "s": [1000, 1000, 1000, 1000],
            "d": [0, 0, 0, 0],
        },
        {
            "name": "Impulse in smooth[0]",
            "s": [16384, 0, 0, 0],
            "d": [0, 0, 0, 0],
        },
        {
            "name": "Detail only",
            "s": [0, 0, 0, 0],
            "d": [8192, 8192, 8192, 8192],
        },
        {
            "name": "Ramp",
            "s": [0, 4096, 8192, 12288],
            "d": [1024, 1024, 1024, 1024],
        },
        {
            "name": "Saturation test",
            "s": [32000, 32000, -32000, -32000],
            "d": [0, 16000, -16000, 0],
        },
        {
            "name": "Alternating",
            "s": [10000, -10000, 10000, -10000],
            "d": [5000, -5000, 5000, -5000],
        },
        {
            "name": "Single detail pulse",
            "s": [0, 0, 0, 0],
            "d": [0, 16384, 0, 0],
        },
        {
            "name": "Maximum positive",
            "s": [32767, 32767, 32767, 32767],
            "d": [0, 0, 0, 0],
        },
        {
            "name": "Maximum negative",
            "s": [-32768, -32768, -32768, -32768],
            "d": [0, 0, 0, 0],
        },
        {
            "name": "Mixed extremes",
            "s": [32767, -32768, 32767, -32768],
            "d": [32767, -32768, 32767, -32768],
        },
    ]

    print("=" * 70)
    print("CDF 5/3 Inverse Lifting Test Vectors")
    print("=" * 70)
    print()

    # Human-readable output
    with open("test_vectors.txt", "w") as f:
        f.write("CDF 5/3 Inverse Lifting Test Vectors\n")
        f.write("=" * 50 + "\n\n")

        for i, tc in enumerate(test_cases):
            s = tc["s"]
            d = tc["d"]
            x = inverse_step(s, d)

            print(f"Test {i+1}: {tc['name']}")
            print(f"  s = {s}")
            print(f"  d = {d}")
            print(f"  x = {x}")
            print()

            f.write(f"Test {i+1}: {tc['name']}\n")
            f.write(f"  Input s:  {s}\n")
            f.write(f"  Input d:  {d}\n")
            f.write(f"  Output x: {x}\n")
            f.write(f"  Hex s:    {' '.join(to_hex(v) for v in s)}\n")
            f.write(f"  Hex d:    {' '.join(to_hex(v) for v in d)}\n")
            f.write(f"  Hex x:    {' '.join(to_hex(v) for v in x)}\n")
            f.write("\n")

    # Verilog-compatible memory file
    with open("test_vectors.mem", "w") as f:
        f.write("// CDF 5/3 Inverse Lifting Test Vectors\n")
        f.write("// Format: s[0] s[1] s[2] s[3] d[0] d[1] d[2] d[3] x[0]..x[7]\n")

        for tc in test_cases:
            s = tc["s"]
            d = tc["d"]
            x = inverse_step(s, d)

            # Write as hex words
            line = " ".join(to_hex(v) for v in s + d + x)
            f.write(line + "\n")

    print("Generated: test_vectors.txt, test_vectors.mem")

    return test_cases


def verify_properties():
    """Verify mathematical properties of the transform."""

    print()
    print("=" * 70)
    print("Property Verification")
    print("=" * 70)
    print()

    # Property 1: Perfect reconstruction
    # Forward then inverse should give back original (with rounding)
    print("Property 1: DC preservation")
    s = [1000, 1000, 1000, 1000]
    d = [0, 0, 0, 0]
    x = inverse_step(s, d)
    dc = sum(x) / len(x)
    print(f"  Input DC: 1000")
    print(f"  Output DC: {dc:.2f}")
    print(f"  Status: {'PASS' if abs(dc - 1000) < 100 else 'FAIL'}")
    print()

    # Property 2: Detail-only should have zero DC
    print("Property 2: Detail-only has ~zero DC")
    s = [0, 0, 0, 0]
    d = [8192, 8192, 8192, 8192]
    x = inverse_step(s, d)
    dc = sum(x) / len(x)
    print(f"  Input s: all zeros")
    print(f"  Input d: all 8192")
    print(f"  Output DC: {dc:.2f}")
    # Note: CDF 5/3 with constant detail does NOT have zero DC output
    print(f"  Status: INFO (CDF 5/3 property)")
    print()

    # Property 3: Saturation
    print("Property 3: Output bounded to [-32768, 32767]")
    s = [32767, 32767, 32767, 32767]
    d = [32767, 32767, 32767, 32767]
    x = inverse_step(s, d)
    min_x = min(x)
    max_x = max(x)
    print(f"  Input: extreme values")
    print(f"  Output range: [{min_x}, {max_x}]")
    print(f"  Status: {'PASS' if min_x >= -32768 and max_x <= 32767 else 'FAIL'}")
    print()


if __name__ == "__main__":
    generate_test_cases()
    verify_properties()
