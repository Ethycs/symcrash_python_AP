"""Test if saturation is causing the roundtrip errors."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "symcrash_python_ap"))

import numpy as np
from symcrash_python_ap.wavelet_packet_trees import CDF53Lifting
from symcrash_python_ap.carrier_lift import Q15_ONE


def test_with_bounded_signal():
    """Test with a signal that stays within Q15 bounds."""
    print("=" * 60)
    print("Testing with bounded signal (no saturation)")
    print("=" * 60)

    lifting = CDF53Lifting()

    # Signal that stays within bounds: max amplitude = 0.5 * Q15_ONE
    x = [int(0.3 * Q15_ONE * np.sin(2 * np.pi * 5 * i / 64)) for i in range(64)]
    print(f"Input max: {max(abs(s) for s in x)}, Q15_ONE: {Q15_ONE}")
    print(f"Input: {x[:8]}...")

    # Full 6-level decomposition
    tree = {0: {0: list(x)}}

    for d in range(6):
        tree[d + 1] = {}
        for k in range(2 ** d):
            data = tree[d].get(k, [])
            if len(data) >= 2:
                s, det = lifting.forward_step(data)
                tree[d + 1][2 * k] = s
                tree[d + 1][2 * k + 1] = det

    # Check coefficients for saturation
    print(f"\nLeaf coefficient max values:")
    max_coeff = 0
    for k in range(64):
        for val in tree[6].get(k, []):
            max_coeff = max(max_coeff, abs(val))
    print(f"  Max coefficient: {max_coeff}")
    if max_coeff >= Q15_ONE:
        print("  WARNING: Coefficients saturated!")

    # Full reconstruction
    recon = {d: {} for d in range(7)}
    for k in range(64):
        recon[6][k] = list(tree[6].get(k, []))

    for d in range(5, -1, -1):
        for k in range(2 ** d):
            left = recon[d + 1].get(2 * k, [])
            right = recon[d + 1].get(2 * k + 1, [])
            if left and right:
                combined = lifting.inverse_step(left, right)
                recon[d][k] = combined

    result = recon[0].get(0, [])

    max_diff = max(abs(a - b) for a, b in zip(x, result))
    print(f"\nMax diff: {max_diff}")

    if max_diff == 0:
        print("PASS: Perfect roundtrip with bounded signal!")
    else:
        print(f"FAIL: Still has errors even with bounded signal")


def test_lifting_overflow():
    """Check for overflow in lifting operations."""
    print("\n" + "=" * 60)
    print("Checking for overflow in lifting")
    print("=" * 60)

    lifting = CDF53Lifting()

    # Values that might cause overflow in predict step
    # predict: d[n] = x[2n+1] - (x[2n] + x[2n+2]) / 2
    # Worst case: x[2n+1] = 32767, x[2n] = -32768, x[2n+2] = -32768
    #             d[n] = 32767 - (-32768) = 65535 -> overflow!

    x = [32767, -32768, 32767, -32768, 32767, -32768, 32767, -32768]
    print(f"Worst case input: {x}")

    s, d = lifting.forward_step(x)
    print(f"Approx: {s}")
    print(f"Detail: {d}")

    x_rec = lifting.inverse_step(s, d)
    print(f"Reconstructed: {x_rec}")

    diff = [a - b for a, b in zip(x, x_rec)]
    print(f"Diff: {diff}")


def test_intermediate_values():
    """Check intermediate values during decomposition."""
    print("\n" + "=" * 60)
    print("Checking intermediate values")
    print("=" * 60)

    lifting = CDF53Lifting()

    # Use the original sine signal
    x = [int(Q15_ONE * 0.5 * np.sin(2 * np.pi * 5 * i / 64)) for i in range(64)]

    tree = {0: {0: list(x)}}
    max_vals = {0: max(abs(v) for v in x)}

    for d in range(6):
        tree[d + 1] = {}
        max_at_level = 0
        for k in range(2 ** d):
            data = tree[d].get(k, [])
            if len(data) >= 2:
                s, det = lifting.forward_step(data)
                tree[d + 1][2 * k] = s
                tree[d + 1][2 * k + 1] = det
                max_at_level = max(max_at_level, max(abs(v) for v in s))
                max_at_level = max(max_at_level, max(abs(v) for v in det))
        max_vals[d + 1] = max_at_level
        if max_at_level >= Q15_ONE:
            print(f"Level {d + 1}: MAX VALUE = {max_at_level} (SATURATED!)")
        else:
            print(f"Level {d + 1}: max = {max_at_level}")


if __name__ == "__main__":
    test_with_bounded_signal()
    test_lifting_overflow()
    test_intermediate_values()
