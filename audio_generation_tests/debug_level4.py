"""Debug level 4 specifically."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "symcrash_python_ap"))

import numpy as np
from symcrash_python_ap.wavelet_packet_trees import CDF53Lifting
from symcrash_python_ap.carrier_lift import Q15_ONE


def test_level4_detailed():
    """Debug level 4 decomposition step by step."""
    print("=" * 60)
    print("Debug Level 4 (16 leaves of 4 samples each)")
    print("=" * 60)

    lifting = CDF53Lifting()

    # Simple test signal
    x = [i * 1000 for i in range(64)]  # 0, 1000, 2000, ..., 63000
    print(f"Input: {x[:8]}... (64 samples)")

    # Level 1: 64 -> (32, 32)
    tree = {0: {0: list(x)}}
    s1, d1 = lifting.forward_step(tree[0][0])
    tree[1] = {0: s1, 1: d1}
    print(f"\nLevel 1: {len(s1)} approx, {len(d1)} detail")

    # Verify level 1 roundtrip
    x_rec_1 = lifting.inverse_step(s1, d1)
    diff_1 = max(abs(a - b) for a, b in zip(x, x_rec_1))
    print(f"  Level 1 roundtrip diff: {diff_1}")

    # Level 2: each (32) -> (16, 16)
    tree[2] = {}
    for k in range(2):
        data = tree[1][k]
        s, d = lifting.forward_step(data)
        tree[2][2*k] = s
        tree[2][2*k + 1] = d

    # Verify level 2 roundtrip
    tree_2_rec = {}
    for k in range(2):
        left = tree[2][2*k]
        right = tree[2][2*k + 1]
        tree_2_rec[k] = lifting.inverse_step(left, right)
    x_rec_2 = lifting.inverse_step(tree_2_rec[0], tree_2_rec[1])
    diff_2 = max(abs(a - b) for a, b in zip(x, x_rec_2))
    print(f"  Level 2 roundtrip diff: {diff_2}")

    # Level 3: each (16) -> (8, 8)
    tree[3] = {}
    for k in range(4):
        data = tree[2][k]
        s, d = lifting.forward_step(data)
        tree[3][2*k] = s
        tree[3][2*k + 1] = d

    # Verify level 3 roundtrip
    tree_3_rec = {}
    for k in range(4):
        left = tree[3][2*k]
        right = tree[3][2*k + 1]
        tree_3_rec[k] = lifting.inverse_step(left, right)
    tree_2_rec = {}
    for k in range(2):
        left = tree_3_rec[2*k]
        right = tree_3_rec[2*k + 1]
        tree_2_rec[k] = lifting.inverse_step(left, right)
    x_rec_3 = lifting.inverse_step(tree_2_rec[0], tree_2_rec[1])
    diff_3 = max(abs(a - b) for a, b in zip(x, x_rec_3))
    print(f"  Level 3 roundtrip diff: {diff_3}")

    # Level 4: each (8) -> (4, 4)
    tree[4] = {}
    for k in range(8):
        data = tree[3][k]
        s, d = lifting.forward_step(data)
        tree[4][2*k] = s
        tree[4][2*k + 1] = d

    print(f"\nLevel 4 nodes:")
    for k in range(min(4, 16)):
        print(f"  Node {k}: {tree[4][k]}")

    # Verify level 4 roundtrip
    tree_4_rec = {}
    for k in range(8):
        left = tree[4][2*k]
        right = tree[4][2*k + 1]
        tree_4_rec[k] = lifting.inverse_step(left, right)

    print(f"\nLevel 3 reconstructed:")
    for k in range(min(4, 8)):
        print(f"  Node {k}: {tree_4_rec[k]}")
        print(f"  Original: {tree[3][k]}")
        diff = [a - b for a, b in zip(tree[3][k], tree_4_rec[k])]
        print(f"  Diff: {diff}")

    # Continue reconstruction
    tree_3_rec = {}
    for k in range(4):
        left = tree_4_rec[2*k]
        right = tree_4_rec[2*k + 1]
        tree_3_rec[k] = lifting.inverse_step(left, right)

    tree_2_rec = {}
    for k in range(2):
        left = tree_3_rec[2*k]
        right = tree_3_rec[2*k + 1]
        tree_2_rec[k] = lifting.inverse_step(left, right)

    x_rec_4 = lifting.inverse_step(tree_2_rec[0], tree_2_rec[1])
    diff_4 = max(abs(a - b) for a, b in zip(x, x_rec_4))
    print(f"\n  Level 4 roundtrip diff: {diff_4}")


def test_8sample_roundtrip():
    """Test 8-sample -> 4-sample -> 8-sample specifically."""
    print("\n" + "=" * 60)
    print("Testing 8-sample roundtrip")
    print("=" * 60)

    lifting = CDF53Lifting()

    x8 = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    print(f"Input: {x8}")

    s, d = lifting.forward_step(x8)
    print(f"After forward: approx={s}, detail={d}")

    x8_rec = lifting.inverse_step(s, d)
    print(f"After inverse: {x8_rec}")

    diff = [a - b for a, b in zip(x8, x8_rec)]
    print(f"Diff: {diff}")


if __name__ == "__main__":
    test_level4_detailed()
    test_8sample_roundtrip()
