"""Test each level of wavelet packet decomposition."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "symcrash_python_ap"))

import numpy as np
from symcrash_python_ap.wavelet.wavelet_packet_trees import CDF53Lifting
from symcrash_python_ap.wavelet.carrier_lift import Q15_ONE


def test_level_by_level():
    """Test roundtrip at each level of decomposition."""
    print("=" * 60)
    print("Testing roundtrip at each decomposition level")
    print("=" * 60)

    lifting = CDF53Lifting()

    # 64 samples
    x = [int(Q15_ONE * 0.5 * np.sin(2 * np.pi * 5 * i / 64)) for i in range(64)]

    for max_depth in range(1, 7):  # Test 1 to 6 levels
        # Forward decomposition up to max_depth
        tree = {d: {} for d in range(max_depth + 1)}
        tree[0][0] = list(x)

        for d in range(max_depth):
            num_nodes = 2 ** d
            for k in range(num_nodes):
                data = tree[d].get(k, [])
                if len(data) >= 2:
                    s, d_coeff = lifting.forward_step(data)
                    tree[d + 1][2 * k] = s
                    tree[d + 1][2 * k + 1] = d_coeff

        # Inverse reconstruction
        recon = {d: {} for d in range(max_depth + 1)}
        for k in range(2 ** max_depth):
            recon[max_depth][k] = list(tree[max_depth].get(k, []))

        for d in range(max_depth - 1, -1, -1):
            num_nodes = 2 ** d
            for k in range(num_nodes):
                left = recon[d + 1].get(2 * k, [])
                right = recon[d + 1].get(2 * k + 1, [])
                if left and right:
                    combined = lifting.inverse_step(left, right)
                    recon[d][k] = combined

        result = recon[0].get(0, [])
        max_diff = max(abs(a - b) for a, b in zip(x, result[:len(x)]))

        status = "PASS" if max_diff == 0 else "FAIL"
        print(f"  Level {max_depth}: {2 ** max_depth} leaves, "
              f"samples per leaf = {64 // (2 ** max_depth)}, "
              f"max diff = {max_diff} [{status}]")


def test_boundary_extension():
    """Test if boundary extension is the issue."""
    print("\n" + "=" * 60)
    print("Testing boundary extension behavior")
    print("=" * 60)

    lifting = CDF53Lifting()

    # Test with 2 samples (edge case)
    x2 = [1000, 2000]
    s2, d2 = lifting.forward_step(x2)
    x2_rec = lifting.inverse_step(s2, d2)
    print(f"2 samples: {x2} -> s={s2}, d={d2} -> {x2_rec}")
    print(f"  Diff: {[a - b for a, b in zip(x2, x2_rec)]}")

    # Test with 4 samples
    x4 = [1000, 2000, 3000, 4000]
    s4, d4 = lifting.forward_step(x4)
    x4_rec = lifting.inverse_step(s4, d4)
    print(f"4 samples: {x4} -> {x4_rec}")
    print(f"  Diff: {[a - b for a, b in zip(x4, x4_rec)]}")


def test_1_sample_leaf():
    """What happens with 1-sample leaves?"""
    print("\n" + "=" * 60)
    print("Testing 1-sample leaf behavior")
    print("=" * 60)

    lifting = CDF53Lifting()

    # At depth 6 with 64 samples, we have 1 sample per leaf
    # But forward_step on 2 samples gives 1 approx, 1 detail

    x2 = [5000, 10000]
    s, d = lifting.forward_step(x2)
    print(f"Forward: {x2} -> approx={s}, detail={d}")

    x2_rec = lifting.inverse_step(s, d)
    print(f"Inverse: approx={s}, detail={d} -> {x2_rec}")

    # Check difference
    diff = [a - b for a, b in zip(x2, x2_rec)]
    print(f"Diff: {diff}")

    # Now test chained: go from 4 -> 2 -> 1 and back
    print("\nChained test: 4 -> 2 -> 1 -> 2 -> 4")
    x4 = [1000, 2000, 3000, 4000]
    print(f"  Level 0: {x4}")

    # Level 1
    s1, d1 = lifting.forward_step(x4)
    print(f"  Level 1: approx={s1}, detail={d1}")

    # Level 2
    s1_s, s1_d = lifting.forward_step(s1)
    d1_s, d1_d = lifting.forward_step(d1)
    print(f"  Level 2: LL={s1_s}, LH={s1_d}, HL={d1_s}, HH={d1_d}")

    # Reconstruct
    s1_rec = lifting.inverse_step(s1_s, s1_d)
    d1_rec = lifting.inverse_step(d1_s, d1_d)
    print(f"  Level 1 (rec): approx={s1_rec}, detail={d1_rec}")

    x4_rec = lifting.inverse_step(s1_rec, d1_rec)
    print(f"  Level 0 (rec): {x4_rec}")

    diff = [a - b for a, b in zip(x4, x4_rec)]
    print(f"  Diff: {diff}")


if __name__ == "__main__":
    test_level_by_level()
    test_boundary_extension()
    test_1_sample_leaf()
