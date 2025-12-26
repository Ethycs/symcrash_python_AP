"""Simple test: 2-level decomposition and reconstruction."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "symcrash_python_ap"))

import numpy as np
from symcrash_python_ap.wavelet.wavelet_packet_trees import CDF53Lifting
from symcrash_python_ap.wavelet.carrier_lift import Q15_ONE


def test_two_level():
    """Test 2-level decomposition: 8 samples -> 4 leaves of 2 samples each."""
    print("=" * 60)
    print("2-Level Wavelet Packet Decomposition (8 samples)")
    print("=" * 60)

    lifting = CDF53Lifting()

    # 8 samples
    x = [1000, 2000, 3000, 4000, 3000, 2000, 1000, 0]
    print(f"Input x: {x}")

    # Level 1: 8 -> (4 approx, 4 detail)
    s1, d1 = lifting.forward_step(x)
    print(f"\nLevel 1:")
    print(f"  s1 (approx): {s1}")
    print(f"  d1 (detail): {d1}")

    # Level 2a: s1 (4) -> (2, 2)
    s1_s, s1_d = lifting.forward_step(s1)
    print(f"\nLevel 2 (from approx s1):")
    print(f"  s1_s: {s1_s}")
    print(f"  s1_d: {s1_d}")

    # Level 2b: d1 (4) -> (2, 2)
    d1_s, d1_d = lifting.forward_step(d1)
    print(f"\nLevel 2 (from detail d1):")
    print(f"  d1_s: {d1_s}")
    print(f"  d1_d: {d1_d}")

    # Now we have 4 leaves, each with 2 samples:
    # Leaf 0 (k=0): s1_s  (LL)
    # Leaf 1 (k=1): s1_d  (LH)
    # Leaf 2 (k=2): d1_s  (HL)
    # Leaf 3 (k=3): d1_d  (HH)

    print("\n=== RECONSTRUCTION ===")

    # Reconstruct s1 from s1_s, s1_d
    s1_rec = lifting.inverse_step(s1_s, s1_d)
    print(f"s1 reconstructed: {s1_rec}")
    print(f"  Original s1: {s1}")
    print(f"  Match: {s1_rec == s1}")

    # Reconstruct d1 from d1_s, d1_d
    d1_rec = lifting.inverse_step(d1_s, d1_d)
    print(f"d1 reconstructed: {d1_rec}")
    print(f"  Original d1: {d1}")
    print(f"  Match: {d1_rec == d1}")

    # Reconstruct x from s1_rec, d1_rec
    x_rec = lifting.inverse_step(s1_rec, d1_rec)
    print(f"\nx reconstructed: {x_rec}")
    print(f"  Original x: {x}")
    print(f"  Match: {x_rec == x}")

    max_diff = max(abs(a - b) for a, b in zip(x, x_rec))
    print(f"  Max diff: {max_diff}")


def test_frequency_ordering():
    """Check if the issue is frequency ordering of leaves."""
    print("\n" + "=" * 60)
    print("Wavelet Packet Frequency Ordering")
    print("=" * 60)

    # In wavelet packets, the leaves are often in 'paley' or 'sequency' order
    # not natural order. The standard reconstruction order is:
    #
    # For a 2-level decomposition:
    #   Level 0: x
    #   Level 1: (s1, d1) = (L, H)
    #   Level 2: (s1_s, s1_d, d1_s, d1_d) = (LL, LH, HL, HH)
    #
    # In sequency order (gray code): LL, LH, HH, HL
    # But for reconstruction, we need natural binary order: LL, LH, HL, HH

    print("Natural ordering (standard reconstruction):")
    print("  Node 0 (k=0): LL (from s1_s)")
    print("  Node 1 (k=1): LH (from s1_d)")
    print("  Node 2 (k=2): HL (from d1_s)")
    print("  Node 3 (k=3): HH (from d1_d)")
    print("\nReconstruction pairs:")
    print("  (0,1) -> s1, (2,3) -> d1, then (s1,d1) -> x")


if __name__ == "__main__":
    test_two_level()
    test_frequency_ordering()
