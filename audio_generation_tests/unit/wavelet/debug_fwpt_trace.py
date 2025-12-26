"""Trace through FWPT/IDWPT step by step to find the bug."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "symcrash_python_ap"))

import numpy as np
from symcrash_python_ap.wavelet.wavelet_packet_trees import (
    CDF53Lifting, PacketAnalyzer, PacketSynthesizer,
    BLOCK_SIZE, MAX_DEPTH, NODE_SIZE, LEAF_COUNT
)
from symcrash_python_ap.wavelet_trees import build_tree_0_balanced
from symcrash_python_ap.wavelet.carrier_lift import Q15_ONE


def test_manual_roundtrip():
    """Manually trace the decomposition/reconstruction to find bug."""
    print("=" * 60)
    print("Manual wavelet packet tree roundtrip")
    print("=" * 60)

    lifting = CDF53Lifting()
    rom = build_tree_0_balanced()

    # Input
    x = [int(Q15_ONE * 0.5 * np.sin(2 * np.pi * 5 * i / BLOCK_SIZE))
         for i in range(BLOCK_SIZE)]
    print(f"Input: {x[:8]}... (64 samples)")

    # === FORWARD: Full tree decomposition ===
    # tree_data[d][k] = data at node (d, k)
    tree_data = {d: {} for d in range(MAX_DEPTH + 1)}
    tree_data[0][0] = list(x)

    print("\n=== FORWARD DECOMPOSITION ===")
    for d in range(MAX_DEPTH):
        nodes_at_d = 2 ** d
        for k in range(nodes_at_d):
            if rom.split_mask.get(d, {}).get(k, False):
                parent = tree_data[d].get(k, [])
                if not parent:
                    continue

                approx, detail = lifting.forward_step(parent)
                tree_data[d + 1][2 * k] = approx
                tree_data[d + 1][2 * k + 1] = detail

                if d < 3 or (d == MAX_DEPTH - 1 and k < 4):
                    print(f"  d={d}, k={k}: {len(parent)} samples -> "
                          f"approx[{len(approx)}], detail[{len(detail)}]")

    # Check leaves at depth 6
    print(f"\nLeaves at depth {MAX_DEPTH}:")
    for k in range(min(8, 2 ** MAX_DEPTH)):
        data = tree_data[MAX_DEPTH].get(k, [])
        print(f"  k={k}: {data}")

    # === INVERSE: Full tree reconstruction ===
    print("\n=== INVERSE RECONSTRUCTION ===")

    # Start by copying leaves
    recon_data = {d: {} for d in range(MAX_DEPTH + 1)}
    for k in range(2 ** MAX_DEPTH):
        if k in tree_data[MAX_DEPTH]:
            recon_data[MAX_DEPTH][k] = list(tree_data[MAX_DEPTH][k])

    # Bottom-up
    for d in range(MAX_DEPTH - 1, -1, -1):
        nodes_at_d = 2 ** d
        for k in range(nodes_at_d):
            if rom.split_mask.get(d, {}).get(k, False):
                left = recon_data[d + 1].get(2 * k, [])
                right = recon_data[d + 1].get(2 * k + 1, [])

                if left and right:
                    combined = lifting.inverse_step(left, right)
                    recon_data[d][k] = combined

                    if d < 3 or (d == MAX_DEPTH - 1 and k < 4):
                        print(f"  d={d}, k={k}: approx[{len(left)}] + detail[{len(right)}] "
                              f"-> {len(combined)} samples")

    # Get result
    result = recon_data[0].get(0, [])
    print(f"\nReconstructed: {result[:8]}... ({len(result)} samples)")

    # Compare
    max_diff = max(abs(a - b) for a, b in zip(x, result[:len(x)]))
    print(f"Max diff: {max_diff}")

    if max_diff > 1:
        print("\nDifferences at each position:")
        for i in range(min(16, len(x))):
            diff = result[i] - x[i] if i < len(result) else 0
            if abs(diff) > 1:
                print(f"  [{i}]: {x[i]} vs {result[i] if i < len(result) else 'N/A'} (diff={diff})")


def test_compare_schedules():
    """Compare what the analyzer/synthesizer schedules are doing."""
    print("\n" + "=" * 60)
    print("Comparing analyzer/synthesizer schedules")
    print("=" * 60)

    rom = build_tree_0_balanced()
    analyzer = PacketAnalyzer(rom)
    synthesizer = PacketSynthesizer(rom)

    print(f"\nAnalyzer split schedule ({len(analyzer.split_schedule)} items):")
    print(f"  First 10: {analyzer.split_schedule[:10]}")
    print(f"  Last 10: {analyzer.split_schedule[-10:]}")

    print(f"\nSynthesizer combine schedule ({len(synthesizer.combine_schedule)} items):")
    print(f"  First 10: {synthesizer.combine_schedule[:10]}")
    print(f"  Last 10: {synthesizer.combine_schedule[-10:]}")

    # Check if combine schedule is reverse of split
    reversed_split = list(reversed(analyzer.split_schedule))
    if synthesizer.combine_schedule == reversed_split:
        print("\nSchedules are correctly reversed")
    else:
        print("\nWARNING: Schedules are NOT simple reverses!")
        print(f"  Split reversed: {reversed_split[:5]}...")
        print(f"  Combine: {synthesizer.combine_schedule[:5]}...")


if __name__ == "__main__":
    test_manual_roundtrip()
    test_compare_schedules()
