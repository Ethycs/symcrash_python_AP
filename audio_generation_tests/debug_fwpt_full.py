"""Test full 6-level wavelet packet decomposition with 64 samples."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "symcrash_python_ap"))

import numpy as np
from symcrash_python_ap.wavelet_packet_trees import CDF53Lifting
from symcrash_python_ap.carrier_lift import Q15_ONE


def test_full_tree():
    """Full 6-level wavelet packet decomposition."""
    print("=" * 60)
    print("6-Level Wavelet Packet Decomposition (64 samples)")
    print("=" * 60)

    lifting = CDF53Lifting()

    # 64 samples
    x = [int(Q15_ONE * 0.5 * np.sin(2 * np.pi * 5 * i / 64)) for i in range(64)]
    print(f"Input: {x[:8]}...")

    # === FORWARD ===
    # tree[d][k] = data at (depth d, position k)
    tree = {d: {} for d in range(7)}  # 0 to 6
    tree[0][0] = list(x)

    # Decompose all nodes at each level
    for d in range(6):  # 0 to 5
        num_nodes = 2 ** d
        for k in range(num_nodes):
            data = tree[d].get(k, [])
            if len(data) >= 2:
                s, d_coeff = lifting.forward_step(data)
                tree[d + 1][2 * k] = s      # Left child (approx)
                tree[d + 1][2 * k + 1] = d_coeff  # Right child (detail)

    print("\n=== FORWARD COMPLETE ===")
    print(f"Leaves at depth 6 ({len(tree[6])} nodes):")
    for k in range(min(8, 64)):
        print(f"  k={k}: {tree[6].get(k, [])}")

    # === INVERSE ===
    # Start from leaves
    recon = {d: {} for d in range(7)}
    for k in range(64):
        recon[6][k] = list(tree[6].get(k, [0]))

    # Reconstruct bottom-up
    for d in range(5, -1, -1):  # 5 down to 0
        num_nodes = 2 ** d
        for k in range(num_nodes):
            left = recon[d + 1].get(2 * k, [])
            right = recon[d + 1].get(2 * k + 1, [])
            if left and right:
                combined = lifting.inverse_step(left, right)
                recon[d][k] = combined

    result = recon[0].get(0, [])
    print(f"\n=== INVERSE COMPLETE ===")
    print(f"Reconstructed: {result[:8]}...")

    # Compare
    max_diff = max(abs(a - b) for a, b in zip(x, result[:len(x)]))
    print(f"\nMax diff: {max_diff}")

    if max_diff == 0:
        print("PASS: Perfect reconstruction!")
    else:
        print(f"FAIL: Differences found")
        print("First 16 differences:")
        for i in range(min(16, len(x))):
            if i < len(result):
                diff = result[i] - x[i]
                if abs(diff) > 0:
                    print(f"  [{i}]: {x[i]} vs {result[i]} (diff={diff})")


def test_compare_to_analyzer():
    """Compare manual decomposition to analyzer."""
    print("\n" + "=" * 60)
    print("Comparing manual decomposition to PacketAnalyzer")
    print("=" * 60)

    from symcrash_python_ap.wavelet_packet_trees import PacketAnalyzer, PacketSynthesizer
    from symcrash_python_ap.wavelet_trees import build_tree_0_balanced

    lifting = CDF53Lifting()
    rom = build_tree_0_balanced()
    analyzer = PacketAnalyzer(rom)
    synthesizer = PacketSynthesizer(rom)

    x = [int(Q15_ONE * 0.5 * np.sin(2 * np.pi * 5 * i / 64)) for i in range(64)]

    # Manual decomposition
    tree = {d: {} for d in range(7)}
    tree[0][0] = list(x)
    for d in range(6):
        num_nodes = 2 ** d
        for k in range(num_nodes):
            data = tree[d].get(k, [])
            if len(data) >= 2:
                s, d_coeff = lifting.forward_step(data)
                tree[d + 1][2 * k] = s
                tree[d + 1][2 * k + 1] = d_coeff

    # Analyzer decomposition
    coeff_nodes = analyzer.analyze(x)

    print("\nComparing leaf coefficients:")
    mismatches = 0
    for k in range(64):
        manual = tree[6].get(k, [0])
        analyzer_result = coeff_nodes[k] if k < len(coeff_nodes) else [0]

        if manual != analyzer_result:
            mismatches += 1
            if mismatches <= 8:
                print(f"  k={k}: manual={manual}, analyzer={analyzer_result}")

    if mismatches == 0:
        print("  All coefficients match!")
    else:
        print(f"\n  Total mismatches: {mismatches}")

    # Now test synthesis
    print("\nComparing reconstruction:")

    # Manual reconstruction
    recon = {d: {} for d in range(7)}
    for k in range(64):
        recon[6][k] = list(tree[6].get(k, [0]))

    for d in range(5, -1, -1):
        num_nodes = 2 ** d
        for k in range(num_nodes):
            left = recon[d + 1].get(2 * k, [])
            right = recon[d + 1].get(2 * k + 1, [])
            if left and right:
                combined = lifting.inverse_step(left, right)
                recon[d][k] = combined

    manual_result = recon[0].get(0, [])

    # Synthesizer reconstruction
    synth_result = synthesizer.synthesize_full(coeff_nodes)

    manual_diff = max(abs(a - b) for a, b in zip(x, manual_result[:len(x)]))
    synth_diff = max(abs(a - b) for a, b in zip(x, synth_result[:len(x)]))

    print(f"  Manual reconstruction max diff: {manual_diff}")
    print(f"  Synthesizer reconstruction max diff: {synth_diff}")


if __name__ == "__main__":
    test_full_tree()
    test_compare_to_analyzer()
