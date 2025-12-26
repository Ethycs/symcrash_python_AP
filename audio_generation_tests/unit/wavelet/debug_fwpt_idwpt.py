"""Debug FWPT/IDWPT roundtrip at each level."""
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


def test_single_level_roundtrip():
    """Test a single CDF 5/3 lifting step roundtrip."""
    print("=" * 60)
    print("Testing single CDF 5/3 level roundtrip")
    print("=" * 60)

    lifting = CDF53Lifting()

    # Test with simple signal
    x = [int(1000 * np.sin(2 * np.pi * i / 16)) for i in range(16)]
    print(f"Input ({len(x)} samples): {x[:8]}...")

    # Forward
    approx, detail = lifting.forward_step(x)
    print(f"Approx ({len(approx)} samples): {approx}")
    print(f"Detail ({len(detail)} samples): {detail}")

    # Inverse
    x_recovered = lifting.inverse_step(approx, detail)
    print(f"Recovered ({len(x_recovered)} samples): {x_recovered[:8]}...")

    # Compare
    max_diff = max(abs(a - b) for a, b in zip(x, x_recovered[:len(x)]))
    print(f"Max diff: {max_diff}")

    if max_diff > 1:
        print("FAIL: Single level roundtrip not exact!")
    else:
        print("PASS: Single level roundtrip is bit-accurate")


def test_multilevel_roundtrip():
    """Test multiple levels of decomposition and reconstruction."""
    print("\n" + "=" * 60)
    print("Testing multi-level CDF 5/3 roundtrip")
    print("=" * 60)

    lifting = CDF53Lifting()

    # Start with 64 samples
    x = [int(1000 * np.sin(2 * np.pi * i / 64)) for i in range(64)]
    print(f"Input ({len(x)} samples): first 8 = {x[:8]}")

    # Decompose to depth 2 (for simplicity)
    # Level 1: 64 -> (32 approx, 32 detail)
    a1, d1 = lifting.forward_step(x)
    print(f"\nLevel 1:")
    print(f"  Approx: {len(a1)} samples, first 4 = {a1[:4]}")
    print(f"  Detail: {len(d1)} samples, first 4 = {d1[:4]}")

    # Level 2: 32 approx -> (16 approx, 16 detail)
    a2, d2 = lifting.forward_step(a1)
    print(f"\nLevel 2:")
    print(f"  Approx: {len(a2)} samples, first 4 = {a2[:4]}")
    print(f"  Detail: {len(d2)} samples, first 4 = {d2[:4]}")

    # Reconstruct level 2 -> level 1
    a1_rec = lifting.inverse_step(a2, d2)
    print(f"\nReconstruct Level 2 -> 1:")
    print(f"  a1_rec: {len(a1_rec)} samples, first 4 = {a1_rec[:4]}")
    max_diff_a1 = max(abs(a - b) for a, b in zip(a1, a1_rec[:len(a1)]))
    print(f"  Max diff from original a1: {max_diff_a1}")

    # Reconstruct level 1 -> level 0
    x_rec = lifting.inverse_step(a1_rec[:len(d1)], d1)
    print(f"\nReconstruct Level 1 -> 0:")
    print(f"  x_rec: {len(x_rec)} samples, first 8 = {x_rec[:8]}")
    max_diff_x = max(abs(a - b) for a, b in zip(x, x_rec[:len(x)]))
    print(f"  Max diff from original x: {max_diff_x}")

    if max_diff_x > 2:
        print("FAIL: Multi-level roundtrip not accurate!")
    else:
        print("PASS: Multi-level roundtrip is accurate")


def test_tree_structure_analysis():
    """Examine tree 0 structure."""
    print("\n" + "=" * 60)
    print("Analyzing tree 0 (balanced) structure")
    print("=" * 60)

    rom = build_tree_0_balanced()

    print(f"BLOCK_SIZE: {BLOCK_SIZE}")
    print(f"MAX_DEPTH: {MAX_DEPTH}")
    print(f"LEAF_COUNT: {LEAF_COUNT}")
    print(f"NODE_SIZE: {NODE_SIZE}")

    print(f"\nLeaf list ({len(rom.leaf_list)} leaves):")
    for i, (d, k) in enumerate(rom.leaf_list[:8]):
        print(f"  Leaf {i}: depth={d}, k={k}")
    print("  ...")

    print(f"\nSplit mask:")
    for d in range(MAX_DEPTH):
        splits = [k for k in range(2 ** d) if rom.split_mask.get(d, {}).get(k, False)]
        if splits:
            print(f"  Depth {d}: split nodes = {splits}")


def test_analyzer_synthesizer_direct():
    """Test PacketAnalyzer + PacketSynthesizer.synthesize_full directly."""
    print("\n" + "=" * 60)
    print("Testing PacketAnalyzer + PacketSynthesizer.synthesize_full")
    print("=" * 60)

    rom = build_tree_0_balanced()
    analyzer = PacketAnalyzer(rom)
    synthesizer = PacketSynthesizer(rom)

    # Test signal
    x = [int(Q15_ONE * 0.5 * np.sin(2 * np.pi * 5 * i / BLOCK_SIZE))
         for i in range(BLOCK_SIZE)]
    print(f"Input: {len(x)} samples, first 8 = {x[:8]}")
    print(f"Input energy: {sum(abs(s) for s in x)}")

    # Analyze (FWPT)
    coeff_nodes = analyzer.analyze(x)
    print(f"\nAnalysis produced {len(coeff_nodes)} nodes")

    # Check what's in the nodes
    total_energy = 0
    non_empty = 0
    for i, node in enumerate(coeff_nodes):
        node_energy = sum(abs(c) for c in node)
        total_energy += node_energy
        if node_energy > 0:
            non_empty += 1
            if i < 4 or node_energy > 1000:
                print(f"  Node {i}: {node}, energy={node_energy}")

    print(f"Total coefficient energy: {total_energy}")
    print(f"Non-empty nodes: {non_empty}/{len(coeff_nodes)}")

    # Synthesize with synthesize_full (proper CDF 5/3 inverse)
    x_rec = synthesizer.synthesize_full(coeff_nodes)
    print(f"\nSynthesize_full: {len(x_rec)} samples, first 8 = {x_rec[:8]}")
    print(f"Output energy: {sum(abs(s) for s in x_rec)}")

    # Compare
    max_diff = max(abs(a - b) for a, b in zip(x, x_rec))
    rms_diff = np.sqrt(np.mean([(a - b) ** 2 for a, b in zip(x, x_rec)]))
    correlation = np.corrcoef(np.array(x), np.array(x_rec))[0, 1]

    print(f"\nComparison:")
    print(f"  Max diff: {max_diff}")
    print(f"  RMS diff: {rms_diff:.1f}")
    print(f"  Correlation: {correlation:.6f}")


if __name__ == "__main__":
    test_single_level_roundtrip()
    test_multilevel_roundtrip()
    test_tree_structure_analysis()
    test_analyzer_synthesizer_direct()
