"""Replicate the working test from debug_wavelet_phase.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "symcrash_python_ap"))

import math
import numpy as np
from symcrash_python_ap.wavelet_packet_trees import PacketTreeSet, LEAF_COUNT, NODE_SIZE, BLOCK_SIZE
from symcrash_python_ap.leaf_generator import LeafGenerator
from symcrash_python_ap.carrier_lift import Q15_ONE

FS = 48000
BLOCKS_PER_SEC = FS // BLOCK_SIZE


def analyze(samples, name):
    arr = np.array(samples, dtype=np.float64)
    deltas = np.diff(arr)
    d2 = np.diff(deltas)

    print(f"\n{name}:")
    print(f"  Samples: {len(samples)}")
    print(f"  Peak: {int(np.max(np.abs(arr)))}")
    print(f"  Max delta: {int(np.max(np.abs(deltas)))}")
    print(f"  Max d2: {int(np.max(np.abs(d2)))}")
    print(f"  Spikes: >500={np.sum(np.abs(d2) > 500)}, >1k={np.sum(np.abs(d2) > 1000)}")


def test_original():
    """Exact copy of the working test from debug_wavelet_phase.py"""
    print("\n" + "=" * 60)
    print("TEST: Original working method")
    print("=" * 60)

    trees = PacketTreeSet()
    leaf_gen = LeafGenerator()

    # Generate leaf content for C4
    f0_hz = 261.63
    harmonic_amps = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]  # Pure sine
    leaf_content = leaf_gen.generate_from_fundamental(f0_hz, harmonic_amps)

    print(f"  leaf_content.coeffs_q15 length: {len(leaf_content.coeffs_q15)}")
    print(f"  Non-zero leaf_content entries:")
    for i, c in enumerate(leaf_content.coeffs_q15):
        if c != 0:
            print(f"    coeffs_q15[{i}] = {c}")

    # Build nodes - EXACTLY as in debug_wavelet_phase.py
    nodes = []
    for leaf in range(LEAF_COUNT):
        node_coeffs = []
        for c in range(NODE_SIZE):
            if leaf < len(leaf_content.coeffs_q15) // NODE_SIZE:  # 64//4 = 16
                idx = leaf * NODE_SIZE + c
                if idx < len(leaf_content.coeffs_q15):
                    node_coeffs.append(leaf_content.coeffs_q15[idx])
                else:
                    node_coeffs.append(0)
            else:
                node_coeffs.append(0)
        nodes.append(node_coeffs)

    # Show non-zero nodes
    print(f"\n  Non-zero nodes after building:")
    for i, node in enumerate(nodes):
        if any(c != 0 for c in node):
            print(f"    node[{i}] = {node}")

    # Initialize phase state
    phase_state = []
    for leaf_idx in range(LEAF_COUNT):
        for coeff_idx in range(NODE_SIZE):
            phase = coeff_idx * (math.pi * 2 / NODE_SIZE) + leaf_idx * (math.pi / LEAF_COUNT)
            phase_state.append(phase)

    # Render
    samples = []
    synth = trees.synthesizers[0]  # Tree 0 = Balanced

    for _ in range(int(0.5 * BLOCKS_PER_SEC)):
        block = synth.synthesize(nodes, phase_state=phase_state)
        samples.extend(block)

    analyze(samples, "Original working method")


def test_with_tree1():
    """Same as above but with Tree 1 (Harmonic) like other tests use"""
    print("\n" + "=" * 60)
    print("TEST: Original method with Tree 1 (Harmonic)")
    print("=" * 60)

    trees = PacketTreeSet()
    leaf_gen = LeafGenerator()

    f0_hz = 261.63
    harmonic_amps = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]
    leaf_content = leaf_gen.generate_from_fundamental(f0_hz, harmonic_amps)

    # Build nodes - EXACTLY as in debug_wavelet_phase.py
    nodes = []
    for leaf in range(LEAF_COUNT):
        node_coeffs = []
        for c in range(NODE_SIZE):
            if leaf < len(leaf_content.coeffs_q15) // NODE_SIZE:
                idx = leaf * NODE_SIZE + c
                if idx < len(leaf_content.coeffs_q15):
                    node_coeffs.append(leaf_content.coeffs_q15[idx])
                else:
                    node_coeffs.append(0)
            else:
                node_coeffs.append(0)
        nodes.append(node_coeffs)

    phase_state = []
    for leaf_idx in range(LEAF_COUNT):
        for coeff_idx in range(NODE_SIZE):
            phase = coeff_idx * (math.pi * 2 / NODE_SIZE) + leaf_idx * (math.pi / LEAF_COUNT)
            phase_state.append(phase)

    samples = []
    synth = trees.synthesizers[1]  # Tree 1 = Harmonic

    for _ in range(int(0.5 * BLOCKS_PER_SEC)):
        block = synth.synthesize(nodes, phase_state=phase_state)
        samples.extend(block)

    analyze(samples, "Original method with Tree 1")


if __name__ == "__main__":
    print("=" * 60)
    print("REPLICATING WORKING TEST")
    print("=" * 60)

    test_original()
    test_with_tree1()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
