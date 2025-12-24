"""Debug wavelet phase continuity."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "symcrash_python_ap"))

import numpy as np
from symcrash_python_ap.wavelet_packet_trees import PacketTreeSet, LEAF_COUNT, NODE_SIZE, BLOCK_SIZE
from symcrash_python_ap.carrier_lift import Q15_ONE, Q15_HALF

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

    # Check block boundaries
    for block in range(1, min(10, len(samples) // BLOCK_SIZE)):
        boundary = block * BLOCK_SIZE
        if boundary + 1 < len(samples):
            delta_at_boundary = abs(samples[boundary] - samples[boundary - 1])
            if delta_at_boundary > 500:
                print(f"    Block {block} boundary delta: {delta_at_boundary}")


def test_constant_single_coeff():
    """Test with single constant coefficient."""
    print("\n" + "=" * 60)
    print("TEST 1: Single constant coefficient (leaf 4, coeff 0)")
    print("=" * 60)

    trees = PacketTreeSet()

    # Build coefficients: only leaf 4, coeff 0 has energy
    coeffs = [0] * 256
    target_leaf = 4  # ~3-3.75 kHz
    target_coeff = 0
    coeff_idx = target_leaf * NODE_SIZE + target_coeff
    coeffs[coeff_idx] = Q15_HALF

    # Create nodes structure
    nodes = []
    for leaf in range(LEAF_COUNT):
        node_coeffs = []
        for c in range(NODE_SIZE):
            node_coeffs.append(coeffs[leaf * NODE_SIZE + c])
        nodes.append(node_coeffs)

    # Initialize phase state
    phase_state = []
    import math
    for leaf_idx in range(LEAF_COUNT):
        for coeff_idx in range(NODE_SIZE):
            phase = coeff_idx * (math.pi * 2 / NODE_SIZE) + leaf_idx * (math.pi / LEAF_COUNT)
            phase_state.append(phase)

    # Render multiple blocks
    samples = []
    synth = trees.synthesizers[0]

    for _ in range(int(0.5 * BLOCKS_PER_SEC)):
        block = synth.synthesize(nodes, phase_state=phase_state)
        samples.extend(block)

    analyze(samples, "Single constant coeff")


def test_fundamental_only():
    """Test with just fundamental (like a pure sine carrier should match)."""
    print("\n" + "=" * 60)
    print("TEST 2: Fundamental only (261.6 Hz / C4)")
    print("=" * 60)

    from symcrash_python_ap.leaf_generator import LeafGenerator

    trees = PacketTreeSet()
    leaf_gen = LeafGenerator()

    # Generate leaf content for C4
    f0_hz = 261.63
    harmonic_amps = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]  # Pure sine
    leaf_content = leaf_gen.generate_from_fundamental(f0_hz, harmonic_amps)

    # Build nodes
    nodes = []
    for leaf in range(LEAF_COUNT):
        node_coeffs = []
        for c in range(NODE_SIZE):
            if leaf < len(leaf_content.coeffs_q15) // NODE_SIZE:
                node_coeffs.append(leaf_content.coeffs_q15[leaf * NODE_SIZE + c] if leaf * NODE_SIZE + c < len(leaf_content.coeffs_q15) else 0)
            else:
                node_coeffs.append(0)
        nodes.append(node_coeffs)

    # Initialize phase state
    phase_state = []
    import math
    for leaf_idx in range(LEAF_COUNT):
        for coeff_idx in range(NODE_SIZE):
            phase = coeff_idx * (math.pi * 2 / NODE_SIZE) + leaf_idx * (math.pi / LEAF_COUNT)
            phase_state.append(phase)

    # Render
    samples = []
    synth = trees.synthesizers[0]

    for _ in range(int(0.5 * BLOCKS_PER_SEC)):
        block = synth.synthesize(nodes, phase_state=phase_state)
        samples.extend(block)

    analyze(samples, "Fundamental only (C4)")


def test_changing_coeffs():
    """Test with coefficients changing each block (simulating real usage)."""
    print("\n" + "=" * 60)
    print("TEST 3: Changing coefficients (simulates dynamic leaf gen)")
    print("=" * 60)

    from symcrash_python_ap.leaf_generator import LeafGenerator

    trees = PacketTreeSet()
    leaf_gen = LeafGenerator()

    f0_hz = 261.63
    harmonic_amps = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    # Initialize phase state
    phase_state = []
    import math
    for leaf_idx in range(LEAF_COUNT):
        for coeff_idx in range(NODE_SIZE):
            phase = coeff_idx * (math.pi * 2 / NODE_SIZE) + leaf_idx * (math.pi / LEAF_COUNT)
            phase_state.append(phase)

    synth = trees.synthesizers[0]
    samples = []

    for block_idx in range(int(0.5 * BLOCKS_PER_SEC)):
        # Generate fresh leaf content each block (as render_voice does)
        leaf_content = leaf_gen.generate_from_fundamental(f0_hz, harmonic_amps)

        # Build nodes
        nodes = []
        for leaf in range(LEAF_COUNT):
            node_coeffs = []
            for c in range(NODE_SIZE):
                idx = leaf * NODE_SIZE + c
                if idx < len(leaf_content.coeffs_q15):
                    node_coeffs.append(leaf_content.coeffs_q15[idx])
                else:
                    node_coeffs.append(0)
            nodes.append(node_coeffs)

        block = synth.synthesize(nodes, phase_state=phase_state)
        samples.extend(block)

    analyze(samples, "Changing coefficients (fresh each block)")


if __name__ == "__main__":
    print("=" * 60)
    print("WAVELET PHASE CONTINUITY DEBUG")
    print("=" * 60)

    test_constant_single_coeff()
    test_fundamental_only()
    test_changing_coeffs()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
