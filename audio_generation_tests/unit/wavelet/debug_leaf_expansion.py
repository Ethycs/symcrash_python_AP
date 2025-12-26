"""Debug different leaf-to-coefficient expansion strategies."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "symcrash_python_ap"))

import numpy as np
import math
from symcrash_python_ap.wavelet.wavelet_packet_trees import PacketTreeSet, LEAF_COUNT, NODE_SIZE, BLOCK_SIZE
from symcrash_python_ap.wavelet.leaf_generator import LeafGenerator
from symcrash_python_ap.wavelet.carrier_lift import Q15_ONE, Q15_HALF
from symcrash_python_ap.fixed import sat16

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


def expand_simple_repeat(leaf_gains: list) -> list:
    """Simple repetition: each leaf repeated NODE_SIZE times."""
    coeffs_256 = []
    for i in range(256):
        leaf_idx = i // NODE_SIZE
        coeffs_256.append(leaf_gains[leaf_idx])
    return coeffs_256


def expand_with_envelope(leaf_gains: list) -> list:
    """Apply Hann envelope within each leaf (like IDWPTLift)."""
    coeffs = []
    for leaf_idx, gain in enumerate(leaf_gains[:LEAF_COUNT]):
        for coeff_idx in range(NODE_SIZE):
            t = (coeff_idx + 0.5) / NODE_SIZE
            envelope = math.sin(t * math.pi)
            coeff = int(gain * envelope)
            coeffs.append(sat16(coeff))
    return coeffs


def expand_flat(leaf_gains: list) -> list:
    """Flat: leaf gain fills all coefficients in node."""
    coeffs = []
    for leaf_idx in range(LEAF_COUNT):
        gain = leaf_gains[leaf_idx] if leaf_idx < len(leaf_gains) else 0
        for _ in range(NODE_SIZE):
            coeffs.append(gain)
    return coeffs


def test_expansion(name, expand_func):
    """Test a specific expansion strategy."""
    print("\n" + "=" * 60)
    print(f"TEST: {name}")
    print("=" * 60)

    trees = PacketTreeSet()
    leaf_gen = LeafGenerator()

    # Generate leaf content for C4 (pure sine)
    f0_hz = 261.63
    harmonic_amps = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]
    leaf_content = leaf_gen.generate_from_fundamental(f0_hz, harmonic_amps)

    # Expand to 256 coefficients
    coeffs_256 = expand_func(leaf_content.coeffs_q15)

    # Initialize phase state
    phase_state = []
    for leaf_idx in range(LEAF_COUNT):
        for coeff_idx in range(NODE_SIZE):
            phase = coeff_idx * (math.pi * 2 / NODE_SIZE) + leaf_idx * (math.pi / LEAF_COUNT)
            phase_state.append(phase)

    # Unpack into nodes
    nodes = trees.unpack_coeffs(1, coeffs_256)

    # Show non-zero nodes
    print(f"  Non-zero nodes:")
    for i, node in enumerate(nodes):
        if any(c != 0 for c in node):
            print(f"    node[{i}] = {node}")

    # Render
    samples = []
    synth = trees.synthesizers[1]  # Harmonic tree

    for _ in range(int(0.5 * BLOCKS_PER_SEC)):
        block = synth.synthesize(nodes, phase_state=phase_state)
        samples.extend(block)

    analyze(samples, name)


if __name__ == "__main__":
    print("=" * 60)
    print("LEAF-TO-COEFF EXPANSION STRATEGIES")
    print("=" * 60)

    test_expansion("Simple Repeat", expand_simple_repeat)
    test_expansion("Hann Envelope", expand_with_envelope)
    test_expansion("Flat Fill", expand_flat)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
