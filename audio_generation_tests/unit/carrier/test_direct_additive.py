"""Test direct additive synthesis vs wavelet-based synthesis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "symcrash_python_ap"))

import numpy as np
from symcrash_python_ap.wavelet.harmonic_injector import (
    synthesize_harmonic_audio, create_harmonic_coeffs, WaveType
)
from symcrash_python_ap.wavelet.wavelet_packet_trees import PacketTreeSet, LEAF_COUNT, NODE_SIZE, BLOCK_SIZE
from symcrash_python_ap.wavelet.carrier_lift import Q15_ONE, Q15_HALF

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


def test_direct_additive():
    """Test direct additive synthesis (Baby Fourier)."""
    print("\n" + "=" * 60)
    print("TEST: Direct additive synthesis (synthesize_harmonic_audio)")
    print("=" * 60)

    samples = []
    phase_state = None

    for _ in range(int(0.5 * BLOCKS_PER_SEC)):
        block, phase_state = synthesize_harmonic_audio(
            frequency=261.63,  # C4
            amplitude=Q15_HALF,
            wave_type=WaveType.SINE,
            partials_n=1,  # Pure sine
            phase_state=phase_state
        )
        samples.extend(block)

    analyze(samples, "Direct additive (pure sine)")


def test_direct_additive_saw():
    """Test direct additive synthesis with saw wave."""
    print("\n" + "=" * 60)
    print("TEST: Direct additive synthesis (saw wave)")
    print("=" * 60)

    samples = []
    phase_state = None

    for _ in range(int(0.5 * BLOCKS_PER_SEC)):
        block, phase_state = synthesize_harmonic_audio(
            frequency=261.63,
            amplitude=Q15_HALF,
            wave_type=WaveType.SAW,
            partials_n=8,
            phase_state=phase_state
        )
        samples.extend(block)

    analyze(samples, "Direct additive (saw)")


def test_wavelet_with_harmonic_coeffs():
    """Test wavelet synthesis with harmonic_injector coefficients."""
    print("\n" + "=" * 60)
    print("TEST: Wavelet synthesis with harmonic_injector coeffs")
    print("=" * 60)

    trees = PacketTreeSet()

    # Generate coefficients using harmonic_injector
    coeffs_256 = create_harmonic_coeffs(
        frequency=261.63,
        amplitude=Q15_HALF,
        wave_type=WaveType.SINE,
        partials_n=1
    )

    print(f"  Non-zero coefficients:")
    for i, c in enumerate(coeffs_256):
        if c != 0:
            leaf_idx = i // NODE_SIZE
            coeff_idx = i % NODE_SIZE
            print(f"    coeffs[{i}] (leaf {leaf_idx}, coeff {coeff_idx}) = {c}")

    # Initialize phase state
    import math
    phase_state = []
    for leaf_idx in range(LEAF_COUNT):
        for coeff_idx in range(NODE_SIZE):
            phase = coeff_idx * (math.pi * 2 / NODE_SIZE) + leaf_idx * (math.pi / LEAF_COUNT)
            phase_state.append(phase)

    # Unpack and render
    nodes = trees.unpack_coeffs(1, coeffs_256)
    synth = trees.synthesizers[1]

    samples = []
    for _ in range(int(0.5 * BLOCKS_PER_SEC)):
        block = synth.synthesize(nodes, phase_state=phase_state)
        samples.extend(block)

    analyze(samples, "Wavelet with harmonic coeffs")


if __name__ == "__main__":
    print("=" * 60)
    print("DIRECT ADDITIVE VS WAVELET SYNTHESIS")
    print("=" * 60)

    test_direct_additive()
    test_direct_additive_saw()
    test_wavelet_with_harmonic_coeffs()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
