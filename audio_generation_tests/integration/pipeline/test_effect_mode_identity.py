"""Test effect mode invariant: FWPT -> identity mask -> IDWPT = original."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "symcrash_python_ap"))

import numpy as np
from symcrash_python_ap.wavelet.wavelet_packet_trees import PacketTreeSet
from symcrash_python_ap.wavelet.carrier_lift import Q15_ONE, BLOCK_SIZE


def test_identity_mask_roundtrip():
    """Test that effect mode with identity mask (None) returns original audio."""
    print("=" * 60)
    print("Testing effect mode identity: FWPT -> IDWPT = original")
    print("=" * 60)

    tree_set = PacketTreeSet()

    # Test with various input signals
    test_signals = {
        "pure_sine": [int(Q15_ONE * 0.5 * np.sin(2 * np.pi * 5 * i / BLOCK_SIZE))
                      for i in range(BLOCK_SIZE)],
        "square": [Q15_ONE // 2 if i < BLOCK_SIZE // 2 else -Q15_ONE // 2
                   for i in range(BLOCK_SIZE)],
        "ramp": [int(-Q15_ONE // 2 + (Q15_ONE * i) // BLOCK_SIZE)
                 for i in range(BLOCK_SIZE)],
        "impulse": [Q15_ONE // 2 if i == BLOCK_SIZE // 4 else 0
                    for i in range(BLOCK_SIZE)],
    }

    all_passed = True

    for name, signal in test_signals.items():
        print(f"\n{name}:")
        print(f"  Input energy: {sum(abs(s) for s in signal)}")

        # Effect mode with identity mask (None) should return original
        output = tree_set.effect_mode(
            tree_id=0,
            carrier_audio=signal,
            mask_q15=None  # Identity mask
        )

        print(f"  Output energy: {sum(abs(s) for s in output)}")

        # Compare input vs output
        max_diff = max(abs(a - b) for a, b in zip(signal, output))
        rms_diff = np.sqrt(np.mean([(a - b) ** 2 for a, b in zip(signal, output)]))

        # Calculate correlation
        signal_arr = np.array(signal, dtype=float)
        output_arr = np.array(output, dtype=float)
        if np.std(signal_arr) > 0 and np.std(output_arr) > 0:
            correlation = np.corrcoef(signal_arr, output_arr)[0, 1]
        else:
            correlation = 1.0 if np.allclose(signal_arr, output_arr) else 0.0

        print(f"  Max diff: {max_diff}")
        print(f"  RMS diff: {rms_diff:.1f}")
        print(f"  Correlation: {correlation:.6f}")

        # Check if result is acceptable
        # Note: CDF 5/3 may have small numerical differences but should be close
        if max_diff > 100 or rms_diff > 50 or correlation < 0.99:
            print(f"  FAIL: Roundtrip error too high!")
            all_passed = False
        else:
            print(f"  PASS")

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
    print("=" * 60)

    return all_passed


def test_mask_shapes_spectrum():
    """Test that non-identity mask modifies the signal."""
    print("\n" + "=" * 60)
    print("Testing that mask modifies signal")
    print("=" * 60)

    tree_set = PacketTreeSet()

    # Input: wide-band signal
    signal = [int(Q15_ONE * 0.5 * np.sin(2 * np.pi * 5 * i / BLOCK_SIZE))
              for i in range(BLOCK_SIZE)]

    # Create a mask that attenuates some leaves
    # 64 leaves * 4 coeffs = 256 mask values
    from symcrash_python_ap.wavelet_trees import NODE_SIZE, LEAF_COUNT

    mask_len = LEAF_COUNT * NODE_SIZE

    # Low-pass mask: leaves 0-31 = full, leaves 32-63 = zero
    lowpass_mask = [Q15_ONE if (i // NODE_SIZE) < LEAF_COUNT // 2 else 0
                    for i in range(mask_len)]

    output_identity = tree_set.effect_mode(tree_id=0, carrier_audio=signal, mask_q15=None)
    output_lowpass = tree_set.effect_mode(tree_id=0, carrier_audio=signal, mask_q15=lowpass_mask)

    identity_energy = sum(abs(s) for s in output_identity)
    lowpass_energy = sum(abs(s) for s in output_lowpass)

    print(f"Identity mask energy: {identity_energy}")
    print(f"Low-pass mask energy: {lowpass_energy}")
    print(f"Difference: {abs(identity_energy - lowpass_energy)}")

    # Lowpass should be different (attenuated high frequencies)
    if abs(identity_energy - lowpass_energy) < 100:
        print("WARNING: Mask doesn't seem to affect output much")
    else:
        print("Mask successfully modifies spectrum")


if __name__ == "__main__":
    test_identity_mask_roundtrip()
    test_mask_shapes_spectrum()
