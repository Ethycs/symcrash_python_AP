"""
Diagnostic: Where do discontinuities actually occur?
====================================================

This script analyzes the actual location of discontinuities in wavelet synthesis
to verify whether they're at block boundaries or elsewhere.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "symcrash_python_ap"))

from test_config import FS, BLOCK_SIZE, BLOCKS_PER_SEC
from symcrash_python_ap.wavelet.wavelet_engine import WaveletEngine
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.wavelet.wavelet_packet_trees import COEFF_LEN
from symcrash_python_ap.fixed import sat16

Q15_ONE = 32767
Q15_HALF = 16384
NODE_SIZE = 8
N_LEAVES = 32


def generate_simple_coeffs(freq: float, amplitude: int) -> list:
    """Generate wavelet coefficients for a single frequency."""
    coeffs = [0] * COEFF_LEN
    nyquist = FS / 2

    def freq_to_leaf(f: float) -> int:
        if f < 20:
            return 0
        if f > 20000:
            return N_LEAVES - 1
        log_pos = np.log2(f / 20) / np.log2(20000 / 20)
        return int(log_pos * (N_LEAVES - 1))

    leaf_idx = freq_to_leaf(freq)
    node_start = leaf_idx * NODE_SIZE

    # Put energy in first coefficient only (simplest case)
    coeffs[node_start] = amplitude

    return coeffs


def find_discontinuities(samples: list, threshold: int = 2000) -> list:
    """
    Find all discontinuities and their positions relative to block boundaries.

    Returns list of (sample_index, delta, position_in_block, block_number)
    """
    discontinuities = []

    for i in range(1, len(samples)):
        delta = abs(samples[i] - samples[i - 1])
        if delta > threshold:
            block_num = i // BLOCK_SIZE
            pos_in_block = i % BLOCK_SIZE
            discontinuities.append((i, delta, pos_in_block, block_num))

    return discontinuities


def run_diagnostic():
    print("=" * 70)
    print("DISCONTINUITY LOCATION DIAGNOSTIC")
    print("=" * 70)
    print()

    # Test 1: Constant coefficients across multiple blocks
    print("Test 1: Constant coefficients (same frequency for 10 blocks)")
    print("-" * 70)

    wavelet = WaveletEngine(block_size=BLOCK_SIZE)
    wavelet.voice_allocate(0)

    coeffs = generate_simple_coeffs(440.0, 8000)

    all_samples = []
    for block in range(10):
        vf = VoiceFrame(
            voice_id=0,
            tree_a_id=1,
            tree_b_id=1,
            tree_blend_q15=0,
            gain_q15=Q15_HALF,
            pan_q15=0,
            coeff_q15=coeffs,
        )
        audio = wavelet.render_voice(vf)
        all_samples.extend(audio)

    discontinuities = find_discontinuities(all_samples)

    print(f"  Total samples: {len(all_samples)}")
    print(f"  Discontinuities (>2000): {len(discontinuities)}")

    if discontinuities:
        # Analyze position distribution
        boundary_count = 0
        mid_block_count = 0

        print("\n  First 10 discontinuities:")
        for i, (idx, delta, pos_in_block, block_num) in enumerate(discontinuities[:10]):
            is_boundary = pos_in_block == 0
            marker = "<<< BOUNDARY" if is_boundary else ""
            print(f"    [{i}] sample={idx}, delta={delta}, block={block_num}, pos_in_block={pos_in_block} {marker}")
            if is_boundary:
                boundary_count += 1
            else:
                mid_block_count += 1

        print(f"\n  Summary:")
        print(f"    At block boundaries (pos=0): {sum(1 for d in discontinuities if d[2] == 0)}")
        print(f"    Mid-block: {sum(1 for d in discontinuities if d[2] != 0)}")

        # Distribution histogram
        positions = [d[2] for d in discontinuities]
        print(f"\n  Position distribution within block:")
        for bucket_start in range(0, BLOCK_SIZE, 8):
            bucket_end = bucket_start + 8
            count = sum(1 for p in positions if bucket_start <= p < bucket_end)
            bar = "#" * (count * 2)
            print(f"    [{bucket_start:2d}-{bucket_end:2d}): {count:3d} {bar}")
    else:
        print("  No discontinuities found!")

    # Test 2: Alternating coefficients (frequency changes)
    print("\n")
    print("Test 2: Alternating coefficients (freq change every block)")
    print("-" * 70)

    wavelet2 = WaveletEngine(block_size=BLOCK_SIZE)
    wavelet2.voice_allocate(0)

    coeffs_a = generate_simple_coeffs(220.0, 8000)
    coeffs_b = generate_simple_coeffs(440.0, 8000)

    all_samples2 = []
    for block in range(10):
        coeffs = coeffs_a if block % 2 == 0 else coeffs_b
        vf = VoiceFrame(
            voice_id=0,
            tree_a_id=1,
            tree_b_id=1,
            tree_blend_q15=0,
            gain_q15=Q15_HALF,
            pan_q15=0,
            coeff_q15=coeffs,
        )
        audio = wavelet2.render_voice(vf)
        all_samples2.extend(audio)

    discontinuities2 = find_discontinuities(all_samples2)

    print(f"  Total samples: {len(all_samples2)}")
    print(f"  Discontinuities (>2000): {len(discontinuities2)}")

    if discontinuities2:
        print("\n  First 10 discontinuities:")
        for i, (idx, delta, pos_in_block, block_num) in enumerate(discontinuities2[:10]):
            is_boundary = pos_in_block == 0
            marker = "<<< BOUNDARY" if is_boundary else ""
            print(f"    [{i}] sample={idx}, delta={delta}, block={block_num}, pos_in_block={pos_in_block} {marker}")

        print(f"\n  Summary:")
        print(f"    At block boundaries (pos=0): {sum(1 for d in discontinuities2 if d[2] == 0)}")
        print(f"    Mid-block: {sum(1 for d in discontinuities2 if d[2] != 0)}")

    # Test 3: Analyze the actual waveform around boundaries
    print("\n")
    print("Test 3: Waveform values around block boundaries")
    print("-" * 70)

    for boundary_block in [1, 2, 3]:
        boundary_idx = boundary_block * BLOCK_SIZE

        print(f"\n  Block {boundary_block-1} -> Block {boundary_block} boundary (sample {boundary_idx}):")

        # Show 4 samples before and after
        for offset in range(-4, 5):
            idx = boundary_idx + offset
            if 0 <= idx < len(all_samples):
                val = all_samples[idx]
                marker = " <<< BOUNDARY" if offset == 0 else ""
                prev_val = all_samples[idx - 1] if idx > 0 else 0
                delta = val - prev_val
                print(f"    sample[{idx}] = {val:6d}  (delta = {delta:+6d}){marker}")

    # Test 4: Check if it's phase discontinuity
    print("\n")
    print("Test 4: Phase continuity check")
    print("-" * 70)

    # Generate expected continuous sine at 440Hz
    freq_hz = 440.0
    omega = 2 * np.pi * freq_hz / FS

    # Find actual signal in the 440Hz leaf
    leaf_idx = 10  # Approximate leaf for 440Hz

    print(f"  For {freq_hz}Hz, expected angular frequency omega = {omega:.6f} rad/sample")
    print(f"  Expected sample-to-sample phase advance: {omega:.6f} rad")
    print(f"  This means samples should follow: sin(n * {omega:.4f})")

    # Check correlation with expected sine
    n_samples = 128
    expected = [np.sin(omega * i) * 8000 for i in range(n_samples)]
    actual = all_samples[:n_samples]

    # Phase check at boundaries
    for boundary_block in [1, 2]:
        boundary_idx = boundary_block * BLOCK_SIZE

        # What phase do we expect?
        expected_phase = (omega * boundary_idx) % (2 * np.pi)

        # What does the actual sample suggest?
        if abs(all_samples[boundary_idx]) < 8000:
            actual_phase = np.arcsin(all_samples[boundary_idx] / 8000)
        else:
            actual_phase = np.pi / 2

        print(f"\n  At boundary {boundary_block} (sample {boundary_idx}):")
        print(f"    Expected phase: {expected_phase:.4f} rad ({np.degrees(expected_phase):.1f}°)")
        print(f"    Estimated actual phase: {actual_phase:.4f} rad ({np.degrees(actual_phase):.1f}°)")


def verify_continuity():
    """
    Verify that wavelet synthesis produces truly continuous output
    by checking if each block boundary is smoother than mid-block average.
    """
    print("\n")
    print("=" * 70)
    print("CONTINUITY VERIFICATION")
    print("=" * 70)

    wavelet = WaveletEngine(block_size=BLOCK_SIZE)
    wavelet.voice_allocate(0)

    coeffs = generate_simple_coeffs(440.0, 8000)

    all_samples = []
    for block in range(100):  # More blocks for statistics
        vf = VoiceFrame(
            voice_id=0,
            tree_a_id=1,
            tree_b_id=1,
            tree_blend_q15=0,
            gain_q15=Q15_HALF,
            pan_q15=0,
            coeff_q15=coeffs,
        )
        audio = wavelet.render_voice(vf)
        all_samples.extend(audio)

    # Calculate delta at EVERY sample
    deltas = [abs(all_samples[i] - all_samples[i-1]) for i in range(1, len(all_samples))]

    # Split into "at boundary" vs "not at boundary"
    boundary_deltas = []
    non_boundary_deltas = []

    for i, delta in enumerate(deltas):
        sample_idx = i + 1  # deltas[0] corresponds to samples[1] - samples[0]
        if sample_idx % BLOCK_SIZE == 0:
            boundary_deltas.append(delta)
        else:
            non_boundary_deltas.append(delta)

    avg_boundary = np.mean(boundary_deltas)
    avg_non_boundary = np.mean(non_boundary_deltas)
    max_boundary = max(boundary_deltas)
    max_non_boundary = max(non_boundary_deltas)

    print(f"\n  Statistics over {len(all_samples)} samples ({len(boundary_deltas)} boundaries):")
    print(f"\n  AVERAGE delta:")
    print(f"    At boundaries:     {avg_boundary:.1f}")
    print(f"    Not at boundaries: {avg_non_boundary:.1f}")
    print(f"    Ratio (boundary/non-boundary): {avg_boundary/avg_non_boundary:.2f}x")

    print(f"\n  MAXIMUM delta:")
    print(f"    At boundaries:     {max_boundary}")
    print(f"    Not at boundaries: {max_non_boundary}")

    # If there were true discontinuities at boundaries, we'd expect
    # boundary deltas to be significantly LARGER than non-boundary deltas
    if avg_boundary <= avg_non_boundary * 1.1:
        print(f"\n  CONCLUSION: No discontinuity at block boundaries!")
        print(f"             Boundary deltas are {'smaller' if avg_boundary < avg_non_boundary else 'similar'} to mid-block deltas.")
    else:
        print(f"\n  WARNING: Boundaries may have discontinuities!")
        print(f"           Boundary deltas are {avg_boundary/avg_non_boundary:.1f}x larger than mid-block.")


if __name__ == "__main__":
    run_diagnostic()
    verify_continuity()
