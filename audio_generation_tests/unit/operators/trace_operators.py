"""
Trace Operator Application
==========================

Verify that operators actually modify the coefficients in the wet path.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np


def test_operator_application():
    print("=" * 70)
    print("OPERATOR APPLICATION TRACE")
    print("=" * 70)

    from symcrash_python_ap.operator_library import (
        build_operator_library, get_operator_id, apply_operator
    )
    from symcrash_python_ap.wavelet.harmonic_injector import create_harmonic_coeffs, WaveType

    Q15_ONE = 32767
    Q15_HALF = 16384

    # Build operator library
    op_library = build_operator_library()
    print(f"\nOperator library: {len(op_library)} operators")

    # Create test coefficients (like in render_voice)
    print("\n[1] GENERATING TEST COEFFICIENTS")
    print("-" * 40)

    coeffs_256 = create_harmonic_coeffs(
        frequency=440.0,
        amplitude=Q15_HALF,
        wave_type=WaveType.SAW,
        partials_n=8,
        brightness_q15=Q15_ONE
    )

    coeffs_np = np.array(coeffs_256, dtype=np.int16)
    print(f"Original coeffs shape: {coeffs_np.shape}")
    print(f"Original coeffs range: [{coeffs_np.min()}, {coeffs_np.max()}]")
    print(f"Original coeffs energy: {np.sum(coeffs_np.astype(np.float64)**2):.0f}")
    print(f"Original coeffs non-zero: {np.count_nonzero(coeffs_np)}")

    # Test each operator from brass_stab
    ops_enable = ["SHEAF_PROJECT", "TILT", "COMPRESS_K", "INTRA_MIX", "NORMALIZE_SCALE"]
    families = {"SHEAF_PROJECT": 0.40, "TILT": 0.45, "COMPRESS_K": 0.30, "INTRA_MIX": 0.25, "NORMALIZE_SCALE": 0.80}

    print("\n[2] APPLYING OPERATORS")
    print("-" * 40)

    for op_name in ops_enable:
        op_id = get_operator_id(op_name)
        if op_id not in op_library:
            print(f"  {op_name}: NOT FOUND (ID {op_id})")
            continue

        op = op_library[op_id]
        strength = families.get(op_name, 1.0)

        # Store original for comparison
        before = coeffs_np.copy()

        # Apply operator
        processed = apply_operator(coeffs_np, op)

        # Blend based on strength
        if strength < 1.0:
            coeffs_np = (
                (coeffs_np.astype(np.int32) * int((1.0 - strength) * 32768) +
                 processed.astype(np.int32) * int(strength * 32768)) >> 15
            ).astype(np.int16)
        else:
            coeffs_np = processed

        # Measure change
        diff = np.abs(coeffs_np.astype(np.int32) - before.astype(np.int32))
        max_diff = diff.max()
        mean_diff = diff.mean()
        changed_count = np.count_nonzero(diff)

        print(f"\n  {op_name} (ID {op_id}, strength {strength:.2f}):")
        print(f"    Type: {op.op_type.value}")
        print(f"    Description: {op.description}")
        print(f"    Max change: {max_diff}")
        print(f"    Mean change: {mean_diff:.1f}")
        print(f"    Coeffs changed: {changed_count}/256")

        if max_diff == 0:
            print(f"    WARNING: No change!")

    print("\n[3] FINAL RESULT")
    print("-" * 40)

    print(f"Final coeffs range: [{coeffs_np.min()}, {coeffs_np.max()}]")
    print(f"Final coeffs energy: {np.sum(coeffs_np.astype(np.float64)**2):.0f}")
    print(f"Final coeffs non-zero: {np.count_nonzero(coeffs_np)}")

    # Compare with original
    original = np.array(create_harmonic_coeffs(
        frequency=440.0,
        amplitude=Q15_HALF,
        wave_type=WaveType.SAW,
        partials_n=8,
        brightness_q15=Q15_ONE
    ), dtype=np.int16)

    total_diff = np.abs(coeffs_np.astype(np.int32) - original.astype(np.int32))
    print(f"\nTotal change from original:")
    print(f"  Max diff: {total_diff.max()}")
    print(f"  Mean diff: {total_diff.mean():.1f}")
    print(f"  Coeffs changed: {np.count_nonzero(total_diff)}/256")

    if total_diff.max() > 0:
        print("\nOPERATORS ARE MODIFYING COEFFICIENTS")
    else:
        print("\nWARNING: OPERATORS HAD NO EFFECT!")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_operator_application()
