#!/usr/bin/env python3
"""
Discontinuity detection tool for VM pipeline.

Detects *unexpected* discontinuities by analyzing second-order differences
(changes in the rate of change). A pure sine wave has smooth first-order
differences, so large second-order differences indicate actual discontinuities.

The threshold is applied to second-order differences relative to peak amplitude.

Usage:
    pixi run python tools/check_discontinuity.py [--verbose] [--threshold DB]
"""
import argparse
import sys
sys.path.insert(0, 'src')

from dataclasses import dataclass, field
from typing import List
import numpy as np

from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES


BLOCK_SIZE = 64
SAMPLE_RATE = 48000


@dataclass
class DiscontinuityResult:
    """Results from a discontinuity test scenario."""
    test_name: str
    passed: bool
    peak_amplitude: int
    threshold_lsb: int
    max_boundary_accel: int  # Max second-order difference at boundaries
    max_within_block_accel: int  # Max second-order difference within blocks
    worst_boundary_block: int
    worst_within_block: int
    worst_within_sample: int
    num_boundary_violations: int
    num_within_violations: int
    details: List[str] = field(default_factory=list)


def compute_threshold_lsb(peak: int, threshold_db: float) -> int:
    """Convert dB threshold to LSB threshold for second-order differences."""
    return int(peak * (10 ** (threshold_db / 20)))


def compute_expected_accel(peak_amplitude: int, f0_hz: float, sample_rate: int = SAMPLE_RATE) -> float:
    """
    Compute expected maximum acceleration for a sine wave.

    For sin(ωt): d²/dt² = -ω² sin(ωt), so max |accel| = ω² × amplitude
    """
    omega = 2 * np.pi * f0_hz / sample_rate
    return omega * omega * peak_amplitude


def detect_discontinuities(
    samples: np.ndarray,
    threshold_lsb: int,
    expected_accel: float,
    block_size: int = BLOCK_SIZE,
) -> tuple[int, int, int, int, int, int, int, List[str]]:
    """
    Detect discontinuities using excess acceleration (beyond expected sine wave behavior).

    A smooth sine wave has a predictable maximum acceleration based on frequency.
    Discontinuities are detected when acceleration exceeds expected + threshold.

    Returns:
        max_boundary_excess: Maximum excess acceleration at block boundaries
        max_within_block_excess: Maximum excess acceleration within blocks
        worst_boundary_block: Block index with largest boundary excess
        worst_within_block: Block index with largest within-block excess
        worst_within_sample: Sample index within worst block
        num_boundary_violations: Count of boundary excesses exceeding threshold
        num_within_violations: Count of within-block excesses exceeding threshold
        details: List of violation descriptions
    """
    samples = samples.astype(np.int32)
    num_blocks = len(samples) // block_size

    # Compute first-order differences (velocity)
    velocity = np.diff(samples)

    # Compute second-order differences (acceleration)
    acceleration = np.diff(velocity)

    # Threshold includes expected acceleration + threshold margin
    # A discontinuity is when accel > expected + margin
    total_threshold = int(expected_accel + threshold_lsb)

    max_boundary_excess = 0
    max_within_block_excess = 0
    worst_boundary_block = 0
    worst_within_block = 0
    worst_within_sample = 0
    num_boundary_violations = 0
    num_within_violations = 0
    details = []

    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = start + block_size

        # Get acceleration values for this block's interior
        interior_start = start
        interior_end = min(end - 2, len(acceleration))

        if interior_start < interior_end:
            block_accel = np.abs(acceleration[interior_start:interior_end])
            if len(block_accel) > 0:
                # Compute excess above expected
                block_excess = block_accel - expected_accel
                max_excess = float(np.max(block_excess))

                if max_excess > max_within_block_excess:
                    max_within_block_excess = max_excess
                    worst_within_block = block_idx
                    worst_within_sample = int(np.argmax(block_excess)) + 1

                # Count violations (excess > threshold margin)
                violations = np.where(block_excess > threshold_lsb)[0]
                if len(violations) > 0:
                    num_within_violations += len(violations)
                    for v_idx in violations[:3]:  # Report first 3 per block
                        if len(details) < 20:  # Global limit
                            details.append(
                                f"Block {block_idx}: excess accel={block_excess[v_idx]:.0f} LSB "
                                f"at sample {v_idx + 1} (expected={expected_accel:.0f}, threshold={threshold_lsb})"
                            )

        # Check block boundary
        if block_idx > 0:
            boundary_accel_idx = start - 1
            if 0 <= boundary_accel_idx < len(acceleration):
                boundary_accel = abs(int(acceleration[boundary_accel_idx]))
                boundary_excess = boundary_accel - expected_accel

                if boundary_excess > max_boundary_excess:
                    max_boundary_excess = boundary_excess
                    worst_boundary_block = block_idx

                if boundary_excess > threshold_lsb:
                    num_boundary_violations += 1
                    if num_boundary_violations <= 5:  # Report first 5
                        details.append(
                            f"Block {block_idx-1}->{block_idx}: excess accel={boundary_excess:.0f} LSB "
                            f"(expected={expected_accel:.0f}, threshold={threshold_lsb})"
                        )

    return (
        int(max_boundary_excess),
        int(max_within_block_excess),
        worst_boundary_block,
        worst_within_block,
        worst_within_sample,
        num_boundary_violations,
        num_within_violations,
        details,
    )


def run_scenario(
    name: str,
    num_blocks: int,
    eta_start: int,
    eta_end: int,
    gain_start: int,
    gain_end: int,
    f0_start: float,
    f0_end: float,
    threshold_db: float = -60.0,
    warmup_blocks: int = 0,
    verbose: bool = False,
) -> DiscontinuityResult:
    """
    Run a test scenario and check for discontinuities.

    Parameters:
        name: Test scenario name
        num_blocks: Number of blocks to analyze
        eta_start/end: Start and end beta_target_q15 values (for ramping)
        gain_start/end: Start and end gain_q15 values
        f0_start/end: Start and end frequency values
        threshold_db: Discontinuity threshold in dB (default -60)
        warmup_blocks: Number of warmup blocks before analysis
        verbose: Print detailed block-by-block info
    """
    state = VMState.create(0)
    state.reset_for_note_on(f0_hz=f0_start)

    all_samples = []

    # Warmup phase (if any)
    if warmup_blocks > 0:
        warmup_frame = VoiceFrame(
            voice_id=0,
            note=69,
            f0_hz=f0_start,
            gain_q15=gain_start,
            beta_target_q15=eta_start,
            partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
            is_active=True,
        )
        for _ in range(warmup_blocks):
            state, _ = vm_step_block(state, warmup_frame, DEFAULT_TABLES)

    # Analysis phase
    for block_idx in range(num_blocks):
        # Interpolate parameters for ramps
        t = block_idx / max(1, num_blocks - 1)
        eta = int(eta_start + t * (eta_end - eta_start))
        gain = int(gain_start + t * (gain_end - gain_start))
        f0 = f0_start + t * (f0_end - f0_start)

        frame = VoiceFrame(
            voice_id=0,
            note=69,
            f0_hz=f0,
            gain_q15=gain,
            beta_target_q15=eta,
            partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
            is_active=True,
        )

        state, output = vm_step_block(state, frame, DEFAULT_TABLES)
        all_samples.extend(output.tolist())

    samples = np.array(all_samples, dtype=np.int16)
    peak = int(np.max(np.abs(samples))) if len(samples) > 0 else 1
    threshold_lsb = compute_threshold_lsb(peak, threshold_db)

    # Ensure minimum threshold of 1 LSB
    threshold_lsb = max(1, threshold_lsb)

    # Compute expected acceleration for the highest frequency in the test
    # For pitch glides, use the max frequency; otherwise use start/end average
    max_f0 = max(f0_start, f0_end)
    expected_accel = compute_expected_accel(peak, max_f0)

    (
        max_boundary_accel,
        max_within_block_accel,
        worst_boundary_block,
        worst_within_block,
        worst_within_sample,
        num_boundary_violations,
        num_within_violations,
        details,
    ) = detect_discontinuities(samples, threshold_lsb, expected_accel)

    passed = (num_boundary_violations == 0) and (num_within_violations == 0)

    return DiscontinuityResult(
        test_name=name,
        passed=passed,
        peak_amplitude=peak,
        threshold_lsb=threshold_lsb,
        max_boundary_accel=max_boundary_accel,
        max_within_block_accel=max_within_block_accel,
        worst_boundary_block=worst_boundary_block,
        worst_within_block=worst_within_block,
        worst_within_sample=worst_within_sample,
        num_boundary_violations=num_boundary_violations,
        num_within_violations=num_within_violations,
        details=details,
    )


def run_note_retrigger_scenario(
    threshold_db: float = -60.0,
    verbose: bool = False,
) -> DiscontinuityResult:
    """
    Special scenario: note-off then note-on retrigger.

    Tests for discontinuities when a note is retriggered mid-stream.
    """
    state = VMState.create(0)
    state.reset_for_note_on(f0_hz=440.0)

    all_samples = []

    frame = VoiceFrame(
        voice_id=0,
        note=69,
        f0_hz=440.0,
        gain_q15=32767,
        beta_target_q15=16384,
        partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
        is_active=True,
    )

    # First note: 50 blocks
    for _ in range(50):
        state, output = vm_step_block(state, frame, DEFAULT_TABLES)
        all_samples.extend(output.tolist())

    # Retrigger note
    state.reset_for_note_on(f0_hz=440.0)

    # Second note: 50 blocks
    for _ in range(50):
        state, output = vm_step_block(state, frame, DEFAULT_TABLES)
        all_samples.extend(output.tolist())

    samples = np.array(all_samples, dtype=np.int16)
    peak = int(np.max(np.abs(samples))) if len(samples) > 0 else 1
    threshold_lsb = compute_threshold_lsb(peak, threshold_db)
    threshold_lsb = max(1, threshold_lsb)

    # Expected accel for 440Hz
    expected_accel = compute_expected_accel(peak, 440.0)

    (
        max_boundary_accel,
        max_within_block_accel,
        worst_boundary_block,
        worst_within_block,
        worst_within_sample,
        num_boundary_violations,
        num_within_violations,
        details,
    ) = detect_discontinuities(samples, threshold_lsb, expected_accel)

    passed = (num_boundary_violations == 0) and (num_within_violations == 0)

    return DiscontinuityResult(
        test_name="Note Retrigger",
        passed=passed,
        peak_amplitude=peak,
        threshold_lsb=threshold_lsb,
        max_boundary_accel=max_boundary_accel,
        max_within_block_accel=max_within_block_accel,
        worst_boundary_block=worst_boundary_block,
        worst_within_block=worst_within_block,
        worst_within_sample=worst_within_sample,
        num_boundary_violations=num_boundary_violations,
        num_within_violations=num_within_violations,
        details=details,
    )


def print_result(result: DiscontinuityResult, verbose: bool = False):
    """Print a single test result."""
    status = "PASS" if result.passed else "FAIL"

    print(f"\nTest: {result.test_name}")
    print(f"  Peak amplitude: {result.peak_amplitude}")
    print(f"  Threshold (-60 dB): {result.threshold_lsb} LSB")
    print(f"  Max boundary accel: {result.max_boundary_accel} LSB (block {result.worst_boundary_block})")
    print(f"  Max within-block accel: {result.max_within_block_accel} LSB "
          f"(block {result.worst_within_block}, sample {result.worst_within_sample})")

    if result.num_boundary_violations > 0 or result.num_within_violations > 0:
        print(f"  Violations: {result.num_boundary_violations} boundary, "
              f"{result.num_within_violations} within-block")

    print(f"  Result: {status}")

    if verbose and result.details:
        print(f"  Details:")
        for detail in result.details[:10]:  # Limit to 10 details
            print(f"    {detail}")
        if len(result.details) > 10:
            print(f"    ... and {len(result.details) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Discontinuity detection tool for VM pipeline"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed violation information"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=-60.0,
        help="Discontinuity threshold in dB (default: -60)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Discontinuity Detection Tool")
    print(f"Threshold: {args.threshold} dB")
    print("=" * 60)

    results = []

    # Test 1: Pure carrier (eta=0) - steady state
    results.append(run_scenario(
        name="Pure Carrier (eta=0)",
        num_blocks=500,
        eta_start=0, eta_end=0,
        gain_start=32767, gain_end=32767,
        f0_start=440.0, f0_end=440.0,
        threshold_db=args.threshold,
        warmup_blocks=10,
        verbose=args.verbose,
    ))

    # Test 2: Pure wet (eta=32767) - steady state
    results.append(run_scenario(
        name="Pure Wet (eta=32767)",
        num_blocks=500,
        eta_start=32767, eta_end=32767,
        gain_start=32767, gain_end=32767,
        f0_start=440.0, f0_end=440.0,
        threshold_db=args.threshold,
        warmup_blocks=10,
        verbose=args.verbose,
    ))

    # Test 3: 50/50 mix (eta=16384) - steady state
    results.append(run_scenario(
        name="50/50 Mix (eta=16384)",
        num_blocks=500,
        eta_start=16384, eta_end=16384,
        gain_start=32767, gain_end=32767,
        f0_start=440.0, f0_end=440.0,
        threshold_db=args.threshold,
        warmup_blocks=10,
        verbose=args.verbose,
    ))

    # Test 4: Eta ramp (0->32767) - dry to wet transition
    results.append(run_scenario(
        name="Eta Ramp (0->32767)",
        num_blocks=100,
        eta_start=0, eta_end=32767,
        gain_start=32767, gain_end=32767,
        f0_start=440.0, f0_end=440.0,
        threshold_db=args.threshold,
        warmup_blocks=10,
        verbose=args.verbose,
    ))

    # Test 5: Gain ramp (0->32767) - fade in
    results.append(run_scenario(
        name="Gain Fade In (0->32767)",
        num_blocks=100,
        eta_start=16384, eta_end=16384,
        gain_start=0, gain_end=32767,
        f0_start=440.0, f0_end=440.0,
        threshold_db=args.threshold,
        warmup_blocks=10,
        verbose=args.verbose,
    ))

    # Test 6: Gain ramp (32767->0) - fade out
    results.append(run_scenario(
        name="Gain Fade Out (32767->0)",
        num_blocks=100,
        eta_start=16384, eta_end=16384,
        gain_start=32767, gain_end=0,
        f0_start=440.0, f0_end=440.0,
        threshold_db=args.threshold,
        warmup_blocks=10,
        verbose=args.verbose,
    ))

    # Test 7: Pitch glide (220->880) - octave sweep
    results.append(run_scenario(
        name="Pitch Glide (220->880 Hz)",
        num_blocks=300,
        eta_start=0, eta_end=0,
        gain_start=32767, gain_end=32767,
        f0_start=220.0, f0_end=880.0,
        threshold_db=args.threshold,
        warmup_blocks=10,
        verbose=args.verbose,
    ))

    # Test 8: Note retrigger
    results.append(run_note_retrigger_scenario(
        threshold_db=args.threshold,
        verbose=args.verbose,
    ))

    # Print all results
    for result in results:
        print_result(result, verbose=args.verbose)

    # Summary
    passed = sum(1 for r in results if r.passed)
    print("\n" + "=" * 60)
    print(f"Summary: {passed}/{len(results)} tests passed")

    if passed < len(results):
        print("\nFailed tests:")
        for r in results:
            if not r.passed:
                print(f"  - {r.test_name}")
        sys.exit(1)
    else:
        print("\nAll discontinuity checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
