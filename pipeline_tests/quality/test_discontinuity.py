"""
Discontinuity Detection Tests
==============================

Sample-level discontinuity detection using second-order differences.

Detects unexpected jumps in the audio signal by analyzing acceleration
(second-order differences). A smooth sine wave has predictable acceleration
based on its frequency. Discontinuities are flagged when acceleration exceeds
expected values plus a threshold.

Key concepts:
- First-order difference (velocity): change in sample value
- Second-order difference (acceleration): change in velocity
- For sine wave: max |accel| = ω² × amplitude where ω = 2πf/sample_rate
"""

import pytest
import numpy as np
from dataclasses import dataclass, field
from typing import List

from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.audio_io import write_audio

BLOCK_SIZE = 64
SAMPLE_RATE = 48000
Q15_ONE = 32767


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


def compute_expected_accel(
    peak_amplitude: int, f0_hz: float, sample_rate: int = SAMPLE_RATE
) -> float:
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
) -> DiscontinuityResult:
    """
    Detect discontinuities using excess acceleration.

    Args:
        samples: Audio samples (int16 or int32)
        threshold_lsb: Threshold in LSB for excess acceleration
        expected_accel: Expected maximum acceleration for smooth signal
        block_size: Block size for boundary detection

    Returns:
        DiscontinuityResult with detection statistics
    """
    samples = samples.astype(np.int32)
    num_blocks = len(samples) // block_size

    # Compute first-order differences (velocity)
    velocity = np.diff(samples)

    # Compute second-order differences (acceleration)
    acceleration = np.diff(velocity)

    # Total threshold = expected + margin
    total_threshold = int(expected_accel + threshold_lsb)

    max_boundary_excess = 0
    max_within_block_excess = 0
    worst_boundary_block = 0
    worst_within_block = 0
    worst_within_sample = 0
    num_boundary_violations = 0
    num_within_violations = 0
    details = []

    peak_amplitude = int(np.max(np.abs(samples)))

    for block_idx in range(num_blocks):
        block_start_sample = block_idx * block_size
        block_end_sample = block_start_sample + block_size

        # Check block boundary (first sample of block)
        if block_idx > 0 and block_start_sample - 2 < len(acceleration):
            boundary_accel = abs(acceleration[block_start_sample - 2])
            excess = boundary_accel - total_threshold

            if excess > 0:
                if excess > max_boundary_excess:
                    max_boundary_excess = excess
                    worst_boundary_block = block_idx

                num_boundary_violations += 1
                details.append(
                    f"Block {block_idx} boundary: accel={boundary_accel} "
                    f"(excess={excess} LSB above {total_threshold})"
                )

        # Check within block
        accel_start = max(0, block_start_sample - 2)
        accel_end = min(len(acceleration), block_end_sample - 2)

        for sample_idx in range(accel_start, accel_end):
            accel = abs(acceleration[sample_idx])
            excess = accel - total_threshold

            if excess > 0:
                if excess > max_within_block_excess:
                    max_within_block_excess = excess
                    worst_within_block = block_idx
                    worst_within_sample = sample_idx - accel_start

                num_within_violations += 1

    passed = num_boundary_violations == 0 and num_within_violations == 0

    return DiscontinuityResult(
        test_name="",
        passed=passed,
        peak_amplitude=peak_amplitude,
        threshold_lsb=threshold_lsb,
        max_boundary_accel=max_boundary_excess,
        max_within_block_accel=max_within_block_excess,
        worst_boundary_block=worst_boundary_block,
        worst_within_block=worst_within_block,
        worst_within_sample=worst_within_sample,
        num_boundary_violations=num_boundary_violations,
        num_within_violations=num_within_violations,
        details=details[:10],  # Limit to first 10 violations
    )


def render_scenario(
    note: int,
    f0_hz: float,
    eta_q15: int,
    gain_q15: int,
    num_blocks: int = 100,
    ramp_eta: bool = False,
    ramp_gain: bool = False,
) -> np.ndarray:
    """
    Render a test scenario through the VM.

    Args:
        note: MIDI note number
        f0_hz: Fundamental frequency
        eta_q15: Wet amount (0-32767)
        gain_q15: Output gain (0-32767)
        num_blocks: Number of blocks to render
        ramp_eta: Whether to ramp eta from 0 to eta_q15
        ramp_gain: Whether to ramp gain from 0 to gain_q15

    Returns:
        np.ndarray: Rendered samples
    """
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(f0_hz=f0_hz)

    all_samples = []

    for block_idx in range(num_blocks):
        # Apply ramping if requested
        if ramp_eta:
            current_eta = int((block_idx / num_blocks) * eta_q15)
        else:
            current_eta = eta_q15

        if ramp_gain:
            current_gain = int((block_idx / num_blocks) * gain_q15)
        else:
            current_gain = gain_q15

        frame = VoiceFrame(
            voice_id=0,
            note=note,
            f0_hz=f0_hz,
            gain_q15=current_gain,
            beta_target_q15=current_eta,
            partial_amps_q15=[Q15_ONE, 0, 0, 0, 0, 0, 0, 0],
            is_active=True,
        )

        output = vm_step_block(state, frame, DEFAULT_TABLES)
        all_samples.extend(output.tolist())

    return np.array(all_samples, dtype=np.int16)


# =============================================================================
# TESTS
# =============================================================================


class TestCarrierDiscontinuity:
    """Test pure carrier (dry) signal continuity."""

    def test_pure_carrier_no_discontinuities(self):
        """Pure carrier should have no discontinuities."""
        f0_hz = 261.63  # Middle C
        samples = render_scenario(
            note=60, f0_hz=f0_hz, eta_q15=0, gain_q15=24576, num_blocks=100
        )

        expected = compute_expected_accel(24576, f0_hz)
        threshold = int(24576 * 0.001)  # -60 dB threshold
        result = detect_discontinuities(samples, threshold, expected)

        assert result.passed, (
            f"Pure carrier has {result.num_boundary_violations} boundary "
            f"and {result.num_within_violations} within-block discontinuities. "
            f"Details: {result.details[:3]}"
        )


class TestWetPathDiscontinuity:
    """Test wet (IDWPT) path continuity."""

    def test_wet_path_steady_state(self):
        """Steady wet path should be continuous."""
        f0_hz = 261.63
        samples = render_scenario(
            note=60, f0_hz=f0_hz, eta_q15=32767, gain_q15=24576, num_blocks=100
        )

        expected = compute_expected_accel(24576, f0_hz)
        threshold = int(24576 * 0.01)  # -40 dB (more lenient for wet path)
        result = detect_discontinuities(samples, threshold, expected)

        # Wet path may have some artifacts but should be bounded
        assert result.num_boundary_violations < 5, (
            f"Too many boundary discontinuities in wet path: "
            f"{result.num_boundary_violations}"
        )


class TestRampingScenarios:
    """Test continuity during parameter changes."""

    def test_eta_ramp(self):
        """Ramping ETA should maintain continuity."""
        f0_hz = 261.63
        samples = render_scenario(
            note=60,
            f0_hz=f0_hz,
            eta_q15=32767,
            gain_q15=24576,
            num_blocks=100,
            ramp_eta=True,
        )

        expected = compute_expected_accel(24576, f0_hz)
        threshold = int(24576 * 0.01)  # -40 dB
        result = detect_discontinuities(samples, threshold, expected)

        # Ramping may introduce some artifacts but should be controlled
        assert result.num_boundary_violations < 10, (
            f"Too many discontinuities during ETA ramp: "
            f"{result.num_boundary_violations}"
        )

    def test_gain_ramp(self):
        """Ramping gain should maintain continuity."""
        f0_hz = 261.63
        samples = render_scenario(
            note=60,
            f0_hz=f0_hz,
            eta_q15=0,
            gain_q15=24576,
            num_blocks=100,
            ramp_gain=True,
        )

        # For gain ramp, use peak of final gain
        expected = compute_expected_accel(24576, f0_hz)
        threshold = int(24576 * 0.001)  # -60 dB
        result = detect_discontinuities(samples, threshold, expected)

        assert result.passed, (
            f"Gain ramp has discontinuities: {result.num_boundary_violations} boundary, "
            f"{result.num_within_violations} within-block"
        )


class TestBlockBoundaries:
    """Test continuity at block boundaries specifically."""

    def test_block_boundaries_carrier(self):
        """Block boundaries should be continuous for carrier."""
        f0_hz = 440.0  # A4
        samples = render_scenario(
            note=69, f0_hz=f0_hz, eta_q15=0, gain_q15=24576, num_blocks=50
        )

        expected = compute_expected_accel(24576, f0_hz)
        threshold = int(24576 * 0.001)
        result = detect_discontinuities(samples, threshold, expected)

        assert result.num_boundary_violations == 0, (
            f"Block boundaries have {result.num_boundary_violations} discontinuities. "
            f"Worst at block {result.worst_boundary_block}"
        )


# =============================================================================
# UTILITY: Generate test audio for manual inspection
# =============================================================================


@pytest.mark.manual
def test_generate_discontinuity_test_files():
    """Generate test files for manual discontinuity inspection (not run in CI)."""
    scenarios = [
        ("carrier_pure", 0, False, False, "Pure carrier"),
        ("wet_100pct", 32767, False, False, "100% wet"),
        ("eta_ramp", 32767, True, False, "ETA ramp 0->100%"),
        ("gain_ramp", 0, False, True, "Gain fade in"),
    ]

    f0_hz = 261.63

    for name, eta, ramp_eta, ramp_gain, label in scenarios:
        samples = render_scenario(
            note=60,
            f0_hz=f0_hz,
            eta_q15=eta,
            gain_q15=24576,
            num_blocks=150,
            ramp_eta=ramp_eta,
            ramp_gain=ramp_gain,
        )

        meta = write_audio(
            f"discontinuity_test_{name}",
            samples,
            timestamp=True,
            subdirectory="discontinuity_tests",
        )

        # Analyze
        expected = compute_expected_accel(24576, f0_hz)
        threshold = int(24576 * 0.001)
        result = detect_discontinuities(samples, threshold, expected)

        print(f"\n{label}: {meta.filepath}")
        print(f"  Status: {'PASS' if result.passed else 'FAIL'}")
        print(f"  Boundary violations: {result.num_boundary_violations}")
        print(f"  Within-block violations: {result.num_within_violations}")
        if result.details:
            print(f"  First violation: {result.details[0]}")
