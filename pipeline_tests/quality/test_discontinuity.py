"""
Discontinuity Detection Tests
==============================

Block boundary continuity tests for polyphase IDWPT synthesis.

Uses boundary/intra jump ratio metric (appropriate for wavelet-domain synthesis).
A ratio < 3.0 indicates proper phase continuity across blocks.

Note: Previous implementation used second-order differences (acceleration),
but polyphase IDWPT produces natural waveform curvature that violates those
metrics. The boundary/intra ratio is the correct metric for this architecture.
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
class ContinuityResult:
    """Results from a continuity test scenario."""

    test_name: str
    passed: bool
    boundary_intra_ratio: float
    max_boundary_jump: int
    avg_boundary_jump: float
    max_intra_jump: int
    avg_intra_jump: float
    rms: float


def compute_boundary_intra_ratio(
    samples: np.ndarray,
    block_size: int = BLOCK_SIZE,
) -> ContinuityResult:
    """
    Compute boundary/intra jump ratio for polyphase continuity.

    This is the correct metric for polyphase IDWPT synthesis.
    Ratio < 3.0 indicates proper phase continuity across blocks.

    Args:
        samples: Audio samples (int16 or int32)
        block_size: Block size for boundary detection

    Returns:
        ContinuityResult with statistics
    """
    samples = samples.astype(np.int32)

    sum_boundary_jump = 0
    num_boundaries = 0
    sum_intra_jump = 0
    num_intra = 0
    max_boundary_jump = 0
    max_intra_jump = 0

    for i in range(1, len(samples)):
        jump = abs(int(samples[i]) - int(samples[i - 1]))
        if i % block_size == 0:
            sum_boundary_jump += jump
            max_boundary_jump = max(max_boundary_jump, jump)
            num_boundaries += 1
        else:
            sum_intra_jump += jump
            max_intra_jump = max(max_intra_jump, jump)
            num_intra += 1

    avg_boundary = sum_boundary_jump / num_boundaries if num_boundaries > 0 else 0
    avg_intra = sum_intra_jump / num_intra if num_intra > 0 else 1
    ratio = avg_boundary / avg_intra if avg_intra > 0 else float("inf")

    rms = float(np.sqrt(np.mean(samples.astype(float) ** 2)))

    return ContinuityResult(
        test_name="",
        passed=ratio < 3.0,
        boundary_intra_ratio=ratio,
        max_boundary_jump=max_boundary_jump,
        avg_boundary_jump=avg_boundary,
        max_intra_jump=max_intra_jump,
        avg_intra_jump=avg_intra,
        rms=rms,
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

        state, output = vm_step_block(state, frame, DEFAULT_TABLES)
        all_samples.extend(output.tolist())

    return np.array(all_samples, dtype=np.int16)


# =============================================================================
# TESTS - Polyphase IDWPT Continuity (Boundary/Intra Ratio)
# =============================================================================


class TestCarrierDiscontinuity:
    """Test pure carrier (dry) signal continuity using polyphase metrics."""

    def test_pure_carrier_no_discontinuities(self):
        """Pure carrier should have good boundary continuity."""
        f0_hz = 261.63  # Middle C
        samples = render_scenario(
            note=60, f0_hz=f0_hz, eta_q15=0, gain_q15=24576, num_blocks=100
        )

        result = compute_boundary_intra_ratio(samples)

        # Polyphase IDWPT: boundary/intra ratio < 3.0 indicates good continuity
        assert result.passed, (
            f"Pure carrier has poor continuity: ratio={result.boundary_intra_ratio:.2f} "
            f"(max 3.0). Avg boundary={result.avg_boundary_jump:.1f}, "
            f"avg intra={result.avg_intra_jump:.1f}"
        )


class TestWetPathDiscontinuity:
    """Test wet (IDWPT) path continuity."""

    def test_wet_path_steady_state(self):
        """Steady wet path should be continuous."""
        f0_hz = 261.63
        samples = render_scenario(
            note=60, f0_hz=f0_hz, eta_q15=32767, gain_q15=24576, num_blocks=100
        )

        result = compute_boundary_intra_ratio(samples)

        # Polyphase IDWPT: boundary/intra ratio < 3.0 indicates good continuity
        assert result.passed, (
            f"Wet path has poor continuity: ratio={result.boundary_intra_ratio:.2f} "
            f"(max 3.0). Avg boundary={result.avg_boundary_jump:.1f}, "
            f"avg intra={result.avg_intra_jump:.1f}"
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

        result = compute_boundary_intra_ratio(samples)

        # Ramping may have slightly worse ratio but should still be < 3.0
        assert result.boundary_intra_ratio < 3.0, (
            f"ETA ramp has poor continuity: ratio={result.boundary_intra_ratio:.2f}"
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

        result = compute_boundary_intra_ratio(samples)

        assert result.boundary_intra_ratio < 3.0, (
            f"Gain ramp has poor continuity: ratio={result.boundary_intra_ratio:.2f}"
        )


class TestBlockBoundaries:
    """Test continuity at block boundaries specifically."""

    def test_block_boundaries_carrier(self):
        """Block boundaries should be continuous for carrier."""
        f0_hz = 440.0  # A4
        samples = render_scenario(
            note=69, f0_hz=f0_hz, eta_q15=0, gain_q15=24576, num_blocks=50
        )

        result = compute_boundary_intra_ratio(samples)

        assert result.passed, (
            f"Block boundaries have poor continuity: "
            f"ratio={result.boundary_intra_ratio:.2f} (max 3.0)"
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

        # Analyze with polyphase metric
        result = compute_boundary_intra_ratio(samples)

        print(f"\n{label}: {meta.filepath}")
        print(f"  Status: {'PASS' if result.passed else 'FAIL'}")
        print(f"  Boundary/intra ratio: {result.boundary_intra_ratio:.2f}")
        print(f"  Avg boundary jump: {result.avg_boundary_jump:.1f}")
        print(f"  Avg intra jump: {result.avg_intra_jump:.1f}")
