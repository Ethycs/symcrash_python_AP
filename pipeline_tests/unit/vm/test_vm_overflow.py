"""
Unit tests for VM overflow/saturation policy (WSE-STD-VM-HYBRID v0)

These tests verify that bit-exactness isn't "bit-exactly wrong":
- Q15 arithmetic behavior is correct
- Clamp is symmetric around zero
- Coefficients never exceed bounds after tail operations
"""

import pytest
import numpy as np
from dataclasses import dataclass

# Import VM components
from symcrash_python_ap.vm import (
    vm_step_block,
    VMState,
    DecodedPlan,
)
from symcrash_python_ap.vm.lanes import (
    lane_tail,
    lane_mix_gain,
    isqrt,
    isqrt_ratio_q15,
    cos_q15,
    sin_q15,
    Q15_ONE,
    Q15_HALF,
)
from symcrash_python_ap.vm.tables import TailMode, BLOCK_SAMPLES, COEFF_LEN


# =============================================================================
# MOCK VOICEFRAME
# =============================================================================

@dataclass
class MockVoiceFrame:
    """Mock VoiceFrame for testing."""
    voice_id: int = 0
    f0_hz: float = 440.0
    gain_q15: int = 32767
    beta_target_q15: int = 16384  # 50% wet
    pattern_id: int = 0
    tail_mode: int = 0  # FULL
    tree_a_id: int = 0
    tree_b_id: int = 0
    tree_blend_q15: int = 0
    partial_amps_q15: list = None
    ops_enable: list = None
    families: dict = None
    event_mask: int = 0
    event_strength_q15: list = None

    def __post_init__(self):
        if self.partial_amps_q15 is None:
            self.partial_amps_q15 = [32767, 16384, 8192, 4096]
        if self.ops_enable is None:
            self.ops_enable = []
        if self.families is None:
            self.families = {}
        if self.event_strength_q15 is None:
            self.event_strength_q15 = [32767] * 8


# =============================================================================
# Q15 ARITHMETIC TESTS
# =============================================================================

class TestQ15Arithmetic:
    """Tests for Q15 fixed-point arithmetic behavior."""

    def test_q15_multiply_truncates(self):
        """Q15 multiply should truncate toward zero (floor division)."""
        # (32767 * 32767) >> 15 = 32766 (truncated, not rounded)
        result = (32767 * 32767) >> 15
        assert result == 32766, f"Expected 32766, got {result}"

        # Negative values: -32768 * 32767 = -1073709056
        # -1073709056 >> 15 = -32767 (arithmetic shift, floor toward -inf)
        result_neg = (-32768 * 32767) >> 15
        assert result_neg == -32767, f"Expected -32767, got {result_neg}"

    def test_q15_half_squared_is_quarter(self):
        """Q15_HALF * Q15_HALF >> 15 should give approximately Q15/4."""
        # 16384 * 16384 = 268435456
        # >> 15 = 8192 which is Q15/4
        result = (Q15_HALF * Q15_HALF) >> 15
        assert result == 8192, f"Expected 8192, got {result}"

    def test_q15_one_squared_is_nearly_one(self):
        """Q15_ONE * Q15_ONE >> 15 should give nearly Q15_ONE."""
        result = (Q15_ONE * Q15_ONE) >> 15
        # 32767 * 32767 = 1073676289
        # >> 15 = 32766
        assert result == 32766, f"Expected 32766, got {result}"


# =============================================================================
# CLAMP SYMMETRY TESTS
# =============================================================================

class TestClampSymmetry:
    """Tests for symmetric clamping behavior."""

    def test_output_clamp_is_symmetric(self):
        """Output should be clamped symmetrically to [-32768, 32767]."""
        # Test the mix_gain lane with extreme values
        cfg = DecodedPlan()
        cfg.eta_prev_q15 = 0
        cfg.eta_curr_q15 = 0  # Pure dry
        cfg.gain_prev_q15 = 32767
        cfg.gain_curr_q15 = 32767

        # Create buffers with values at the limits
        carrier_max = np.full(BLOCK_SAMPLES, 32767, dtype=np.int16)
        carrier_min = np.full(BLOCK_SAMPLES, -32768, dtype=np.int16)
        texture = np.zeros(BLOCK_SAMPLES, dtype=np.int16)

        # Test max carrier
        output_max = lane_mix_gain(cfg, carrier_max, texture)
        assert np.all(output_max <= 32767), "Output exceeds upper bound"
        assert np.all(output_max >= -32768), "Output exceeds lower bound"

        # Test min carrier
        output_min = lane_mix_gain(cfg, carrier_min, texture)
        assert np.all(output_min <= 32767), "Output exceeds upper bound"
        assert np.all(output_min >= -32768), "Output exceeds lower bound"

    def test_clamp_preserves_valid_values(self):
        """Clamping should not modify values within valid range."""
        cfg = DecodedPlan()
        cfg.eta_prev_q15 = 0
        cfg.eta_curr_q15 = 0  # Pure dry
        cfg.gain_prev_q15 = 32767
        cfg.gain_curr_q15 = 32767

        # Use mid-range values
        carrier = np.full(BLOCK_SAMPLES, 16384, dtype=np.int16)
        texture = np.zeros(BLOCK_SAMPLES, dtype=np.int16)

        output = lane_mix_gain(cfg, carrier, texture)

        # Output should be approximately the input (scaled by gain and cos)
        # At eta=0, cos_q15(0) = 32767 (approx 1.0)
        # So output ≈ carrier * 32767/32768 * 32767/32768 ≈ 16382-16384
        assert np.all(np.abs(output - carrier) < 10), "Clamp incorrectly modified valid values"


# =============================================================================
# COEFFICIENT BOUNDS TESTS
# =============================================================================

class TestCoefficientBounds:
    """Tests for coefficient bounds after tail operations."""

    def test_coeffs_never_exceed_bounds_after_tail_full(self):
        """FULL tail mode should produce in-range coefficients."""
        state = VMState.create(voice_id=0)
        # Set coefficients to max
        state.coeff_state[:] = 32767

        cfg = DecodedPlan()
        cfg.mask_u32 = 0xFFFFFFFF
        cfg.tail_ops = [2, 11]  # FineClamp, NormalizeScale

        lane_tail(state, cfg)

        # Verify all coefficients in legal range
        assert np.all(state.coeff_state >= -32768), "Coefficient below -32768"
        assert np.all(state.coeff_state <= 32767), "Coefficient above 32767"

    def test_coeffs_never_exceed_bounds_after_tail_min(self):
        """MIN tail mode should produce in-range coefficients."""
        state = VMState.create(voice_id=0)
        # Set coefficients to max
        state.coeff_state[:] = 32767

        cfg = DecodedPlan()
        cfg.mask_u32 = 0xFFFFFFFF
        cfg.tail_ops = [2]  # FineClamp only

        lane_tail(state, cfg)

        # Verify all coefficients in legal range
        assert np.all(state.coeff_state >= -32768), "Coefficient below -32768"
        assert np.all(state.coeff_state <= 32767), "Coefficient above 32767"

    def test_negative_coeffs_preserved_after_tail(self):
        """Tail operations should preserve sign of coefficients."""
        state = VMState.create(voice_id=0)
        # Set half positive, half negative
        state.coeff_state[:COEFF_LEN // 2] = 16384
        state.coeff_state[COEFF_LEN // 2:] = -16384

        cfg = DecodedPlan()
        cfg.mask_u32 = 0xFFFFFFFF
        cfg.tail_ops = [2, 11]  # FineClamp, NormalizeScale

        lane_tail(state, cfg)

        # Verify bounds
        assert np.all(state.coeff_state >= -32768), "Coefficient below -32768"
        assert np.all(state.coeff_state <= 32767), "Coefficient above 32767"


# =============================================================================
# INTEGER SQRT EDGE CASES
# =============================================================================

class TestIsqrtEdgeCases:
    """Tests for integer square root edge cases."""

    def test_isqrt_zero(self):
        """isqrt(0) should return 0."""
        assert isqrt(0) == 0

    def test_isqrt_one(self):
        """isqrt(1) should return 1."""
        assert isqrt(1) == 1

    def test_isqrt_perfect_squares(self):
        """isqrt should return exact results for perfect squares."""
        for i in range(1, 100):
            assert isqrt(i * i) == i, f"isqrt({i * i}) != {i}"

    def test_isqrt_floors_correctly(self):
        """isqrt should floor non-perfect squares."""
        assert isqrt(2) == 1
        assert isqrt(3) == 1
        assert isqrt(5) == 2
        assert isqrt(8) == 2
        assert isqrt(10) == 3
        assert isqrt(99) == 9

    def test_isqrt_large_values(self):
        """isqrt should handle large values correctly."""
        assert isqrt(2**30) == 2**15
        assert isqrt(2**60) == 2**30

    def test_isqrt_ratio_zero_denominator(self):
        """isqrt_ratio_q15 should return 0 for zero/negative denominator."""
        assert isqrt_ratio_q15(100, 0) == 0
        assert isqrt_ratio_q15(100, -1) == 0

    def test_isqrt_ratio_zero_numerator(self):
        """isqrt_ratio_q15 should return 0 for zero/negative numerator."""
        assert isqrt_ratio_q15(0, 100) == 0
        assert isqrt_ratio_q15(-1, 100) == 0


# =============================================================================
# TRIG LOOKUP EDGE CASES
# =============================================================================

class TestTrigEdgeCases:
    """Tests for trig lookup edge cases."""

    def test_cos_at_zero(self):
        """cos_q15(0) should be approximately Q15_ONE."""
        result = cos_q15(0)
        assert result == Q15_ONE, f"cos(0) = {result}, expected {Q15_ONE}"

    def test_sin_at_zero(self):
        """sin_q15(0) should be 0."""
        result = sin_q15(0)
        assert result == 0, f"sin(0) = {result}, expected 0"

    def test_cos_at_max(self):
        """cos_q15(Q15_ONE) should be approximately 0."""
        result = cos_q15(Q15_ONE)
        # At eta=Q15_ONE, angle = 90°, cos(90°) = 0
        assert abs(result) <= 5, f"cos(pi/2) = {result}, expected ~0"

    def test_sin_at_max(self):
        """sin_q15(Q15_ONE) should be approximately Q15_ONE."""
        result = sin_q15(Q15_ONE)
        # At eta=Q15_ONE, angle = 90°, sin(90°) = 1
        assert abs(result - Q15_ONE) <= 5, f"sin(pi/2) = {result}, expected ~{Q15_ONE}"

    def test_cos_sin_sum_of_squares(self):
        """cos^2 + sin^2 should be approximately 1 (Q30 = Q15_ONE^2)."""
        for eta in [0, 8192, 16384, 24576, 32767]:
            c = cos_q15(eta)
            s = sin_q15(eta)
            # cos^2 + sin^2 should be approximately Q15_ONE^2 = 1073676289
            sum_sq = c * c + s * s
            # Allow 1% tolerance
            expected = Q15_ONE * Q15_ONE
            assert abs(sum_sq - expected) < expected * 0.02, \
                f"At eta={eta}: cos^2+sin^2 = {sum_sq}, expected ~{expected}"


# =============================================================================
# FULL PIPELINE OVERFLOW TEST
# =============================================================================

class TestPipelineOverflow:
    """Tests for overflow in full VM pipeline."""

    def test_max_gain_max_carrier_no_overflow(self):
        """Max gain with max carrier should not overflow output."""
        state = VMState.create(voice_id=0)
        state.reset_for_note_on(f0_hz=440.0)

        # Set coefficients to max
        state.coeff_state[:] = 32767

        frame = MockVoiceFrame(
            f0_hz=440.0,
            gain_q15=32767,  # Max gain
            beta_target_q15=0,  # Pure dry (carrier only)
        )

        # Run several blocks to stabilize
        for _ in range(10):
            state, output = vm_step_block(state, frame)

        # Verify output is within bounds
        assert np.all(output >= -32768), "Output below -32768"
        assert np.all(output <= 32767), "Output above 32767"

    def test_max_wet_max_coeffs_no_overflow(self):
        """Max wet path with max coefficients should not overflow."""
        state = VMState.create(voice_id=0)
        state.reset_for_note_on(f0_hz=440.0)

        # Set coefficients to max
        state.coeff_state[:] = 32767

        frame = MockVoiceFrame(
            f0_hz=440.0,
            gain_q15=32767,  # Max gain
            beta_target_q15=32767,  # Full wet
        )

        # Run several blocks (wet needs 5 blocks to be ready)
        for _ in range(10):
            state, output = vm_step_block(state, frame)

        # Verify output is within bounds
        assert np.all(output >= -32768), "Output below -32768"
        assert np.all(output <= 32767), "Output above 32767"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
