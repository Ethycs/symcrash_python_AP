"""
Sine LUT Tests
==============

Tests the 64K sine lookup table (LUT) implementation.

The sine_core module provides fast Q15 sine/cosine lookups using a
16-bit phase index (0-65535 maps to 0-360 degrees).

Validates:
- Boundary values (0°, 90°, 180°, 270°)
- Scalar vs vectorized consistency
- Accuracy (≤1 LSB error at key points)
- Performance characteristics
"""

import pytest
import numpy as np

from symcrash_python_ap.wavelet.sine_core import (
    sine_q15_fast,
    cosine_q15_fast,
    sine_q15_vectorized,
    cosine_q15_vectorized,
)

Q15_ONE = 32767


class TestSineLUTBoundaryValues:
    """Test sine LUT at key boundary values."""

    @pytest.mark.parametrize(
        "phase,expected_sin,expected_cos,name",
        [
            (0, 0, 32767, "0°"),
            (16384, 32767, 0, "90°"),
            (32768, 0, -32767, "180°"),
            (49152, -32767, 0, "270°"),
        ],
    )
    def test_boundary_value(self, phase, expected_sin, expected_cos, name):
        """Test sine/cos at key angles."""
        actual_sin = sine_q15_fast(phase)
        actual_cos = cosine_q15_fast(phase)

        # Allow ±1 LSB tolerance
        assert abs(actual_sin - expected_sin) <= 1, (
            f"Sine at {name}: got {actual_sin}, expected {expected_sin}"
        )
        assert abs(actual_cos - expected_cos) <= 1, (
            f"Cosine at {name}: got {actual_cos}, expected {expected_cos}"
        )


class TestSineLUTVectorized:
    """Test vectorized sine/cosine operations."""

    def test_vectorized_sine_matches_scalar(self):
        """Vectorized sine should match scalar results."""
        phases = np.array([0, 16384, 32768, 49152], dtype=np.int32)
        expected = np.array(
            [sine_q15_fast(p) for p in phases], dtype=np.int32
        )

        actual = sine_q15_vectorized(phases)

        np.testing.assert_array_equal(actual, expected)

    def test_vectorized_cosine_matches_scalar(self):
        """Vectorized cosine should match scalar results."""
        phases = np.array([0, 16384, 32768, 49152], dtype=np.int32)
        expected = np.array(
            [cosine_q15_fast(p) for p in phases], dtype=np.int32
        )

        actual = cosine_q15_vectorized(phases)

        np.testing.assert_array_equal(actual, expected)

    def test_vectorized_sine_boundary_accuracy(self):
        """Vectorized sine should have ≤1 LSB error at boundaries."""
        phases = np.array([0, 16384, 32768, 49152], dtype=np.int32)
        expected = np.array([0, 32767, 0, -32767], dtype=np.int32)

        actual = sine_q15_vectorized(phases)
        max_error = np.abs(actual - expected).max()

        assert max_error <= 1, f"Max error: {max_error} LSB"

    def test_vectorized_cosine_boundary_accuracy(self):
        """Vectorized cosine should have ≤1 LSB error at boundaries."""
        phases = np.array([0, 16384, 32768, 49152], dtype=np.int32)
        expected = np.array([32767, 0, -32767, 0], dtype=np.int32)

        actual = cosine_q15_vectorized(phases)
        max_error = np.abs(actual - expected).max()

        assert max_error <= 1, f"Max error: {max_error} LSB"


class TestSineLUTProperties:
    """Test mathematical properties of sine LUT."""

    def test_sine_range(self):
        """Sine values should be within [-32767, 32767]."""
        # Sample across full range
        phases = np.arange(0, 65536, 256, dtype=np.int32)
        values = sine_q15_vectorized(phases)

        assert values.min() >= -32767
        assert values.max() <= 32767

    def test_cosine_range(self):
        """Cosine values should be within [-32767, 32767]."""
        phases = np.arange(0, 65536, 256, dtype=np.int32)
        values = cosine_q15_vectorized(phases)

        assert values.min() >= -32767
        assert values.max() <= 32767

    def test_pythagorean_identity(self):
        """sin² + cos² should equal 1 (within rounding)."""
        phases = np.arange(0, 65536, 256, dtype=np.int32)
        sin_vals = sine_q15_vectorized(phases).astype(np.int64)
        cos_vals = cosine_q15_vectorized(phases).astype(np.int64)

        # Compute sin² + cos² in Q15*Q15 >> 15 = Q15
        sum_of_squares = ((sin_vals**2 + cos_vals**2) >> 15).astype(np.int32)
        expected = 32767

        # Should be very close to 32767 (1.0 in Q15)
        max_error = np.abs(sum_of_squares - expected).max()
        assert max_error <= 10, f"Max identity error: {max_error}"


class TestSineLUTSymmetry:
    """Test symmetry properties."""

    def test_sine_antisymmetry(self):
        """sin(-x) should equal -sin(x)."""
        # Test at a few points
        phases = np.array([1000, 5000, 10000], dtype=np.int32)
        neg_phases = ((-phases) & 0xFFFF).astype(np.int32)

        sin_pos = sine_q15_vectorized(phases)
        sin_neg = sine_q15_vectorized(neg_phases)

        # sin(-x) ≈ -sin(x)
        max_diff = np.abs(sin_pos + sin_neg).max()
        assert max_diff <= 2, f"Antisymmetry violated: {max_diff}"

    def test_cosine_symmetry(self):
        """cos(-x) should equal cos(x)."""
        phases = np.array([1000, 5000, 10000], dtype=np.int32)
        neg_phases = ((-phases) & 0xFFFF).astype(np.int32)

        cos_pos = cosine_q15_vectorized(phases)
        cos_neg = cosine_q15_vectorized(neg_phases)

        # cos(-x) ≈ cos(x)
        max_diff = np.abs(cos_pos - cos_neg).max()
        assert max_diff <= 2, f"Symmetry violated: {max_diff}"


class TestSineLUTPerformance:
    """Performance tests for sine LUT."""

    def test_vectorized_performance(self):
        """Vectorized sine lookup over a full phase sweep is fast.

        (Was a pytest-benchmark test; rewritten with manual timing since
        pytest-benchmark is not part of the environment.)
        """
        import time

        phases = np.arange(65536, dtype=np.int32)

        # Warm up (JIT / cache effects)
        result = sine_q15_vectorized(phases)
        assert len(result) == 65536

        start = time.perf_counter()
        for _ in range(10):
            result = sine_q15_vectorized(phases)
        elapsed = (time.perf_counter() - start) / 10

        assert len(result) == 65536
        # Generous threshold: a vectorized 64K lookup should be well under 100 ms
        assert elapsed < 0.1, f"sine_q15_vectorized too slow: {elapsed*1000:.1f} ms per 64K sweep"
