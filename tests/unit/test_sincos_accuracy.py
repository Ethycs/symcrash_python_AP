"""
Sincos Interpolation Accuracy Tests
====================================

Tests the sincos lookup table interpolation used for equal-power crossfading.

The _sincos_batch_q15 function provides sin/cos values in Q15 format for
mapping ETA (wet amount) to crossfade coefficients.

Key properties tested:
- Accuracy vs floating point reference
- Equal-power property: cos² + sin² = 1
- Boundary values (eta=0, eta=32767)
- Interpolation smoothness across full range
"""

import pytest
import numpy as np
import math

from symcrash_python_ap.vm.lanes import _sincos_batch_q15

Q15_ONE = 32767


class TestSincosAccuracy:
    """Test sincos interpolation accuracy."""

    def test_sincos_boundary_values(self):
        """Test sin/cos at key ETA boundary values."""
        test_cases = [
            (0, 32767, 0),  # eta=0: cos=1, sin=0 (full dry)
            (16384, 23170, 23170),  # eta=16384: cos≈sin≈0.707 (equal power)
            (32767, 0, 32767),  # eta=32767: cos=0, sin=1 (full wet)
        ]

        for eta, expected_cos, expected_sin in test_cases:
            phase = (eta * 0x40000000) >> 15
            phases = np.array([phase], dtype=np.uint32)
            cos_val, sin_val = _sincos_batch_q15(phases)

            # Allow small error (±50 LSB)
            assert abs(cos_val[0] - expected_cos) <= 50, (
                f"At eta={eta}: cos={cos_val[0]}, expected≈{expected_cos}"
            )
            assert abs(sin_val[0] - expected_sin) <= 50, (
                f"At eta={eta}: sin={sin_val[0]}, expected≈{expected_sin}"
            )

    def test_sincos_vs_floating_point(self):
        """Test accuracy vs floating point reference."""
        # Test across full range
        eta_vals = np.arange(0, 32768, dtype=np.int32)
        phases = ((eta_vals.astype(np.int64) * 0x40000000) >> 15).astype(np.uint32)

        cos_vals, sin_vals = _sincos_batch_q15(phases)

        # Compute expected values using floating point
        eta_float = eta_vals / 32767.0  # 0 to 1
        expected_cos = (np.cos(eta_float * np.pi / 2) * 32767).astype(np.int32)
        expected_sin = (np.sin(eta_float * np.pi / 2) * 32767).astype(np.int32)

        # Compute errors
        cos_error = np.abs(cos_vals - expected_cos)
        sin_error = np.abs(sin_vals - expected_sin)

        # Max error should be small (interpolation accuracy)
        assert cos_error.max() <= 2, f"Cos max error: {cos_error.max()}"
        assert sin_error.max() <= 2, f"Sin max error: {sin_error.max()}"

        # RMS error should be very small
        cos_rms = np.sqrt(np.mean(cos_error**2))
        sin_rms = np.sqrt(np.mean(sin_error**2))

        assert cos_rms < 1.0, f"Cos RMS error: {cos_rms:.3f}"
        assert sin_rms < 1.0, f"Sin RMS error: {sin_rms:.3f}"


class TestEqualPowerProperty:
    """Test equal-power crossfade property."""

    def test_equal_power_sum(self):
        """cos² + sin² should equal 1 (32767 in Q15)."""
        eta_vals = np.arange(0, 32768, dtype=np.int32)
        phases = ((eta_vals.astype(np.int64) * 0x40000000) >> 15).astype(np.uint32)

        cos_vals, sin_vals = _sincos_batch_q15(phases)

        # Compute power: (cos² + sin²) >> 15
        # In Q15: cos*cos = Q30, so >> 15 = Q15
        power_sum = (
            (cos_vals.astype(np.int64) ** 2 + sin_vals.astype(np.int64) ** 2) >> 15
        ).astype(np.int32)

        expected_power = 32767

        # Power sum should be very close to 32767 everywhere
        power_error = np.abs(power_sum - expected_power)
        max_error = power_error.max()

        assert max_error <= 10, (
            f"Equal-power property violated: max error = {max_error}, "
            f"power range = [{power_sum.min()}, {power_sum.max()}]"
        )

    def test_equal_power_at_key_points(self):
        """Test equal-power at specific ETA values."""
        test_etas = [0, 8192, 16384, 24576, 32767]

        for eta in test_etas:
            phase = (eta * 0x40000000) >> 15
            phases = np.array([phase], dtype=np.uint32)
            cos_val, sin_val = _sincos_batch_q15(phases)

            # Compute power
            power = (cos_val[0] ** 2 + sin_val[0] ** 2) >> 15

            # Should be very close to 32767
            error = abs(power - 32767)
            assert error <= 10, (
                f"At eta={eta}: power={power}, error={error} "
                f"(cos={cos_val[0]}, sin={sin_val[0]})"
            )


class TestSmoothness:
    """Test interpolation smoothness."""

    def test_no_large_jumps(self):
        """Consecutive values should change smoothly."""
        eta_vals = np.arange(0, 32768, dtype=np.int32)
        phases = ((eta_vals.astype(np.int64) * 0x40000000) >> 15).astype(np.uint32)

        cos_vals, sin_vals = _sincos_batch_q15(phases)

        # Check consecutive differences
        cos_diff = np.abs(np.diff(cos_vals.astype(np.int32)))
        sin_diff = np.abs(np.diff(sin_vals.astype(np.int32)))

        # Max step should be small (smooth interpolation)
        assert cos_diff.max() <= 2, f"Cos has large jump: {cos_diff.max()}"
        assert sin_diff.max() <= 2, f"Sin has large jump: {sin_diff.max()}"


class TestCrossfadeWeights:
    """Test that crossfade weights behave correctly."""

    def test_dry_weight_decreases_with_eta(self):
        """Dry weight (cos) should decrease as ETA increases."""
        etas = np.array([0, 8192, 16384, 24576, 32767], dtype=np.int32)
        phases = ((etas.astype(np.int64) * 0x40000000) >> 15).astype(np.uint32)

        cos_vals, _ = _sincos_batch_q15(phases)

        # cos should be monotonically decreasing
        for i in range(len(cos_vals) - 1):
            assert cos_vals[i] >= cos_vals[i + 1], (
                f"Dry weight not decreasing: "
                f"cos[{etas[i]}]={cos_vals[i]}, cos[{etas[i+1]}]={cos_vals[i+1]}"
            )

    def test_wet_weight_increases_with_eta(self):
        """Wet weight (sin) should increase as ETA increases."""
        etas = np.array([0, 8192, 16384, 24576, 32767], dtype=np.int32)
        phases = ((etas.astype(np.int64) * 0x40000000) >> 15).astype(np.uint32)

        _, sin_vals = _sincos_batch_q15(phases)

        # sin should be monotonically increasing
        for i in range(len(sin_vals) - 1):
            assert sin_vals[i] <= sin_vals[i + 1], (
                f"Wet weight not increasing: "
                f"sin[{etas[i]}]={sin_vals[i]}, sin[{etas[i+1]}]={sin_vals[i+1]}"
            )
