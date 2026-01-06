"""
Wavelet Transform Tests
========================

Tests the separated FWPT (Forward Wavelet Packet Transform) and
IDWPT (Inverse Discrete Wavelet Packet Transform) functions.

Validates:
- FWPT->IDWPT roundtrip accuracy
- Separated vs fused implementation consistency
- Transform error bounds (≤4 LSB)
- Coefficient preservation
"""

import pytest
import numpy as np

from symcrash_python_ap.wavelet.fast_lifting import (
    fwpt_jit,
    idwpt_jit,
    effect_mode_jit,
    NUMBA_AVAILABLE,
)

BLOCK_SIZE = 64


def generate_test_signal(freq=440, amplitude=16000):
    """Generate test sine wave block."""
    return [int(amplitude * np.sin(2 * np.pi * freq * i / BLOCK_SIZE)) for i in range(BLOCK_SIZE)]


class TestWaveletRoundtrip:
    """Test FWPT->IDWPT roundtrip accuracy."""

    def test_sine_wave_roundtrip(self):
        """Sine wave should roundtrip with minimal error."""
        samples = generate_test_signal(freq=440, amplitude=16000)

        # Forward transform
        coeffs = fwpt_jit(samples)
        assert len(coeffs) == BLOCK_SIZE

        # Inverse transform
        reconstructed = idwpt_jit(coeffs)
        assert len(reconstructed) == BLOCK_SIZE

        # Check error
        error = [abs(s - r) for s, r in zip(samples, reconstructed)]
        max_error = max(error)
        mean_error = sum(error) / len(error)

        assert max_error <= 4, f"Max roundtrip error: {max_error} LSB (limit: 4)"
        assert mean_error < 2.0, f"Mean roundtrip error: {mean_error:.2f} LSB"

    def test_dc_signal_roundtrip(self):
        """DC signal should roundtrip exactly."""
        samples = [10000] * BLOCK_SIZE

        coeffs = fwpt_jit(samples)
        reconstructed = idwpt_jit(coeffs)

        error = [abs(s - r) for s, r in zip(samples, reconstructed)]
        max_error = max(error)

        assert max_error <= 1, f"DC roundtrip error: {max_error}"

    def test_zero_signal_roundtrip(self):
        """Zero signal should roundtrip exactly."""
        samples = [0] * BLOCK_SIZE

        coeffs = fwpt_jit(samples)
        reconstructed = idwpt_jit(coeffs)

        assert all(r == 0 for r in reconstructed), "Zero signal not preserved"

    @pytest.mark.parametrize("amplitude", [1000, 5000, 16000, 30000])
    def test_amplitude_independence(self, amplitude):
        """Roundtrip error should be amplitude-independent."""
        samples = generate_test_signal(freq=440, amplitude=amplitude)

        coeffs = fwpt_jit(samples)
        reconstructed = idwpt_jit(coeffs)

        error = [abs(s - r) for s, r in zip(samples, reconstructed)]
        max_error = max(error)

        assert max_error <= 4, (
            f"At amplitude {amplitude}: max error = {max_error}"
        )


class TestSeparatedVsFused:
    """Test separated transforms match fused implementation."""

    def test_separated_matches_fused_identity_mask(self):
        """Separated with identity mask should match fused."""
        samples = generate_test_signal()

        # Fused path with identity mask
        mask_identity = [32767] * 64
        fused_out = effect_mode_jit(samples, mask_identity, headroom_bits=1)

        # Separated path
        coeffs = fwpt_jit(samples)
        separated_out = idwpt_jit(coeffs)

        # Compare
        diff = [abs(f - s) for f, s in zip(fused_out, separated_out)]
        max_diff = max(diff)

        assert max_diff <= 2, (
            f"Fused vs separated mismatch: {max_diff} LSB"
        )

    def test_separated_matches_fused_passthrough(self):
        """Separated should match fused passthrough (mask=None)."""
        samples = generate_test_signal()

        # Fused passthrough
        fused_out = effect_mode_jit(samples, None, headroom_bits=1)

        # Separated
        coeffs = fwpt_jit(samples)
        separated_out = idwpt_jit(coeffs)

        # Should be identical or very close
        diff = [abs(f - s) for f, s in zip(fused_out, separated_out)]
        max_diff = max(diff)

        assert max_diff <= 2, f"Passthrough mismatch: {max_diff} LSB"


class TestTransformProperties:
    """Test mathematical properties of transforms."""

    def test_fwpt_preserves_length(self):
        """FWPT should output same length as input."""
        samples = generate_test_signal()
        coeffs = fwpt_jit(samples)

        assert len(coeffs) == len(samples)

    def test_idwpt_preserves_length(self):
        """IDWPT should output same length as input."""
        samples = generate_test_signal()
        coeffs = fwpt_jit(samples)
        reconstructed = idwpt_jit(coeffs)

        assert len(reconstructed) == len(coeffs)

    def test_fwpt_coefficient_range(self):
        """FWPT coefficients should be bounded."""
        samples = generate_test_signal(amplitude=30000)
        coeffs = fwpt_jit(samples)

        # Coefficients shouldn't exceed reasonable bounds
        # (allowing for some expansion due to transform)
        max_coeff = max(abs(c) for c in coeffs)
        assert max_coeff < 100000, f"Coefficient too large: {max_coeff}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_large_amplitude_roundtrip(self):
        """Large amplitude signal should still roundtrip."""
        samples = generate_test_signal(amplitude=30000)

        coeffs = fwpt_jit(samples)
        reconstructed = idwpt_jit(coeffs)

        error = [abs(s - r) for s, r in zip(samples, reconstructed)]
        max_error = max(error)

        assert max_error <= 10, (
            f"Large amplitude error: {max_error} (relaxed limit)"
        )

    def test_negative_amplitude_roundtrip(self):
        """Negative values should roundtrip correctly."""
        samples = [-10000] * BLOCK_SIZE

        coeffs = fwpt_jit(samples)
        reconstructed = idwpt_jit(coeffs)

        error = [abs(s - r) for s, r in zip(samples, reconstructed)]
        max_error = max(error)

        assert max_error <= 1, f"Negative DC error: {max_error}"

    def test_mixed_sign_roundtrip(self):
        """Mixed positive/negative should roundtrip."""
        samples = generate_test_signal()  # Sine wave has both signs

        coeffs = fwpt_jit(samples)
        reconstructed = idwpt_jit(coeffs)

        # Check both positive and negative samples present
        assert any(r > 0 for r in reconstructed)
        assert any(r < 0 for r in reconstructed)

        error = [abs(s - r) for s, r in zip(samples, reconstructed)]
        assert max(error) <= 4


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
class TestNumbaOptimization:
    """Test Numba-optimized paths."""

    def test_numba_enabled(self):
        """Numba should be available for optimization."""
        assert NUMBA_AVAILABLE, "Numba not available"

    def test_numba_functions_callable(self):
        """Numba-compiled functions should be callable."""
        samples = generate_test_signal()

        # Should not raise
        coeffs = fwpt_jit(samples)
        reconstructed = idwpt_jit(coeffs)

        assert len(coeffs) == BLOCK_SIZE
        assert len(reconstructed) == BLOCK_SIZE
