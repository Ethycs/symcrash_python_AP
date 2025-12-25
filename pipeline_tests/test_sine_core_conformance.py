"""
SINE CORE CONFORMANCE TESTS
===========================

Tests for sine_core.py LUT-based sine implementation.
Extracted from sine_core.py self_test().
"""

import pytest
import math


class TestSineCoreValues:
    """Test boundary values for sine/cosine functions."""

    def test_sine_zero(self):
        """sin(0°) should be 0."""
        from symcrash_python_ap.wavelet.sine_core import sine_q15, Q_SCALE

        actual = sine_q15(0x00000000)
        assert abs(actual - 0) <= 1, f"sin(0°) = {actual}, expected 0"

    def test_sine_90(self):
        """sin(90°) should be Q_SCALE (32767)."""
        from symcrash_python_ap.wavelet.sine_core import sine_q15, Q_SCALE

        actual = sine_q15(0x40000000)
        assert abs(actual - Q_SCALE) <= 1, f"sin(90°) = {actual}, expected {Q_SCALE}"

    def test_sine_180(self):
        """sin(180°) should be 0."""
        from symcrash_python_ap.wavelet.sine_core import sine_q15

        actual = sine_q15(0x80000000)
        assert abs(actual - 0) <= 1, f"sin(180°) = {actual}, expected 0"

    def test_sine_270(self):
        """sin(270°) should be -Q_SCALE (-32767)."""
        from symcrash_python_ap.wavelet.sine_core import sine_q15, Q_SCALE

        actual = sine_q15(0xC0000000)
        assert abs(actual - (-Q_SCALE)) <= 1, f"sin(270°) = {actual}, expected {-Q_SCALE}"


class TestCosineIdentity:
    """Test that cosine = sin + 90°."""

    def test_cosine_zero(self):
        """cos(0°) should be Q_SCALE."""
        from symcrash_python_ap.wavelet.sine_core import cosine_q15, Q_SCALE

        actual = cosine_q15(0x00000000)
        assert abs(actual - Q_SCALE) <= 1, f"cos(0°) = {actual}, expected {Q_SCALE}"

    def test_cosine_90(self):
        """cos(90°) should be 0."""
        from symcrash_python_ap.wavelet.sine_core import cosine_q15

        actual = cosine_q15(0x40000000)
        assert abs(actual - 0) <= 1, f"cos(90°) = {actual}, expected 0"

    def test_cosine_180(self):
        """cos(180°) should be -Q_SCALE."""
        from symcrash_python_ap.wavelet.sine_core import cosine_q15, Q_SCALE

        actual = cosine_q15(0x80000000)
        assert abs(actual - (-Q_SCALE)) <= 1, f"cos(180°) = {actual}, expected {-Q_SCALE}"

    def test_cosine_270(self):
        """cos(270°) should be 0."""
        from symcrash_python_ap.wavelet.sine_core import cosine_q15

        actual = cosine_q15(0xC0000000)
        assert abs(actual - 0) <= 1, f"cos(270°) = {actual}, expected 0"


class TestSineCoreAccuracy:
    """Test sine implementation accuracy."""

    def test_max_error_within_2_lsb(self):
        """Maximum error should be less than 2 LSB."""
        from symcrash_python_ap.wavelet.sine_core import verify_accuracy

        max_err, rms_err, thd = verify_accuracy(num_samples=10000)
        assert max_err < 2.0, f"Max error {max_err:.2f} LSB exceeds 2 LSB limit"

    def test_thd_below_0_1_percent(self):
        """THD should be below 0.1%."""
        from symcrash_python_ap.wavelet.sine_core import verify_accuracy

        max_err, rms_err, thd = verify_accuracy(num_samples=10000)
        assert thd < 0.1, f"THD {thd:.4f}% exceeds 0.1% limit"


class TestSineCoreMonotonicity:
    """Test sine monotonicity in first quadrant."""

    def test_first_quadrant_monotonic(self):
        """Sine should be monotonically increasing in Q1 (0° to 90°)."""
        from symcrash_python_ap.wavelet.sine_core import sine_q15, PHASE_BITS

        prev = -1
        for i in range(1024):
            phase = int((i / 4096) * (1 << PHASE_BITS))  # 0 to 90°
            val = sine_q15(phase)
            assert val >= prev, f"Non-monotonic at {i}: {prev} -> {val}"
            prev = val


class TestSincosFunction:
    """Test combined sincos function."""

    def test_sincos_matches_individual(self):
        """sincos should match individual sin/cos calls."""
        from symcrash_python_ap.wavelet.sine_core import sine_q15, cosine_q15, sincos_q15

        test_phases = [0, 0x10000000, 0x40000000, 0x80000000, 0xC0000000]

        for phase in test_phases:
            sin_val, cos_val = sincos_q15(phase)
            expected_sin = sine_q15(phase)
            expected_cos = cosine_q15(phase)

            assert sin_val == expected_sin, f"sincos sin mismatch at phase {phase:08X}"
            assert cos_val == expected_cos, f"sincos cos mismatch at phase {phase:08X}"


class TestFrequencyConversion:
    """Test frequency to phase increment conversion."""

    def test_midi_to_freq_a4(self):
        """MIDI note 69 (A4) should be 440 Hz."""
        from symcrash_python_ap.wavelet.sine_core import midi_to_freq

        freq = midi_to_freq(69)
        assert abs(freq - 440.0) < 0.1, f"A4 = {freq:.1f} Hz, expected 440 Hz"

    def test_midi_to_freq_c4(self):
        """MIDI note 60 (C4) should be ~261.6 Hz."""
        from symcrash_python_ap.wavelet.sine_core import midi_to_freq

        freq = midi_to_freq(60)
        assert abs(freq - 261.6) < 0.5, f"C4 = {freq:.1f} Hz, expected 261.6 Hz"

    def test_freq_to_phase_inc_round_trip(self):
        """Phase increment should produce correct frequency."""
        from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, PHASE_BITS

        sample_rate = 48000
        test_freq = 440.0

        phase_inc = freq_to_phase_inc(test_freq, sample_rate)

        # Verify: freq = phase_inc * sample_rate / 2^32
        recovered_freq = phase_inc * sample_rate / (1 << 32)
        assert abs(recovered_freq - test_freq) < 0.01, \
            f"Recovered {recovered_freq:.2f} Hz, expected {test_freq} Hz"
