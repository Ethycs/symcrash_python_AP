"""
Wavelet Round-Trip Tests
=========================

Tests FWPT->IDWPT round-trip accuracy across different input signals.

The wavelet transform should be invertible with minimal error. These tests
verify that effect_mode (passthrough with mask=None) accurately reconstructs
the input signal.

Key metrics:
- Max error: Maximum absolute sample error
- RMS error: Root-mean-square error
- Headroom bits: Effect of internal scaling on accuracy
"""

import pytest
import numpy as np

from symcrash_python_ap.wavelet.fast_lifting import (
    effect_mode_jit,
    _get_jit_processor,
)

BLOCK_SIZE = 64
SAMPLE_RATE = 48000


def compute_roundtrip_error(samples, mask_q15=None):
    """
    Compute FWPT->IDWPT round-trip error.

    Args:
        samples: Input samples (list of int)
        mask_q15: Optional mask (None for passthrough)

    Returns:
        dict with max_err, rms_err, samples_in, samples_out
    """
    carrier_list = [int(s) for s in samples]
    wet = effect_mode_jit(carrier_list, mask_q15)

    wet_arr = np.array(wet, dtype=np.int32)
    samples_arr = np.array(samples, dtype=np.int32)

    diff = samples_arr - wet_arr
    max_err = np.max(np.abs(diff))
    rms_err = np.sqrt(np.mean(diff**2))

    return {
        "max_err": int(max_err),
        "rms_err": float(rms_err),
        "samples_in": samples_arr,
        "samples_out": wet_arr,
        "diff": diff,
    }


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================


def generate_pure_sine(freq=440.0, amplitude=32767, samples=64, sr=SAMPLE_RATE):
    """Generate pure sine wave centered at 0."""
    return [int(amplitude * np.sin(2 * np.pi * freq * i / sr)) for i in range(samples)]


def generate_offset_sine(freq=440.0, amplitude=32767, offset=5000, samples=64):
    """Generate sine wave with DC offset."""
    pure = generate_pure_sine(freq, amplitude, samples)
    return [s + offset for s in pure]


def generate_dc(value=10000, samples=64):
    """Generate pure DC signal."""
    return [value] * samples


def generate_actual_carrier():
    """Real carrier values from debug output."""
    return [
        0,
        1779,
        3553,
        5310,
        7042,
        8750,
        10424,
        12059,
        13646,
        15186,
        16663,
        18072,
        19413,
        20684,
        21875,
        22983,
        24005,
        24940,
        25783,
        26533,
        27188,
        27745,
        28201,
        28557,
        28812,
        28964,
        29013,
        28958,
        28800,
        28540,
        28179,
        27718,
        27159,
        26504,
        25757,
        24919,
        23996,
        22991,
        21907,
        20751,
        19527,
        18241,
        16897,
        15502,
        14060,
        12579,
        11063,
        9519,
        7954,
        6373,
        4782,
        3187,
        1594,
        8,
        -1566,
        -3124,
        -4663,
        -6180,
        -7670,
        -9132,
        -10561,
        -11955,
        -13309,
        -14622,
    ]


# =============================================================================
# TESTS
# =============================================================================


class TestPassthroughRoundtrip:
    """Test passthrough (mask=None) round-trip accuracy."""

    def test_pure_sine_roundtrip(self):
        """Pure sine should round-trip with minimal error."""
        samples = generate_pure_sine(freq=440.0, amplitude=32767)
        result = compute_roundtrip_error(samples, mask_q15=None)

        # Error should be very small (< 1 LSB on average)
        assert result["max_err"] <= 2, f"Max error too high: {result['max_err']}"
        assert result["rms_err"] < 1.0, f"RMS error too high: {result['rms_err']:.2f}"

    def test_offset_sine_roundtrip(self):
        """Sine with DC offset should round-trip accurately."""
        samples = generate_offset_sine(freq=440.0, amplitude=32767, offset=5000)
        result = compute_roundtrip_error(samples, mask_q15=None)

        # DC offset may introduce slightly higher error due to headroom
        assert result["max_err"] <= 5, f"Max error too high: {result['max_err']}"
        assert result["rms_err"] < 2.0, f"RMS error too high: {result['rms_err']:.2f}"

    def test_half_amplitude_roundtrip(self):
        """Half amplitude sine should round-trip well."""
        samples = generate_pure_sine(freq=440.0, amplitude=16384)
        result = compute_roundtrip_error(samples, mask_q15=None)

        assert result["max_err"] <= 2
        assert result["rms_err"] < 1.0

    def test_small_amplitude_roundtrip(self):
        """Small amplitude sine should round-trip."""
        samples = generate_pure_sine(freq=440.0, amplitude=1000)
        result = compute_roundtrip_error(samples, mask_q15=None)

        # Small signals may have proportionally higher relative error
        # but absolute error should still be < 1 LSB
        assert result["max_err"] <= 1
        assert result["rms_err"] < 0.5

    def test_dc_roundtrip(self):
        """Pure DC should round-trip exactly."""
        samples = generate_dc(value=10000)
        result = compute_roundtrip_error(samples, mask_q15=None)

        # DC should be preserved exactly (or very close)
        assert result["max_err"] <= 1
        assert result["rms_err"] < 0.5

    def test_zero_roundtrip(self):
        """All zeros should round-trip exactly."""
        samples = generate_dc(value=0)
        result = compute_roundtrip_error(samples, mask_q15=None)

        assert result["max_err"] == 0
        assert result["rms_err"] == 0.0

    def test_actual_carrier_roundtrip(self):
        """Real carrier values should round-trip accurately."""
        samples = generate_actual_carrier()
        result = compute_roundtrip_error(samples, mask_q15=None)

        assert result["max_err"] <= 2, f"Max error: {result['max_err']}"
        assert result["rms_err"] < 1.0, f"RMS error: {result['rms_err']:.2f}"


class TestUnityMaskRoundtrip:
    """Test round-trip with unity mask (all coefficients = 32767)."""

    def test_unity_mask_matches_passthrough(self):
        """Unity mask should match passthrough behavior."""
        samples = generate_pure_sine(freq=440.0, amplitude=32767)

        result_passthrough = compute_roundtrip_error(samples, mask_q15=None)
        result_unity = compute_roundtrip_error(samples, mask_q15=[32767] * 64)

        # Both should produce identical or very similar results
        diff = np.abs(result_passthrough["samples_out"] - result_unity["samples_out"])
        max_diff = np.max(diff)

        assert max_diff <= 1, f"Unity mask differs from passthrough by {max_diff}"


class TestHeadroomEffect:
    """Test effect of headroom bits on round-trip accuracy."""

    def test_headroom_bits_accuracy(self):
        """Different headroom values should maintain accuracy."""
        samples = generate_actual_carrier()
        processor = _get_jit_processor()

        errors = []
        for headroom in [0, 1, 2, 3, 4]:
            wet = processor.effect_mode(samples, None, headroom)
            wet_arr = np.array(wet, dtype=np.int32)
            samples_arr = np.array(samples, dtype=np.int32)
            diff = samples_arr - wet_arr
            max_err = np.max(np.abs(diff))
            rms_err = np.sqrt(np.mean(diff**2))
            errors.append((headroom, max_err, rms_err))

        # All headroom values should maintain good accuracy
        for headroom, max_err, rms_err in errors:
            assert max_err <= 4, (
                f"Headroom {headroom}: max_err={max_err} exceeds threshold"
            )
            assert rms_err < 2.0, (
                f"Headroom {headroom}: rms_err={rms_err:.2f} exceeds threshold"
            )

    def test_headroom_consistency(self):
        """Higher headroom should not degrade accuracy significantly."""
        samples = generate_pure_sine(freq=440.0, amplitude=20000)
        processor = _get_jit_processor()

        # Compare headroom=0 vs headroom=2
        wet_0 = processor.effect_mode(samples, None, 0)
        wet_2 = processor.effect_mode(samples, None, 2)

        err_0 = np.max(np.abs(np.array(samples) - np.array(wet_0)))
        err_2 = np.max(np.abs(np.array(samples) - np.array(wet_2)))

        # Both should be similarly accurate
        assert err_0 <= 2
        assert err_2 <= 2


class TestFrequencyDependence:
    """Test round-trip accuracy across frequencies."""

    @pytest.mark.parametrize("freq", [100.0, 440.0, 1000.0, 4000.0])
    def test_roundtrip_at_frequency(self, freq):
        """Round-trip should work well across frequencies."""
        samples = generate_pure_sine(freq=freq, amplitude=20000)
        result = compute_roundtrip_error(samples, mask_q15=None)

        assert result["max_err"] <= 2, (
            f"At {freq} Hz: max_err={result['max_err']}"
        )
        assert result["rms_err"] < 1.0, (
            f"At {freq} Hz: rms_err={result['rms_err']:.2f}"
        )


# =============================================================================
# UTILITY: Detailed analysis
# =============================================================================


@pytest.mark.manual
def test_analyze_roundtrip_detailed():
    """Detailed round-trip analysis (not run in CI)."""
    test_cases = [
        ("Pure 440Hz sine", generate_pure_sine(440.0, 32767)),
        ("440Hz + 5000 offset", generate_offset_sine(440.0, 32767, 5000)),
        ("Half amplitude", generate_pure_sine(440.0, 16384)),
        ("Small amplitude", generate_pure_sine(440.0, 1000)),
        ("Pure DC=10000", generate_dc(10000)),
        ("All zeros", generate_dc(0)),
        ("Actual carrier", generate_actual_carrier()),
    ]

    print("\n" + "=" * 70)
    print("DETAILED ROUND-TRIP ANALYSIS")
    print("=" * 70)

    for name, samples in test_cases:
        result = compute_roundtrip_error(samples, mask_q15=None)
        print(f"\n{name}:")
        print(f"  Input range: [{result['samples_in'].min()}, "
              f"{result['samples_in'].max()}]")
        print(f"  Max error: {result['max_err']}")
        print(f"  RMS error: {result['rms_err']:.2f}")

        # Show first few error values
        if result['max_err'] > 0:
            nonzero_errors = np.where(result['diff'] != 0)[0]
            if len(nonzero_errors) > 0:
                print(f"  Non-zero errors at indices: {nonzero_errors[:5].tolist()}")

    # Test headroom effect
    print("\n" + "=" * 70)
    print("HEADROOM EFFECT")
    print("=" * 70)

    samples = generate_actual_carrier()
    processor = _get_jit_processor()

    print(f"\n{'Headroom':<10} {'Max Error':<12} {'RMS Error':<12}")
    print("-" * 34)

    for headroom in [0, 1, 2, 3, 4]:
        wet = processor.effect_mode(samples, None, headroom)
        diff = np.array(samples) - np.array(wet)
        max_err = np.max(np.abs(diff))
        rms_err = np.sqrt(np.mean(diff**2))
        print(f"{headroom:<10} {max_err:<12} {rms_err:<12.2f}")
