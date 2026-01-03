"""
Spectrum Analysis Tests
========================

FFT-based verification of frequency domain purity and harmonic content.

This module provides:
1. Full-pipeline spectral analysis with windowing
2. Harmonic purity verification
3. Non-harmonic artifact detection
4. Noise floor measurement
"""

import pytest
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import windows

from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import write_audio

Q15_ONE = 32767
BLOCK_SIZE = 64
SAMPLE_RATE = 48000


def render_pipeline(
    note=60,
    beta_target_q15=0,
    duration_blocks=1500,
    gain_q15=24576,
    partial_amps_q15=None,
):
    """
    Render audio through the full pipeline.

    Args:
        note: MIDI note number
        beta_target_q15: Wet amount (0=dry, 32767=100% wet)
        duration_blocks: Number of blocks to render
        gain_q15: Output gain in Q15
        partial_amps_q15: Harmonic amplitudes (default: pure sine)

    Returns:
        np.ndarray: Rendered samples (mono, int16)
    """
    if partial_amps_q15 is None:
        partial_amps_q15 = [32767, 0, 0, 0, 0, 0, 0, 0]

    stage = UnifiedOutputStage(block_size=BLOCK_SIZE)
    f0_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))

    frame = VoiceFrame(
        voice_id=0,
        note=note,
        cents=0,
        f0_hz=f0_hz,
        gain_q15=gain_q15,
        pan_q15=0,
        tree_a_id=0,
        tree_b_id=0,
        tree_blend_q15=0,
        is_active=True,
        beta_target_q15=beta_target_q15,
        partial_amps_q15=partial_amps_q15,
    )

    stage.voice_on(0, frame.note, frame.cents)

    all_samples = []
    for _ in range(duration_blocks):
        left, right = stage.render_block(frame)
        all_samples.extend(left)

    return np.array(all_samples, dtype=np.int16)


def analyze_spectrum(
    samples,
    use_window=True,
    threshold_db=-60,
    deviation_threshold=0.005,
):
    """
    Analyze spectrum with proper windowing.

    Args:
        samples: Audio samples (int16 or float)
        use_window: Whether to apply Blackman window
        threshold_db: Threshold for peak detection (dB relative to fundamental)
        deviation_threshold: Threshold for non-harmonic detection (0.005 = 0.5%)

    Returns:
        dict with keys:
            - fundamental_freq: Fundamental frequency (Hz)
            - fundamental_mag: Fundamental magnitude
            - harmonics: List of (freq, db, harmonic_number)
            - non_harmonic: List of (freq, db, ratio, deviation)
            - noise_floor_db: Noise floor in dB
            - is_pure: Boolean indicating purity
    """
    samples_float = samples.astype(np.float64)

    # Apply window to reduce spectral leakage
    if use_window:
        window = windows.blackman(len(samples))
        windowed = samples_float * window
        window_gain = np.sum(window) / len(window)
    else:
        windowed = samples_float
        window_gain = 1.0

    # Compute spectrum
    spectrum = np.abs(rfft(windowed)) / window_gain
    freqs = rfftfreq(len(samples), 1 / SAMPLE_RATE)

    # Find fundamental (skip DC and very low frequencies)
    start_idx = int(50 * len(freqs) / (SAMPLE_RATE / 2))
    fund_idx = start_idx + np.argmax(spectrum[start_idx : start_idx + 2000])
    fund_freq = freqs[fund_idx]
    fund_mag = spectrum[fund_idx]

    # Find peaks above threshold
    threshold_mag = fund_mag * (10 ** (threshold_db / 20))
    peak_indices = []
    for i in range(10, len(spectrum) - 10):
        if spectrum[i] > threshold_mag:
            if spectrum[i] > spectrum[i - 1] and spectrum[i] > spectrum[i + 1]:
                peak_indices.append(i)

    # Sort by magnitude
    peak_indices.sort(key=lambda i: spectrum[i], reverse=True)

    # Classify peaks as harmonic or non-harmonic
    harmonics = []
    non_harmonic = []

    for idx in peak_indices[:20]:
        freq = freqs[idx]
        if freq < 50:
            continue

        mag = spectrum[idx]
        rel_db = 20 * np.log10(mag / fund_mag + 1e-10)
        harmonic_ratio = freq / fund_freq
        nearest_harmonic = round(harmonic_ratio)

        if nearest_harmonic > 0:
            deviation = abs(harmonic_ratio - nearest_harmonic) / nearest_harmonic

            if deviation <= deviation_threshold:
                # This is a harmonic
                harmonics.append((freq, rel_db, nearest_harmonic))
            else:
                # This is non-harmonic
                non_harmonic.append((freq, rel_db, harmonic_ratio, deviation))

    # Measure noise floor (bins far from harmonics)
    noise_bins = []
    for i in range(len(spectrum)):
        freq = freqs[i]
        if freq < 50 or freq > 20000:
            continue

        # Check if far from any harmonic
        is_harmonic = False
        for h in range(1, 30):
            if abs(freq - fund_freq * h) < 10:
                is_harmonic = True
                break

        if not is_harmonic:
            noise_bins.append(spectrum[i])

    if noise_bins:
        noise_floor = np.mean(noise_bins)
        noise_floor_db = 20 * np.log10(noise_floor / fund_mag + 1e-10)
    else:
        noise_floor_db = -120.0

    is_pure = len(non_harmonic) == 0

    return {
        "fundamental_freq": fund_freq,
        "fundamental_mag": fund_mag,
        "harmonics": harmonics,
        "non_harmonic": non_harmonic,
        "noise_floor_db": noise_floor_db,
        "is_pure": is_pure,
        "spectrum": spectrum,
        "freqs": freqs,
    }


# =============================================================================
# TESTS
# =============================================================================


class TestCarrierPurity:
    """Test carrier (dry) signal purity."""

    def test_pure_carrier_sine(self):
        """Pure carrier should have no non-harmonic content."""
        samples = render_pipeline(note=60, beta_target_q15=0, duration_blocks=1500)
        result = analyze_spectrum(samples, use_window=True)

        assert result["is_pure"], (
            f"Carrier has non-harmonic content: {result['non_harmonic']}"
        )
        assert result["fundamental_freq"] == pytest.approx(261.63, rel=0.01)
        assert result["noise_floor_db"] < -60, (
            f"Noise floor too high: {result['noise_floor_db']:.1f} dB"
        )

    def test_carrier_harmonic_decay(self):
        """For pure sine, only fundamental should be present."""
        samples = render_pipeline(note=69, beta_target_q15=0)  # A4 = 440 Hz
        result = analyze_spectrum(samples)

        # Check that 2nd harmonic is at least 60 dB down
        h2_db = None
        for freq, db, h in result["harmonics"]:
            if h == 2:
                h2_db = db
                break

        # Pure sine should have no significant 2nd harmonic
        assert h2_db is None or h2_db < -60, (
            f"2nd harmonic too strong: {h2_db:.1f} dB"
        )


class TestWetPathPurity:
    """Test wet (IDWPT) path purity."""

    def test_wet_path_25_percent(self):
        """25% wet should maintain purity."""
        samples = render_pipeline(note=60, beta_target_q15=8192, duration_blocks=1500)
        result = analyze_spectrum(samples)

        # Allow some tolerance for IDWPT artifacts
        assert len(result["non_harmonic"]) <= 2, (
            f"Too many non-harmonic peaks at 25% wet: {result['non_harmonic']}"
        )

    def test_wet_path_100_percent(self):
        """100% wet should still be relatively pure."""
        samples = render_pipeline(note=60, beta_target_q15=32767, duration_blocks=1500)
        result = analyze_spectrum(samples)

        # IDWPT may introduce some artifacts but should be controlled
        assert len(result["non_harmonic"]) <= 5, (
            f"Too many non-harmonic peaks at 100% wet: {result['non_harmonic']}"
        )
        assert result["noise_floor_db"] < -40, (
            f"Noise floor too high at 100% wet: {result['noise_floor_db']:.1f} dB"
        )


class TestMixedSignals:
    """Test carrier+wet mix."""

    def test_50_50_mix(self):
        """50/50 carrier+wet mix."""
        samples = render_pipeline(note=60, beta_target_q15=16384, duration_blocks=1500)
        result = analyze_spectrum(samples)

        # Mixed signals may have more artifacts but should be bounded
        assert len(result["non_harmonic"]) <= 3, (
            f"Too many artifacts in 50/50 mix: {result['non_harmonic']}"
        )


class TestWindowingEffect:
    """Test effect of windowing on spectrum analysis."""

    def test_windowing_reduces_leakage(self):
        """Windowing should reduce spectral leakage."""
        samples = render_pipeline(note=60, beta_target_q15=0, duration_blocks=1000)

        result_no_window = analyze_spectrum(samples, use_window=False)
        result_with_window = analyze_spectrum(samples, use_window=True)

        # Windowing should reduce non-harmonic content
        assert len(result_with_window["non_harmonic"]) <= len(
            result_no_window["non_harmonic"]
        )


# =============================================================================
# UTILITY: Generate test audio for manual inspection
# =============================================================================


@pytest.mark.manual
def test_generate_spectrum_test_files():
    """Generate test files for manual spectrum inspection (not run in CI)."""
    configs = [
        ("carrier_pure", 0, "Pure carrier (0% wet)"),
        ("mix_25pct", 8192, "25% wet"),
        ("mix_50pct", 16384, "50% wet"),
        ("mix_75pct", 24576, "75% wet"),
        ("wet_100pct", 32767, "100% wet"),
    ]

    for name, beta, label in configs:
        samples = render_pipeline(note=60, beta_target_q15=beta, duration_blocks=1500)
        meta = write_audio(
            f"spectrum_test_{name}",
            samples,
            timestamp=True,
            subdirectory="spectral_tests",
        )
        print(f"Generated {label}: {meta.filepath}")

        # Analyze and print
        result = analyze_spectrum(samples)
        print(f"  Fundamental: {result['fundamental_freq']:.2f} Hz")
        print(f"  Non-harmonic peaks: {len(result['non_harmonic'])}")
        print(f"  Noise floor: {result['noise_floor_db']:.1f} dB")
        print()
