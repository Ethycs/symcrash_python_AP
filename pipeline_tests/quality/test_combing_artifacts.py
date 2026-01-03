"""
Combing Artifact Detection Tests
=================================

Tests for comb filtering artifacts in wet path and mixed signals.

Combing artifacts appear as non-harmonic frequency content, typically
caused by phase misalignment or improper mixing of carrier and wet paths.

Detection methods:
- Non-harmonic peak detection in FFT
- Sample-to-sample discontinuity analysis
- Block boundary continuity checks
"""

import pytest
import numpy as np
from scipy import signal as sp_signal

from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import write_audio

SAMPLE_RATE = 48000
BLOCK_SIZE = 64


def generate_vm_audio(
    f0_hz=440.0,
    gain_q15=24000,
    beta_target_q15=0,
    num_blocks=200,
    warmup_blocks=10,
):
    """
    Generate audio using VM with specified wet amount.

    Args:
        f0_hz: Fundamental frequency
        gain_q15: Output gain
        beta_target_q15: Wet amount (0=dry, 32767=full wet)
        num_blocks: Total blocks to render
        warmup_blocks: Blocks to skip for warmup

    Returns:
        np.ndarray: Rendered samples (after warmup)
    """
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(f0_hz=f0_hz)

    frame = VoiceFrame(
        voice_id=0,
        note=69,
        f0_hz=f0_hz,
        gain_q15=gain_q15,
        beta_target_q15=beta_target_q15,
        partial_amps_q15=[32767] + [0] * 15,  # Pure sine input
        is_active=True,
    )

    all_samples = []
    for _ in range(num_blocks):
        output = vm_step_block(state, frame, DEFAULT_TABLES)
        all_samples.extend(output.tolist())

    samples = np.array(all_samples, dtype=np.int16)
    return samples[warmup_blocks * BLOCK_SIZE :]


def analyze_non_harmonic_content(samples, f0_hz, threshold_db=-60, sr=SAMPLE_RATE):
    """
    Detect non-harmonic peaks in spectrum.

    Args:
        samples: Audio samples
        f0_hz: Expected fundamental frequency
        threshold_db: Detection threshold relative to fundamental
        sr: Sample rate

    Returns:
        dict with:
            - fundamental_freq: Detected fundamental (Hz)
            - fundamental_db: Fundamental magnitude (dB)
            - non_harmonic_peaks: List of (freq, db_relative)
            - is_pure: True if no non-harmonic content found
    """
    # Compute FFT
    n = len(samples)
    fft = np.fft.rfft(samples.astype(np.float64))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    magnitude_db = 20 * np.log10(np.abs(fft) + 1e-10)

    # Find fundamental
    fund_search = np.where((freqs > f0_hz * 0.9) & (freqs < f0_hz * 1.1))[0]
    fund_idx = fund_search[np.argmax(magnitude_db[fund_search])]
    fund_freq = freqs[fund_idx]
    fund_db = magnitude_db[fund_idx]

    # Detection threshold
    detection_threshold = fund_db + threshold_db

    # Find non-harmonic peaks
    non_harmonic_peaks = []

    for i, (f, db) in enumerate(zip(freqs, magnitude_db)):
        if db < detection_threshold:
            continue
        if f < 50 or f > 15000:  # Skip DC and very high frequencies
            continue

        # Check if it's a harmonic (within 2% of expected harmonic frequency)
        is_harmonic = False
        for h in range(1, 30):
            expected = fund_freq * h
            if abs(f - expected) < expected * 0.02:
                is_harmonic = True
                break

        if not is_harmonic:
            non_harmonic_peaks.append((f, db - fund_db))

    is_pure = len(non_harmonic_peaks) == 0

    return {
        "fundamental_freq": fund_freq,
        "fundamental_db": fund_db,
        "non_harmonic_peaks": non_harmonic_peaks,
        "is_pure": is_pure,
    }


def detect_sample_discontinuities(samples, threshold=1000, warmup_samples=640):
    """
    Detect large sample-to-sample jumps.

    Args:
        samples: Audio samples
        threshold: Threshold for discontinuity detection (LSB)
        warmup_samples: Samples to skip before analysis

    Returns:
        dict with:
            - discontinuity_indices: List of sample indices
            - block_boundary_count: Count at block boundaries
            - total_count: Total discontinuities found
    """
    # Skip warmup
    samples = samples[warmup_samples:]

    diffs = np.abs(np.diff(samples.astype(np.int32)))
    disc_indices = np.where(diffs > threshold)[0]

    # Check if at block boundaries
    block_boundary_count = sum(1 for i in disc_indices if i % BLOCK_SIZE < 3)

    return {
        "discontinuity_indices": disc_indices.tolist(),
        "block_boundary_count": block_boundary_count,
        "total_count": len(disc_indices),
    }


# =============================================================================
# TESTS
# =============================================================================


class TestDrySignalPurity:
    """Test dry (carrier only) signal has no combing."""

    def test_dry_no_non_harmonic_peaks(self):
        """Dry signal should have no non-harmonic content."""
        samples = generate_vm_audio(f0_hz=440.0, beta_target_q15=0, num_blocks=200)
        result = analyze_non_harmonic_content(samples, f0_hz=440.0, threshold_db=-60)

        assert result["is_pure"], (
            f"Dry signal has {len(result['non_harmonic_peaks'])} non-harmonic peaks: "
            f"{result['non_harmonic_peaks'][:5]}"
        )

    def test_dry_no_discontinuities(self):
        """Dry signal should have no discontinuities."""
        samples = generate_vm_audio(f0_hz=440.0, beta_target_q15=0, num_blocks=200)
        result = detect_sample_discontinuities(samples, threshold=1000)

        assert result["total_count"] == 0, (
            f"Dry signal has {result['total_count']} discontinuities"
        )


class TestWetSignalQuality:
    """Test wet (IDWPT) signal quality."""

    def test_full_wet_limited_artifacts(self):
        """Full wet should have limited non-harmonic content."""
        samples = generate_vm_audio(
            f0_hz=440.0, beta_target_q15=32767, num_blocks=200
        )
        result = analyze_non_harmonic_content(samples, f0_hz=440.0, threshold_db=-60)

        # Wet path may have some artifacts but should be bounded
        assert len(result["non_harmonic_peaks"]) <= 5, (
            f"Full wet has {len(result['non_harmonic_peaks'])} non-harmonic peaks, "
            f"exceeds limit of 5"
        )

    def test_full_wet_controlled_discontinuities(self):
        """Full wet discontinuities should be controlled."""
        samples = generate_vm_audio(
            f0_hz=440.0, beta_target_q15=32767, num_blocks=200
        )
        result = detect_sample_discontinuities(samples, threshold=1500)

        # Some discontinuities expected due to IDWPT but should be limited
        assert result["total_count"] <= 10, (
            f"Full wet has {result['total_count']} discontinuities (>1500 LSB)"
        )


class TestMixedSignalQuality:
    """Test carrier+wet mixed signals."""

    @pytest.mark.parametrize(
        "beta_q15,max_peaks",
        [
            (8192, 3),  # 25% wet - allow up to 3 peaks
            (16384, 4),  # 50% wet - allow up to 4 peaks
            (24576, 5),  # 75% wet - allow up to 5 peaks
        ],
    )
    def test_mixed_signal_purity(self, beta_q15, max_peaks):
        """Test mixed signal has limited artifacts."""
        samples = generate_vm_audio(
            f0_hz=440.0, beta_target_q15=beta_q15, num_blocks=200
        )
        result = analyze_non_harmonic_content(samples, f0_hz=440.0, threshold_db=-60)

        pct = int(beta_q15 / 32767 * 100)
        assert len(result["non_harmonic_peaks"]) <= max_peaks, (
            f"At {pct}% wet: {len(result['non_harmonic_peaks'])} non-harmonic peaks "
            f"exceeds limit of {max_peaks}"
        )


class TestFrequencyDependence:
    """Test combing across different frequencies."""

    @pytest.mark.parametrize("f0_hz", [220.0, 440.0, 880.0])
    def test_dry_purity_at_frequency(self, f0_hz):
        """Dry signal should be pure across frequencies."""
        samples = generate_vm_audio(f0_hz=f0_hz, beta_target_q15=0, num_blocks=200)
        result = analyze_non_harmonic_content(samples, f0_hz=f0_hz, threshold_db=-60)

        assert result["is_pure"], (
            f"At {f0_hz} Hz: Found non-harmonic peaks {result['non_harmonic_peaks']}"
        )


# =============================================================================
# UTILITY: Generate test audio for manual inspection
# =============================================================================


@pytest.mark.manual
def test_generate_combing_test_files():
    """Generate test files for manual combing inspection (not run in CI)."""
    configs = [
        ("dry", 0, "Dry (carrier only)"),
        ("wet_25pct", 8192, "25% wet"),
        ("wet_50pct", 16384, "50% wet"),
        ("wet_75pct", 24576, "75% wet"),
        ("wet_100pct", 32767, "100% wet"),
    ]

    print("\n" + "=" * 70)
    print("COMBING ARTIFACT ANALYSIS")
    print("=" * 70)

    for name, beta_q15, label in configs:
        samples = generate_vm_audio(
            f0_hz=440.0, beta_target_q15=beta_q15, num_blocks=200
        )

        meta = write_audio(
            f"combing_test_{name}",
            samples,
            timestamp=True,
            subdirectory="combing_tests",
        )

        # Analyze
        spectral = analyze_non_harmonic_content(samples, f0_hz=440.0, threshold_db=-60)
        discontinuity = detect_sample_discontinuities(samples, threshold=1000)

        print(f"\n{label}: {meta.filepath}")
        print(f"  Fundamental: {spectral['fundamental_freq']:.1f} Hz")
        print(f"  Non-harmonic peaks: {len(spectral['non_harmonic_peaks'])}")
        if spectral["non_harmonic_peaks"]:
            for freq, db in spectral["non_harmonic_peaks"][:3]:
                print(f"    {freq:.1f} Hz: {db:+.1f} dB")
        print(f"  Discontinuities: {discontinuity['total_count']}")
        print(f"  At block boundaries: {discontinuity['block_boundary_count']}")
