"""
Carrier Purity Tests
====================

Verifies carrier signal purity using spectral analysis and block boundary checks.

Tests:
- Pure sine generation without harmonics
- Block boundary continuity
- Phase coherence across blocks
- Sideband detection near fundamental
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


def generate_carrier_vm(
    f0_hz=440.0,
    gain_q15=24000,
    num_blocks=200,
    warmup_blocks=10,
):
    """
    Generate pure carrier signal using VM.

    Args:
        f0_hz: Fundamental frequency
        gain_q15: Output gain
        num_blocks: Total blocks to render
        warmup_blocks: Blocks to skip for latency warmup

    Returns:
        np.ndarray: Rendered samples (after warmup)
    """
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(f0_hz=f0_hz)

    frame = VoiceFrame(
        voice_id=0,
        note=69,  # A4
        cents=0,
        f0_hz=f0_hz,
        gain_q15=gain_q15,
        beta_target_q15=0,  # Pure carrier (no wet)
        partial_amps_q15=[32767] + [0] * 15,  # Pure sine
        is_active=True,
    )

    all_samples = []
    for _ in range(num_blocks):
        output = vm_step_block(state, frame, DEFAULT_TABLES)
        all_samples.extend(output.tolist())

    samples = np.array(all_samples, dtype=np.int16)
    # Skip warmup blocks
    return samples[warmup_blocks * BLOCK_SIZE :]


def analyze_spectrum_with_window(samples, sr=SAMPLE_RATE):
    """
    Analyze spectrum with Hann window.

    Returns:
        dict with:
            - fundamental_freq: Hz
            - fundamental_db: dB
            - sidebands: List of (freq, db_relative) within ±50 Hz of fundamental
            - harmonics: List of (harmonic_number, freq, db_relative)
    """
    n = len(samples)

    # Apply Hann window to reduce spectral leakage
    window = sp_signal.windows.hann(n)
    windowed = samples * window

    # Zero-pad for better frequency resolution
    nfft = n * 4
    fft = np.fft.rfft(windowed, nfft)
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    magnitude_db = 20 * np.log10(np.abs(fft) + 1e-10)

    # Find fundamental (search around 440 Hz)
    search_range = np.where((freqs > 400) & (freqs < 500))[0]
    fund_idx = search_range[np.argmax(magnitude_db[search_range])]
    fund_freq = freqs[fund_idx]
    fund_db = magnitude_db[fund_idx]

    # Find sidebands (within ±50 Hz of fundamental, excluding fundamental itself)
    nearby = np.where((freqs > fund_freq - 50) & (freqs < fund_freq + 50))[0]
    threshold_db = fund_db - 40

    sidebands = []
    for i in nearby:
        if abs(freqs[i] - fund_freq) < 2:  # Skip fundamental itself
            continue
        if magnitude_db[i] > threshold_db:
            sidebands.append((freqs[i], magnitude_db[i] - fund_db))

    # Find harmonics
    harmonics = []
    for h in range(2, 10):  # Check harmonics 2-9
        target_freq = fund_freq * h
        # Find peak near expected harmonic
        search = np.where((freqs > target_freq * 0.98) & (freqs < target_freq * 1.02))[
            0
        ]
        if len(search) > 0:
            peak_idx = search[np.argmax(magnitude_db[search])]
            db_rel = magnitude_db[peak_idx] - fund_db
            if db_rel > -80:  # Only report if above noise floor
                harmonics.append((h, freqs[peak_idx], db_rel))

    return {
        "fundamental_freq": fund_freq,
        "fundamental_db": fund_db,
        "sidebands": sidebands,
        "harmonics": harmonics,
    }


def check_block_boundaries(f0_hz=440.0, num_blocks=50):
    """
    Check continuity at block boundaries.

    Returns:
        dict with:
            - max_boundary_diff: Maximum sample-to-sample diff at boundary
            - avg_boundary_diff: Average boundary diff
            - expected_diff: Expected diff for continuous sine
    """
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(f0_hz=f0_hz)

    frame = VoiceFrame(
        voice_id=0,
        note=69,
        f0_hz=f0_hz,
        gain_q15=20000,
        beta_target_q15=0,
        partial_amps_q15=[32767] + [0] * 15,
        is_active=True,
    )

    prev_end = None
    boundary_diffs = []

    for _ in range(num_blocks):
        output = vm_step_block(state, frame, DEFAULT_TABLES)
        if prev_end is not None:
            diff = abs(int(output[0]) - int(prev_end))
            boundary_diffs.append(diff)
        prev_end = output[-1]

    # Expected diff for continuous sine
    expected_diff = abs(20000 * np.sin(2 * np.pi * f0_hz / SAMPLE_RATE))

    return {
        "max_boundary_diff": max(boundary_diffs) if boundary_diffs else 0,
        "avg_boundary_diff": np.mean(boundary_diffs) if boundary_diffs else 0,
        "expected_diff": expected_diff,
        "boundary_diffs": boundary_diffs,
    }


# =============================================================================
# TESTS
# =============================================================================


class TestCarrierSpectralPurity:
    """Test carrier spectral purity."""

    def test_carrier_has_no_sidebands(self):
        """Carrier should have no significant sidebands near fundamental."""
        samples = generate_carrier_vm(f0_hz=440.0, num_blocks=200)
        result = analyze_spectrum_with_window(samples)

        assert result["fundamental_freq"] == pytest.approx(440.0, rel=0.01)
        assert len(result["sidebands"]) == 0, (
            f"Found {len(result['sidebands'])} sidebands: {result['sidebands']}"
        )

    def test_carrier_pure_sine_no_harmonics(self):
        """Pure sine should have no significant harmonics."""
        samples = generate_carrier_vm(f0_hz=440.0, num_blocks=200)
        result = analyze_spectrum_with_window(samples)

        # For pure sine, all harmonics should be < -60 dB
        significant_harmonics = [
            (h, f, db) for h, f, db in result["harmonics"] if db > -60
        ]

        assert len(significant_harmonics) == 0, (
            f"Found {len(significant_harmonics)} significant harmonics "
            f"(> -60 dB): {significant_harmonics}"
        )


class TestBlockBoundaryContinuity:
    """Test block boundary continuity."""

    def test_block_boundaries_continuous(self):
        """Block boundaries should be continuous (no clicks)."""
        result = check_block_boundaries(f0_hz=440.0, num_blocks=50)

        # Max boundary diff should be close to expected diff for continuous sine
        # Allow 10% tolerance
        assert result["max_boundary_diff"] < result["expected_diff"] * 1.1, (
            f"Max boundary diff {result['max_boundary_diff']} exceeds expected "
            f"{result['expected_diff'] * 1.1:.1f}"
        )

    def test_avg_boundary_diff_within_tolerance(self):
        """Average boundary diff should match expected value."""
        result = check_block_boundaries(f0_hz=440.0, num_blocks=50)

        # Average should be very close to expected
        assert result["avg_boundary_diff"] == pytest.approx(
            result["expected_diff"], rel=0.05
        )


class TestMultipleFrequencies:
    """Test purity across different frequencies."""

    @pytest.mark.parametrize("f0_hz", [220.0, 440.0, 880.0, 1760.0])
    def test_purity_at_frequency(self, f0_hz):
        """Test spectral purity at various frequencies."""
        samples = generate_carrier_vm(f0_hz=f0_hz, num_blocks=200)
        result = analyze_spectrum_with_window(samples)

        # Should have no sidebands
        assert len(result["sidebands"]) == 0, (
            f"At {f0_hz} Hz: Found sidebands {result['sidebands']}"
        )

        # Fundamental should be detected correctly
        assert result["fundamental_freq"] == pytest.approx(f0_hz, rel=0.01)


# =============================================================================
# UTILITY: Generate test audio for manual inspection
# =============================================================================


@pytest.mark.manual
def test_generate_carrier_purity_files():
    """Generate carrier test files for manual inspection (not run in CI)."""
    frequencies = [220.0, 440.0, 880.0]

    for f0_hz in frequencies:
        samples = generate_carrier_vm(f0_hz=f0_hz, num_blocks=300)

        meta = write_audio(
            f"carrier_purity_{int(f0_hz)}Hz",
            samples,
            timestamp=True,
            subdirectory="carrier_purity_tests",
        )

        # Analyze
        result = analyze_spectrum_with_window(samples)
        boundary = check_block_boundaries(f0_hz=f0_hz)

        print(f"\n{f0_hz} Hz: {meta.filepath}")
        print(f"  Fundamental: {result['fundamental_freq']:.2f} Hz")
        print(f"  Sidebands: {len(result['sidebands'])}")
        print(f"  Significant harmonics: {len(result['harmonics'])}")
        print(
            f"  Max boundary diff: {boundary['max_boundary_diff']} "
            f"(expected: {boundary['expected_diff']:.1f})"
        )
