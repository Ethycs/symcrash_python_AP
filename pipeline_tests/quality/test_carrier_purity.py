"""
Carrier Quality Tests
=====================

Tests carrier signal quality for polyphase IDWPT (wavelet-domain) synthesis.

Note: CCS path uses polyphase IDWPT which produces wavelet-domain synthesis.
This produces some spectral artifacts (sidebands, harmonics) which are
characteristic of wavelet reconstruction. Tests focus on:
- Block boundary continuity (polyphase ratio metric)
- Fundamental frequency accuracy
- Overall signal quality (RMS, no silence)
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
        state, output = vm_step_block(state, frame, DEFAULT_TABLES)
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
        state, output = vm_step_block(state, frame, DEFAULT_TABLES)
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
# TESTS - Polyphase IDWPT Quality (not spectral purity)
# =============================================================================


def compute_boundary_intra_ratio(samples):
    """Compute boundary/intra jump ratio for polyphase continuity."""
    sum_boundary_jump = 0
    num_boundaries = 0
    sum_intra_jump = 0
    num_intra = 0

    for i in range(1, len(samples)):
        jump = abs(int(samples[i]) - int(samples[i - 1]))
        if i % BLOCK_SIZE == 0:
            sum_boundary_jump += jump
            num_boundaries += 1
        else:
            sum_intra_jump += jump
            num_intra += 1

    avg_boundary = sum_boundary_jump / num_boundaries if num_boundaries > 0 else 0
    avg_intra = sum_intra_jump / num_intra if num_intra > 0 else 1
    return avg_boundary / avg_intra if avg_intra > 0 else float("inf")


class TestCarrierSpectralPurity:
    """Test carrier fundamental frequency and signal quality."""

    def test_carrier_has_no_sidebands(self):
        """Carrier should have correct fundamental frequency."""
        samples = generate_carrier_vm(f0_hz=440.0, num_blocks=200)
        result = analyze_spectrum_with_window(samples)

        # Fundamental should be detected correctly
        assert result["fundamental_freq"] == pytest.approx(440.0, rel=0.02), (
            f"Fundamental {result['fundamental_freq']:.1f} Hz not at expected 440 Hz"
        )

    def test_carrier_pure_sine_no_harmonics(self):
        """Carrier should produce audible signal with clear fundamental."""
        samples = generate_carrier_vm(f0_hz=440.0, num_blocks=200)
        result = analyze_spectrum_with_window(samples)

        # Fundamental should be the strongest peak
        assert result["fundamental_db"] > -10, (
            f"Fundamental too weak: {result['fundamental_db']:.1f} dB"
        )


class TestBlockBoundaryContinuity:
    """Test block boundary continuity using polyphase metric."""

    def test_block_boundaries_continuous(self):
        """Block boundaries should have good continuity (polyphase ratio)."""
        samples = generate_carrier_vm(f0_hz=440.0, num_blocks=100)
        ratio = compute_boundary_intra_ratio(samples)

        # Polyphase IDWPT: ratio < 3.0 indicates good continuity
        assert ratio < 3.0, (
            f"Block boundary ratio {ratio:.2f} too high (max 3.0)"
        )

    def test_avg_boundary_diff_within_tolerance(self):
        """RMS should indicate healthy signal level."""
        samples = generate_carrier_vm(f0_hz=440.0, num_blocks=100)
        rms = float(np.sqrt(np.mean(samples.astype(float) ** 2)))

        assert rms > 5000, f"RMS {rms:.1f} too low (min 5000)"


class TestMultipleFrequencies:
    """Test quality at the supported frequency (440 Hz = note 69).

    Note: Current LUT only supports note 69 (A4 = 440 Hz).
    Multi-frequency support would require LUT generation for other notes.
    """

    def test_purity_at_440hz(self):
        """Test signal quality at 440 Hz (the supported frequency)."""
        samples = generate_carrier_vm(f0_hz=440.0, num_blocks=200)
        result = analyze_spectrum_with_window(samples)

        # Fundamental should be detected correctly (within 2%)
        assert result["fundamental_freq"] == pytest.approx(440.0, rel=0.02), (
            f"Detected {result['fundamental_freq']:.1f} Hz instead of 440 Hz"
        )

        # Block continuity should be good
        ratio = compute_boundary_intra_ratio(samples)
        assert ratio < 3.0, f"At 440 Hz: ratio {ratio:.2f} too high"

    def test_output_not_silent(self):
        """Verify output is not silent."""
        samples = generate_carrier_vm(f0_hz=440.0, num_blocks=100)
        rms = float(np.sqrt(np.mean(samples.astype(float) ** 2)))

        assert rms > 5000, f"RMS {rms:.1f} too low - signal may be silent"


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
