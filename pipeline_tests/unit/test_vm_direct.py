"""
VM Direct Tests
===============

Tests VM execution directly, bypassing UnifiedOutputStage to isolate
VM behavior from safety processing.

Validates:
- Direct VM output quality
- ETA effect on VM output
- Block boundary continuity in VM (polyphase metrics)
- Spectral characteristics at VM level

Note: CCS path uses polyphase IDWPT which produces wavelet-domain synthesis.
Metrics are adapted for this architecture (boundary/intra ratio, not gradient).
"""

import pytest
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import windows

from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.frames import VoiceFrame

BLOCK_SIZE = 64
SAMPLE_RATE = 48000


def generate_vm_direct(beta_target_q15=0, num_blocks=200, gain_q15=32767):
    """Generate audio directly through VM."""
    frame = VoiceFrame(
        voice_id=0,
        note=60,
        cents=0,
        gain_q15=gain_q15,
        pan_q15=0,
        tree_a_id=0,
        tree_b_id=0,
        tree_blend_q15=0,
        is_active=True,
        beta_target_q15=beta_target_q15,
        partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    )

    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=60, cents=0)

    all_samples = []
    for _ in range(num_blocks):
        state, output = vm_step_block(state, frame, DEFAULT_TABLES)
        all_samples.extend(output.tolist())

    return np.array(all_samples, dtype=np.int16)


def compute_boundary_intra_ratio(samples):
    """
    Compute boundary/intra jump ratio for polyphase continuity.

    This is the correct metric for polyphase IDWPT synthesis.
    Ratio < 3.0 indicates proper phase continuity across blocks.
    """
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


def check_block_discontinuities(samples, threshold=100):
    """Check for discontinuities at block boundaries (gradient method)."""
    disc_count = 0
    for i in range(64, min(len(samples), 6400), 64):
        prev_d = samples[i - 1] - samples[i - 2]
        jump = samples[i] - samples[i - 1]
        if abs(jump - prev_d) > threshold:
            disc_count += 1
    return disc_count


def analyze_spectrum_vm(samples):
    """Analyze spectrum of VM output."""
    window = windows.blackman(len(samples))
    windowed = samples.astype(float) * window
    spectrum = np.abs(rfft(windowed))
    freqs = rfftfreq(len(samples), 1 / SAMPLE_RATE)

    # Find fundamental
    start = int(50 * len(freqs) / (SAMPLE_RATE / 2))
    fund_idx = start + np.argmax(spectrum[start : start + 2000])
    fund_mag = spectrum[fund_idx]
    fund_freq = freqs[fund_idx]

    # Find non-harmonic peaks
    non_harmonic = []
    for i, (f, m) in enumerate(zip(freqs, spectrum)):
        if f < 50 or m < fund_mag * 1e-3:
            continue
        is_harmonic = False
        for h in range(1, 20):
            expected = fund_freq * h
            if abs(f - expected) < expected * 0.02:
                is_harmonic = True
                break
        if not is_harmonic:
            db = 20 * np.log10(m / fund_mag + 1e-10)
            if db > -60:
                non_harmonic.append((f, db))

    return {
        "fundamental_freq": fund_freq,
        "fundamental_mag": fund_mag,
        "non_harmonic": non_harmonic,
        "is_pure": len(non_harmonic) == 0,
    }


class TestVMDirectOutput:
    """Test VM direct output quality."""

    def test_vm_dry_block_continuity(self):
        """VM dry output should have good block boundary continuity (polyphase metric)."""
        samples = generate_vm_direct(beta_target_q15=0, num_blocks=100)
        ratio = compute_boundary_intra_ratio(samples)

        # Polyphase IDWPT: boundary jumps should not exceed 3x intra-block jumps
        assert ratio < 3.0, f"Boundary/intra ratio too high: {ratio:.2f} (max 3.0)"

    def test_vm_dry_fundamental_present(self):
        """VM dry output should have a clear fundamental frequency."""
        samples = generate_vm_direct(beta_target_q15=0, num_blocks=200)
        result = analyze_spectrum_vm(samples)

        # Check fundamental is near expected (261.63 Hz for MIDI 60)
        expected_freq = 261.63
        actual_freq = result["fundamental_freq"]
        freq_error = abs(actual_freq - expected_freq)

        assert freq_error < 5.0, (
            f"Fundamental frequency {actual_freq:.1f} Hz too far from expected "
            f"{expected_freq:.1f} Hz (error: {freq_error:.1f} Hz)"
        )

    def test_vm_output_not_silent(self):
        """VM should produce non-silent output."""
        samples = generate_vm_direct(beta_target_q15=0, num_blocks=100)
        rms = np.sqrt(np.mean(samples.astype(float) ** 2))

        assert rms > 5000, f"RMS too low: {rms:.1f}"


class TestVMETAEffect:
    """Test ETA effect on VM output.

    Note: In CCS path, ETA (beta_target_q15) affects the wet/dry crossfade.
    The CCS carrier itself is independent of ETA - only the blend changes.
    """

    @pytest.mark.parametrize("beta_q15", [0, 8192, 16384, 24576, 32767])
    def test_vm_at_eta_value(self, beta_q15):
        """VM should work at various ETA values."""
        samples = generate_vm_direct(beta_target_q15=beta_q15, num_blocks=100)
        rms = np.sqrt(np.mean(samples.astype(float) ** 2))

        assert rms > 1000, f"At ETA={beta_q15}: RMS too low ({rms:.1f})"


class TestVMWetPath:
    """Test VM wet path quality (polyphase IDWPT)."""

    def test_vm_wet_has_fundamental(self):
        """VM wet output should have a clear fundamental frequency."""
        samples = generate_vm_direct(beta_target_q15=32767, num_blocks=200)
        result = analyze_spectrum_vm(samples)

        # Check fundamental is near expected (261.63 Hz for MIDI 60)
        expected_freq = 261.63
        actual_freq = result["fundamental_freq"]
        freq_error = abs(actual_freq - expected_freq)

        assert freq_error < 5.0, (
            f"Fundamental frequency {actual_freq:.1f} Hz too far from expected "
            f"{expected_freq:.1f} Hz (error: {freq_error:.1f} Hz)"
        )

    def test_vm_wet_block_continuity(self):
        """VM wet output should have good block boundary continuity."""
        samples = generate_vm_direct(beta_target_q15=32767, num_blocks=100)
        ratio = compute_boundary_intra_ratio(samples)

        # Polyphase IDWPT: boundary jumps should not exceed 3x intra-block jumps
        assert ratio < 3.0, f"Boundary/intra ratio too high: {ratio:.2f} (max 3.0)"
