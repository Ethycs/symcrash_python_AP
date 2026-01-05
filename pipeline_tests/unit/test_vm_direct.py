"""
VM Direct Tests
===============

Tests VM execution directly, bypassing UnifiedOutputStage to isolate
VM behavior from safety processing.

Validates:
- Direct VM output quality
- ETA effect on VM output
- Block boundary continuity in VM
- Spectral purity at VM level
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
        output = vm_step_block(state, frame, DEFAULT_TABLES)
        all_samples.extend(output.tolist())

    return np.array(all_samples, dtype=np.int16)


def check_block_discontinuities(samples, threshold=100):
    """Check for discontinuities at block boundaries."""
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

    def test_vm_dry_no_discontinuities(self):
        """VM dry output should have no block discontinuities."""
        samples = generate_vm_direct(beta_target_q15=0, num_blocks=100)
        disc_count = check_block_discontinuities(samples)

        assert disc_count == 0, f"Found {disc_count} block discontinuities"

    def test_vm_dry_spectral_purity(self):
        """VM dry output should be spectrally pure."""
        samples = generate_vm_direct(beta_target_q15=0, num_blocks=200)
        result = analyze_spectrum_vm(samples)

        assert result["is_pure"], (
            f"VM dry output has non-harmonic content: {result['non_harmonic']}"
        )

    def test_vm_output_not_silent(self):
        """VM should produce non-silent output."""
        samples = generate_vm_direct(beta_target_q15=0, num_blocks=100)
        rms = np.sqrt(np.mean(samples.astype(float) ** 2))

        assert rms > 5000, f"RMS too low: {rms:.1f}"


class TestVMETAEffect:
    """Test ETA effect on VM output."""

    def test_vm_eta_changes_output(self):
        """ETA should affect VM output."""
        samples_dry = generate_vm_direct(beta_target_q15=0, num_blocks=100)
        samples_wet = generate_vm_direct(beta_target_q15=32767, num_blocks=100)

        diff = np.abs(samples_dry.astype(int) - samples_wet.astype(int)).max()
        assert diff > 100, f"ETA has no effect on VM output (max diff={diff})"

    @pytest.mark.parametrize("beta_q15", [0, 8192, 16384, 24576, 32767])
    def test_vm_at_eta_value(self, beta_q15):
        """VM should work at various ETA values."""
        samples = generate_vm_direct(beta_target_q15=beta_q15, num_blocks=100)
        rms = np.sqrt(np.mean(samples.astype(float) ** 2))

        assert rms > 1000, f"At ETA={beta_q15}: RMS too low ({rms:.1f})"


class TestVMWetPath:
    """Test VM wet path quality."""

    def test_vm_wet_limited_artifacts(self):
        """VM wet output should have limited non-harmonic content."""
        samples = generate_vm_direct(beta_target_q15=32767, num_blocks=200)
        result = analyze_spectrum_vm(samples)

        assert len(result["non_harmonic"]) <= 5, (
            f"Too many non-harmonic peaks: {len(result['non_harmonic'])}"
        )

    def test_vm_wet_controlled_discontinuities(self):
        """VM wet output discontinuities should be controlled."""
        samples = generate_vm_direct(beta_target_q15=32767, num_blocks=100)
        disc_count = check_block_discontinuities(samples, threshold=200)

        assert disc_count <= 10, f"Too many discontinuities: {disc_count}"
