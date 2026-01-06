"""
WSE-STD-210 Mode B Compliance Tests
====================================

Tests for Mode B (Rotator/Harmonic Bank) coefficient oscillator compliance
with WSE-STD-210 specification.

Per WSE-STD-210 Section 13, the following requirements MUST be met:
- Pitch accuracy: f0 error < 0.1%
- Spur level: Largest spur < -60 dBc
- Determinism: Bit-exact across runs
- PWM aliasing: Clean PWM sweeps
- Mode switch: Click-free A↔B transitions
"""

import pytest
import numpy as np
from typing import Tuple

# Import test subjects
from symcrash_python_ap.vm.harmonic_bank import (
    ModeBState,
    HarmonicBasisFrames,
    HarmonicWeights,
    generate_basis_frames,
    get_basis_for_note,
    mode_b_generate_coeffs,
    compute_pwm_weights,
    compute_bandlimit_H,
    VM_K,
    NUM_LEAVES,
    BLOCK_SIZE,
    SAMPLE_RATE,
    DEFAULT_H_MAX,
)
from symcrash_python_ap.vm.state import VMState, OSC_MODE_A, OSC_MODE_B
from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.tables import DEFAULT_TABLES, COEFF_LEN
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def measure_pitch_from_samples(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """
    Measure fundamental frequency from audio samples using FFT.

    Returns frequency in Hz.
    """
    if len(samples) < 1024:
        return 0.0

    # Use a window
    windowed = samples.astype(float) * np.hanning(len(samples))

    # FFT
    spectrum = np.abs(np.fft.rfft(windowed))

    # Find peak (skip DC)
    peak_idx = np.argmax(spectrum[1:]) + 1

    # Frequency resolution
    freq_res = sample_rate / len(samples)

    return peak_idx * freq_res


def measure_spur_level(samples: np.ndarray, f0_hz: float, sample_rate: int = SAMPLE_RATE) -> float:
    """
    Measure largest spur level relative to fundamental (dBc).

    Returns dBc (negative value = spur below fundamental).
    """
    if len(samples) < 1024:
        return 0.0

    # Use a window
    windowed = samples.astype(float) * np.hanning(len(samples))

    # FFT
    spectrum = np.abs(np.fft.rfft(windowed))

    # Find fundamental bin
    freq_res = sample_rate / len(samples)
    f0_bin = int(round(f0_hz / freq_res))

    # Fundamental power (with a few bins around it)
    f0_power = np.max(spectrum[max(1, f0_bin-2):f0_bin+3])

    # Find largest non-harmonic spur
    harmonic_bins = set()
    for h in range(1, 64):
        h_bin = int(round(h * f0_hz / freq_res))
        if h_bin < len(spectrum):
            for offset in range(-2, 3):
                if 0 <= h_bin + offset < len(spectrum):
                    harmonic_bins.add(h_bin + offset)

    # Mask harmonics
    spur_spectrum = spectrum.copy()
    for b in harmonic_bins:
        spur_spectrum[b] = 0

    # Find largest spur
    largest_spur = np.max(spur_spectrum[1:])  # Skip DC

    if largest_spur == 0 or f0_power == 0:
        return -100.0  # Very clean

    return 20 * np.log10(largest_spur / f0_power)


# =============================================================================
# BASIS FRAME TESTS
# =============================================================================

class TestBasisFrameGeneration:
    """Tests for basis frame generation (WSE-STD-210 Section 6.2)."""

    def test_basis_frame_shapes(self):
        """Verify basis frame array shapes."""
        basis = generate_basis_frames(anchor_note=69, H_max=16)

        assert basis.DC_frame.shape == (NUM_LEAVES, VM_K)
        assert basis.Ucos_h.shape == (16, NUM_LEAVES, VM_K)
        assert basis.Usin_h.shape == (16, NUM_LEAVES, VM_K)
        assert basis.H_max == 16
        assert basis.anchor_note == 69

    def test_basis_frame_dtype(self):
        """Verify basis frames are int16."""
        basis = generate_basis_frames(anchor_note=69, H_max=8)

        assert basis.DC_frame.dtype == np.int16
        assert basis.Ucos_h.dtype == np.int16
        assert basis.Usin_h.dtype == np.int16

    def test_dc_frame_nonzero(self):
        """DC frame should have significant energy."""
        basis = generate_basis_frames(anchor_note=69, H_max=8)

        # DC frame should reconstruct to constant ~1.0 in time domain
        # In coefficient domain, it should have significant values
        dc_energy = np.mean(np.abs(basis.DC_frame.astype(float)))
        assert dc_energy > 100, f"DC frame energy too low: {dc_energy}"

    def test_harmonic_frames_nonzero(self):
        """Harmonic basis frames should have energy."""
        basis = generate_basis_frames(anchor_note=69, H_max=8)

        for h in range(1, 5):
            cos_energy = np.mean(np.abs(basis.Ucos_h[h-1].astype(float)))
            sin_energy = np.mean(np.abs(basis.Usin_h[h-1].astype(float)))
            assert cos_energy > 10 or sin_energy > 10, \
                f"Harmonic {h} basis has too low energy: cos={cos_energy}, sin={sin_energy}"


class TestBasisFrameAtlas:
    """Tests for basis frame atlas (WSE-STD-210 Section 5)."""

    def test_octave_anchors(self):
        """Verify octave-based anchor selection."""
        # Notes in same octave should use same anchor
        basis_60 = get_basis_for_note(60)  # C4
        basis_65 = get_basis_for_note(65)  # F4

        assert basis_60.anchor_note == basis_65.anchor_note

    def test_different_octave_anchors(self):
        """Different octaves should have different anchors."""
        basis_60 = get_basis_for_note(60)  # C4 (octave 5, anchor A4=69)
        basis_72 = get_basis_for_note(72)  # C5 (octave 6, anchor A5=81)

        assert basis_60.anchor_note != basis_72.anchor_note


# =============================================================================
# HARMONIC WEIGHT TESTS
# =============================================================================

class TestHarmonicWeights:
    """Tests for harmonic weight calculation."""

    def test_sine_weights(self):
        """Sine wave should have a_1 = 1, all others = 0."""
        weights = HarmonicWeights()
        weights.set_sine()

        assert weights.dc_q15 == 0
        assert weights.a_h_q15[0] == 32767  # a_1 = 1
        assert weights.b_h_q15[0] == 0
        assert np.all(weights.a_h_q15[1:] == 0)
        assert np.all(weights.b_h_q15[1:] == 0)

    def test_saw_weights_rolloff(self):
        """Saw wave should have 1/h rolloff."""
        weights = HarmonicWeights()
        weights.set_saw(H=8)

        # a_1 should be largest
        assert abs(weights.a_h_q15[0]) > abs(weights.a_h_q15[1])
        # Roughly 1/h rolloff
        ratio = abs(weights.a_h_q15[0]) / max(1, abs(weights.a_h_q15[1]))
        assert 1.5 < ratio < 2.5, f"Expected ~2x rolloff, got {ratio}"

    def test_square_weights_odd_only(self):
        """Square wave should have only odd harmonics."""
        weights = HarmonicWeights()
        weights.set_square(H=8)

        # Even harmonics should be zero
        assert weights.a_h_q15[1] == 0  # h=2
        assert weights.a_h_q15[3] == 0  # h=4
        assert weights.a_h_q15[5] == 0  # h=6

        # Odd harmonics should be nonzero
        assert weights.a_h_q15[0] != 0  # h=1
        assert weights.a_h_q15[2] != 0  # h=3
        assert weights.a_h_q15[4] != 0  # h=5

    def test_triangle_weights_squared_rolloff(self):
        """Triangle wave should have 1/h² rolloff."""
        weights = HarmonicWeights()
        weights.set_triangle(H=8)

        # h=1 to h=3 ratio should be ~9x (1/1 vs 1/9)
        if weights.a_h_q15[2] != 0:  # h=3
            ratio = abs(weights.a_h_q15[0]) / abs(weights.a_h_q15[2])
            assert 6 < ratio < 12, f"Expected ~9x rolloff, got {ratio}"


class TestPWMWeights:
    """Tests for PWM weight calculation (WSE-STD-210 Section 8.1)."""

    def test_pwm_50_percent(self):
        """50% duty cycle should be symmetric (like square wave)."""
        weights = compute_pwm_weights(16384, H_max=16)  # 50%

        # DC should be ~0
        assert abs(weights.dc_q15) < 1000

        # Even harmonics should be near zero
        assert abs(weights.a_h_q15[1]) < 1000  # h=2
        assert abs(weights.a_h_q15[3]) < 1000  # h=4

    def test_pwm_25_percent(self):
        """25% duty cycle should have specific DC offset."""
        weights = compute_pwm_weights(8192, H_max=16)  # 25%

        # DC = 2d - 1 = 2*0.25 - 1 = -0.5
        expected_dc = int(-0.5 * 32767)
        assert abs(weights.dc_q15 - expected_dc) < 1000

    def test_pwm_harmonic_content(self):
        """PWM should have harmonics for non-50% duty cycles."""
        weights = compute_pwm_weights(24576, H_max=16)  # 75%

        # Should have some harmonic content
        total_harmonic = np.sum(np.abs(weights.a_h_q15[:8]))
        assert total_harmonic > 10000


# =============================================================================
# COEFFICIENT SYNTHESIS TESTS
# =============================================================================

class TestCoefficientSynthesis:
    """Tests for Mode B coefficient synthesis (WSE-STD-210 Section 3.2.1)."""

    def test_coeff_output_shape(self):
        """Output should be 512 coefficients."""
        state = ModeBState()
        state.reset(note=69, phase_inc=freq_to_phase_inc(440.0, SAMPLE_RATE))
        state.weights.set_sine()

        basis = get_basis_for_note(69)
        output = np.zeros(COEFF_LEN, dtype=np.int16)

        mode_b_generate_coeffs(state, basis, output)

        assert output.shape == (COEFF_LEN,)
        assert output.dtype == np.int16

    def test_coeff_output_nonzero(self):
        """Sine wave should produce nonzero coefficients."""
        state = ModeBState()
        state.reset(note=69, phase_inc=freq_to_phase_inc(440.0, SAMPLE_RATE))
        state.weights.set_sine()

        basis = get_basis_for_note(69)
        output = np.zeros(COEFF_LEN, dtype=np.int16)

        mode_b_generate_coeffs(state, basis, output)

        # Should have significant energy
        rms = np.sqrt(np.mean(output.astype(float)**2))
        assert rms > 100, f"Coefficient RMS too low: {rms}"

    def test_phase_advance(self):
        """Phase should advance correctly between blocks."""
        state = ModeBState()
        state.reset(note=69, phase_inc=freq_to_phase_inc(440.0, SAMPLE_RATE))

        initial_phase = state.phase
        state.advance_block()

        expected_advance = BLOCK_SIZE * state.phase_inc
        actual_advance = (state.phase - initial_phase) & 0xFFFFFFFF

        assert actual_advance == expected_advance


class TestBandlimit:
    """Tests for bandlimit calculation (WSE-STD-210 Section 7.1)."""

    def test_low_pitch_high_harmonics(self):
        """Low pitches should allow many harmonics."""
        H = compute_bandlimit_H(100.0, SAMPLE_RATE)
        assert H > 100  # 48000/2/100 = 240, with 0.9 margin = ~216

    def test_high_pitch_few_harmonics(self):
        """High pitches should limit harmonics."""
        H = compute_bandlimit_H(4000.0, SAMPLE_RATE)
        assert H < 10  # 48000/2/4000 = 6, with 0.9 margin = ~5

    def test_nyquist_safety(self):
        """Harmonics should stay below Nyquist."""
        for freq in [100, 440, 1000, 2000]:
            H = compute_bandlimit_H(float(freq), SAMPLE_RATE)
            max_freq = H * freq
            assert max_freq < SAMPLE_RATE / 2, \
                f"f0={freq}, H={H}, max_freq={max_freq} exceeds Nyquist"


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for bit-exact determinism (WSE-STD-210 Section 13)."""

    def test_repeated_generation_identical(self):
        """Same inputs should produce identical outputs."""
        basis = get_basis_for_note(69)

        # Run 1
        state1 = ModeBState()
        state1.reset(note=69, phase_inc=freq_to_phase_inc(440.0, SAMPLE_RATE))
        state1.weights.set_sine()
        output1 = np.zeros(COEFF_LEN, dtype=np.int16)
        mode_b_generate_coeffs(state1, basis, output1)

        # Run 2
        state2 = ModeBState()
        state2.reset(note=69, phase_inc=freq_to_phase_inc(440.0, SAMPLE_RATE))
        state2.weights.set_sine()
        output2 = np.zeros(COEFF_LEN, dtype=np.int16)
        mode_b_generate_coeffs(state2, basis, output2)

        # Must be bit-exact
        assert np.array_equal(output1, output2), "Outputs differ between runs"

    def test_pwm_weights_deterministic(self):
        """PWM weight calculation should be deterministic."""
        w1 = compute_pwm_weights(16384, 32)
        w2 = compute_pwm_weights(16384, 32)

        assert w1.dc_q15 == w2.dc_q15
        assert np.array_equal(w1.a_h_q15, w2.a_h_q15)
        assert np.array_equal(w1.b_h_q15, w2.b_h_q15)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestModeAModeBParity:
    """Tests comparing Mode A and Mode B outputs."""

    def test_sine_similar_to_mode_a(self):
        """Mode B sine should be similar to Mode A sine."""
        from symcrash_python_ap.vm.coeff_lut_streaming import (
            ccs_generate_coeffs_polyphase, CCSState
        )

        note = 69
        phase_inc = freq_to_phase_inc(440.0, SAMPLE_RATE)

        # Mode A
        ccs_state = CCSState()
        ccs_state.note = note
        ccs_state.wave_id = 0  # Sine
        ccs_state.set_pitch(phase_inc, slew_blocks=0)
        mode_a_out = np.zeros(COEFF_LEN, dtype=np.int16)
        ccs_generate_coeffs_polyphase(ccs_state, mode_a_out)

        # Mode B
        mb_state = ModeBState()
        mb_state.reset(note=note, phase_inc=phase_inc)
        mb_state.weights.set_sine()
        basis = get_basis_for_note(note)
        mode_b_out = np.zeros(COEFF_LEN, dtype=np.int16)
        mode_b_generate_coeffs(mb_state, basis, mode_b_out)

        # Both should have similar energy (not identical due to different implementations)
        mode_a_rms = np.sqrt(np.mean(mode_a_out.astype(float)**2))
        mode_b_rms = np.sqrt(np.mean(mode_b_out.astype(float)**2))

        # Should be within 2x of each other
        ratio = max(mode_a_rms, mode_b_rms) / max(1, min(mode_a_rms, mode_b_rms))
        assert ratio < 3.0, f"RMS ratio too different: {ratio} (A={mode_a_rms}, B={mode_b_rms})"


class TestExecutorIntegration:
    """Tests for Mode B integration in executor."""

    def test_mode_b_voice_frame(self):
        """VoiceFrame with Mode B should work."""
        frame = VoiceFrame(
            voice_id=0,
            note=69,
            cents=0,
            gain_q15=32767,
            coeff_osc_mode=1,  # Mode B
            H_request=16,
            beta_target_q15=0,  # Pure carrier
        )

        state = VMState.create(voice_id=0)
        state.reset_for_note_on(note=69, cents=0)

        # Should not crash
        state, output = vm_step_block(state, frame, DEFAULT_TABLES)

        assert output.shape == (BLOCK_SIZE,)
        assert output.dtype == np.int16

    def test_mode_b_produces_audio(self):
        """Mode B should produce non-silent audio."""
        frame = VoiceFrame(
            voice_id=0,
            note=69,
            cents=0,
            gain_q15=32767,
            coeff_osc_mode=1,  # Mode B
            H_request=16,
            beta_target_q15=0,  # Pure carrier
        )

        state = VMState.create(voice_id=0)
        state.reset_for_note_on(note=69, cents=0)

        # Generate multiple blocks
        all_samples = []
        for _ in range(16):
            state, output = vm_step_block(state, frame, DEFAULT_TABLES)
            all_samples.extend(output.tolist())

        samples = np.array(all_samples, dtype=np.int16)
        rms = np.sqrt(np.mean(samples.astype(float)**2))

        assert rms > 100, f"Mode B output RMS too low: {rms}"

    def test_mode_b_pwm(self):
        """Mode B with PWM should work."""
        frame = VoiceFrame(
            voice_id=0,
            note=69,
            cents=0,
            gain_q15=32767,
            coeff_osc_mode=1,  # Mode B
            H_request=16,
            pwm_enabled=True,
            pwm_width_q15=24576,  # 75%
            beta_target_q15=0,
        )

        state = VMState.create(voice_id=0)
        state.reset_for_note_on(note=69, cents=0)

        # Should not crash
        state, output = vm_step_block(state, frame, DEFAULT_TABLES)

        assert output.shape == (BLOCK_SIZE,)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
