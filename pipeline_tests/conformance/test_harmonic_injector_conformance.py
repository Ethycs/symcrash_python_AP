"""
HARMONIC INJECTOR CONFORMANCE TESTS
===================================

Tests for harmonic_injector.py Baby Fourier / Harmonic Target Generator.
Extracted from harmonic_injector.py test_harmonic_injector().
"""

import pytest


class TestNoteToFrequency:
    """Test MIDI note to frequency conversion."""

    def test_a4_is_440hz(self):
        """MIDI note 69 (A4) should be 440 Hz."""
        from symcrash_python_ap.wavelet.harmonic_injector import note_to_freq

        freq = note_to_freq(69, 0)
        assert abs(freq - 440.0) < 0.1, f"A4 = {freq:.1f} Hz, expected 440 Hz"

    def test_c4_is_261hz(self):
        """MIDI note 60 (C4) should be ~261.6 Hz."""
        from symcrash_python_ap.wavelet.harmonic_injector import note_to_freq

        freq = note_to_freq(60, 0)
        assert abs(freq - 261.6) < 0.5, f"C4 = {freq:.1f} Hz, expected 261.6 Hz"

    def test_c3_is_130hz(self):
        """MIDI note 48 (C3) should be ~130.8 Hz."""
        from symcrash_python_ap.wavelet.harmonic_injector import note_to_freq

        freq = note_to_freq(48, 0)
        assert abs(freq - 130.8) < 0.5, f"C3 = {freq:.1f} Hz, expected 130.8 Hz"

    def test_a5_is_880hz(self):
        """MIDI note 81 (A5) should be 880 Hz."""
        from symcrash_python_ap.wavelet.harmonic_injector import note_to_freq

        freq = note_to_freq(81, 0)
        assert abs(freq - 880.0) < 0.5, f"A5 = {freq:.1f} Hz, expected 880 Hz"


class TestFrequencyToLeaf:
    """Test frequency to leaf mapping."""

    def test_low_frequency_maps_to_low_leaf(self):
        """100 Hz should map to a low leaf index."""
        from symcrash_python_ap.wavelet.harmonic_injector import freq_to_leaf

        leaf = freq_to_leaf(100)
        assert leaf < 10, f"100 Hz should map to low leaf, got {leaf}"

    def test_high_frequency_maps_to_high_leaf(self):
        """10000 Hz should map to a high leaf index."""
        from symcrash_python_ap.wavelet.harmonic_injector import freq_to_leaf

        leaf = freq_to_leaf(10000)
        assert leaf > 20, f"10000 Hz should map to high leaf, got {leaf}"

    def test_frequency_in_band_range(self):
        """Mapped frequency should be within band limits."""
        from symcrash_python_ap.wavelet.harmonic_injector import freq_to_leaf, LEAF_EDGES_HZ

        test_freqs = [220, 440, 880, 1760]
        for freq in test_freqs:
            leaf = freq_to_leaf(freq)
            low = LEAF_EDGES_HZ[leaf]
            high = LEAF_EDGES_HZ[leaf + 1]
            assert low <= freq <= high, \
                f"{freq} Hz -> leaf {leaf} but band is {low}-{high} Hz"


class TestWaveTypeAmplitudes:
    """Test wave type amplitude laws."""

    def test_sine_only_fundamental(self):
        """Sine wave should only have fundamental."""
        from symcrash_python_ap.wavelet.harmonic_injector import WaveType, get_partial_amplitude, Q15_ONE

        # Fundamental (k=1) should be full amplitude
        amp_1 = get_partial_amplitude(WaveType.SINE, 1)
        assert amp_1 == Q15_ONE

        # Higher partials should be zero
        for k in range(2, 9):
            amp_k = get_partial_amplitude(WaveType.SINE, k)
            assert amp_k == 0, f"Sine partial {k} should be 0, got {amp_k}"

    def test_saw_has_all_harmonics(self):
        """Saw wave should have all harmonics with 1/k rolloff."""
        from symcrash_python_ap.wavelet.harmonic_injector import WaveType, get_partial_amplitude, Q15_ONE

        for k in range(1, 9):
            amp_k = get_partial_amplitude(WaveType.SAW, k)
            expected_ratio = 1.0 / k
            actual_ratio = amp_k / Q15_ONE

            # Allow some tolerance for Q15 quantization
            assert abs(actual_ratio - expected_ratio) < 0.02, \
                f"Saw partial {k}: got {actual_ratio:.3f}, expected {expected_ratio:.3f}"

    def test_triangle_only_odd_harmonics(self):
        """Triangle wave should have only odd harmonics."""
        from symcrash_python_ap.wavelet.harmonic_injector import WaveType, get_partial_amplitude

        for k in range(1, 9):
            amp_k = get_partial_amplitude(WaveType.TRIANGLE, k)
            if k % 2 == 0:  # Even harmonics
                assert amp_k == 0, f"Triangle even partial {k} should be 0, got {amp_k}"
            else:  # Odd harmonics
                assert amp_k > 0, f"Triangle odd partial {k} should be > 0, got {amp_k}"


class TestHarmonicTargetGenerator:
    """Test HarmonicTargetGenerator class."""

    def test_generate_sparse_target(self):
        """Sparse target should have active leaves for harmonics."""
        from symcrash_python_ap.wavelet.harmonic_injector import HarmonicTargetGenerator, WaveType

        gen = HarmonicTargetGenerator()
        gen.set_note(0, 57)  # A3 = 220 Hz
        gen.set_wave_params(0, WaveType.SAW, partials_n=8)

        target = gen.generate_target(0)

        assert len(target.leaf_indices) > 0, "Should have active leaves"
        assert len(target.leaf_amplitudes) == len(target.leaf_indices)

    def test_generate_dense_target(self):
        """Dense target should have 256 coefficients."""
        from symcrash_python_ap.wavelet.harmonic_injector import HarmonicTargetGenerator, WaveType, COEFF_LEN

        gen = HarmonicTargetGenerator()
        gen.set_note(0, 57)  # A3 = 220 Hz
        gen.set_wave_params(0, WaveType.SAW, partials_n=8)

        target = gen.generate_target(0, include_dense=True)

        assert len(target.dense_target) == COEFF_LEN
        non_zero = sum(1 for c in target.dense_target if c != 0)
        assert non_zero > 0, "Dense target should have non-zero coefficients"

    def test_apply_injection(self):
        """Injection should modify coefficient vector."""
        from symcrash_python_ap.wavelet.harmonic_injector import HarmonicTargetGenerator, WaveType, COEFF_LEN, Q15_ONE

        gen = HarmonicTargetGenerator()
        gen.set_note(0, 57)
        gen.set_wave_params(0, WaveType.SAW, partials_n=8)
        gen.set_injection_params(0, harmonic_dial_q15=Q15_ONE, pull_strength_q15=Q15_ONE)

        target = gen.generate_target(0, include_dense=True)
        current = [0] * COEFF_LEN

        result = gen.apply_injection(0, current, target)

        assert len(result) == COEFF_LEN
        non_zero = sum(1 for c in result if c != 0)
        assert non_zero > 0, "Injection should produce non-zero coefficients"


class TestConvenienceFunctions:
    """Test convenience functions for coefficient generation."""

    def test_create_harmonic_coeffs_length(self):
        """create_harmonic_coeffs should return 256 coefficients."""
        from symcrash_python_ap.wavelet.harmonic_injector import create_harmonic_coeffs, WaveType, COEFF_LEN

        coeffs = create_harmonic_coeffs(
            frequency=220.0,
            amplitude=0x5000,
            wave_type=WaveType.SAW,
            partials_n=12
        )

        assert len(coeffs) == COEFF_LEN

    def test_create_harmonic_coeffs_non_zero(self):
        """create_harmonic_coeffs should have non-zero coefficients."""
        from symcrash_python_ap.wavelet.harmonic_injector import create_harmonic_coeffs, WaveType

        coeffs = create_harmonic_coeffs(
            frequency=220.0,
            amplitude=0x5000,
            wave_type=WaveType.SAW,
            partials_n=12
        )

        non_zero = sum(1 for c in coeffs if c != 0)
        assert non_zero > 0, "Should have non-zero coefficients"

    def test_synthesize_harmonic_audio_length(self):
        """synthesize_harmonic_audio should return correct block size."""
        from symcrash_python_ap.wavelet.harmonic_injector import synthesize_harmonic_audio, WaveType

        audio, phase = synthesize_harmonic_audio(
            frequency=440.0,
            amplitude=0x5000,
            wave_type=WaveType.SAW,
            partials_n=8,
            block_size=64
        )

        assert len(audio) == 64

    def test_synthesize_harmonic_audio_phase_continuity(self):
        """synthesize_harmonic_audio should return updated phase state."""
        from symcrash_python_ap.wavelet.harmonic_injector import synthesize_harmonic_audio, WaveType

        audio1, phase1 = synthesize_harmonic_audio(
            frequency=440.0,
            amplitude=0x5000,
            wave_type=WaveType.SAW,
            partials_n=8,
            block_size=64
        )

        audio2, phase2 = synthesize_harmonic_audio(
            frequency=440.0,
            amplitude=0x5000,
            wave_type=WaveType.SAW,
            partials_n=8,
            block_size=64,
            phase_state=phase1
        )

        # Phase should have advanced
        assert phase1 != phase2, "Phase state should advance between blocks"
