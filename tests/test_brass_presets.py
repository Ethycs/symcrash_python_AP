"""
Test FM6 Brass Preset Configurations
====================================

Verifies that the brass preset configurations produce valid audio output.
"""

import numpy as np
import pytest

from symcrash_python_ap.vm.fm6_state import FM6State, Q15_ONE, Q16_ONE
from symcrash_python_ap.vm.fm6_engine import (
    fm6_generate_audio,
    configure_brass,
    configure_synth_brass,
    configure_jump_brass,
)


# Test constants
SAMPLE_RATE = 48000
A4_FREQ = 440.0
A4_PHASE_INC = int((A4_FREQ / SAMPLE_RATE) * (1 << 32))


class TestBrassPresets:
    """Test FM6 brass preset configurations."""

    def test_configure_brass_produces_audio(self):
        """Verify configure_brass generates non-zero audio."""
        state = FM6State()
        configure_brass(state)

        # Generate 10 blocks of audio
        audio = fm6_generate_audio(state, A4_PHASE_INC, num_blocks=10)

        # Verify audio is non-zero
        assert len(audio) > 0
        assert np.max(np.abs(audio)) > 0, "Brass should produce non-zero audio"

        # Verify audio has reasonable amplitude
        peak = np.max(np.abs(audio))
        assert peak > 1000, f"Brass peak {peak} too quiet"
        assert peak <= 32767, f"Brass peak {peak} exceeds Q15 range"

    def test_configure_brass_algorithm(self):
        """Verify configure_brass uses Algorithm 22."""
        state = FM6State()
        configure_brass(state)

        assert state.algorithm_id == 22

    def test_configure_brass_ratios(self):
        """Verify configure_brass has correct operator ratios."""
        state = FM6State()
        configure_brass(state)

        # Op1 should be 1.0 (anchor)
        assert state.operators[0].ratio_q16 == Q16_ONE

        # Op6 should be 2.0 (modulator with edge)
        assert state.operators[5].ratio_q16 == Q16_ONE * 2

        # Other carriers should have slight detune
        assert state.operators[1].ratio_q16 != Q16_ONE
        assert state.operators[2].ratio_q16 != Q16_ONE
        assert state.operators[3].ratio_q16 != Q16_ONE

    def test_configure_synth_brass_produces_audio(self):
        """Verify configure_synth_brass generates non-zero audio."""
        state = FM6State()
        configure_synth_brass(state)

        audio = fm6_generate_audio(state, A4_PHASE_INC, num_blocks=10)

        assert len(audio) > 0
        assert np.max(np.abs(audio)) > 0

        peak = np.max(np.abs(audio))
        assert peak > 1000, f"Synth brass peak {peak} too quiet"

    def test_configure_synth_brass_algorithm(self):
        """Verify configure_synth_brass uses Algorithm 16."""
        state = FM6State()
        configure_synth_brass(state)

        assert state.algorithm_id == 16

    def test_configure_synth_brass_harmonic_ratios(self):
        """Verify synth_brass uses harmonic series ratios."""
        state = FM6State()
        configure_synth_brass(state)

        # Harmonic series: 1, 2, 3, 4, 5, 2
        assert state.operators[0].ratio_q16 == Q16_ONE
        assert state.operators[1].ratio_q16 == Q16_ONE * 2
        assert state.operators[2].ratio_q16 == Q16_ONE * 3
        assert state.operators[3].ratio_q16 == Q16_ONE * 4
        assert state.operators[4].ratio_q16 == Q16_ONE * 5
        assert state.operators[5].ratio_q16 == Q16_ONE * 2

    def test_configure_jump_brass_produces_audio(self):
        """Verify configure_jump_brass generates non-zero audio."""
        state = FM6State()
        configure_jump_brass(state)

        audio = fm6_generate_audio(state, A4_PHASE_INC, num_blocks=10)

        assert len(audio) > 0
        assert np.max(np.abs(audio)) > 0

        peak = np.max(np.abs(audio))
        assert peak > 1000, f"Jump brass peak {peak} too quiet"

    def test_configure_jump_brass_algorithm(self):
        """Verify configure_jump_brass uses Algorithm 19."""
        state = FM6State()
        configure_jump_brass(state)

        assert state.algorithm_id == 19

    def test_configure_jump_brass_detune(self):
        """Verify jump_brass has heavy carrier detune."""
        state = FM6State()
        configure_jump_brass(state)

        # Op1 is anchor at 1.0
        assert state.operators[0].ratio_q16 == Q16_ONE

        # Op2 should be detuned up
        assert state.operators[1].ratio_q16 > Q16_ONE

        # Op3 should be detuned down
        assert state.operators[2].ratio_q16 < Q16_ONE

        # Op4 should be detuned up more
        assert state.operators[3].ratio_q16 > state.operators[1].ratio_q16

    def test_brass_presets_have_distinct_spectra(self):
        """Verify different brass presets produce different spectra."""
        state1 = FM6State()
        configure_brass(state1)
        audio1 = fm6_generate_audio(state1, A4_PHASE_INC, num_blocks=20)

        state2 = FM6State()
        configure_synth_brass(state2)
        audio2 = fm6_generate_audio(state2, A4_PHASE_INC, num_blocks=20)

        state3 = FM6State()
        configure_jump_brass(state3)
        audio3 = fm6_generate_audio(state3, A4_PHASE_INC, num_blocks=20)

        # Simple spectrum comparison: RMS should differ
        rms1 = np.sqrt(np.mean(audio1.astype(np.float64) ** 2))
        rms2 = np.sqrt(np.mean(audio2.astype(np.float64) ** 2))
        rms3 = np.sqrt(np.mean(audio3.astype(np.float64) ** 2))

        # At least two should be different
        assert not (rms1 == rms2 == rms3), "All brass presets have identical RMS"

    def test_brass_feedback_stability(self):
        """Verify brass presets remain stable with extended playback."""
        for config_fn in [configure_brass, configure_synth_brass, configure_jump_brass]:
            state = FM6State()
            config_fn(state)

            # Generate 100 blocks (more than typical note)
            audio = fm6_generate_audio(state, A4_PHASE_INC, num_blocks=100)

            # Check for runaway (energy explosion)
            initial_energy = np.mean(audio[:80].astype(np.float64) ** 2)
            final_energy = np.mean(audio[-80:].astype(np.float64) ** 2)

            # Energy shouldn't grow more than 4x (feedback stability)
            assert final_energy <= initial_energy * 4, \
                f"{config_fn.__name__} feedback unstable: {final_energy/initial_energy:.1f}x growth"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
