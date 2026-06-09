"""
Test emphasis → family dispatch.

Verifies that per-regime emphasis weights from presets are correctly
translated into family_base values and consumed by the voice engine's
family weight computation.
"""

import pytest
from symcrash_python_ap.instrument_ir import (
    emphasis_to_family_base,
    EMPHASIS_FAMILY_MAP,
)
from symcrash_python_ap.primitives import Family, Regime, Q15_ONE, Q15_HALF


# =========================================================================
# emphasis_to_family_base() unit tests
# =========================================================================

class TestEmphasisToFamilyBase:
    """Test the emphasis → family_base translation function."""

    def test_empty_emphasis(self):
        result = emphasis_to_family_base({})
        assert result == [0] * 8

    def test_single_emphasis_tone(self):
        result = emphasis_to_family_base({'tone': 1.0})
        assert result[Family.F_COLOR] == Q15_ONE
        # All others zero
        for i in range(8):
            if i != Family.F_COLOR:
                assert result[i] == 0

    def test_single_emphasis_energy(self):
        result = emphasis_to_family_base({'energy': 0.5})
        assert result[Family.F_EVENTS] == Q15_ONE // 2

    def test_multiple_emphasis(self):
        result = emphasis_to_family_base({
            'tone': 0.9,
            'energy': 1.0,
            'body': 0.5,
        })
        assert result[Family.F_COLOR] == int(0.9 * Q15_ONE)
        assert result[Family.F_EVENTS] == Q15_ONE
        assert result[Family.F_LOCAL] == int(0.5 * Q15_ONE)
        # Others zero
        assert result[Family.F_RESERVED] == 0
        assert result[Family.F_TRANSPORT] == 0
        assert result[Family.F_TOPOLOGY] == 0

    def test_additive_glue_slots(self):
        """clarity, stability, settling all map to F_GLUE and should sum."""
        result = emphasis_to_family_base({
            'clarity': 0.3,
            'stability': 0.3,
            'settling': 0.3,
        })
        expected = min(Q15_ONE, int(0.3 * Q15_ONE) * 3)
        assert result[Family.F_GLUE] == expected

    def test_additive_glue_clamped(self):
        """Sum of multiple F_GLUE contributions should clamp to Q15_ONE."""
        result = emphasis_to_family_base({
            'clarity': 1.0,
            'stability': 1.0,
        })
        assert result[Family.F_GLUE] == Q15_ONE

    def test_unknown_emphasis_ignored(self):
        result = emphasis_to_family_base({'mystery': 0.8, 'tone': 0.5})
        assert result[Family.F_COLOR] == int(0.5 * Q15_ONE)
        # No crash, unknown key silently ignored

    def test_all_emphasis_names_have_valid_slots(self):
        """Every key in EMPHASIS_FAMILY_MAP maps to a valid family index."""
        for name, slot in EMPHASIS_FAMILY_MAP.items():
            assert 0 <= slot < 8, f"{name} maps to out-of-range slot {slot}"

    def test_chaos_maps_to_catastrophe(self):
        result = emphasis_to_family_base({'chaos': 0.7})
        assert result[Family.F_CATASTROPHE] == int(0.7 * Q15_ONE)


# =========================================================================
# InstrumentProgram.per_regime_emphasis integration
# =========================================================================

class TestPerRegimeEmphasis:
    """Test that per_regime_emphasis on InstrumentProgram affects family weights."""

    def test_program_default_has_no_emphasis(self):
        from symcrash_python_ap.voice.voice_engine import InstrumentProgram
        prog = InstrumentProgram()
        assert prog.per_regime_emphasis == {}

    def test_program_with_emphasis(self):
        from symcrash_python_ap.voice.voice_engine import InstrumentProgram
        emphasis = {
            int(Regime.R1_ATTACK): {'tone': 0.9, 'energy': 1.0},
            int(Regime.R2_STABLE): {'tone': 0.8, 'body': 0.6},
        }
        prog = InstrumentProgram(per_regime_emphasis=emphasis)
        assert int(Regime.R1_ATTACK) in prog.per_regime_emphasis
        assert prog.per_regime_emphasis[int(Regime.R1_ATTACK)]['tone'] == 0.9

    def test_emphasis_overrides_regime_spec_base(self):
        """
        When a program has per_regime_emphasis for R1_ATTACK,
        the family weights should use emphasis-derived values
        instead of the hardcoded RegimeSpec defaults.
        """
        from symcrash_python_ap.voice.voice_engine import (
            InstrumentProgram, VoiceEngine, VoiceStateWithProgram,
            IntentVector, REGIME_SPECS,
        )

        # Create program with emphasis that strongly weights F_COLOR (tone)
        emphasis = {
            int(Regime.R1_ATTACK): {'tone': 1.0},
        }
        prog = InstrumentProgram(per_regime_emphasis=emphasis)

        # Create engine
        engine = VoiceEngine(max_voices=1)

        # Create voice state
        state = VoiceStateWithProgram(
            voice_id=0,
            program=prog,
            program_id=0,
        )
        state.regime = Regime.R1_ATTACK
        state.intent = IntentVector()  # Zero intent

        # Compute weights
        weights = engine._compute_family_weights_with_regime(state)

        # F_COLOR should reflect emphasis (tone=1.0 → Q15_ONE),
        # not the hardcoded RegimeSpec default
        hardcoded_base = REGIME_SPECS[Regime.R1_ATTACK].family_base
        assert weights[Family.F_COLOR] != hardcoded_base[Family.F_COLOR], \
            "Emphasis should override the hardcoded regime base"

        # F_COLOR base from emphasis is Q15_ONE; after program mods
        # (default Q15_ONE) it should still be Q15_ONE
        assert weights[Family.F_COLOR] > 0

    def test_no_emphasis_falls_back_to_hardcoded(self):
        """Without per_regime_emphasis, behavior is unchanged."""
        from symcrash_python_ap.voice.voice_engine import (
            InstrumentProgram, VoiceEngine, VoiceStateWithProgram,
            IntentVector, REGIME_SPECS,
        )

        prog = InstrumentProgram()  # No emphasis
        engine = VoiceEngine(max_voices=1)

        state = VoiceStateWithProgram(
            voice_id=0,
            program=prog,
            program_id=0,
        )
        state.regime = Regime.R2_STABLE
        state.intent = IntentVector()  # Zero intent

        weights = engine._compute_family_weights_with_regime(state)

        # Should use hardcoded defaults
        hardcoded_base = REGIME_SPECS[Regime.R2_STABLE].family_base
        # With zero intent and default mods (Q15_ONE), weights == base
        # (caps permitting)
        for i in range(8):
            cap = REGIME_SPECS[Regime.R2_STABLE].family_cap[i]
            expected = min(hardcoded_base[i], cap)
            assert weights[i] == expected, \
                f"Family {i}: expected {expected}, got {weights[i]}"
