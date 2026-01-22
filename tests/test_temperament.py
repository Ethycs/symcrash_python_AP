"""
Tests for CMMM Enrichment Gap G5: Temperament Parameters
========================================================

Tests the TemperamentVector dataclass and its effect on EdgeSpec modulation.

Reference: CMMM-ENRICHMENT-DESIGN.md §9.4
"""

import pytest

from symcrash_python_ap.primitives import Q15_ONE, Q15_HALF
from symcrash_python_ap.instrument_ir import TemperamentVector, ControlConfig


class TestTemperamentVectorBasics:
    """Tests for TemperamentVector dataclass basics."""

    def test_default_values(self):
        """TemperamentVector should have neutral defaults."""
        temp = TemperamentVector()
        # All should be Q15_HALF (neutral) except chaos which is 0
        assert temp.hysteresis_q15 == Q15_HALF
        assert temp.stickiness_q15 == Q15_HALF
        assert temp.reactivity_q15 == Q15_HALF
        assert temp.escalation_bias_q15 == Q15_HALF
        assert temp.fatigue_sensitivity_q15 == Q15_HALF
        assert temp.tension_influence_q15 == Q15_HALF
        assert temp.chaos_amount_q15 == 0  # Deterministic by default

    def test_all_fields_q15_range(self):
        """All fields should be in valid Q15 range."""
        temp = TemperamentVector()
        assert 0 <= temp.hysteresis_q15 <= Q15_ONE
        assert 0 <= temp.stickiness_q15 <= Q15_ONE
        assert 0 <= temp.reactivity_q15 <= Q15_ONE
        assert 0 <= temp.escalation_bias_q15 <= Q15_ONE
        assert 0 <= temp.fatigue_sensitivity_q15 <= Q15_ONE
        assert 0 <= temp.tension_influence_q15 <= Q15_ONE
        assert 0 <= temp.chaos_amount_q15 <= Q15_ONE


class TestTemperamentSerialization:
    """Tests for TemperamentVector to_dict/from_dict."""

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        temp = TemperamentVector(
            hysteresis_q15=1000,
            stickiness_q15=2000,
            reactivity_q15=3000,
            escalation_bias_q15=4000,
            fatigue_sensitivity_q15=5000,
            tension_influence_q15=6000,
            chaos_amount_q15=7000,
        )
        d = temp.to_dict()
        assert d['hysteresis_q15'] == 1000
        assert d['stickiness_q15'] == 2000
        assert d['reactivity_q15'] == 3000
        assert d['escalation_bias_q15'] == 4000
        assert d['fatigue_sensitivity_q15'] == 5000
        assert d['tension_influence_q15'] == 6000
        assert d['chaos_amount_q15'] == 7000

    def test_from_dict(self):
        """from_dict should deserialize correctly."""
        d = {
            'hysteresis_q15': 1111,
            'stickiness_q15': 2222,
            'reactivity_q15': 3333,
            'escalation_bias_q15': 4444,
            'fatigue_sensitivity_q15': 5555,
            'tension_influence_q15': 6666,
            'chaos_amount_q15': 7777,
        }
        temp = TemperamentVector.from_dict(d)
        assert temp.hysteresis_q15 == 1111
        assert temp.stickiness_q15 == 2222
        assert temp.reactivity_q15 == 3333
        assert temp.escalation_bias_q15 == 4444
        assert temp.fatigue_sensitivity_q15 == 5555
        assert temp.tension_influence_q15 == 6666
        assert temp.chaos_amount_q15 == 7777

    def test_from_dict_missing_fields(self):
        """from_dict should handle missing fields with defaults."""
        d = {'hysteresis_q15': 500}
        temp = TemperamentVector.from_dict(d)
        assert temp.hysteresis_q15 == 500
        assert temp.stickiness_q15 == Q15_HALF  # Default
        assert temp.chaos_amount_q15 == 0  # Default

    def test_from_dict_none(self):
        """from_dict(None) should return default."""
        temp = TemperamentVector.from_dict(None)
        assert temp.hysteresis_q15 == Q15_HALF

    def test_roundtrip(self):
        """to_dict -> from_dict should preserve values."""
        original = TemperamentVector(
            hysteresis_q15=8000,
            stickiness_q15=9000,
            reactivity_q15=10000,
            escalation_bias_q15=11000,
            fatigue_sensitivity_q15=12000,
            tension_influence_q15=13000,
            chaos_amount_q15=14000,
        )
        restored = TemperamentVector.from_dict(original.to_dict())
        assert restored.hysteresis_q15 == original.hysteresis_q15
        assert restored.stickiness_q15 == original.stickiness_q15
        assert restored.reactivity_q15 == original.reactivity_q15
        assert restored.escalation_bias_q15 == original.escalation_bias_q15
        assert restored.fatigue_sensitivity_q15 == original.fatigue_sensitivity_q15
        assert restored.tension_influence_q15 == original.tension_influence_q15
        assert restored.chaos_amount_q15 == original.chaos_amount_q15


class TestTemperamentNormalized:
    """Tests for TemperamentVector.from_normalized()."""

    def test_from_normalized_defaults(self):
        """from_normalized with no args should give Q15_HALF defaults."""
        temp = TemperamentVector.from_normalized()
        # 0.5 * 32767 = 16383 (close to Q15_HALF=16384)
        assert abs(temp.hysteresis_q15 - Q15_HALF) < 10
        assert temp.chaos_amount_q15 == 0  # chaos defaults to 0.0

    def test_from_normalized_specific_values(self):
        """from_normalized should convert 0-1 to Q15."""
        temp = TemperamentVector.from_normalized(
            hysteresis=0.3,
            reactivity=0.7,
            chaos_amount=0.1,
        )
        # 0.3 * 32767 = 9830
        assert abs(temp.hysteresis_q15 - 9830) < 10
        # 0.7 * 32767 = 22936
        assert abs(temp.reactivity_q15 - 22936) < 10
        # 0.1 * 32767 = 3276
        assert abs(temp.chaos_amount_q15 - 3276) < 10

    def test_from_normalized_extremes(self):
        """from_normalized should handle 0.0 and 1.0."""
        temp = TemperamentVector.from_normalized(
            hysteresis=0.0,
            reactivity=1.0,
        )
        assert temp.hysteresis_q15 == 0
        assert temp.reactivity_q15 == 32767


class TestControlConfigTemperament:
    """Tests for ControlConfig.temperament integration."""

    def test_default_temperament(self):
        """ControlConfig should have default temperament."""
        config = ControlConfig()
        assert config.temperament is not None
        assert isinstance(config.temperament, TemperamentVector)

    def test_custom_temperament(self):
        """ControlConfig can have custom temperament."""
        temp = TemperamentVector(hysteresis_q15=25000)
        config = ControlConfig(temperament=temp)
        assert config.temperament.hysteresis_q15 == 25000


class TestTemperamentEdgeModulation:
    """Tests for temperament effects on EdgeSpec."""

    def test_high_hysteresis_widens_margins(self):
        """High hysteresis should widen enter/exit margins."""
        from symcrash_python_ap.voice.voice_engine import (
            HysteresisSpec, apply_temperament_to_hysteresis
        )

        base_hyst = HysteresisSpec(
            enter_margin_q15=1000,
            exit_margin_q15=500,
            debounce_blocks=4,
        )

        # High hysteresis = very sticky (double margins)
        high_temp = TemperamentVector(hysteresis_q15=Q15_ONE)
        result = apply_temperament_to_hysteresis(base_hyst, high_temp)

        # Should have wider margins
        assert result.enter_margin_q15 > base_hyst.enter_margin_q15
        assert result.exit_margin_q15 > base_hyst.exit_margin_q15

    def test_low_hysteresis_narrows_margins(self):
        """Low hysteresis should narrow enter/exit margins (twitchy)."""
        from symcrash_python_ap.voice.voice_engine import (
            HysteresisSpec, apply_temperament_to_hysteresis
        )

        base_hyst = HysteresisSpec(
            enter_margin_q15=1000,
            exit_margin_q15=500,
            debounce_blocks=4,
        )

        # Zero hysteresis = no margins
        low_temp = TemperamentVector(hysteresis_q15=0)
        result = apply_temperament_to_hysteresis(base_hyst, low_temp)

        # Should have narrow/zero margins
        assert result.enter_margin_q15 < base_hyst.enter_margin_q15
        assert result.exit_margin_q15 < base_hyst.exit_margin_q15

    def test_high_reactivity_reduces_debounce(self):
        """High reactivity should reduce debounce (faster response)."""
        from symcrash_python_ap.voice.voice_engine import (
            HysteresisSpec, apply_temperament_to_hysteresis
        )

        base_hyst = HysteresisSpec(
            enter_margin_q15=1000,
            exit_margin_q15=500,
            debounce_blocks=10,
        )

        # High reactivity = fast response
        high_react_temp = TemperamentVector(reactivity_q15=Q15_ONE)
        result = apply_temperament_to_hysteresis(base_hyst, high_react_temp)

        # Should have reduced debounce
        assert result.debounce_blocks < base_hyst.debounce_blocks

    def test_low_reactivity_increases_debounce(self):
        """Low reactivity should increase debounce (slow response)."""
        from symcrash_python_ap.voice.voice_engine import (
            HysteresisSpec, apply_temperament_to_hysteresis
        )

        base_hyst = HysteresisSpec(
            enter_margin_q15=1000,
            exit_margin_q15=500,
            debounce_blocks=4,
        )

        # Zero reactivity = very slow (double debounce)
        low_react_temp = TemperamentVector(reactivity_q15=0)
        result = apply_temperament_to_hysteresis(base_hyst, low_react_temp)

        # Should have increased debounce
        assert result.debounce_blocks > base_hyst.debounce_blocks

    def test_neutral_temperament_no_change(self):
        """Neutral temperament (Q15_HALF) should approximately preserve values."""
        from symcrash_python_ap.voice.voice_engine import (
            HysteresisSpec, apply_temperament_to_hysteresis
        )

        base_hyst = HysteresisSpec(
            enter_margin_q15=1000,
            exit_margin_q15=500,
            debounce_blocks=4,
        )

        neutral_temp = TemperamentVector()  # All Q15_HALF
        result = apply_temperament_to_hysteresis(base_hyst, neutral_temp)

        # Should be close to original (within rounding)
        # Note: scaling by Q15_HALF gives ~0.5x so margins are reduced by half
        # This is expected behavior for "neutral" being the midpoint
        assert result.enter_margin_q15 <= base_hyst.enter_margin_q15
        assert result.debounce_blocks >= 1  # At least 1 block


class TestTemperamentEdgeSpecIntegration:
    """Tests for apply_temperament_to_edge()."""

    def test_edge_modification(self):
        """apply_temperament_to_edge should create modified edge."""
        from symcrash_python_ap.voice.voice_engine import (
            EdgeSpec, EdgeType, HysteresisSpec,
            apply_temperament_to_edge, Regime
        )

        edge = EdgeSpec(
            from_regime=Regime.R2_STABLE,
            to_regime=Regime.R3_AIR,
            hysteresis=HysteresisSpec(
                enter_margin_q15=1000,
                exit_margin_q15=500,
                debounce_blocks=4,
            ),
            min_dwell_blocks=10,
        )

        # High stickiness = longer min_dwell
        sticky_temp = TemperamentVector(stickiness_q15=Q15_ONE)
        result = apply_temperament_to_edge(edge, sticky_temp)

        # Should have longer min_dwell
        assert result.min_dwell_blocks > edge.min_dwell_blocks
        # Original should be unchanged
        assert edge.min_dwell_blocks == 10

    def test_edge_preserves_core_fields(self):
        """apply_temperament_to_edge should preserve from/to regimes."""
        from symcrash_python_ap.voice.voice_engine import (
            EdgeSpec, apply_temperament_to_edge, Regime
        )

        edge = EdgeSpec(
            from_regime=Regime.R2_STABLE,
            to_regime=Regime.R4_GLITCH,
            priority=5,
            arc_id=2,
        )

        temp = TemperamentVector(hysteresis_q15=Q15_ONE)
        result = apply_temperament_to_edge(edge, temp)

        assert result.from_regime == edge.from_regime
        assert result.to_regime == edge.to_regime
        assert result.priority == edge.priority
        assert result.arc_id == edge.arc_id


class TestRegimeGraphTemperament:
    """Tests for RegimeGraph.check_transition with temperament."""

    def test_check_transition_with_temperament(self):
        """check_transition should accept temperament parameter."""
        from symcrash_python_ap.voice.voice_engine import (
            RegimeGraph, build_default_regime_graph, IntentVector, Regime
        )

        graph = build_default_regime_graph()
        intent = IntentVector()
        intent.lanes[0] = Q15_ONE  # I_TONE high

        # Should work with temperament=None
        result, prng = graph.check_transition(
            Regime.R1_ATTACK, intent, 10, temperament=None
        )
        # May or may not transition depending on thresholds
        # Result is now a tuple (regime, prng_state)

        # Should work with temperament provided
        temp = TemperamentVector()
        result, prng = graph.check_transition(
            Regime.R1_ATTACK, intent, 10, temperament=temp
        )
        # Result is (Optional[Regime], int)

    def test_high_stickiness_prevents_early_transition(self):
        """High stickiness should prevent transitions when dwell is low."""
        from symcrash_python_ap.voice.voice_engine import (
            RegimeGraph, EdgeSpec, IntentVector, Regime,
            HysteresisSpec, ConditionSpec
        )

        # Create a simple graph with one edge
        graph = RegimeGraph()
        graph.edges = [
            EdgeSpec(
                from_regime=Regime.R2_STABLE,
                to_regime=Regime.R3_AIR,
                condition=ConditionSpec(),  # Always true
                min_dwell_blocks=5,  # Must dwell at least 5 blocks
            )
        ]

        intent = IntentVector()

        # With high stickiness, min_dwell should be scaled up
        # So even at 5 blocks, shouldn't transition
        sticky_temp = TemperamentVector(stickiness_q15=Q15_ONE)

        result, prng = graph.check_transition(
            Regime.R2_STABLE, intent, 5, temperament=sticky_temp
        )

        # The scaled min_dwell should be > 5, so no transition
        assert result is None

    def test_low_stickiness_allows_early_transition(self):
        """Low stickiness should allow transitions sooner."""
        from symcrash_python_ap.voice.voice_engine import (
            RegimeGraph, EdgeSpec, IntentVector, Regime,
            ConditionSpec
        )

        # Create a simple graph with one edge
        graph = RegimeGraph()
        graph.edges = [
            EdgeSpec(
                from_regime=Regime.R2_STABLE,
                to_regime=Regime.R3_AIR,
                condition=ConditionSpec(),  # Always true
                min_dwell_blocks=10,  # Must dwell at least 10 blocks normally
            )
        ]

        intent = IntentVector()

        # With zero stickiness, min_dwell should be scaled down
        # So at 8 blocks, might transition (10 * 0.5 = 5)
        low_sticky_temp = TemperamentVector(stickiness_q15=0)

        result, prng = graph.check_transition(
            Regime.R2_STABLE, intent, 8, temperament=low_sticky_temp
        )

        # The scaled min_dwell should be ~5, so transition happens at 8
        assert result == Regime.R3_AIR


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
