"""
Tests for CMMM Enrichment Gap G3: Stochastic Transition Kernel
==============================================================

Tests the stochastic transition kernel P_θ(s'|s,u) implementation.

Reference: CMMM-ENRICHMENT-DESIGN.md §3.4 and §4.6 (Phase P4)
"""

import pytest

from symcrash_python_ap.primitives import Q15_ONE, Q15_HALF, Regime
from symcrash_python_ap.instrument_ir import TemperamentVector
from symcrash_python_ap.vm.state import VMState, CMMMemory


class TestEdgeSpecStochasticFields:
    """Tests for EdgeSpec stochastic fields."""

    def test_default_base_probability_is_q15_one(self):
        """Default base_probability_q15 should be Q15_ONE (deterministic)."""
        from symcrash_python_ap.voice.voice_engine import EdgeSpec

        edge = EdgeSpec(
            from_regime=Regime.R2_STABLE,
            to_regime=Regime.R3_AIR,
        )

        assert edge.base_probability_q15 == Q15_ONE

    def test_default_intent_influences_is_empty(self):
        """Default stochastic_intent_influences should be empty dict."""
        from symcrash_python_ap.voice.voice_engine import EdgeSpec

        edge = EdgeSpec(
            from_regime=Regime.R2_STABLE,
            to_regime=Regime.R3_AIR,
        )

        assert edge.stochastic_intent_influences == {}

    def test_custom_base_probability(self):
        """EdgeSpec should accept custom base_probability_q15."""
        from symcrash_python_ap.voice.voice_engine import EdgeSpec

        edge = EdgeSpec(
            from_regime=Regime.R2_STABLE,
            to_regime=Regime.R3_AIR,
            base_probability_q15=Q15_HALF,  # 50% probability
        )

        assert edge.base_probability_q15 == Q15_HALF

    def test_custom_intent_influences(self):
        """EdgeSpec should accept custom stochastic_intent_influences."""
        from symcrash_python_ap.voice.voice_engine import EdgeSpec

        influences = {0: 5000, 2: -3000}  # I_TONE positive, I_AIR negative

        edge = EdgeSpec(
            from_regime=Regime.R2_STABLE,
            to_regime=Regime.R3_AIR,
            stochastic_intent_influences=influences,
        )

        assert edge.stochastic_intent_influences == influences


class TestComputeTransitionProbability:
    """Tests for compute_transition_probability function."""

    def test_base_probability_preserved(self):
        """With no modulation, base probability should be returned."""
        from symcrash_python_ap.voice.voice_engine import (
            EdgeSpec, compute_transition_probability, IntentVector
        )

        edge = EdgeSpec(
            from_regime=Regime.R2_STABLE,
            to_regime=Regime.R3_AIR,
            base_probability_q15=20000,  # ~61%
        )

        intent = IntentVector()  # All zeros
        memory = CMMMemory()
        memory.blocks_since_transition = 100  # Bypass hysteresis threshold
        temperament = TemperamentVector(
            chaos_amount_q15=0,  # No chaos
            tension_influence_q15=0,  # No tension influence
            fatigue_sensitivity_q15=0,  # No fatigue influence
        )

        prob, _ = compute_transition_probability(
            edge, intent, memory, temperament, prng_state=0x12345678
        )

        # Should be close to base probability (no modulation factors active)
        assert abs(prob - 20000) < 1000  # Should be very close

    def test_intent_influence_increases_probability(self):
        """Positive intent influence should increase probability."""
        from symcrash_python_ap.voice.voice_engine import (
            EdgeSpec, compute_transition_probability, IntentVector
        )

        edge = EdgeSpec(
            from_regime=Regime.R2_STABLE,
            to_regime=Regime.R3_AIR,
            base_probability_q15=10000,
            stochastic_intent_influences={0: 10000},  # I_TONE increases prob
        )

        # Low intent
        intent_low = IntentVector()
        intent_low.lanes[0] = 0

        memory = CMMMemory()
        memory.blocks_since_transition = 100  # Past hysteresis threshold
        temperament = TemperamentVector(
            chaos_amount_q15=0,
            tension_influence_q15=0,
            fatigue_sensitivity_q15=0,
        )

        prob_low, _ = compute_transition_probability(
            edge, intent_low, memory, temperament, 0x12345678
        )

        # High intent
        intent_high = IntentVector()
        intent_high.lanes[0] = Q15_ONE

        prob_high, _ = compute_transition_probability(
            edge, intent_high, memory, temperament, 0x12345678
        )

        # High intent should give higher probability
        assert prob_high > prob_low

    def test_intent_influence_decreases_probability(self):
        """Negative intent influence should decrease probability."""
        from symcrash_python_ap.voice.voice_engine import (
            EdgeSpec, compute_transition_probability, IntentVector
        )

        edge = EdgeSpec(
            from_regime=Regime.R2_STABLE,
            to_regime=Regime.R3_AIR,
            base_probability_q15=20000,
            stochastic_intent_influences={2: -10000},  # I_AIR decreases prob
        )

        memory = CMMMemory()
        memory.blocks_since_transition = 100
        temperament = TemperamentVector(
            chaos_amount_q15=0,
            tension_influence_q15=0,
            fatigue_sensitivity_q15=0,
        )

        # Low air intent
        intent_low = IntentVector()
        intent_low.lanes[2] = 0

        prob_low, _ = compute_transition_probability(
            edge, intent_low, memory, temperament, 0x12345678
        )

        # High air intent
        intent_high = IntentVector()
        intent_high.lanes[2] = Q15_ONE

        prob_high, _ = compute_transition_probability(
            edge, intent_high, memory, temperament, 0x12345678
        )

        # High air should give lower probability
        assert prob_high < prob_low

    def test_tension_increases_escalation_probability(self):
        """High tension should increase probability for escalating transitions."""
        from symcrash_python_ap.voice.voice_engine import (
            EdgeSpec, compute_transition_probability, IntentVector
        )

        # Escalating transition: R2 -> R4 (higher regime)
        edge = EdgeSpec(
            from_regime=Regime.R2_STABLE,
            to_regime=Regime.R4_GLITCH,
            base_probability_q15=10000,
        )

        intent = IntentVector()
        temperament = TemperamentVector(
            chaos_amount_q15=0,
            tension_influence_q15=Q15_ONE,  # Full tension influence
            fatigue_sensitivity_q15=0,
        )

        # Low tension
        memory_low = CMMMemory()
        memory_low.tension_q15 = 0
        memory_low.blocks_since_transition = 100

        prob_low, _ = compute_transition_probability(
            edge, intent, memory_low, temperament, 0x12345678
        )

        # High tension
        memory_high = CMMMemory()
        memory_high.tension_q15 = Q15_ONE
        memory_high.blocks_since_transition = 100

        prob_high, _ = compute_transition_probability(
            edge, intent, memory_high, temperament, 0x12345678
        )

        # High tension should increase escalation probability
        assert prob_high > prob_low

    def test_fatigue_decreases_probability(self):
        """High fatigue should decrease transition probability."""
        from symcrash_python_ap.voice.voice_engine import (
            EdgeSpec, compute_transition_probability, IntentVector
        )

        edge = EdgeSpec(
            from_regime=Regime.R2_STABLE,
            to_regime=Regime.R3_AIR,
            base_probability_q15=20000,
        )

        intent = IntentVector()
        temperament = TemperamentVector(
            chaos_amount_q15=0,
            tension_influence_q15=0,
            fatigue_sensitivity_q15=Q15_ONE,  # Full fatigue sensitivity
        )

        # Low fatigue
        memory_low = CMMMemory()
        memory_low.fatigue_q15 = 0
        memory_low.blocks_since_transition = 100

        prob_low, _ = compute_transition_probability(
            edge, intent, memory_low, temperament, 0x12345678
        )

        # High fatigue
        memory_high = CMMMemory()
        memory_high.fatigue_q15 = Q15_ONE
        memory_high.blocks_since_transition = 100

        prob_high, _ = compute_transition_probability(
            edge, intent, memory_high, temperament, 0x12345678
        )

        # High fatigue should decrease probability
        assert prob_high < prob_low

    def test_hysteresis_reduces_early_probability(self):
        """Low dwell time (below hysteresis) should reduce probability."""
        from symcrash_python_ap.voice.voice_engine import (
            EdgeSpec, compute_transition_probability, IntentVector
        )

        edge = EdgeSpec(
            from_regime=Regime.R2_STABLE,
            to_regime=Regime.R3_AIR,
            base_probability_q15=Q15_ONE,  # Should always transition
        )

        intent = IntentVector()
        temperament = TemperamentVector(
            hysteresis_q15=Q15_ONE,  # Max hysteresis (~32 blocks threshold)
            chaos_amount_q15=0,
            tension_influence_q15=0,
            fatigue_sensitivity_q15=0,
        )

        # Very low dwell (below hysteresis threshold)
        memory_low = CMMMemory()
        memory_low.blocks_since_transition = 2

        prob_low, _ = compute_transition_probability(
            edge, intent, memory_low, temperament, 0x12345678
        )

        # High dwell (above hysteresis threshold)
        memory_high = CMMMemory()
        memory_high.blocks_since_transition = 100

        prob_high, _ = compute_transition_probability(
            edge, intent, memory_high, temperament, 0x12345678
        )

        # Low dwell should have reduced probability
        assert prob_low < prob_high

    def test_chaos_adds_variation(self):
        """Non-zero chaos should add variation to probability."""
        from symcrash_python_ap.voice.voice_engine import (
            EdgeSpec, compute_transition_probability, IntentVector
        )

        edge = EdgeSpec(
            from_regime=Regime.R2_STABLE,
            to_regime=Regime.R3_AIR,
            base_probability_q15=Q15_HALF,
        )

        intent = IntentVector()
        memory = CMMMemory()
        memory.blocks_since_transition = 100

        temperament = TemperamentVector(
            chaos_amount_q15=10000,  # Moderate chaos
            tension_influence_q15=0,
            fatigue_sensitivity_q15=0,
        )

        # Run multiple times with different PRNG states
        probs = []
        for seed in range(100):
            prob, _ = compute_transition_probability(
                edge, intent, memory, temperament, prng_state=seed * 12345
            )
            probs.append(prob)

        # With chaos, probabilities should vary
        min_prob = min(probs)
        max_prob = max(probs)
        assert max_prob - min_prob > 1000  # Should have noticeable variation

    def test_zero_chaos_is_deterministic(self):
        """Zero chaos should give deterministic probability."""
        from symcrash_python_ap.voice.voice_engine import (
            EdgeSpec, compute_transition_probability, IntentVector
        )

        edge = EdgeSpec(
            from_regime=Regime.R2_STABLE,
            to_regime=Regime.R3_AIR,
            base_probability_q15=20000,
        )

        intent = IntentVector()
        memory = CMMMemory()
        memory.blocks_since_transition = 100

        temperament = TemperamentVector(
            chaos_amount_q15=0,  # No chaos
            tension_influence_q15=0,
            fatigue_sensitivity_q15=0,
        )

        # Run multiple times with different PRNG states
        probs = set()
        for seed in range(10):
            prob, _ = compute_transition_probability(
                edge, intent, memory, temperament, prng_state=seed * 12345
            )
            probs.add(prob)

        # With no chaos, all probabilities should be the same
        assert len(probs) == 1


class TestShouldTransitionStochastic:
    """Tests for should_transition_stochastic function."""

    def test_high_probability_usually_transitions(self):
        """High probability (Q15_ONE) should almost always transition."""
        from symcrash_python_ap.voice.voice_engine import (
            EdgeSpec, should_transition_stochastic, IntentVector
        )

        edge = EdgeSpec(
            from_regime=Regime.R2_STABLE,
            to_regime=Regime.R3_AIR,
            base_probability_q15=Q15_ONE,  # 100%
        )

        intent = IntentVector()
        memory = CMMMemory()
        memory.blocks_since_transition = 100

        temperament = TemperamentVector(
            chaos_amount_q15=0,
            tension_influence_q15=0,
            fatigue_sensitivity_q15=0,
        )

        # Run many times
        transitions = 0
        prng = 0x12345678
        for _ in range(100):
            result, prng = should_transition_stochastic(
                edge, intent, memory, temperament, prng
            )
            if result:
                transitions += 1

        # Should transition almost every time
        assert transitions > 95

    def test_zero_probability_never_transitions(self):
        """Zero probability should never transition."""
        from symcrash_python_ap.voice.voice_engine import (
            EdgeSpec, should_transition_stochastic, IntentVector
        )

        edge = EdgeSpec(
            from_regime=Regime.R2_STABLE,
            to_regime=Regime.R3_AIR,
            base_probability_q15=0,  # 0%
        )

        intent = IntentVector()
        memory = CMMMemory()
        memory.blocks_since_transition = 100

        temperament = TemperamentVector(
            chaos_amount_q15=0,
            tension_influence_q15=0,
            fatigue_sensitivity_q15=0,
        )

        # Run many times
        transitions = 0
        prng = 0x12345678
        for _ in range(100):
            result, prng = should_transition_stochastic(
                edge, intent, memory, temperament, prng
            )
            if result:
                transitions += 1

        # Should never transition
        assert transitions == 0

    def test_half_probability_roughly_half_transitions(self):
        """50% probability should transition roughly half the time."""
        from symcrash_python_ap.voice.voice_engine import (
            EdgeSpec, should_transition_stochastic, IntentVector
        )

        edge = EdgeSpec(
            from_regime=Regime.R2_STABLE,
            to_regime=Regime.R3_AIR,
            base_probability_q15=Q15_HALF,  # 50%
        )

        intent = IntentVector()
        memory = CMMMemory()
        memory.blocks_since_transition = 100

        temperament = TemperamentVector(
            chaos_amount_q15=0,
            tension_influence_q15=0,
            fatigue_sensitivity_q15=0,
        )

        # Run many times
        transitions = 0
        prng = 0x12345678
        for _ in range(1000):
            result, prng = should_transition_stochastic(
                edge, intent, memory, temperament, prng
            )
            if result:
                transitions += 1

        # Should be roughly 50% (allow 35-65% for statistical variation)
        assert 350 < transitions < 650

    def test_prng_state_advances(self):
        """PRNG state should advance after each call."""
        from symcrash_python_ap.voice.voice_engine import (
            EdgeSpec, should_transition_stochastic, IntentVector
        )

        edge = EdgeSpec(
            from_regime=Regime.R2_STABLE,
            to_regime=Regime.R3_AIR,
        )

        intent = IntentVector()
        memory = CMMMemory()
        temperament = TemperamentVector()

        prng1 = 0x12345678
        _, prng2 = should_transition_stochastic(
            edge, intent, memory, temperament, prng1
        )

        # PRNG state should have changed
        assert prng2 != prng1


class TestRegimeGraphStochasticTransition:
    """Tests for RegimeGraph.check_transition with stochastic kernel."""

    def test_deterministic_by_default(self):
        """Default parameters should give deterministic behavior."""
        from symcrash_python_ap.voice.voice_engine import (
            RegimeGraph, EdgeSpec, IntentVector, ConditionSpec
        )

        graph = RegimeGraph()
        graph.edges = [
            EdgeSpec(
                from_regime=Regime.R2_STABLE,
                to_regime=Regime.R3_AIR,
                condition=ConditionSpec(),  # Always true
            )
        ]

        intent = IntentVector()

        # Run with different PRNG states
        results = set()
        for seed in range(10):
            result, _ = graph.check_transition(
                Regime.R2_STABLE, intent, dwell_blocks=10, prng_state=seed * 12345
            )
            results.add(result)

        # All should be the same (deterministic)
        assert len(results) == 1
        assert Regime.R3_AIR in results

    def test_stochastic_with_low_probability(self):
        """Low base_probability should sometimes block transitions."""
        from symcrash_python_ap.voice.voice_engine import (
            RegimeGraph, EdgeSpec, IntentVector, ConditionSpec
        )

        graph = RegimeGraph()
        graph.edges = [
            EdgeSpec(
                from_regime=Regime.R2_STABLE,
                to_regime=Regime.R3_AIR,
                condition=ConditionSpec(),  # Always true
                base_probability_q15=Q15_HALF,  # 50% chance
            )
        ]

        intent = IntentVector()
        memory = CMMMemory()
        memory.blocks_since_transition = 100
        temperament = TemperamentVector(
            chaos_amount_q15=0,
            tension_influence_q15=0,
            fatigue_sensitivity_q15=0,
        )

        # Run many times
        transitions = 0
        no_transitions = 0
        prng = 0x12345678
        for _ in range(100):
            result, prng = graph.check_transition(
                Regime.R2_STABLE, intent, dwell_blocks=10,
                temperament=temperament, memory=memory, prng_state=prng
            )
            if result == Regime.R3_AIR:
                transitions += 1
            else:
                no_transitions += 1

        # Should see a mix of transitions and non-transitions
        assert transitions > 20
        assert no_transitions > 20

    def test_returns_tuple(self):
        """check_transition should return (regime, prng_state) tuple."""
        from symcrash_python_ap.voice.voice_engine import (
            RegimeGraph, EdgeSpec, IntentVector, ConditionSpec
        )

        graph = RegimeGraph()
        graph.edges = [
            EdgeSpec(
                from_regime=Regime.R2_STABLE,
                to_regime=Regime.R3_AIR,
                condition=ConditionSpec(),
            )
        ]

        intent = IntentVector()
        result = graph.check_transition(
            Regime.R2_STABLE, intent, dwell_blocks=10
        )

        # Should be a tuple
        assert isinstance(result, tuple)
        assert len(result) == 2

        regime, prng = result
        assert regime == Regime.R3_AIR or regime is None
        assert isinstance(prng, int)


class TestDeterminismPreservation:
    """Tests to verify determinism with same seed + inputs."""

    def test_same_seed_same_result(self):
        """Same PRNG seed should produce same stochastic results."""
        from symcrash_python_ap.voice.voice_engine import (
            EdgeSpec, should_transition_stochastic, IntentVector
        )

        edge = EdgeSpec(
            from_regime=Regime.R2_STABLE,
            to_regime=Regime.R3_AIR,
            base_probability_q15=Q15_HALF,  # 50%
        )

        intent = IntentVector()
        memory = CMMMemory()
        memory.blocks_since_transition = 100

        temperament = TemperamentVector(
            chaos_amount_q15=5000,  # Some chaos
        )

        # Run sequence 1
        results1 = []
        prng = 0xDEADBEEF
        for _ in range(50):
            result, prng = should_transition_stochastic(
                edge, intent, memory, temperament, prng
            )
            results1.append(result)

        # Run sequence 2 with same seed
        results2 = []
        prng = 0xDEADBEEF
        for _ in range(50):
            result, prng = should_transition_stochastic(
                edge, intent, memory, temperament, prng
            )
            results2.append(result)

        # Sequences should be identical
        assert results1 == results2

    def test_different_seed_different_results(self):
        """Different PRNG seeds should produce different stochastic results."""
        from symcrash_python_ap.voice.voice_engine import (
            EdgeSpec, should_transition_stochastic, IntentVector
        )

        edge = EdgeSpec(
            from_regime=Regime.R2_STABLE,
            to_regime=Regime.R3_AIR,
            base_probability_q15=Q15_HALF,
        )

        intent = IntentVector()
        memory = CMMMemory()
        memory.blocks_since_transition = 100

        temperament = TemperamentVector(
            chaos_amount_q15=5000,
        )

        # Run with seed 1
        results1 = []
        prng = 0xDEADBEEF
        for _ in range(50):
            result, prng = should_transition_stochastic(
                edge, intent, memory, temperament, prng
            )
            results1.append(result)

        # Run with seed 2
        results2 = []
        prng = 0xCAFEBABE
        for _ in range(50):
            result, prng = should_transition_stochastic(
                edge, intent, memory, temperament, prng
            )
            results2.append(result)

        # Sequences should be different
        assert results1 != results2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
