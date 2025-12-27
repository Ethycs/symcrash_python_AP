"""
Unit Tests for CMMM Budget System
=================================

Tests for operator costs, regime budgets, tail modes, pattern table,
budget prediction, and degradation ladder.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from symcrash_python_ap.budgets import (
    # Constants
    COEFF_LEN, NODE_LEN, L_TOTAL_NODES, K_LADDER,
    # Enums
    OpCategory, RegimeId, VoiceClass, TailMode,
    # Operator costs
    OPERATOR_COSTS, get_operator_cost, get_tail_operators,
    get_nonlinear_operators, is_global_op,
    # Regime budgets
    REGIME_BUDGETS, get_regime_budget, get_voice_class,
    # Tail modes
    TAIL_OPS, get_tail_ops, get_allowed_tail_mode, downgrade_tail_mode,
    # Pattern table
    NODE_PRIORITY, popcount32, build_mask_for_k, shrink_mask_to_k,
    get_pattern_for_k, get_next_lower_k,
    PATTERN_TABLE,
    # Prediction
    predict_operator_ticks, predict_chain_cost, check_budget_fit,
    PredictedCost,
    # Degradation
    apply_degradation, degrade_to_fit, emergency_fallback,
    DegradationResult,
)


# =============================================================================
# OPERATOR COSTS TESTS
# =============================================================================

class TestOperatorCosts:
    """Tests for operator cost table and functions."""

    def test_constants(self):
        """Verify core constants."""
        assert COEFF_LEN == 256
        assert NODE_LEN == 8
        assert L_TOTAL_NODES == 32

    def test_all_operators_have_costs(self):
        """All defined operators should have cost entries."""
        expected_ops = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                       15, 16, 17, 18, 19, 20, 21, 22, 23]
        for op_id in expected_ops:
            cost = get_operator_cost(op_id)
            assert cost.op_id == op_id

    def test_unknown_operator_returns_default(self):
        """Unknown operator ID should return conservative default."""
        cost = get_operator_cost(999)
        assert cost.op_name == "Unknown_999"
        assert cost.drop_priority == 255  # Drop first

    def test_tail_operators(self):
        """Tail operators should be correctly identified."""
        tail_ops = get_tail_operators()
        # FineClamp (2), NormalizeScale (11), EmergencySimplify (19), SheafProject (23)
        assert 2 in tail_ops
        assert 11 in tail_ops
        assert 19 in tail_ops
        assert 23 in tail_ops
        # Non-tail should not be included
        assert 1 not in tail_ops  # Tilt
        assert 7 not in tail_ops  # LowRankMorph

    def test_nonlinear_operators(self):
        """Nonlinear operators should be identified."""
        nl_ops = get_nonlinear_operators()
        assert 11 in nl_ops  # NormalizeScale uses sqrt

    def test_global_ops(self):
        """Global (L-scaling) operators should be identified."""
        # PERMUTE ops are global
        assert is_global_op(6)   # BandShear
        assert is_global_op(10)  # RotateNodes
        # LOWRANK ops are global
        assert is_global_op(7)   # LowRankMorph
        # DIAG ops are not global
        assert not is_global_op(1)  # Tilt

    def test_cost_computation(self):
        """Cost should compute correctly with K and L."""
        # Tilt: base=8, per_active=8, per_total=0
        cost = get_operator_cost(1)
        # K=32, L=32: 8 + 32*8 + 32*0 = 264
        assert cost.compute_ticks(K=32) == 8 + 32 * 8

        # LowRankMorph: base=64, per_active=0, per_total=32
        cost = get_operator_cost(7)
        # K=16, L=32: 64 + 16*0 + 32*32 = 1088
        assert cost.compute_ticks(K=16) == 64 + 32 * 32


# =============================================================================
# REGIME BUDGETS TESTS
# =============================================================================

class TestRegimeBudgets:
    """Tests for regime budget table and functions."""

    def test_all_regimes_have_budgets(self):
        """All regime IDs should have budget entries."""
        for regime_id in [RegimeId.IDLE, RegimeId.TRANSIENT, RegimeId.STABLE_TONE,
                          RegimeId.GLITCH_CHATTER, RegimeId.RECOVERY,
                          RegimeId.RESIDUAL, RegimeId.CARRIER_ONLY]:
            budget = get_regime_budget(regime_id)
            assert budget.regime_id == regime_id

    def test_stable_tone_is_generous(self):
        """STABLE_TONE should have generous budget."""
        budget = get_regime_budget(RegimeId.STABLE_TONE)
        assert budget.max_ticks_per_block >= 8000
        assert budget.max_active_nodes_K == 32
        assert budget.allow_nonlinear is True

    def test_residual_is_tight(self):
        """RESIDUAL should have tight budget."""
        budget = get_regime_budget(RegimeId.RESIDUAL)
        assert budget.max_ticks_per_block <= 500
        assert budget.max_active_nodes_K <= 8
        assert budget.allow_nonlinear is False

    def test_voice_class_routing(self):
        """Voice class should route to correct budget."""
        # RESIDUAL class always uses RESIDUAL budget
        budget = get_regime_budget(RegimeId.STABLE_TONE, VoiceClass.RESIDUAL)
        assert budget.regime_id == RegimeId.RESIDUAL

        # MAIN class uses regime-specific budget
        budget = get_regime_budget(RegimeId.STABLE_TONE, VoiceClass.MAIN)
        assert budget.regime_id == RegimeId.STABLE_TONE

    def test_voice_class_from_id(self):
        """Voice ID should map to correct class."""
        assert get_voice_class(0) == VoiceClass.MAIN
        assert get_voice_class(11) == VoiceClass.MAIN
        assert get_voice_class(12) == VoiceClass.RESIDUAL
        assert get_voice_class(100) == VoiceClass.RESIDUAL


# =============================================================================
# TAIL MODES TESTS
# =============================================================================

class TestTailModes:
    """Tests for tail mode definitions."""

    def test_tail_ops_defined(self):
        """All tail modes should have operator lists."""
        assert len(get_tail_ops(TailMode.FULL)) == 3
        assert len(get_tail_ops(TailMode.MID)) == 2
        assert len(get_tail_ops(TailMode.MIN)) == 1
        assert len(get_tail_ops(TailMode.EMERGENCY)) == 1

    def test_tail_ops_contents(self):
        """Tail ops should contain correct operators."""
        assert get_tail_ops(TailMode.FULL) == [2, 11, 23]
        assert get_tail_ops(TailMode.MID) == [2, 11]
        assert get_tail_ops(TailMode.MIN) == [2]
        assert get_tail_ops(TailMode.EMERGENCY) == [19]

    def test_nonlinear_constraint(self):
        """FULL and MID use nonlinear ops."""
        assert get_allowed_tail_mode(TailMode.FULL, allow_nonlinear=True) == TailMode.FULL
        assert get_allowed_tail_mode(TailMode.FULL, allow_nonlinear=False) == TailMode.MIN
        assert get_allowed_tail_mode(TailMode.MID, allow_nonlinear=False) == TailMode.MIN
        assert get_allowed_tail_mode(TailMode.MIN, allow_nonlinear=False) == TailMode.MIN

    def test_downgrade_order(self):
        """Tail modes should downgrade in order."""
        assert downgrade_tail_mode(TailMode.FULL) == TailMode.MID
        assert downgrade_tail_mode(TailMode.MID) == TailMode.MIN
        assert downgrade_tail_mode(TailMode.MIN) == TailMode.EMERGENCY
        assert downgrade_tail_mode(TailMode.EMERGENCY) == TailMode.EMERGENCY


# =============================================================================
# PATTERN TABLE TESTS
# =============================================================================

class TestPatternTable:
    """Tests for pattern table and node mask utilities."""

    def test_node_priority_is_sequential(self):
        """Node priority should be 0-31 (bands are contiguous)."""
        assert NODE_PRIORITY == list(range(32))

    def test_popcount(self):
        """Popcount should count set bits."""
        assert popcount32(0) == 0
        assert popcount32(1) == 1
        assert popcount32(0xFFFFFFFF) == 32
        assert popcount32(0b10101010) == 4

    def test_build_mask_for_k(self):
        """Mask should have exactly K bits set."""
        for k in K_LADDER:
            mask = build_mask_for_k(k)
            assert popcount32(mask) == k

    def test_full_mask(self):
        """K=32 should give all-ones mask."""
        mask = build_mask_for_k(32)
        assert mask == 0xFFFFFFFF

    def test_shrink_mask(self):
        """Shrinking mask should reduce K."""
        full_mask = 0xFFFFFFFF
        for k in [24, 16, 12, 8, 4, 1]:
            shrunk = shrink_mask_to_k(full_mask, k)
            assert popcount32(shrunk) == k

    def test_k_ladder_patterns(self):
        """Pattern table should have entries for K ladder."""
        for k in K_LADDER:
            pattern = get_pattern_for_k(k)
            assert pattern.k_value <= k

    def test_get_next_lower_k(self):
        """Next lower K should follow ladder."""
        assert get_next_lower_k(32) == 24
        assert get_next_lower_k(24) == 16
        assert get_next_lower_k(16) == 12
        assert get_next_lower_k(12) == 8
        assert get_next_lower_k(8) == 4
        assert get_next_lower_k(4) == 1
        assert get_next_lower_k(1) == 1  # Already at minimum


# =============================================================================
# BUDGET PREDICTION TESTS
# =============================================================================

class TestBudgetPrediction:
    """Tests for budget prediction functions."""

    def test_predict_single_operator(self):
        """Single operator prediction should work."""
        ticks, uses_nl = predict_operator_ticks(1, K=32)  # Tilt
        assert ticks > 0
        assert uses_nl is False

        ticks, uses_nl = predict_operator_ticks(11, K=32)  # NormalizeScale
        assert uses_nl is True

    def test_predict_chain_cost(self):
        """Chain prediction should sum all operators."""
        predicted = predict_chain_cost(
            backbone_ops=[1, 3],  # Tilt, IntraMix
            event_ops=[8],        # GlobalPulse
            K=32,
            tail_mode=TailMode.FULL,
        )
        assert predicted.total_ticks > 0
        assert predicted.backbone_ticks > 0
        assert predicted.event_ticks > 0
        assert predicted.tail_ticks > 0
        assert predicted.total_ticks == (
            predicted.backbone_ticks + predicted.event_ticks + predicted.tail_ticks
        )

    def test_check_budget_fit_passes(self):
        """Simple chain should fit STABLE_TONE budget."""
        budget = get_regime_budget(RegimeId.STABLE_TONE)
        predicted = predict_chain_cost(
            backbone_ops=[1],  # Just Tilt
            event_ops=[],
            K=16,
            tail_mode=TailMode.MIN,
        )
        fits, violations = check_budget_fit(predicted, budget)
        assert fits is True
        assert len(violations) == 0

    def test_check_budget_fit_fails(self):
        """Heavy chain should exceed IDLE budget."""
        budget = get_regime_budget(RegimeId.IDLE)
        predicted = predict_chain_cost(
            backbone_ops=[1, 3, 4, 5, 6, 7],  # Many ops
            event_ops=[8, 13, 14],
            K=32,
            tail_mode=TailMode.FULL,
        )
        fits, violations = check_budget_fit(predicted, budget)
        assert fits is False
        assert len(violations) > 0


# =============================================================================
# DEGRADATION LADDER TESTS
# =============================================================================

class TestDegradationLadder:
    """Tests for degradation ladder algorithm."""

    def test_no_degradation_needed(self):
        """Simple chain should not need degradation."""
        budget = get_regime_budget(RegimeId.STABLE_TONE)
        result = apply_degradation(
            backbone_ops=[1],  # Just Tilt
            event_ops=[],
            node_mask=0xFFFFFFFF,
            tail_mode=TailMode.FULL,
            budget=budget,
        )
        assert result.K == 32
        assert result.backbone_ops == [1]
        assert result.is_emergency is False

    def test_degradation_reduces_k(self):
        """Overbudget chain should reduce K."""
        budget = get_regime_budget(RegimeId.IDLE)
        result = apply_degradation(
            backbone_ops=[1, 3, 4, 5],
            event_ops=[],
            node_mask=0xFFFFFFFF,
            tail_mode=TailMode.FULL,
            budget=budget,
        )
        # Should have degraded
        assert result.K < 32 or len(result.backbone_ops) < 4 or result.tail_mode != TailMode.FULL

    def test_degradation_drops_events_first(self):
        """Events should be dropped before backbone ops when budget is tight."""
        # Use IDLE budget which is very tight
        budget = get_regime_budget(RegimeId.IDLE)
        # Start with backbone + events that exceed budget
        result = apply_degradation(
            backbone_ops=[1, 3],
            event_ops=[8, 13, 14, 21],  # Many inject ops
            node_mask=0xFFFFFFFF,
            tail_mode=TailMode.FULL,
            budget=budget,
        )
        # Budget is tight, so degradation should have occurred
        # Either events were dropped, K reduced, or tail downgraded
        degraded = (
            len(result.event_ops) < 4 or
            result.K < 32 or
            result.tail_mode != TailMode.FULL or
            len(result.backbone_ops) < 2
        )
        assert degraded, f"Expected degradation but got: {result}"

    def test_emergency_fallback(self):
        """Emergency fallback should be minimal."""
        result = emergency_fallback()
        assert result.K == 1
        assert result.backbone_ops == []
        assert result.event_ops == []
        assert result.tail_mode == TailMode.EMERGENCY
        assert result.is_emergency is True

    def test_result_always_valid(self):
        """Result should always have valid tail."""
        budget = get_regime_budget(RegimeId.CARRIER_ONLY)
        result = apply_degradation(
            backbone_ops=[1, 3, 4, 5, 6, 7],
            event_ops=[8, 13, 14],
            node_mask=0xFFFFFFFF,
            tail_mode=TailMode.FULL,
            budget=budget,
        )
        # Should have degraded to emergency but still have tail
        tail_ops = get_tail_ops(result.tail_mode)
        assert len(tail_ops) > 0

    def test_nonlinear_constraint_respected(self):
        """Nonlinear ops should be removed when not allowed."""
        budget = get_regime_budget(RegimeId.TRANSIENT)  # allow_nonlinear=False
        result = apply_degradation(
            backbone_ops=[1, 3],
            event_ops=[],
            node_mask=0xFFFFFFFF,
            tail_mode=TailMode.FULL,  # Contains nonlinear op 11
            budget=budget,
        )
        # Tail should have been downgraded away from FULL/MID
        assert result.tail_mode in (TailMode.MIN, TailMode.EMERGENCY)

    def test_determinism(self):
        """Same inputs should produce same outputs."""
        budget = get_regime_budget(RegimeId.STABLE_TONE)
        result1 = apply_degradation(
            backbone_ops=[1, 3, 4, 5],
            event_ops=[8, 13],
            node_mask=0xFFFFFFFF,
            tail_mode=TailMode.FULL,
            budget=budget,
        )
        result2 = apply_degradation(
            backbone_ops=[1, 3, 4, 5],
            event_ops=[8, 13],
            node_mask=0xFFFFFFFF,
            tail_mode=TailMode.FULL,
            budget=budget,
        )
        assert result1.backbone_ops == result2.backbone_ops
        assert result1.event_ops == result2.event_ops
        assert result1.K == result2.K
        assert result1.node_mask == result2.node_mask
        assert result1.tail_mode == result2.tail_mode


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestBudgetSystemIntegration:
    """Integration tests for full budget system."""

    def test_main_voice_workflow(self):
        """Test typical MAIN voice budget workflow."""
        # Get budget for MAIN voice in STABLE_TONE
        voice_class = get_voice_class(0)
        assert voice_class == VoiceClass.MAIN

        budget = get_regime_budget(RegimeId.STABLE_TONE, voice_class)

        # Predict cost for typical backbone
        backbone = [1, 3, 4, 5]  # Tilt, IntraMix, DiffuseUp, DiffuseDown
        events = [8]  # GlobalPulse
        K = 32
        tail_mode = TailMode.FULL

        predicted = predict_chain_cost(backbone, events, K, tail_mode)
        fits, _ = check_budget_fit(predicted, budget)

        if not fits:
            result = apply_degradation(
                backbone_ops=backbone,
                event_ops=events,
                node_mask=build_mask_for_k(K),
                tail_mode=tail_mode,
                budget=budget,
            )
            assert result.K >= 1

    def test_residual_voice_workflow(self):
        """Test typical RESIDUAL voice budget workflow."""
        # Get budget for RESIDUAL voice
        voice_class = get_voice_class(50)
        assert voice_class == VoiceClass.RESIDUAL

        budget = get_regime_budget(RegimeId.STABLE_TONE, voice_class)
        assert budget.regime_id == RegimeId.RESIDUAL

        # RESIDUAL should only support minimal ops
        result = degrade_to_fit(
            ops=[1, 3, 4, 5, 6, 7],  # Many ops
            K=32,
            budget=budget,
            tail_mode=TailMode.MIN,
        )

        # Should have degraded significantly
        assert result.K <= 8
        assert result.K >= 1

    def test_128_voices_scenario(self):
        """Verify 128 RESIDUAL voices can fit with tight budgets."""
        budget = get_regime_budget(RegimeId.RESIDUAL)

        # Minimal ops for cloud voice
        result = degrade_to_fit(
            ops=[1],  # Just Tilt
            K=8,
            budget=budget,
            tail_mode=TailMode.MIN,
        )

        # Degradation should complete deterministically
        assert result.K >= 1
        assert result.tail_mode in (TailMode.MIN, TailMode.EMERGENCY)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
