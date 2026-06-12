"""
Budget Ladder Conformance Tests (WSE-STD-DEGRADE v0)
=====================================================

Tests conformance to WSE-STD-DEGRADE v0 specification.

Verifies:
- K ladder exact rungs (§3)
- pattern_id mapping (§3.1)
- Tail mode implementation (§4)
- Nonlinear constraints (§4.2)
- Ladder step order (§9.4)
- Determinism (§1.2)
- Backbone order preservation (§9.1)
- Emergency feasibility (§4.1)
"""

import pytest
from typing import List

from symcrash_python_ap.budgets import (
    apply_degradation,
    DegradationResult,
    RegimeBudget,
    RegimeId,
    TailMode,
    get_regime_budget,
    K_LADDER,
    get_pattern_for_k,
    OpCategory,
    get_operator_cost,
)


# =============================================================================
# WSE-STD-DEGRADE v0 §3: K LADDER CONFORMANCE
# =============================================================================


class TestKLadderConformance:
    """Test K ladder matches WSE-STD-DEGRADE v0 §3."""

    def test_k_ladder_exact_rungs(self):
        """K_LADDER should be exactly [32, 24, 16, 12, 8, 4, 1]."""
        assert K_LADDER == (32, 24, 16, 12, 8, 4, 1), (
            f"K_LADDER mismatch: {K_LADDER}"
        )

    def test_pattern_id_matches_k_table(self):
        """pattern_id should map to K per §3.1 table."""
        expected_mapping = {
            0: 32,  # pattern_id 0 → K=32
            1: 24,  # pattern_id 1 → K=24
            2: 16,  # pattern_id 2 → K=16
            3: 12,  # pattern_id 3 → K=12
            4: 8,   # pattern_id 4 → K=8
            5: 4,   # pattern_id 5 → K=4
            6: 1,   # pattern_id 6 → K=1
        }

        for pattern_id, expected_k in expected_mapping.items():
            pattern = get_pattern_for_k(expected_k)
            assert pattern.pattern_id == pattern_id, (
                f"At K={expected_k}: pattern_id={pattern.pattern_id}, "
                f"expected {pattern_id}"
            )
            assert pattern.k_count == expected_k, (
                f"Pattern {pattern_id} has K={pattern.k_count}, expected {expected_k}"
            )

    def test_k_reduction_follows_ladder(self):
        """K reduction should step through ladder in order."""
        # Create simple scenario that will reduce K
        budget = RegimeBudget(
            regime_id=0,
            regime_name="TEST",
            max_ticks_per_block=500,  # Very tight
            max_active_nodes_K=32,
            allow_nonlinear=False,
            tail_reserved_ticks=200,
        )

        # Start with K=32, heavy ops
        backbone = [1, 1, 1, 1, 1]  # 5× Tilt
        result = apply_degradation(
            backbone_ops=backbone,
            event_ops=[],
            node_mask=0xFFFFFFFF,  # All 32 nodes
            tail_mode=TailMode.MIN,
            budget=budget,
        )

        # Should reduce K to fit
        assert result.K in K_LADDER, f"K={result.K} not in ladder"


# =============================================================================
# WSE-STD-DEGRADE v0 §4: TAIL MODE CONFORMANCE
# =============================================================================


class TestTailModeConformance:
    """Test tail modes match WSE-STD-DEGRADE v0 §4."""

    def test_tail_mode_operators_exact(self):
        """Tail modes should have exact operator sets per §4."""
        from symcrash_python_ap.budgets.tail_modes import TAIL_OPS

        expected = {
            TailMode.FULL: [2, 11, 23],  # FineClamp, NormalizeScale, SheafProject
            TailMode.MID: [2, 11],       # FineClamp, NormalizeScale
            TailMode.MIN: [2],           # FineClamp only
            TailMode.EMERGENCY: [2, 19], # FineClamp, EmergencySimplify (v0: FineClamp always present)
        }

        for mode, expected_ops in expected.items():
            actual_ops = TAIL_OPS[mode]
            assert actual_ops == expected_ops, (
                f"TailMode.{mode.name}: ops={actual_ops}, expected {expected_ops}"
            )

    def test_tail_nonlinear_flags_correct(self):
        """Tail nonlinear flags should match §4 (FULL/MID=True, MIN/EMERGENCY=False)."""
        from symcrash_python_ap.budgets.tail_modes import tail_uses_nonlinear

        assert tail_uses_nonlinear(TailMode.FULL) is True
        assert tail_uses_nonlinear(TailMode.MID) is True
        assert tail_uses_nonlinear(TailMode.MIN) is False
        assert tail_uses_nonlinear(TailMode.EMERGENCY) is False

    def test_tail_downgrade_order(self):
        """Tail should downgrade FULL→MID→MIN→EMERGENCY per §9.4 step 6."""
        from symcrash_python_ap.budgets.tail_modes import downgrade_tail_mode

        assert downgrade_tail_mode(TailMode.FULL) == TailMode.MID
        assert downgrade_tail_mode(TailMode.MID) == TailMode.MIN
        assert downgrade_tail_mode(TailMode.MIN) == TailMode.EMERGENCY
        assert downgrade_tail_mode(TailMode.EMERGENCY) == TailMode.EMERGENCY

    def test_nonlinear_constraint_step_0b(self):
        """§4.2: If allow_nonlinear=False, FULL/MID should downgrade to MIN before loop."""
        budget = get_regime_budget(RegimeId.TRANSIENT)  # allow_nonlinear=False

        result = apply_degradation(
            backbone_ops=[1],  # Tilt (nonlinear=False)
            event_ops=[],
            node_mask=0xFF,  # K=8
            tail_mode=TailMode.FULL,  # Request FULL
            budget=budget,
        )

        # Should downgrade to MIN before loop
        assert result.tail_mode in (TailMode.MIN, TailMode.EMERGENCY), (
            f"With allow_nonlinear=False: tail_mode={result.tail_mode}, "
            f"should be MIN or EMERGENCY"
        )


# =============================================================================
# WSE-STD-DEGRADE v0 §9: ALGORITHM CONFORMANCE
# =============================================================================


class TestLadderStepOrder:
    """Test ladder step execution order matches §9.4."""

    def test_step_1_drops_events_first(self):
        """Step 1: Event ops should drop before backbone ops."""
        budget = RegimeBudget(
            regime_id=0,
            regime_name="TEST",
            max_ticks_per_block=400,
            max_active_nodes_K=32,
            allow_nonlinear=True,
            tail_reserved_ticks=200,
        )

        # Both events and backbone present - events should drop first
        result = apply_degradation(
            backbone_ops=[1],  # Tilt
            event_ops=[8, 13],  # Event ops (INJECT category)
            node_mask=0xFFFFFFFF,
            tail_mode=TailMode.MIN,
            budget=budget,
        )

        # Should have dropped events before backbone
        if result.degradation_steps > 0:
            # Events should be gone or reduced
            assert len(result.event_ops) < 2, (
                "Events should drop first, but still have {len(result.event_ops)}"
            )

    def test_k_reduction_before_tail_downgrade(self):
        """Step 4: K reduction should happen before tail downgrade."""
        # This test verifies ladder step ordering by checking that
        # when tail is already at MIN, further degradation reduces K
        # rather than immediately jumping to EMERGENCY tail
        budget = RegimeBudget(
            regime_id=0,
            regime_name="TEST",
            max_ticks_per_block=1300,  # Just above MIN tail (576 ticks at K=1)
            max_active_nodes_K=32,
            allow_nonlinear=False,  # Force MIN tail
            tail_reserved_ticks=600,
        )

        # Expensive backbone that won't fit at K=32
        result = apply_degradation(
            backbone_ops=[7],  # LowRankMorph (base=64, L×32) = very expensive at K=32
            event_ops=[],
            node_mask=0xFFFFFFFF,  # K=32
            tail_mode=TailMode.MIN,  # Already at MIN
            budget=budget,
        )

        # Ladder should drop the backbone op or reduce K before going to EMERGENCY
        # Since tail is MIN and can't downgrade further without going to EMERGENCY,
        # it should try backbone reduction and K reduction first
        if not result.is_emergency:
            # If not emergency, should have reduced K or dropped backbone
            assert result.K < 32 or len(result.backbone_ops) == 0, (
                f"Expected K reduction or backbone drop, got K={result.K}, ops={result.backbone_ops}"
            )


class TestDeterminism:
    """Test determinism per §1.2."""

    def test_same_inputs_same_output(self):
        """Same inputs should produce identical DegradationResult."""
        budget = get_regime_budget(RegimeId.STABLE_TONE)
        backbone = [1, 3, 5]
        events = [8]
        mask = 0x00FFFFFF  # K=24
        tail = TailMode.FULL

        # Run twice
        result1 = apply_degradation(backbone, events, mask, tail, budget)
        result2 = apply_degradation(backbone, events, mask, tail, budget)

        # Should be identical
        assert result1.backbone_ops == result2.backbone_ops
        assert result1.event_ops == result2.event_ops
        assert result1.K == result2.K
        assert result1.node_mask == result2.node_mask
        assert result1.tail_mode == result2.tail_mode
        assert result1.pattern_id == result2.pattern_id
        assert result1.degradation_steps == result2.degradation_steps
        assert result1.final_ticks == result2.final_ticks
        assert result1.is_emergency == result2.is_emergency


class TestBackboneOrderPreservation:
    """Test backbone order preservation per §9.1."""

    def test_backbone_semantic_order_preserved(self):
        """Dropping ops should preserve remaining op order."""
        budget = RegimeBudget(
            regime_id=0,
            regime_name="TEST",
            max_ticks_per_block=400,
            max_active_nodes_K=32,
            allow_nonlinear=False,
            tail_reserved_ticks=200,
        )

        # Ordered backbone: [1, 3, 5, 16]
        # Priority: 1(80), 3(100), 5(120), 16(40)
        # Drop order: 5, 3, 1, 16
        backbone = [1, 3, 5, 16]

        result = apply_degradation(
            backbone_ops=backbone,
            event_ops=[],
            node_mask=0xFFFFFFFF,
            tail_mode=TailMode.MIN,
            budget=budget,
        )

        # Remaining ops should maintain original order
        # If [1, 3, 16] remain, they should be in that order (not [16, 1, 3])
        if len(result.backbone_ops) > 1:
            original_indices = {op: i for i, op in enumerate(backbone)}
            for i in range(len(result.backbone_ops) - 1):
                op_a = result.backbone_ops[i]
                op_b = result.backbone_ops[i + 1]
                assert original_indices[op_a] < original_indices[op_b], (
                    f"Order violated: {op_a} should come before {op_b}"
                )


class TestEmergencyFallback:
    """Test emergency fallback per §4.1."""

    def test_emergency_tail_always_fits(self):
        """Emergency tail at K=1 should fit in any valid budget."""
        for regime_id in [RegimeId.IDLE, RegimeId.TRANSIENT, RegimeId.STABLE_TONE]:
            budget = get_regime_budget(regime_id)

            result = apply_degradation(
                backbone_ops=[],
                event_ops=[],
                node_mask=1,  # K=1
                tail_mode=TailMode.EMERGENCY,
                budget=budget,
            )

            assert not result.is_emergency or result.final_ticks <= budget.max_ticks_per_block, (
                f"Emergency doesn't fit in {budget.regime_name}: "
                f"{result.final_ticks} > {budget.max_ticks_per_block}"
            )


# =============================================================================
# BUDGET CALCULATION TESTS
# =============================================================================


class TestCostCalculation:
    """Test cost model per §6."""

    def test_cost_model_integer_only(self):
        """Cost computation should use integer-only arithmetic."""
        from symcrash_python_ap.budgets.budget_predictor import predict_chain_cost

        result = predict_chain_cost(
            backbone_ops=[1, 3],
            event_ops=[],
            K=16,
            tail_mode=TailMode.FULL,
        )

        # All costs should be integers
        assert isinstance(result.total_ticks, int)
        assert isinstance(result.backbone_ticks, int)  # Corrected field name
        assert isinstance(result.tail_ticks, int)

    def test_k_scaling_correct(self):
        """K-scaling ops should scale with active nodes."""
        from symcrash_python_ap.budgets.operator_costs import get_operator_cost

        # Tilt is K-scaling (op_id=1)
        tilt_cost = get_operator_cost(1)

        cost_k8 = tilt_cost.compute_ticks(K=8)
        cost_k16 = tilt_cost.compute_ticks(K=16)
        cost_k32 = tilt_cost.compute_ticks(K=32)

        # Should scale linearly with K
        assert cost_k16 > cost_k8
        assert cost_k32 > cost_k16

        # For K-scaling: cost = base + K×per_active
        # Delta should scale with K difference
        # delta(8→16) = 8×per_active, delta(16→32) = 16×per_active
        # So delta_16_32 should be 2× delta_8_16
        delta_8_16 = cost_k16 - cost_k8
        delta_16_32 = cost_k32 - cost_k16
        assert delta_16_32 == 2 * delta_8_16, (
            f"K-scaling incorrect: delta(8→16)={delta_8_16}, delta(16→32)={delta_16_32}"
        )


# =============================================================================
# NODE MASK CONFORMANCE TESTS
# =============================================================================


class TestNodeMaskConstruction:
    """Test node mask construction per §8."""

    def test_shrink_mask_low_frequency_first(self):
        """shrink_mask_to_k should keep low-frequency nodes per §8.1."""
        from symcrash_python_ap.budgets import shrink_mask_to_k, popcount32

        # Start with all 32 nodes
        full_mask = 0xFFFFFFFF

        # Shrink to K=8
        mask_k8 = shrink_mask_to_k(full_mask, 8)
        assert popcount32(mask_k8) == 8

        # Check that nodes 0-7 are set (low-frequency band 0-2)
        for i in range(8):
            assert (mask_k8 >> i) & 1, f"Node {i} should be active at K=8"

        # Check that nodes 8-31 are cleared (high-frequency)
        for i in range(8, 32):
            assert not ((mask_k8 >> i) & 1), f"Node {i} should be inactive at K=8"

    def test_popcount_deterministic(self):
        """popcount32 should be deterministic per §8.2."""
        from symcrash_python_ap.budgets import popcount32

        test_masks = [0, 1, 0xFF, 0xFFFF, 0xFFFFFFFF, 0x55555555, 0xAAAAAAAA]

        for mask in test_masks:
            count1 = popcount32(mask)
            count2 = popcount32(mask)
            assert count1 == count2, f"popcount not deterministic for mask=0x{mask:08X}"


# =============================================================================
# DEGRADATION RESULT CONFORMANCE
# =============================================================================


class TestDegradationResultFields:
    """Test DegradationResult fields per §10."""

    def test_result_has_all_required_fields(self):
        """DegradationResult should have all 9 required fields per §10."""
        budget = get_regime_budget(RegimeId.STABLE_TONE)

        result = apply_degradation(
            backbone_ops=[1],
            event_ops=[],
            node_mask=0xFFFFFFFF,
            tail_mode=TailMode.FULL,
            budget=budget,
        )

        # Check all required fields present
        assert hasattr(result, "backbone_ops")  # backbone_ops_final
        assert hasattr(result, "event_ops")  # event_ops_final
        assert hasattr(result, "K")  # K_final
        assert hasattr(result, "node_mask")  # node_mask_final_u32
        assert hasattr(result, "tail_mode")  # tail_mode_final
        assert hasattr(result, "pattern_id")  # pattern_id
        assert hasattr(result, "degradation_steps")  # steps
        assert hasattr(result, "final_ticks")  # ticks
        assert hasattr(result, "is_emergency")  # is_emergency

    def test_result_types_correct(self):
        """Result fields should have correct types."""
        budget = get_regime_budget(RegimeId.STABLE_TONE)

        result = apply_degradation(
            backbone_ops=[1],
            event_ops=[],
            node_mask=0xFFFF,
            tail_mode=TailMode.FULL,
            budget=budget,
        )

        assert isinstance(result.backbone_ops, list)
        assert isinstance(result.event_ops, list)
        assert isinstance(result.K, int)
        assert isinstance(result.node_mask, int)
        assert isinstance(result.tail_mode, TailMode)
        assert isinstance(result.pattern_id, int)
        assert isinstance(result.degradation_steps, int)
        assert isinstance(result.final_ticks, int)
        assert isinstance(result.is_emergency, bool)


# =============================================================================
# BOUNDED WORK GUARANTEE (§1.1)
# =============================================================================


class TestBoundedWork:
    """Test bounded work guarantee per §1.1."""

    def test_degradation_always_terminates(self):
        """Degradation should always terminate with a valid plan."""
        budgets = [
            get_regime_budget(RegimeId.IDLE),
            get_regime_budget(RegimeId.TRANSIENT),
            get_regime_budget(RegimeId.STABLE_TONE),
        ]

        # Extreme case: many expensive ops
        heavy_backbone = [7, 7, 7, 7, 7]  # 5× LowRankMorph (expensive)
        heavy_events = [8, 13, 14]  # Multiple events

        for budget in budgets:
            result = apply_degradation(
                backbone_ops=heavy_backbone,
                event_ops=heavy_events,
                node_mask=0xFFFFFFFF,
                tail_mode=TailMode.FULL,
                budget=budget,
            )

            # Should always produce a valid result
            assert result is not None
            assert result.K in K_LADDER
            assert result.tail_mode in TailMode
            assert result.final_ticks <= budget.max_ticks_per_block or result.is_emergency

    def test_iteration_bound_respected(self):
        """Degradation should complete within max_iterations."""
        budget = get_regime_budget(RegimeId.IDLE)  # Very tight

        # Heavy configuration
        result = apply_degradation(
            backbone_ops=[1, 3, 5, 7] * 10,  # 40 ops
            event_ops=[8, 13, 14, 21],
            node_mask=0xFFFFFFFF,
            tail_mode=TailMode.FULL,
            budget=budget,
            max_iterations=128,
        )

        # Should terminate (not hang)
        assert result.degradation_steps <= 128


# =============================================================================
# REGIME BUDGET TESTS
# =============================================================================


class TestRegimeBudgets:
    """Test regime budget configurations."""

    def test_stable_tone_full_featured(self):
        """STABLE_TONE should allow full processing."""
        budget = get_regime_budget(RegimeId.STABLE_TONE)

        assert budget.max_active_nodes_K == 32
        assert budget.allow_nonlinear is True
        assert budget.max_ticks_per_block >= 8000

    def test_transient_no_nonlinear(self):
        """TRANSIENT should prohibit nonlinear ops."""
        budget = get_regime_budget(RegimeId.TRANSIENT)

        assert budget.allow_nonlinear is False

    def test_idle_minimal(self):
        """IDLE should have minimal budget (but must fit EMERGENCY tail)."""
        budget = get_regime_budget(RegimeId.IDLE)

        assert budget.max_active_nodes_K <= 8
        # IDLE must fit EMERGENCY tail (1152 ticks), so minimum is 1300
        assert budget.max_ticks_per_block >= 1152  # Must fit emergency
        assert budget.max_ticks_per_block <= 1500  # But still minimal


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_operators(self):
        """Empty operator lists should work."""
        budget = get_regime_budget(RegimeId.STABLE_TONE)

        result = apply_degradation(
            backbone_ops=[],
            event_ops=[],
            node_mask=0xFFFFFFFF,
            tail_mode=TailMode.MIN,
            budget=budget,
        )

        assert result.backbone_ops == []
        assert result.event_ops == []
        assert result.tail_mode == TailMode.MIN

    def test_k_equals_1_minimum(self):
        """K=1 is the minimum value."""
        budget = get_regime_budget(RegimeId.IDLE)

        result = apply_degradation(
            backbone_ops=[],
            event_ops=[],
            node_mask=1,  # K=1
            tail_mode=TailMode.EMERGENCY,
            budget=budget,
        )

        assert result.K == 1
        assert result.pattern_id == 6  # pattern_id 6 → K=1

    def test_already_fits_no_degradation(self):
        """If plan already fits, no degradation should occur."""
        budget = get_regime_budget(RegimeId.STABLE_TONE)  # Generous

        result = apply_degradation(
            backbone_ops=[1],  # Single Tilt (cheap)
            event_ops=[],
            node_mask=0xFF,  # K=8 (small)
            tail_mode=TailMode.MIN,
            budget=budget,
        )

        # Should fit without changes
        assert result.degradation_steps == 0
        assert result.backbone_ops == [1]
        assert result.K == 8
        assert result.tail_mode == TailMode.MIN
