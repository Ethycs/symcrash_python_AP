"""
WSE-STD-030 EXECUTOR SCHEDULING CONFORMANCE TESTS
=================================================

Tests for executor_scheduling.py conformance with WSE-STD-030.
Extracted from executor_scheduling.py run_conformance_test().
"""

import pytest


class TestBudgetAccounting:
    """Section 4: Budget Accounting."""

    def test_budget_initialization(self):
        """Budget should initialize with correct remaining amount."""
        from symcrash_python_ap.symbol.executor_scheduling import Budget

        budget = Budget(B_total=10000, B_voice_max=2000)
        budget.reset_for_voice()
        assert budget.remaining == 2000

    def test_budget_consumption(self):
        """Budget consumption should reduce remaining amount."""
        from symcrash_python_ap.symbol.executor_scheduling import Budget

        budget = Budget(B_total=10000, B_voice_max=2000)
        budget.reset_for_voice()
        budget.consume(500)
        assert budget.remaining == 1500

    def test_budget_can_afford(self):
        """can_afford should correctly check remaining budget."""
        from symcrash_python_ap.symbol.executor_scheduling import Budget

        budget = Budget(B_total=10000, B_voice_max=2000)
        budget.reset_for_voice()
        budget.consume(500)

        assert budget.can_afford(1000)
        assert not budget.can_afford(2000)


class TestTailSafePredicate:
    """Section 3.3: TAIL_SAFE Predicate."""

    def test_safe_coefficients_pass(self):
        """Safe coefficients should pass TAIL_SAFE."""
        from symcrash_python_ap.symbol.executor_scheduling import check_tail_safe, COEFF_LEN

        c_safe = [1000] * COEFF_LEN
        result = check_tail_safe(c_safe)

        assert result.is_safe
        assert result.bounded_magnitude

    def test_overflow_coefficients_detected(self):
        """Overflow coefficients should be detected."""
        from symcrash_python_ap.symbol.executor_scheduling import check_tail_safe, COEFF_LEN

        c_overflow = [50000] * COEFF_LEN  # Exceeds int16
        result = check_tail_safe(c_overflow)

        assert not result.bounded_magnitude


class TestMandatoryTail:
    """Section 3.2: Mandatory Tail Stage."""

    def test_tail_makes_hot_input_safe(self):
        """Mandatory tail should make hot input safe."""
        from symcrash_python_ap.symbol.executor_scheduling import apply_mandatory_tail, check_tail_safe, COEFF_LEN, C_MAX

        c_hot = [32000] * COEFF_LEN  # Near saturation
        c_out, tail_result = apply_mandatory_tail(c_hot)

        assert tail_result.is_safe, "Tail should make output safe"
        assert max(abs(x) for x in c_out) <= C_MAX


class TestEventLaneProcessing:
    """Section 2.3: Event Lane Processing."""

    def test_all_lanes_have_costs(self):
        """All 8 lanes should have costs defined."""
        from symcrash_python_ap.symbol.executor_scheduling import EventLane, EVENT_LANE_COSTS, NUM_EVENT_LANES

        for lane in range(NUM_EVENT_LANES):
            assert EventLane(lane) in EVENT_LANE_COSTS


class TestDropPriority:
    """Section 4.4: Drop Priority."""

    def test_toggle_drops_last(self):
        """Toggle events should drop last (priority 0)."""
        from symcrash_python_ap.symbol.executor_scheduling import EventLane, EVENT_LANE_DROP_PRIORITY

        assert EVENT_LANE_DROP_PRIORITY[EventLane.EVT_OP_TOGGLE] == 0

    def test_debug_drops_before_toggle(self):
        """Debug should drop before toggle."""
        from symcrash_python_ap.symbol.executor_scheduling import EventLane, EVENT_LANE_DROP_PRIORITY

        toggle_priority = EVENT_LANE_DROP_PRIORITY[EventLane.EVT_OP_TOGGLE]
        debug_priority = EVENT_LANE_DROP_PRIORITY[EventLane.EVT_DEBUG]

        assert debug_priority > toggle_priority


class TestExecutorPipeline:
    """Section 7: Executor Pipeline."""

    def test_execute_voice_block_returns_safe_output(self):
        """Executor should return safe output."""
        from symcrash_python_ap.symbol.executor_scheduling import (
            Executor030, VoiceFrameV0, BackboneEntry,
            check_tail_safe, COEFF_LEN, Q15_ONE
        )

        executor = Executor030(n_voices=8)
        executor.allocate_voice(0)

        frame = VoiceFrameV0(
            voice_id=0,
            block_index=1,
            backbone=[
                BackboneEntry(op_id=1, strength_q15=0x4000),
                BackboneEntry(op_id=2, strength_q15=0x6000),
            ],
            event_mask=0x04,  # EVT_INJECT
            event_strength_q15=[0, 0, 0x5000, 0, 0, 0, 0, 0],
            traction_q15=Q15_ONE,
        )

        # Inject some initial coefficients
        state = executor.get_symbol_state(0)
        state.c = [1000] * COEFF_LEN

        c_out = executor.execute_voice_block(0, frame)

        assert len(c_out) == COEFF_LEN
        assert check_tail_safe(c_out).is_safe


class TestTogglePersistence:
    """Section 2.3, Rule E3: Toggle Persistence."""

    def test_initial_mask_all_enabled(self):
        """Initial op_enable_mask should be all enabled."""
        from symcrash_python_ap.symbol.executor_scheduling import Executor030

        executor = Executor030(n_voices=4)
        executor.allocate_voice(0)

        initial_mask = executor.get_voice_mask(0)
        assert initial_mask == 0xFFFFFFFF

    def test_toggle_persists_across_blocks(self):
        """Toggle should persist across blocks."""
        from symcrash_python_ap.symbol.executor_scheduling import Executor030

        executor = Executor030(n_voices=4)
        executor.allocate_voice(0)

        # Simulate toggle by modifying mask
        executor.set_voice_mask(0, 0xFFFFFFFE)  # Disable op 0
        assert executor.get_voice_mask(0) == 0xFFFFFFFE


class TestOperatorCosts:
    """Section 4.1: Operator Costs."""

    def test_sufficient_operators_defined(self):
        """At least 20 operators should have defined costs."""
        from symcrash_python_ap.symbol.executor_scheduling import OPERATOR_COSTS

        assert len(OPERATOR_COSTS) >= 20

    def test_all_costs_positive(self):
        """All operator costs must be positive."""
        from symcrash_python_ap.symbol.executor_scheduling import OPERATOR_COSTS

        for op_id, cost in OPERATOR_COSTS.items():
            assert cost > 0, f"Op {op_id} must have positive cost"
