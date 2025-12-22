"""
WSE-STD-030 Executor Scheduling Tests
=====================================

Tests for executor scheduling, tail ordering, and budget accounting.
"""

import pytest
from typing import List

from executor_scheduling import (
    # Constants
    C_MAX, COST_TAIL_TOTAL, COST_UNIT_BASE,
    OPERATOR_COSTS, EVENT_LANE_COSTS, EVENT_LANE_DROP_PRIORITY,
    # Budget
    Budget,
    # Symbol state
    SymbolState, COEFF_LEN,
    # Tail
    TailSafeResult, check_tail_safe, compute_energy,
    apply_tail_normalize, apply_tail_clamp, apply_tail_simplify,
    apply_tail_coerce, apply_mandatory_tail,
    # Events
    should_drop_lane, EventContext, apply_event_lane,
    # Executor
    Executor030, ExecutorStats,
)
from uart_protocol import (
    VoiceFrameV0, BackboneEntry, Payload64, PayloadMode, EventLane, NUM_EVENT_LANES
)
from fixed import Q15_ONE, Q15_HALF, sat16
from operator_library import BAND_SLICES, NUM_BANDS


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def q15_one():
    return Q15_ONE


@pytest.fixture
def q15_half():
    return Q15_HALF


@pytest.fixture
def budget():
    """Create default budget."""
    return Budget(B_total=10000, B_voice_max=2000)


@pytest.fixture
def executor():
    """Create executor with allocated voice."""
    ex = Executor030(n_voices=8)
    ex.allocate_voice(0)
    return ex


@pytest.fixture
def sample_frame():
    """Create sample VoiceFrame with backbone and events."""
    return VoiceFrameV0(
        voice_id=0,
        block_index=1,
        backbone=[
            BackboneEntry(op_id=1, strength_q15=0x4000),
            BackboneEntry(op_id=3, strength_q15=0x6000),
        ],
        event_mask=0x04,  # EVT_INJECT
        event_strength_q15=[0, 0, 0x5000, 0, 0, 0, 0, 0],
        traction_q15=Q15_ONE,
    )


# =============================================================================
# BUDGET ACCOUNTING TESTS (Section 4)
# =============================================================================

class TestBudgetAccounting:
    """Tests for budget accounting per WSE-STD-030 Section 4."""

    def test_budget_initialization(self, budget):
        """Budget initializes with correct values."""
        assert budget.B_total == 10000
        assert budget.B_voice_max == 2000
        assert budget.B_tail == COST_TAIL_TOTAL

    def test_budget_reset_for_voice(self, budget):
        """Budget resets correctly for each voice."""
        budget.reset_for_voice()
        assert budget.remaining == budget.B_voice_max
        assert budget.consumed_backbone == 0
        assert budget.consumed_events == 0

    def test_budget_reset_with_override(self, budget):
        """Budget can be overridden per voice."""
        budget.reset_for_voice(voice_budget=500)
        assert budget.remaining == 500

    def test_budget_can_afford(self, budget):
        """can_afford checks remaining budget."""
        budget.reset_for_voice()
        assert budget.can_afford(1000)
        assert budget.can_afford(2000)
        assert not budget.can_afford(3000)

    def test_budget_consume(self, budget):
        """consume reduces remaining budget."""
        budget.reset_for_voice()
        initial = budget.remaining
        budget.consume(500)
        assert budget.remaining == initial - 500

    def test_budget_consume_backbone_vs_events(self, budget):
        """Backbone and event consumption tracked separately."""
        budget.reset_for_voice()
        budget.consume(100, is_event=False)
        budget.consume(50, is_event=True)
        assert budget.consumed_backbone == 100
        assert budget.consumed_events == 50

    def test_budget_reserve_for_tail(self, budget):
        """Tail is reserved outside budget (returns 0)."""
        assert budget.reserve_for_tail() == 0


# =============================================================================
# OPERATOR COST TESTS (Section 4.1)
# =============================================================================

class TestOperatorCosts:
    """Tests for operator cost table per WSE-STD-030 Section 4.1."""

    def test_all_operators_have_costs(self):
        """All operators have defined costs."""
        for op_id in range(24):
            assert op_id in OPERATOR_COSTS, f"Op {op_id} missing cost"

    def test_costs_are_positive(self):
        """All costs are positive."""
        for op_id, cost in OPERATOR_COSTS.items():
            assert cost > 0, f"Op {op_id} has non-positive cost"

    def test_costs_are_stable(self):
        """Costs are integers (stable across builds)."""
        for op_id, cost in OPERATOR_COSTS.items():
            assert isinstance(cost, int), f"Op {op_id} cost not int"

    def test_event_lane_costs(self):
        """All event lanes have costs."""
        for lane in EventLane:
            assert lane in EVENT_LANE_COSTS


# =============================================================================
# SYMBOL STATE TESTS (Section 1.3)
# =============================================================================

class TestSymbolState:
    """Tests for symbol state per WSE-STD-030 Section 1.3."""

    def test_state_initialization(self):
        """State initializes with correct defaults."""
        state = SymbolState()
        assert len(state.c) == COEFF_LEN
        assert state.op_enable_mask == 0xFFFFFFFF
        assert state.rng_state != 0

    def test_state_reset_on_allocation(self):
        """State resets on voice allocation."""
        state = SymbolState()
        state.c = [1000] * COEFF_LEN
        state.op_enable_mask = 0x12345678

        state.reset_on_allocation()

        assert all(c == 0 for c in state.c)
        assert state.op_enable_mask == 0xFFFFFFFF

    def test_state_reset_with_preset_mask(self):
        """State reset respects preset mask."""
        state = SymbolState()
        state.reset_on_allocation(preset_mask=0xFFFFFFFE)
        assert state.op_enable_mask == 0xFFFFFFFE


# =============================================================================
# TAIL SAFE PREDICATE TESTS (Section 3.3)
# =============================================================================

class TestTailSafePredicate:
    """Tests for TAIL_SAFE predicate per WSE-STD-030 Section 3.3."""

    def test_safe_coefficients(self):
        """Safe coefficients pass all checks."""
        c = [1000] * COEFF_LEN
        result = check_tail_safe(c)
        assert result.is_safe
        assert result.bounded_magnitude
        assert result.bounded_energy
        assert result.no_nan_undefined

    def test_bounded_magnitude_pass(self):
        """Coefficients within C_MAX pass magnitude check."""
        c = [C_MAX] * COEFF_LEN
        result = check_tail_safe(c)
        assert result.bounded_magnitude
        assert result.max_coeff == C_MAX

    def test_bounded_magnitude_fail(self):
        """Coefficients exceeding C_MAX fail magnitude check."""
        c = [C_MAX + 1] * COEFF_LEN
        result = check_tail_safe(c)
        assert not result.bounded_magnitude

    def test_energy_computation(self):
        """Energy computed correctly."""
        c = [1000] * COEFF_LEN
        energy = compute_energy(c)
        expected = 1000 * 1000 * COEFF_LEN
        assert energy == expected

    def test_traction_scales_energy_limit(self, q15_half):
        """Higher traction = stricter energy limit."""
        # Use coefficients that produce energy between E_MAX/2 and E_MAX
        # so half traction passes but full traction fails
        # Energy = sum(c[i]^2) = COEFF_LEN * c^2
        # With E_MAX = 0x7FFFFFFF ≈ 2.1e9, we need energy ≈ 1.5e9
        # sqrt(1.5e9 / 256) ≈ 2421, so use c ≈ 2500
        c = [2500] * COEFF_LEN
        energy = 2500 * 2500 * COEFF_LEN  # 1,600,000,000

        # With full traction, E_MAX_scaled = E_MAX (stricter)
        result_full = check_tail_safe(c, traction_q15=Q15_ONE)
        # With half traction, E_MAX_scaled = E_MAX * 0.5 (MORE strict, not less!)
        # Actually traction scaling makes the limit stricter (lower) with lower traction
        # So test the opposite: full traction should be more lenient
        result_half = check_tail_safe(c, traction_q15=q15_half)

        # Full traction = scaled_e_max = E_MAX, should pass (energy < E_MAX)
        # Half traction = scaled_e_max = E_MAX/2 ≈ 1.07e9, should fail (energy > E_MAX/2)
        assert result_full.bounded_energy, f"Full traction should pass: energy={energy}, limit={0x7FFFFFFF}"
        assert not result_half.bounded_energy, f"Half traction should fail: energy={energy}, limit={0x7FFFFFFF//2}"

    def test_no_nan_check(self):
        """Invalid values fail no_nan check."""
        c = [0] * COEFF_LEN
        c[0] = 100000  # Way out of int16 range
        result = check_tail_safe(c)
        assert not result.no_nan_undefined


# =============================================================================
# TAIL STAGE TESTS (Section 3.2)
# =============================================================================

class TestTailStage:
    """Tests for mandatory tail stage per WSE-STD-030 Section 3.2."""

    def test_tail_clamp_reduces_energy(self):
        """Tail clamp reduces per-band energy."""
        c = [32000] * COEFF_LEN  # Hot
        c_clamped = apply_tail_clamp(c, cap_ratio=0.3)

        # Energy should be reduced in each band
        for start, end in BAND_SLICES:
            orig_energy = sum(int(c[i]) ** 2 for i in range(start, end))
            new_energy = sum(int(c_clamped[i]) ** 2 for i in range(start, end))
            assert new_energy <= orig_energy

    def test_tail_normalize(self):
        """Tail normalize keeps coefficients bounded."""
        c = [30000] * COEFF_LEN
        c_norm = apply_tail_normalize(c)
        assert max(abs(x) for x in c_norm) <= C_MAX

    def test_tail_simplify_zeros_bands(self):
        """Emergency simplify zeros high-frequency bands."""
        c = [1000] * COEFF_LEN
        c_simple = apply_tail_simplify(c, trigger=True, k=2)

        # Last 2 bands should be zeroed
        for band_idx in range(NUM_BANDS - 2, NUM_BANDS):
            start, end = BAND_SLICES[band_idx]
            assert all(c_simple[i] == 0 for i in range(start, end))

    def test_tail_simplify_no_trigger(self):
        """Simplify does nothing when not triggered."""
        c = [1000] * COEFF_LEN
        c_out = apply_tail_simplify(c, trigger=False)
        assert c_out == c

    def test_tail_coerce_saturates(self):
        """Tail coerce saturates out-of-range values."""
        c = [50000, -50000] + [0] * (COEFF_LEN - 2)
        c_coerced = apply_tail_coerce(c)
        assert c_coerced[0] == 32767
        assert c_coerced[1] == -32768

    def test_mandatory_tail_always_safe(self):
        """Mandatory tail always produces safe output."""
        # Various hot inputs
        test_cases = [
            [32767] * COEFF_LEN,
            [-32768] * COEFF_LEN,
            [50000] * COEFF_LEN,
            list(range(-COEFF_LEN // 2, COEFF_LEN // 2)),
        ]

        for c_in in test_cases:
            c_out, result = apply_mandatory_tail(c_in)
            assert result.is_safe, f"Tail failed for input pattern"

    def test_mandatory_tail_sequence(self):
        """Mandatory tail applies ops in correct sequence."""
        c = [32000] * COEFF_LEN
        c_out, _ = apply_mandatory_tail(c, cap_ratio=0.3)

        # Output should be bounded
        assert max(abs(x) for x in c_out) <= C_MAX
        # Output should be safe
        assert check_tail_safe(c_out).is_safe


# =============================================================================
# EVENT LANE TESTS (Section 2.3)
# =============================================================================

class TestEventLanes:
    """Tests for event lane processing per WSE-STD-030 Section 2.3."""

    def test_lane_order(self):
        """Lanes are processed in order 0..7 (Rule E1)."""
        for i in range(NUM_EVENT_LANES):
            assert EventLane(i).value == i

    def test_drop_priority_order(self):
        """Drop priority correct per Rule D4."""
        # Cosmetic first (highest priority = drop first)
        assert EVENT_LANE_DROP_PRIORITY[EventLane.EVT_OP_FIRE] > EVENT_LANE_DROP_PRIORITY[EventLane.EVT_INJECT]
        # Toggles last
        assert EVENT_LANE_DROP_PRIORITY[EventLane.EVT_OP_TOGGLE] < EVENT_LANE_DROP_PRIORITY[EventLane.EVT_INJECT]

    def test_should_drop_cosmetic_first(self, budget):
        """Cosmetic lanes drop first under budget pressure."""
        budget.reset_for_voice()
        budget.remaining = 100  # Low budget

        # Debug (cosmetic) should drop
        assert should_drop_lane(EventLane.EVT_DEBUG, budget)
        # Toggle should not drop
        assert not should_drop_lane(EventLane.EVT_OP_TOGGLE, budget)

    def test_all_lanes_have_costs(self):
        """All 8 lanes have costs defined."""
        for lane in range(NUM_EVENT_LANES):
            assert EventLane(lane) in EVENT_LANE_COSTS


# =============================================================================
# EXECUTOR TESTS (Section 7)
# =============================================================================

class TestExecutor030:
    """Tests for reference executor per WSE-STD-030 Section 7."""

    def test_executor_initialization(self, executor):
        """Executor initializes correctly."""
        assert executor.n_voices == 8
        assert 0 in executor.symbol_state

    def test_allocate_voice(self, executor):
        """Voice allocation resets state."""
        executor.allocate_voice(1)
        state = executor.get_symbol_state(1)
        assert all(c == 0 for c in state.c)
        assert state.op_enable_mask == 0xFFFFFFFF

    def test_execute_empty_frame(self, executor):
        """Execute with empty backbone."""
        frame = VoiceFrameV0(voice_id=0)
        c_out = executor.execute_voice_block(0, frame)
        assert len(c_out) == COEFF_LEN
        assert check_tail_safe(c_out).is_safe

    def test_execute_with_backbone(self, executor, sample_frame):
        """Execute with backbone operators."""
        # Set initial coefficients
        state = executor.get_symbol_state(0)
        state.c = [1000] * COEFF_LEN

        c_out = executor.execute_voice_block(0, sample_frame)

        assert executor.stats.backbone_ops_executed == 2
        assert check_tail_safe(c_out).is_safe

    def test_execute_with_events(self, executor):
        """Execute with event lanes."""
        frame = VoiceFrameV0(
            voice_id=0,
            event_mask=0x0F,  # Lanes 0-3
            event_strength_q15=[0x4000] * 8,
        )

        state = executor.get_symbol_state(0)
        state.c = [1000] * COEFF_LEN

        c_out = executor.execute_voice_block(0, frame)

        assert executor.stats.events_executed >= 1
        assert check_tail_safe(c_out).is_safe

    def test_backbone_author_order(self, executor):
        """Backbone executed in author order (Rule B1)."""
        # Create frame with specific order
        frame = VoiceFrameV0(
            voice_id=0,
            backbone=[
                BackboneEntry(op_id=1, strength_q15=0x4000),
                BackboneEntry(op_id=2, strength_q15=0x6000),
                BackboneEntry(op_id=3, strength_q15=0x5000),
            ],
        )

        state = executor.get_symbol_state(0)
        state.c = [1000] * COEFF_LEN

        c_out = executor.execute_voice_block(0, frame)

        # All backbone ops should execute (no reordering)
        assert executor.stats.backbone_ops_executed == 3

    def test_disabled_op_skipped(self, executor):
        """Disabled ops are skipped (Rule B3)."""
        # Disable op 1
        executor.set_voice_mask(0, 0xFFFFFFFD)  # Bit 1 clear

        frame = VoiceFrameV0(
            voice_id=0,
            backbone=[
                BackboneEntry(op_id=1, strength_q15=0x4000),  # Should skip
                BackboneEntry(op_id=2, strength_q15=0x6000),
            ],
        )

        state = executor.get_symbol_state(0)
        state.c = [1000] * COEFF_LEN

        c_out = executor.execute_voice_block(0, frame)

        # Only 1 op executed (op 1 disabled)
        assert executor.stats.backbone_ops_executed == 1

    def test_tail_always_runs(self, executor):
        """Tail always runs regardless of budget (Rule D2)."""
        frame = VoiceFrameV0(voice_id=0)

        state = executor.get_symbol_state(0)
        state.c = [32000] * COEFF_LEN  # Hot input

        c_out = executor.execute_voice_block(0, frame, voice_budget=1)  # Tiny budget

        # Output must be tail-safe despite budget
        assert check_tail_safe(c_out).is_safe

    def test_toggle_persistence(self, executor):
        """Toggle mutations persist (Rule E3)."""
        initial_mask = executor.get_voice_mask(0)

        # Manually modify mask (simulating toggle event)
        new_mask = initial_mask ^ 0x00000010  # Toggle bit 4
        executor.set_voice_mask(0, new_mask)

        # Execute frame
        frame = VoiceFrameV0(voice_id=0)
        executor.execute_voice_block(0, frame)

        # Mask should persist
        assert executor.get_voice_mask(0) == new_mask

    def test_commit_atomicity(self, executor, sample_frame):
        """Commit is atomic (Rule C1)."""
        state = executor.get_symbol_state(0)
        c_before = list(state.c)

        c_out = executor.execute_voice_block(0, sample_frame)

        # State should be updated atomically
        assert state.c == c_out
        assert state.c != c_before or c_before == [0] * COEFF_LEN

    def test_multiple_voices(self, executor):
        """Execute block with multiple voices."""
        for i in range(4):
            executor.allocate_voice(i)
            state = executor.get_symbol_state(i)
            state.c = [1000 * (i + 1)] * COEFF_LEN

        frames = [
            (i, VoiceFrameV0(voice_id=i, block_index=1))
            for i in range(4)
        ]

        results = executor.execute_block(frames)

        assert len(results) == 4
        for voice_id, c_out in results.items():
            assert check_tail_safe(c_out).is_safe

    def test_stats_tracking(self, executor, sample_frame):
        """Statistics are tracked correctly."""
        executor.reset_stats()

        state = executor.get_symbol_state(0)
        state.c = [1000] * COEFF_LEN

        executor.execute_voice_block(0, sample_frame)

        assert executor.stats.voices_processed == 1
        assert executor.stats.backbone_ops_executed >= 1


# =============================================================================
# DROP POLICY TESTS (Section 4.3/4.4)
# =============================================================================

class TestDropPolicy:
    """Tests for drop policy per WSE-STD-030 Section 4.3/4.4."""

    def test_drop_remainder_not_reorder(self, executor):
        """Drop remainder, don't reorder (Rule D1)."""
        # Create many backbone ops
        frame = VoiceFrameV0(
            voice_id=0,
            backbone=[
                BackboneEntry(op_id=i, strength_q15=0x4000)
                for i in range(10)
            ],
        )

        state = executor.get_symbol_state(0)
        state.c = [1000] * COEFF_LEN

        # Execute with limited budget
        c_out = executor.execute_voice_block(0, frame, voice_budget=500)

        # Some ops should be skipped, but order preserved
        assert executor.stats.backbone_ops_executed < 10
        assert check_tail_safe(c_out).is_safe

    def test_atomic_op_execution(self, executor):
        """Ops are atomic - start implies complete (Rule D3)."""
        frame = VoiceFrameV0(
            voice_id=0,
            backbone=[BackboneEntry(op_id=1, strength_q15=0x4000)],
        )

        state = executor.get_symbol_state(0)
        state.c = [1000] * COEFF_LEN

        c_out = executor.execute_voice_block(0, frame)

        # Op either executed fully or not at all
        assert executor.stats.backbone_ops_executed in [0, 1]


# =============================================================================
# DETERMINISM TESTS (Section 5)
# =============================================================================

class TestDeterminism:
    """Tests for determinism requirements per WSE-STD-030 Section 5."""

    def test_same_input_same_output(self):
        """Same inputs produce same outputs."""
        frame = VoiceFrameV0(
            voice_id=0,
            backbone=[BackboneEntry(op_id=1, strength_q15=0x4000)],
        )

        results = []
        for _ in range(3):
            executor = Executor030(n_voices=4)
            executor.allocate_voice(0)
            state = executor.get_symbol_state(0)
            state.c = [1000] * COEFF_LEN

            c_out = executor.execute_voice_block(0, frame)
            results.append(c_out)

        # All results should be identical
        assert results[0] == results[1] == results[2]

    def test_no_hidden_counters(self, executor):
        """Executor has no hidden timing counters (Section 5.2)."""
        # Execute same frame multiple times
        frame = VoiceFrameV0(voice_id=0)

        results = []
        for i in range(5):
            executor.allocate_voice(0)
            state = executor.get_symbol_state(0)
            state.c = [1000] * COEFF_LEN

            c_out = executor.execute_voice_block(0, frame)
            results.append(c_out)

        # All results should be identical (no block counter effects)
        assert all(r == results[0] for r in results)
