"""
WSE-STD-050 Degradation Conformance Tests
==========================================

Tests for degradation and voice scheduling semantics.
"""

import pytest
from src.symcrash_python_ap.degradation import (
    # Constants
    ENERGY_EPS_Q15,
    CLOUD_DECAY_Q15,
    RETIRE_EPS_Q15,
    RETIRE_BLOCKS,
    # Enums
    VoiceClass,
    VoiceStatus,
    EventDropPolicy,
    # State
    VoiceState,
    FreezeState,
    # Semantics
    apply_freeze,
    apply_cloud_decay,
    check_retirement,
    # Scheduler
    VoiceScheduler,
    # Events
    handle_skipped_voice_events,
    # Validation
    DeterminismValidator,
    check_compliance,
    # Behavior tables
    get_skip_behavior,
    get_service_behavior,
)


# =============================================================================
# VOICE CLASSIFICATION TESTS (Section 3)
# =============================================================================

class TestVoiceClassification:
    """Test voice classification per Section 3."""

    def test_gate_active_is_foreground(self):
        """Voice with non-zero gate should be Foreground."""
        voice = VoiceState(voice_id=0, gate=True)
        assert voice.classify() == VoiceClass.FOREGROUND

    def test_intent_updated_is_foreground(self):
        """Voice with IntentFrame update should be Foreground."""
        voice = VoiceState(voice_id=0, intent_updated=True)
        assert voice.classify() == VoiceClass.FOREGROUND

    def test_energy_above_threshold_is_foreground(self):
        """Voice with energy > ENERGY_EPS should be Foreground."""
        voice = VoiceState(voice_id=0, energy_q15=ENERGY_EPS_Q15 + 100)
        assert voice.classify() == VoiceClass.FOREGROUND

    def test_energy_at_threshold_is_cloud(self):
        """Voice with energy == ENERGY_EPS should be Cloud."""
        voice = VoiceState(voice_id=0, energy_q15=ENERGY_EPS_Q15)
        assert voice.classify() == VoiceClass.CLOUD

    def test_solo_is_foreground(self):
        """Voice marked SOLO should be Foreground."""
        voice = VoiceState(voice_id=0, solo=True)
        assert voice.classify() == VoiceClass.FOREGROUND

    def test_armed_is_foreground(self):
        """Voice marked ARMED should be Foreground."""
        voice = VoiceState(voice_id=0, armed=True)
        assert voice.classify() == VoiceClass.FOREGROUND

    def test_no_conditions_is_cloud(self):
        """Voice with no Foreground conditions should be Cloud."""
        voice = VoiceState(voice_id=0)
        assert voice.classify() == VoiceClass.CLOUD

    def test_classification_recomputed_each_block(self):
        """Classification should change when conditions change."""
        voice = VoiceState(voice_id=0, gate=True)
        assert voice.classify() == VoiceClass.FOREGROUND

        voice.gate = False
        assert voice.classify() == VoiceClass.CLOUD

    def test_multiple_conditions(self):
        """Multiple Foreground conditions should still be Foreground."""
        voice = VoiceState(
            voice_id=0,
            gate=True,
            energy_q15=ENERGY_EPS_Q15 + 1000,
            solo=True,
        )
        assert voice.classify() == VoiceClass.FOREGROUND


# =============================================================================
# FREEZE SEMANTICS TESTS (Section 5.1)
# =============================================================================

class TestFreezeSemantic:
    """Test freeze semantics for Foreground voices."""

    def test_apply_freeze_captures_state(self):
        """Freeze should capture all timing state."""
        voice = VoiceState(
            voice_id=0,
            adsr_phase=2,
            adsr_time=100,
            regime_id=3,
            regime_dwell=50,
            energy_q15=10000,
        )

        frozen = apply_freeze(voice)

        assert frozen.adsr_phase == 2
        assert frozen.adsr_time == 100
        assert frozen.regime_id == 3
        assert frozen.regime_dwell == 50
        assert frozen.energy_q15 == 10000
        assert frozen.coefficients_frozen

    def test_freeze_preserves_voice_state(self):
        """Freeze should not modify voice state."""
        voice = VoiceState(
            voice_id=0,
            adsr_time=100,
            energy_q15=10000,
        )

        original_time = voice.adsr_time
        original_energy = voice.energy_q15

        apply_freeze(voice)

        assert voice.adsr_time == original_time
        assert voice.energy_q15 == original_energy

    def test_skip_behavior_foreground(self):
        """Foreground skip behavior should be all frozen."""
        behavior = get_skip_behavior(VoiceClass.FOREGROUND)

        assert behavior['adsr'] == 'frozen'
        assert behavior['regime'] == 'frozen'
        assert behavior['prng'] == 'frozen'
        assert behavior['energy'] == 'frozen'
        assert behavior['audio'] == 'held'


# =============================================================================
# DECAY SEMANTICS TESTS (Section 5.2)
# =============================================================================

class TestDecaySemantic:
    """Test decay semantics for Cloud voices."""

    def test_apply_cloud_decay(self):
        """Cloud decay should reduce energy deterministically."""
        voice = VoiceState(voice_id=0, energy_q15=32767)

        new_energy = apply_cloud_decay(voice)

        # Should be energy * 0.98
        expected = (32767 * CLOUD_DECAY_Q15) >> 15
        assert new_energy == expected
        assert voice.energy_q15 == expected

    def test_cloud_decay_deterministic(self):
        """Decay should be deterministic (same input = same output)."""
        voice1 = VoiceState(voice_id=0, energy_q15=16384)
        voice2 = VoiceState(voice_id=1, energy_q15=16384)

        e1 = apply_cloud_decay(voice1)
        e2 = apply_cloud_decay(voice2)

        assert e1 == e2

    def test_cloud_decay_converges_to_zero(self):
        """Repeated decay should converge to zero."""
        voice = VoiceState(voice_id=0, energy_q15=32767)

        for _ in range(500):  # Many blocks
            apply_cloud_decay(voice)

        assert voice.energy_q15 < RETIRE_EPS_Q15

    def test_skip_behavior_cloud(self):
        """Cloud skip behavior should have decaying energy."""
        behavior = get_skip_behavior(VoiceClass.CLOUD)

        assert behavior['adsr'] == 'frozen'
        assert behavior['regime'] == 'frozen'
        assert behavior['prng'] == 'frozen'
        assert behavior['energy'] == 'decays'
        assert behavior['audio'] == 'held'


# =============================================================================
# RETIREMENT TESTS (Section 9)
# =============================================================================

class TestRetirement:
    """Test voice retirement per Section 9."""

    def test_retirement_requires_cloud(self):
        """Only Cloud voices can be retired."""
        voice = VoiceState(
            voice_id=0,
            gate=True,  # Foreground
            energy_q15=0,
        )
        voice.classify()

        # Even with zero energy, Foreground can't retire
        for _ in range(RETIRE_BLOCKS + 5):
            check_retirement(voice)

        assert not voice.retired

    def test_retirement_requires_low_energy(self):
        """Retirement requires energy < RETIRE_EPS."""
        voice = VoiceState(voice_id=0, energy_q15=1000)
        voice.classify()

        for _ in range(RETIRE_BLOCKS + 5):
            check_retirement(voice)

        assert not voice.retired

    def test_retirement_requires_n_blocks(self):
        """Retirement requires N consecutive blocks below threshold."""
        voice = VoiceState(voice_id=0, energy_q15=0)
        voice.classify()

        for i in range(RETIRE_BLOCKS - 1):
            check_retirement(voice)
            assert not voice.retired

        check_retirement(voice)
        assert voice.retired

    def test_retirement_counter_resets(self):
        """Retirement counter should reset if energy rises."""
        voice = VoiceState(voice_id=0, energy_q15=0)
        voice.classify()

        # Almost retire
        for _ in range(RETIRE_BLOCKS - 1):
            check_retirement(voice)

        # Energy rises
        voice.energy_q15 = RETIRE_EPS_Q15 + 100
        check_retirement(voice)

        # Counter should be reset
        assert voice.below_retire_count == 0
        assert not voice.retired


# =============================================================================
# SCHEDULER TESTS (Section 4)
# =============================================================================

class TestVoiceScheduler:
    """Test voice scheduler per Section 4."""

    def test_allocate_voice(self):
        """Scheduler should allocate voices."""
        scheduler = VoiceScheduler()
        state = scheduler.allocate_voice(5)

        assert state.voice_id == 5
        assert 5 in scheduler.active_voices
        assert scheduler.get_voice(5) is state

    def test_deallocate_voice(self):
        """Scheduler should deallocate voices."""
        scheduler = VoiceScheduler()
        scheduler.allocate_voice(5)

        scheduler.deallocate_voice(5)

        assert 5 not in scheduler.active_voices
        assert scheduler.get_voice(5) is None

    def test_schedule_all_fit(self):
        """All voices should be serviced if budget allows."""
        scheduler = VoiceScheduler()
        for i in range(5):
            scheduler.allocate_voice(i)

        serviced, skipped = scheduler.schedule_block(budget=10, cost_per_voice=1)

        assert len(serviced) == 5
        assert len(skipped) == 0

    def test_schedule_budget_limit(self):
        """Voices should be skipped when budget exhausted."""
        scheduler = VoiceScheduler()
        for i in range(5):
            scheduler.allocate_voice(i)

        serviced, skipped = scheduler.schedule_block(budget=3, cost_per_voice=1)

        assert len(serviced) == 3
        assert len(skipped) == 2

    def test_rolling_cursor(self):
        """Scheduler should use rolling cursor for fairness."""
        scheduler = VoiceScheduler()
        for i in range(4):
            scheduler.allocate_voice(i)

        # First block: service first 2
        serviced1, _ = scheduler.schedule_block(budget=2, cost_per_voice=1)

        # Second block: should start from where we left off
        serviced2, _ = scheduler.schedule_block(budget=2, cost_per_voice=1)

        # All voices should get serviced across two blocks
        all_serviced = set(serviced1 + serviced2)
        assert len(all_serviced) == 4

    def test_classify_all(self):
        """Scheduler should classify all voices."""
        scheduler = VoiceScheduler()

        v0 = scheduler.allocate_voice(0)
        v0.gate = True  # Foreground

        v1 = scheduler.allocate_voice(1)
        # Cloud

        fg, cloud = scheduler.classify_all()

        assert 0 in fg
        assert 1 in cloud

    def test_apply_skip_semantics_foreground(self):
        """Skip semantics for Foreground should preserve state."""
        scheduler = VoiceScheduler()
        v0 = scheduler.allocate_voice(0)
        v0.gate = True
        v0.energy_q15 = 10000
        v0.classify()

        original_energy = v0.energy_q15

        scheduler.apply_skip_semantics([0], block_num=1)

        # Foreground energy should not change
        assert v0.energy_q15 == original_energy

    def test_apply_skip_semantics_cloud(self):
        """Skip semantics for Cloud should decay energy."""
        scheduler = VoiceScheduler()
        v0 = scheduler.allocate_voice(0)
        # Set energy at threshold (not above) so voice is Cloud
        v0.energy_q15 = ENERGY_EPS_Q15
        v0.classify()

        # Verify it's classified as Cloud
        assert v0.is_cloud()

        original_energy = v0.energy_q15

        scheduler.apply_skip_semantics([0], block_num=1)

        # Cloud energy should decay
        assert v0.energy_q15 < original_energy

    def test_retire_eligible(self):
        """Scheduler should retire eligible voices."""
        scheduler = VoiceScheduler()
        v0 = scheduler.allocate_voice(0)
        v0.energy_q15 = 0
        v0.classify()

        # Trigger retirement
        for _ in range(RETIRE_BLOCKS):
            check_retirement(v0)

        retired = scheduler.retire_eligible()

        assert 0 in retired
        assert 0 not in scheduler.active_voices


# =============================================================================
# EVENT HANDLING TESTS (Section 6)
# =============================================================================

class TestEventHandling:
    """Test event handling for skipped voices."""

    def test_events_dropped_for_skipped_voice(self):
        """All events should be dropped for skipped voices."""
        voice = VoiceState(voice_id=0)
        events = [1, 2, 3, 4, 5]

        result = handle_skipped_voice_events(voice, events)

        assert result == []

    def test_no_event_queuing(self):
        """Events should NOT be queued (per spec rationale)."""
        voice = VoiceState(voice_id=0)

        # Simulate multiple blocks of skipped events
        for _ in range(5):
            result = handle_skipped_voice_events(voice, [1, 2, 3])
            assert result == []


# =============================================================================
# DETERMINISM TESTS (Section 10)
# =============================================================================

class TestDeterminism:
    """Test determinism guarantees per Section 10."""

    def test_identical_runs_match(self):
        """Identical inputs should produce identical state."""
        def run_simulation(seed: int):
            scheduler = VoiceScheduler()
            validator = DeterminismValidator()

            # Allocate voices
            for i in range(4):
                v = scheduler.allocate_voice(i)
                v.energy_q15 = 10000 + i * 1000

            # Run several blocks with limited budget
            for block in range(10):
                scheduler.classify_all()
                serviced, skipped = scheduler.schedule_block(budget=2, cost_per_voice=1)
                scheduler.apply_skip_semantics(skipped, block)
                scheduler.apply_service_semantics(serviced, block)
                validator.capture_state(scheduler, block)

            return validator.state_snapshots

        run1 = run_simulation(42)
        run2 = run_simulation(42)

        validator = DeterminismValidator()
        is_identical, msg = validator.compare_runs(run1, run2)
        assert is_identical, msg

    def test_different_inputs_differ(self):
        """Different inputs should produce different state."""
        def run_simulation(budget: int):
            scheduler = VoiceScheduler()
            validator = DeterminismValidator()

            for i in range(4):
                v = scheduler.allocate_voice(i)
                # Set energy at threshold (Cloud) so decay occurs when skipped
                v.energy_q15 = ENERGY_EPS_Q15

            for block in range(5):
                scheduler.classify_all()
                serviced, skipped = scheduler.schedule_block(budget=budget, cost_per_voice=1)
                scheduler.apply_skip_semantics(skipped, block)
                validator.capture_state(scheduler, block)

            return validator.state_snapshots

        run1 = run_simulation(budget=4)  # All serviced (no decay)
        run2 = run_simulation(budget=2)  # Some skipped (Cloud decay)

        # Cloud voices should have different energy due to decay when skipped
        validator = DeterminismValidator()
        is_identical, _ = validator.compare_runs(run1, run2)
        assert not is_identical


# =============================================================================
# COMPLIANCE TESTS (Section 13)
# =============================================================================

class TestCompliance:
    """Test compliance checking per Section 13."""

    def test_compliant_implementation(self):
        """Standard implementation should be compliant."""
        scheduler = VoiceScheduler()
        scheduler.allocate_voice(0)
        scheduler.allocate_voice(1)

        serviced, skipped = scheduler.schedule_block(budget=1, cost_per_voice=1)

        is_compliant, violations = check_compliance(scheduler, serviced, skipped)
        assert is_compliant
        assert len(violations) == 0


# =============================================================================
# BEHAVIOR TABLE TESTS (Section 11)
# =============================================================================

class TestBehaviorTables:
    """Test behavior tables per Section 11."""

    def test_foreground_skip_behavior(self):
        """Foreground skip behavior should match spec table."""
        behavior = get_skip_behavior(VoiceClass.FOREGROUND)

        assert behavior['adsr'] == 'frozen'
        assert behavior['regime'] == 'frozen'
        assert behavior['prng'] == 'frozen'
        assert behavior['energy'] == 'frozen'
        assert behavior['audio'] == 'held'

    def test_cloud_skip_behavior(self):
        """Cloud skip behavior should match spec table."""
        behavior = get_skip_behavior(VoiceClass.CLOUD)

        assert behavior['adsr'] == 'frozen'
        assert behavior['regime'] == 'frozen'
        assert behavior['prng'] == 'frozen'
        assert behavior['energy'] == 'decays'
        assert behavior['audio'] == 'held'

    def test_service_behavior(self):
        """Service behavior should match spec table."""
        behavior = get_service_behavior()

        assert behavior['adsr'] == 'advances'
        assert behavior['regime'] == 'advances'
        assert behavior['prng'] == 'advances'
        assert behavior['energy'] == 'updated'
        assert behavior['audio'] == 'recomputed'


# =============================================================================
# INTEGRATION WITH PRNG TESTS
# =============================================================================

class TestPRNGIntegration:
    """Test PRNG integration with degradation."""

    def test_skipped_voice_prng_frozen(self):
        """Skipped voices should have frozen PRNG (verified by behavior table)."""
        fg_behavior = get_skip_behavior(VoiceClass.FOREGROUND)
        cloud_behavior = get_skip_behavior(VoiceClass.CLOUD)

        assert fg_behavior['prng'] == 'frozen'
        assert cloud_behavior['prng'] == 'frozen'

    def test_serviced_voice_prng_advances(self):
        """Serviced voices should have advancing PRNG."""
        behavior = get_service_behavior()
        assert behavior['prng'] == 'advances'
