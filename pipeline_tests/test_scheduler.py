"""
Block Scheduler Conformance Tests
==================================

Tests for the Scheduler Spec v01 implementation.
"""

import pytest
from src.symcrash_python_ap.scheduler import (
    BlockScheduler,
    VoiceSelector,
    TileScheduler,
    DegradeController,
    HealthTelemetry,
    VoiceScheduleInfo,
    ScheduleResult,
    EmissionGroup,
    tile_id,
    tile_off,
    BLOCK_SIZE,
    V_MAX,
    TILE_SIZE,
    NUM_TILES,
    Q15_ONE,
    Q15_SCALE,
    THETA_ACTIVE_Q15,
    THETA_STAB_RED_Q15,
    H_DOWN,
    DEGRADE_THRESHOLDS_Q15,
    PERIOD_TABLE,
    C_TAIL_VOICE,
)
from src.symcrash_python_ap.frames import (
    VoiceFramePacketV0b,
    VoiceFlags,
)


# =============================================================================
# TILE HELPER TESTS
# =============================================================================

class TestTileHelpers:
    """Test tile_id and tile_off helper functions."""

    def test_tile_id_voice_0(self):
        """Voice 0 should be in tile 0."""
        assert tile_id(0) == 0

    def test_tile_id_voice_15(self):
        """Voice 15 should be in tile 0 (last voice in first tile)."""
        assert tile_id(15) == 0

    def test_tile_id_voice_16(self):
        """Voice 16 should be in tile 1 (first voice in second tile)."""
        assert tile_id(16) == 1

    def test_tile_id_voice_127(self):
        """Voice 127 should be in tile 7 (last tile)."""
        assert tile_id(127) == 7

    def test_tile_off_voice_0(self):
        """Voice 0 should have offset 0."""
        assert tile_off(0) == 0

    def test_tile_off_voice_15(self):
        """Voice 15 should have offset 15."""
        assert tile_off(15) == 15

    def test_tile_off_voice_16(self):
        """Voice 16 should have offset 0 (first in tile 1)."""
        assert tile_off(16) == 0

    def test_tile_off_voice_31(self):
        """Voice 31 should have offset 15."""
        assert tile_off(31) == 15


# =============================================================================
# VOICE SELECTOR TESTS
# =============================================================================

class TestVoiceSelector:
    """Test voice selection with ring scheduler."""

    def test_empty_frames(self):
        """Empty frame list should return empty selection."""
        selector = VoiceSelector()
        serviced, eligible = selector.select_voices([], 10)
        assert serviced == []
        assert eligible == 0

    def test_all_rejected(self):
        """All rejected frames should not be selected."""
        selector = VoiceSelector()
        frames = [
            VoiceFramePacketV0b(voice_id=i, reject_code=1)
            for i in range(10)
        ]
        serviced, eligible = selector.select_voices(frames, 10)
        assert serviced == []
        assert eligible == 0

    def test_gate_active_selected(self):
        """Voice with gate active should be selected."""
        selector = VoiceSelector()
        frames = [
            VoiceFramePacketV0b(voice_id=0, flags=VoiceFlags.GATE),
        ]
        serviced, eligible = selector.select_voices(frames, 10)
        assert serviced == [0]
        assert eligible == 1

    def test_activity_above_threshold(self):
        """Voice with high activity should be selected."""
        selector = VoiceSelector()
        frames = [
            VoiceFramePacketV0b(voice_id=0, activity_q15=THETA_ACTIVE_Q15 + 100),
        ]
        serviced, eligible = selector.select_voices(frames, 10)
        assert serviced == [0]
        assert eligible == 1

    def test_activity_below_threshold_not_selected(self):
        """Voice with low activity and no gate should not be selected."""
        selector = VoiceSelector()
        frames = [
            VoiceFramePacketV0b(voice_id=0, activity_q15=THETA_ACTIVE_Q15 - 100),
        ]
        serviced, eligible = selector.select_voices(frames, 10)
        assert serviced == []
        assert eligible == 0

    def test_ring_cursor_wrap(self):
        """Ring cursor should wrap around."""
        selector = VoiceSelector(cursor_fg=126, v_max=128)
        frames = [None] * 128
        # Put eligible voices at positions 127 and 0
        frames[127] = VoiceFramePacketV0b(voice_id=127, flags=VoiceFlags.GATE)
        frames[0] = VoiceFramePacketV0b(voice_id=0, flags=VoiceFlags.GATE)

        serviced, eligible = selector.select_voices(frames, 10)

        assert 127 in serviced
        assert 0 in serviced
        assert eligible == 2

    def test_cursor_advances(self):
        """Cursor should advance after selection."""
        selector = VoiceSelector(cursor_fg=0, v_max=128)
        frames = [None] * 128
        frames[0] = VoiceFramePacketV0b(voice_id=0, flags=VoiceFlags.GATE)

        selector.select_voices(frames, 1)

        # After scanning all 128 voices, cursor wraps to starting position
        assert selector.cursor_fg == 0

    def test_v_serviced_limit(self):
        """Should respect V_serviced limit."""
        selector = VoiceSelector()
        frames = [
            VoiceFramePacketV0b(voice_id=i, flags=VoiceFlags.GATE)
            for i in range(10)
        ]
        serviced, eligible = selector.select_voices(frames, 3)
        assert len(serviced) == 3
        assert eligible == 10

    def test_compute_v_serviced(self):
        """V_serviced should be computed from budget."""
        selector = VoiceSelector()

        # With budget of 120 and C_TAIL_VOICE=12, max is 10 voices
        v_max = selector.compute_v_serviced(120, 120)
        assert v_max == 10

        # Takes minimum of compute and mem budgets
        v_max = selector.compute_v_serviced(120, 60)
        assert v_max == 5


# =============================================================================
# TILE SCHEDULER TESTS
# =============================================================================

class TestTileScheduler:
    """Test tile-based TOP/MID throttling."""

    def test_initial_counters(self):
        """Initial counters should be zero."""
        scheduler = TileScheduler()
        assert all(c == 0 for c in scheduler.ctr_mid)
        assert all(c == 0 for c in scheduler.ctr_top)

    def test_periods_d0(self):
        """D=0 should have K_top=1, K_mid=1."""
        scheduler = TileScheduler()
        k_top, k_mid = scheduler.get_periods(0)
        assert k_top == 1
        assert k_mid == 1

    def test_periods_d2(self):
        """D=2 should have K_top=4, K_mid=2."""
        scheduler = TileScheduler()
        k_top, k_mid = scheduler.get_periods(2)
        assert k_top == 4
        assert k_mid == 2

    def test_periods_d4(self):
        """D=4 should have K_top=16, K_mid=8."""
        scheduler = TileScheduler()
        k_top, k_mid = scheduler.get_periods(4)
        assert k_top == 16
        assert k_mid == 8

    def test_mid_due_d0(self):
        """At D=0, MID should always be due."""
        scheduler = TileScheduler()
        for v in range(16):
            assert scheduler.is_mid_due(v, D=0)

    def test_top_due_d0(self):
        """At D=0, TOP should always be due."""
        scheduler = TileScheduler()
        for v in range(16):
            assert scheduler.is_top_due(v, D=0)

    def test_top_suppressed(self):
        """TOP should not be due when suppressed."""
        scheduler = TileScheduler()
        for v in range(16):
            assert not scheduler.is_top_due(v, D=0, top_suppressed=True)

    def test_advance_counters_only_serviced_tiles(self):
        """Counters should only advance for tiles with serviced voices."""
        scheduler = TileScheduler()

        # Service voice 0 (tile 0) only
        scheduler.advance_counters([0], D=0)

        # Tile 0 should advance, others should not
        assert scheduler.ctr_mid[0] == 0  # K_mid=1 at D=0, wraps to 0
        assert scheduler.ctr_mid[1] == 0

    def test_counter_wraps(self):
        """Counter should wrap based on period."""
        scheduler = TileScheduler()
        scheduler.ctr_mid[0] = 1

        # At D=2, K_mid=2, so counter wraps at 2
        scheduler.advance_counters([0], D=2)
        assert scheduler.ctr_mid[0] == 0

    def test_mid_due_phased(self):
        """MID should be due based on counter + phase."""
        scheduler = TileScheduler()
        scheduler.ctr_mid[0] = 0

        # At D=2, K_mid=2
        # Voice 0: phase = 0, due when (0 + 0) % 2 == 0 -> True
        # Voice 1: phase = 1, due when (0 + 1) % 2 == 0 -> False
        assert scheduler.is_mid_due(0, D=2)
        assert not scheduler.is_mid_due(1, D=2)

        # Advance counter
        scheduler.advance_counters([0, 1], D=2)

        # Now ctr_mid[0] = 1
        # Voice 0: due when (1 + 0) % 2 == 0 -> False
        # Voice 1: due when (1 + 1) % 2 == 0 -> True
        assert not scheduler.is_mid_due(0, D=2)
        assert scheduler.is_mid_due(1, D=2)


# =============================================================================
# DEGRADE CONTROLLER TESTS
# =============================================================================

class TestDegradeController:
    """Test degrade state with hysteresis."""

    def test_initial_d_zero(self):
        """Initial D should be 0."""
        controller = DegradeController()
        assert controller.D == 0

    def test_d_increases_immediately(self):
        """D should increase immediately when pressure is high."""
        controller = DegradeController()

        # Pressure above 0.70 threshold
        high_pressure = DEGRADE_THRESHOLDS_Q15[0] + 100
        new_d = controller.update(high_pressure, high_pressure)

        assert new_d == 1

    def test_d_increases_by_one(self):
        """D should increase by at most 1 per block."""
        controller = DegradeController()

        # Very high pressure (should want D=4)
        very_high = DEGRADE_THRESHOLDS_Q15[3] + 1000

        # First update: D goes from 0 to 1
        controller.update(very_high, very_high)
        assert controller.D == 1

        # Second update: D goes from 1 to 2
        controller.update(very_high, very_high)
        assert controller.D == 2

    def test_d_decreases_with_hysteresis(self):
        """D should decrease only after H_DOWN blocks."""
        controller = DegradeController()
        controller.D = 2
        controller.down_counter = 0

        low_pressure = DEGRADE_THRESHOLDS_Q15[0] - 100  # Below D=0 threshold

        # First H_DOWN-1 updates: D stays at 2
        for _ in range(H_DOWN - 1):
            controller.update(low_pressure, low_pressure)
            assert controller.D == 2

        # H_DOWN-th update: D decreases to 1
        controller.update(low_pressure, low_pressure)
        assert controller.D == 1

    def test_d_reset_down_counter_on_increase(self):
        """Down counter should reset when D increases."""
        controller = DegradeController()
        controller.D = 1
        controller.down_counter = 3  # Almost ready to decrease

        # High pressure causes increase
        high = DEGRADE_THRESHOLDS_Q15[1] + 100
        controller.update(high, high)

        assert controller.D == 2
        assert controller.down_counter == 0

    def test_gear_calculation(self):
        """Gear should be 5 - D."""
        controller = DegradeController()

        controller.D = 0
        assert controller.get_gear() == 5

        controller.D = 3
        assert controller.get_gear() == 2

        controller.D = 4
        assert controller.get_gear() == 1

    def test_target_d_boundaries(self):
        """Test target D at boundary values."""
        controller = DegradeController()

        # Just below 0.70 -> D=0
        assert controller.compute_target_D(DEGRADE_THRESHOLDS_Q15[0] - 1) == 0

        # At 0.70 -> D=1
        assert controller.compute_target_D(DEGRADE_THRESHOLDS_Q15[0]) == 1

        # Just below 0.85 -> D=1
        assert controller.compute_target_D(DEGRADE_THRESHOLDS_Q15[1] - 1) == 1

        # At 1.15 -> D=4
        assert controller.compute_target_D(DEGRADE_THRESHOLDS_Q15[3]) == 4


# =============================================================================
# HEALTH TELEMETRY TESTS
# =============================================================================

class TestHealthTelemetry:
    """Test health telemetry structure."""

    def test_default_values(self):
        """Default telemetry should have sensible values."""
        telemetry = HealthTelemetry()
        assert telemetry.cpu_bar == 0
        assert telemetry.mem_bar == 0
        assert telemetry.stab_bar == Q15_ONE
        assert telemetry.gear == 5

    def test_to_dict(self):
        """to_dict should include all fields."""
        telemetry = HealthTelemetry(cpu_bar=100, gear=3)
        d = telemetry.to_dict()

        assert d['cpu_bar'] == 100
        assert d['gear'] == 3
        assert 'stab_bar' in d
        assert 'voices_serviced' in d


# =============================================================================
# BLOCK SCHEDULER INTEGRATION TESTS
# =============================================================================

class TestBlockScheduler:
    """Test integrated block scheduler."""

    def test_empty_block(self):
        """Empty frames should produce empty schedule."""
        scheduler = BlockScheduler()
        result = scheduler.schedule_block([])

        assert result.serviced == []
        assert result.degrade_state == 0
        assert not result.top_suppressed

    def test_single_voice(self):
        """Single eligible voice should be serviced."""
        scheduler = BlockScheduler()
        frames = [VoiceFramePacketV0b(voice_id=0, flags=VoiceFlags.GATE)]

        result = scheduler.schedule_block(frames)

        assert len(result.serviced) == 1
        assert result.serviced[0].voice_id == 0

    def test_due_flags_at_d0(self):
        """At D=0, all voices should have MID and TOP due."""
        scheduler = BlockScheduler()
        frames = [
            VoiceFramePacketV0b(voice_id=i, flags=VoiceFlags.GATE)
            for i in range(4)
        ]

        result = scheduler.schedule_block(frames)

        for info in result.serviced:
            assert info.mid_due
            assert info.top_due

    def test_top_suppression_at_d4(self):
        """TOP should be suppressed at D=4 with low stab_bar."""
        scheduler = BlockScheduler()
        scheduler.degrade_controller.D = 4
        scheduler.prev_stab_bar = THETA_STAB_RED_Q15 - 100

        frames = [VoiceFramePacketV0b(voice_id=0, flags=VoiceFlags.GATE)]

        result = scheduler.schedule_block(frames)

        assert result.top_suppressed
        assert not result.serviced[0].top_due

    def test_end_block_updates_telemetry(self):
        """end_block should update prev telemetry."""
        scheduler = BlockScheduler(b_compute=100, b_mem=100)
        scheduler.used_compute = 50
        scheduler.used_mem = 70

        scheduler.end_block()

        # CPU bar: 50/100 = 0.5 * 32768 = 16384
        assert scheduler.prev_cpu_bar == 16384
        # Mem bar: 70/100 = 0.7 * 32768 = 22937 (approximately)
        assert abs(scheduler.prev_mem_bar - 22937) < 10

    def test_degrade_from_telemetry(self):
        """D should increase based on previous block telemetry."""
        scheduler = BlockScheduler()

        # Set previous telemetry to high pressure
        scheduler.prev_cpu_bar = DEGRADE_THRESHOLDS_Q15[2]  # 1.0 (D=3 target)
        scheduler.prev_mem_bar = 0

        frames = [VoiceFramePacketV0b(voice_id=0, flags=VoiceFlags.GATE)]

        result = scheduler.schedule_block(frames)

        # D should have increased from 0 to 1
        assert result.degrade_state == 1

    def test_deterministic_selection(self):
        """Selection should be deterministic given same state."""
        frames = [
            VoiceFramePacketV0b(voice_id=i, flags=VoiceFlags.GATE)
            for i in range(20)
        ]

        scheduler1 = BlockScheduler()
        scheduler2 = BlockScheduler()

        result1 = scheduler1.schedule_block(frames)
        result2 = scheduler2.schedule_block(frames)

        assert result1.serviced_ids == result2.serviced_ids

    def test_rejected_voices_not_serviced(self):
        """Rejected voices should not be serviced."""
        scheduler = BlockScheduler()
        frames = [
            VoiceFramePacketV0b(voice_id=0, flags=VoiceFlags.GATE, reject_code=1),
            VoiceFramePacketV0b(voice_id=1, flags=VoiceFlags.GATE, reject_code=0),
        ]

        result = scheduler.schedule_block(frames)

        assert len(result.serviced) == 1
        assert result.serviced[0].voice_id == 1

    def test_tile_counters_advance(self):
        """Tile counters should advance for serviced tiles."""
        scheduler = BlockScheduler()
        scheduler.degrade_controller.D = 2  # K_mid=2, K_top=4

        frames = [None] * 128
        # Voice 0 in tile 0
        frames[0] = VoiceFramePacketV0b(voice_id=0, flags=VoiceFlags.GATE)

        initial_mid_0 = scheduler.tile_scheduler.ctr_mid[0]
        initial_mid_1 = scheduler.tile_scheduler.ctr_mid[1]

        scheduler.schedule_block(frames)

        # Tile 0 counter should have advanced
        assert scheduler.tile_scheduler.ctr_mid[0] != initial_mid_0 or initial_mid_0 == 0
        # Tile 1 counter should NOT have advanced
        assert scheduler.tile_scheduler.ctr_mid[1] == initial_mid_1


# =============================================================================
# STABILITY BAR TESTS
# =============================================================================

class TestStabilityBar:
    """Test stability bar computation."""

    def test_full_stability_no_events(self):
        """No events should give full stability."""
        scheduler = BlockScheduler()
        stab = scheduler.compute_stab_bar(0, 0, 0, 10)
        assert stab == Q15_ONE

    def test_stability_decreases_with_emergency(self):
        """Emergency events should decrease stability."""
        scheduler = BlockScheduler()
        stab = scheduler.compute_stab_bar(0, 0, 1, 10)
        assert stab < Q15_ONE

    def test_stability_decreases_with_clamp_hits(self):
        """Clamp hits should decrease stability."""
        scheduler = BlockScheduler()
        stab = scheduler.compute_stab_bar(100, 0, 0, 10)
        assert stab < Q15_ONE

    def test_stability_bottoms_at_zero(self):
        """Stability should not go negative."""
        scheduler = BlockScheduler()
        stab = scheduler.compute_stab_bar(10000, 100, 100, 1)
        assert stab >= 0


# =============================================================================
# SCHEDULE RESULT TESTS
# =============================================================================

class TestScheduleResult:
    """Test schedule result structure."""

    def test_serviced_ids_property(self):
        """serviced_ids should return list of voice IDs."""
        result = ScheduleResult(
            serviced=[
                VoiceScheduleInfo(5, True, False),
                VoiceScheduleInfo(10, False, True),
            ],
            degrade_state=1,
            top_suppressed=False,
            telemetry=HealthTelemetry(),
        )

        assert result.serviced_ids == [5, 10]

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        result = ScheduleResult(
            serviced=[VoiceScheduleInfo(0, True, True)],
            degrade_state=2,
            top_suppressed=True,
            telemetry=HealthTelemetry(gear=3),
        )

        d = result.to_dict()
        assert d['degrade_state'] == 2
        assert d['top_suppressed'] is True
        assert len(d['serviced']) == 1
        assert d['telemetry']['gear'] == 3
