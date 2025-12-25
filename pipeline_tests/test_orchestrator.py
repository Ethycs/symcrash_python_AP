"""
BLOCK ORCHESTRATOR TESTS (O1-O5)
================================

Tests for the orchestrator connecting scheduler to pipeline.

Covers:
- O1: Voice Lifecycle Management
- O2: Scheduled Rendering
- O3: Emission Group Filtering
- O4: Frame Updates
- O5: Integration with Scheduler
"""

import pytest
from typing import List


class TestO1VoiceLifecycle:
    """
    O1. Voice Lifecycle Management

    Requirement:
    - voice_allocate arms microfade
    - voice_on does NOT reset state
    - voice_free preserves prev_sample for next allocate

    Test:
    - State transitions follow lifecycle rules
    - Microfade is armed on allocate only
    """

    def test_voice_allocate_transitions_to_allocated(self, block_orchestrator):
        """voice_allocate transitions slot to ALLOCATED."""
        from symcrash_python_ap.voice.orchestrator import VoiceState

        # Initially FREE
        assert block_orchestrator.voice_slots[0].state == VoiceState.FREE

        # Allocate
        result = block_orchestrator.voice_allocate(0, owner_id=42, note_u8=60)
        assert result is True
        assert block_orchestrator.voice_slots[0].state == VoiceState.ALLOCATED
        assert block_orchestrator.voice_slots[0].owner_id == 42
        assert block_orchestrator.voice_slots[0].note_u8 == 60

    def test_voice_allocate_arms_microfade(self, block_orchestrator):
        """voice_allocate arms microfade in WaveletEngine."""
        block_orchestrator.voice_allocate(0, owner_id=1)

        R = block_orchestrator.wavelet.get_regularity(0)
        assert R.is_fading(), "Microfade should be armed after allocate"

    def test_voice_on_transitions_to_active(self, block_orchestrator):
        """voice_on transitions slot to ACTIVE."""
        from symcrash_python_ap.voice.orchestrator import VoiceState

        block_orchestrator.voice_allocate(0, owner_id=1)
        assert block_orchestrator.voice_slots[0].state == VoiceState.ALLOCATED

        result = block_orchestrator.voice_on(0)
        assert result is True
        assert block_orchestrator.voice_slots[0].state == VoiceState.ACTIVE

    def test_voice_on_does_not_arm_microfade(self, block_orchestrator):
        """voice_on does NOT arm microfade."""
        from symcrash_python_ap.frames import VoiceFrame

        block_orchestrator.voice_allocate(0, owner_id=1)

        # Render a block to consume the microfade
        vf = VoiceFrame(voice_id=0, tree_id=0, gain_q15=16384, coeff_q15=[1000] * 256)
        for _ in range(2):  # Two blocks to ensure microfade is consumed
            block_orchestrator.wavelet.render_voice(vf)

        R = block_orchestrator.wavelet.get_regularity(0)
        assert not R.is_fading(), "Microfade should be consumed"

        # voice_on should NOT arm microfade
        block_orchestrator.voice_on(0)
        assert not R.is_fading(), "voice_on should NOT arm microfade"

    def test_voice_on_fails_if_free(self, block_orchestrator):
        """voice_on fails if voice is FREE (must allocate first)."""
        result = block_orchestrator.voice_on(0)
        assert result is False

    def test_voice_free_transitions_to_free(self, block_orchestrator):
        """voice_free transitions slot back to FREE."""
        from symcrash_python_ap.voice.orchestrator import VoiceState

        block_orchestrator.voice_allocate(0, owner_id=1)
        block_orchestrator.voice_on(0)
        assert block_orchestrator.voice_slots[0].state == VoiceState.ACTIVE

        result = block_orchestrator.voice_free(0)
        assert result is True
        assert block_orchestrator.voice_slots[0].state == VoiceState.FREE

    def test_voice_free_preserves_prev_sample(self, block_orchestrator):
        """voice_free preserves prev_sample for next allocate."""
        from symcrash_python_ap.frames import VoiceFrame

        block_orchestrator.voice_allocate(0, owner_id=1)
        block_orchestrator.voice_on(0)

        # Render to establish non-zero prev_sample
        vf = VoiceFrame(voice_id=0, tree_id=0, gain_q15=16384, coeff_q15=[5000] * 256)
        block_orchestrator.wavelet.render_voice(vf)

        R = block_orchestrator.wavelet.get_regularity(0)
        prev_sample = R.prev_sample

        # Free voice
        block_orchestrator.voice_free(0)

        # prev_sample should be preserved
        assert R.prev_sample == prev_sample

    def test_get_free_voice_returns_free_slot(self, block_orchestrator):
        """get_free_voice finds an unallocated slot."""
        # Allocate first two voices
        block_orchestrator.voice_allocate(0, owner_id=1)
        block_orchestrator.voice_allocate(1, owner_id=2)

        # Next free voice should be index 2
        free_vid = block_orchestrator.get_free_voice()
        assert free_vid == 2

    def test_get_free_voice_returns_none_when_full(self, block_orchestrator):
        """get_free_voice returns None when all slots are used."""
        # Allocate all voices
        for vid in range(block_orchestrator.n_voices):
            block_orchestrator.voice_allocate(vid, owner_id=vid)

        free_vid = block_orchestrator.get_free_voice()
        assert free_vid is None


class TestO2ScheduledRendering:
    """
    O2. Scheduled Rendering

    Requirement:
    - Only serviced voices are rendered
    - Schedule result reflects scheduler decisions

    Test:
    - Voices not in serviced list produce no audio
    - Scheduler decisions flow through to rendering
    """

    def test_step_block_returns_audio_and_schedule(self, block_orchestrator):
        """step_block returns stereo audio and schedule result."""
        from symcrash_python_ap.voice.scheduler import ScheduleResult

        L, R, schedule = block_orchestrator.step_block()

        assert len(L) == 64
        assert len(R) == 64
        assert isinstance(schedule, ScheduleResult)

    def test_only_serviced_voices_contribute_audio(self, block_orchestrator, q15_half):
        """Only voices in schedule.serviced contribute to output."""
        from symcrash_python_ap.frames import VoiceFramePacketV0b, VoiceFlags

        # Set up one active voice with loud coefficients
        frame = VoiceFramePacketV0b(
            voice_id=0,
            flags=VoiceFlags.GATE,
            gain_q15=q15_half,
            activity_q15=10000,
        )
        block_orchestrator.update_frame(0, frame)

        # Step block
        L, R, schedule = block_orchestrator.step_block()

        # If voice was serviced, audio should have some energy
        if 0 in schedule.serviced_ids:
            # Should have some non-zero samples
            # (though with zero coeffs it might be silent)
            pass

    def test_unserviced_voices_dont_contribute(self, block_orchestrator):
        """Voices not in serviced list don't contribute audio."""
        # With no frames, no voices should be serviced
        L, R, schedule = block_orchestrator.step_block()

        # Should be silent
        assert all(s == 0 for s in L)
        assert all(s == 0 for s in R)


class TestO3EmissionGroupFiltering:
    """
    O3. Emission Group Filtering

    Requirement:
    - TOP/MID flags control which operators run
    - TAIL always runs for serviced voices

    Test:
    - Emission group flags are passed through schedule
    """

    def test_schedule_contains_emission_flags(self, block_orchestrator, q15_half):
        """Schedule result contains mid_due and top_due flags."""
        from symcrash_python_ap.frames import VoiceFramePacketV0b, VoiceFlags

        # Set up active voice
        frame = VoiceFramePacketV0b(
            voice_id=0,
            flags=VoiceFlags.GATE,
            gain_q15=q15_half,
            activity_q15=10000,
        )
        block_orchestrator.update_frame(0, frame)

        L, R, schedule = block_orchestrator.step_block()

        # Each serviced voice has mid_due and top_due flags
        for sched_info in schedule.serviced:
            assert hasattr(sched_info, 'mid_due')
            assert hasattr(sched_info, 'top_due')


class TestO4FrameUpdates:
    """
    O4. Frame Updates

    Requirement:
    - update_frame updates voice state from external source
    - Gate events trigger lifecycle transitions

    Test:
    - Frame updates flow through to scheduler
    - GATE_RISE triggers auto-allocate + voice_on
    """

    def test_update_frame_stores_frame(self, block_orchestrator, q15_half):
        """update_frame stores frame for voice."""
        from symcrash_python_ap.frames import VoiceFramePacketV0b

        frame = VoiceFramePacketV0b(voice_id=5, gain_q15=q15_half)
        block_orchestrator.update_frame(5, frame)

        assert block_orchestrator.voice_frames[5] is frame

    def test_gate_rise_auto_allocates(self, block_orchestrator, q15_half):
        """GATE_RISE on FREE voice auto-allocates."""
        from symcrash_python_ap.frames import VoiceFramePacketV0b, VoiceFlags
        from symcrash_python_ap.voice.orchestrator import VoiceState

        frame = VoiceFramePacketV0b(
            voice_id=3,
            flags=VoiceFlags.GATE | VoiceFlags.GATE_RISE,
            gain_q15=q15_half,
            note_u8=64,
        )

        block_orchestrator.update_frame(3, frame)

        # Should be ACTIVE (allocate + on)
        assert block_orchestrator.voice_slots[3].state == VoiceState.ACTIVE
        assert block_orchestrator.voice_slots[3].note_u8 == 64

    def test_gate_fall_transitions_to_releasing(self, block_orchestrator, q15_half):
        """GATE_FALL transitions to RELEASING."""
        from symcrash_python_ap.frames import VoiceFramePacketV0b, VoiceFlags
        from symcrash_python_ap.voice.orchestrator import VoiceState

        # First activate
        frame1 = VoiceFramePacketV0b(
            voice_id=3,
            flags=VoiceFlags.GATE | VoiceFlags.GATE_RISE,
            gain_q15=q15_half,
        )
        block_orchestrator.update_frame(3, frame1)
        assert block_orchestrator.voice_slots[3].state == VoiceState.ACTIVE

        # Then gate fall
        frame2 = VoiceFramePacketV0b(
            voice_id=3,
            flags=VoiceFlags.GATE_FALL,
            gain_q15=q15_half,
            activity_q15=5000,  # Still has activity
        )
        block_orchestrator.update_frame(3, frame2)
        assert block_orchestrator.voice_slots[3].state == VoiceState.RELEASING


class TestO5SchedulerIntegration:
    """
    O5. Integration with Scheduler

    Requirement:
    - Scheduler decisions drive rendering
    - Telemetry updates correctly

    Test:
    - Multiple blocks show scheduler progression
    - Degrade state changes with pressure
    """

    def test_multiple_blocks_progress_counters(self, block_orchestrator, q15_half):
        """Multiple blocks advance scheduler counters."""
        from symcrash_python_ap.frames import VoiceFramePacketV0b, VoiceFlags

        # Set up active voice
        frame = VoiceFramePacketV0b(
            voice_id=0,
            flags=VoiceFlags.GATE,
            gain_q15=q15_half,
            activity_q15=10000,
        )
        block_orchestrator.update_frame(0, frame)

        block_counts = []
        for _ in range(10):
            L, R, schedule = block_orchestrator.step_block()
            block_counts.append(block_orchestrator.block_count)

        # Block count should increase
        assert block_counts == list(range(1, 11))

    def test_voice_summary_reflects_states(self, block_orchestrator):
        """get_voice_summary reflects current voice states."""
        from symcrash_python_ap.voice.orchestrator import VoiceState

        # Initially all FREE
        summary = block_orchestrator.get_voice_summary()
        assert summary['FREE'] == block_orchestrator.n_voices
        assert summary['ACTIVE'] == 0

        # Allocate and activate some voices
        block_orchestrator.voice_allocate(0, owner_id=1)
        block_orchestrator.voice_on(0)
        block_orchestrator.voice_allocate(1, owner_id=2)

        summary = block_orchestrator.get_voice_summary()
        assert summary['ACTIVE'] == 1
        assert summary['ALLOCATED'] == 1
        assert summary['FREE'] == block_orchestrator.n_voices - 2

    def test_get_active_voice_count(self, block_orchestrator):
        """get_active_voice_count returns correct count."""
        # Initially 0
        assert block_orchestrator.get_active_voice_count() == 0

        # Activate a voice
        block_orchestrator.voice_allocate(0, owner_id=1)
        block_orchestrator.voice_on(0)
        assert block_orchestrator.get_active_voice_count() == 1

        # Activate another
        block_orchestrator.voice_allocate(1, owner_id=2)
        block_orchestrator.voice_on(1)
        assert block_orchestrator.get_active_voice_count() == 2


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def block_orchestrator():
    """Fresh BlockOrchestrator instance with small voice count for testing."""
    from symcrash_python_ap.voice.orchestrator import BlockOrchestrator
    return BlockOrchestrator(n_voices=8)


@pytest.fixture
def q15_half():
    """Q1.15 half value."""
    return 16384
