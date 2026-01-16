"""
Test CMMM event roundtrip: VoiceFrame → Executor → Latch.

Verifies WSE-STD-070 event bindings:
- 0x30 (NODEMASK_SET): Node mask changes with ramp
- 0x31 (TAILMODE_SET): Tail mode changes with traction
"""

import numpy as np
import pytest
from dataclasses import dataclass
from typing import Optional

from symcrash_python_ap.vm.state import VMState, ExecutorLatch
from symcrash_python_ap.vm.executor import _process_cmmm_events


@dataclass
class MockVoiceFrame:
    """Mock VoiceFrame for testing CMMM event processing."""
    # CMMM node mask event (0x30)
    cmmm_node_mask: Optional[int] = None
    cmmm_mask_ramp_blocks: int = 5

    # CMMM tail mode event (0x31)
    cmmm_tail_mode: Optional[int] = None
    cmmm_traction_q15: int = 32767


class TestNodeMaskEventRoundtrip:
    """Test 0x30 NODEMASK_SET event roundtrip."""

    def test_nodemask_event_updates_latch(self):
        """NODEMASK_SET event should update executor latch."""
        state = VMState()
        frame = MockVoiceFrame(
            cmmm_node_mask=0x0000FFFF,  # 16 nodes
            cmmm_mask_ramp_blocks=10,
        )

        _process_cmmm_events(state, frame)

        # Latch should be ramping to new mask
        latch = state.executor_latch
        assert latch.mask_target_u32 == 0x0000FFFF
        assert latch.mask_ramp_remaining == 10
        assert latch.is_ramping()

    def test_nodemask_event_preserves_start_mask(self):
        """Start mask should be preserved for crossfade."""
        state = VMState()
        state.executor_latch.node_mask_u32 = 0xFFFFFFFF  # 32 nodes

        frame = MockVoiceFrame(
            cmmm_node_mask=0x000000FF,  # 8 nodes
            cmmm_mask_ramp_blocks=5,
        )

        _process_cmmm_events(state, frame)

        # Check crossfade weights show correct start mask
        start_w, target_w, start_m, target_m = state.executor_latch.get_crossfade_weights()
        assert start_m == 0xFFFFFFFF  # Original mask
        assert target_m == 0x000000FF  # New target

    def test_no_nodemask_event_advances_ramp(self):
        """Without event, ramp should still advance."""
        state = VMState()
        state.executor_latch.apply_nodemask_event(0x0000FFFF, ramp_blocks=5)
        initial_remaining = state.executor_latch.mask_ramp_remaining

        # Process frame without event
        frame = MockVoiceFrame()  # No events
        _process_cmmm_events(state, frame)

        # Ramp should have advanced by 1
        assert state.executor_latch.mask_ramp_remaining == initial_remaining - 1

    def test_nodemask_ramp_completes(self):
        """Ramp should complete after specified blocks."""
        state = VMState()
        state.executor_latch.node_mask_u32 = 0xFFFFFFFF

        # Start ramp
        frame = MockVoiceFrame(cmmm_node_mask=0x0000FFFF, cmmm_mask_ramp_blocks=3)
        _process_cmmm_events(state, frame)

        # Process 3 more frames (first frame already stepped once)
        for _ in range(2):
            _process_cmmm_events(state, MockVoiceFrame())

        # Still ramping after 2 more steps
        assert state.executor_latch.mask_ramp_remaining == 1

        # Final step
        _process_cmmm_events(state, MockVoiceFrame())

        # Now complete
        assert not state.executor_latch.is_ramping()
        assert state.executor_latch.node_mask_u32 == 0x0000FFFF


class TestTailModeEventRoundtrip:
    """Test 0x31 TAILMODE_SET event roundtrip."""

    def test_tailmode_event_updates_latch(self):
        """TAILMODE_SET event should update executor latch."""
        state = VMState()
        frame = MockVoiceFrame(
            cmmm_tail_mode=2,  # MIN tail mode
            cmmm_traction_q15=16384,
        )

        _process_cmmm_events(state, frame)

        latch = state.executor_latch
        assert latch.tail_mode == 2
        assert latch.traction_q15 == 16384

    def test_tailmode_event_only_when_present(self):
        """Tail mode should not change without event."""
        state = VMState()
        state.executor_latch.tail_mode = 1
        state.executor_latch.traction_q15 = 8000

        # Frame without tail mode event
        frame = MockVoiceFrame()
        _process_cmmm_events(state, frame)

        # Should be unchanged
        assert state.executor_latch.tail_mode == 1
        assert state.executor_latch.traction_q15 == 8000


class TestCombinedEvents:
    """Test multiple CMMM events in same frame."""

    def test_both_events_in_one_frame(self):
        """Both node mask and tail mode can be set in one frame."""
        state = VMState()
        frame = MockVoiceFrame(
            cmmm_node_mask=0x00FF00FF,
            cmmm_mask_ramp_blocks=8,
            cmmm_tail_mode=3,  # EMERGENCY
            cmmm_traction_q15=32767,
        )

        _process_cmmm_events(state, frame)

        latch = state.executor_latch
        # Node mask
        assert latch.mask_target_u32 == 0x00FF00FF
        assert latch.mask_ramp_remaining == 8
        # Tail mode
        assert latch.tail_mode == 3
        assert latch.traction_q15 == 32767

    def test_new_mask_event_interrupts_ramp(self):
        """New mask event during ramp should start new ramp."""
        state = VMState()
        state.executor_latch.node_mask_u32 = 0xFFFFFFFF

        # Start first ramp
        frame1 = MockVoiceFrame(cmmm_node_mask=0x0000FFFF, cmmm_mask_ramp_blocks=10)
        _process_cmmm_events(state, frame1)

        # Advance a few blocks
        for _ in range(3):
            _process_cmmm_events(state, MockVoiceFrame())

        # Interrupt with new mask event
        frame2 = MockVoiceFrame(cmmm_node_mask=0x000000FF, cmmm_mask_ramp_blocks=5)
        _process_cmmm_events(state, frame2)

        # Should be ramping to new target
        assert state.executor_latch.mask_target_u32 == 0x000000FF
        assert state.executor_latch.mask_ramp_remaining == 5


class TestKDegradationStaircase:
    """Test K (active node count) computation from masks."""

    def test_full_mask_k32(self):
        """Full mask (32 bits) should give K=32."""
        latch = ExecutorLatch()
        latch.node_mask_u32 = 0xFFFFFFFF
        assert latch.get_effective_k() == 32

    def test_half_mask_k16(self):
        """16-bit mask should give K=16."""
        latch = ExecutorLatch()
        latch.node_mask_u32 = 0x0000FFFF
        assert latch.get_effective_k() == 16

    def test_sparse_mask(self):
        """Sparse mask should count only set bits."""
        latch = ExecutorLatch()
        latch.node_mask_u32 = 0b10101010101010101010101010101010  # 16 bits set
        assert latch.get_effective_k() == 16

    def test_minimal_mask_k1(self):
        """Single bit mask should give K=1."""
        latch = ExecutorLatch()
        latch.node_mask_u32 = 0x00000001
        assert latch.get_effective_k() == 1

    def test_empty_mask_k0(self):
        """Empty mask should give K=0."""
        latch = ExecutorLatch()
        latch.node_mask_u32 = 0x00000000
        assert latch.get_effective_k() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
