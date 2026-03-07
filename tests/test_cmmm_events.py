"""
Test CMMM event roundtrip: VoiceFrame -> Executor -> Latch.

Verifies WSE-STD-070 event bindings:
- 0x30 (NODEMASK_SET): Node mask changes (immediate)
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
        """NODEMASK_SET event should update executor latch immediately."""
        state = VMState()
        frame = MockVoiceFrame(
            cmmm_node_mask=0x0000FFFF,  # 16 nodes
            cmmm_mask_ramp_blocks=10,
        )

        _process_cmmm_events(state, frame)

        # Latch should have new mask applied immediately
        latch = state.executor_latch
        assert latch.node_mask_u32 == 0x0000FFFF
        assert latch.get_effective_k() == 16

    def test_nodemask_event_immediate_regardless_of_ramp(self):
        """Mask change should be immediate even with ramp_blocks specified."""
        state = VMState()
        state.executor_latch.node_mask_u32 = 0xFFFFFFFF  # 32 nodes

        frame = MockVoiceFrame(
            cmmm_node_mask=0x000000FF,  # 8 nodes
            cmmm_mask_ramp_blocks=5,
        )

        _process_cmmm_events(state, frame)

        # Should be immediately at target
        assert state.executor_latch.node_mask_u32 == 0x000000FF
        assert state.executor_latch.get_effective_k() == 8

    def test_no_nodemask_event_preserves_mask(self):
        """Without event, mask should be unchanged."""
        state = VMState()
        state.executor_latch.node_mask_u32 = 0x0000FFFF

        # Process frame without event
        frame = MockVoiceFrame()  # No events
        _process_cmmm_events(state, frame)

        # Mask should be unchanged
        assert state.executor_latch.node_mask_u32 == 0x0000FFFF

    def test_sequential_nodemask_events(self):
        """Sequential mask events should each apply immediately."""
        state = VMState()
        state.executor_latch.node_mask_u32 = 0xFFFFFFFF

        # First mask change
        frame1 = MockVoiceFrame(cmmm_node_mask=0x0000FFFF)
        _process_cmmm_events(state, frame1)
        assert state.executor_latch.node_mask_u32 == 0x0000FFFF

        # Second mask change
        frame2 = MockVoiceFrame(cmmm_node_mask=0x000000FF)
        _process_cmmm_events(state, frame2)
        assert state.executor_latch.node_mask_u32 == 0x000000FF

        # Third mask change
        frame3 = MockVoiceFrame(cmmm_node_mask=0x00000001)
        _process_cmmm_events(state, frame3)
        assert state.executor_latch.node_mask_u32 == 0x00000001


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
        # Node mask — immediate
        assert latch.node_mask_u32 == 0x00FF00FF
        assert latch.get_effective_k() == 16
        # Tail mode
        assert latch.tail_mode == 3
        assert latch.traction_q15 == 32767

    def test_new_mask_event_replaces_previous(self):
        """New mask event should simply overwrite previous mask."""
        state = VMState()
        state.executor_latch.node_mask_u32 = 0xFFFFFFFF

        # First mask change
        frame1 = MockVoiceFrame(cmmm_node_mask=0x0000FFFF)
        _process_cmmm_events(state, frame1)
        assert state.executor_latch.node_mask_u32 == 0x0000FFFF

        # A few frames without events
        for _ in range(3):
            _process_cmmm_events(state, MockVoiceFrame())
        assert state.executor_latch.node_mask_u32 == 0x0000FFFF

        # New mask event overwrites
        frame2 = MockVoiceFrame(cmmm_node_mask=0x000000FF)
        _process_cmmm_events(state, frame2)
        assert state.executor_latch.node_mask_u32 == 0x000000FF


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
