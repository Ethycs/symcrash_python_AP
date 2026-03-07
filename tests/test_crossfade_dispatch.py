"""
Test mask-aware dispatch for CMMM node mask transitions.

Verifies WSE-STD-070: immediate mask changes with continuous coefficient state.
Crossfade double-dispatch was removed — coefficient continuity and tail operators
guarantee smooth transitions without it.
"""

import numpy as np
import pytest

from symcrash_python_ap.vm.state import VMState, ExecutorLatch


class TestExecutorLatch:
    """Test ExecutorLatch node mask behavior."""

    def test_default_mask_is_full(self):
        """Default mask should be all-nodes-active (0xFFFFFFFF)."""
        latch = ExecutorLatch()
        assert latch.node_mask_u32 == 0xFFFFFFFF

    def test_apply_nodemask_is_immediate(self):
        """Mask change should be applied immediately."""
        latch = ExecutorLatch()
        latch.node_mask_u32 = 0xFFFFFFFF

        latch.apply_nodemask_event(0x0000FFFF, ramp_blocks=10)

        # Mask should be updated immediately (ramp_blocks ignored)
        assert latch.node_mask_u32 == 0x0000FFFF

    def test_apply_nodemask_zero_ramp(self):
        """With ramp_blocks=0, change should be immediate."""
        latch = ExecutorLatch()
        latch.node_mask_u32 = 0xFFFFFFFF

        latch.apply_nodemask_event(0x0000FFFF, ramp_blocks=0)

        assert latch.node_mask_u32 == 0x0000FFFF

    def test_apply_nodemask_default_ramp(self):
        """With default ramp_blocks, change should still be immediate."""
        latch = ExecutorLatch()
        latch.node_mask_u32 = 0xFFFFFFFF

        latch.apply_nodemask_event(0x0000FFFF)

        assert latch.node_mask_u32 == 0x0000FFFF

    def test_effective_k_after_mask_change(self):
        """K should reflect new mask immediately."""
        latch = ExecutorLatch()
        assert latch.get_effective_k() == 32

        latch.apply_nodemask_event(0x0000FFFF)
        assert latch.get_effective_k() == 16

        latch.apply_nodemask_event(0x000000FF)
        assert latch.get_effective_k() == 8


class TestMaskAwareDispatch:
    """Test mask-aware operator dispatch (no crossfade)."""

    def test_dispatch_uses_current_mask(self):
        """Dispatch should use the current node_mask_u32 directly."""
        state = VMState()
        state.coeff_state = np.zeros(512, dtype=np.int16)
        state.coeff_state[0] = 1000

        # Set a specific mask
        state.executor_latch.node_mask_u32 = 0x0000FFFF

        # The mask should be directly accessible
        assert state.executor_latch.node_mask_u32 == 0x0000FFFF
        assert state.executor_latch.get_effective_k() == 16

    def test_sequential_mask_changes(self):
        """Multiple mask changes should each take effect immediately."""
        state = VMState()

        state.executor_latch.apply_nodemask_event(0x0000FFFF)
        assert state.executor_latch.node_mask_u32 == 0x0000FFFF

        state.executor_latch.apply_nodemask_event(0x000000FF)
        assert state.executor_latch.node_mask_u32 == 0x000000FF

        state.executor_latch.apply_nodemask_event(0x00000001)
        assert state.executor_latch.node_mask_u32 == 0x00000001


class TestTailModeEvent:
    """Test tail mode event handling."""

    def test_apply_tailmode_event(self):
        """Tail mode event should update latch state."""
        latch = ExecutorLatch()

        # Apply tail mode change
        latch.apply_tailmode_event(tail_mode=2, traction_q15=16384)

        assert latch.tail_mode == 2
        assert latch.traction_q15 == 16384

    def test_default_tail_mode_is_full(self):
        """Default tail mode should be FULL (0)."""
        latch = ExecutorLatch()

        assert latch.tail_mode == 0
        assert latch.traction_q15 == 32767


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
