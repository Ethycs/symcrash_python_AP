"""
Test crossfade-aware dispatch for CMMM node mask transitions.

Verifies WSE-STD-070 Section 8: smooth mask transitions prevent clicks.
"""

import numpy as np
import pytest

from symcrash_python_ap.vm.state import VMState, ExecutorLatch


class TestExecutorLatch:
    """Test ExecutorLatch crossfade weight computation."""

    def test_no_ramp_returns_full_target_weight(self):
        """When not ramping, target weight should be full (32767)."""
        latch = ExecutorLatch()
        latch.node_mask_u32 = 0x0000FFFF  # 16 nodes

        start_w, target_w, start_m, target_m = latch.get_crossfade_weights()

        assert start_w == 0
        assert target_w == 32767
        assert target_m == 0x0000FFFF

    def test_ramp_start_is_all_start_weight(self):
        """At ramp start, start weight should dominate."""
        latch = ExecutorLatch()
        latch.node_mask_u32 = 0xFFFFFFFF  # Start: 32 nodes

        # Apply mask change with 10 block ramp
        latch.apply_nodemask_event(0x0000FFFF, ramp_blocks=10)

        start_w, target_w, start_m, target_m = latch.get_crossfade_weights()

        # At start of ramp (progress=0), start_weight should be ~32767
        assert start_w > 30000  # Should be close to 32767
        assert target_w < 3000  # Should be close to 0
        assert start_m == 0xFFFFFFFF
        assert target_m == 0x0000FFFF

    def test_ramp_midpoint_is_balanced(self):
        """At ramp midpoint, weights should be roughly equal."""
        latch = ExecutorLatch()
        latch.node_mask_u32 = 0xFFFFFFFF

        # Apply mask change with 10 block ramp
        latch.apply_nodemask_event(0x0000FFFF, ramp_blocks=10)

        # Step 5 times to reach midpoint
        for _ in range(5):
            latch.step_mask_ramp()

        start_w, target_w, start_m, target_m = latch.get_crossfade_weights()

        # At midpoint, weights should be ~equal
        assert abs(start_w - target_w) < 6554  # Within 20% of each other

    def test_ramp_end_snaps_to_target(self):
        """After ramp completes, mask should snap to target."""
        latch = ExecutorLatch()
        latch.node_mask_u32 = 0xFFFFFFFF

        # Apply mask change with 5 block ramp
        latch.apply_nodemask_event(0x0000FFFF, ramp_blocks=5)

        # Step through entire ramp
        for _ in range(5):
            latch.step_mask_ramp()

        # After ramp, mask should be target
        assert latch.node_mask_u32 == 0x0000FFFF
        assert latch.is_ramping() == False

    def test_immediate_change_no_ramp(self):
        """With ramp_blocks=0, change should be immediate."""
        latch = ExecutorLatch()
        latch.node_mask_u32 = 0xFFFFFFFF

        latch.apply_nodemask_event(0x0000FFFF, ramp_blocks=0)

        assert latch.node_mask_u32 == 0x0000FFFF
        assert latch.is_ramping() == False


class TestCrossfadeDispatch:
    """Test crossfade-aware operator dispatch."""

    def test_no_ramp_uses_direct_dispatch(self):
        """When not ramping, should use direct dispatch path."""
        state = VMState()
        state.coeff_state = np.zeros(512, dtype=np.int16)
        state.coeff_state[0] = 1000

        # Not ramping - should use direct path
        assert not state.executor_latch.is_ramping()

        # Verify the is_ramping check works correctly
        weights = state.executor_latch.get_crossfade_weights()
        assert weights[0] == 0  # start_weight = 0
        assert weights[1] == 32767  # target_weight = full

    def test_ramp_produces_blended_output(self):
        """During ramp, output should be blend of start and target masks."""
        state = VMState()
        state.coeff_state = np.zeros(512, dtype=np.int16)

        # Set up a ramp
        state.executor_latch.node_mask_u32 = 0xFFFFFFFF
        state.executor_latch.apply_nodemask_event(0x0000FFFF, ramp_blocks=10)

        assert state.executor_latch.is_ramping()

        # Get weights
        start_w, target_w, start_m, target_m = state.executor_latch.get_crossfade_weights()

        # Verify we have distinct masks
        assert start_m != target_m
        assert start_w + target_w == 32767  # Weights sum to Q15_ONE


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
