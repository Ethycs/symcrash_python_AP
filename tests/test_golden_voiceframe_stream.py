"""
Golden VoiceFrame Stream Test - Deterministic Replay Verification.

Verifies WSE-STD-070 conformance: same inputs → bit-exact outputs.

This test creates a sequence of VoiceFrames with CMMM events and verifies
that replay produces identical output.
"""

import numpy as np
import pytest
from typing import Optional

from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.frames import VoiceFrame


def create_test_voiceframe(
    note: int = 69,
    gain_q15: int = 16384,
    coeff_osc_mode: int = 1,  # Mode B for consistent behavior
    cmmm_node_mask: Optional[int] = None,
    cmmm_mask_ramp_blocks: int = 5,
    cmmm_tail_mode: Optional[int] = None,
    cmmm_traction_q15: int = 32767,
) -> VoiceFrame:
    """Create a VoiceFrame for testing."""
    frame = VoiceFrame(
        note=note,
        gain_q15=gain_q15,
        coeff_osc_mode=coeff_osc_mode,
    )
    # Add CMMM fields if provided
    if cmmm_node_mask is not None:
        frame.cmmm_node_mask = cmmm_node_mask
        frame.cmmm_mask_ramp_blocks = cmmm_mask_ramp_blocks
    if cmmm_tail_mode is not None:
        frame.cmmm_tail_mode = cmmm_tail_mode
        frame.cmmm_traction_q15 = cmmm_traction_q15
    return frame


class TestDeterministicReplay:
    """Test that identical frame sequences produce identical outputs."""

    def test_basic_replay_determinism(self):
        """Basic frames without CMMM events should replay identically."""
        frames = [
            create_test_voiceframe(note=60),
            create_test_voiceframe(note=60),
            create_test_voiceframe(note=60),
        ]

        # First run
        state1 = VMState()
        outputs1 = []
        for frame in frames:
            state1, output = vm_step_block(state1, frame)
            outputs1.append(output.copy())

        # Second run with fresh state
        state2 = VMState()
        outputs2 = []
        for frame in frames:
            state2, output = vm_step_block(state2, frame)
            outputs2.append(output.copy())

        # Verify bit-exact match
        for i, (o1, o2) in enumerate(zip(outputs1, outputs2)):
            assert np.array_equal(o1, o2), f"Block {i} mismatch"

    def test_cmmm_mask_event_replay(self):
        """Frames with node mask events should replay identically."""
        frames = [
            create_test_voiceframe(note=60),
            create_test_voiceframe(note=60, cmmm_node_mask=0x0000FFFF, cmmm_mask_ramp_blocks=3),
            create_test_voiceframe(note=60),  # Ramp continues
            create_test_voiceframe(note=60),  # Ramp continues
            create_test_voiceframe(note=60),  # Ramp complete
        ]

        # First run
        state1 = VMState()
        outputs1 = []
        masks1 = []
        for frame in frames:
            state1, output = vm_step_block(state1, frame)
            outputs1.append(output.copy())
            masks1.append(state1.executor_latch.node_mask_u32)

        # Second run
        state2 = VMState()
        outputs2 = []
        masks2 = []
        for frame in frames:
            state2, output = vm_step_block(state2, frame)
            outputs2.append(output.copy())
            masks2.append(state2.executor_latch.node_mask_u32)

        # Verify outputs match
        for i, (o1, o2) in enumerate(zip(outputs1, outputs2)):
            assert np.array_equal(o1, o2), f"Block {i} output mismatch"

        # Verify mask state matches
        assert masks1 == masks2, "Mask state diverged"

    def test_cmmm_tailmode_event_replay(self):
        """Frames with tail mode events should replay identically."""
        frames = [
            create_test_voiceframe(note=60),
            create_test_voiceframe(note=60, cmmm_tail_mode=1, cmmm_traction_q15=16384),
            create_test_voiceframe(note=60),
            create_test_voiceframe(note=60, cmmm_tail_mode=2, cmmm_traction_q15=8192),
            create_test_voiceframe(note=60),
        ]

        # First run
        state1 = VMState()
        outputs1 = []
        for frame in frames:
            state1, output = vm_step_block(state1, frame)
            outputs1.append(output.copy())

        # Second run
        state2 = VMState()
        outputs2 = []
        for frame in frames:
            state2, output = vm_step_block(state2, frame)
            outputs2.append(output.copy())

        # Verify bit-exact match
        for i, (o1, o2) in enumerate(zip(outputs1, outputs2)):
            assert np.array_equal(o1, o2), f"Block {i} mismatch"

    def test_combined_events_replay(self):
        """Complex event sequences should replay identically."""
        frames = [
            create_test_voiceframe(note=60),
            create_test_voiceframe(
                note=62,
                cmmm_node_mask=0x00FF00FF,
                cmmm_mask_ramp_blocks=5,
                cmmm_tail_mode=1,
            ),
            create_test_voiceframe(note=64),
            create_test_voiceframe(note=64, cmmm_node_mask=0x0F0F0F0F, cmmm_mask_ramp_blocks=3),
            create_test_voiceframe(note=62),
            create_test_voiceframe(note=60, cmmm_tail_mode=0),
        ]

        # First run
        state1 = VMState()
        outputs1 = []
        for frame in frames:
            state1, output = vm_step_block(state1, frame)
            outputs1.append(output.copy())

        # Second run
        state2 = VMState()
        outputs2 = []
        for frame in frames:
            state2, output = vm_step_block(state2, frame)
            outputs2.append(output.copy())

        # Verify bit-exact match
        for i, (o1, o2) in enumerate(zip(outputs1, outputs2)):
            assert np.array_equal(o1, o2), f"Block {i} mismatch"


class TestImmediateMaskBehavior:
    """Test that mask changes apply immediately (no crossfade ramp)."""

    def test_mask_applied_immediately(self):
        """Mask event should take effect on the same block."""
        frames = [
            create_test_voiceframe(note=60),
            create_test_voiceframe(note=60, cmmm_node_mask=0x0000FFFF, cmmm_mask_ramp_blocks=5),
            create_test_voiceframe(note=60),
        ]

        state = VMState()
        masks = []

        for frame in frames:
            state, _ = vm_step_block(state, frame)
            masks.append(state.executor_latch.node_mask_u32)

        # Frame 0: default mask
        assert masks[0] == 0xFFFFFFFF
        # Frame 1: mask applied immediately (ramp_blocks ignored)
        assert masks[1] == 0x0000FFFF
        # Frame 2: stays at new mask
        assert masks[2] == 0x0000FFFF

    def test_mask_arrives_at_target(self):
        """Mask should be at target immediately after event."""
        state = VMState()
        target_mask = 0x0000FFFF

        # Initial frame
        frame1 = create_test_voiceframe(note=60)
        state, _ = vm_step_block(state, frame1)

        # Event frame
        frame2 = create_test_voiceframe(
            note=60,
            cmmm_node_mask=target_mask,
            cmmm_mask_ramp_blocks=2,
        )
        state, _ = vm_step_block(state, frame2)

        # Verify mask is already at target
        assert state.executor_latch.node_mask_u32 == target_mask


class TestStateSnapshot:
    """Test that state snapshot/restore works with CMMM latch."""

    def test_latch_state_survives_snapshot(self):
        """CMMM latch state should survive snapshot/restore."""
        state = VMState()

        # Process frames with CMMM events
        frames = [
            create_test_voiceframe(note=60),
            create_test_voiceframe(note=60, cmmm_node_mask=0x00FF00FF, cmmm_mask_ramp_blocks=3),
            create_test_voiceframe(note=60, cmmm_tail_mode=2, cmmm_traction_q15=16384),
        ]

        for frame in frames:
            state, _ = vm_step_block(state, frame)

        # Snapshot
        snapshot = state.snapshot()

        # Restore
        restored = VMState.restore(snapshot)

        # Verify latch state
        assert restored.executor_latch.node_mask_u32 == state.executor_latch.node_mask_u32
        assert restored.executor_latch.tail_mode == state.executor_latch.tail_mode
        assert restored.executor_latch.traction_q15 == state.executor_latch.traction_q15

    def test_restored_state_produces_same_output(self):
        """Restored state should produce same output as original."""
        state = VMState()

        # Process some frames
        for _ in range(3):
            state, _ = vm_step_block(state, create_test_voiceframe(note=60))

        # Snapshot mid-stream
        snapshot = state.snapshot()
        restored = VMState.restore(snapshot)

        # Continue from both
        frame = create_test_voiceframe(note=62)
        _, output_original = vm_step_block(state, frame)
        _, output_restored = vm_step_block(restored, frame)

        # Should match
        assert np.array_equal(output_original, output_restored)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
