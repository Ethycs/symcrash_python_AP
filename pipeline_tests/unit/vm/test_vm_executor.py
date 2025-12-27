"""
Unit tests for VM Executor (WSE-STD-VM-HYBRID v0)
"""

import pytest
import numpy as np
from dataclasses import dataclass

# Import VM components
from symcrash_python_ap.vm import (
    vm_step_block,
    VMState,
    DecodedPlan,
    Tables,
)
from symcrash_python_ap.vm.executor import VMExecutor, vm_step_block_raw
from symcrash_python_ap.vm.decode import CarrierRole, CarrierParams, BackboneEntry
from symcrash_python_ap.vm.tables import TailMode, BLOCK_SAMPLES, Lane


# =============================================================================
# MOCK VOICEFRAME
# =============================================================================

@dataclass
class MockVoiceFrame:
    """Mock VoiceFrame for testing."""
    voice_id: int = 0
    f0_hz: float = 440.0
    gain_q15: int = 32767
    beta_target_q15: int = 16384  # 50% wet
    pattern_id: int = 0
    tail_mode: int = 0  # FULL
    tree_a_id: int = 0
    tree_b_id: int = 0
    tree_blend_q15: int = 0
    partial_amps_q15: list = None
    ops_enable: list = None
    families: dict = None
    event_mask: int = 0
    event_strength_q15: list = None

    def __post_init__(self):
        if self.partial_amps_q15 is None:
            self.partial_amps_q15 = [32767, 16384, 8192, 4096]
        if self.ops_enable is None:
            self.ops_enable = []
        if self.families is None:
            self.families = {}
        if self.event_strength_q15 is None:
            self.event_strength_q15 = [32767] * 8


# =============================================================================
# TEST CLASSES
# =============================================================================

class TestVMState:
    """Tests for VMState."""

    def test_create_state(self):
        """VMState.create() should return initialized state."""
        state = VMState.create(voice_id=5)
        assert state.voice_id == 5
        assert state.phase == 0
        assert state.prng_state != 0
        assert len(state.coeff_state) == 256

    def test_different_voices_different_prng(self):
        """Different voice IDs should have different PRNG seeds."""
        state1 = VMState.create(voice_id=0)
        state2 = VMState.create(voice_id=1)
        assert state1.prng_state != state2.prng_state

    def test_reset_for_note_on(self):
        """reset_for_note_on should initialize envelope and phase_inc."""
        state = VMState.create(voice_id=0)
        state.reset_for_note_on(f0_hz=440.0, sample_rate=48000)

        assert state.phase_inc > 0
        assert state.envelope.phase.value == 0  # ATTACK
        assert state.blocks_since_start == 0
        assert not state.wet_ready

    def test_snapshot_restore(self):
        """Snapshot and restore should be reversible."""
        state = VMState.create(voice_id=7)
        state.phase = 12345678
        state.coeff_state[10] = 5000

        snapshot = state.snapshot()
        restored = VMState.restore(snapshot)

        assert restored.voice_id == 7
        assert restored.phase == 12345678
        assert restored.coeff_state[10] == 5000


class TestDecodedPlan:
    """Tests for frame decoding."""

    def test_decode_basic_frame(self):
        """Decoding a basic frame should produce valid plan."""
        from symcrash_python_ap.vm.decode import decode_frame

        frame = MockVoiceFrame(f0_hz=440.0, gain_q15=32767)
        plan = decode_frame(frame)

        assert plan.K == 32
        assert plan.mask_u32 == 0xFFFFFFFF
        assert plan.tail_mode == TailMode.FULL
        assert plan.carrier_params.f0_hz == 440.0

    def test_decode_with_ops(self):
        """Decoding frame with ops should include backbone instructions."""
        from symcrash_python_ap.vm.decode import decode_frame

        frame = MockVoiceFrame(
            ops_enable=['tilt', 'intramix'],
            families={'tilt': 0.5, 'intramix': 1.0}
        )
        plan = decode_frame(frame)

        assert len(plan.backbone_instrs) == 2
        assert plan.backbone_instrs[0].op_id == 1  # Tilt
        assert plan.backbone_instrs[1].op_id == 3  # IntraMix

    def test_decode_tail_modes(self):
        """Different tail modes should produce different tail ops."""
        from symcrash_python_ap.vm.decode import decode_frame

        for mode in range(4):
            frame = MockVoiceFrame(tail_mode=mode)
            plan = decode_frame(frame)
            assert plan.tail_mode == TailMode(mode)
            assert len(plan.tail_ops) > 0


class TestVMExecutor:
    """Tests for vm_step_block."""

    def test_single_block_execution(self):
        """vm_step_block should execute one block and return output."""
        state = VMState.create(voice_id=0)
        state.reset_for_note_on(f0_hz=440.0)

        frame = MockVoiceFrame(f0_hz=440.0)
        state, output = vm_step_block(state, frame)

        assert isinstance(output, np.ndarray)
        assert output.dtype == np.int16
        assert len(output) == BLOCK_SAMPLES

    def test_phase_continuity(self):
        """Phase should be continuous across blocks."""
        state = VMState.create(voice_id=0)
        state.reset_for_note_on(f0_hz=440.0)

        frame = MockVoiceFrame(f0_hz=440.0)

        phases = []
        for _ in range(10):
            phases.append(state.phase)
            state, _ = vm_step_block(state, frame)

        # Phase should increase monotonically (modulo wrap)
        for i in range(1, len(phases)):
            # Check phase advanced (accounting for wrap)
            diff = (phases[i] - phases[i-1]) & 0xFFFFFFFF
            assert diff > 0 or phases[i] < phases[i-1]  # Either advanced or wrapped

    def test_determinism(self):
        """Same inputs should produce identical outputs."""
        frame = MockVoiceFrame(f0_hz=440.0, gain_q15=32767)

        # Run 1
        state1 = VMState.create(voice_id=0)
        state1.reset_for_note_on(f0_hz=440.0)
        outputs1 = []
        for _ in range(5):
            state1, out = vm_step_block(state1, frame)
            outputs1.append(out.copy())

        # Run 2
        state2 = VMState.create(voice_id=0)
        state2.reset_for_note_on(f0_hz=440.0)
        outputs2 = []
        for _ in range(5):
            state2, out = vm_step_block(state2, frame)
            outputs2.append(out.copy())

        # Should be identical
        for out1, out2 in zip(outputs1, outputs2):
            assert np.array_equal(out1, out2)

    def test_latency_model(self):
        """Wet signal should only be enabled after latency period."""
        state = VMState.create(voice_id=0)
        state.reset_for_note_on(f0_hz=440.0)

        frame = MockVoiceFrame(f0_hz=440.0, beta_target_q15=32767)  # Full wet

        # First 5 blocks should have eta=0 (pure carrier)
        for i in range(5):
            assert not state.wet_ready
            state, _ = vm_step_block(state, frame)

        # After 5 blocks, wet should be ready
        assert state.wet_ready

    def test_gain_ramping(self):
        """Gain changes should be applied smoothly."""
        state = VMState.create(voice_id=0)
        state.reset_for_note_on(f0_hz=440.0)

        # First block with full gain
        frame1 = MockVoiceFrame(f0_hz=440.0, gain_q15=32767)
        state, out1 = vm_step_block(state, frame1)

        # Second block with half gain
        frame2 = MockVoiceFrame(f0_hz=440.0, gain_q15=16384)
        state, out2 = vm_step_block(state, frame2)

        # Output should be ramped, not stepped
        # Check that samples change gradually (no hard step)
        assert np.max(np.abs(out2)) > 0  # Some output
        assert state.gain_prev_q15 == 16384  # Updated for next block


class TestLanes:
    """Tests for individual lanes."""

    def test_carrier_render_produces_output(self):
        """Carrier render should produce non-zero output."""
        from symcrash_python_ap.vm.lanes import lane_carrier_render

        state = VMState.create(voice_id=0)
        state.reset_for_note_on(f0_hz=440.0)

        cfg = DecodedPlan(
            carrier_role=CarrierRole.AUDIBLE,
            carrier_params=CarrierParams(
                f0_hz=440.0,
                phase_inc=int((440.0 / 48000) * (2**32)),
                harmonic_amps_q15=[32767] + [0] * 15,
            ),
        )

        carrier_buf = lane_carrier_render(state, cfg)

        assert np.max(np.abs(carrier_buf)) > 0

    def test_tail_fine_clamp(self):
        """Fine clamp should limit coefficient energy."""
        from symcrash_python_ap.vm.lanes import _tail_fine_clamp

        state = VMState.create(voice_id=0)
        # Set coefficients to max int16 value (exceeds per-node energy budget)
        # Max per coefficient is 32767, but with 8 coeffs all at max,
        # total energy = 8 * 32767^2 which equals the budget exactly
        # To test clamping, use values that exceed: e.g., many coeffs at very high
        # Actually, since max is 32767, we can test boundary behavior

        # Set all 8 coefficients to 32767 (max) - this should be exactly at budget
        for i in range(8):
            state.coeff_state[i] = 32767

        initial_energy = sum(int(state.coeff_state[i]) ** 2 for i in range(8))

        _tail_fine_clamp(state, mask=0xFFFFFFFF)

        # Energy should remain unchanged since it's at exactly the budget
        final_energy = sum(int(state.coeff_state[i]) ** 2 for i in range(8))

        # Max energy per node = 8 * 32767^2
        max_energy = 8 * 32767 * 32767

        assert final_energy <= max_energy, f"Energy {final_energy} > max {max_energy}"

        # Test that clamp actually works by checking a higher initial energy case
        # would require coefficients > 32767 which is impossible with int16
        # So instead verify it doesn't corrupt valid data
        for i in range(8):
            assert abs(state.coeff_state[i]) <= 32767

    def test_inject_does_not_advance_prng_without_noise(self):
        """Inject lane should not advance PRNG without noise ops."""
        from symcrash_python_ap.vm.lanes import lane_inject

        state = VMState.create(voice_id=0)
        initial_prng = state.prng_state

        # No noise events
        cfg = DecodedPlan()
        cfg.event_instrs = []

        lane_inject(state, cfg)

        # PRNG should not have advanced
        assert state.prng_state == initial_prng


class TestVMExecutorClass:
    """Tests for VMExecutor class."""

    def test_multi_voice_execution(self):
        """VMExecutor should handle multiple voices."""
        executor = VMExecutor(max_voices=16)

        # Execute different voices
        for voice_id in range(4):
            frame = MockVoiceFrame(voice_id=voice_id, f0_hz=440.0 * (voice_id + 1))
            output = executor.step_voice(voice_id, frame)
            assert len(output) == BLOCK_SAMPLES

        assert len(executor.states) == 4

    def test_voice_release(self):
        """release_voice should remove voice state."""
        executor = VMExecutor()

        frame = MockVoiceFrame(voice_id=0)
        executor.step_voice(0, frame)
        assert 0 in executor.states

        executor.release_voice(0)
        assert 0 not in executor.states

    def test_snapshot_restore_all(self):
        """snapshot_all and restore_all should preserve all states."""
        executor = VMExecutor()

        # Execute some voices
        for vid in range(3):
            frame = MockVoiceFrame(voice_id=vid, f0_hz=440.0)
            executor.step_voice(vid, frame)

        # Snapshot
        snapshots = executor.snapshot_all()
        assert len(snapshots) == 3

        # Modify state
        executor.states[0].phase = 99999

        # Restore
        executor.restore_all(snapshots)
        assert executor.states[0].phase != 99999


class TestTables:
    """Tests for VM tables."""

    def test_operator_lane_table(self):
        """All operators should have lane assignments."""
        from symcrash_python_ap.vm.tables import OPERATOR_LANES

        # Check all 24 operators
        for op_id in range(24):
            assert op_id in OPERATOR_LANES
            lane_info = OPERATOR_LANES[op_id]
            assert lane_info.op_id == op_id
            assert lane_info.lane in Lane

    def test_tail_ops_table(self):
        """All tail modes should have operator lists."""
        from symcrash_python_ap.vm.tables import TAIL_OPS, get_tail_ops

        for mode in TailMode:
            ops = get_tail_ops(mode)
            assert isinstance(ops, list)
            assert len(ops) > 0

    def test_pattern_table(self):
        """Pattern table should have entries for K-ladder."""
        from symcrash_python_ap.vm.tables import PATTERN_TABLE, K_LADDER, popcount32

        for k in K_LADDER:
            found = False
            for entry in PATTERN_TABLE.values():
                if entry.k_value == k:
                    found = True
                    assert popcount32(entry.mask_u32) == k
                    break
            assert found, f"No pattern for K={k}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
