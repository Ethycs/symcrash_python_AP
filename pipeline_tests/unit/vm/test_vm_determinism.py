"""
Unit tests for VM determinism (WSE-STD-VM-HYBRID v0)

These tests verify bit-exact output across runs.
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
from symcrash_python_ap.vm.tables import DEFAULT_TABLES, BLOCK_SAMPLES


# =============================================================================
# MOCK VOICEFRAME
# =============================================================================

@dataclass
class MockVoiceFrame:
    """Mock VoiceFrame for testing."""
    voice_id: int = 0
    note: int = 69        # A4 (440 Hz)
    cents: int = 0
    f0_hz: float = 440.0  # Deprecated, kept for legacy
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
# DETERMINISM TESTS
# =============================================================================

class TestBitExactDeterminism:
    """Tests for bit-exact deterministic output."""

    def test_byte_exact_output(self):
        """Same VoiceFrame sequence must produce identical output."""
        frames = [MockVoiceFrame(f0_hz=440.0, gain_q15=32767) for _ in range(10)]

        # Run 1
        state1 = VMState.create(voice_id=0)
        state1.reset_for_note_on(f0_hz=440.0)
        outputs1 = []
        for f in frames:
            state1, out = vm_step_block(state1, f, DEFAULT_TABLES)
            outputs1.append(out.copy())
        prng1 = state1.prng_state

        # Run 2
        state2 = VMState.create(voice_id=0)
        state2.reset_for_note_on(f0_hz=440.0)
        outputs2 = []
        for f in frames:
            state2, out = vm_step_block(state2, f, DEFAULT_TABLES)
            outputs2.append(out.copy())
        prng2 = state2.prng_state

        # Verify byte-exact
        for i, (o1, o2) in enumerate(zip(outputs1, outputs2)):
            assert np.array_equal(o1, o2), f"Output block {i} not byte-exact"

        assert prng1 == prng2, "PRNG state diverged"

    def test_prng_only_advances_on_noise(self):
        """PRNG should not advance without noise operations."""
        state = VMState.create(voice_id=0)
        state.reset_for_note_on(f0_hz=440.0)
        initial_prng = state.prng_state

        # Execute 10 blocks without noise ops
        frame = MockVoiceFrame(f0_hz=440.0, event_mask=0)  # No events
        for _ in range(10):
            state, _ = vm_step_block(state, frame, DEFAULT_TABLES)

        # PRNG should not have changed
        assert state.prng_state == initial_prng, "PRNG advanced without noise ops"

    def test_different_voices_produce_different_prng(self):
        """Different voice IDs should produce different PRNG sequences."""
        state0 = VMState.create(voice_id=0)
        state1 = VMState.create(voice_id=1)

        assert state0.prng_state != state1.prng_state, "Same PRNG for different voices"

    def test_snapshot_restore_determinism(self):
        """Restoring from snapshot should produce identical output."""
        state = VMState.create(voice_id=0)
        state.reset_for_note_on(f0_hz=440.0)

        # Run 5 blocks
        frame = MockVoiceFrame(f0_hz=440.0)
        for _ in range(5):
            state, _ = vm_step_block(state, frame, DEFAULT_TABLES)

        # Snapshot
        snapshot = state.snapshot()

        # Continue for 5 more blocks
        outputs_a = []
        for _ in range(5):
            state, out = vm_step_block(state, frame, DEFAULT_TABLES)
            outputs_a.append(out.copy())

        # Restore and run again
        state = VMState.restore(snapshot)
        outputs_b = []
        for _ in range(5):
            state, out = vm_step_block(state, frame, DEFAULT_TABLES)
            outputs_b.append(out.copy())

        # Should be identical
        for i, (a, b) in enumerate(zip(outputs_a, outputs_b)):
            assert np.array_equal(a, b), f"Block {i} differs after restore"

    def test_gain_ramp_is_smooth(self):
        """Gain ramping should produce smooth output without clicks."""
        state = VMState.create(voice_id=0)
        state.reset_for_note_on(f0_hz=440.0)

        # Run a few blocks to let carrier stabilize
        frame = MockVoiceFrame(f0_hz=440.0, gain_q15=32767)
        for _ in range(5):
            state, _ = vm_step_block(state, frame, DEFAULT_TABLES)

        # Block with full gain
        frame1 = MockVoiceFrame(f0_hz=440.0, gain_q15=32767)
        state, out1 = vm_step_block(state, frame1, DEFAULT_TABLES)
        rms1 = np.sqrt(np.mean(out1.astype(np.float64) ** 2))

        # Block with half gain (ramping from full)
        frame2 = MockVoiceFrame(f0_hz=440.0, gain_q15=16384)
        state, out2 = vm_step_block(state, frame2, DEFAULT_TABLES)
        rms2 = np.sqrt(np.mean(out2.astype(np.float64) ** 2))

        # Block with half gain (now stable at half)
        frame3 = MockVoiceFrame(f0_hz=440.0, gain_q15=16384)
        state, out3 = vm_step_block(state, frame3, DEFAULT_TABLES)
        rms3 = np.sqrt(np.mean(out3.astype(np.float64) ** 2))

        # After stabilization, rms3 should be roughly half of rms1
        # Allow some tolerance for waveform shape differences
        assert rms3 < rms1 * 0.75, f"Gain reduction not reflected: rms3={rms3}, rms1={rms1}"
        assert rms3 > rms1 * 0.25, f"Gain too low: rms3={rms3}, rms1={rms1}"

    def test_eta_ramp_is_smooth(self):
        """Eta (dry/wet) ramping should be smooth."""
        state = VMState.create(voice_id=0)
        state.reset_for_note_on(f0_hz=440.0)

        # Wait for wet to be ready (5 blocks)
        frame_dry = MockVoiceFrame(f0_hz=440.0, beta_target_q15=0)
        for _ in range(6):
            state, _ = vm_step_block(state, frame_dry, DEFAULT_TABLES)

        # Now ramp from dry to wet
        frame_wet = MockVoiceFrame(f0_hz=440.0, beta_target_q15=32767)
        state, out = vm_step_block(state, frame_wet, DEFAULT_TABLES)

        # Should have non-zero output (not silent)
        assert np.max(np.abs(out)) > 0, "No output during eta ramp"

    def test_integer_sqrt_determinism(self):
        """Integer sqrt should be deterministic across all inputs."""
        from symcrash_python_ap.vm.lanes import isqrt, isqrt_ratio_q15

        # Test known values
        assert isqrt(0) == 0
        assert isqrt(1) == 1
        assert isqrt(4) == 2
        assert isqrt(9) == 3
        assert isqrt(100) == 10
        assert isqrt(1000000) == 1000

        # Test floor behavior
        assert isqrt(2) == 1
        assert isqrt(3) == 1
        assert isqrt(5) == 2
        assert isqrt(8) == 2

        # Test large values
        assert isqrt(2**30) == 2**15
        assert isqrt(2**60) == 2**30

        # Test ratio function
        # sqrt(1) * Q15 should be close to Q15_ONE (allow ±1 for rounding)
        ratio_1 = isqrt_ratio_q15(32767, 32767)
        assert 32766 <= ratio_1 <= 32768, f"sqrt(1) ratio out of range: {ratio_1}"
        assert isqrt_ratio_q15(0, 100) == 0
        assert isqrt_ratio_q15(100, 0) == 0

    def test_trig_lookup_determinism(self):
        """Trig lookup tables should return consistent values."""
        from symcrash_python_ap.vm.lanes import cos_q15, sin_q15

        # Test boundary values
        assert cos_q15(0) == 32767  # cos(0) = 1
        assert sin_q15(0) == 0      # sin(0) = 0

        # Test quarter angle
        mid = 16384  # Q15_ONE / 2
        cos_mid = cos_q15(mid)
        sin_mid = sin_q15(mid)

        # At pi/4, cos and sin should be roughly equal (both ~0.707)
        # In Q15, that's ~23170
        assert 20000 < cos_mid < 26000, f"cos(pi/4) out of range: {cos_mid}"
        assert 20000 < sin_mid < 26000, f"sin(pi/4) out of range: {sin_mid}"

        # Test multiple runs produce same result
        for angle in [0, 8192, 16384, 24576, 32767]:
            c1 = cos_q15(angle)
            c2 = cos_q15(angle)
            s1 = sin_q15(angle)
            s2 = sin_q15(angle)
            assert c1 == c2, f"cos_q15({angle}) not deterministic"
            assert s1 == s2, f"sin_q15({angle}) not deterministic"


class TestPhaseConsistency:
    """Tests for phase continuity."""

    def _get_phase(self, state: VMState) -> int:
        """Get the active phase accumulator (CCS or legacy)."""
        if state.use_coeff_osc:
            return int(state.coeff_osc.node_phases[0])
        return state.phase

    def test_phase_increases_monotonically(self):
        """Phase should increase monotonically (modulo wrap)."""
        state = VMState.create(voice_id=0)
        state.reset_for_note_on(f0_hz=440.0)

        frame = MockVoiceFrame(f0_hz=440.0)
        phases = [self._get_phase(state)]

        for _ in range(10):
            state, _ = vm_step_block(state, frame, DEFAULT_TABLES)
            phases.append(self._get_phase(state))

        # Phase should advance (or wrap)
        for i in range(1, len(phases)):
            diff = (phases[i] - phases[i-1]) & 0xFFFFFFFF
            # Either advanced or wrapped around
            assert diff > 0 or phases[i] < phases[i-1], f"Phase not advancing at block {i}"

    def test_phase_not_reset_between_blocks(self):
        """Phase should not reset to 0 between blocks."""
        state = VMState.create(voice_id=0)
        state.reset_for_note_on(f0_hz=440.0)

        frame = MockVoiceFrame(f0_hz=440.0)

        # Run a few blocks
        for _ in range(3):
            state, _ = vm_step_block(state, frame, DEFAULT_TABLES)

        # Phase should not be 0 (unless very unlikely coincidence)
        phase = self._get_phase(state)
        assert phase != 0, "Phase reset to 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
