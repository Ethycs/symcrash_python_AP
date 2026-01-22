"""
Tests for CMMM Enrichment Gap G2: CMMMemory
==========================================

Tests the CMMMemory dataclass, tension/fatigue dynamics, and VMState integration.

Reference: CMMM-ENRICHMENT-DESIGN.md §9.3
"""

import pytest

from symcrash_python_ap.primitives import Q15_ONE, Q15_HALF
from symcrash_python_ap.vm.state import VMState, CMMMemory


class TestCMMMemoryBasics:
    """Tests for CMMMemory dataclass basics."""

    def test_default_values(self):
        """CMMMemory should start with all zeros."""
        mem = CMMMemory()
        assert mem.tension_q15 == 0
        assert mem.fatigue_q15 == 0
        assert mem.harmonic_memory_q15 == 0
        assert mem.blocks_since_transition == 0

    def test_decay_constants_defined(self):
        """Decay constants should be defined."""
        mem = CMMMemory()
        assert hasattr(mem, 'TENSION_DECAY')
        assert hasattr(mem, 'FATIGUE_DECAY')
        assert hasattr(mem, 'HARMONIC_DECAY')
        assert mem.TENSION_DECAY > 0
        assert mem.FATIGUE_DECAY > 0
        assert mem.HARMONIC_DECAY > 0


class TestCMMMemoryStep:
    """Tests for CMMMemory.step() behavior."""

    def test_tension_builds_with_high_energy(self):
        """High intent energy should increase tension."""
        mem = CMMMemory()
        initial_tension = mem.tension_q15

        # Step with high energy
        mem.step(Q15_ONE, is_transition=False)

        # Tension should have increased (minus decay)
        # High energy gain = Q15_ONE >> 3 = 4096
        # Net = 4096 - decay(328) = 3768 > 0
        assert mem.tension_q15 > initial_tension

    def test_tension_decays_with_low_energy(self):
        """Low intent energy should allow tension to decay."""
        mem = CMMMemory()
        mem.tension_q15 = 10000  # Start with tension

        # Step with zero energy
        mem.step(0, is_transition=False)

        # Tension should have decreased
        assert mem.tension_q15 < 10000

    def test_tension_capped_at_q15_one(self):
        """Tension should not exceed Q15_ONE."""
        mem = CMMMemory()
        mem.tension_q15 = Q15_ONE - 100

        # Step with max energy many times
        for _ in range(100):
            mem.step(Q15_ONE, is_transition=False)

        assert mem.tension_q15 <= Q15_ONE

    def test_tension_floor_at_zero(self):
        """Tension should not go below zero."""
        mem = CMMMemory()
        mem.tension_q15 = 100

        # Step with zero energy many times to decay
        for _ in range(1000):
            mem.step(0, is_transition=False)

        assert mem.tension_q15 >= 0

    def test_fatigue_builds_above_threshold(self):
        """Fatigue should build when intent energy > Q15_HALF."""
        mem = CMMMemory()
        initial_fatigue = mem.fatigue_q15

        # Step with high energy (above Q15_HALF threshold)
        for _ in range(100):
            mem.step(Q15_ONE, is_transition=False)

        # Fatigue should have increased
        assert mem.fatigue_q15 > initial_fatigue

    def test_fatigue_does_not_build_below_threshold(self):
        """Fatigue should not build when intent energy < Q15_HALF."""
        mem = CMMMemory()

        # Step with low energy many times
        # But start with some fatigue to see decay
        mem.fatigue_q15 = 5000
        initial_fatigue = mem.fatigue_q15

        for _ in range(10):
            mem.step(Q15_HALF - 1, is_transition=False)

        # Fatigue should only decay, not build
        assert mem.fatigue_q15 < initial_fatigue

    def test_harmonic_memory_decays(self):
        """Harmonic memory should decay each step."""
        mem = CMMMemory()
        mem.harmonic_memory_q15 = 10000

        mem.step(0, is_transition=False)

        assert mem.harmonic_memory_q15 < 10000

    def test_blocks_since_transition_increments(self):
        """blocks_since_transition should increment each step."""
        mem = CMMMemory()
        assert mem.blocks_since_transition == 0

        mem.step(0, is_transition=False)
        assert mem.blocks_since_transition == 1

        mem.step(0, is_transition=False)
        assert mem.blocks_since_transition == 2

    def test_transition_resets_counter(self):
        """is_transition=True should reset blocks_since_transition."""
        mem = CMMMemory()
        mem.blocks_since_transition = 100

        mem.step(0, is_transition=True)

        assert mem.blocks_since_transition == 0


class TestCMMMemorySetters:
    """Tests for CMMMemory setter methods."""

    def test_set_harmonic_memory(self):
        """set_harmonic_memory should set value within bounds."""
        mem = CMMMemory()

        mem.set_harmonic_memory(12345)
        assert mem.harmonic_memory_q15 == 12345

    def test_set_harmonic_memory_clamped_high(self):
        """set_harmonic_memory should clamp to Q15_ONE."""
        mem = CMMMemory()

        mem.set_harmonic_memory(Q15_ONE + 1000)
        assert mem.harmonic_memory_q15 == Q15_ONE

    def test_set_harmonic_memory_clamped_low(self):
        """set_harmonic_memory should clamp to 0."""
        mem = CMMMemory()

        mem.set_harmonic_memory(-1000)
        assert mem.harmonic_memory_q15 == 0


class TestCMMMemoryReset:
    """Tests for CMMMemory.reset_for_note_on()."""

    def test_reset_clears_all_fields(self):
        """reset_for_note_on should clear all fields."""
        mem = CMMMemory()
        mem.tension_q15 = 10000
        mem.fatigue_q15 = 8000
        mem.harmonic_memory_q15 = 5000
        mem.blocks_since_transition = 500

        mem.reset_for_note_on()

        assert mem.tension_q15 == 0
        assert mem.fatigue_q15 == 0
        assert mem.harmonic_memory_q15 == 0
        assert mem.blocks_since_transition == 0


class TestCMMMemorySerialization:
    """Tests for CMMMemory to_dict/from_dict."""

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        mem = CMMMemory()
        mem.tension_q15 = 1234
        mem.fatigue_q15 = 5678
        mem.harmonic_memory_q15 = 9012
        mem.blocks_since_transition = 42

        d = mem.to_dict()

        assert d['tension_q15'] == 1234
        assert d['fatigue_q15'] == 5678
        assert d['harmonic_memory_q15'] == 9012
        assert d['blocks_since_transition'] == 42

    def test_from_dict(self):
        """from_dict should deserialize correctly."""
        d = {
            'tension_q15': 1111,
            'fatigue_q15': 2222,
            'harmonic_memory_q15': 3333,
            'blocks_since_transition': 99,
        }

        mem = CMMMemory.from_dict(d)

        assert mem.tension_q15 == 1111
        assert mem.fatigue_q15 == 2222
        assert mem.harmonic_memory_q15 == 3333
        assert mem.blocks_since_transition == 99

    def test_from_dict_missing_fields(self):
        """from_dict should handle missing fields with defaults."""
        d = {'tension_q15': 500}

        mem = CMMMemory.from_dict(d)

        assert mem.tension_q15 == 500
        assert mem.fatigue_q15 == 0  # Default
        assert mem.harmonic_memory_q15 == 0  # Default
        assert mem.blocks_since_transition == 0  # Default

    def test_roundtrip(self):
        """to_dict -> from_dict should preserve values."""
        original = CMMMemory()
        original.tension_q15 = 7777
        original.fatigue_q15 = 8888
        original.harmonic_memory_q15 = 9999
        original.blocks_since_transition = 123

        restored = CMMMemory.from_dict(original.to_dict())

        assert restored.tension_q15 == original.tension_q15
        assert restored.fatigue_q15 == original.fatigue_q15
        assert restored.harmonic_memory_q15 == original.harmonic_memory_q15
        assert restored.blocks_since_transition == original.blocks_since_transition


class TestVMStateMemoryIntegration:
    """Tests for CMMMemory integration with VMState."""

    def test_vmstate_has_cmmm_memory(self):
        """VMState should have cmmm_memory field."""
        state = VMState.create(voice_id=0)
        assert hasattr(state, 'cmmm_memory')
        assert isinstance(state.cmmm_memory, CMMMemory)

    def test_vmstate_reset_resets_memory(self):
        """reset_for_note_on should reset cmmm_memory."""
        state = VMState.create(voice_id=0)
        state.cmmm_memory.tension_q15 = 9999
        state.cmmm_memory.fatigue_q15 = 8888

        state.reset_for_note_on(note=60)

        assert state.cmmm_memory.tension_q15 == 0
        assert state.cmmm_memory.fatigue_q15 == 0

    def test_vmstate_snapshot_includes_memory(self):
        """VMState snapshot should include cmmm_memory."""
        state = VMState.create(voice_id=0)
        state.cmmm_memory.tension_q15 = 5000
        state.cmmm_memory.fatigue_q15 = 3000

        snapshot = state.snapshot()

        assert 'cmmm_memory' in snapshot
        assert snapshot['cmmm_memory']['tension_q15'] == 5000
        assert snapshot['cmmm_memory']['fatigue_q15'] == 3000

    def test_vmstate_restore_recovers_memory(self):
        """VMState.restore should recover cmmm_memory."""
        state = VMState.create(voice_id=0)
        state.cmmm_memory.tension_q15 = 12000
        state.cmmm_memory.fatigue_q15 = 6000
        state.cmmm_memory.blocks_since_transition = 42

        snapshot = state.snapshot()
        restored = VMState.restore(snapshot)

        assert restored.cmmm_memory.tension_q15 == 12000
        assert restored.cmmm_memory.fatigue_q15 == 6000
        assert restored.cmmm_memory.blocks_since_transition == 42


class TestMemoryDynamicsRealistic:
    """Realistic tests for memory dynamics over many blocks."""

    def test_tension_stabilizes(self):
        """With constant input, tension should stabilize."""
        mem = CMMMemory()
        constant_energy = Q15_HALF

        prev_tension = -1
        for _ in range(1000):
            mem.step(constant_energy, is_transition=False)
            # Check for stabilization (small change)
            if prev_tension >= 0 and abs(mem.tension_q15 - prev_tension) < 10:
                break
            prev_tension = mem.tension_q15

        # Should have stabilized
        mem.step(constant_energy, is_transition=False)
        assert abs(mem.tension_q15 - prev_tension) < 100

    def test_fatigue_builds_over_sustained_activity(self):
        """Fatigue should build significantly over sustained high activity."""
        mem = CMMMemory()

        # Simulate 1 second (~750 blocks at 750 Hz block rate) of high activity
        for _ in range(750):
            mem.step(Q15_ONE, is_transition=False)

        # Fatigue should be substantial
        assert mem.fatigue_q15 > Q15_HALF // 2  # At least 25%


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
