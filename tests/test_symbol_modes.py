"""
Tests for CMMM Enrichment Gap G1: Symbol Engine Modes
=====================================================

Tests the SymbolMode enum, ExecutorLatch mode fields, evolution rate scaling,
and regime-to-mode mapping.

Reference: CMMM-ENRICHMENT-DESIGN.md §9.2
"""

import pytest
import numpy as np

from symcrash_python_ap.primitives import (
    SymbolMode, SYMBOL_MODE_TABLE, get_symbol_mode_evolution_rate,
    REGIME_TO_SYMBOL_MODE, Q15_ONE, Q15_HALF,
)
from symcrash_python_ap.vm.state import VMState, ExecutorLatch


class TestSymbolModeEnum:
    """Tests for the SymbolMode enum and related tables."""

    def test_symbol_mode_values(self):
        """SymbolMode enum should have correct values."""
        assert SymbolMode.HOLD == 0
        assert SymbolMode.EVOLVE == 1
        assert SymbolMode.PHRASE_QUANTIZE == 2
        assert SymbolMode.FREEZE == 3

    def test_symbol_mode_table_complete(self):
        """SYMBOL_MODE_TABLE should have entries for all modes."""
        for mode in SymbolMode:
            assert mode.value in SYMBOL_MODE_TABLE
            info = SYMBOL_MODE_TABLE[mode.value]
            assert info.id == mode.value
            assert isinstance(info.evolution_rate_q15, int)

    def test_evolution_rate_ranges(self):
        """Evolution rates should be in valid Q15 range."""
        for info in SYMBOL_MODE_TABLE.values():
            assert 0 <= info.evolution_rate_q15 <= Q15_ONE

    def test_hold_rate_is_low(self):
        """HOLD mode should have low evolution rate (~3%)."""
        rate = get_symbol_mode_evolution_rate(SymbolMode.HOLD)
        assert rate > 0
        assert rate < Q15_ONE // 10  # Less than 10%

    def test_evolve_rate_is_full(self):
        """EVOLVE mode should have full evolution rate."""
        rate = get_symbol_mode_evolution_rate(SymbolMode.EVOLVE)
        assert rate == Q15_ONE

    def test_freeze_rate_is_zero(self):
        """FREEZE mode should have zero evolution rate."""
        rate = get_symbol_mode_evolution_rate(SymbolMode.FREEZE)
        assert rate == 0

    def test_phrase_quantize_rate_is_full(self):
        """PHRASE_QUANTIZE mode should have full rate (gating is separate)."""
        rate = get_symbol_mode_evolution_rate(SymbolMode.PHRASE_QUANTIZE)
        assert rate == Q15_ONE


class TestRegimeToModeMapping:
    """Tests for regime-to-symbol-mode mapping."""

    def test_mapping_complete(self):
        """All 7 regimes should have mode mappings."""
        for regime_id in range(7):
            assert regime_id in REGIME_TO_SYMBOL_MODE

    def test_idle_maps_to_hold(self):
        """R0_IDLE should map to HOLD mode."""
        assert REGIME_TO_SYMBOL_MODE[0] == SymbolMode.HOLD

    def test_attack_maps_to_evolve(self):
        """R1_ATTACK should map to EVOLVE mode."""
        assert REGIME_TO_SYMBOL_MODE[1] == SymbolMode.EVOLVE

    def test_stable_maps_to_evolve(self):
        """R2_STABLE should map to EVOLVE mode."""
        assert REGIME_TO_SYMBOL_MODE[2] == SymbolMode.EVOLVE

    def test_catastrophe_maps_to_freeze(self):
        """R5_CATASTROPHE should map to FREEZE mode."""
        assert REGIME_TO_SYMBOL_MODE[5] == SymbolMode.FREEZE

    def test_recovery_maps_to_hold(self):
        """R6_RECOVERY should map to HOLD mode."""
        assert REGIME_TO_SYMBOL_MODE[6] == SymbolMode.HOLD


class TestExecutorLatchModeFields:
    """Tests for ExecutorLatch symbol mode fields."""

    def test_default_mode_is_evolve(self):
        """Default symbol mode should be EVOLVE."""
        latch = ExecutorLatch()
        assert latch.symbol_mode == SymbolMode.EVOLVE

    def test_default_dwell_is_zero(self):
        """Default mode dwell should be zero."""
        latch = ExecutorLatch()
        assert latch.mode_dwell_blocks == 0

    def test_set_symbol_mode(self):
        """set_symbol_mode should update mode and reset dwell."""
        latch = ExecutorLatch()
        latch.mode_dwell_blocks = 100  # Simulate some time passing

        latch.set_symbol_mode(SymbolMode.FREEZE)

        assert latch.symbol_mode == SymbolMode.FREEZE
        assert latch.mode_dwell_blocks == 0

    def test_set_same_mode_no_reset(self):
        """Setting same mode should NOT reset dwell (no-op)."""
        latch = ExecutorLatch()
        latch.symbol_mode = SymbolMode.HOLD
        latch.mode_dwell_blocks = 50

        # set_symbol_mode checks if mode changed
        latch.set_symbol_mode(SymbolMode.HOLD)

        # Dwell should NOT be reset since mode didn't change
        assert latch.mode_dwell_blocks == 50

    def test_step_symbol_mode_increments_dwell(self):
        """step_symbol_mode should increment dwell counter."""
        latch = ExecutorLatch()
        assert latch.mode_dwell_blocks == 0

        latch.step_symbol_mode()
        assert latch.mode_dwell_blocks == 1

        latch.step_symbol_mode()
        assert latch.mode_dwell_blocks == 2

    def test_get_evolution_rate(self):
        """get_evolution_rate_q15 should return correct rate for mode."""
        latch = ExecutorLatch()

        latch.symbol_mode = SymbolMode.EVOLVE
        assert latch.get_evolution_rate_q15() == Q15_ONE

        latch.symbol_mode = SymbolMode.FREEZE
        assert latch.get_evolution_rate_q15() == 0


class TestTimbralModeFields:
    """Tests for ExecutorLatch timbral mode fields (Gap G4 preview)."""

    def test_default_timbral_mode(self):
        """Default timbral mode should be T0_PURE (0)."""
        latch = ExecutorLatch()
        assert latch.timbral_mode == 0

    def test_set_timbral_mode(self):
        """set_timbral_mode should set target and ramp."""
        latch = ExecutorLatch()

        latch.set_timbral_mode(3, ramp_blocks=8)

        assert latch.timbral_target == 3
        assert latch.timbral_ramp_blocks == 8

    def test_step_timbral_mode_ramps(self):
        """step_timbral_mode should decrement ramp and snap at end."""
        latch = ExecutorLatch()
        latch.timbral_mode = 0
        latch.timbral_target = 5
        latch.timbral_ramp_blocks = 3

        latch.step_timbral_mode()
        assert latch.timbral_ramp_blocks == 2
        assert latch.timbral_mode == 0  # Not yet at target

        latch.step_timbral_mode()
        assert latch.timbral_ramp_blocks == 1

        latch.step_timbral_mode()
        assert latch.timbral_ramp_blocks == 0
        assert latch.timbral_mode == 5  # Now at target

    def test_is_timbral_ramping(self):
        """is_timbral_ramping should reflect ramp state."""
        latch = ExecutorLatch()
        assert not latch.is_timbral_ramping()

        latch.timbral_ramp_blocks = 5
        assert latch.is_timbral_ramping()


class TestVMStateModePersistence:
    """Tests for VMState snapshot/restore of mode fields."""

    def test_snapshot_includes_mode_fields(self):
        """VMState snapshot should include mode fields."""
        state = VMState.create(voice_id=0)
        state.executor_latch.symbol_mode = SymbolMode.FREEZE
        state.executor_latch.mode_dwell_blocks = 42
        state.executor_latch.timbral_mode = 3

        snapshot = state.snapshot()

        assert 'executor_latch_symbol_mode' in snapshot
        assert snapshot['executor_latch_symbol_mode'] == SymbolMode.FREEZE
        assert snapshot['executor_latch_mode_dwell'] == 42
        assert snapshot['executor_latch_timbral_mode'] == 3

    def test_restore_recovers_mode_fields(self):
        """VMState.restore should recover mode fields."""
        state = VMState.create(voice_id=0)
        state.executor_latch.symbol_mode = SymbolMode.HOLD
        state.executor_latch.mode_dwell_blocks = 100
        state.executor_latch.timbral_mode = 7
        state.executor_latch.timbral_target = 7

        snapshot = state.snapshot()
        restored = VMState.restore(snapshot)

        assert restored.executor_latch.symbol_mode == SymbolMode.HOLD
        assert restored.executor_latch.mode_dwell_blocks == 100
        assert restored.executor_latch.timbral_mode == 7

    def test_reset_for_note_on_resets_latch(self):
        """reset_for_note_on should reset executor latch."""
        state = VMState.create(voice_id=0)
        state.executor_latch.symbol_mode = SymbolMode.FREEZE
        state.executor_latch.mode_dwell_blocks = 999

        state.reset_for_note_on(note=60)

        # ExecutorLatch is replaced entirely
        assert state.executor_latch.symbol_mode == SymbolMode.EVOLVE  # Default
        assert state.executor_latch.mode_dwell_blocks == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
