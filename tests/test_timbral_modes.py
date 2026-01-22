"""
Tests for CMMM Enrichment Gap G4: Timbral Sub-states
====================================================

Tests the TimbralMode enum, REGIME_SYMBOL_OPS table, intent→timbral mapping,
and dynamic operator selection.

Reference: CMMM-ENRICHMENT-DESIGN.md §4.5 (Phase P3)
"""

import pytest

from symcrash_python_ap.primitives import (
    Q15_ONE, Q15_HALF,
    TimbralMode, TimbralModeInfo, TIMBRAL_MODE_TABLE,
    Regime,
    REGIME_SYMBOL_OPS, get_operators_for_state,
    compute_timbral_mode_from_intent,
)
from symcrash_python_ap.vm.state import VMState, ExecutorLatch


class TestTimbralModeEnum:
    """Tests for the TimbralMode enum and related tables."""

    def test_timbral_mode_values(self):
        """TimbralMode enum should have correct values (0-7)."""
        assert TimbralMode.T0_PURE == 0
        assert TimbralMode.T1_BRIGHT == 1
        assert TimbralMode.T2_DARK == 2
        assert TimbralMode.T3_HOLLOW == 3
        assert TimbralMode.T4_NASAL == 4
        assert TimbralMode.T5_BREATHY == 5
        assert TimbralMode.T6_GROWL == 6
        assert TimbralMode.T7_CHAOS == 7

    def test_timbral_mode_table_complete(self):
        """TIMBRAL_MODE_TABLE should have entries for all 8 modes."""
        for mode in TimbralMode:
            assert mode.value in TIMBRAL_MODE_TABLE
            info = TIMBRAL_MODE_TABLE[mode.value]
            assert info.id == mode.value
            assert isinstance(info.name, str)
            assert isinstance(info.description, str)

    def test_timbral_mode_count(self):
        """There should be exactly 8 timbral modes."""
        assert len(TimbralMode) == 8
        assert len(TIMBRAL_MODE_TABLE) == 8


class TestRegimeSymbolOpsTable:
    """Tests for the REGIME_SYMBOL_OPS table."""

    def test_table_not_empty(self):
        """REGIME_SYMBOL_OPS should not be empty."""
        assert len(REGIME_SYMBOL_OPS) > 0

    def test_r2_stable_has_all_timbral_modes(self):
        """R2_STABLE should have entries for all 8 timbral modes."""
        for timbral_mode in range(8):
            key = (Regime.R2_STABLE, timbral_mode)
            assert key in REGIME_SYMBOL_OPS, f"Missing entry for {key}"
            ops = REGIME_SYMBOL_OPS[key]
            assert isinstance(ops, list)
            assert len(ops) > 0

    def test_r5_catastrophe_consistent(self):
        """R5_CATASTROPHE should have same ops for all timbral modes."""
        base_ops = REGIME_SYMBOL_OPS[(Regime.R5_CATASTROPHE, 0)]
        for timbral_mode in range(1, 8):
            key = (Regime.R5_CATASTROPHE, timbral_mode)
            assert key in REGIME_SYMBOL_OPS
            assert REGIME_SYMBOL_OPS[key] == base_ops

    def test_all_entries_have_valid_op_ids(self):
        """All operator IDs in table should be in valid range (0-23)."""
        for key, ops in REGIME_SYMBOL_OPS.items():
            regime, timbral = key
            for op_id in ops:
                assert 0 <= op_id <= 23, f"Invalid op_id {op_id} for {key}"

    def test_r0_idle_minimal_ops(self):
        """R0_IDLE should have minimal operator lists."""
        for key in REGIME_SYMBOL_OPS:
            regime, timbral = key
            if regime == Regime.R0_IDLE:
                ops = REGIME_SYMBOL_OPS[key]
                assert len(ops) <= 2, f"R0_IDLE should have ≤2 operators, got {len(ops)}"


class TestGetOperatorsForState:
    """Tests for get_operators_for_state() function."""

    def test_known_key_returns_correct_ops(self):
        """Known (regime, timbral) key should return correct operators."""
        ops = get_operators_for_state(Regime.R2_STABLE, TimbralMode.T0_PURE)
        assert ops == [1, 3, 12]  # Tilt + IntraMix + GluePair

    def test_unknown_timbral_falls_back_to_t0(self):
        """Unknown timbral mode should fallback to timbral_mode=0."""
        # Create a regime that exists but test with hypothetical mode 99
        ops = get_operators_for_state(Regime.R2_STABLE, 99)
        # Should fallback to (2, 0)
        assert ops == [1, 3, 12]

    def test_unknown_regime_returns_identity(self):
        """Completely unknown regime should return [0] (Identity)."""
        ops = get_operators_for_state(99, 0)
        assert ops == [0]

    def test_r1_attack_t7_chaos(self):
        """R1_ATTACK + T7_CHAOS should return catastrophe-style operators."""
        ops = get_operators_for_state(Regime.R1_ATTACK, TimbralMode.T7_CHAOS)
        # Should be Impulse + CuspInjection + SwallowtailMask
        assert 13 in ops  # Impulse
        assert 21 in ops  # CuspInjection

    def test_r3_air_t5_breathy(self):
        """R3_AIR + T5_BREATHY should emphasize noise."""
        ops = get_operators_for_state(Regime.R3_AIR, TimbralMode.T5_BREATHY)
        # Should have NoiseBand (14) prominently
        assert 14 in ops


class TestComputeTimbralModeFromIntent:
    """Tests for compute_timbral_mode_from_intent() function."""

    def test_empty_intent_returns_pure(self):
        """Empty or short intent should return T0_PURE."""
        assert compute_timbral_mode_from_intent([]) == TimbralMode.T0_PURE
        assert compute_timbral_mode_from_intent([0, 0]) == TimbralMode.T0_PURE

    def test_zero_intent_returns_pure(self):
        """All-zero intent should return T0_PURE."""
        intent = [0, 0, 0, 0, 0, 0, 0, 0]
        assert compute_timbral_mode_from_intent(intent) == TimbralMode.T0_PURE

    def test_high_glitch_returns_chaos(self):
        """Very high I_GLITCH should return T7_CHAOS."""
        intent = [0, 0, 0, 30000, 0, 0, 0, 0]  # High glitch (~91%)
        assert compute_timbral_mode_from_intent(intent) == TimbralMode.T7_CHAOS

    def test_medium_glitch_returns_growl(self):
        """Medium I_GLITCH should return T6_GROWL."""
        intent = [0, 0, 0, 20000, 0, 0, 0, 0]  # ~61% glitch
        assert compute_timbral_mode_from_intent(intent) == TimbralMode.T6_GROWL

    def test_high_air_returns_breathy(self):
        """High I_AIR should return T5_BREATHY."""
        intent = [0, 0, 28000, 0, 0, 0, 0, 0]  # High air (~85%)
        assert compute_timbral_mode_from_intent(intent) == TimbralMode.T5_BREATHY

    def test_high_bow_tone_returns_nasal(self):
        """High I_BOW + I_TONE should return T4_NASAL."""
        intent = [20000, 28000, 0, 0, 0, 0, 0, 0]  # High bow+tone
        assert compute_timbral_mode_from_intent(intent) == TimbralMode.T4_NASAL

    def test_high_tone_low_air_returns_bright(self):
        """High I_TONE + low I_AIR should return T1_BRIGHT."""
        intent = [20000, 0, 5000, 0, 0, 0, 0, 0]  # High tone, low air
        assert compute_timbral_mode_from_intent(intent) == TimbralMode.T1_BRIGHT

    def test_low_tone_med_air_returns_hollow(self):
        """Low I_TONE + medium I_AIR should return T3_HOLLOW."""
        intent = [5000, 0, 10000, 0, 0, 0, 0, 0]  # Low tone, medium air
        assert compute_timbral_mode_from_intent(intent) == TimbralMode.T3_HOLLOW

    def test_medium_tone_bow_returns_dark(self):
        """Medium I_TONE + medium I_BOW + low I_AIR should return T2_DARK."""
        intent = [12000, 20000, 5000, 0, 0, 0, 0, 0]
        assert compute_timbral_mode_from_intent(intent) == TimbralMode.T2_DARK

    def test_priority_glitch_over_air(self):
        """I_GLITCH should take priority over I_AIR."""
        intent = [0, 0, 30000, 28000, 0, 0, 0, 0]  # Both high
        # Glitch priority should win
        assert compute_timbral_mode_from_intent(intent) == TimbralMode.T7_CHAOS


class TestExecutorLatchTimbralMode:
    """Tests for ExecutorLatch timbral mode handling."""

    def test_default_timbral_mode(self):
        """Default timbral mode should be T0_PURE (0)."""
        latch = ExecutorLatch()
        assert latch.timbral_mode == 0

    def test_set_timbral_mode_changes_target(self):
        """set_timbral_mode should set target and ramp."""
        latch = ExecutorLatch()
        latch.set_timbral_mode(TimbralMode.T5_BREATHY, ramp_blocks=8)

        assert latch.timbral_target == TimbralMode.T5_BREATHY
        assert latch.timbral_ramp_blocks == 8
        assert latch.timbral_mode == 0  # Not yet changed

    def test_timbral_ramp_completes(self):
        """Timbral mode should change after ramp completes."""
        latch = ExecutorLatch()
        latch.set_timbral_mode(TimbralMode.T3_HOLLOW, ramp_blocks=3)

        # Step through ramp
        latch.step_timbral_mode()
        assert latch.timbral_ramp_blocks == 2
        assert latch.timbral_mode == 0  # Still old mode

        latch.step_timbral_mode()
        assert latch.timbral_ramp_blocks == 1

        latch.step_timbral_mode()
        assert latch.timbral_ramp_blocks == 0
        assert latch.timbral_mode == TimbralMode.T3_HOLLOW  # Now changed

    def test_is_timbral_ramping(self):
        """is_timbral_ramping should reflect ramp state."""
        latch = ExecutorLatch()
        assert not latch.is_timbral_ramping()

        latch.set_timbral_mode(TimbralMode.T7_CHAOS, ramp_blocks=5)
        assert latch.is_timbral_ramping()

        # Complete the ramp
        for _ in range(5):
            latch.step_timbral_mode()
        assert not latch.is_timbral_ramping()

    def test_set_same_mode_no_ramp(self):
        """Setting same timbral mode should not trigger ramp."""
        latch = ExecutorLatch()
        latch.timbral_mode = TimbralMode.T2_DARK
        latch.timbral_target = TimbralMode.T2_DARK

        latch.set_timbral_mode(TimbralMode.T2_DARK, ramp_blocks=8)

        # No ramp should be triggered
        assert latch.timbral_ramp_blocks == 0


class TestVMStateTimbralMode:
    """Tests for VMState timbral mode handling."""

    def test_vmstate_has_executor_latch(self):
        """VMState should have executor_latch with timbral mode."""
        state = VMState.create(voice_id=0)
        assert hasattr(state, 'executor_latch')
        assert hasattr(state.executor_latch, 'timbral_mode')

    def test_vmstate_reset_resets_timbral(self):
        """reset_for_note_on should reset timbral mode."""
        state = VMState.create(voice_id=0)
        state.executor_latch.timbral_mode = TimbralMode.T6_GROWL
        state.executor_latch.timbral_target = TimbralMode.T7_CHAOS
        state.executor_latch.timbral_ramp_blocks = 10

        state.reset_for_note_on(note=60)

        assert state.executor_latch.timbral_mode == 0  # Default T0_PURE
        assert state.executor_latch.timbral_ramp_blocks == 0

    def test_vmstate_snapshot_includes_timbral(self):
        """VMState snapshot should include timbral mode fields."""
        state = VMState.create(voice_id=0)
        state.executor_latch.timbral_mode = TimbralMode.T4_NASAL
        state.executor_latch.timbral_target = TimbralMode.T4_NASAL

        snapshot = state.snapshot()

        assert 'executor_latch_timbral_mode' in snapshot
        assert snapshot['executor_latch_timbral_mode'] == TimbralMode.T4_NASAL

    def test_vmstate_restore_recovers_timbral(self):
        """VMState.restore should recover timbral mode fields."""
        state = VMState.create(voice_id=0)
        state.executor_latch.timbral_mode = TimbralMode.T1_BRIGHT
        state.executor_latch.timbral_target = TimbralMode.T2_DARK
        state.executor_latch.timbral_ramp_blocks = 5

        snapshot = state.snapshot()
        restored = VMState.restore(snapshot)

        assert restored.executor_latch.timbral_mode == TimbralMode.T1_BRIGHT
        assert restored.executor_latch.timbral_target == TimbralMode.T2_DARK
        assert restored.executor_latch.timbral_ramp_blocks == 5


class TestDynamicOperatorSelection:
    """Tests for dynamic operator selection in executor."""

    def test_get_dynamic_backbone_with_static(self):
        """Without dynamic mode, should return static backbone."""
        from symcrash_python_ap.vm.executor import _get_dynamic_backbone
        from symcrash_python_ap.vm.decode import BackboneEntry

        state = VMState.create(voice_id=0)

        # Mock frame without dynamic mode
        class MockFrame:
            dynamic_timbral_mode = False
            regime_id = 2

        class MockCfg:
            backbone_instrs = [BackboneEntry(op_id=1), BackboneEntry(op_id=3)]

        result = _get_dynamic_backbone(state, MockFrame(), MockCfg())
        assert result == MockCfg.backbone_instrs

    def test_get_dynamic_backbone_with_intent(self):
        """With dynamic mode and intent, should compute timbral mode."""
        from symcrash_python_ap.vm.executor import _get_dynamic_backbone
        from symcrash_python_ap.vm.decode import BackboneEntry

        state = VMState.create(voice_id=0)

        # Mock frame with dynamic mode and high glitch intent
        class MockFrame:
            dynamic_timbral_mode = True
            regime_id = 2  # R2_STABLE
            intent_vector = [0, 0, 0, 30000, 0, 0, 0, 0]  # High glitch
            families = {}

        class MockCfg:
            backbone_instrs = [BackboneEntry(op_id=0)]

        result = _get_dynamic_backbone(state, MockFrame(), MockCfg())

        # Should get operators for (R2_STABLE, T7_CHAOS)
        expected_ops = REGIME_SYMBOL_OPS[(2, 7)]  # [21, 22, 8]
        result_op_ids = [entry.op_id for entry in result]
        assert result_op_ids == expected_ops

    def test_get_dynamic_backbone_fallback(self):
        """With dynamic mode but no table entry, should fallback to static."""
        from symcrash_python_ap.vm.executor import _get_dynamic_backbone
        from symcrash_python_ap.vm.decode import BackboneEntry

        state = VMState.create(voice_id=0)

        # Mock frame with unknown regime
        class MockFrame:
            dynamic_timbral_mode = True
            regime_id = 99  # Unknown regime
            intent_vector = [0, 0, 0, 0, 0, 0, 0, 0]
            families = {}

        class MockCfg:
            backbone_instrs = [BackboneEntry(op_id=7)]

        result = _get_dynamic_backbone(state, MockFrame(), MockCfg())

        # Should fallback to static backbone since (99, 0) not in table
        assert result == MockCfg.backbone_instrs

    def test_dynamic_backbone_updates_latch(self):
        """Dynamic backbone should update executor latch timbral mode."""
        from symcrash_python_ap.vm.executor import _get_dynamic_backbone
        from symcrash_python_ap.vm.decode import BackboneEntry

        state = VMState.create(voice_id=0)
        assert state.executor_latch.timbral_mode == 0

        # Mock frame with high air intent (-> T5_BREATHY)
        class MockFrame:
            dynamic_timbral_mode = True
            regime_id = 2
            intent_vector = [0, 0, 28000, 0, 0, 0, 0, 0]  # High air
            families = {}

        class MockCfg:
            backbone_instrs = []

        _get_dynamic_backbone(state, MockFrame(), MockCfg())

        # Latch should have set timbral target (with ramp)
        assert state.executor_latch.timbral_target == TimbralMode.T5_BREATHY
        assert state.executor_latch.timbral_ramp_blocks > 0


class TestTimbralModeWithRegimes:
    """Tests for timbral modes across different regimes."""

    def test_r1_attack_all_timbral_modes(self):
        """R1_ATTACK should have operators for all timbral modes."""
        for timbral in range(8):
            ops = get_operators_for_state(Regime.R1_ATTACK, timbral)
            assert len(ops) > 0
            # All attack modes should have Impulse (13)
            assert 13 in ops or 21 in ops  # Impulse or CuspInjection

    def test_r6_recovery_all_timbral_modes(self):
        """R6_RECOVERY should have operators for all timbral modes."""
        for timbral in range(8):
            ops = get_operators_for_state(Regime.R6_RECOVERY, timbral)
            assert len(ops) > 0
            # All recovery modes should have GluePair (12) and SheafProject (23)
            assert 12 in ops

    def test_timbral_modes_differ_in_r2_stable(self):
        """Different timbral modes in R2_STABLE should have different operators."""
        ops_pure = get_operators_for_state(Regime.R2_STABLE, TimbralMode.T0_PURE)
        ops_bright = get_operators_for_state(Regime.R2_STABLE, TimbralMode.T1_BRIGHT)
        ops_chaos = get_operators_for_state(Regime.R2_STABLE, TimbralMode.T7_CHAOS)

        # These should be different operator sets
        assert ops_pure != ops_chaos
        assert ops_bright != ops_chaos


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
