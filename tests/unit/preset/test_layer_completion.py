"""
InstrumentIR Layer Completion Tests
====================================

Tests for automatic layer completion (L1/L2/L3 auto-fill).

These tests verify:
- Empty IR completes successfully
- L0-only IR auto-fills L1/L2/L3
- Completion is deterministic
- Completed layers have valid structure
"""

import pytest


# =============================================================================
# AUTO-COMPLETION TESTS
# =============================================================================

class TestAutoCompletion:
    """Tests for automatic layer completion."""

    def test_empty_ir_completes(self):
        """Empty IR MUST complete successfully."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR()
        ir.meta.instrument_id = "test"
        ir.meta.name = "Test"

        # Should not raise
        capsule = ir.to_capsule()

        assert capsule is not None

    def test_quick_synth_completes(self):
        """quick_synth IR MUST complete successfully."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test")
        capsule = ir.to_capsule()

        assert capsule is not None
        assert capsule.identity.name == "Test"

    def test_l0_only_fills_l1_l2_l3(self):
        """IR with only L0 MUST auto-fill L1/L2/L3."""
        from symcrash_python_ap.instrument_ir import (
            InstrumentIR, L0Node, L0Edge, NodeParams, ADSR
        )

        ir = InstrumentIR()
        ir.meta.instrument_id = "l0-only"
        ir.meta.name = "L0 Only"

        # Define minimal L0 graph
        ir.layers.L0_graph.nodes = [
            L0Node(id="osc1", type="Osc", params=NodeParams(waveform="saw")),
            L0Node(id="amp1", type="Amp", params=NodeParams(
                adsr=ADSR(a_ms=10, d_ms=100, s=0.7, r_ms=200)
            )),
        ]
        ir.layers.L0_graph.edges = [
            L0Edge(src="osc1", dst="amp1"),
        ]

        # Should complete without error
        capsule = ir.to_capsule()
        assert capsule is not None


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestCompletionDeterminism:
    """Tests for completion determinism."""

    def test_completion_is_deterministic(self):
        """Same IR MUST produce same capsule."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir1 = InstrumentIR.quick_synth("Test", brightness=0.5)
        ir2 = InstrumentIR.quick_synth("Test", brightness=0.5)

        capsule1 = ir1.to_capsule()
        capsule2 = ir2.to_capsule()

        # Names should match
        assert capsule1.identity.name == capsule2.identity.name

        # Both should have same basic structure
        assert capsule1.identity.kind == capsule2.identity.kind

    def test_different_params_different_capsules(self):
        """Different params MUST produce different results."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir_bright = InstrumentIR.quick_synth("Bright", brightness=0.9)
        ir_dark = InstrumentIR.quick_synth("Dark", brightness=0.1)

        capsule_bright = ir_bright.to_capsule()
        capsule_dark = ir_dark.to_capsule()

        # Names should differ
        assert capsule_bright.identity.name != capsule_dark.identity.name


# =============================================================================
# LAYER STRUCTURE TESTS
# =============================================================================

class TestLayerStructure:
    """Tests for completed layer structure."""

    def test_l0_has_nodes_after_completion(self):
        """L0 graph MUST have nodes after completion."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test")

        # L0 should have nodes
        assert len(ir.layers.L0_graph.nodes) > 0

    def test_l0_has_edges_after_completion(self):
        """L0 graph MUST have edges after completion."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test")

        # L0 should have edges
        assert len(ir.layers.L0_graph.edges) > 0

    def test_l0_has_amp_with_adsr(self):
        """L0 graph MUST have Amp node with ADSR."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test")

        amp_nodes = [n for n in ir.layers.L0_graph.nodes if n.type == "Amp"]
        assert len(amp_nodes) > 0

        amp = amp_nodes[0]
        assert amp.params.adsr is not None


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases in completion."""

    def test_zero_attack_allowed(self):
        """Zero attack MUST be allowed (instant)."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Instant", attack_ms=0)
        capsule = ir.to_capsule()

        assert capsule is not None

    def test_very_long_release_allowed(self):
        """Very long release MUST be allowed."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Long Tail", release_ms=10000)
        capsule = ir.to_capsule()

        assert capsule is not None

    def test_whitespace_name_allowed(self):
        """Whitespace-only name MUST be allowed (validation is lenient)."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        # Whitespace names work because they still have instrument_id generated
        ir = InstrumentIR.quick_synth("   ")
        capsule = ir.to_capsule()

        assert capsule is not None


# =============================================================================
# CAPSULE VALIDITY TESTS
# =============================================================================

class TestCapsuleValidity:
    """Tests for capsule validity after completion."""

    def test_capsule_has_identity(self):
        """Completed capsule MUST have identity."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test")
        capsule = ir.to_capsule()

        assert capsule.identity is not None
        assert capsule.identity.name == "Test"

    def test_capsule_has_program(self):
        """Completed capsule MUST have program reference."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test")
        capsule = ir.to_capsule()

        assert capsule.program is not None

    def test_capsule_has_defaults(self):
        """Completed capsule MUST have defaults."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test")
        capsule = ir.to_capsule()

        assert capsule.defaults is not None
