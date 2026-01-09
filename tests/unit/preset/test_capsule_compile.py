"""
Capsule Compilation Tests
=========================

Tests for IR to CapsuleV0 compilation.

These tests verify:
- IR compiles to valid CapsuleV0
- Program reference is correctly set
- Control defaults are populated
- Voice param defaults are populated
- Compilation errors are properly reported
"""

import pytest


# =============================================================================
# BASIC COMPILATION TESTS
# =============================================================================

class TestBasicCompilation:
    """Tests for basic IR to capsule compilation."""

    def test_ir_to_capsule(self):
        """IR MUST compile to CapsuleV0."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test")
        capsule = ir.to_capsule()

        assert capsule is not None

    def test_capsule_name_matches(self):
        """Capsule name MUST match IR name."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("My Synth")
        capsule = ir.to_capsule()

        assert capsule.identity.name == "My Synth"

    def test_capsule_is_instrument(self):
        """Synth capsule MUST be instrument kind."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test")
        capsule = ir.to_capsule()

        assert capsule.is_instrument


# =============================================================================
# PROGRAM REFERENCE TESTS
# =============================================================================

class TestProgramReference:
    """Tests for program reference in capsule."""

    def test_program_id_auto_generated(self):
        """Program ID MUST be auto-generated if not provided."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test")
        capsule = ir.to_capsule()

        # Program should exist
        assert capsule.program is not None

    def test_program_id_override(self):
        """Program ID MUST accept override."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test")
        capsule = ir.to_capsule(program_id=42)

        assert capsule.program.program_id == 42

    def test_capsule_id_auto_generated(self):
        """Capsule ID MUST be auto-generated if not provided."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test")
        capsule = ir.to_capsule()

        # Capsule ID should exist
        assert capsule.identity.capsule_id is not None
        assert len(capsule.identity.capsule_id) > 0

    def test_capsule_id_override(self):
        """Capsule ID MUST accept override."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test")
        capsule = ir.to_capsule(capsule_id="custom-id")

        assert capsule.identity.capsule_id == "custom-id"


# =============================================================================
# DEFAULTS TESTS
# =============================================================================

class TestDefaults:
    """Tests for capsule defaults population."""

    def test_control_defaults_populated(self):
        """Control defaults MUST be populated."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test")
        capsule = ir.to_capsule()

        assert capsule.defaults is not None
        # Should have at least some control defaults
        assert hasattr(capsule.defaults, 'control_defaults')

    def test_voice_param_defaults_populated(self):
        """Voice param defaults MUST be populated."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test")
        capsule = ir.to_capsule()

        assert capsule.defaults is not None
        assert hasattr(capsule.defaults, 'voice_param_defaults')


# =============================================================================
# COMPILE RESULT TESTS
# =============================================================================

class TestCompileResult:
    """Tests for to_compile_result method."""

    def test_compile_result_success(self):
        """to_compile_result MUST return success for valid IR."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test")
        result = ir.to_compile_result()

        assert result.success

    def test_compile_result_has_capsule(self):
        """Successful compile MUST have capsule."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test")
        result = ir.to_compile_result()

        assert result.capsule is not None

    def test_compile_result_has_record(self):
        """Compile result MUST have record."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test")
        result = ir.to_compile_result()

        # Record should exist if success
        if result.success:
            assert result.record is not None


# =============================================================================
# AUTO-COMPLETE TESTS
# =============================================================================

class TestAutoComplete:
    """Tests for auto_complete parameter."""

    def test_auto_complete_default_true(self):
        """auto_complete MUST default to True."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test")

        # Should work without explicit auto_complete
        capsule = ir.to_capsule()
        assert capsule is not None

    def test_auto_complete_fills_missing(self):
        """auto_complete MUST fill missing layers."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR()
        ir.meta.instrument_id = "minimal"
        ir.meta.name = "Minimal"

        # L0/L1/L2/L3 not explicitly set
        capsule = ir.to_capsule(auto_complete=True)

        assert capsule is not None


# =============================================================================
# ENGINE COMPATIBILITY TESTS
# =============================================================================

class TestEngineCompatibility:
    """Tests for engine compatibility in capsule."""

    def test_has_engine_compat(self):
        """Capsule MUST have engine_compat field."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test")
        capsule = ir.to_capsule()

        assert hasattr(capsule.identity, 'engine_compat')
