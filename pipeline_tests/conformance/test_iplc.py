"""
WSE-STD-IPLC v0 Conformance Tests
==================================

Tests for Instrument Preset Lifecycle standard.

Conformance Checklist (§8):
- [ ] Implements lifecycle states + allowed transitions (§2)
- [ ] Implements capsule bundle layout + required manifest fields (§3)
- [ ] Implements compiler outputs with ABI compat declared (§4)
- [ ] Implements runtime apply semantics exactly (§5)
- [ ] Enforces preset vs snapshot boundary (§5.3)
- [ ] Deterministic application order + seed policy enforcement (§6)
"""

import pytest

from symcrash_python_ap.library import (
    LifecycleState,
    CapsuleRecord,
    ALLOWED_TRANSITIONS,
    TRANSITION_TRIGGERS,
    validate_transition,
    get_transition_trigger,
    transition,
    TransitionError,
    create_draft_record,
    create_compiled_record,
    create_published_record,
    get_allowed_transitions,
    is_terminal_state,
    is_library_state,
    is_runtime_state,
    CompileResult,
    CapsuleCompiler,
    CapsuleV0,
    CapsuleKind,
    create_instrument_capsule,
)
from symcrash_python_ap.control_registry import ControlID
from symcrash_python_ap.voice.voice_engine_registry import VoiceParamID


# =============================================================================
# §2: LIFECYCLE STATES
# =============================================================================

class TestLifecycleStates:
    """Test lifecycle state definitions per §2.1."""

    def test_all_states_defined(self):
        """All 7 lifecycle states are defined."""
        assert LifecycleState.DRAFT == 0
        assert LifecycleState.COMPILED == 1
        assert LifecycleState.PUBLISHED == 2
        assert LifecycleState.APPLIED == 3
        assert LifecycleState.RUNNING == 4
        assert LifecycleState.DEPRECATED == 5
        assert LifecycleState.RETIRED == 6
        assert len(LifecycleState) == 7

    def test_state_names(self):
        """States have correct names."""
        assert LifecycleState.DRAFT.name == "DRAFT"
        assert LifecycleState.COMPILED.name == "COMPILED"
        assert LifecycleState.PUBLISHED.name == "PUBLISHED"
        assert LifecycleState.APPLIED.name == "APPLIED"
        assert LifecycleState.RUNNING.name == "RUNNING"
        assert LifecycleState.DEPRECATED.name == "DEPRECATED"
        assert LifecycleState.RETIRED.name == "RETIRED"


# =============================================================================
# §2.2: TRANSITIONS
# =============================================================================

class TestTransitions:
    """Test allowed transitions per §2.2."""

    def test_draft_to_compiled(self):
        """DRAFT → COMPILED is allowed."""
        assert validate_transition(LifecycleState.DRAFT, LifecycleState.COMPILED)

    def test_compiled_to_published(self):
        """COMPILED → PUBLISHED is allowed."""
        assert validate_transition(LifecycleState.COMPILED, LifecycleState.PUBLISHED)

    def test_published_to_applied(self):
        """PUBLISHED → APPLIED is allowed."""
        assert validate_transition(LifecycleState.PUBLISHED, LifecycleState.APPLIED)

    def test_applied_to_running(self):
        """APPLIED → RUNNING is allowed."""
        assert validate_transition(LifecycleState.APPLIED, LifecycleState.RUNNING)

    def test_published_to_deprecated(self):
        """PUBLISHED → DEPRECATED is allowed."""
        assert validate_transition(LifecycleState.PUBLISHED, LifecycleState.DEPRECATED)

    def test_deprecated_to_retired(self):
        """DEPRECATED → RETIRED is allowed."""
        assert validate_transition(LifecycleState.DEPRECATED, LifecycleState.RETIRED)

    def test_deprecated_to_applied(self):
        """DEPRECATED → APPLIED is allowed (deprecated can still be applied)."""
        assert validate_transition(LifecycleState.DEPRECATED, LifecycleState.APPLIED)

    def test_backward_transitions_not_allowed(self):
        """Backward transitions are not allowed."""
        # COMPILED → DRAFT
        assert not validate_transition(LifecycleState.COMPILED, LifecycleState.DRAFT)
        # PUBLISHED → COMPILED
        assert not validate_transition(LifecycleState.PUBLISHED, LifecycleState.COMPILED)
        # APPLIED → PUBLISHED
        assert not validate_transition(LifecycleState.APPLIED, LifecycleState.PUBLISHED)
        # RUNNING → APPLIED
        assert not validate_transition(LifecycleState.RUNNING, LifecycleState.APPLIED)

    def test_terminal_states(self):
        """RUNNING and RETIRED are terminal states."""
        assert is_terminal_state(LifecycleState.RUNNING)
        assert is_terminal_state(LifecycleState.RETIRED)
        assert not is_terminal_state(LifecycleState.PUBLISHED)

    def test_transition_function_valid(self):
        """transition() works for valid transitions."""
        capsule = create_instrument_capsule(
            capsule_id="test",
            name="Test",
            program_id=1001,
        )
        record = create_draft_record(capsule)
        assert record.state == LifecycleState.DRAFT

        # DRAFT → COMPILED
        record2 = transition(record, LifecycleState.COMPILED)
        assert record2.state == LifecycleState.COMPILED

    def test_transition_function_invalid_raises(self):
        """transition() raises for invalid transitions."""
        capsule = create_instrument_capsule(
            capsule_id="test",
            name="Test",
            program_id=1001,
        )
        record = create_draft_record(capsule)

        # DRAFT → PUBLISHED (skipping COMPILED)
        with pytest.raises(TransitionError):
            transition(record, LifecycleState.PUBLISHED)

    def test_transition_triggers_documented(self):
        """All allowed transitions have documented triggers."""
        for from_state, to_states in ALLOWED_TRANSITIONS.items():
            for to_state in to_states:
                trigger = get_transition_trigger(from_state, to_state)
                assert trigger is not None, f"Missing trigger for {from_state.name} → {to_state.name}"


# =============================================================================
# §2: CAPSULE RECORD
# =============================================================================

class TestCapsuleRecord:
    """Test CapsuleRecord structure."""

    def test_record_has_required_fields(self):
        """CapsuleRecord has required fields per §1.2."""
        capsule = create_instrument_capsule(
            capsule_id="test-001",
            name="Test",
            program_id=1001,
        )
        record = CapsuleRecord(
            capsule=capsule,
            state=LifecycleState.PUBLISHED,
            revision_id=1,
            storage_path="capsules/test-001/1",
        )

        assert record.capsule_id == "test-001"
        assert record.revision_id == 1
        assert record.storage_path == "capsules/test-001/1"
        assert not record.deprecated
        assert not record.retired

    def test_record_is_loadable(self):
        """is_loadable() respects retired flag."""
        capsule = create_instrument_capsule(
            capsule_id="test",
            name="Test",
            program_id=1001,
        )
        record = CapsuleRecord(
            capsule=capsule,
            state=LifecycleState.PUBLISHED,
            revision_id=1,
        )

        assert record.is_loadable()

        # Deprecated is still loadable
        record.deprecated = True
        assert record.is_loadable()

        # Retired is not loadable by default
        record.retired = True
        assert not record.is_loadable()

        # But can be overridden
        assert record.is_loadable(allow_retired=True)

    def test_record_is_recommended(self):
        """is_recommended() checks both flags."""
        capsule = create_instrument_capsule(
            capsule_id="test",
            name="Test",
            program_id=1001,
        )
        record = CapsuleRecord(
            capsule=capsule,
            state=LifecycleState.PUBLISHED,
            revision_id=1,
        )

        assert record.is_recommended()

        record.deprecated = True
        assert not record.is_recommended()


# =============================================================================
# §4: COMPILER CONTRACT
# =============================================================================

class TestCompilerContract:
    """Test compiler contract per §4."""

    def test_compiler_produces_capsule(self):
        """Compiler produces valid CapsuleV0."""
        compiler = CapsuleCompiler()
        result = compiler.compile_instrument(
            program_id=1001,
            name="Glitch Violin",
            author="Test",
        )

        assert result.success
        assert result.capsule is not None
        assert result.capsule.identity.name == "Glitch Violin"
        assert result.capsule.program.program_id == 1001

    def test_compiler_clamps_control_defaults(self):
        """Compiler clamps out-of-range control defaults (§4.1)."""
        compiler = CapsuleCompiler()
        result = compiler.compile_instrument(
            program_id=1001,
            name="Test",
            control_defaults={
                ControlID.TIMBRE: 50000,  # Max is 32767
            },
        )

        assert result.success
        assert ControlID.TIMBRE in result.capsule.defaults.control_defaults
        assert result.capsule.defaults.control_defaults[ControlID.TIMBRE] == 32767
        assert any("clamped" in w for w in result.warnings)

    def test_compiler_clamps_voice_param_defaults(self):
        """Compiler clamps out-of-range voice param defaults (§4.1)."""
        compiler = CapsuleCompiler()
        result = compiler.compile_instrument(
            program_id=1001,
            name="Test",
            voice_param_defaults={
                VoiceParamID.TREE_ID: 100,  # Max is 4
            },
        )

        assert result.success
        assert VoiceParamID.TREE_ID in result.capsule.defaults.voice_param_defaults
        assert result.capsule.defaults.voice_param_defaults[VoiceParamID.TREE_ID] == 4

    def test_compiler_ignores_unknown_ids(self):
        """Compiler ignores unknown control/param IDs."""
        compiler = CapsuleCompiler()
        result = compiler.compile_instrument(
            program_id=1001,
            name="Test",
            control_defaults={255: 100},  # Unknown
            voice_param_defaults={9999: 100},  # Unknown
        )

        assert result.success
        assert 255 not in result.capsule.defaults.control_defaults
        assert 9999 not in result.capsule.defaults.voice_param_defaults
        assert any("Unknown" in w for w in result.warnings)

    def test_compiler_declares_abi(self):
        """Compiler declares ABI versions in manifest (§4.4)."""
        compiler = CapsuleCompiler()
        result = compiler.compile_instrument(
            program_id=1001,
            name="Test",
        )

        assert result.success
        compat = result.capsule.identity.engine_compat
        assert compat.control_registry_abi == 0
        assert compat.voice_registry_abi == 0
        assert compat.voiceframe_abi == 0
        assert compat.sample_rate_hz == 48000
        assert compat.block_size == 64

    def test_compiler_generates_stable_capsule_id(self):
        """Same identity produces same capsule_id (§7)."""
        compiler = CapsuleCompiler()

        result1 = compiler.compile_instrument(
            program_id=1001,
            name="Glitch Violin",
            author="Test",
        )
        result2 = compiler.compile_instrument(
            program_id=1001,
            name="Glitch Violin",
            author="Test",
        )

        assert result1.capsule_id == result2.capsule_id

    def test_compiler_different_identity_different_id(self):
        """Different identity produces different capsule_id (§7)."""
        compiler = CapsuleCompiler()

        result1 = compiler.compile_instrument(
            program_id=1001,
            name="Violin A",
        )
        result2 = compiler.compile_instrument(
            program_id=1001,
            name="Violin B",
        )

        assert result1.capsule_id != result2.capsule_id

    def test_compiler_explicit_capsule_id(self):
        """Explicit capsule_id is used."""
        compiler = CapsuleCompiler()
        result = compiler.compile_instrument(
            program_id=1001,
            name="Test",
            capsule_id="my-custom-id",
        )

        assert result.capsule_id == "my-custom-id"
        assert result.capsule.identity.capsule_id == "my-custom-id"

    def test_compiler_requires_program_id(self):
        """Compiler requires program_id (§4.2)."""
        compiler = CapsuleCompiler()
        result = compiler.compile_instrument(
            program_id=None,
            name="Test",
        )

        assert not result.success
        assert any("program_id" in e for e in result.errors)

    def test_compiler_creates_compiled_record(self):
        """Compiler returns record in COMPILED state."""
        compiler = CapsuleCompiler()
        result = compiler.compile_instrument(
            program_id=1001,
            name="Test",
        )

        assert result.success
        assert result.record is not None
        assert result.record.state == LifecycleState.COMPILED

    def test_compile_synth(self):
        """compile_synth creates SYNTH capsule."""
        compiler = CapsuleCompiler()
        result = compiler.compile_synth(
            program_id=1002,
            name="Warm Pad",
        )

        assert result.success
        assert result.capsule.identity.kind == CapsuleKind.SYNTH


# =============================================================================
# §5: RUNTIME APPLY SEMANTICS
# =============================================================================

class TestRuntimeApplySemantics:
    """Test runtime apply semantics per §5."""

    def test_apply_not_restore_runtime_state(self):
        """Apply MUST NOT restore runtime histories (§5.3)."""
        # CapsuleDefaults only contains default overrides
        # No coeff_state, phase, PRNG state, etc.
        capsule = create_instrument_capsule(
            capsule_id="test",
            name="Test",
            program_id=1001,
        )

        defaults = capsule.defaults
        assert not hasattr(defaults, 'coeff_state')
        assert not hasattr(defaults, 'phase')
        assert not hasattr(defaults, 'prng_state')
        assert not hasattr(defaults, 'envelope_state')


# =============================================================================
# §6: DETERMINISM
# =============================================================================

class TestDeterminism:
    """Test determinism requirements per §6."""

    def test_compiler_deterministic(self):
        """Same inputs produce bit-identical capsule."""
        compiler = CapsuleCompiler()

        result1 = compiler.compile_instrument(
            program_id=1001,
            name="Test",
            control_defaults={ControlID.TIMBRE: 16384},
        )
        result2 = compiler.compile_instrument(
            program_id=1001,
            name="Test",
            control_defaults={ControlID.TIMBRE: 16384},
        )

        # Same capsule_id
        assert result1.capsule_id == result2.capsule_id

        # Same defaults
        assert result1.capsule.defaults.control_defaults == result2.capsule.defaults.control_defaults

    def test_seed_policy_in_capsule(self):
        """Capsule contains seed policy for determinism."""
        compiler = CapsuleCompiler()
        result = compiler.compile_instrument(
            program_id=1001,
            name="Test",
            master_seed=0xDEADBEEF,
        )

        assert result.capsule.seeds.master_seed == 0xDEADBEEF
        assert result.capsule.seeds.derivation_rule_id == 0


# =============================================================================
# STATE QUERY HELPERS
# =============================================================================

class TestStateQueries:
    """Test state query helpers."""

    def test_get_allowed_transitions(self):
        """get_allowed_transitions returns correct states."""
        allowed = get_allowed_transitions(LifecycleState.PUBLISHED)
        assert LifecycleState.APPLIED in allowed
        assert LifecycleState.DEPRECATED in allowed
        assert LifecycleState.COMPILED not in allowed

    def test_is_library_state(self):
        """is_library_state identifies library states."""
        assert is_library_state(LifecycleState.PUBLISHED)
        assert is_library_state(LifecycleState.DEPRECATED)
        assert is_library_state(LifecycleState.RETIRED)
        assert not is_library_state(LifecycleState.APPLIED)
        assert not is_library_state(LifecycleState.RUNNING)

    def test_is_runtime_state(self):
        """is_runtime_state identifies runtime states."""
        assert is_runtime_state(LifecycleState.APPLIED)
        assert is_runtime_state(LifecycleState.RUNNING)
        assert not is_runtime_state(LifecycleState.PUBLISHED)
