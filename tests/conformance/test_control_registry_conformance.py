"""
WSE-STD-CONTROLREGISTRY Conformance Tests
==========================================

Tests for ControlRegistry conformance with frozen v0 specification.

Per WSE-STD-CONTROLREGISTRY v0, the registry MUST:
- Define frozen ControlID enum values (0-6, 9, 16-19)
- Specify value types (Q15U, I16, U16, BOOL, ENUM8)
- Specify slew policies (INSTANT, NBLOCK, ONE_POLE)
- Provide default values and valid ranges
- Support binding targets for each control
"""

import pytest
from dataclasses import dataclass
from typing import Dict, List


# =============================================================================
# FROZEN SPEC VALUES (from WSE-STD-CONTROLREGISTRY v0)
# =============================================================================

FROZEN_CONTROL_IDS = {
    "TIMBRE": 0,
    "TEXTURE": 1,
    "DYNAMICS": 2,
    "MORPH": 3,
    "CHAOS": 4,
    "DETUNE": 5,
    "PITCH_BEND": 6,
    "MASTER_GAIN": 9,
    "ATK_BLOCKS": 16,
    "DCY_BLOCKS": 17,
    "SUS_Q15": 18,
    "REL_BLOCKS": 19,
}

FROZEN_VALUE_TYPES = {
    "Q15U": 0,
    "I16": 1,
    "U16": 2,
    "BOOL": 3,
    "ENUM8": 4,
}

FROZEN_SLEW_POLICIES = {
    "INSTANT": 0,
    "NBLOCK": 1,
    "ONE_POLE": 2,
}


# =============================================================================
# CONFORMANCE: ENUM VALUES
# =============================================================================

class TestControlIDEnum:
    """Verify ControlID enum values match frozen spec."""

    def test_timbre_id(self):
        """TIMBRE MUST be 0."""
        from symcrash_python_ap.control_registry import ControlID
        assert ControlID.TIMBRE == 0

    def test_texture_id(self):
        """TEXTURE MUST be 1."""
        from symcrash_python_ap.control_registry import ControlID
        assert ControlID.TEXTURE == 1

    def test_dynamics_id(self):
        """DYNAMICS MUST be 2."""
        from symcrash_python_ap.control_registry import ControlID
        assert ControlID.DYNAMICS == 2

    def test_morph_id(self):
        """MORPH MUST be 3."""
        from symcrash_python_ap.control_registry import ControlID
        assert ControlID.MORPH == 3

    def test_chaos_id(self):
        """CHAOS MUST be 4."""
        from symcrash_python_ap.control_registry import ControlID
        assert ControlID.CHAOS == 4

    def test_detune_id(self):
        """DETUNE MUST be 5."""
        from symcrash_python_ap.control_registry import ControlID
        assert ControlID.DETUNE == 5

    def test_pitch_bend_id(self):
        """PITCH_BEND MUST be 6."""
        from symcrash_python_ap.control_registry import ControlID
        assert ControlID.PITCH_BEND == 6

    def test_master_gain_id(self):
        """MASTER_GAIN MUST be 9."""
        from symcrash_python_ap.control_registry import ControlID
        assert ControlID.MASTER_GAIN == 9

    def test_atk_blocks_id(self):
        """ATK_BLOCKS MUST be 16."""
        from symcrash_python_ap.control_registry import ControlID
        assert ControlID.ATK_BLOCKS == 16

    def test_dcy_blocks_id(self):
        """DCY_BLOCKS MUST be 17."""
        from symcrash_python_ap.control_registry import ControlID
        assert ControlID.DCY_BLOCKS == 17

    def test_sus_q15_id(self):
        """SUS_Q15 MUST be 18."""
        from symcrash_python_ap.control_registry import ControlID
        assert ControlID.SUS_Q15 == 18

    def test_rel_blocks_id(self):
        """REL_BLOCKS MUST be 19."""
        from symcrash_python_ap.control_registry import ControlID
        assert ControlID.REL_BLOCKS == 19


class TestValueTypeEnum:
    """Verify ValueType enum values match frozen spec."""

    def test_q15u_value(self):
        """Q15U MUST be 0."""
        from symcrash_python_ap.control_registry import ValueType
        assert ValueType.Q15U == 0

    def test_i16_value(self):
        """I16 MUST be 1."""
        from symcrash_python_ap.control_registry import ValueType
        assert ValueType.I16 == 1

    def test_u16_value(self):
        """U16 MUST be 2."""
        from symcrash_python_ap.control_registry import ValueType
        assert ValueType.U16 == 2

    def test_bool_value(self):
        """BOOL MUST be 3."""
        from symcrash_python_ap.control_registry import ValueType
        assert ValueType.BOOL == 3

    def test_enum8_value(self):
        """ENUM8 MUST be 4."""
        from symcrash_python_ap.control_registry import ValueType
        assert ValueType.ENUM8 == 4


class TestSlewPolicyEnum:
    """Verify SlewPolicy enum values match frozen spec."""

    def test_instant_value(self):
        """INSTANT MUST be 0."""
        from symcrash_python_ap.control_registry import SlewPolicy
        assert SlewPolicy.INSTANT == 0

    def test_nblock_value(self):
        """NBLOCK MUST be 1."""
        from symcrash_python_ap.control_registry import SlewPolicy
        assert SlewPolicy.NBLOCK == 1

    def test_one_pole_value(self):
        """ONE_POLE MUST be 2."""
        from symcrash_python_ap.control_registry import SlewPolicy
        assert SlewPolicy.ONE_POLE == 2


# =============================================================================
# CONFORMANCE: CONTROL ENTRIES
# =============================================================================

class TestControlEntries:
    """Verify control entries have required fields."""

    def test_all_frozen_ids_present(self):
        """Registry MUST contain all frozen control IDs."""
        from symcrash_python_ap.control_registry import ControlRegistry, ControlID

        registry = ControlRegistry()
        required_ids = [
            ControlID.TIMBRE, ControlID.TEXTURE, ControlID.DYNAMICS,
            ControlID.MORPH, ControlID.CHAOS, ControlID.DETUNE,
            ControlID.PITCH_BEND, ControlID.MASTER_GAIN,
            ControlID.ATK_BLOCKS, ControlID.DCY_BLOCKS,
            ControlID.SUS_Q15, ControlID.REL_BLOCKS,
        ]

        for control_id in required_ids:
            entry = registry.get_entry(control_id)
            assert entry is not None, f"Missing entry for {control_id.name}"

    def test_entry_has_name(self):
        """Each entry MUST have a name."""
        from symcrash_python_ap.control_registry import ControlRegistry

        registry = ControlRegistry()
        for entry in registry.all_entries():
            assert entry.name, f"Control {entry.control_id} missing name"

    def test_entry_has_binding_target(self):
        """Each entry MUST have a binding target."""
        from symcrash_python_ap.control_registry import ControlRegistry

        registry = ControlRegistry()
        for entry in registry.all_entries():
            assert entry.binding_target, f"Control {entry.name} missing binding_target"


# =============================================================================
# CONFORMANCE: VALUE TYPES AND RANGES
# =============================================================================

class TestValueTypesAndRanges:
    """Verify value types and ranges match spec."""

    def test_timbre_q15u(self):
        """TIMBRE MUST be Q15U type with range 0-32767."""
        from symcrash_python_ap.control_registry import ControlRegistry, ControlID, ValueType

        registry = ControlRegistry()
        entry = registry.get_entry(ControlID.TIMBRE)

        assert entry.value_type == ValueType.Q15U
        assert entry.min_value == 0
        assert entry.max_value == 32767

    def test_detune_i16(self):
        """DETUNE MUST be I16 type with cents range."""
        from symcrash_python_ap.control_registry import ControlRegistry, ControlID, ValueType

        registry = ControlRegistry()
        entry = registry.get_entry(ControlID.DETUNE)

        assert entry.value_type == ValueType.I16
        assert entry.min_value < 0  # Negative cents allowed
        assert entry.max_value > 0  # Positive cents allowed

    def test_atk_blocks_u16(self):
        """ATK_BLOCKS MUST be U16 type with positive range."""
        from symcrash_python_ap.control_registry import ControlRegistry, ControlID, ValueType

        registry = ControlRegistry()
        entry = registry.get_entry(ControlID.ATK_BLOCKS)

        assert entry.value_type == ValueType.U16
        assert entry.min_value >= 1  # At least 1 block
        assert entry.max_value > entry.min_value


# =============================================================================
# CONFORMANCE: DEFAULTS
# =============================================================================

class TestDefaults:
    """Verify default values are sensible."""

    def test_timbre_default_max(self):
        """TIMBRE default MUST be max (full brightness)."""
        from symcrash_python_ap.control_registry import ControlRegistry, ControlID

        registry = ControlRegistry()
        entry = registry.get_entry(ControlID.TIMBRE)

        assert entry.default_value == 32767

    def test_chaos_default_zero(self):
        """CHAOS default MUST be 0 (no chaos)."""
        from symcrash_python_ap.control_registry import ControlRegistry, ControlID

        registry = ControlRegistry()
        entry = registry.get_entry(ControlID.CHAOS)

        assert entry.default_value == 0

    def test_detune_default_zero(self):
        """DETUNE default MUST be 0 (no detuning)."""
        from symcrash_python_ap.control_registry import ControlRegistry, ControlID

        registry = ControlRegistry()
        entry = registry.get_entry(ControlID.DETUNE)

        assert entry.default_value == 0

    def test_sustain_default_reasonable(self):
        """SUS_Q15 default MUST be reasonable (0.7-0.8 of max)."""
        from symcrash_python_ap.control_registry import ControlRegistry, ControlID

        registry = ControlRegistry()
        entry = registry.get_entry(ControlID.SUS_Q15)

        # Default should be around 0.7-0.8 of max (24576 / 32767 ≈ 0.75)
        assert 20000 < entry.default_value < 28000


# =============================================================================
# CONFORMANCE: SLEW POLICIES
# =============================================================================

class TestSlewPolicies:
    """Verify slew policies are appropriate."""

    def test_pitch_bend_instant(self):
        """PITCH_BEND MUST use INSTANT slew."""
        from symcrash_python_ap.control_registry import ControlRegistry, ControlID, SlewPolicy

        registry = ControlRegistry()
        entry = registry.get_entry(ControlID.PITCH_BEND)

        assert entry.slew_policy == SlewPolicy.INSTANT

    def test_timbre_smoothed(self):
        """TIMBRE MUST use ONE_POLE slew."""
        from symcrash_python_ap.control_registry import ControlRegistry, ControlID, SlewPolicy

        registry = ControlRegistry()
        entry = registry.get_entry(ControlID.TIMBRE)

        assert entry.slew_policy == SlewPolicy.ONE_POLE

    def test_morph_nblock(self):
        """MORPH MUST use NBLOCK slew."""
        from symcrash_python_ap.control_registry import ControlRegistry, ControlID, SlewPolicy

        registry = ControlRegistry()
        entry = registry.get_entry(ControlID.MORPH)

        assert entry.slew_policy == SlewPolicy.NBLOCK


# =============================================================================
# CONFORMANCE: CLAMPING
# =============================================================================

class TestClamping:
    """Verify clamping behavior."""

    def test_clamp_above_max(self):
        """Clamp MUST enforce max value."""
        from symcrash_python_ap.control_registry import ControlRegistry, ControlID

        registry = ControlRegistry()
        result = registry.clamp(ControlID.TIMBRE, 50000)

        assert result == 32767

    def test_clamp_below_min(self):
        """Clamp MUST enforce min value."""
        from symcrash_python_ap.control_registry import ControlRegistry, ControlID

        registry = ControlRegistry()
        result = registry.clamp(ControlID.ATK_BLOCKS, 0)

        # ATK_BLOCKS min is 1
        assert result >= 1

    def test_clamp_within_range(self):
        """Clamp MUST preserve values within range."""
        from symcrash_python_ap.control_registry import ControlRegistry, ControlID

        registry = ControlRegistry()
        result = registry.clamp(ControlID.TIMBRE, 16384)

        assert result == 16384

    def test_clamp_signed_negative(self):
        """Clamp MUST handle signed negative values."""
        from symcrash_python_ap.control_registry import ControlRegistry, ControlID

        registry = ControlRegistry()
        result = registry.clamp(ControlID.DETUNE, -100)

        # Should preserve negative value within range
        assert result == -100


# =============================================================================
# CONFORMANCE: BINDING TARGETS
# =============================================================================

class TestBindingTargets:
    """Verify binding targets are valid."""

    def test_timbre_binds_to_family_scale(self):
        """TIMBRE MUST bind to family_scale[COLOR]."""
        from symcrash_python_ap.control_registry import ControlRegistry, ControlID

        registry = ControlRegistry()
        entry = registry.get_entry(ControlID.TIMBRE)

        assert "family_scale" in entry.binding_target

    def test_dynamics_binds_to_gain(self):
        """DYNAMICS MUST bind to gain_scale."""
        from symcrash_python_ap.control_registry import ControlRegistry, ControlID

        registry = ControlRegistry()
        entry = registry.get_entry(ControlID.DYNAMICS)

        assert "gain" in entry.binding_target.lower()

    def test_detune_binds_to_pitch(self):
        """DETUNE MUST bind to pitch_offset."""
        from symcrash_python_ap.control_registry import ControlRegistry, ControlID

        registry = ControlRegistry()
        entry = registry.get_entry(ControlID.DETUNE)

        assert "pitch" in entry.binding_target.lower()


# =============================================================================
# CONFORMANCE: ABI STABILITY
# =============================================================================

class TestABIStability:
    """Verify registry is ABI-stable."""

    def test_registry_count_stable(self):
        """Registry MUST have at least 12 entries (frozen v0)."""
        from symcrash_python_ap.control_registry import ControlRegistry

        registry = ControlRegistry()
        assert len(registry.all_entries()) >= 12

    def test_entry_structure_complete(self):
        """Each entry MUST have all required fields."""
        from symcrash_python_ap.control_registry import ControlRegistry

        registry = ControlRegistry()
        for entry in registry.all_entries():
            # All required fields must exist
            assert hasattr(entry, 'control_id')
            assert hasattr(entry, 'name')
            assert hasattr(entry, 'value_type')
            assert hasattr(entry, 'min_value')
            assert hasattr(entry, 'max_value')
            assert hasattr(entry, 'default_value')
            assert hasattr(entry, 'slew_policy')
            assert hasattr(entry, 'slew_param')
            assert hasattr(entry, 'binding_target')
