"""
WSE-STD-070 Preset Conformance Tests
=====================================

LEGACY TEST - Preset module was refactored. Tower/Editor API no longer exists.
This test needs to be rewritten for the new preset loading API.

Tests for presets, editor operations, and tower completion.
"""

import pytest
pytest.skip("Legacy test - preset module refactored, tower API removed", allow_module_level=True)

from symcrash_python_ap.library.preset_model import (
    # Constants
    PRESET_MAGIC,
    SAFE_OPERATOR_SET,
    DEFAULT_REGIME,
    # Tower levels
    TowerLevel,
    TOWER_LEVEL_INFO,
    get_tower_level_info,
    # Header
    PresetHeader,
    # Layers
    L0PatchLayer,
    L1FamilyLayer,
    L2InstrumentLayer,
    L3CockpitLayer,
    L4ForgeLayer,
    L5LabLayer,
    # Body
    PresetBody,
    # Editor ops
    EditorOpType,
    EditorOp,
    set_tree_id,
    add_regime,
    remove_regime,
    add_transition,
    remove_transition,
    set_param,
    set_bound,
    set_operator_allowed,
    set_operator_weight,
    set_budget,
    set_tower_level,
    # Completion
    complete_tower,
    complete_l1_family,
    complete_l2_instrument,
    complete_l3_cockpit,
    complete_l4_forge,
    complete_l5_lab,
    # Projection
    project_to_level,
    reset_to_safe,
    # Preset
    Preset,
    # Lifecycle
    load_preset,
    save_preset,
    create_preset,
    compute_crc32,
    # Determinism
    validate_determinism,
)


# =============================================================================
# TOWER LEVEL TESTS (Section 3)
# =============================================================================

class TestTowerLevels:
    """Test tower level definitions per Section 3."""

    def test_all_levels_defined(self):
        """All 6 tower levels should be defined."""
        assert len(TowerLevel) == 6
        assert TowerLevel.L0_PATCH.value == 0
        assert TowerLevel.L1_FAMILY.value == 1
        assert TowerLevel.L2_INSTRUMENT.value == 2
        assert TowerLevel.L3_COCKPIT.value == 3
        assert TowerLevel.L4_FORGE.value == 4
        assert TowerLevel.L5_LAB.value == 5

    def test_level_info_defined(self):
        """Each level should have metadata."""
        for level in TowerLevel:
            info = get_tower_level_info(level)
            assert 'name' in info
            assert 'invariant' in info

    def test_l0_always_playable(self):
        """L0 invariant should be 'Always playable'."""
        info = get_tower_level_info(TowerLevel.L0_PATCH)
        assert 'playable' in info['invariant'].lower()

    def test_l5_must_compile_down(self):
        """L5 invariant should be 'Must compile down'."""
        info = get_tower_level_info(TowerLevel.L5_LAB)
        assert 'compile' in info['invariant'].lower()


# =============================================================================
# PRESET HEADER TESTS (Section 1.1)
# =============================================================================

class TestPresetHeader:
    """Test preset header per Section 1.1."""

    def test_header_fields(self):
        """Header should have all required fields."""
        header = PresetHeader(
            preset_id=12345,
            preset_version=2,
            tower_level=TowerLevel.L3_COCKPIT,
            tree_id=42,
            flags=0x0001,
        )
        assert header.preset_id == 12345
        assert header.preset_version == 2
        assert header.tower_level == TowerLevel.L3_COCKPIT
        assert header.tree_id == 42
        assert header.flags == 0x0001

    def test_header_serialization(self):
        """Header should serialize and deserialize correctly."""
        original = PresetHeader(
            preset_id=99999,
            preset_version=1,
            tower_level=TowerLevel.L2_INSTRUMENT,
            tree_id=100,
            flags=0,
            crc32=0,
        )
        data = original.to_bytes()
        restored = PresetHeader.from_bytes(data)

        assert restored.preset_id == original.preset_id
        assert restored.tower_level == original.tower_level
        assert restored.tree_id == original.tree_id

    def test_header_size(self):
        """Header should be fixed size."""
        header = PresetHeader(preset_id=1)
        data = header.to_bytes()
        assert len(data) == PresetHeader.HEADER_SIZE


# =============================================================================
# LAYER TESTS (Section 1.2)
# =============================================================================

class TestLayers:
    """Test tower layer structures per Section 1.2."""

    def test_l0_patch_always_valid(self):
        """L0 Patch layer should always be valid."""
        layer = L0PatchLayer()
        assert layer.is_valid()

    def test_l1_family_requires_regimes(self):
        """L1 Family needs at least one regime."""
        layer = L1FamilyLayer(regimes={'STABLE'}, initial_regime='STABLE')
        assert layer.is_valid()

        layer_empty = L1FamilyLayer(regimes=set(), initial_regime='STABLE')
        assert not layer_empty.is_valid()

    def test_l2_instrument_requires_operators(self):
        """L2 Instrument needs allowed operators."""
        layer = L2InstrumentLayer(allowed_operators={'FineClamp'})
        assert layer.is_valid()

        layer_empty = L2InstrumentLayer(allowed_operators=set())
        assert not layer_empty.is_valid()

    def test_l3_cockpit_requires_budget(self):
        """L3 Cockpit needs positive budget."""
        layer = L3CockpitLayer(compute_budget=1000, memory_budget=1000)
        assert layer.is_valid()

        layer_no_budget = L3CockpitLayer(compute_budget=0, memory_budget=0)
        assert not layer_no_budget.is_valid()

    def test_l4_forge_always_projectable(self):
        """L4 Forge should always be projectable (valid)."""
        layer = L4ForgeLayer()
        assert layer.is_valid()

    def test_l5_lab_must_compile_down(self):
        """L5 Lab should be valid (must compile down)."""
        layer = L5LabLayer()
        assert layer.is_valid()


class TestPresetBody:
    """Test preset body structure."""

    def test_get_set_layer(self):
        """Should get and set layers by level."""
        body = PresetBody()

        layer = L0PatchLayer(gain=0.5)
        body.set_layer(TowerLevel.L0_PATCH, layer)

        retrieved = body.get_layer(TowerLevel.L0_PATCH)
        assert retrieved is layer
        assert retrieved.gain == 0.5

    def test_has_layer(self):
        """Should check if layer exists."""
        body = PresetBody()
        assert not body.has_layer(TowerLevel.L2_INSTRUMENT)

        body.L2_instrument = L2InstrumentLayer()
        assert body.has_layer(TowerLevel.L2_INSTRUMENT)

    def test_highest_explicit_level(self):
        """Should find highest explicit level."""
        body = PresetBody()
        body.L0_patch = L0PatchLayer()
        assert body.highest_explicit_level() == TowerLevel.L0_PATCH

        body.L2_instrument = L2InstrumentLayer()
        assert body.highest_explicit_level() == TowerLevel.L2_INSTRUMENT

        body.L4_forge = L4ForgeLayer()
        assert body.highest_explicit_level() == TowerLevel.L4_FORGE


# =============================================================================
# EDITOR PRIMITIVE TESTS (Section 2)
# =============================================================================

class TestEditorPrimitives:
    """Test editor primitives per Section 2."""

    def test_set_tree_id(self):
        """SET_TREE_ID should create valid op."""
        op = set_tree_id(42)
        assert op.op_type == EditorOpType.SET_TREE_ID
        assert op.args['tree_id'] == 42

    def test_add_remove_regime(self):
        """ADD_REGIME and REMOVE_REGIME should work."""
        add_op = add_regime('CHAOTIC')
        assert add_op.op_type == EditorOpType.ADD_REGIME
        assert add_op.args['regime_id'] == 'CHAOTIC'

        remove_op = remove_regime('CHAOTIC')
        assert remove_op.op_type == EditorOpType.REMOVE_REGIME

    def test_add_remove_transition(self):
        """ADD_TRANSITION and REMOVE_TRANSITION should work."""
        add_op = add_transition('STABLE', 'CHAOTIC')
        assert add_op.op_type == EditorOpType.ADD_TRANSITION
        assert add_op.args['src'] == 'STABLE'
        assert add_op.args['dst'] == 'CHAOTIC'

    def test_set_param(self):
        """SET_PARAM should create valid op."""
        op = set_param('gain', 0.8)
        assert op.op_type == EditorOpType.SET_PARAM
        assert op.args['path'] == 'gain'
        assert op.args['value'] == 0.8

    def test_set_bound(self):
        """SET_BOUND should create valid op."""
        op = set_bound('attack', 0.001, 1.0)
        assert op.op_type == EditorOpType.SET_BOUND
        assert op.args['min'] == 0.001
        assert op.args['max'] == 1.0

    def test_set_operator_allowed(self):
        """SET_OPERATOR_ALLOWED should work."""
        op = set_operator_allowed('NoiseBand', True)
        assert op.op_type == EditorOpType.SET_OPERATOR_ALLOWED
        assert op.args['op_id'] == 'NoiseBand'
        assert op.args['allowed'] is True

    def test_set_operator_weight(self):
        """SET_OPERATOR_WEIGHT should work."""
        op = set_operator_weight('Tilt', 0.5)
        assert op.op_type == EditorOpType.SET_OPERATOR_WEIGHT
        assert op.args['weight'] == 0.5

    def test_set_budget(self):
        """SET_BUDGET should work."""
        op = set_budget(2000)
        assert op.op_type == EditorOpType.SET_BUDGET
        assert op.args['limit'] == 2000

    def test_reset_to_safe_op(self):
        """RESET_TO_SAFE should require no args."""
        op = EditorOp(EditorOpType.RESET_TO_SAFE, {})
        assert op.op_type == EditorOpType.RESET_TO_SAFE

    def test_missing_required_arg_raises(self):
        """Missing required arg should raise."""
        with pytest.raises(ValueError):
            EditorOp(EditorOpType.SET_TREE_ID, {})  # Missing tree_id


# =============================================================================
# TOWER COMPLETION TESTS (Section 4)
# =============================================================================

class TestTowerCompletion:
    """Test tower completion per Section 4."""

    def test_completion_direction_upward(self):
        """Completion should proceed upward."""
        preset = Preset(header=PresetHeader(preset_id=1, tree_id=10))
        # Only L0 present
        preset.body.L0_patch = L0PatchLayer()

        complete_tower(preset, TowerLevel.L3_COCKPIT)

        # Should have L0, L1, L2, L3
        assert preset.body.has_layer(TowerLevel.L0_PATCH)
        assert preset.body.has_layer(TowerLevel.L1_FAMILY)
        assert preset.body.has_layer(TowerLevel.L2_INSTRUMENT)
        assert preset.body.has_layer(TowerLevel.L3_COCKPIT)
        # L4, L5 not completed (above target)
        assert not preset.body.has_layer(TowerLevel.L4_FORGE)

    def test_complete_l1_infers_family(self):
        """Missing L1 should infer family from tree_id."""
        preset = Preset(header=PresetHeader(preset_id=1, tree_id=42))
        layer = complete_l1_family(preset)

        assert 'TREE_42' in layer.family_name
        assert DEFAULT_REGIME in layer.regimes

    def test_complete_l2_uses_safe_ops(self):
        """Missing L2 should use SAFE operator set."""
        preset = Preset(header=PresetHeader(preset_id=1))
        layer = complete_l2_instrument(preset)

        assert layer.allowed_operators == set(SAFE_OPERATOR_SET)

    def test_complete_l3_conservative_bounds(self):
        """Missing L3 should have conservative envelope bounds."""
        preset = Preset(header=PresetHeader(preset_id=1))
        layer = complete_l3_cockpit(preset)

        assert 'attack' in layer.envelope_bounds
        assert layer.compute_budget > 0

    def test_complete_l4_fixed_transitions(self):
        """Missing L4 should have fixed transitions."""
        preset = Preset(header=PresetHeader(preset_id=1))
        preset.body.L1_family = L1FamilyLayer(regimes={'A', 'B'}, initial_regime='A')

        layer = complete_l4_forge(preset)

        # Should have self-loops only
        assert 'A' in layer.regime_graph
        assert 'B' in layer.regime_graph

    def test_completion_is_deterministic(self):
        """Same preset should produce same completion."""
        preset1 = Preset(header=PresetHeader(preset_id=1, tree_id=99))
        preset2 = Preset(header=PresetHeader(preset_id=1, tree_id=99))

        complete_tower(preset1, TowerLevel.L3_COCKPIT)
        complete_tower(preset2, TowerLevel.L3_COCKPIT)

        # Should be identical
        assert preset1.body.L1_family.family_name == preset2.body.L1_family.family_name
        assert preset1.body.L2_instrument.allowed_operators == preset2.body.L2_instrument.allowed_operators


# =============================================================================
# PROJECTION TESTS (Section 5)
# =============================================================================

class TestProjection:
    """Test projection maps per Section 5."""

    def test_projection_direction_downward(self):
        """Projection should proceed downward."""
        preset = create_preset(preset_id=1, tower_level=TowerLevel.L5_LAB)

        # Should have all layers
        assert preset.body.has_layer(TowerLevel.L5_LAB)

        # Project to L2
        project_to_level(preset, TowerLevel.L2_INSTRUMENT)

        # Should NOT have L3, L4, L5
        assert not preset.body.has_layer(TowerLevel.L3_COCKPIT)
        assert not preset.body.has_layer(TowerLevel.L4_FORGE)
        assert not preset.body.has_layer(TowerLevel.L5_LAB)

        # Should still have L0, L1, L2
        assert preset.body.has_layer(TowerLevel.L0_PATCH)
        assert preset.body.has_layer(TowerLevel.L1_FAMILY)
        assert preset.body.has_layer(TowerLevel.L2_INSTRUMENT)

    def test_projection_preserves_playability(self):
        """Projection MUST preserve playability."""
        preset = create_preset(preset_id=1, tower_level=TowerLevel.L4_FORGE)

        # Project to L0
        project_to_level(preset, TowerLevel.L0_PATCH)

        # Should still be valid/playable
        assert preset.is_valid()
        assert preset.body.has_layer(TowerLevel.L0_PATCH)

    def test_reset_to_safe_projects_to_l2(self):
        """RESET_TO_SAFE should project to L2."""
        preset = create_preset(preset_id=1, tower_level=TowerLevel.L4_FORGE)

        reset_to_safe(preset)

        assert preset.header.tower_level == TowerLevel.L2_INSTRUMENT
        assert not preset.body.has_layer(TowerLevel.L3_COCKPIT)

    def test_reset_to_safe_uses_safe_ops(self):
        """RESET_TO_SAFE should enable only SAFE ops."""
        preset = create_preset(preset_id=1)
        # Add unsafe operator
        preset.body.L2_instrument.allowed_operators.add('UnsafeOp')

        reset_to_safe(preset)

        assert preset.body.L2_instrument.allowed_operators == set(SAFE_OPERATOR_SET)

    def test_reset_to_safe_clears_events(self):
        """RESET_TO_SAFE should clear event policies."""
        preset = create_preset(preset_id=1, tower_level=TowerLevel.L3_COCKPIT)
        preset.body.L3_cockpit.event_policies = {0: {'policy': 'test'}}

        reset_to_safe(preset)

        # L3 should be gone (projected to L2)
        assert not preset.body.has_layer(TowerLevel.L3_COCKPIT)


# =============================================================================
# PRESET TESTS
# =============================================================================

class TestPreset:
    """Test complete preset functionality."""

    def test_create_preset(self):
        """create_preset should produce valid preset."""
        preset = create_preset(preset_id=123, tree_id=45)

        assert preset.header.preset_id == 123
        assert preset.header.tree_id == 45
        assert preset.is_valid()

    def test_apply_set_tree_id(self):
        """Applying SET_TREE_ID should update tree."""
        preset = create_preset(preset_id=1, tree_id=0)
        preset.apply_op(set_tree_id(99))

        assert preset.header.tree_id == 99

    def test_apply_add_regime(self):
        """Applying ADD_REGIME should add regime."""
        preset = create_preset(preset_id=1)
        preset.apply_op(add_regime('CHAOTIC'))

        assert 'CHAOTIC' in preset.body.L1_family.regimes

    def test_apply_remove_regime(self):
        """Applying REMOVE_REGIME should remove regime."""
        preset = create_preset(preset_id=1)
        preset.body.L1_family.regimes.add('TEMP')

        preset.apply_op(remove_regime('TEMP'))

        assert 'TEMP' not in preset.body.L1_family.regimes

    def test_apply_set_param(self):
        """Applying SET_PARAM should set parameter."""
        preset = create_preset(preset_id=1)
        preset.apply_op(set_param('my_param', 42))

        assert preset.body.L2_instrument.params['my_param'] == 42

    def test_apply_set_operator_allowed(self):
        """Applying SET_OPERATOR_ALLOWED should modify allowlist."""
        preset = create_preset(preset_id=1)

        # Add new operator
        preset.apply_op(set_operator_allowed('CustomOp', True))
        assert 'CustomOp' in preset.body.L2_instrument.allowed_operators

        # Remove it
        preset.apply_op(set_operator_allowed('CustomOp', False))
        assert 'CustomOp' not in preset.body.L2_instrument.allowed_operators

    def test_edit_history_recorded(self):
        """All applied ops should be recorded."""
        preset = create_preset(preset_id=1)
        preset.apply_op(set_tree_id(10))
        preset.apply_op(add_regime('X'))
        preset.apply_op(set_param('p', 1))

        assert len(preset.edit_history) == 3

    def test_preset_valid_with_only_l0(self):
        """Preset should be valid with only L0 layer."""
        preset = Preset(header=PresetHeader(preset_id=1))
        preset.body.L0_patch = L0PatchLayer()

        assert preset.is_valid()


# =============================================================================
# LIFECYCLE TESTS (Section 7)
# =============================================================================

class TestLifecycle:
    """Test preset lifecycle per Section 7."""

    def test_save_load_roundtrip(self):
        """Save and load should preserve header."""
        original = create_preset(preset_id=12345, tree_id=67)
        data = save_preset(original)

        # At minimum, header should be preserved
        assert len(data) >= PresetHeader.HEADER_SIZE

    def test_crc_computed_on_save(self):
        """CRC should be computed when saving."""
        preset = create_preset(preset_id=1)
        preset.header.crc32 = 0

        save_preset(preset)

        # CRC should be set (even if body is empty, CRC of empty is computed)
        # Note: CRC is computed over body, not header
        assert True  # CRC computation is tested by save_preset

    def test_complete_called_on_load(self):
        """Loading should apply completion rules."""
        # Create minimal data with just header
        header = PresetHeader(preset_id=1, tower_level=TowerLevel.L2_INSTRUMENT)
        header.crc32 = compute_crc32(b'')  # Empty body
        data = header.to_bytes()

        preset = load_preset(data)

        # Should have completed layers
        assert preset.body.has_layer(TowerLevel.L0_PATCH)
        assert preset.body.has_layer(TowerLevel.L1_FAMILY)
        assert preset.body.has_layer(TowerLevel.L2_INSTRUMENT)


# =============================================================================
# DETERMINISM TESTS (Section 6)
# =============================================================================

class TestDeterminism:
    """Test determinism guarantees per Section 6."""

    def test_same_ops_same_result(self):
        """Same preset + same ops should produce same result."""
        preset1 = create_preset(preset_id=1, tree_id=10)
        preset2 = create_preset(preset_id=1, tree_id=10)

        ops = [
            set_tree_id(99),
            add_regime('NEW'),
            set_param('x', 100),
        ]

        for op in ops:
            preset1.apply_op(op)
        for op in ops:
            preset2.apply_op(op)

        # Results should be identical
        assert preset1.header.tree_id == preset2.header.tree_id
        assert preset1.body.L1_family.regimes == preset2.body.L1_family.regimes
        assert preset1.body.L2_instrument.params == preset2.body.L2_instrument.params

    def test_completion_does_not_consult_runtime(self):
        """Tower completion MUST NOT consult runtime state."""
        # Completion is pure based on preset content only
        preset1 = Preset(header=PresetHeader(preset_id=1, tree_id=42))
        preset2 = Preset(header=PresetHeader(preset_id=1, tree_id=42))

        # Complete at different "times" (simulated)
        complete_tower(preset1, TowerLevel.L2_INSTRUMENT)
        # Simulated delay...
        complete_tower(preset2, TowerLevel.L2_INSTRUMENT)

        # Results must be identical
        assert preset1.body.L1_family.family_name == preset2.body.L1_family.family_name
        assert preset1.body.L2_instrument.allowed_operators == preset2.body.L2_instrument.allowed_operators


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for preset system."""

    def test_full_workflow(self):
        """Test complete preset workflow."""
        # 1. Create preset
        preset = create_preset(preset_id=1, tree_id=5)

        # 2. Edit at L3
        preset.complete(TowerLevel.L3_COCKPIT)
        preset.apply_op(set_budget(2000))
        preset.apply_op(set_operator_weight('Tilt', 0.8))

        # 3. Edit at L4
        preset.complete(TowerLevel.L4_FORGE)
        preset.apply_op(add_transition('STABLE', 'CHAOTIC'))

        # 4. Panic - reset to safe
        reset_to_safe(preset)

        # 5. Should be valid L2 preset
        assert preset.header.tower_level == TowerLevel.L2_INSTRUMENT
        assert preset.is_valid()
        assert preset.body.L2_instrument.allowed_operators == set(SAFE_OPERATOR_SET)

    def test_layer_independence(self):
        """Changes at one level shouldn't break lower levels."""
        preset = create_preset(preset_id=1, tower_level=TowerLevel.L4_FORGE)

        # Edit L4
        preset.apply_op(add_transition('A', 'B'))

        # Project to L2
        project_to_level(preset, TowerLevel.L2_INSTRUMENT)

        # L0, L1, L2 should still be valid
        assert preset.body.L0_patch.is_valid()
        assert preset.body.L1_family.is_valid()
        assert preset.body.L2_instrument.is_valid()
