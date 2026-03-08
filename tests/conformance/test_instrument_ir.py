"""
InstrumentIR v0 Conformance Tests
==================================

Tests for the layered InstrumentIR schema and loader.
"""

import pytest
import tempfile
import os
import json

from symcrash_python_ap.instrument_ir import (
    # Enums
    NodeType,
    WaveformType,
    NoiseColor,
    FilterMode,
    RegimeKind,
    TransitionType,
    OverlayKind,
    EventLane,
    # L0 types
    Macro,
    ADSR,
    NodeParams,
    L0Node,
    L0Edge,
    L0Graph,
    # L1 types
    GestureSource,
    MacroAffect,
    MacroSemantic,
    L1Lift,
    # L2 types
    Hysteresis,
    RegimeNode,
    RegimeTransition,
    L2Regimes,
    # L3 types
    StateVariable,
    CmmmState,
    CmmmInputs,
    CmmmOutputs,
    TreeTarget,
    EventPayloadTemplate,
    EventPolicy,
    EmitTemplate,
    TransitionChoice,
    TransitionRule,
    SuppressionRule,
    OverlayAdd,
    Overlay,
    L3Cmmm,
    # Compile types
    CompletionPolicy,
    DegradeGroup,
    DegradeProfile,
    EventsPolicy,
    FrameEmitterPolicy,
    CompileConfig,
    # Top-level types
    VoiceModel,
    InstrumentMeta,
    IOConfig,
    Layers,
    InstrumentIR,
    # Functions
    load_instrument,
    load_instrument_from_dict,
    save_instrument,
    complete_instrument,
    create_default_l1_lift,
    create_default_l2_regimes,
    create_default_emit_templates,
)


# =============================================================================
# ENUM TESTS
# =============================================================================

class TestEnums:
    """Test enum definitions."""

    def test_node_types(self):
        """Node types should match spec."""
        assert NodeType.OSC.value == "Osc"
        assert NodeType.NOISE.value == "Noise"
        assert NodeType.FILTER.value == "Filter"

    def test_event_lanes(self):
        """Event lanes should match spec assignments."""
        assert EventLane.EVT_OP_PULSE == 0
        assert EventLane.EVT_OP_TOGGLE == 1
        assert EventLane.EVT_TREE_MORPH == 2
        assert EventLane.EVT_K_MODE == 3
        assert EventLane.EVT_INJECT_NOTE == 4
        assert EventLane.EVT_MATERIAL_HIT == 5
        assert EventLane.EVT_CLOUD_PUSH == 6
        assert EventLane.EVT_RESERVED == 7

    def test_regime_kinds(self):
        """Regime kinds should match spec."""
        assert RegimeKind.IDLE.value == "Idle"
        assert RegimeKind.TRANSIENT.value == "Transient"
        assert RegimeKind.STABLE_TONE.value == "StableTone"
        assert RegimeKind.GLITCH_CHATTER.value == "GlitchChatter"
        assert RegimeKind.RECOVERY.value == "Recovery"


# =============================================================================
# L0 GRAPH TESTS
# =============================================================================

class TestL0Graph:
    """Test L0 DSP graph structures."""

    def test_macro_defaults(self):
        """Macro should have sensible defaults."""
        macro = Macro(id="M_GAIN", name="Gain")
        assert macro.range == (0.0, 1.0)
        assert macro.default == 0.5

    def test_adsr_defaults(self):
        """ADSR should have standard defaults."""
        adsr = ADSR()
        assert adsr.a_ms == 8.0
        assert adsr.d_ms == 80.0
        assert adsr.s == 0.7
        assert adsr.r_ms == 180.0

    def test_l0_node_creation(self):
        """L0 nodes should be creatable."""
        node = L0Node(
            id="N_OSC",
            type="Osc",
            params=NodeParams(waveform="saw"),
        )
        assert node.id == "N_OSC"
        assert node.params.waveform == "saw"

    def test_l0_edge_creation(self):
        """L0 edges should be creatable."""
        edge = L0Edge(src="N_OSC.out", dst="N_MIX.in1")
        assert edge.src == "N_OSC.out"
        assert edge.dst == "N_MIX.in1"

    def test_l0_graph_creation(self):
        """L0 graph should be creatable with nodes and edges."""
        graph = L0Graph(
            macros=[Macro(id="M_GAIN", name="Gain")],
            nodes=[L0Node(id="N_OSC", type="Osc")],
            edges=[L0Edge(src="N_OSC.out", dst="N_AMP.in")],
        )
        assert len(graph.macros) == 1
        assert len(graph.nodes) == 1
        assert len(graph.edges) == 1


# =============================================================================
# L1 LIFT TESTS
# =============================================================================

class TestL1Lift:
    """Test L1 gesture basis and macro semantics."""

    def test_gesture_source(self):
        """Gesture source should store range and source."""
        gesture = GestureSource(range=(0, 1), source="note_on_off")
        assert gesture.range == (0, 1)
        assert gesture.source == "note_on_off"

    def test_macro_affect(self):
        """Macro affect should store target and map."""
        affect = MacroAffect(target="frame.gain_q15", map="adsr_vca")
        assert affect.target == "frame.gain_q15"
        assert affect.map == "adsr_vca"

    def test_l1_lift_creation(self):
        """L1 lift should be creatable with gesture basis."""
        lift = L1Lift(
            gesture_basis={
                'Gate': GestureSource(range=(0, 1), source='note_on_off'),
            },
            macro_semantics={
                'M_GAIN': MacroSemantic(
                    affects=[MacroAffect(target="frame.gain_q15", map="adsr_vca")]
                ),
            },
        )
        assert 'Gate' in lift.gesture_basis
        assert 'M_GAIN' in lift.macro_semantics


# =============================================================================
# L2 REGIMES TESTS
# =============================================================================

class TestL2Regimes:
    """Test L2 regime graph structures."""

    def test_hysteresis(self):
        """Hysteresis should store enter/exit thresholds."""
        hyst = Hysteresis(enter=0.6, exit=0.45)
        assert hyst.enter == 0.6
        assert hyst.exit == 0.45

    def test_regime_node(self):
        """Regime node should store kind and template."""
        node = RegimeNode(
            id="R0_IDLE",
            kind="Idle",
            emit_template="T_IDLE",
        )
        assert node.id == "R0_IDLE"
        assert node.kind == "Idle"
        assert node.emit_template == "T_IDLE"

    def test_regime_transition(self):
        """Regime transition should store from/to/when."""
        trans = RegimeTransition(
            from_regime="R0_IDLE",
            to="R1_ATTACK",
            when="Gate rises",
        )
        assert trans.from_regime == "R0_IDLE"
        assert trans.to == "R1_ATTACK"
        assert trans.when == "Gate rises"

    def test_l2_regimes_creation(self):
        """L2 regimes should be creatable with nodes and transitions."""
        regimes = L2Regimes(
            regime_nodes=[
                RegimeNode(id="R0_IDLE", kind="Idle", emit_template="T_IDLE"),
                RegimeNode(id="R1_ATTACK", kind="Transient", emit_template="T_ATTACK"),
            ],
            transitions=[
                RegimeTransition(from_regime="R0_IDLE", to="R1_ATTACK", when="Gate rises"),
            ],
        )
        assert len(regimes.regime_nodes) == 2
        assert len(regimes.transitions) == 1


# =============================================================================
# L3 CMMM TESTS
# =============================================================================

class TestL3Cmmm:
    """Test L3-L5 CMMM structures."""

    def test_tree_target_defaults(self):
        """Tree target should have sensible defaults."""
        target = TreeTarget()
        assert target.A == "harmonic"
        assert target.B == "harmonic"
        assert target.beta == 0.0

    def test_emit_template(self):
        """Emit template should store families and ops."""
        template = EmitTemplate(
            families={'tone': 0.5, 'settling': 0.7},
            ops_enable=['Tilt', 'FineClamp'],
            trees=TreeTarget(A='harmonic', B='texture', beta=0.5),
        )
        assert template.families['tone'] == 0.5
        assert 'Tilt' in template.ops_enable
        assert template.trees.beta == 0.5

    def test_event_policy(self):
        """Event policy should store lane and condition."""
        policy = EventPolicy(
            lane='EVT_OP_PULSE',
            condition='Gate rises',
            payload_template=EventPayloadTemplate(variant=0, op='Impulse'),
            strength='0.8',
        )
        assert policy.lane == 'EVT_OP_PULSE'
        assert policy.condition == 'Gate rises'
        assert policy.payload_template.op == 'Impulse'

    def test_transition_rule_deterministic(self):
        """Deterministic transition rule should have no choices."""
        rule = TransitionRule(
            name='attack',
            from_regime='R0_IDLE',
            to='R1_ATTACK',
            type='deterministic',
            when='Gate rises',
        )
        assert rule.type == 'deterministic'
        assert len(rule.choices) == 0

    def test_transition_rule_weighted(self):
        """Weighted transition rule should have choices."""
        rule = TransitionRule(
            name='glitch_enter',
            from_regime='R2_STABLE',
            to='R4_GLITCH',
            type='weighted',
            when='Chaos > 0.6',
            choices=[
                TransitionChoice(to='R4_GLITCH', weight=0.8),
                TransitionChoice(to='R3_AIR', weight=0.2),
            ],
            prng_draw_site='CMMM.edge_choice',
        )
        assert rule.type == 'weighted'
        assert len(rule.choices) == 2
        assert rule.prng_draw_site == 'CMMM.edge_choice'

    def test_overlay(self):
        """Overlay should store enable condition and additions."""
        overlay = Overlay(
            id='O_CHAOS',
            enable_when='Chaos > 0.5',
            kind='conditional',
            add=OverlayAdd(
                families={'warp': 0.5},
                events=[],
            ),
        )
        assert overlay.id == 'O_CHAOS'
        assert overlay.add.families['warp'] == 0.5


# =============================================================================
# TOP-LEVEL IR TESTS
# =============================================================================

class TestInstrumentIR:
    """Test top-level InstrumentIR structure."""

    def test_default_creation(self):
        """InstrumentIR should be creatable with defaults."""
        ir = InstrumentIR()
        assert ir.meta.voice_model.max_voices == 128
        assert ir.io.sample_rate_hz == 48000
        assert ir.io.block_size == 64

    def test_validation_tile_size_power_of_two(self):
        """Validation should fail for non-power-of-two tile_size."""
        ir = InstrumentIR()
        ir.meta.voice_model.tile_size = 15  # Not power of two

        valid, err = ir.validate()
        assert not valid
        assert "tile_size" in err

    def test_validation_valid_default(self):
        """Default IR should validate."""
        ir = InstrumentIR()
        valid, err = ir.validate()
        assert valid, f"Validation failed: {err}"

    def test_validation_lane_bounds(self):
        """Lane IDs should be in 0-7 range."""
        ir = InstrumentIR()
        ir.compile.frame_emitter_policy.lanes = {'EVT_BAD': 10}

        valid, err = ir.validate()
        assert not valid
        assert "Lane" in err


# =============================================================================
# LOADER TESTS
# =============================================================================

class TestLoader:
    """Test loading InstrumentIR from dict/files."""

    def test_load_from_dict_minimal(self):
        """Load minimal IR from dict."""
        data = {
            'meta': {
                'name': 'Test Instrument',
                'version': 1,
            },
        }
        ir = load_instrument_from_dict(data)
        assert ir.meta.name == 'Test Instrument'
        assert ir.meta.version == 1

    def test_load_from_dict_with_wrapper(self):
        """Load IR with instrument_ir_v0 wrapper."""
        data = {
            'instrument_ir_v0': {
                'meta': {
                    'name': 'Wrapped Instrument',
                },
            },
        }
        ir = load_instrument_from_dict(data)
        assert ir.meta.name == 'Wrapped Instrument'

    def test_load_from_dict_l0_graph(self):
        """Load IR with L0 graph."""
        data = {
            'layers': {
                'L0_graph': {
                    'macros': [
                        {'id': 'M_GAIN', 'name': 'Gain', 'range': [0, 1], 'default': 0.8},
                    ],
                    'nodes': [
                        {'id': 'N_OSC', 'type': 'Osc', 'params': {'waveform': 'saw'}},
                    ],
                    'edges': [
                        {'src': 'N_OSC.out', 'dst': 'N_AMP.in'},
                    ],
                },
            },
        }
        ir = load_instrument_from_dict(data)
        assert len(ir.layers.L0_graph.macros) == 1
        assert ir.layers.L0_graph.macros[0].default == 0.8
        assert ir.layers.L0_graph.nodes[0].params.waveform == 'saw'

    def test_load_from_dict_l2_regimes(self):
        """Load IR with L2 regimes."""
        data = {
            'layers': {
                'L2_regimes': {
                    'regime_nodes': [
                        {'id': 'R0_IDLE', 'kind': 'Idle', 'emit_template': 'T_IDLE'},
                        {'id': 'R1_ATTACK', 'kind': 'Transient', 'emit_template': 'T_ATTACK', 'dwell_ms': 20},
                    ],
                    'transitions': [
                        {'from': 'R0_IDLE', 'to': 'R1_ATTACK', 'when': 'Gate rises'},
                    ],
                },
            },
        }
        ir = load_instrument_from_dict(data)
        assert len(ir.layers.L2_regimes.regime_nodes) == 2
        assert ir.layers.L2_regimes.regime_nodes[1].dwell_ms == 20

    def test_load_from_dict_emit_templates(self):
        """Load IR with emit templates (legacy format translates to sound-eng names)."""
        data = {
            'layers': {
                'L3_cmmm': {
                    'emit_templates': {
                        'T_IDLE': {
                            'families': {'CALM': 1.0},
                            'ops_enable': ['FineClamp', 'NormalizeScale'],
                            'trees': {'A': 'T1', 'B': 'T1', 'beta': 0.0},
                        },
                    },
                },
            },
        }
        ir = load_instrument_from_dict(data)
        assert 'T_IDLE' in ir.layers.L3_cmmm.emit_templates
        template = ir.layers.L3_cmmm.emit_templates['T_IDLE']
        assert template.families['settling'] == 1.0
        assert 'FineClamp' in template.ops_enable
        assert template.trees.A == 'harmonic'

    @pytest.mark.skip(reason="yaml module not installed")
    def test_load_from_yaml_file(self):
        """Load IR from YAML file."""
        yaml_content = """
instrument_ir_v0:
  meta:
    name: YAML Instrument
    version: 2
    seed_u32: 123456
  io:
    sample_rate_hz: 44100
    block_size: 128
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            ir = load_instrument(path)
            assert ir.meta.name == 'YAML Instrument'
            assert ir.meta.version == 2
            assert ir.meta.seed_u32 == 123456
            assert ir.io.sample_rate_hz == 44100
            assert ir.io.block_size == 128
        finally:
            os.unlink(path)

    def test_load_from_json_file(self):
        """Load IR from JSON file."""
        json_content = {
            'instrument_ir_v0': {
                'meta': {
                    'name': 'JSON Instrument',
                    'version': 3,
                },
            },
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_content, f)
            path = f.name

        try:
            ir = load_instrument(path)
            assert ir.meta.name == 'JSON Instrument'
            assert ir.meta.version == 3
        finally:
            os.unlink(path)


# =============================================================================
# COMPLETION TESTS
# =============================================================================

class TestCompletion:
    """Test IR completion for partial specifications."""

    def test_create_default_l1_lift(self):
        """Default L1 lift should have standard gestures."""
        lift = create_default_l1_lift()
        assert 'Gate' in lift.gesture_basis
        assert 'Pressure' in lift.gesture_basis
        assert 'Chaos' in lift.gesture_basis
        assert lift.gesture_basis['Gate'].source == 'note_on_off'

    def test_create_default_l2_regimes(self):
        """Default L2 regimes should have skeleton regimes."""
        regimes = create_default_l2_regimes()
        regime_ids = [r.id for r in regimes.regime_nodes]
        assert 'R0_IDLE' in regime_ids
        assert 'R1_ATTACK' in regime_ids
        assert 'R2_STABLE' in regime_ids
        assert 'R6_RECOVERY' in regime_ids

    def test_create_default_emit_templates(self):
        """Default emit templates should have standard templates."""
        templates = create_default_emit_templates()
        assert 'T_IDLE' in templates
        assert 'T_ATTACK' in templates
        assert 'T_STABLE' in templates
        assert 'T_RECOVERY' in templates

    def test_complete_empty_ir(self):
        """Empty IR should be completed with defaults."""
        ir = InstrumentIR()
        ir = complete_instrument(ir)

        assert len(ir.layers.L1_lift.gesture_basis) > 0
        assert len(ir.layers.L2_regimes.regime_nodes) > 0
        assert len(ir.layers.L3_cmmm.emit_templates) > 0

    def test_complete_adds_tail_ops(self):
        """Completion should add tail ops to all templates."""
        ir = InstrumentIR()
        ir.compile.completion_policy.always_add_tail_ops = ['FineClamp', 'NormalizeScale', 'SheafProject']

        ir = complete_instrument(ir)

        for template in ir.layers.L3_cmmm.emit_templates.values():
            assert 'FineClamp' in template.ops_enable
            assert 'NormalizeScale' in template.ops_enable
            assert 'SheafProject' in template.ops_enable

    def test_complete_preserves_existing(self):
        """Completion should preserve existing data."""
        ir = InstrumentIR()
        ir.layers.L1_lift.gesture_basis = {
            'CustomGesture': GestureSource(range=(0, 10), source='custom'),
        }

        ir = complete_instrument(ir)

        # Should preserve existing gesture
        assert 'CustomGesture' in ir.layers.L1_lift.gesture_basis
        assert ir.layers.L1_lift.gesture_basis['CustomGesture'].source == 'custom'


# =============================================================================
# COMPILE CONFIG TESTS
# =============================================================================

class TestCompileConfig:
    """Test compile configuration structures."""

    def test_frame_emitter_policy_defaults(self):
        """Frame emitter policy should have sensible defaults."""
        policy = FrameEmitterPolicy()
        assert policy.voiceframe_version == 1
        assert policy.endianness == "big"
        assert policy.lane_limit == 8

    def test_degrade_profile_structure(self):
        """Degrade profile should have group structure."""
        profile = DegradeProfile(
            top_group=DegradeGroup(
                includes=['overlays', 'catastrophe_ops'],
                priority='drop first',
            ),
            mid_group=DegradeGroup(
                includes=['diffuse', 'lowrank'],
                priority='drop second',
            ),
            never_drop=DegradeGroup(
                includes=['tail_ops', 'gain_envelope'],
            ),
        )
        assert 'overlays' in profile.top_group.includes
        assert 'diffuse' in profile.mid_group.includes
        assert 'tail_ops' in profile.never_drop.includes

    def test_map_library(self):
        """Map library should store mapping functions."""
        config = CompileConfig(
            map_library={
                'map(Tilt)': 'clamp_bipolar(gesture.Tilt)',
                'map(Chaos)': 'clamp01(gesture.Chaos)',
            },
        )
        assert 'map(Tilt)' in config.map_library
        assert 'clamp_bipolar' in config.map_library['map(Tilt)']
