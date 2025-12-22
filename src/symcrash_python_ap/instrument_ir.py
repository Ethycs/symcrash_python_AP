"""
InstrumentIR v0 (Layered Tower IR)
==================================

A layered, editor-friendly IR that starts at L0 conventional DSP graph,
lifts to L2 regimes, and culminates in a CMMM (L3-L5) that emits
VoiceFramePacket v0b.

Layers:
- L0: Conventional DSP graph (OSC -> MIX -> FILTER -> AMP -> FX)
- L1: Gesture basis + macro meaning (control manifold)
- L2: Regime graph with typed modes and transitions
- L3-L5: CMMM (Controlled Markov Mealy Machine) with overlays

The IR can be stored as YAML or JSON and supports completion rules
for partial specifications.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum, IntEnum
import json

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None


# =============================================================================
# ENUMS
# =============================================================================

class NodeType(Enum):
    """L0 DSP node types."""
    OSC = "Osc"
    NOISE = "Noise"
    MIXER = "Mixer"
    FILTER = "Filter"
    AMP = "Amp"
    FX_SEND = "FXSend"


class WaveformType(Enum):
    """Oscillator waveform types."""
    SINE = "sine"
    SAW = "saw"
    PULSE = "pulse"
    TRI = "tri"


class NoiseColor(Enum):
    """Noise generator color types."""
    WHITE = "white"
    PINK = "pink"
    BROWN = "brown"


class FilterMode(Enum):
    """Filter mode types."""
    LP = "lp"
    HP = "hp"
    BP = "bp"
    NOTCH = "notch"


class RegimeKind(Enum):
    """Regime type classification."""
    IDLE = "Idle"
    TRANSIENT = "Transient"
    STABLE_TONE = "StableTone"
    NOISE_TEXTURE = "NoiseTexture"
    GLITCH_CHATTER = "GlitchChatter"
    RECOVERY = "Recovery"


class TransitionType(Enum):
    """Transition rule types."""
    DETERMINISTIC = "deterministic"
    WEIGHTED = "weighted"


class OverlayKind(Enum):
    """Overlay behavior types."""
    CONDITIONAL = "conditional"
    ALWAYS = "always"


class EventLane(IntEnum):
    """Fixed event lane assignments."""
    EVT_OP_PULSE = 0
    EVT_OP_TOGGLE = 1
    EVT_TREE_MORPH = 2
    EVT_K_MODE = 3
    EVT_INJECT_NOTE = 4
    EVT_MATERIAL_HIT = 5
    EVT_CLOUD_PUSH = 6
    EVT_RESERVED = 7


# =============================================================================
# L0 GRAPH IR (Conventional DSP)
# =============================================================================

@dataclass
class Macro:
    """User-facing macro parameter."""
    id: str
    name: str
    range: Tuple[float, float] = (0.0, 1.0)
    default: float = 0.5


@dataclass
class ADSR:
    """Envelope parameters."""
    a_ms: float = 8.0
    d_ms: float = 80.0
    s: float = 0.7
    r_ms: float = 180.0


@dataclass
class NodeParams:
    """Parameters for DSP nodes."""
    # Oscillator
    waveform: Optional[str] = None
    detune_cents: float = 0.0
    pw: float = 0.5  # Pulse width

    # Noise
    color: Optional[str] = None

    # Filter
    mode: Optional[str] = None

    # Amp
    adsr: Optional[ADSR] = None

    # FX Send
    send_a: float = 0.0
    send_b: float = 0.0


@dataclass
class L0Node:
    """L0 DSP graph node."""
    id: str
    type: str
    params: NodeParams = field(default_factory=NodeParams)


@dataclass
class L0Edge:
    """L0 DSP graph edge."""
    src: str
    dst: str


@dataclass
class L0Graph:
    """L0 conventional DSP graph."""
    macros: List[Macro] = field(default_factory=list)
    nodes: List[L0Node] = field(default_factory=list)
    edges: List[L0Edge] = field(default_factory=list)


# =============================================================================
# L1 LIFT IR (Gestures + Macro Meaning)
# =============================================================================

@dataclass
class GestureSource:
    """Gesture basis definition."""
    range: Tuple[float, float]
    source: str


@dataclass
class MacroAffect:
    """How a macro affects a target."""
    target: str
    map: str


@dataclass
class MacroSemantic:
    """Macro semantics: how L0 macros map to internal primitives."""
    affects: List[MacroAffect] = field(default_factory=list)


@dataclass
class L1Lift:
    """L1 gesture basis and macro semantics."""
    gesture_basis: Dict[str, GestureSource] = field(default_factory=dict)
    macro_semantics: Dict[str, MacroSemantic] = field(default_factory=dict)


# =============================================================================
# L2 REGIMES IR (Regime Graph)
# =============================================================================

@dataclass
class Hysteresis:
    """Hysteresis thresholds for transitions."""
    enter: float
    exit: float


@dataclass
class RegimeNode:
    """L2 regime graph node."""
    id: str
    kind: str
    emit_template: str
    dwell_ms: Optional[int] = None


@dataclass
class RegimeTransition:
    """L2 regime transition."""
    from_regime: str  # 'from' is reserved in Python
    to: str
    when: str
    hysteresis: Optional[Hysteresis] = None


@dataclass
class L2Regimes:
    """L2 regime graph."""
    regime_nodes: List[RegimeNode] = field(default_factory=list)
    transitions: List[RegimeTransition] = field(default_factory=list)


# =============================================================================
# L3-L5 CMMM IR (Controlled Markov Mealy Machine)
# =============================================================================

@dataclass
class StateVariable:
    """CMMM state variable definition."""
    type: str
    init: Any


@dataclass
class CmmmState:
    """CMMM persistent per-voice state."""
    variables: Dict[str, StateVariable] = field(default_factory=dict)


@dataclass
class CmmmInputs:
    """CMMM per-block inputs."""
    gesture: List[str] = field(default_factory=list)
    note: Dict[str, str] = field(default_factory=dict)
    block_index: str = "u32"


@dataclass
class CmmmOutputs:
    """CMMM per-block outputs."""
    frame_fields: List[str] = field(default_factory=list)


@dataclass
class TreeTarget:
    """Tree blend target."""
    A: str = "T1"
    B: str = "T1"
    beta: Union[float, str] = 0.0


@dataclass
class EventPayloadTemplate:
    """Event payload template."""
    variant: int = 0
    op: Optional[str] = None
    target_kind: Optional[str] = None
    target_id: int = 0
    pattern_id: int = 0
    p0: int = 0
    p1: int = 0
    p2: int = 0
    # For EVT_INJECT_NOTE
    wave_type: Optional[str] = None
    partials_N: Optional[int] = None
    pull_q8: Optional[int] = None


@dataclass
class EventPolicy:
    """Event emission policy."""
    lane: str
    condition: str
    payload_template: EventPayloadTemplate = field(default_factory=EventPayloadTemplate)
    strength: str = "0.0"


@dataclass
class EmitTemplate:
    """Emission template for a regime."""
    families: Dict[str, Union[float, str]] = field(default_factory=dict)
    ops_enable: List[str] = field(default_factory=list)
    trees: TreeTarget = field(default_factory=TreeTarget)
    events: List[EventPolicy] = field(default_factory=list)


@dataclass
class TransitionChoice:
    """Weighted transition choice."""
    to: str
    weight: float


@dataclass
class TransitionRule:
    """CMMM transition rule."""
    name: str
    from_regime: str
    to: str
    type: str  # "deterministic" or "weighted"
    when: str
    choices: List[TransitionChoice] = field(default_factory=list)
    prng_draw_site: Optional[str] = None


@dataclass
class SuppressionRule:
    """Overlay suppression rule."""
    when: str
    suppress: List[str] = field(default_factory=list)


@dataclass
class OverlayAdd:
    """What an overlay adds to emissions."""
    families: Dict[str, Union[float, str]] = field(default_factory=dict)
    events: List[EventPolicy] = field(default_factory=list)


@dataclass
class Overlay:
    """CMMM overlay (conditional superposition)."""
    id: str
    enable_when: str
    kind: str = "conditional"
    add: OverlayAdd = field(default_factory=OverlayAdd)
    suppression_rules: List[SuppressionRule] = field(default_factory=list)


@dataclass
class L3Cmmm:
    """L3-L5 Controlled Markov Mealy Machine."""
    state: CmmmState = field(default_factory=CmmmState)
    inputs: CmmmInputs = field(default_factory=CmmmInputs)
    outputs: CmmmOutputs = field(default_factory=CmmmOutputs)
    emit_templates: Dict[str, EmitTemplate] = field(default_factory=dict)
    transition_kernel: List[TransitionRule] = field(default_factory=list)
    overlays: List[Overlay] = field(default_factory=list)


# =============================================================================
# COMPILE SECTION
# =============================================================================

@dataclass
class CompletionPolicy:
    """Policy for completing missing layers."""
    if_missing_L1_lift: str = ""
    if_missing_L2_regimes: str = ""
    if_missing_L3_cmmm: str = ""
    always_add_tail_ops: List[str] = field(default_factory=list)
    default_tree: TreeTarget = field(default_factory=TreeTarget)


@dataclass
class DegradeGroup:
    """Degrade group for scheduler integration."""
    includes: List[str] = field(default_factory=list)
    priority: str = ""


@dataclass
class DegradeProfile:
    """How features sacrifice under load."""
    top_group: DegradeGroup = field(default_factory=DegradeGroup)
    mid_group: DegradeGroup = field(default_factory=DegradeGroup)
    never_drop: DegradeGroup = field(default_factory=DegradeGroup)


@dataclass
class EventsPolicy:
    """Event selection and merge policy."""
    choose_order: str = "lane_id_ascending_then_template_order"
    merge_policy: str = "at most one payload per lane per block; strongest wins"


@dataclass
class FrameEmitterPolicy:
    """VoiceFrame emission policy."""
    voiceframe_version: int = 1
    endianness: str = "big"
    lane_limit: int = 8
    lanes: Dict[str, int] = field(default_factory=dict)
    events: EventsPolicy = field(default_factory=EventsPolicy)
    reject_on: List[str] = field(default_factory=list)


@dataclass
class CompileConfig:
    """Compile section configuration."""
    map_library: Dict[str, str] = field(default_factory=dict)
    completion_policy: CompletionPolicy = field(default_factory=CompletionPolicy)
    degrade_profile: DegradeProfile = field(default_factory=DegradeProfile)
    frame_emitter_policy: FrameEmitterPolicy = field(default_factory=FrameEmitterPolicy)


# =============================================================================
# TOP-LEVEL IR
# =============================================================================

@dataclass
class VoiceModel:
    """Voice model configuration."""
    max_voices: int = 128
    tile_size: int = 16  # Must be power of two


@dataclass
class InstrumentMeta:
    """Instrument metadata."""
    instrument_id: str = ""
    name: str = ""
    version: int = 0
    author: str = ""
    seed_u32: int = 0
    voice_model: VoiceModel = field(default_factory=VoiceModel)


@dataclass
class IOConfig:
    """I/O configuration."""
    sample_rate_hz: int = 48000
    block_size: int = 64


@dataclass
class Layers:
    """All IR layers."""
    L0_graph: L0Graph = field(default_factory=L0Graph)
    L1_lift: L1Lift = field(default_factory=L1Lift)
    L2_regimes: L2Regimes = field(default_factory=L2Regimes)
    L3_cmmm: L3Cmmm = field(default_factory=L3Cmmm)


@dataclass
class InstrumentIR:
    """
    InstrumentIR v0 - Complete instrument specification.

    A layered IR that compiles down to VoiceFramePacket v0b emissions.
    """
    meta: InstrumentMeta = field(default_factory=InstrumentMeta)
    io: IOConfig = field(default_factory=IOConfig)
    layers: Layers = field(default_factory=Layers)
    compile: CompileConfig = field(default_factory=CompileConfig)

    def validate(self) -> Tuple[bool, str]:
        """
        Validate the IR structure.

        Returns:
            Tuple of (valid, error_message)
        """
        # Check voice model tile_size is power of two
        ts = self.meta.voice_model.tile_size
        if ts & (ts - 1) != 0 or ts == 0:
            return False, f"tile_size must be power of two, got {ts}"

        # Check regime nodes have valid templates
        template_names = set(self.layers.L3_cmmm.emit_templates.keys())
        for node in self.layers.L2_regimes.regime_nodes:
            if node.emit_template and node.emit_template not in template_names:
                # Allow if no templates defined (completion will fill)
                if template_names:
                    return False, f"Regime {node.id} references unknown template {node.emit_template}"

        # Check frame emitter policy lane assignments
        for lane_name, lane_id in self.compile.frame_emitter_policy.lanes.items():
            if lane_id < 0 or lane_id >= 8:
                return False, f"Lane {lane_name} has invalid ID {lane_id} (must be 0-7)"

        return True, ""


# =============================================================================
# PARSER HELPERS
# =============================================================================

def _parse_macro(data: dict) -> Macro:
    """Parse a macro from dict."""
    return Macro(
        id=data.get('id', ''),
        name=data.get('name', ''),
        range=tuple(data.get('range', [0.0, 1.0])),
        default=data.get('default', 0.5),
    )


def _parse_adsr(data: dict) -> ADSR:
    """Parse ADSR from dict."""
    return ADSR(
        a_ms=data.get('a_ms', 8.0),
        d_ms=data.get('d_ms', 80.0),
        s=data.get('s', 0.7),
        r_ms=data.get('r_ms', 180.0),
    )


def _parse_node_params(data: dict) -> NodeParams:
    """Parse node params from dict."""
    params = NodeParams()
    if 'waveform' in data:
        params.waveform = data['waveform']
    if 'detune_cents' in data:
        params.detune_cents = data['detune_cents']
    if 'pw' in data:
        params.pw = data['pw']
    if 'color' in data:
        params.color = data['color']
    if 'mode' in data:
        params.mode = data['mode']
    if 'adsr' in data:
        params.adsr = _parse_adsr(data['adsr'])
    if 'send_a' in data:
        params.send_a = data['send_a']
    if 'send_b' in data:
        params.send_b = data['send_b']
    return params


def _parse_l0_node(data: dict) -> L0Node:
    """Parse L0 node from dict."""
    return L0Node(
        id=data.get('id', ''),
        type=data.get('type', ''),
        params=_parse_node_params(data.get('params', {})),
    )


def _parse_l0_edge(data: dict) -> L0Edge:
    """Parse L0 edge from dict."""
    return L0Edge(
        src=data.get('src', ''),
        dst=data.get('dst', ''),
    )


def _parse_l0_graph(data: dict) -> L0Graph:
    """Parse L0 graph from dict."""
    return L0Graph(
        macros=[_parse_macro(m) for m in data.get('macros', [])],
        nodes=[_parse_l0_node(n) for n in data.get('nodes', [])],
        edges=[_parse_l0_edge(e) for e in data.get('edges', [])],
    )


def _parse_gesture_source(data: dict) -> GestureSource:
    """Parse gesture source from dict."""
    return GestureSource(
        range=tuple(data.get('range', [0, 1])),
        source=data.get('source', ''),
    )


def _parse_macro_affect(data: dict) -> MacroAffect:
    """Parse macro affect from dict."""
    return MacroAffect(
        target=data.get('target', ''),
        map=data.get('map', ''),
    )


def _parse_macro_semantic(data: dict) -> MacroSemantic:
    """Parse macro semantic from dict."""
    return MacroSemantic(
        affects=[_parse_macro_affect(a) for a in data.get('affects', [])],
    )


def _parse_l1_lift(data: dict) -> L1Lift:
    """Parse L1 lift from dict."""
    gesture_basis = {}
    for name, gdata in data.get('gesture_basis', {}).items():
        gesture_basis[name] = _parse_gesture_source(gdata)

    macro_semantics = {}
    for name, mdata in data.get('macro_semantics', {}).items():
        macro_semantics[name] = _parse_macro_semantic(mdata)

    return L1Lift(
        gesture_basis=gesture_basis,
        macro_semantics=macro_semantics,
    )


def _parse_hysteresis(data: dict) -> Hysteresis:
    """Parse hysteresis from dict."""
    return Hysteresis(
        enter=data.get('enter', 0.5),
        exit=data.get('exit', 0.4),
    )


def _parse_regime_node(data: dict) -> RegimeNode:
    """Parse regime node from dict."""
    return RegimeNode(
        id=data.get('id', ''),
        kind=data.get('kind', ''),
        emit_template=data.get('emit_template', ''),
        dwell_ms=data.get('dwell_ms'),
    )


def _parse_regime_transition(data: dict) -> RegimeTransition:
    """Parse regime transition from dict."""
    hysteresis = None
    if 'hysteresis' in data:
        hysteresis = _parse_hysteresis(data['hysteresis'])

    return RegimeTransition(
        from_regime=data.get('from', ''),
        to=data.get('to', ''),
        when=data.get('when', ''),
        hysteresis=hysteresis,
    )


def _parse_l2_regimes(data: dict) -> L2Regimes:
    """Parse L2 regimes from dict."""
    return L2Regimes(
        regime_nodes=[_parse_regime_node(n) for n in data.get('regime_nodes', [])],
        transitions=[_parse_regime_transition(t) for t in data.get('transitions', [])],
    )


def _parse_tree_target(data: dict) -> TreeTarget:
    """Parse tree target from dict."""
    return TreeTarget(
        A=data.get('A', 'T1'),
        B=data.get('B', 'T1'),
        beta=data.get('beta', 0.0),
    )


def _parse_event_payload_template(data: dict) -> EventPayloadTemplate:
    """Parse event payload template from dict."""
    return EventPayloadTemplate(
        variant=data.get('variant', 0),
        op=data.get('op'),
        target_kind=data.get('target_kind'),
        target_id=data.get('target_id', 0),
        pattern_id=data.get('pattern_id', 0),
        p0=data.get('p0', 0),
        p1=data.get('p1', 0),
        p2=data.get('p2', 0),
        wave_type=data.get('wave_type'),
        partials_N=data.get('partials_N'),
        pull_q8=data.get('pull_q8'),
    )


def _parse_event_policy(data: dict) -> EventPolicy:
    """Parse event policy from dict."""
    return EventPolicy(
        lane=data.get('lane', ''),
        condition=data.get('condition', ''),
        payload_template=_parse_event_payload_template(data.get('payload_template', {})),
        strength=str(data.get('strength', '0.0')),
    )


def _parse_emit_template(data: dict) -> EmitTemplate:
    """Parse emit template from dict."""
    trees = TreeTarget()
    if 'trees' in data:
        trees = _parse_tree_target(data['trees'])

    return EmitTemplate(
        families=data.get('families', {}),
        ops_enable=data.get('ops_enable', []),
        trees=trees,
        events=[_parse_event_policy(e) for e in data.get('events', [])],
    )


def _parse_transition_choice(data: dict) -> TransitionChoice:
    """Parse transition choice from dict."""
    return TransitionChoice(
        to=data.get('to', ''),
        weight=data.get('weight', 0.0),
    )


def _parse_transition_rule(data: dict) -> TransitionRule:
    """Parse transition rule from dict."""
    return TransitionRule(
        name=data.get('name', ''),
        from_regime=data.get('from', ''),
        to=data.get('to', ''),
        type=data.get('type', 'deterministic'),
        when=data.get('when', ''),
        choices=[_parse_transition_choice(c) for c in data.get('choices', [])],
        prng_draw_site=data.get('prng_draw_site'),
    )


def _parse_suppression_rule(data: dict) -> SuppressionRule:
    """Parse suppression rule from dict."""
    return SuppressionRule(
        when=data.get('when', ''),
        suppress=data.get('suppress', []),
    )


def _parse_overlay_add(data: dict) -> OverlayAdd:
    """Parse overlay add from dict."""
    return OverlayAdd(
        families=data.get('families', {}),
        events=[_parse_event_policy(e) for e in data.get('events', [])],
    )


def _parse_overlay(data: dict) -> Overlay:
    """Parse overlay from dict."""
    return Overlay(
        id=data.get('id', ''),
        enable_when=data.get('enable_when', ''),
        kind=data.get('kind', 'conditional'),
        add=_parse_overlay_add(data.get('add', {})),
        suppression_rules=[_parse_suppression_rule(s) for s in data.get('suppression_rules', [])],
    )


def _parse_state_variable(data: dict) -> StateVariable:
    """Parse state variable from dict."""
    return StateVariable(
        type=data.get('type', ''),
        init=data.get('init'),
    )


def _parse_cmmm_state(data: dict) -> CmmmState:
    """Parse CMMM state from dict."""
    variables = {}
    for name, vdata in data.get('variables', {}).items():
        variables[name] = _parse_state_variable(vdata)
    return CmmmState(variables=variables)


def _parse_cmmm_inputs(data: dict) -> CmmmInputs:
    """Parse CMMM inputs from dict."""
    return CmmmInputs(
        gesture=data.get('gesture', []),
        note=data.get('note', {}),
        block_index=data.get('block_index', 'u32'),
    )


def _parse_cmmm_outputs(data: dict) -> CmmmOutputs:
    """Parse CMMM outputs from dict."""
    return CmmmOutputs(
        frame_fields=data.get('frame_fields', []),
    )


def _parse_l3_cmmm(data: dict) -> L3Cmmm:
    """Parse L3 CMMM from dict."""
    emit_templates = {}
    for name, tdata in data.get('emit_templates', {}).items():
        emit_templates[name] = _parse_emit_template(tdata)

    return L3Cmmm(
        state=_parse_cmmm_state(data.get('state', {})),
        inputs=_parse_cmmm_inputs(data.get('inputs', {})),
        outputs=_parse_cmmm_outputs(data.get('outputs', {})),
        emit_templates=emit_templates,
        transition_kernel=[_parse_transition_rule(r) for r in data.get('transition_kernel', [])],
        overlays=[_parse_overlay(o) for o in data.get('overlays', [])],
    )


def _parse_completion_policy(data: dict) -> CompletionPolicy:
    """Parse completion policy from dict."""
    default_tree = TreeTarget()
    if 'default_tree' in data:
        default_tree = _parse_tree_target(data['default_tree'])

    return CompletionPolicy(
        if_missing_L1_lift=data.get('if_missing_L1_lift', ''),
        if_missing_L2_regimes=data.get('if_missing_L2_regimes', ''),
        if_missing_L3_cmmm=data.get('if_missing_L3_cmmm', ''),
        always_add_tail_ops=data.get('always_add_tail_ops', []),
        default_tree=default_tree,
    )


def _parse_degrade_group(data: dict) -> DegradeGroup:
    """Parse degrade group from dict."""
    return DegradeGroup(
        includes=data.get('includes', []),
        priority=data.get('priority', ''),
    )


def _parse_degrade_profile(data: dict) -> DegradeProfile:
    """Parse degrade profile from dict."""
    return DegradeProfile(
        top_group=_parse_degrade_group(data.get('top_group', {})),
        mid_group=_parse_degrade_group(data.get('mid_group', {})),
        never_drop=_parse_degrade_group(data.get('never_drop', {})),
    )


def _parse_events_policy(data: dict) -> EventsPolicy:
    """Parse events policy from dict."""
    return EventsPolicy(
        choose_order=data.get('choose_order', 'lane_id_ascending_then_template_order'),
        merge_policy=data.get('merge_policy', 'at most one payload per lane per block; strongest wins'),
    )


def _parse_frame_emitter_policy(data: dict) -> FrameEmitterPolicy:
    """Parse frame emitter policy from dict."""
    return FrameEmitterPolicy(
        voiceframe_version=data.get('voiceframe_version', 1),
        endianness=data.get('endianness', 'big'),
        lane_limit=data.get('lane_limit', 8),
        lanes=data.get('lanes', {}),
        events=_parse_events_policy(data.get('events', {})),
        reject_on=data.get('reject_on', []),
    )


def _parse_compile_config(data: dict) -> CompileConfig:
    """Parse compile config from dict."""
    return CompileConfig(
        map_library=data.get('map_library', {}),
        completion_policy=_parse_completion_policy(data.get('completion_policy', {})),
        degrade_profile=_parse_degrade_profile(data.get('degrade_profile', {})),
        frame_emitter_policy=_parse_frame_emitter_policy(data.get('frame_emitter_policy', {})),
    )


def _parse_voice_model(data: dict) -> VoiceModel:
    """Parse voice model from dict."""
    return VoiceModel(
        max_voices=data.get('max_voices', 128),
        tile_size=data.get('tile_size', 16),
    )


def _parse_instrument_meta(data: dict) -> InstrumentMeta:
    """Parse instrument meta from dict."""
    return InstrumentMeta(
        instrument_id=data.get('instrument_id', ''),
        name=data.get('name', ''),
        version=data.get('version', 0),
        author=data.get('author', ''),
        seed_u32=data.get('seed_u32', 0),
        voice_model=_parse_voice_model(data.get('voice_model', {})),
    )


def _parse_io_config(data: dict) -> IOConfig:
    """Parse IO config from dict."""
    return IOConfig(
        sample_rate_hz=data.get('sample_rate_hz', 48000),
        block_size=data.get('block_size', 64),
    )


def _parse_layers(data: dict) -> Layers:
    """Parse layers from dict."""
    return Layers(
        L0_graph=_parse_l0_graph(data.get('L0_graph', {})),
        L1_lift=_parse_l1_lift(data.get('L1_lift', {})),
        L2_regimes=_parse_l2_regimes(data.get('L2_regimes', {})),
        L3_cmmm=_parse_l3_cmmm(data.get('L3_cmmm', {})),
    )


# =============================================================================
# LOADER FUNCTIONS
# =============================================================================

def load_instrument_from_dict(data: dict) -> InstrumentIR:
    """
    Load InstrumentIR from a dictionary.

    Args:
        data: Dictionary with instrument IR data

    Returns:
        InstrumentIR object
    """
    # Handle top-level wrapper if present
    if 'instrument_ir_v0' in data:
        data = data['instrument_ir_v0']

    return InstrumentIR(
        meta=_parse_instrument_meta(data.get('meta', {})),
        io=_parse_io_config(data.get('io', {})),
        layers=_parse_layers(data.get('layers', {})),
        compile=_parse_compile_config(data.get('compile', {})),
    )


def load_instrument(path: str) -> InstrumentIR:
    """
    Load InstrumentIR from a YAML or JSON file.

    Args:
        path: Path to the instrument file (.yaml, .yml, or .json)

    Returns:
        InstrumentIR object

    Raises:
        ImportError: If trying to load YAML without pyyaml installed
    """
    with open(path, 'r', encoding='utf-8') as f:
        if path.endswith('.json'):
            data = json.load(f)
        else:
            if not HAS_YAML:
                raise ImportError(
                    "YAML support requires pyyaml. Install with: pip install pyyaml"
                )
            data = yaml.safe_load(f)

    return load_instrument_from_dict(data)


def save_instrument(ir: InstrumentIR, path: str) -> None:
    """
    Save InstrumentIR to a YAML or JSON file.

    Args:
        ir: InstrumentIR object
        path: Output path (.yaml, .yml, or .json)
    """
    # Convert to dict (simplified, for demonstration)
    data = {
        'instrument_ir_v0': {
            'meta': {
                'instrument_id': ir.meta.instrument_id,
                'name': ir.meta.name,
                'version': ir.meta.version,
                'author': ir.meta.author,
                'seed_u32': ir.meta.seed_u32,
                'voice_model': {
                    'max_voices': ir.meta.voice_model.max_voices,
                    'tile_size': ir.meta.voice_model.tile_size,
                },
            },
            'io': {
                'sample_rate_hz': ir.io.sample_rate_hz,
                'block_size': ir.io.block_size,
            },
            # Simplified - full serialization would include all layers
        }
    }

    with open(path, 'w', encoding='utf-8') as f:
        if path.endswith('.json'):
            json.dump(data, f, indent=2)
        else:
            if not HAS_YAML:
                raise ImportError(
                    "YAML support requires pyyaml. Install with: pip install pyyaml"
                )
            yaml.safe_dump(data, f, default_flow_style=False)


# =============================================================================
# COMPLETION HELPERS
# =============================================================================

def create_default_l1_lift() -> L1Lift:
    """Create default L1 lift with standard gesture basis."""
    return L1Lift(
        gesture_basis={
            'Gate': GestureSource(range=(0, 1), source='note_on_off'),
            'Pressure': GestureSource(range=(0, 1), source='aftertouch_or_modwheel'),
            'Position': GestureSource(range=(0, 1), source='cc74_or_macro'),
            'Speed': GestureSource(range=(0, 1), source='derivative(Position)'),
            'Tilt': GestureSource(range=(-1, 1), source='macro(M_BRIGHT) mapped bipolar'),
            'Chaos': GestureSource(range=(0, 1), source='macro(M_GLITCH)'),
        },
    )


def create_default_l2_regimes() -> L2Regimes:
    """Create default L2 regimes with standard skeleton."""
    return L2Regimes(
        regime_nodes=[
            RegimeNode(id='R0_IDLE', kind='Idle', emit_template='T_IDLE'),
            RegimeNode(id='R1_ATTACK', kind='Transient', emit_template='T_ATTACK', dwell_ms=20),
            RegimeNode(id='R2_STABLE', kind='StableTone', emit_template='T_STABLE'),
            RegimeNode(id='R6_RECOVERY', kind='Recovery', emit_template='T_RECOVERY', dwell_ms=80),
        ],
        transitions=[
            RegimeTransition(from_regime='R0_IDLE', to='R1_ATTACK', when='Gate rises'),
            RegimeTransition(from_regime='R1_ATTACK', to='R2_STABLE', when='dwell_done'),
            RegimeTransition(from_regime='any', to='R0_IDLE', when='Gate falls'),
        ],
    )


def create_default_emit_templates() -> Dict[str, EmitTemplate]:
    """Create default emission templates."""
    return {
        'T_IDLE': EmitTemplate(
            families={'CONSTRAINT': 1.0, 'CALM': 1.0},
            ops_enable=['NormalizeScale', 'FineClamp'],
            trees=TreeTarget(A='T1', B='T1', beta=0.0),
        ),
        'T_ATTACK': EmitTemplate(
            families={'COLOR': 0.6, 'EXCITE': 1.0, 'CONSTRAINT': 1.0, 'CALM': 0.5},
            ops_enable=['Tilt', 'IntraMix', 'NormalizeScale', 'FineClamp'],
            trees=TreeTarget(A='T1', B='T1', beta=0.0),
        ),
        'T_STABLE': EmitTemplate(
            families={'COLOR': 0.5, 'COHERENCE': 0.4, 'CONSTRAINT': 1.0, 'CALM': 0.7},
            ops_enable=['Tilt', 'LowRankMorph', 'NormalizeScale', 'FineClamp', 'SheafProject'],
            trees=TreeTarget(A='T1', B='T1', beta=0.0),
        ),
        'T_RECOVERY': EmitTemplate(
            families={'COHERENCE': 0.3, 'CONSTRAINT': 1.0, 'CALM': 1.0},
            ops_enable=['GluePair', 'SheafProject', 'NormalizeScale', 'FineClamp'],
            trees=TreeTarget(A='T1', B='T1', beta=0.0),
        ),
    }


def complete_instrument(ir: InstrumentIR) -> InstrumentIR:
    """
    Complete a partial InstrumentIR with defaults.

    Follows the completion policy to fill in missing layers.

    Args:
        ir: Partial InstrumentIR

    Returns:
        Completed InstrumentIR
    """
    # Complete L1 if missing
    if not ir.layers.L1_lift.gesture_basis:
        ir.layers.L1_lift = create_default_l1_lift()

    # Complete L2 if missing
    if not ir.layers.L2_regimes.regime_nodes:
        ir.layers.L2_regimes = create_default_l2_regimes()

    # Complete L3 emit templates if missing
    if not ir.layers.L3_cmmm.emit_templates:
        ir.layers.L3_cmmm.emit_templates = create_default_emit_templates()

    # Add tail ops if specified
    tail_ops = ir.compile.completion_policy.always_add_tail_ops
    if tail_ops:
        for template in ir.layers.L3_cmmm.emit_templates.values():
            for op in tail_ops:
                if op not in template.ops_enable:
                    template.ops_enable.append(op)

    return ir


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'NodeType', 'WaveformType', 'NoiseColor', 'FilterMode',
    'RegimeKind', 'TransitionType', 'OverlayKind', 'EventLane',
    # L0 types
    'Macro', 'ADSR', 'NodeParams', 'L0Node', 'L0Edge', 'L0Graph',
    # L1 types
    'GestureSource', 'MacroAffect', 'MacroSemantic', 'L1Lift',
    # L2 types
    'Hysteresis', 'RegimeNode', 'RegimeTransition', 'L2Regimes',
    # L3 types
    'StateVariable', 'CmmmState', 'CmmmInputs', 'CmmmOutputs',
    'TreeTarget', 'EventPayloadTemplate', 'EventPolicy', 'EmitTemplate',
    'TransitionChoice', 'TransitionRule', 'SuppressionRule', 'OverlayAdd', 'Overlay',
    'L3Cmmm',
    # Compile types
    'CompletionPolicy', 'DegradeGroup', 'DegradeProfile',
    'EventsPolicy', 'FrameEmitterPolicy', 'CompileConfig',
    # Top-level types
    'VoiceModel', 'InstrumentMeta', 'IOConfig', 'Layers', 'InstrumentIR',
    # Functions
    'load_instrument', 'load_instrument_from_dict', 'save_instrument',
    'complete_instrument',
    'create_default_l1_lift', 'create_default_l2_regimes', 'create_default_emit_templates',
]
