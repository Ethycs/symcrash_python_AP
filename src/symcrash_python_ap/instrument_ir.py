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
# UNIT CONVERSION HELPERS
# =============================================================================

def ms_to_blocks(ms: float, block_size: int = 64, sample_rate: int = 48000) -> int:
    """
    Convert milliseconds to block count.

    Args:
        ms: Time in milliseconds
        block_size: Samples per block (default 64)
        sample_rate: Sample rate in Hz (default 48000)

    Returns:
        Number of blocks (minimum 1)

    Example:
        >>> ms_to_blocks(10)  # 10ms at 48kHz/64 samples
        8
    """
    samples = (ms / 1000.0) * sample_rate
    return max(1, int(samples / block_size))


def pct_to_q15(pct: float) -> int:
    """
    Convert 0.0-1.0 percentage to Q15 fixed-point (0-32767).

    Args:
        pct: Value from 0.0 to 1.0

    Returns:
        Q15 value (0-32767)

    Example:
        >>> pct_to_q15(0.5)
        16383
        >>> pct_to_q15(1.0)
        32767
    """
    return int(max(0.0, min(1.0, pct)) * 32767)


def q15_to_pct(q15: int) -> float:
    """
    Convert Q15 fixed-point (0-32767) to 0.0-1.0 percentage.

    Args:
        q15: Q15 value (0-32767)

    Returns:
        Float from 0.0 to 1.0

    Example:
        >>> q15_to_pct(16383)
        0.4999847...
    """
    return q15 / 32767.0


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
    """Fixed event lane assignments (v1 with carrier-first semantics)."""
    EVT_OP_PULSE = 0
    EVT_OP_TOGGLE = 1
    EVT_TREE_MORPH = 2
    EVT_K_MODE = 3
    EVT_CARRIER_CFG = 4     # Was EVT_INJECT_NOTE in v0
    EVT_MATERIAL_HIT = 5
    EVT_CLOUD_PUSH = 6
    EVT_RESERVED = 7

    # Backwards compatibility alias
    EVT_INJECT_NOTE = EVT_CARRIER_CFG


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
class VibratoParams:
    """Vibrato/LFO modulation parameters."""
    rate_hz: float = 5.5         # LFO rate in Hz (typical vibrato 5-7 Hz)
    depth_amp: float = 0.0       # Amplitude modulation depth (0-1)
    depth_timbre: float = 0.0    # Timbre modulation depth (0-1)
    delay_ms: float = 40.0       # Delay before vibrato onset
    ramp_ms: float = 30.0        # Ramp-in time for vibrato


@dataclass
class BodyWarbleParams:
    """Body resonance warble parameters (slow formant drift)."""
    rate_hz: float = 1.2         # Slow LFO rate (0.5-2 Hz typical)
    depth: float = 0.0           # Warble depth (0-1)
    target_harmonic: int = 4     # Which harmonic to modulate (1-indexed)


@dataclass
class AttackTextureParams:
    """Attack texture parameters (brightness settling)."""
    bright_ms: float = 60.0      # Time for brightness to settle
    bright_amount: float = 0.0   # Extra brightness on attack (0-1)
    harmonics: List[int] = field(default_factory=lambda: [5, 6])  # Which harmonics to boost


@dataclass
class HarmonicProfile:
    """Custom harmonic amplitude profile."""
    amplitudes: List[float] = field(default_factory=lambda: [1.0])  # H1, H2, H3... (0-1 scale)


@dataclass
class NodeParams:
    """
    Parameters for DSP nodes.

    Per WSE-STD-001: Q1.15 for gains/strengths, Q0.16 for unipolar, ENUM8 for modes.
    """
    # Oscillator
    waveform: Optional[str] = None
    detune_cents: float = 0.0
    pw: float = 0.5  # Pulse width (legacy, use pwm_width_q15 for precision)

    # Oscillator mode (0=A/LUT, 1=B/Rotator, 2=Sample, 3=FM6)
    osc_mode: int = 0

    # Waveform type (0=sine, 1=square, 2=pulse, 3=saw, 4=saw_down, 5=tri)
    waveform_type: int = 0

    # Carrier-specific fields (v1 Addendum A)
    role: Optional[str] = None        # "off" | "audible" | "exciter_only"
    archetype: Optional[str] = None   # TreeArchetype name

    # Harmonic profile (for Mode B synthesis)
    harmonics: Optional[HarmonicProfile] = None

    # Modulation parameters
    vibrato: Optional[VibratoParams] = None
    body_warble: Optional[BodyWarbleParams] = None
    attack_texture: Optional[AttackTextureParams] = None

    # PWM parameters (Q0.15 per WSE-STD-001)
    pwm_enable: bool = False
    pwm_width_q15: int = 16384        # Duty cycle Q0.15 (16384=50%)

    # Hard sync parameters
    sync_enable: bool = False
    sync_ratio_q16: int = 65536       # Master/slave ratio Q16.16 (65536=1.0)

    # Mode B FM parameters
    fm_enable: bool = False
    fm_depth_q15: int = 0             # FM depth Q0.15
    fm_source: int = 0                # 0=off, 1=feedback, 2=LFO

    # Unison parameters
    unison_voices: int = 1            # Voice count (1-8)
    unison_spread: int = 10           # Detune spread in cents

    # Mode flags
    drone_mode: bool = False          # Sustained coefficient mode
    legato_mode: bool = False         # Smooth pitch transitions

    # Sample playback parameters
    sample_rate_q16: int = 65536      # Playback rate Q16.16 (65536=1.0x)

    # Noise
    color: Optional[str] = None

    # Filter
    mode: Optional[str] = None

    # Amp
    adsr: Optional[ADSR] = None

    # FX Send
    send_a: float = 0.0
    send_b: float = 0.0

    # FM6 parameters (6-operator FM synthesis, OSC_MODE=3)
    algorithm: Optional[int] = None          # DX7 algorithm 1-32
    feedback: Optional[int] = None           # Feedback level 0-7
    op_ratios: Optional[List[int]] = None    # 6 Q16.16 frequency ratios
    op_levels: Optional[List[int]] = None    # 6 Q0.15 output levels
    fm6_topo_a: int = 0                      # Source archetype 0-7
    fm6_topo_b: int = 0                      # Target archetype 0-7
    fm6_topo_blend_q15: int = 0              # Topology morph Q0.15


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
    # For EVT_INJECT_NOTE (legacy)
    wave_type: Optional[str] = None
    partials_N: Optional[int] = None
    pull_q8: Optional[int] = None
    # For EVT_CARRIER_CFG (v1 Addendum A)
    carrier_model: Optional[str] = None    # CarrierModel name
    carrier_role: Optional[str] = None     # CarrierRole name
    carrier_archetype: Optional[str] = None  # TreeArchetype name


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

    Natural compilation to CapsuleV0:
        ir = load_instrument("my_synth.yaml")
        capsule = ir.to_capsule()  # Returns CapsuleV0

        # Or with explicit compiler for validation:
        from symcrash_python_ap.library.compiler import CapsuleCompiler
        compiler = CapsuleCompiler()
        result = compiler.compile_from_ir(ir)
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

    def to_capsule(self, program_id: int = None, capsule_id: str = None, auto_complete: bool = True) -> Any:
        """
        Compile this IR to a CapsuleV0.

        This is the natural compilation path - the IR contains all metadata,
        control mappings, regime graphs, and emission templates needed.

        Missing L1/L2/L3 layers are automatically filled with sensible defaults
        unless auto_complete=False.

        Args:
            program_id: Override program_id (uses hash of instrument_id if None)
            capsule_id: Override capsule_id (generated from IR if None)
            auto_complete: If True (default), fill missing layers with defaults

        Returns:
            CapsuleV0 ready for library storage or runtime loading

        Raises:
            ValueError: If IR validation fails or compilation errors occur

        Example:
            # Minimal IR with auto-completion:
            ir = InstrumentIR()
            ir.meta.instrument_id = "my-synth"
            ir.meta.name = "My Synth"
            capsule = ir.to_capsule()  # L1/L2/L3 auto-filled!

            # From YAML:
            ir = load_instrument("brass_stab.yaml")
            capsule = ir.to_capsule()
        """
        from .library.compiler import CapsuleCompiler

        # Auto-complete missing layers if requested
        if auto_complete:
            completed_ir = complete_instrument(self)
        else:
            completed_ir = self

        compiler = CapsuleCompiler(
            sample_rate=completed_ir.io.sample_rate_hz,
            block_size=completed_ir.io.block_size,
        )

        result = compiler.compile_from_ir(completed_ir, program_id=program_id, capsule_id=capsule_id)

        if not result.success:
            errors = "; ".join(result.errors)
            raise ValueError(f"IR compilation failed: {errors}")

        return result.capsule

    def to_compile_result(self, program_id: int = None, capsule_id: str = None, auto_complete: bool = True) -> Any:
        """
        Compile this IR to a CompileResult (includes warnings and CapsuleRecord).

        Use this instead of to_capsule() when you need the full compilation
        result including warnings and the CapsuleRecord in COMPILED state.

        Args:
            program_id: Override program_id (uses hash of instrument_id if None)
            capsule_id: Override capsule_id (generated from IR if None)
            auto_complete: If True (default), fill missing layers with defaults

        Returns:
            CompileResult with capsule, record, warnings, and errors

        Example:
            ir = load_instrument("brass_stab.yaml")
            result = ir.to_compile_result()
            if result.success:
                capsule = result.capsule
                record = result.record  # CapsuleRecord in COMPILED state
            for warning in result.warnings:
                print(f"Warning: {warning}")
        """
        from .library.compiler import CapsuleCompiler

        # Auto-complete missing layers if requested
        if auto_complete:
            completed_ir = complete_instrument(self)
        else:
            completed_ir = self

        compiler = CapsuleCompiler(
            sample_rate=completed_ir.io.sample_rate_hz,
            block_size=completed_ir.io.block_size,
        )

        return compiler.compile_from_ir(completed_ir, program_id=program_id, capsule_id=capsule_id)

    # =========================================================================
    # QUICK-BUILD FACTORY METHODS
    # =========================================================================

    @classmethod
    def quick_synth(
        cls,
        name: str,
        waveform: str = "saw",
        attack_ms: float = 10,
        decay_ms: float = 100,
        sustain: float = 0.7,
        release_ms: float = 200,
        brightness: float = 0.5,
        author: str = "",
    ) -> 'InstrumentIR':
        """
        Create a minimal synth preset in one line.

        This is the recommended way to create simple presets programmatically.
        For complex presets, use YAML files with load_instrument().

        Args:
            name: Human-readable preset name (also generates instrument_id)
            waveform: Oscillator waveform ("saw", "sine", "pulse", "tri")
            attack_ms: Attack time in milliseconds
            decay_ms: Decay time in milliseconds
            sustain: Sustain level (0.0-1.0)
            release_ms: Release time in milliseconds
            brightness: Timbre brightness (0.0-1.0)
            author: Optional author name

        Returns:
            Complete InstrumentIR with L0-L3 auto-filled

        Example:
            ir = InstrumentIR.quick_synth("Bright Saw", brightness=0.8)
            capsule = ir.to_capsule()

            # Or chain it:
            capsule = InstrumentIR.quick_synth("Pad", attack_ms=500).to_capsule()
        """
        ir = cls()
        ir.meta.instrument_id = name.lower().replace(" ", "-").replace("_", "-")
        ir.meta.name = name
        ir.meta.author = author

        # L0: Single osc + amp chain
        ir.layers.L0_graph.nodes = [
            L0Node(id="osc1", type="Osc", params=NodeParams(waveform=waveform)),
            L0Node(id="amp1", type="Amp", params=NodeParams(
                adsr=ADSR(a_ms=attack_ms, d_ms=decay_ms, s=sustain, r_ms=release_ms)
            )),
        ]
        ir.layers.L0_graph.edges = [L0Edge(src="osc1", dst="amp1")]

        # L0: Brightness macro
        ir.layers.L0_graph.macros = [
            Macro(id="M_BRIGHT", name="Brightness", default=brightness)
        ]

        # Auto-complete L1/L2/L3 with defaults
        return complete_instrument(ir)

    @classmethod
    def quick_pad(cls, name: str, **kwargs) -> 'InstrumentIR':
        """
        Create a pad preset with slow attack/release defaults.

        Args:
            name: Preset name
            **kwargs: Override any quick_synth parameter

        Returns:
            InstrumentIR configured for pad sounds

        Example:
            ir = InstrumentIR.quick_pad("Warm Pad", brightness=0.3)
        """
        defaults = dict(
            attack_ms=300,
            decay_ms=200,
            sustain=0.8,
            release_ms=800,
        )
        defaults.update(kwargs)
        return cls.quick_synth(name, **defaults)

    @classmethod
    def quick_pluck(cls, name: str, **kwargs) -> 'InstrumentIR':
        """
        Create a pluck preset with fast attack, short decay.

        Args:
            name: Preset name
            **kwargs: Override any quick_synth parameter

        Returns:
            InstrumentIR configured for pluck/percussive sounds

        Example:
            ir = InstrumentIR.quick_pluck("Sharp Pluck", waveform="pulse")
        """
        defaults = dict(
            attack_ms=2,
            decay_ms=150,
            sustain=0.0,
            release_ms=100,
        )
        defaults.update(kwargs)
        return cls.quick_synth(name, **defaults)

    @classmethod
    def quick_bass(cls, name: str, **kwargs) -> 'InstrumentIR':
        """
        Create a bass preset with punchy attack and medium sustain.

        Args:
            name: Preset name
            **kwargs: Override any quick_synth parameter

        Returns:
            InstrumentIR configured for bass sounds

        Example:
            ir = InstrumentIR.quick_bass("Sub Bass", waveform="sine")
        """
        defaults = dict(
            attack_ms=5,
            decay_ms=80,
            sustain=0.6,
            release_ms=150,
            brightness=0.3,
        )
        defaults.update(kwargs)
        return cls.quick_synth(name, **defaults)

    @classmethod
    def quick_lead(cls, name: str, **kwargs) -> 'InstrumentIR':
        """
        Create a lead preset with medium attack and high sustain.

        Args:
            name: Preset name
            **kwargs: Override any quick_synth parameter

        Returns:
            InstrumentIR configured for lead sounds

        Example:
            ir = InstrumentIR.quick_lead("Screaming Lead", brightness=0.9)
        """
        defaults = dict(
            attack_ms=15,
            decay_ms=100,
            sustain=0.85,
            release_ms=200,
            brightness=0.7,
        )
        defaults.update(kwargs)
        return cls.quick_synth(name, **defaults)


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

    # Oscillator basics
    if 'waveform' in data:
        params.waveform = data['waveform']
    if 'detune_cents' in data:
        params.detune_cents = data['detune_cents']
    if 'pw' in data:
        params.pw = data['pw']
    if 'osc_mode' in data:
        params.osc_mode = data['osc_mode']
    if 'waveform_type' in data:
        params.waveform_type = data['waveform_type']

    # Carrier-specific fields (v1 Addendum A)
    if 'role' in data:
        params.role = data['role']
    if 'archetype' in data:
        params.archetype = data['archetype']

    # PWM parameters
    if 'pwm_enable' in data:
        params.pwm_enable = data['pwm_enable']
    if 'pwm_width_q15' in data:
        params.pwm_width_q15 = data['pwm_width_q15']

    # Hard sync parameters
    if 'sync_enable' in data:
        params.sync_enable = data['sync_enable']
    if 'sync_ratio_q16' in data:
        params.sync_ratio_q16 = data['sync_ratio_q16']

    # Mode B FM parameters
    if 'fm_enable' in data:
        params.fm_enable = data['fm_enable']
    if 'fm_depth_q15' in data:
        params.fm_depth_q15 = data['fm_depth_q15']
    if 'fm_source' in data:
        params.fm_source = data['fm_source']

    # Unison parameters
    if 'unison_voices' in data:
        params.unison_voices = data['unison_voices']
    if 'unison_spread' in data:
        params.unison_spread = data['unison_spread']

    # Mode flags
    if 'drone_mode' in data:
        params.drone_mode = data['drone_mode']
    if 'legato_mode' in data:
        params.legato_mode = data['legato_mode']

    # Sample playback
    if 'sample_rate_q16' in data:
        params.sample_rate_q16 = data['sample_rate_q16']

    # Noise / Filter / Amp
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

    # FM6 parameters (6-operator synthesis)
    if 'algorithm' in data:
        params.algorithm = data['algorithm']
    if 'feedback' in data:
        params.feedback = data['feedback']
    if 'op_ratios' in data:
        params.op_ratios = data['op_ratios']
    if 'op_levels' in data:
        params.op_levels = data['op_levels']
    if 'fm6_topo_a' in data:
        params.fm6_topo_a = data['fm6_topo_a']
    if 'fm6_topo_b' in data:
        params.fm6_topo_b = data['fm6_topo_b']
    if 'fm6_topo_blend_q15' in data:
        params.fm6_topo_blend_q15 = data['fm6_topo_blend_q15']

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
        # EVT_CARRIER_CFG fields (v1 Addendum A)
        carrier_model=data.get('carrier_model'),
        carrier_role=data.get('carrier_role'),
        carrier_archetype=data.get('carrier_archetype'),
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
    # Unit conversion helpers
    'ms_to_blocks', 'pct_to_q15', 'q15_to_pct',
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
