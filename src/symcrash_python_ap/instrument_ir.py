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

# Import canonical primitives (single source of truth)
from .primitives import (
    EventLane as _EventLane,
    TailMode as _TailMode,
    K_LADDER_MASKS,
    get_k_mask,
)

# Lazy import for central registry (avoid circular imports)
def _get_central_registry():
    """Lazy import of CentralRegistry to avoid circular imports."""
    from .library.central_registry import get_central_registry
    return get_central_registry()


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


# EventLane imported from primitives (single source of truth)
# Re-export for backwards compatibility
EventLane = _EventLane


# =============================================================================
# L0 GRAPH IR (Conventional DSP)
# =============================================================================

@dataclass
class ADSR:
    """Envelope parameters."""
    a_ms: float = 8.0
    d_ms: float = 80.0
    s: float = 0.7
    r_ms: float = 180.0


@dataclass
class EnvelopeBinding:
    """
    Named envelope with routing to targets.

    Part of the Matrix Envelope System - allows one envelope to control
    multiple targets (e.g., carrier envelope → FM_OP1-4).

    See envelope_targets.py for valid target names:
    - FM_OP1-FM_OP6: FM operator envelope values
    - GAIN: Voice gain multiplier
    - FILTER_CUTOFF, FILTER_RESO: Filter parameters (future)
    - PITCH, VIBRATO_DEPTH: Pitch modulation (future)
    """
    id: str                          # e.g., "carrier_env"
    adsr: ADSR = field(default_factory=ADSR)  # Envelope shape
    targets: List[str] = field(default_factory=list)  # Target names


@dataclass
class ControlConfig:
    """
    Control axis defaults for cockpit integration.

    Defines default values for semantic control axes that presets can specify.
    These values are pushed to MacroRuntime at preset load time via ControlProgram.

    All values are Q15 unsigned (0-32767) representing 0.0-1.0 normalized.

    Semantic meanings (per WSE-STD-MACRO v0):
    - energy_intent: Playing intensity / attack character
    - motion: Movement / modulation amount (gesture tracking)
    - focus: Attention / emphasis (spectral concentration)
    - brightness: Timbral brightness (harmonic content)
    - density: Body / thickness (low-mid presence)
    - risk: Chaos / instability (stochastic behavior)
    - space: Spatial / reverb character
    - material: Surface texture (wood=low, metal=mid, glass=high)
    - unity: Coherence / blend (how much voices blend together)

    YAML example:
        control:
          brightness: 0.75    # High brightness for brass
          density: 0.5        # Neutral body
          energy_intent: 0.6  # Slightly aggressive
          risk: 0.1           # Some instability

    Or using Q15 directly:
        control:
          brightness_q15: 24576
          density_q15: 16384
    """
    # Controller → CMMM axes (normalized 0.0-1.0 or Q15 0-32767)
    energy_intent: float = 0.5        # 0.5 = neutral intensity
    motion: float = 0.0               # 0.0 = no modulation
    focus: float = 0.5                # 0.5 = balanced emphasis
    brightness: float = 0.5           # 0.5 = neutral brightness
    density: float = 0.5              # 0.5 = neutral body
    risk: float = 0.0                 # 0.0 = stable, no chaos
    space: float = 0.25               # 0.25 = subtle spatial
    material: float = 0.5             # 0.5 = neutral (neither wood nor glass)
    unity: float = 0.5                # 0.5 = balanced blend

    def to_control_program(self) -> 'ControlProgram':
        """
        Convert to ControlProgram for capsule compilation.

        Returns:
            ControlProgram with Q15 values
        """
        from .library.capsule import ControlProgram

        return ControlProgram(
            energy_intent_q15=int(self.energy_intent * 32767),
            motion_q15=int(self.motion * 32767),
            focus_q15=int(self.focus * 32767),
            brightness_q15=int(self.brightness * 32767),
            density_q15=int(self.density * 32767),
            risk_q15=int(self.risk * 32767),
            space_q15=int(self.space * 32767),
            material_q15=int(self.material * 32767),
            unity_q15=int(self.unity * 32767),
        )


@dataclass
class Macro:
    """User-facing macro parameter."""
    id: str
    name: str
    range: Tuple[float, float] = (0.0, 1.0)
    default: float = 0.5


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

    # Modulation parameters (DEPRECATED: moved to L1Lift for Tower conformance)
    # Keep for backwards compatibility with existing presets
    vibrato: Optional[VibratoParams] = None           # Prefer L1Lift.vibrato
    body_warble: Optional[BodyWarbleParams] = None    # Prefer L1Lift.body_warble
    attack_texture: Optional[AttackTextureParams] = None  # Prefer L1Lift.attack_texture

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

    # Filter (legacy mode string)
    mode: Optional[str] = None

    # Filter parameters (subtractive synthesis, Q0.15 per WSE-STD-001)
    filter_enable: bool = False
    filter_type: int = 0              # 0=LP, 1=HP, 2=BP, 3=Notch
    filter_cutoff_q15: int = 32767    # Cutoff Q0.15 (log scale 20Hz-20kHz)
    filter_resonance_q15: int = 0     # Resonance Q0.15 (0=flat, 32767=self-osc)
    filter_env_depth_q15: int = 0     # Envelope mod Q1.15 (signed)
    filter_key_track_q15: int = 0     # Key tracking Q0.15 (16384=50%)
    filter_drive_q15: int = 0         # Pre-filter saturation Q0.15
    filter_slope: int = 1             # 0=12dB/oct, 1=24dB/oct, 2=6dB/oct

    # Amp
    adsr: Optional[ADSR] = None

    # FX Send
    send_a: float = 0.0
    send_b: float = 0.0

    # FM6 parameters (6-operator FM synthesis, OSC_MODE=3)
    # Static algorithm/feedback/ratios/levels are L0 (patch-level, fixed)
    # Topology morphing (topo_a, topo_b, blend) moved to L1Lift.fm6_morph
    algorithm: Optional[int] = None          # DX7 algorithm 1-32
    feedback: Optional[int] = None           # Feedback level 0-7
    op_ratios: Optional[List[int]] = None    # 6 Q16.16 frequency ratios
    op_levels: Optional[List[int]] = None    # 6 Q0.15 output levels
    # Legacy fields for backwards compatibility (prefer L1Lift.fm6_morph)
    fm6_topo_a: int = 0                      # Source archetype 0-7 (DEPRECATED: use L1Lift.fm6_morph)
    fm6_topo_b: int = 0                      # Target archetype 0-7 (DEPRECATED: use L1Lift.fm6_morph)
    fm6_topo_blend_q15: int = 0              # Topology morph Q0.15 (DEPRECATED: use L1Lift.fm6_morph)


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
class FM6MorphConfig:
    """
    FM6 topology morphing configuration (L1 family primitive).

    Defines interpolation between two FM topologies for dynamic timbral morphing.
    Static algorithm/feedback/ratios/levels stay in L0 NodeParams, but morphing
    between archetypes is an L1 family behavior controlled by gesture/macro.

    Per Full Tower v0: L1 defines "family template" including morph pairs.
    """
    topo_a: int = 0                   # Source archetype 0-7
    topo_b: int = 0                   # Target archetype 0-7
    blend_q15: int = 0                # Initial blend Q0.15 (gesture can modulate)
    blend_source: str = "static"      # "static" | "gesture:motion" | "macro:M_BODY"


@dataclass
class IntentMappingConfig:
    """
    Gesture → Intent mapping configuration (L1 primitive).

    Per InstrumentIR v0 §2.3:
    - mode="default" uses WSE-STD-GESTURE default_gesture_to_intent()
    - mode="custom" uses the custom_map expressions

    Gesture Basis (8 raw inputs):
        Gate, Pressure, Position, Speed, Tilt, Chaos, Pitch, Pan

    Intent Vector (8 semantic outputs):
        I_TONE=0, I_BOW=1, I_AIR=2, I_GLITCH=3,
        I_WARP=4, I_SNAP=5, I_GLUE=6, I_CALM=7
    """
    mode: str = "default"             # "default" | "custom"
    custom_map: Dict[str, str] = field(default_factory=dict)  # e.g. {"I_TONE": "expr..."}


@dataclass
class FamilyMappingConfig:
    """
    Intent → Family weight mapping configuration (L1 primitive).

    Per InstrumentIR v0 §2.4:
    - mode="default" uses WSE-STD-GESTURE default family mapping
    - mode="custom" uses the custom_map expressions

    Intent lanes (input):
        I_TONE, I_BOW, I_AIR, I_GLITCH, I_WARP, I_SNAP, I_GLUE, I_CALM

    Family indices (output):
        F_RESERVED=0, F_COLOR=1, F_TRANSPORT=2, F_LOCAL=3,
        F_TOPOLOGY=4, F_GLUE=5, F_EVENTS=6, F_CATASTROPHE=7
    """
    mode: str = "default"             # "default" | "custom"
    custom_map: Dict[str, str] = field(default_factory=dict)  # e.g. {"F_COLOR": "expr..."}


@dataclass
class L1Lift:
    """
    L1 gesture basis, macro semantics, and control manifold.

    Per Full Tower v0 Section L1 — Instrument Family:
    - Gesture basis: raw inputs (MIDI, MPE, etc.) → normalized control signals
    - Macro semantics: how L0 macros map to internal primitives
    - Control manifold: semantic axes defining the family's character space

    The control manifold (ControlConfig) IS the L1 primitive - it defines
    the 9 semantic axes that characterize the instrument family:
    energy_intent, motion, focus, brightness, density, risk, space, material, unity

    Additionally L1 holds:
    - FM6 topology morphing config (morph between FM archetypes)
    - Modulation parameters (vibrato, body_warble, attack_texture)

    These are family-level because they define gesture→sound relationships,
    not per-regime behavior (which is L2).
    """
    gesture_basis: Dict[str, GestureSource] = field(default_factory=dict)
    macro_semantics: Dict[str, MacroSemantic] = field(default_factory=dict)

    # Control manifold axes (the L1 semantic space)
    control: Optional[ControlConfig] = None

    # Gesture → Intent mapping (how raw gesture becomes semantic intent)
    intent_mapping: Optional[IntentMappingConfig] = None

    # Intent → Family weight mapping (how intent becomes operator weights)
    family_mapping: Optional[FamilyMappingConfig] = None

    # FM6 topology morphing (L1 family morph pairs)
    fm6_morph: Optional[FM6MorphConfig] = None

    # Modulation parameters (gesture-based, family-level behavior)
    vibrato: Optional[VibratoParams] = None
    body_warble: Optional[BodyWarbleParams] = None
    attack_texture: Optional[AttackTextureParams] = None


# =============================================================================
# L2 REGIMES IR (Regime Graph)
# =============================================================================

@dataclass
class Hysteresis:
    """Hysteresis thresholds for transitions."""
    enter: float
    exit: float


class EdgeTransitionType(Enum):
    """
    Edge transition types per Voice Engine Tower spec.

    Per symcrash_docs/03 - Audio Pipeline/00 - Voice Engine/Evolution/05:
    These define how the transition between regimes behaves.
    """
    HARD = "hard"                  # Immediate switch, no crossfade
    SOFT = "soft"                  # Smooth crossfade over duration_blocks
    CHAOTIC = "chaotic"            # Stochastic switching with jitter
    CORRIDOR_ENTRY = "corridor"    # Entry into degradation corridor
    RECOVERY = "recovery"          # Return from instability to stable regime


@dataclass
class EdgeSpec:
    """
    Tower-conformant edge specification per Voice Engine Tower spec.

    Per symcrash_docs/03 - Audio Pipeline/00 - Voice Engine/Evolution/05:
    EdgeSpec defines transitions with typed behavior, guards, and overlays.
    """
    src_regime: str
    dst_regime: str
    transition_type: str = "soft"           # EdgeTransitionType value
    guard: str = ""                          # Condition expression (ConditionSpec)
    hysteresis: Optional[Hysteresis] = None
    min_dwell_blocks: int = 0               # Minimum time before transition allowed
    duration_blocks: int = 4                # Transition crossfade duration
    overlay: Optional[str] = None           # Overlay to apply during transition


@dataclass
class RegimeNode:
    """L2 regime graph node."""
    id: str
    kind: str
    emit_template: str
    dwell_ms: Optional[int] = None


@dataclass
class RegimeTransition:
    """L2 regime transition (legacy format, use EdgeSpec for tower-conformant)."""
    from_regime: str  # 'from' is reserved in Python
    to: str
    when: str
    hysteresis: Optional[Hysteresis] = None


@dataclass
class RegimeGraphSpec:
    """
    Tower-conformant regime graph specification.

    Per symcrash_docs/03 - Audio Pipeline/00 - Voice Engine/Evolution/05:
    RegimeGraphSpec defines the full state machine topology.
    """
    start_regime: str = "R0_IDLE"
    edges: List[EdgeSpec] = field(default_factory=list)


@dataclass
class RegimeEnvelopeOverride:
    """
    Per-regime envelope override.

    Allows regimes to modify envelope behavior (e.g., faster release in recovery).
    References envelope by id and provides override ADSR values.
    """
    envelope_id: str                  # Reference to EnvelopeBinding.id
    adsr_override: Optional[ADSR] = None  # Override ADSR for this regime
    scale_q15: int = 32767            # Output scaling Q0.15 (32767 = 1.0)


@dataclass
class L2Regimes:
    """
    L2 regime graph - the graph-based state machine.

    Per Full Tower v0 Section L2 — Instrument Instance:
    - Regime graph with numeric thresholds
    - Policy tables mapping regime → operator program template
    - Per-regime envelope overrides (Matrix Envelope System policy)

    The Matrix Envelope System bindings live here because envelope routing
    is a per-regime policy decision. Different regimes may route envelopes
    to different targets or modify their ADSR shapes.

    Supports both simple (regime_nodes + transitions) and
    full graph spec (regime_graph_spec with typed EdgeSpecs) formats.
    """
    regime_nodes: List[RegimeNode] = field(default_factory=list)
    transitions: List[RegimeTransition] = field(default_factory=list)
    # Full graph spec with typed edges (optional, takes precedence if present)
    regime_graph_spec: Optional[RegimeGraphSpec] = None

    # Matrix Envelope System - envelope definitions with target routing
    # These are L2 because envelope→target routing is regime-dependent policy
    envelopes: List[EnvelopeBinding] = field(default_factory=list)

    # Per-regime envelope overrides (e.g., faster release in R6_RECOVERY)
    envelope_overrides: Dict[str, List[RegimeEnvelopeOverride]] = field(default_factory=dict)


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
    """Tree blend target for topology morphing."""
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
class CmmmControl:
    """
    CMMM runtime control fields (WSE-STD-070 §10).

    Controls executor latch state for K degradation and tail operator selection.

    Attributes:
        node_mask: Target node mask (32-bit) or pattern reference ("pattern:N")
        mask_ramp_blocks: Crossfade duration in blocks (default 5)
        tail_mode: Tail operator set (FULL=0, MID=1, MIN=2, EMERGENCY=3)
        traction_q15: Projection hardness (0-32767, default full)
    """
    node_mask: Union[int, str] = 0xFFFFFFFF  # Can be int or "pattern:N"
    mask_ramp_blocks: int = 5
    tail_mode: Union[int, str] = 0  # Can be int or name (FULL, MID, MIN, EMERGENCY)
    traction_q15: int = 32767

    # Pattern table for K ladder - use primitives.K_LADDER_MASKS (single source of truth)
    # Kept as class attr for backwards compatibility
    PATTERN_TABLE = K_LADDER_MASKS

    # Tail mode name mapping - use primitives.TailMode values
    TAIL_MODE_MAP = {
        "FULL": _TailMode.FULL,
        "MID": _TailMode.MID,
        "MIN": _TailMode.MIN,
        "EMERGENCY": _TailMode.EMERGENCY,
    }

    def get_node_mask_u32(self) -> int:
        """Resolve node_mask to a 32-bit integer."""
        if isinstance(self.node_mask, int):
            return self.node_mask & 0xFFFFFFFF
        if isinstance(self.node_mask, str):
            # Pattern reference: "pattern:N"
            if self.node_mask.startswith("pattern:"):
                pattern_id = int(self.node_mask.split(":")[1])
                return get_k_mask(pattern_id)
            # Hex string
            if self.node_mask.startswith("0x"):
                return int(self.node_mask, 16) & 0xFFFFFFFF
        return 0xFFFFFFFF

    def get_tail_mode_int(self) -> int:
        """Resolve tail_mode to an integer."""
        if isinstance(self.tail_mode, int):
            return self.tail_mode
        if isinstance(self.tail_mode, str):
            return self.TAIL_MODE_MAP.get(self.tail_mode.upper(), 0)
        return 0


@dataclass
class EmitTemplate:
    """
    Emission template for a regime.

    Defines what the CMMM outputs when in a given regime:
    - families: Weight distribution across family operators
    - ops_enable: Which operators are enabled
    - trees: Tree blend target for topology morphing
    - events: Conditional event emissions
    - cmmm: CMMM runtime control (node mask, tail mode) per WSE-STD-070 §10
    """
    families: Dict[str, Union[float, str]] = field(default_factory=dict)
    ops_enable: List[str] = field(default_factory=list)
    trees: TreeTarget = field(default_factory=TreeTarget)
    events: List[EventPolicy] = field(default_factory=list)
    cmmm: Optional[CmmmControl] = None  # CMMM runtime control (WSE-STD-070)


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
    """
    L3-L5 Controlled Markov Mealy Machine (CMMM).

    The CMMM is a graph-based state machine that controls voice synthesis:
    - Regimes define states (Idle, Attack, Stable, etc.)
    - Transitions define edges with typed behaviors
    - Emit templates define what each regime outputs
    - Overlays provide conditional modifications

    Per CMMM spec, this drives:
    - Operator selection and budgeting
    - Family weight computation
    - Tree topology morphing
    - Event lane triggering
    """
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
class MaterialConfig:
    """
    Material engine configuration (WSE-STD-MATERIAL-v0).

    Controls the Material Engine analog processing layer.
    The 6 material programs provide different coloration:
    - 0: CLEAN (pass-through)
    - 1: TAPE (hysteresis saturation)
    - 2: CATASTROPHE (bifurcation maps)
    - 3: RANK (nonlinear filtering)
    - 4: TROPICAL (max operations)
    - 5: RESERVED

    Beta controls the blend between dry carrier and wet CMMM-processed material.
    """
    # Material program selection
    material_id: int = 0             # Material program (0-5)

    # Dry/wet blend
    beta_start_q15: int = 0          # Starting blend (0 = fully dry)
    beta_target_q15: int = 16384     # Target blend (0.5 default)
    beta_ramp_blocks: int = 8        # Blocks to ramp from start to target

    # Per-program parameters (Q15)
    tape_saturation_q15: int = 8192      # Tape hysteresis amount
    catastrophe_threshold_q15: int = 24576  # Bifurcation trigger point
    rank_filter_order: int = 3           # Nonlinear filter order (1-8)
    tropical_blend_q15: int = 16384      # Max operation blend


@dataclass
class SafetyConfig:
    """
    Safety engine configuration (WSE-STD-SAFETY).

    Controls the Safety Engine audio-space tail processing:
    - OLA (overlap-add) for wet path smoothing
    - DC removal (disabled for CCS path)
    - Limiter for amplitude safety
    - Slew limiting for click prevention

    Note: CCS path produces phase-continuous sine waves with zero inherent DC,
    so dc_removal and master_ola are disabled by default.
    """
    enabled: bool = True                    # Enable safety processing
    dc_removal: bool = False                # DC offset removal (disabled for CCS)
    master_ola: bool = False                # Master OLA (handled in lane_idwpt_render)
    limiter_threshold_q15: int = 32000      # Hard limiter threshold
    slew_limit_q15: int = 2000              # Maximum sample-to-sample change
    energy_cap_soft_q32: int = 0x30000000   # Soft energy cap (warning)
    energy_cap_hard_q32: int = 0x40000000   # Hard energy cap (enforce)


@dataclass
class CompileConfig:
    """Compile section configuration."""
    map_library: Dict[str, str] = field(default_factory=dict)
    completion_policy: CompletionPolicy = field(default_factory=CompletionPolicy)
    degrade_profile: DegradeProfile = field(default_factory=DegradeProfile)
    frame_emitter_policy: FrameEmitterPolicy = field(default_factory=FrameEmitterPolicy)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)


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

    Tower Layer Placement (per Full Tower v0):
    - L0 (Patch): layers.L0_graph - static DSP graph, fixed params
    - L1 (Family): layers.L1_lift - control manifold, gesture semantics, FM6 morph
    - L2 (Instance): layers.L2_regimes - regime graph, envelope bindings
    - L3 (Cockpit): layers.L3_cmmm - CMMM emission, overlays

    Access patterns:
    - Control axes: ir.layers.L1_lift.control (or ir.control for backwards compat)
    - Envelopes: ir.layers.L2_regimes.envelopes (or ir.envelopes for backwards compat)
    - FM6 morph: ir.layers.L1_lift.fm6_morph
    - Modulation: ir.layers.L1_lift.vibrato, .body_warble, .attack_texture

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

    # DEPRECATED: These are now in their proper Tower layers
    # Kept for backwards compatibility - will be migrated to layers on load
    # Access via ir.control or ir.envelopes properties instead
    _envelopes_compat: List[EnvelopeBinding] = field(default_factory=list)
    _control_compat: Optional[ControlConfig] = None

    @property
    def envelopes(self) -> List[EnvelopeBinding]:
        """Get envelopes from L2 layer (or backwards-compat field)."""
        if self.layers.L2_regimes.envelopes:
            return self.layers.L2_regimes.envelopes
        return self._envelopes_compat

    @envelopes.setter
    def envelopes(self, value: List[EnvelopeBinding]):
        """Set envelopes - stores in L2 layer."""
        self.layers.L2_regimes.envelopes = value

    @property
    def control(self) -> Optional[ControlConfig]:
        """Get control config from L1 layer (or backwards-compat field)."""
        if self.layers.L1_lift.control is not None:
            return self.layers.L1_lift.control
        return self._control_compat

    @control.setter
    def control(self, value: Optional[ControlConfig]):
        """Set control config - stores in L1 layer."""
        self.layers.L1_lift.control = value

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


def _parse_envelope_binding(data: dict) -> EnvelopeBinding:
    """
    Parse envelope binding from dict.

    Example YAML:
        - id: "carrier_env"
          adsr: { a_ms: 8, d_ms: 150, s: 0.7, r_ms: 100 }
          targets: [FM_OP1, FM_OP2, FM_OP3, FM_OP4]
    """
    return EnvelopeBinding(
        id=data.get('id', ''),
        adsr=_parse_adsr(data.get('adsr', {})),
        targets=data.get('targets', []),
    )


def _parse_control_config(data: dict) -> ControlConfig:
    """
    Parse control config from dict.

    Supports both normalized (0.0-1.0) and Q15 (0-32767) values.
    Q15 values take precedence if both are specified.

    Example YAML:
        control:
          brightness: 0.75      # High brightness for brass
          density: 0.5          # Neutral body
          energy_intent: 0.6    # Slightly aggressive
          risk: 0.1             # Some instability

    Or using Q15 directly:
        control:
          brightness_q15: 24576
          density_q15: 16384
    """
    def get_value(name: str, default: float) -> float:
        """Get value, preferring Q15 over normalized."""
        q15_key = f'{name}_q15'
        if q15_key in data:
            return data[q15_key] / 32767.0
        return data.get(name, default)

    return ControlConfig(
        energy_intent=get_value('energy_intent', 0.5),
        motion=get_value('motion', 0.0),
        focus=get_value('focus', 0.5),
        brightness=get_value('brightness', 0.5),
        density=get_value('density', 0.5),
        risk=get_value('risk', 0.0),
        space=get_value('space', 0.25),
        material=get_value('material', 0.5),
        unity=get_value('unity', 0.5),
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

    # Noise
    if 'color' in data:
        params.color = data['color']

    # Filter (legacy mode string + new Q15 parameters)
    if 'mode' in data:
        params.mode = data['mode']
    if 'filter_enable' in data:
        params.filter_enable = data['filter_enable']
    if 'filter_type' in data:
        params.filter_type = data['filter_type']
    if 'filter_cutoff_q15' in data:
        params.filter_cutoff_q15 = data['filter_cutoff_q15']
    if 'filter_resonance_q15' in data:
        params.filter_resonance_q15 = data['filter_resonance_q15']
    if 'filter_env_depth_q15' in data:
        params.filter_env_depth_q15 = data['filter_env_depth_q15']
    if 'filter_key_track_q15' in data:
        params.filter_key_track_q15 = data['filter_key_track_q15']
    if 'filter_drive_q15' in data:
        params.filter_drive_q15 = data['filter_drive_q15']
    if 'filter_slope' in data:
        params.filter_slope = data['filter_slope']

    # Amp / FX Send
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


def _parse_vibrato_params(data: dict) -> VibratoParams:
    """Parse vibrato params from dict."""
    return VibratoParams(
        rate_hz=data.get('rate_hz', 5.5),
        depth_amp=data.get('depth_amp', 0.0),
        depth_timbre=data.get('depth_timbre', 0.0),
        delay_ms=data.get('delay_ms', 40.0),
        ramp_ms=data.get('ramp_ms', 30.0),
    )


def _parse_body_warble_params(data: dict) -> BodyWarbleParams:
    """Parse body warble params from dict."""
    return BodyWarbleParams(
        rate_hz=data.get('rate_hz', 1.2),
        depth=data.get('depth', 0.0),
        target_harmonic=data.get('target_harmonic', 4),
    )


def _parse_attack_texture_params(data: dict) -> AttackTextureParams:
    """Parse attack texture params from dict."""
    return AttackTextureParams(
        bright_ms=data.get('bright_ms', 60.0),
        bright_amount=data.get('bright_amount', 0.0),
        harmonics=data.get('harmonics', [5, 6]),
    )


def _parse_fm6_morph_config(data: dict) -> FM6MorphConfig:
    """Parse FM6 morph config from dict."""
    return FM6MorphConfig(
        topo_a=data.get('topo_a', 0),
        topo_b=data.get('topo_b', 0),
        blend_q15=data.get('blend_q15', 0),
        blend_source=data.get('blend_source', 'static'),
    )


def _parse_intent_mapping_config(data: dict) -> IntentMappingConfig:
    """Parse intent mapping config from dict."""
    return IntentMappingConfig(
        mode=data.get('mode', 'default'),
        custom_map=data.get('custom_map', {}),
    )


def _parse_family_mapping_config(data: dict) -> FamilyMappingConfig:
    """Parse family mapping config from dict."""
    return FamilyMappingConfig(
        mode=data.get('mode', 'default'),
        custom_map=data.get('custom_map', {}),
    )


def _parse_l1_lift(data: dict) -> L1Lift:
    """
    Parse L1 lift from dict.

    L1 contains:
    - gesture_basis: raw input → normalized control signals
    - macro_semantics: L0 macro → internal primitive mappings
    - control: semantic control axes (the L1 manifold)
    - intent_mapping: gesture → intent mapping config
    - family_mapping: intent → family weight mapping config
    - fm6_morph: FM topology morphing config
    - vibrato, body_warble, attack_texture: modulation params
    """
    gesture_basis = {}
    for name, gdata in data.get('gesture_basis', {}).items():
        gesture_basis[name] = _parse_gesture_source(gdata)

    macro_semantics = {}
    for name, mdata in data.get('macro_semantics', {}).items():
        macro_semantics[name] = _parse_macro_semantic(mdata)

    # Parse control config (L1 manifold axes)
    control = None
    if 'control' in data:
        control = _parse_control_config(data['control'])

    # Parse intent mapping (gesture → intent)
    intent_mapping = None
    if 'intent_mapping' in data:
        intent_mapping = _parse_intent_mapping_config(data['intent_mapping'])

    # Parse family mapping (intent → family weights)
    family_mapping = None
    if 'family_mapping' in data:
        family_mapping = _parse_family_mapping_config(data['family_mapping'])

    # Parse FM6 morph config
    fm6_morph = None
    if 'fm6_morph' in data:
        fm6_morph = _parse_fm6_morph_config(data['fm6_morph'])

    # Parse modulation params
    vibrato = None
    if 'vibrato' in data:
        vibrato = _parse_vibrato_params(data['vibrato'])

    body_warble = None
    if 'body_warble' in data:
        body_warble = _parse_body_warble_params(data['body_warble'])

    attack_texture = None
    if 'attack_texture' in data:
        attack_texture = _parse_attack_texture_params(data['attack_texture'])

    return L1Lift(
        gesture_basis=gesture_basis,
        macro_semantics=macro_semantics,
        control=control,
        intent_mapping=intent_mapping,
        family_mapping=family_mapping,
        fm6_morph=fm6_morph,
        vibrato=vibrato,
        body_warble=body_warble,
        attack_texture=attack_texture,
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


def _parse_regime_envelope_override(data: dict) -> RegimeEnvelopeOverride:
    """Parse regime envelope override from dict."""
    adsr_override = None
    if 'adsr_override' in data:
        adsr_override = _parse_adsr(data['adsr_override'])

    return RegimeEnvelopeOverride(
        envelope_id=data.get('envelope_id', ''),
        adsr_override=adsr_override,
        scale_q15=data.get('scale_q15', 32767),
    )


def _parse_l2_regimes(data: dict) -> L2Regimes:
    """
    Parse L2 regimes from dict.

    L2 contains:
    - regime_nodes: state machine nodes
    - transitions: state machine edges
    - regime_graph_spec: typed edge spec (optional)
    - envelopes: Matrix Envelope System bindings (L2 policy)
    - envelope_overrides: per-regime envelope modifications
    """
    # Parse envelope bindings (Matrix Envelope System)
    envelopes = [
        _parse_envelope_binding(env)
        for env in data.get('envelopes', [])
    ]

    # Parse per-regime envelope overrides
    envelope_overrides = {}
    for regime_id, overrides in data.get('envelope_overrides', {}).items():
        envelope_overrides[regime_id] = [
            _parse_regime_envelope_override(o) for o in overrides
        ]

    return L2Regimes(
        regime_nodes=[_parse_regime_node(n) for n in data.get('regime_nodes', [])],
        transitions=[_parse_regime_transition(t) for t in data.get('transitions', [])],
        envelopes=envelopes,
        envelope_overrides=envelope_overrides,
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


def _parse_cmmm_control(data: dict) -> CmmmControl:
    """Parse CMMM control from dict."""
    return CmmmControl(
        node_mask=data.get('node_mask', 0xFFFFFFFF),
        mask_ramp_blocks=data.get('mask_ramp_blocks', 5),
        tail_mode=data.get('tail_mode', 0),
        traction_q15=data.get('traction_q15', 32767),
    )


def _parse_emit_template(data: dict) -> EmitTemplate:
    """Parse emit template from dict."""
    trees = TreeTarget()
    if 'trees' in data:
        trees = _parse_tree_target(data['trees'])

    # Parse CMMM control if present (WSE-STD-070)
    cmmm = None
    if 'cmmm' in data:
        cmmm = _parse_cmmm_control(data['cmmm'])

    return EmitTemplate(
        families=data.get('families', {}),
        ops_enable=data.get('ops_enable', []),
        trees=trees,
        events=[_parse_event_policy(e) for e in data.get('events', [])],
        cmmm=cmmm,
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


def _parse_material_config(data: dict) -> MaterialConfig:
    """Parse material config from dict."""
    return MaterialConfig(
        beta_start_q15=data.get('beta_start_q15', 0),
        beta_target_q15=data.get('beta_target_q15', 16384),
        beta_ramp_blocks=data.get('beta_ramp_blocks', 8),
    )


def _parse_compile_config(data: dict) -> CompileConfig:
    """Parse compile config from dict."""
    return CompileConfig(
        map_library=data.get('map_library', {}),
        completion_policy=_parse_completion_policy(data.get('completion_policy', {})),
        degrade_profile=_parse_degrade_profile(data.get('degrade_profile', {})),
        frame_emitter_policy=_parse_frame_emitter_policy(data.get('frame_emitter_policy', {})),
        material=_parse_material_config(data.get('material', {})),
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

    The canonical format uses the instrument_ir_v0 wrapper:

        instrument_ir_v0:
          meta:
            instrument_id: "my-synth"
            name: "My Synth"
            ...
          layers:
            L0_graph:
              nodes: [...]
            L3_cmmm:
              emit_templates: {...}
          compile:
            material:
              beta_start_q15: 0
              beta_target_q15: 16384

    Required fields:
    - meta.instrument_id: Unique identifier
    - meta.name: Human-readable name

    Optional fields (with defaults):
    - meta.version: 0
    - meta.author: ""
    - io.sample_rate_hz: 48000
    - io.block_size: 64
    - layers.L0_graph: Auto-completed if missing
    - layers.L3_cmmm: Auto-completed if missing
    - compile.material: Default blend settings

    Args:
        data: Dictionary with instrument_ir_v0 wrapper

    Returns:
        InstrumentIR object

    Raises:
        ValueError: If using deprecated preset: format
    """
    # Handle top-level wrapper
    if 'instrument_ir_v0' in data:
        data = data['instrument_ir_v0']
    elif 'preset' in data:
        raise ValueError(
            "The 'preset:' format is deprecated. Use 'instrument_ir_v0:' format instead. "
            "See PRESET-AUTHORING-GUIDE.md for the canonical format."
        )

    # Parse envelope bindings (Matrix Envelope System)
    envelopes = [
        _parse_envelope_binding(env)
        for env in data.get('envelopes', [])
    ]

    # Parse control config (system macro defaults)
    control = None
    if 'control' in data:
        control = _parse_control_config(data['control'])

    return InstrumentIR(
        meta=_parse_instrument_meta(data.get('meta', {})),
        io=_parse_io_config(data.get('io', {})),
        layers=_parse_layers(data.get('layers', {})),
        compile=_parse_compile_config(data.get('compile', {})),
        envelopes=envelopes,
        control=control,
    )


def load_instrument(path) -> InstrumentIR:
    """
    Load InstrumentIR from a YAML or JSON file.

    Args:
        path: Path to the instrument file (.yaml, .yml, or .json)
              Can be string or Path object.

    Returns:
        InstrumentIR object

    Raises:
        ImportError: If trying to load YAML without pyyaml installed
    """
    from pathlib import Path as PathLib
    # Convert to string for extension checking
    path_str = str(path)

    with open(path, 'r', encoding='utf-8') as f:
        if path_str.endswith('.json'):
            data = json.load(f)
        else:
            if not HAS_YAML:
                raise ImportError(
                    "YAML support requires pyyaml. Install with: pip install pyyaml"
                )
            data = yaml.safe_load(f)

    return load_instrument_from_dict(data)


def save_instrument(ir: InstrumentIR, path) -> None:
    """
    Save InstrumentIR to a YAML or JSON file.

    Args:
        ir: InstrumentIR object
        path: Output path (.yaml, .yml, or .json). Can be string or Path object.
    """
    path_str = str(path)
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
        if path_str.endswith('.json'):
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
# TOWER OBJECT (WSE-STD-070 Tower Runtime)
# =============================================================================

class TowerLevel(IntEnum):
    """
    Tower levels per WSE-STD-070.

    Each level represents a different depth of instrument specification:
    - L0: Playable patch (always sounding)
    - L1: Family (gesture/intent semantics)
    - L2: Instrument (regime state machine)
    - L3: Cockpit (CMMM emission templates)
    - L4: Forge (graph editing, offline)
    - L5: Lab (custom operators/trees, offline)
    """
    L0_PATCH = 0       # Sounding patch
    L1_FAMILY = 1      # Gesture/intent semantics
    L2_INSTRUMENT = 2  # Regime state machine
    L3_COCKPIT = 3     # CMMM emission
    L4_FORGE = 4       # Graph editing (offline)
    L5_LAB = 5         # Custom ops/trees (offline)


@dataclass
class TowerBudget:
    """
    Budget configuration for Tower execution.

    Controls operator execution limits and degradation behavior.
    """
    tick_budget: int = 4096       # Max ticks per block per voice
    k_degradation: List[int] = field(default_factory=lambda: [32, 24, 16, 12, 8, 4, 1])
    tail_mode: int = 0            # Default tail mode (0=FULL)
    allow_nonlinear: bool = True  # Whether nonlinear ops are allowed


@dataclass
class Tower:
    """
    The Tower object wraps InstrumentIR layers with level tracking and
    projection/completion capabilities.

    Per WSE-STD-070:
    - Tower level declares the highest explicitly authored level
    - Missing layers are filled by Tower Completion (upward)
    - Projection maps allow safe descent (downward)

    This is the runtime-meaningful Tower abstraction that ensures:
    1. Every layer compiles to runtime behavior
    2. Deterministic completion and projection
    3. Budget-aware degradation
    """
    tower_level: TowerLevel = TowerLevel.L2_INSTRUMENT
    L0_graph: L0Graph = field(default_factory=L0Graph)
    L1_lift: L1Lift = field(default_factory=L1Lift)
    L2_regimes: L2Regimes = field(default_factory=L2Regimes)
    L3_cmmm: L3Cmmm = field(default_factory=L3Cmmm)
    envelopes: List[EnvelopeBinding] = field(default_factory=list)
    budget: Optional[TowerBudget] = None

    # Registry references (resolved at compile time)
    regime_graph_id: Optional[int] = None
    tree_family_id: Optional[int] = None

    @classmethod
    def from_instrument_ir(cls, ir: 'InstrumentIR') -> 'Tower':
        """
        Create a Tower from an InstrumentIR.

        Infers tower_level from which layers are populated.
        """
        # Infer tower level from populated layers
        level = TowerLevel.L0_PATCH
        if ir.layers.L1_lift.gesture_basis:
            level = TowerLevel.L1_FAMILY
        if ir.layers.L2_regimes.regime_nodes:
            level = TowerLevel.L2_INSTRUMENT
        if ir.layers.L3_cmmm.emit_templates:
            level = TowerLevel.L3_COCKPIT

        return cls(
            tower_level=level,
            L0_graph=ir.layers.L0_graph,
            L1_lift=ir.layers.L1_lift,
            L2_regimes=ir.layers.L2_regimes,
            L3_cmmm=ir.layers.L3_cmmm,
            envelopes=ir.envelopes,
        )

    def to_layers(self) -> 'Layers':
        """Convert Tower back to Layers for InstrumentIR compatibility."""
        return Layers(
            L0_graph=self.L0_graph,
            L1_lift=self.L1_lift,
            L2_regimes=self.L2_regimes,
            L3_cmmm=self.L3_cmmm,
        )

    def complete(self) -> 'Tower':
        """
        Complete missing layers upward from current tower_level.

        Per WSE-STD-070 Section 4.2, completion always proceeds upward:
        Lk → Lk+1 → ... → L3

        Returns:
            New Tower with completed layers
        """
        result = Tower(
            tower_level=self.tower_level,
            L0_graph=self.L0_graph,
            L1_lift=self.L1_lift,
            L2_regimes=self.L2_regimes,
            L3_cmmm=self.L3_cmmm,
            envelopes=list(self.envelopes),
            budget=self.budget,
            regime_graph_id=self.regime_graph_id,
            tree_family_id=self.tree_family_id,
        )

        # Complete L1 if missing
        if not result.L1_lift.gesture_basis:
            result.L1_lift = create_default_l1_lift()

        # Complete L2 if missing
        if not result.L2_regimes.regime_nodes:
            result.L2_regimes = create_default_l2_regimes()

        # Complete L3 if missing
        if not result.L3_cmmm.emit_templates:
            result.L3_cmmm.emit_templates = create_default_emit_templates()

        return result

    def project_to_level(self, target: TowerLevel) -> 'Tower':
        """
        Safely descend the tower to a lower level.

        Per WSE-STD-070 Section 5.1, projection removes data above target_level
        and re-applies completion rules. Must preserve playability.

        Args:
            target: Target level to project to

        Returns:
            New Tower at the target level
        """
        if target > self.tower_level:
            raise ValueError(f"Cannot project upward from {self.tower_level.name} to {target.name}")

        result = Tower(
            tower_level=target,
            L0_graph=self.L0_graph,
            envelopes=list(self.envelopes),
            budget=self.budget,
        )

        # Copy layers up to target level
        if target >= TowerLevel.L1_FAMILY:
            result.L1_lift = self.L1_lift
        if target >= TowerLevel.L2_INSTRUMENT:
            result.L2_regimes = self.L2_regimes
        if target >= TowerLevel.L3_COCKPIT:
            result.L3_cmmm = self.L3_cmmm

        # Re-apply completion for playability
        return result.complete()

    def reset_to_safe(self) -> 'Tower':
        """
        Panic projection: drop to L2 with only safe operators.

        Per WSE-STD-070 Section 5.2:
        - Project to L2
        - Enable only SAFE operator set
        - Reset envelopes to neutral
        - Clear all event policies
        """
        result = self.project_to_level(TowerLevel.L2_INSTRUMENT)

        # Reset L3 to minimal safe configuration
        result.L3_cmmm = L3Cmmm()
        result.L3_cmmm.emit_templates = {
            "T_SAFE": EmitTemplate(
                families={"CONSTRAINT": 1.0, "CALM": 1.0},
                ops_enable=["NormalizeScale", "FineClamp"],
            )
        }

        # Update all regime nodes to use safe template
        for node in result.L2_regimes.regime_nodes:
            node.emit_template = "T_SAFE"

        # Reset budget to conservative
        if result.budget is None:
            result.budget = TowerBudget()
        result.budget.tail_mode = 2  # MIN

        return result

    def get_conformance_profile(self) -> str:
        """
        Get the conformance profile for this Tower.

        Returns one of:
        - "P0": L0 + envelopes (static synthesis)
        - "P1": P0 + L2 regimes (state machine drives emit selection)
        - "P2": P1 + L3 CMMM (full expressive synthesis)
        - "P3": P2 + budgeting (deterministic degradation)
        """
        if self.budget is not None and self.budget.tick_budget > 0:
            return "P3"
        if self.L3_cmmm.emit_templates:
            return "P2"
        if self.L2_regimes.regime_nodes:
            return "P1"
        return "P0"

    def validate_registry_references(self) -> List[str]:
        """
        Validate that registry references exist in CentralRegistry.

        Checks:
        - regime_graph_id exists in RegimeRegistry
        - tree_family_id exists in TreeRegistry
        - Any pattern_id references in emit templates exist

        Returns:
            List of error messages (empty if all valid)
        """
        errors = []
        registry = _get_central_registry()

        # Validate regime_graph_id
        if self.regime_graph_id is not None:
            if registry.regimes.get(self.regime_graph_id) is None:
                errors.append(f"Regime graph ID {self.regime_graph_id} not found in registry")

        # Validate tree_family_id
        if self.tree_family_id is not None:
            if registry.trees.get(self.tree_family_id) is None:
                errors.append(f"Tree family ID {self.tree_family_id} not found in registry")

        # Validate pattern references in emit templates
        for template_id, template in self.L3_cmmm.emit_templates.items():
            if hasattr(template, 'pattern_id') and template.pattern_id is not None:
                if registry.patterns.get(template.pattern_id) is None:
                    # Check K ladder as fallback
                    if template.pattern_id not in K_LADDER_MASKS:
                        errors.append(
                            f"Pattern ID {template.pattern_id} in template '{template_id}' "
                            f"not found in registry or K ladder"
                        )

        return errors

    def resolve_registry_references(self) -> 'Tower':
        """
        Resolve registry IDs to actual objects from CentralRegistry.

        This populates:
        - L2_regimes from regime_graph_id (if set)
        - Tree family info (stored in metadata)

        Returns:
            New Tower with resolved references
        """
        registry = _get_central_registry()
        result = Tower(
            tower_level=self.tower_level,
            L0_graph=self.L0_graph,
            L1_lift=self.L1_lift,
            L2_regimes=self.L2_regimes,
            L3_cmmm=self.L3_cmmm,
            envelopes=list(self.envelopes),
            budget=self.budget,
            regime_graph_id=self.regime_graph_id,
            tree_family_id=self.tree_family_id,
        )

        # Resolve regime graph if ID is set but L2 is empty
        if self.regime_graph_id is not None and not self.L2_regimes.regime_nodes:
            regime_graph = registry.regimes.get(self.regime_graph_id)
            if regime_graph is not None:
                # Convert RegimeGraphSpec to L2Regimes
                result.L2_regimes = L2Regimes(
                    regime_nodes=[
                        RegimeNode(
                            id=node.id,
                            kind=RegimeKind.STABLE if node.kind == "StableTone" else RegimeKind.ONSET,
                            emit_template=node.emit_template,
                        )
                        for node in regime_graph.nodes
                    ],
                    transitions=[
                        RegimeTransition(
                            from_regime=edge.src_regime,
                            to_regime=edge.dst_regime,
                            condition=edge.guard or "auto",
                            priority=1,
                        )
                        for edge in regime_graph.edges
                    ],
                    regime_graph_spec=regime_graph,
                )

        return result


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Unit conversion helpers
    'ms_to_blocks', 'pct_to_q15', 'q15_to_pct',
    # Enums
    'NodeType', 'WaveformType', 'NoiseColor', 'FilterMode',
    'RegimeKind', 'TransitionType', 'OverlayKind', 'EventLane',
    'EdgeTransitionType',  # Graph-based state machine transition types
    # L0 types
    'Macro', 'ADSR', 'EnvelopeBinding', 'ControlConfig', 'NodeParams', 'L0Node', 'L0Edge', 'L0Graph',
    # L1 types
    'GestureSource', 'MacroAffect', 'MacroSemantic', 'L1Lift',
    # L2 types (regime graph - the graph-based state machine)
    'Hysteresis', 'EdgeSpec', 'RegimeGraphSpec',
    'RegimeNode', 'RegimeTransition', 'L2Regimes',
    # L3 types (CMMM - Controlled Markov Mealy Machine)
    'StateVariable', 'CmmmState', 'CmmmInputs', 'CmmmOutputs',
    'TreeTarget', 'EventPayloadTemplate', 'EventPolicy', 'CmmmControl', 'EmitTemplate',
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
    # Tower (WSE-STD-070)
    'TowerLevel', 'TowerBudget', 'Tower',
]
