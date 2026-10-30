"""
Voice Engine Instrument Master Implementation v3.0
===================================================

Implements a Controlled Semi-Markov Mealy Machine (CMMM) for voice control.
Targets Voice Engine Instrument Master Spec v1.0. Verified runtime profiles
and remaining compiler/transition gaps are recorded in the conformance report.

Architecture: CMMM
------------------
- Controlled: External GestureBasis input steers behavior
- Semi-Markov: State durations are non-geometric (ADSR envelopes)
- Mealy: Outputs (VoiceFrame) depend on transitions, not just states

Requirements:
- Total: Always produces a valid VoiceFrame
- Reversible/Replayable: Given seed + inputs, produces identical outputs
- Time-only state evolution: State depends only on explicit state + inputs + block
- FPGA-friendly: Bounded memory + predictable runtime

Key Data Structures (per spec):
- VoiceBank: Container for programs, presets, control maps
- InstrumentProgram: Full instrument definition with all policies
- VoiceState: Complete snapshotable state for replay (includes RNG)
- GestureBasis: 8 normalized physical gesture inputs
- IntentVector: 8 semantic intent lanes
- RegimeGraph: States + edges + hysteresis + dwell
- EdgeSpec: Transition conditions with HysteresisSpec and OverlaySpec
- ArcSpec: LFOs, splines, scripts, patterns
- FamilyPolicySpec: W_q15 matrix mapping intent to family weights
- TreePolicySpec: Selection/blend rules for tree morphing
- EventPolicySpec: Per-lane rules for event generation
- VoiceFrameABI: Complete output per voice per block

Tower Compliance (Levels 0-4):
- L0 Patch: VoiceModel + GestureMap + fixed TreePolicy + static FamilyPolicy
- L1 Modulation: Add ArcSpec for modulation
- L2 Instrument: Enable RegimeGraph and per-regime policies
- L3 Tree morph: Enable full TreePolicy with blending
- L4 Events: Enable EventPolicy and Arc.EventPattern
"""

from typing import List, Dict, Optional, Tuple, Callable, TYPE_CHECKING
from dataclasses import dataclass, field, replace
from enum import IntEnum
import numpy as np
from .control_bindings import VoiceControls
from .event_policy import EventPatternSpec, LaneRuleType, LaneRuleSpec, EventPolicySpec
from .ir_events import CompiledEventProgram, AuthoredEventState, event_context, phase_template, phase_regime
from .ir_regimes import AuthoredRegimeState
from .unison import UnisonMember
from ..frames import CMMMEventPayload as EventPayload
from ..frames import WireFrameData, VoiceFrame as NativeVoiceFrame
from ..control_registry import ControlID
from ..primitives import Q15_3_4, Q15_NEG_ONE

if TYPE_CHECKING:
    from ..library.preset_compiler import CompiledPreset
    from ..envelope_scheduler import EnvelopeScheduler
    from ..control_frame import CaptureMark, ControlFrame, EventType
    from ..frames import ParamFrame, VoiceFrame
    from ..vm.state import CMMMemory

from ..fixed import (
    sat16, mul_q15_sat, slew_q15, RSHIFT_RNE,
    Q15_ONE, Q15_HALF, Q15_SHIFT
)
from ..vm.harmonic_bank import LFOState, LFOTarget

# Import canonical primitives from primitives.py (WSE-STD-PRIMITIVES v0)
from ..primitives import (
    Regime,                    # Voice regime states (R0_IDLE through R6_RECOVERY)
    Intent,                    # Intent lane IDs (I_TONE through I_CALM)
    Family,                    # Operator family IDs
    Gesture,                   # Gesture input IDs
    REGIME_TABLE,              # Regime metadata
    INTENT_TABLE,              # Intent metadata
    BLOCK_SIZE,                # Samples per block (canonical)
    SAMPLE_RATE_HZ,            # Sample rate (canonical)
    Q16_ONE,                   # Q16 unity (FM6 mod_index scaling)
)

# =============================================================================
# CONSTANTS
# =============================================================================

SAMPLE_RATE = float(SAMPLE_RATE_HZ)
BLOCK_TIME = BLOCK_SIZE / SAMPLE_RATE  # ~1.33ms

NUM_INTENTS = 8
NUM_FAMILIES = 8
NUM_EVENTS = 8
NUM_TREES = 5
MAX_ARCS = 8          # Maximum arc modulators per voice
MAX_VOICES = 128      # Maximum polyphony target


# =============================================================================
# VOICE ENGINE OUTPUT (step_block return types)
# =============================================================================

@dataclass
class RenderBatchItem:
    """
    Single voice to render this block.

    Emitted for serviced voices and previously rendered voices being held.
    Consumed by the unified execution layer (Stages 3-10).
    """
    voice_id: int
    frame: 'VoiceFrame'       # VoiceFrame with all parameters
    mid_due: bool = True      # MID emission group due this block
    top_due: bool = True      # TOP emission group due this block
    degrade_level: int = 0    # Degradation level (0=none, 3=emergency)
    serviced: bool = True    # False: render held coefficients without stepping.

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'voice_id': self.voice_id,
            'mid_due': self.mid_due,
            'top_due': self.top_due,
            'degrade_level': self.degrade_level,
            'serviced': self.serviced,
        }


@dataclass
class VoiceEngineOutput:
    """
    Output from VoiceEngine.step_block().

    Per WSE-ARCHITECTURE-v1: VoiceEngine emits what to render,
    not audio. The unified execution layer handles rendering.
    """
    # Voices to render, including held voices with serviced=False.
    render_batch: List[RenderBatchItem] = field(default_factory=list)

    # Global parameters for Material/Safety (from ControlFrame.globals)
    param_frame: 'ParamFrame' = None

    # Scheduler telemetry (D state, drops, health bars)
    telemetry: Optional['HealthTelemetry'] = None

    # Block index
    block_index: int = 0

    # Consumed markers delivered once to the renderer/recorder.
    capture_marks: List['CaptureMark'] = field(default_factory=list)

    def __post_init__(self):
        if self.param_frame is None:
            # Import here to avoid circular import at module load
            from ..frames import ParamFrame
            self.param_frame = ParamFrame()


# =============================================================================
# DETERMINISTIC RNG (xorshift32 per spec Section 11)
# =============================================================================

# Single owner: vm/state.py (identical update; per spec Section 11 all
# randomness derives from VoiceState.rng_u32)
from ..vm.state import xorshift32


def rng_q15(state: int) -> Tuple[int, int]:
    """Get random Q15 value [0, 32767] from RNG state."""
    new_state, value = xorshift32(state)
    return new_state, value & 0x7FFF


def rng_bool(state: int, probability_q15: int) -> Tuple[int, bool]:
    """Get random bool with given probability (Q15) from RNG state."""
    new_state, value = rng_q15(state)
    return new_state, value < probability_q15

# =============================================================================
# TOWER LEVELS (L0-L4) - Hierarchical Coefficient Projections
# =============================================================================

class TowerLevel(IntEnum):
    """
    Tower levels for hierarchical coefficient projection.

    L0: Full 256-coefficient vector (no projection)
    L1: 64 coefficients (4x compression)
    L2: 16 coefficients (16x compression)
    L3: 4 coefficients (64x compression)
    L4: 1 coefficient (scalar - total energy)
    """
    L0_FULL = 0       # 256 coefficients - full resolution
    L1_COARSE = 1     # 64 coefficients - band-level
    L2_SPARSE = 2     # 16 coefficients - region-level
    L3_SKELETON = 3   # 4 coefficients - quadrant-level
    L4_SCALAR = 4     # 1 coefficient - total energy only


# Tower projection dimensions
TOWER_DIMS = {
    TowerLevel.L0_FULL: 256,
    TowerLevel.L1_COARSE: 64,
    TowerLevel.L2_SPARSE: 16,
    TowerLevel.L3_SKELETON: 4,
    TowerLevel.L4_SCALAR: 1,
}


# =============================================================================
# ENGINE MODE (Simple vs Full CMMM)
# =============================================================================

class EngineMode(IntEnum):
    """
    Voice Engine operating mode.

    SIMPLE: Native preset synthesis, optionally with an authored IR graph.
            Uses CompiledPreset for sound definition.
    FULL_CMMM: Full CMMM implementation with regime graph, intent mapping, etc.
               Uses InstrumentProgram for behavior definition.
    """
    SIMPLE = 0        # Native preset carrier/envelopes and optional IR graph
    FULL_CMMM = 1     # Full regime graph with hysteresis


# =============================================================================
# DEGRADATION LEVELS
# =============================================================================

class DegradationLevel(IntEnum):
    """
    Degradation levels for CPU/DSP load management.

    Higher levels disable more expensive operators.
    """
    NONE = 0          # Full processing
    LIGHT = 1         # Disable expensive transport ops
    HEAVY = 2         # Disable transport + topology
    EMERGENCY = 3     # Only tail stages (normalize + clamp)


# =============================================================================
# REGIME SYSTEM (R0-R6) - Imported from primitives.py
# =============================================================================
# Regime enum is now imported from ..primitives
# Canonical names: R0_IDLE, R1_ATTACK, R2_STABLE, R3_AIR, R4_GLITCH, R5_CATASTROPHE, R6_RECOVERY
#
# Legacy name aliases for backward compatibility during migration:
R1_ATTACK_IGNITION = Regime.R1_ATTACK
R2_STABLE_TONE = Regime.R2_STABLE
R3_GRAIN_AIR = Regime.R3_AIR
R4_GLITCH_CHATTER = Regime.R4_GLITCH
R6_RECOVERY_HEAL = Regime.R6_RECOVERY


# =============================================================================
# HYSTERESIS SPEC (per spec Section 7.2)
# =============================================================================

@dataclass
class HysteresisSpec:
    """
    Hysteresis configuration for edge transitions.

    Prevents oscillation between regimes by requiring thresholds
    to cross with margin before triggering.
    """
    # Threshold margin: must exceed threshold by this amount to enter
    enter_margin_q15: int = 1000  # ~3%

    # Exit margin: must drop below threshold minus this to exit
    exit_margin_q15: int = 500   # ~1.5%

    # Debounce: minimum blocks between transitions
    debounce_blocks: int = 4

    # Last transition block (for debounce tracking)
    last_transition_block: int = 0


# =============================================================================
# OVERLAY SPEC (per spec Section 7.2)
# =============================================================================

@dataclass
class OverlaySpec:
    """
    Temporary emission modification during transition.

    Applied for duration_blocks after transition starts.
    """
    # Duration of overlay effect
    duration_blocks: int = 8

    # Family slew tightening/loosening (multiplier Q15)
    family_slew_mod_q15: List[int] = field(default_factory=lambda: [Q15_ONE] * NUM_FAMILIES)

    # Event suppression mask (1 = suppress this event lane)
    event_suppress_mask: int = 0

    # Event enable mask (1 = force enable this event lane)
    event_enable_mask: int = 0

    # Tree blend ramp law override
    tree_blend_law: str = "linear"  # "linear", "ease_in", "ease_out", "s_curve"

    # Tail policy tightening (lower = more aggressive clamping)
    tail_cap_mod_q15: int = Q15_ONE


# =============================================================================
# EDGE SPEC - Transition Conditions (per spec Section 7.2)
# =============================================================================

class EdgeType(IntEnum):
    """Edge transition types per spec."""
    HARD = 0           # Immediate switch
    SOFT = 1           # Gradual blend
    CHAOTIC = 2        # Turbulent transition
    CORRIDOR_ENTRY = 3 # Entering a transitional regime
    RECOVERY = 4       # Returning to stability


@dataclass
class ConditionSpec:
    """
    Condition specification for edge activation.

    Supports threshold, range, and complex conditions on gesture/intent.
    """
    # Intent lane thresholds (must exceed all)
    intent_thresholds: Dict[int, int] = field(default_factory=dict)

    # Intent lane ranges (must be within range)
    intent_ranges: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    # Gesture conditions
    gate_required: Optional[bool] = None
    gate_rise_required: bool = False
    gate_fall_required: bool = False

    def check(self, gesture: Optional['GestureBasis'], intent: 'IntentVector', *,
              gate_rise: Optional[bool] = None, gate_fall: Optional[bool] = None) -> bool:
        """Check if condition is satisfied."""
        # Check gate conditions
        if self.gate_required is not None and (gesture is None or gesture.gate != self.gate_required):
            return False
        rise = getattr(gesture, 'gate_rise', False) if gate_rise is None else gate_rise
        fall = getattr(gesture, 'gate_fall', False) if gate_fall is None else gate_fall
        if self.gate_rise_required and not rise:
            return False
        if self.gate_fall_required and not fall:
            return False

        # Check intent thresholds
        for lane_id, threshold in self.intent_thresholds.items():
            if intent.lanes[lane_id] < threshold:
                return False

        # Check intent ranges
        for lane_id, (lo, hi) in self.intent_ranges.items():
            if not (lo <= intent.lanes[lane_id] <= hi):
                return False

        return True


@dataclass
class EdgeSpec:
    """
    Specification for a regime transition edge (per spec Section 7.2).

    Defines when and how transitions between regimes can occur.
    """
    from_regime: Regime
    to_regime: Regime

    # Edge type
    edge_type: EdgeType = EdgeType.SOFT

    # Condition (replaces simple thresholds)
    condition: ConditionSpec = field(default_factory=ConditionSpec)

    # Legacy: simple intent thresholds (for backward compat)
    intent_thresholds: Dict[int, int] = field(default_factory=dict)

    # Hysteresis configuration
    hysteresis: HysteresisSpec = field(default_factory=HysteresisSpec)

    # Minimum dwell time in source regime (blocks)
    min_dwell_blocks: int = 0

    # Duration of transition effect (for SOFT/CHAOTIC)
    duration_blocks: int = 8

    # Priority (higher = checked first)
    priority: int = 0

    # Arc modulator to apply during transition
    arc_id: Optional[int] = None

    # Overlay for temporary modification
    overlay: Optional[OverlaySpec] = None

    # -------------------------------------------------------------------------
    # Stochastic Transition Parameters (CMMM Enrichment Gap G3 - Phase P4)
    # -------------------------------------------------------------------------
    # Base probability of transition when threshold is met (Q15)
    # Q15_ONE (32767) = always transition (deterministic, default)
    # 0 = never transition (effectively disables edge)
    base_probability_q15: int = Q15_ONE

    # Intent influence on probability: lane_id -> influence_q15
    # Positive values increase probability when lane is high
    # Negative values decrease probability when lane is high
    stochastic_intent_influences: Dict[int, int] = field(default_factory=dict)

    def check_condition(self, intent: 'IntentVector', dwell_blocks: int,
                       gesture: Optional['GestureBasis'] = None, *,
                       gate_rise: Optional[bool] = None, gate_fall: Optional[bool] = None) -> bool:
        """Check if transition condition is met."""
        if dwell_blocks < self.min_dwell_blocks:
            return False

        # Every declared condition applies, including range-only and edge-only
        # guards. A missing gesture cannot satisfy a required gate level.
        if not self.condition.check(gesture, intent, gate_rise=gate_rise, gate_fall=gate_fall):
            return False

        # Fall back to legacy thresholds
        for lane_id, threshold in self.intent_thresholds.items():
            if intent.lanes[lane_id] < threshold:
                return False

        return True


# =============================================================================
# TEMPERAMENT MODULATION (CMMM-ENRICHMENT-DESIGN.md Phase P2)
# =============================================================================

def apply_temperament_to_hysteresis(
    hysteresis: HysteresisSpec,
    temperament: 'TemperamentVector'
) -> HysteresisSpec:
    """
    Modulate HysteresisSpec by temperament parameters.

    Per CMMM-ENRICHMENT-DESIGN.md §9.4:
    - hysteresis_q15: Scales enter/exit margins (high = sticky, wider margins)
    - reactivity_q15: Scales debounce inversely (high = faster response)

    Args:
        hysteresis: Original HysteresisSpec
        temperament: TemperamentVector from preset

    Returns:
        Modified HysteresisSpec with temperament applied
    """
    # Scale margins by temperament.hysteresis_q15
    # hysteresis=Q15_HALF (16384) → 1.0x (no change)
    # hysteresis=Q15_ONE (32767) → ~2.0x (very sticky)
    # hysteresis=0 → ~0.0x (no hysteresis, twitchy)
    margin_scale = temperament.hysteresis_q15
    new_enter = mul_q15_sat(hysteresis.enter_margin_q15 * 2, margin_scale)
    new_exit = mul_q15_sat(hysteresis.exit_margin_q15 * 2, margin_scale)

    # Scale debounce inversely by reactivity
    # reactivity=Q15_HALF (16384) → 1.0x (no change)
    # reactivity=Q15_ONE (32767) → ~0.5x (faster, less debounce)
    # reactivity=0 → ~2.0x (slow, more debounce)
    # Formula: debounce_scale = 2.0 - (reactivity / Q15_HALF)
    #   reactivity=0 → scale=2.0
    #   reactivity=Q15_HALF → scale=1.0
    #   reactivity=Q15_ONE → scale=0.0 (clamped to min)
    reactivity_normalized = (temperament.reactivity_q15 << 15) // Q15_HALF if Q15_HALF > 0 else 0
    # reactivity_normalized: 0 → 0, Q15_HALF → Q15_ONE, Q15_ONE → 2*Q15_ONE
    debounce_scale_q15 = max(0, (2 * Q15_ONE) - reactivity_normalized)
    # debounce_scale_q15: reactivity=0 → 2*Q15_ONE, Q15_HALF → Q15_ONE, Q15_ONE → 0
    new_debounce = max(1, (hysteresis.debounce_blocks * debounce_scale_q15) // Q15_ONE)

    return HysteresisSpec(
        enter_margin_q15=new_enter,
        exit_margin_q15=new_exit,
        debounce_blocks=new_debounce,
        last_transition_block=hysteresis.last_transition_block,
    )


def apply_temperament_to_edge(
    edge: EdgeSpec,
    temperament: 'TemperamentVector'
) -> EdgeSpec:
    """
    Create a modified EdgeSpec with temperament applied.

    Applies temperament modulation to:
    - Hysteresis margins and debounce
    - Minimum dwell time (scaled by stickiness)

    Note: This returns a new EdgeSpec; the original is not modified.

    Args:
        edge: Original EdgeSpec
        temperament: TemperamentVector from preset

    Returns:
        Modified EdgeSpec with temperament applied
    """
    # Modulate hysteresis
    new_hysteresis = apply_temperament_to_hysteresis(edge.hysteresis, temperament)

    # Scale min_dwell by stickiness
    # stickiness=Q15_HALF → 1.0x (no change)
    # stickiness=Q15_ONE → ~2.0x (stay longer before transitioning)
    # stickiness=0 → ~0.5x (transition sooner)
    # Formula: scale = 0.5 + (stickiness / Q15_ONE)
    #   stickiness=0 → scale=0.5
    #   stickiness=Q15_HALF → scale=1.0
    #   stickiness=Q15_ONE → scale=1.5
    dwell_scale_q15 = (Q15_HALF >> 1) + ((temperament.stickiness_q15 * Q15_ONE) // Q15_ONE)
    # Simplify: scale = 0.25*Q15_ONE + stickiness
    # Wait, let's be more explicit:
    # stickiness=0 → 0.5 * Q15_ONE = Q15_HALF
    # stickiness=Q15_HALF → 1.0 * Q15_ONE = Q15_ONE
    # stickiness=Q15_ONE → 1.5 * Q15_ONE = 1.5 * Q15_ONE
    # Formula: dwell_scale = Q15_HALF + stickiness (capped at 1.5*Q15_ONE)
    dwell_scale_q15 = Q15_HALF + temperament.stickiness_q15
    new_min_dwell = max(0, (edge.min_dwell_blocks * dwell_scale_q15) // Q15_ONE)

    # Create new edge with modified parameters
    return EdgeSpec(
        from_regime=edge.from_regime,
        to_regime=edge.to_regime,
        edge_type=edge.edge_type,
        condition=edge.condition,
        intent_thresholds=edge.intent_thresholds,
        hysteresis=new_hysteresis,
        min_dwell_blocks=new_min_dwell,
        duration_blocks=edge.duration_blocks,
        priority=edge.priority,
        arc_id=edge.arc_id,
        overlay=edge.overlay,
        # Preserve stochastic transition parameters (Gap G3 - Phase P4)
        base_probability_q15=edge.base_probability_q15,
        stochastic_intent_influences=edge.stochastic_intent_influences,
    )


# Type alias for TemperamentVector import (avoid circular)
try:
    from ..instrument_ir import TemperamentVector
except ImportError:
    TemperamentVector = None  # type: ignore


# =============================================================================
# STOCHASTIC TRANSITION KERNEL (CMMM Enrichment Gap G3 - Phase P4)
# =============================================================================
# Reference: CMMM-ENRICHMENT-DESIGN.md §3.4 (Stochastic Transition Kernel)
#
# The stochastic kernel computes P_θ(s'|s,u) where:
# - s = current regime
# - s' = target regime
# - u = intent vector
# - θ = temperament parameters
#
# Determinism is preserved by default:
# - base_probability_q15 = Q15_ONE → always transition if condition met
# - chaos_amount_q15 = 0 → no stochastic perturbation

def compute_transition_probability(
    edge: EdgeSpec,
    intent: 'IntentVector',
    memory: 'CMMMemory',
    temperament: 'TemperamentVector',
    prng_state: int
) -> Tuple[int, int]:
    """
    Compute transition probability P_θ(to_state | from_state, intent).

    This implements the stochastic transition kernel from CMMM theory.
    The probability is modulated by:
    1. Base probability from EdgeSpec
    2. Intent vector influences
    3. Memory state (tension increases escalation probability)
    4. Temperament (fatigue reduces probability, chaos adds noise)

    Args:
        edge: EdgeSpec being evaluated
        intent: Current IntentVector
        memory: CMMMemory state
        temperament: TemperamentVector from preset
        prng_state: Current PRNG state (for chaos perturbation)

    Returns:
        (probability_q15, new_prng_state) where probability is clamped to [0, Q15_ONE]
    """
    from ..vm.state import xorshift32

    # Start with base probability
    base_prob = edge.base_probability_q15

    # Modulate by intent influences
    for lane_id, influence in edge.stochastic_intent_influences.items():
        if lane_id < len(intent.lanes):
            # delta = intent[lane] * influence / Q15_ONE
            delta = mul_q15_sat(intent.lanes[lane_id], influence)
            base_prob = sat_add_q15(base_prob, delta)

    # Modulate by memory - tension increases escalation probability
    # Only apply if transitioning to a "higher" regime (escalation)
    if edge.to_regime.value > edge.from_regime.value:
        # High tension → more likely to escalate
        tension_factor = mul_q15_sat(
            memory.tension_q15, temperament.tension_influence_q15)
        # Add 25% of tension factor to probability
        base_prob = sat_add_q15(base_prob, tension_factor >> 2)

    # Modulate by fatigue - high fatigue reduces transition probability
    fatigue_factor = mul_q15_sat(
        memory.fatigue_q15, temperament.fatigue_sensitivity_q15)
    # Multiply by (1 - fatigue/2) to reduce probability
    fatigue_reduction = Q15_ONE - (fatigue_factor >> 1)
    if fatigue_factor:
        base_prob = mul_q15_sat(base_prob, fatigue_reduction)

    # Apply hysteresis - require sustained dwell before full probability
    # hysteresis_q15 >> 10 gives ~0-32 blocks threshold
    hysteresis_blocks = temperament.hysteresis_q15 >> 10
    if memory.blocks_since_transition < hysteresis_blocks:
        # Reduce probability 4x when below hysteresis threshold
        base_prob = base_prob >> 2

    # Add stochastic perturbation from temperament.chaos_amount_q15
    if temperament.chaos_amount_q15 > 0:
        prng_state, noise = xorshift32(prng_state)
        # Extract 15-bit noise value
        noise_q15 = (noise >> 17) & 0x7FFF  # 0-32767
        noise_scaled = mul_q15_sat(noise_q15, temperament.chaos_amount_q15)
        # Bipolar noise: [-chaos/2, +chaos/2]
        chaos_half = temperament.chaos_amount_q15 >> 1
        base_prob = sat_add_q15(base_prob, noise_scaled - chaos_half)

    # Clamp to valid Q15 range
    final_prob = max(0, min(Q15_ONE, base_prob))

    return (final_prob, prng_state)


def should_transition_stochastic(
    edge: EdgeSpec,
    intent: 'IntentVector',
    memory: 'CMMMemory',
    temperament: 'TemperamentVector',
    prng_state: int
) -> Tuple[bool, int]:
    """
    Make stochastic transition decision using PRNG.

    This function:
    1. Computes the transition probability
    2. Draws a random threshold from PRNG
    3. Returns True if threshold < probability

    For deterministic behavior (default):
    - base_probability_q15 = Q15_ONE and chaos_amount_q15 = 0
    - Probability will be Q15_ONE, so any threshold < Q15_ONE → always True

    Args:
        edge: EdgeSpec being evaluated
        intent: Current IntentVector
        memory: CMMMemory state
        temperament: TemperamentVector from preset
        prng_state: Current PRNG state

    Returns:
        (should_transition, new_prng_state)
    """
    from ..vm.state import xorshift32

    # Compute probability
    prob, prng_state = compute_transition_probability(
        edge, intent, memory, temperament, prng_state)

    # Draw random threshold
    prng_state, draw = xorshift32(prng_state)
    threshold = draw & 0x7FFF  # 0-32767

    # Transition if threshold < probability
    return (prob == Q15_ONE or threshold < prob, prng_state)


# Helper: saturating add for Q15 values
def sat_add_q15(a: int, b: int) -> int:
    """Saturating add for Q15 values."""
    result = a + b
    if result > Q15_ONE:
        return Q15_ONE
    if result < -Q15_ONE:
        return -Q15_ONE
    return result


# =============================================================================
# ARC SPEC - Transition Modulators
# =============================================================================

@dataclass
class ArcSpec:
    """
    Specification for arc modulators applied during transitions.

    Arcs are short-duration effects that smooth or emphasize transitions.
    """
    arc_id: int = 0
    name: str = "default"

    # Duration in blocks
    duration_blocks: int = 8

    # Family weight modifiers during arc (added to base)
    family_boost: List[int] = field(default_factory=lambda: [0] * NUM_FAMILIES)

    # Event triggers during arc
    arc_events: int = 0  # Bitmask of events to fire

    # Gain envelope modifier
    gain_curve: str = "linear"  # "linear", "ease_in", "ease_out", "s_curve"

    # Tree morph rate override (0 = use default)
    morph_rate_override: int = 0

    # Pattern IDs are local to this arc; the VoiceEngine owns their phase.
    event_patterns: Dict[int, EventPatternSpec] = field(default_factory=dict)

    def __post_init__(self):
        for pattern_id, pattern in self.event_patterns.items():
            if not isinstance(pattern_id, int) or not 0 <= pattern_id <= 65535:
                raise ValueError('Arc event pattern IDs must be uint16')
            if not isinstance(pattern, EventPatternSpec):
                raise TypeError('Arc patterns must be EventPatternSpec instances')


# Default arc specs
ARC_ATTACK_BURST = ArcSpec(
    arc_id=1,
    name="attack_burst",
    duration_blocks=4,
    family_boost=[0, Q15_HALF // 2, 0, 0, 0, 0, Q15_HALF, 0],
    arc_events=0x01,  # E_IMPULSE
    gain_curve="ease_in",
)

ARC_CATASTROPHE_ENTRY = ArcSpec(
    arc_id=2,
    name="catastrophe_entry",
    duration_blocks=16,
    family_boost=[0, 0, 0, 0, Q15_HALF, 0, Q15_HALF, Q15_ONE],
    arc_events=0x50,  # E_CUSP | E_FOLD
    gain_curve="s_curve",
    morph_rate_override=256,
)

ARC_RECOVERY_SMOOTH = ArcSpec(
    arc_id=3,
    name="recovery_smooth",
    duration_blocks=32,
    family_boost=[0, 0, 0, 0, 0, Q15_HALF, 0, 0],
    arc_events=0x80,  # E_SHEAF
    gain_curve="ease_out",
)

# Arc registry
ARC_SPECS: Dict[int, ArcSpec] = {
    1: ARC_ATTACK_BURST,
    2: ARC_CATASTROPHE_ENTRY,
    3: ARC_RECOVERY_SMOOTH,
}


# =============================================================================
# FAMILY POLICY SPEC (per spec Section 8.1)
# =============================================================================

class FamilySelectionRule(IntEnum):
    """How to compute family weights."""
    FIXED = 0          # Use fixed base weights
    BY_INTENT = 1      # Linear map from intent vector
    BY_GESTURE = 2     # Direct from gesture
    BY_ARC = 3         # Modulated by active arc


@dataclass
class FamilyPolicySpec:
    """
    Computes w_family[8] from intents + regime params.

    Per spec Section 8.1.
    """
    # Enable mask (1 bit per family)
    enable_mask: int = 0xFF

    # Base weights (Q15)
    base_q15: List[int] = field(default_factory=lambda: [0] * NUM_FAMILIES)

    # Linear mapping matrix W[8][8]: w_out = base + W @ intent
    # Stored as flattened list [row0, row1, ..., row7]
    W_q15: List[int] = field(default_factory=lambda: [0] * (NUM_FAMILIES * NUM_INTENTS))

    # Bias (added after matrix multiply)
    bias_q15: List[int] = field(default_factory=lambda: [0] * NUM_FAMILIES)

    # Caps (maximum values)
    caps_q15: List[int] = field(default_factory=lambda: [Q15_ONE] * NUM_FAMILIES)

    # Slew time constants (blocks)
    slew_tc_blocks: List[int] = field(default_factory=lambda: [4] * NUM_FAMILIES)

    # Selection rule
    selection_rule: FamilySelectionRule = FamilySelectionRule.BY_INTENT

    def compute(self, intent: 'IntentVector', current: List[int]) -> List[int]:
        """
        Compute family weights from intent vector.

        Applies: base + W @ intent + bias, then clamp to caps, then slew.
        """
        result = list(self.base_q15)

        if self.selection_rule == FamilySelectionRule.BY_INTENT:
            # Matrix multiply: W[8x8] @ intent[8]
            for f in range(NUM_FAMILIES):
                acc = 0
                for i in range(NUM_INTENTS):
                    w = self.W_q15[f * NUM_INTENTS + i]
                    acc += mul_q15_sat(w, intent.lanes[i])
                result[f] = sat16(result[f] + acc + self.bias_q15[f])

        # Apply caps
        for f in range(NUM_FAMILIES):
            result[f] = min(result[f], self.caps_q15[f])
            # Apply enable mask
            if not (self.enable_mask & (1 << f)):
                result[f] = 0

        # Apply slew
        for f in range(NUM_FAMILIES):
            tc = max(1, self.slew_tc_blocks[f])
            delta = (result[f] - current[f]) // tc
            result[f] = current[f] + delta

        return result


# =============================================================================
# TREE POLICY SPEC (per spec Section 8.2)
# =============================================================================

class TreeSelectionRule(IntEnum):
    """How to select trees."""
    FIXED = 0          # Use default trees
    BY_INTENT = 1      # Select based on intent lanes
    BY_GESTURE = 2     # Select based on gesture (position)
    BY_ARC = 3         # Modulated by active arc


class TreeBlendRule(IntEnum):
    """How to compute tree blend."""
    FIXED = 0          # Fixed blend value
    GESTURE_POSITION = 1  # Blend from gesture.position
    INTENT_WARP = 2    # Blend from intent.warp
    ARC = 3            # Blend from arc modulator


@dataclass
class TreePolicySpec:
    """
    Tree selection and blending policy (per spec Section 8.2).

    Output: (tree_a_id, tree_b_id, tree_blend_q15)
    """
    # Allowed tree pairs (if empty, all pairs allowed)
    allowed_pairs: List[Tuple[int, int]] = field(default_factory=list)

    # Selection rule
    selection_rule: TreeSelectionRule = TreeSelectionRule.BY_INTENT

    # Blend rule
    blend_rule: TreeBlendRule = TreeBlendRule.INTENT_WARP

    # Blend slew time constant
    blend_slew_tc: int = 8

    # Default trees
    default_tree_a: int = 0
    default_tree_b: int = 0

    # Fixed blend (when blend_rule == FIXED)
    fixed_blend_q15: int = 0

    def select_trees(self, intent: 'IntentVector', gesture: 'GestureBasis',
                    current_a: int, current_b: int) -> Tuple[int, int]:
        """Select tree pair based on policy."""
        if self.selection_rule == TreeSelectionRule.FIXED:
            return self.default_tree_a, self.default_tree_b

        elif self.selection_rule == TreeSelectionRule.BY_INTENT:
            # Use intent to select trees
            # High tone -> harmonic (1), high air -> texture (2),
            # high glitch -> transient (3), high calm -> drone (4)
            scores = [
                intent.tone,      # Tree 1: Harmonic
                intent.air,       # Tree 2: Texture
                intent.snap,      # Tree 3: Transient
                intent.calm,      # Tree 4: Drone
            ]
            best = max(range(len(scores)), key=lambda i: scores[i])
            target_a = best + 1  # Trees are 1-indexed

            # Second tree based on warp for morphing
            if intent.warp > Q15_HALF:
                target_b = (target_a % 4) + 1
            else:
                target_b = target_a

            return target_a, target_b

        elif self.selection_rule == TreeSelectionRule.BY_GESTURE:
            # Use gesture position to select
            pos = gesture.position_q15
            tree = (pos * 4) // Q15_ONE
            return min(4, max(1, tree + 1)), self.default_tree_b

        return current_a, current_b

    def compute_blend(self, intent: 'IntentVector', gesture: 'GestureBasis',
                     current_blend: int) -> int:
        """Compute tree blend value based on policy."""
        if self.blend_rule == TreeBlendRule.FIXED:
            target = self.fixed_blend_q15
        elif self.blend_rule == TreeBlendRule.GESTURE_POSITION:
            target = gesture.position_q15
        elif self.blend_rule == TreeBlendRule.INTENT_WARP:
            target = intent.warp
        else:
            target = current_blend

        # Slew toward target
        delta = (target - current_blend) // max(1, self.blend_slew_tc)
        return current_blend + delta


# =============================================================================
# EVENT POLICY SPEC (per spec Section 8.3)
# =============================================================================

# Event policy types are re-exported from voice.event_policy.


# =============================================================================
# REGIME SPEC - Per-Regime Policies
# =============================================================================

@dataclass
class RegimeSpec:
    """
    Complete specification for a regime's behavior.

    Defines family weights, tree policy, event policy, and tail config.
    """
    regime: Regime
    name: str = ""

    # Family Policy: base weights and caps
    family_base: List[int] = field(default_factory=lambda: [0] * NUM_FAMILIES)
    family_cap: List[int] = field(default_factory=lambda: [Q15_ONE] * NUM_FAMILIES)
    family_enable_mask: int = 0xFF  # Bitmask of enabled families

    # Tree Policy
    preferred_tree: int = 0
    tree_lock: bool = False  # If True, ignore intent-based tree selection
    morph_rate: int = 512

    # Event Policy
    event_enable_mask: int = 0xFF
    event_probability: List[int] = field(default_factory=lambda: [Q15_HALF] * NUM_EVENTS)

    # Tail Policy overrides; None inherits the instrument's default.
    normalize_curve: Optional[str] = None  # "flat", "pink", "bright"
    fineclamp_cap_ratio: Optional[float] = None
    slew_max_delta: Optional[int] = None

    # Tower level for this regime
    tower_level: TowerLevel = TowerLevel.L0_FULL


# Default regime specs (using canonical names from primitives.py)
REGIME_SPECS: Dict[Regime, RegimeSpec] = {
    Regime.R0_IDLE: RegimeSpec(
        regime=Regime.R0_IDLE,
        name="Idle",
        family_enable_mask=0x00,  # All disabled
        event_enable_mask=0x00,
        tower_level=TowerLevel.L4_SCALAR,
    ),
    Regime.R1_ATTACK: RegimeSpec(
        regime=Regime.R1_ATTACK,
        name="Attack",
        family_base=[0, Q15_HALF, 0, Q15_HALF // 2, 0, 0, Q15_HALF, 0],
        family_cap=[0, Q15_ONE, Q15_HALF, Q15_ONE, Q15_HALF, Q15_ONE, Q15_ONE, 0],
        preferred_tree=3,  # Transient
        event_enable_mask=0x07,  # Impulse, noise, pulse
        normalize_curve="bright",
        tower_level=TowerLevel.L1_COARSE,
    ),
    Regime.R2_STABLE: RegimeSpec(
        regime=Regime.R2_STABLE,
        name="Stable Tone",
        family_base=[0, Q15_HALF, 0, Q15_HALF, 0, Q15_HALF, 0, 0],
        family_cap=[0, Q15_ONE, Q15_HALF, Q15_ONE, Q15_HALF, Q15_ONE, Q15_HALF, 0],
        preferred_tree=1,  # Harmonic
        event_enable_mask=0x00,
        normalize_curve="pink",
        tower_level=TowerLevel.L0_FULL,
    ),
    Regime.R3_AIR: RegimeSpec(
        regime=Regime.R3_AIR,
        name="Grain/Air",
        family_base=[0, Q15_HALF // 2, Q15_HALF, Q15_HALF // 2, 0, Q15_HALF // 2, Q15_HALF // 2, 0],
        family_cap=[0, Q15_ONE, Q15_ONE, Q15_ONE, Q15_HALF, Q15_ONE, Q15_ONE, 0],
        preferred_tree=2,  # Texture
        event_enable_mask=0x02,  # Noise
        normalize_curve="flat",
        tower_level=TowerLevel.L1_COARSE,
    ),
    Regime.R4_GLITCH: RegimeSpec(
        regime=Regime.R4_GLITCH,
        name="Glitch/Chatter",
        family_base=[0, 0, Q15_HALF, Q15_HALF // 2, Q15_HALF, 0, Q15_HALF, 0],
        family_cap=[0, Q15_ONE, Q15_ONE, Q15_ONE, Q15_ONE, Q15_ONE, Q15_ONE, Q15_HALF],
        preferred_tree=2,  # Texture
        event_enable_mask=0x2B,  # Impulse, noise, gate, swallowtail
        normalize_curve="pink",
        tower_level=TowerLevel.L2_SPARSE,
    ),
    Regime.R5_CATASTROPHE: RegimeSpec(
        regime=Regime.R5_CATASTROPHE,
        name="Catastrophe",
        family_base=[0, 0, Q15_HALF, 0, Q15_HALF, 0, Q15_HALF, Q15_ONE],
        family_cap=[0, Q15_ONE, Q15_ONE, Q15_ONE, Q15_ONE, Q15_ONE, Q15_ONE, Q15_ONE],
        family_enable_mask=0xFF,
        preferred_tree=2,  # Texture
        tree_lock=False,
        morph_rate=256,  # Slow morph during chaos
        event_enable_mask=0xF0,  # Cusp, swallowtail, fold, sheaf
        normalize_curve="pink",
        fineclamp_cap_ratio=0.3,
        tower_level=TowerLevel.L2_SPARSE,
    ),
    Regime.R6_RECOVERY: RegimeSpec(
        regime=Regime.R6_RECOVERY,
        name="Recovery",
        family_base=[0, Q15_HALF // 2, 0, Q15_HALF // 2, 0, Q15_ONE, 0, 0],
        family_cap=[0, Q15_HALF, Q15_HALF // 2, Q15_HALF, 0, Q15_ONE, Q15_HALF // 2, 0],
        preferred_tree=4,  # Drone - stable
        morph_rate=1024,  # Fast morph to stability
        event_enable_mask=0x80,  # Sheaf only
        normalize_curve="pink",
        fineclamp_cap_ratio=0.6,
        tower_level=TowerLevel.L1_COARSE,
    ),
}


# =============================================================================
# REGIME GRAPH - Full Transition Specification
# =============================================================================

@dataclass
class RegimeGraph:
    """
    Complete regime graph with all edges and arc specifications.
    """
    edges: List[EdgeSpec] = field(default_factory=list)
    regime_specs: Dict[Regime, RegimeSpec] = field(default_factory=dict)
    arc_specs: Dict[int, ArcSpec] = field(default_factory=dict)

    def get_valid_transitions(self, from_regime: Regime) -> List[EdgeSpec]:
        """Get all edges leaving a regime, sorted by priority."""
        edges = [e for e in self.edges if e.from_regime == from_regime]
        return sorted(edges, key=lambda e: -e.priority)

    def check_transition(self, from_regime: Regime, intent: 'IntentVector',
                        dwell_blocks: int,
                        temperament: 'TemperamentVector' = None,
                        memory: 'CMMMemory' = None,
                        prng_state: int = None, *, gesture: Optional['GestureBasis'] = None,
                        gate_rise: Optional[bool] = None,
                        gate_fall: Optional[bool] = None) -> Tuple[Optional[Regime], int]:
        """
        Check if any transition should occur.

        Supports both deterministic (default) and stochastic transitions:
        - Deterministic: base_probability_q15 = Q15_ONE, chaos_amount_q15 = 0
        - Stochastic: Uses PRNG to make probabilistic decision

        Args:
            from_regime: Current regime
            intent: Intent vector
            dwell_blocks: Blocks in current regime
            temperament: Optional temperament for edge modulation
            memory: Optional CMMMemory for stochastic kernel
            prng_state: Optional PRNG state for stochastic transitions

        Returns:
            (target_regime, new_prng_state) where target_regime is None if no transition
        """
        # Default PRNG state if not provided
        if prng_state is None:
            prng_state = 0x12345678

        for edge in self.get_valid_transitions(from_regime):
            # Apply temperament modulation if provided
            if temperament is not None:
                modulated_edge = apply_temperament_to_edge(edge, temperament)
            else:
                modulated_edge = edge

            # Check threshold condition first
            if not modulated_edge.check_condition(intent, dwell_blocks, gesture,
                                                   gate_rise=gate_rise, gate_fall=gate_fall):
                continue

            # If threshold met, apply stochastic kernel if enabled
            # Stochastic is enabled when: base_probability < Q15_ONE OR chaos_amount > 0
            is_stochastic = (
                modulated_edge.base_probability_q15 < Q15_ONE or
                any(modulated_edge.stochastic_intent_influences.values()) or
                (temperament is not None and temperament.chaos_amount_q15 > 0)
            )

            if is_stochastic:
                from ..vm.state import CMMMemory
                # Missing optional modulation inputs must not turn a chance
                # (including zero) into an unconditional transition.
                effective_memory = memory if memory is not None else CMMMemory(blocks_since_transition=dwell_blocks)
                effective_temperament = temperament if temperament is not None else TemperamentVector(
                    hysteresis_q15=0, tension_influence_q15=0, fatigue_sensitivity_q15=0)
                # Use stochastic transition decision
                should_transition, prng_state = should_transition_stochastic(
                    modulated_edge, intent, effective_memory, effective_temperament, prng_state
                )
                if should_transition:
                    return (modulated_edge.to_regime, prng_state)
            else:
                # Deterministic: threshold met = transition
                return (modulated_edge.to_regime, prng_state)

        return (None, prng_state)

    def get_arc(self, from_regime: Regime, to_regime: Regime) -> Optional[ArcSpec]:
        """Get arc spec for a transition."""
        for edge in self.edges:
            if edge.from_regime == from_regime and edge.to_regime == to_regime:
                if edge.arc_id is not None:
                    return self.arc_specs.get(edge.arc_id)
        return None


def build_default_regime_graph() -> RegimeGraph:
    """Build the default regime graph with all transitions."""
    graph = RegimeGraph(
        regime_specs=REGIME_SPECS,
        arc_specs=ARC_SPECS,
    )

    # Define edges with conditions
    graph.edges = [
        # From R0_IDLE
        EdgeSpec(
            from_regime=Regime.R0_IDLE,
            to_regime=Regime.R1_ATTACK,
            intent_thresholds={},  # Triggered by note_on
            priority=10,
            arc_id=1,  # Attack burst
        ),

        # From R1_ATTACK_IGNITION
        EdgeSpec(
            from_regime=Regime.R1_ATTACK,
            to_regime=Regime.R2_STABLE,
            intent_thresholds={I_TONE: Q15_HALF},
            min_dwell_blocks=4,
            priority=10,
        ),
        EdgeSpec(
            from_regime=Regime.R1_ATTACK,
            to_regime=Regime.R3_AIR,
            intent_thresholds={I_AIR: int(Q15_ONE * 0.6)},
            min_dwell_blocks=4,
            priority=8,
        ),
        EdgeSpec(
            from_regime=Regime.R1_ATTACK,
            to_regime=Regime.R4_GLITCH,
            intent_thresholds={I_GLITCH: int(Q15_ONE * 0.7)},
            min_dwell_blocks=4,
            priority=6,
        ),

        # From R2_STABLE_TONE
        # Note: R2->R0 is triggered by note_off(), not by intent conditions
        # So we don't include it here (it's handled explicitly)
        EdgeSpec(
            from_regime=Regime.R2_STABLE,
            to_regime=Regime.R3_AIR,
            intent_thresholds={I_AIR: int(Q15_ONE * 0.6)},
            min_dwell_blocks=8,
            priority=5,
        ),
        EdgeSpec(
            from_regime=Regime.R2_STABLE,
            to_regime=Regime.R4_GLITCH,
            intent_thresholds={I_GLITCH: int(Q15_ONE * 0.6)},
            min_dwell_blocks=8,
            priority=5,
        ),
        EdgeSpec(
            from_regime=Regime.R2_STABLE,
            to_regime=Regime.R5_CATASTROPHE,
            intent_thresholds={I_GLITCH: int(Q15_ONE * 0.8), I_WARP: int(Q15_ONE * 0.6)},
            min_dwell_blocks=16,
            priority=3,
            arc_id=2,  # Catastrophe entry
        ),

        # From R3_GRAIN_AIR
        # Note: R3->R0 is triggered by note_off(), not by intent conditions
        EdgeSpec(
            from_regime=Regime.R3_AIR,
            to_regime=Regime.R2_STABLE,
            intent_thresholds={I_TONE: int(Q15_ONE * 0.7), I_AIR: 0},
            min_dwell_blocks=8,
            priority=5,
        ),
        EdgeSpec(
            from_regime=Regime.R3_AIR,
            to_regime=Regime.R4_GLITCH,
            intent_thresholds={I_GLITCH: int(Q15_ONE * 0.6)},
            min_dwell_blocks=8,
            priority=5,
        ),
        EdgeSpec(
            from_regime=Regime.R3_AIR,
            to_regime=Regime.R5_CATASTROPHE,
            intent_thresholds={I_GLITCH: int(Q15_ONE * 0.8), I_WARP: int(Q15_ONE * 0.6)},
            min_dwell_blocks=16,
            priority=3,
            arc_id=2,
        ),

        # From R4_GLITCH_CHATTER
        # Note: R4->R0 is triggered by note_off(), not by intent conditions
        EdgeSpec(
            from_regime=Regime.R4_GLITCH,
            to_regime=Regime.R2_STABLE,
            intent_thresholds={I_CALM: int(Q15_ONE * 0.6)},
            min_dwell_blocks=8,
            priority=5,
        ),
        EdgeSpec(
            from_regime=Regime.R4_GLITCH,
            to_regime=Regime.R3_AIR,
            intent_thresholds={I_AIR: int(Q15_ONE * 0.7)},
            min_dwell_blocks=8,
            priority=5,
        ),
        EdgeSpec(
            from_regime=Regime.R4_GLITCH,
            to_regime=Regime.R5_CATASTROPHE,
            intent_thresholds={I_GLITCH: int(Q15_ONE * 0.9)},
            min_dwell_blocks=16,
            priority=3,
            arc_id=2,
        ),

        # From R5_CATASTROPHE
        EdgeSpec(
            from_regime=Regime.R5_CATASTROPHE,
            to_regime=Regime.R6_RECOVERY,
            intent_thresholds={},  # Automatic after duration
            min_dwell_blocks=64,
            priority=10,
            arc_id=3,  # Recovery smooth
        ),

        # From R6_RECOVERY_HEAL
        # Note: R6->R0 is triggered by note_off(), not by intent conditions
        EdgeSpec(
            from_regime=Regime.R6_RECOVERY,
            to_regime=Regime.R2_STABLE,
            intent_thresholds={I_CALM: int(Q15_ONE * 0.5)},
            min_dwell_blocks=32,
            priority=5,
        ),
        EdgeSpec(
            from_regime=Regime.R6_RECOVERY,
            to_regime=Regime.R3_AIR,
            intent_thresholds={I_AIR: int(Q15_ONE * 0.5)},
            min_dwell_blocks=32,
            priority=5,
        ),
    ]

    return graph


# Regime transition matrix: from_regime -> valid_to_regimes (using canonical names)
REGIME_TRANSITIONS = {
    Regime.R0_IDLE: {Regime.R1_ATTACK},
    Regime.R1_ATTACK: {Regime.R2_STABLE, Regime.R3_AIR, Regime.R4_GLITCH},
    Regime.R2_STABLE: {Regime.R0_IDLE, Regime.R3_AIR, Regime.R4_GLITCH, Regime.R5_CATASTROPHE},
    Regime.R3_AIR: {Regime.R0_IDLE, Regime.R2_STABLE, Regime.R4_GLITCH, Regime.R5_CATASTROPHE},
    Regime.R4_GLITCH: {Regime.R0_IDLE, Regime.R2_STABLE, Regime.R3_AIR, Regime.R5_CATASTROPHE},
    Regime.R5_CATASTROPHE: {Regime.R6_RECOVERY},
    Regime.R6_RECOVERY: {Regime.R0_IDLE, Regime.R2_STABLE, Regime.R3_AIR},
}


# =============================================================================
# GESTURE BASIS (8 fields)
# =============================================================================

@dataclass
class GestureBasis:
    """
    Physical gesture inputs - raw controller state.

    All values normalized to Q1.15 range (-32768 to +32767)
    except gate which is boolean.
    """
    gate: bool = False           # Note on/off
    pressure_q15: int = 0        # Aftertouch / bow pressure (0 to Q15_ONE)
    position_q15: int = 0        # Position on surface (-Q15_ONE to +Q15_ONE)
    speed_q15: int = 0           # Movement speed (0 to Q15_ONE)
    tilt_q15: int = 0            # Surface tilt (-Q15_ONE to +Q15_ONE)
    chaos_q15: int = 0           # Random/noise input (0 to Q15_ONE)
    pitch_q15: int = 0           # Pitch bend (-Q15_ONE to +Q15_ONE)
    pan_q15: int = 0             # Stereo position (-Q15_ONE to +Q15_ONE)

    @classmethod
    def from_note_on(cls, velocity_q15: int = Q15_HALF, pitch_q15: int = 0,
                     pan_q15: int = 0) -> 'GestureBasis':
        """Create gesture from basic note-on parameters."""
        return cls(
            gate=True,
            pressure_q15=velocity_q15,
            position_q15=0,
            speed_q15=velocity_q15,  # Initial speed from velocity
            tilt_q15=0,
            chaos_q15=0,
            pitch_q15=pitch_q15,
            pan_q15=pan_q15,
        )


# =============================================================================
# INTENT VECTOR (8 lanes)
# =============================================================================

# Intent lane indices (use Intent enum from primitives.py)
# These aliases are for backward compatibility with code using bare I_* names
I_TONE = Intent.I_TONE
I_BOW = Intent.I_BOW
I_AIR = Intent.I_AIR
I_GLITCH = Intent.I_GLITCH
I_WARP = Intent.I_WARP
I_SNAP = Intent.I_SNAP
I_GLUE = Intent.I_GLUE
I_CALM = Intent.I_CALM


@dataclass
class IntentVector:
    """
    Semantic intent lanes - what the voice "wants to do".

    Computed from GestureBasis via mapping functions.
    All values in Q1.15 (0 to Q15_ONE, representing 0.0 to 1.0).
    """
    lanes: List[int] = field(default_factory=lambda: [0] * NUM_INTENTS)

    def __getitem__(self, idx: int) -> int:
        return self.lanes[idx]

    def __setitem__(self, idx: int, val: int):
        self.lanes[idx] = val

    @property
    def tone(self) -> int: return self.lanes[I_TONE]
    @property
    def bow(self) -> int: return self.lanes[I_BOW]
    @property
    def air(self) -> int: return self.lanes[I_AIR]
    @property
    def glitch(self) -> int: return self.lanes[I_GLITCH]
    @property
    def warp(self) -> int: return self.lanes[I_WARP]
    @property
    def snap(self) -> int: return self.lanes[I_SNAP]
    @property
    def glue(self) -> int: return self.lanes[I_GLUE]
    @property
    def calm(self) -> int: return self.lanes[I_CALM]


def default_gesture_to_intent(g: GestureBasis) -> IntentVector:
    """
    Default mapping from GestureBasis to IntentVector.

    This is the spec's "Gesture → Intent Mapping".
    """
    intent = IntentVector()

    # I_TONE: high when gate is on and pressure is moderate
    if g.gate:
        intent.lanes[I_TONE] = mul_q15_sat(g.pressure_q15, Q15_HALF) + Q15_HALF // 2

    # I_BOW: direct from pressure
    intent.lanes[I_BOW] = g.pressure_q15

    # I_AIR: increases with speed and low pressure
    air_from_speed = g.speed_q15 >> 1
    air_from_low_pressure = max(0, Q15_HALF - g.pressure_q15)
    intent.lanes[I_AIR] = sat16(air_from_speed + (air_from_low_pressure >> 1))

    # I_GLITCH: from chaos and extreme tilt
    glitch_from_chaos = g.chaos_q15
    glitch_from_tilt = abs(g.tilt_q15) >> 1
    intent.lanes[I_GLITCH] = sat16(glitch_from_chaos + glitch_from_tilt) >> 1

    # I_WARP: from pitch bend and extreme position
    intent.lanes[I_WARP] = sat16(abs(g.pitch_q15) + (abs(g.position_q15) >> 2))

    # I_SNAP: high on gate-on with high speed
    if g.gate and g.speed_q15 > Q15_HALF:
        intent.lanes[I_SNAP] = g.speed_q15
    else:
        intent.lanes[I_SNAP] = 0

    # I_GLUE: inverse of chaos - coherence desire
    intent.lanes[I_GLUE] = max(0, Q15_ONE - g.chaos_q15)

    # I_CALM: low pressure and speed = desire for calm
    calm_pressure = max(0, Q15_HALF - g.pressure_q15)
    calm_speed = max(0, Q15_HALF - g.speed_q15)
    intent.lanes[I_CALM] = (calm_pressure + calm_speed) >> 1

    return intent


# =============================================================================
# FAMILY WEIGHTS (F0-F7)
# =============================================================================

# Family indices (use Family enum from primitives.py)
# These aliases are for backward compatibility with code using bare F_* names
F_RESERVED = Family.F_RESERVED
F_COLOR = Family.F_COLOR
F_TRANSPORT = Family.F_TRANSPORT
F_LOCAL = Family.F_LOCAL
F_TOPOLOGY = Family.F_TOPOLOGY
F_GLUE = Family.F_GLUE
F_EVENTS = Family.F_EVENTS
F_CATASTROPHE = Family.F_CATASTROPHE


def softplus_q15(x: int, k: int = 8) -> int:
    """
    Softplus activation in Q15: log(1 + exp(k*x)) / k

    Approximation: for x > 0, returns ~x; for x < 0, returns ~0.
    k controls sharpness (higher = sharper transition).
    """
    if x > Q15_ONE:
        return Q15_ONE
    elif x < -Q15_ONE:
        return 0
    else:
        # Simple approximation: max(0, x) + soft knee
        if x >= 0:
            return x
        else:
            # Soft curve for negative x
            return max(0, (x + Q15_HALF) >> 2)


def compute_family_weights(intent: IntentVector, regime: Regime) -> List[int]:
    """
    Compute family weights from intent vector and regime.

    Returns 8 weights in Q1.15 (0 to Q15_ONE).
    """
    weights = [0] * NUM_FAMILIES

    # Base mappings (intent -> family)
    # F_COLOR: tone + bow
    weights[F_COLOR] = (intent.tone + intent.bow) >> 1

    # F_TRANSPORT: warp + air
    weights[F_TRANSPORT] = (intent.warp + intent.air) >> 1

    # F_LOCAL: bow + tone
    weights[F_LOCAL] = (intent.bow + intent.tone) >> 1

    # F_TOPOLOGY: glitch + warp
    weights[F_TOPOLOGY] = (intent.glitch + intent.warp) >> 1

    # F_GLUE: glue intent directly
    weights[F_GLUE] = intent.glue

    # F_EVENTS: snap + glitch
    weights[F_EVENTS] = (intent.snap + intent.glitch) >> 1

    # F_CATASTROPHE: only active in R5
    if regime == Regime.R5_CATASTROPHE:
        weights[F_CATASTROPHE] = Q15_ONE
    else:
        weights[F_CATASTROPHE] = 0

    # Regime modifiers
    if regime == Regime.R1_ATTACK:
        # Boost snap/events during attack
        weights[F_EVENTS] = sat16(weights[F_EVENTS] + Q15_HALF)
        weights[F_COLOR] = sat16(weights[F_COLOR] + Q15_HALF // 2)

    elif regime == Regime.R2_STABLE:
        # Boost tone families, reduce chaos
        weights[F_COLOR] = sat16(weights[F_COLOR] + Q15_HALF // 2)
        weights[F_LOCAL] = sat16(weights[F_LOCAL] + Q15_HALF // 2)
        weights[F_GLUE] = sat16(weights[F_GLUE] + Q15_HALF // 2)
        weights[F_TOPOLOGY] = weights[F_TOPOLOGY] >> 1

    elif regime == Regime.R3_AIR:
        # Boost air/transport
        weights[F_TRANSPORT] = sat16(weights[F_TRANSPORT] + Q15_HALF)
        weights[F_EVENTS] = sat16(weights[F_EVENTS] + Q15_HALF // 2)

    elif regime == Regime.R4_GLITCH:
        # Boost topology and events
        weights[F_TOPOLOGY] = sat16(weights[F_TOPOLOGY] + Q15_HALF)
        weights[F_EVENTS] = sat16(weights[F_EVENTS] + Q15_HALF)

    elif regime == Regime.R6_RECOVERY:
        # Boost calm, reduce chaos
        weights[F_GLUE] = sat16(weights[F_GLUE] + Q15_HALF)
        weights[F_CATASTROPHE] = 0
        weights[F_TOPOLOGY] = weights[F_TOPOLOGY] >> 1
        weights[F_EVENTS] = weights[F_EVENTS] >> 1

    # Apply softplus activation
    weights = [softplus_q15(w) for w in weights]

    return weights


# =============================================================================
# EVENT LANES (E0-E7)
# =============================================================================

# Event lane indices
E_IMPULSE = 0      # Single-sample impulse
E_NOISE = 1        # Noise burst
E_PULSE = 2        # Pressure ramp
E_GATE = 3         # Gate mask
E_CUSP = 4         # Cusp injection
E_SWALLOWTAIL = 5  # Swallowtail mask
E_FOLD = 6         # Fold pressure
E_SHEAF = 7        # Sheaf projection


def compute_events(intent: IntentVector, regime: Regime, block_count: int) -> EventPayload:
    """
    Compute event payloads from intent and regime.

    Some events fire probabilistically based on intent strength.
    """
    events = EventPayload()

    # Impulse: fires on high snap
    if intent.snap > Q15_HALF:
        events.impulse_band = 3  # Mid band
        events.impulse_strength_q15 = intent.snap
        events.active_mask |= (1 << E_IMPULSE)

    # Noise: fires when air is high
    if intent.air > Q15_HALF:
        events.noise_band = 5  # Fine band
        events.noise_amount_q15 = intent.air >> 1
        events.active_mask |= (1 << E_NOISE)

    # Pulse: during attack regime
    if regime == Regime.R1_ATTACK:
        events.pulse_pressure_q15 = Q15_HALF
        events.active_mask |= (1 << E_PULSE)

    # Gate: can be modulated by glitch
    if intent.glitch > Q15_HALF:
        # Random gate pattern
        events.gate_mask = (block_count * 17) & 0xFF  # Pseudo-random
        events.active_mask |= (1 << E_GATE)

    # Cusp: during catastrophe
    if regime == Regime.R5_CATASTROPHE:
        events.cusp_amount_q15 = Q15_HALF
        events.active_mask |= (1 << E_CUSP)

    # Swallowtail: glitch and catastrophe corridors.
    if regime in (Regime.R4_GLITCH, Regime.R5_CATASTROPHE):
        events.swallowtail_bands = 0x3C  # Mid bands
        events.active_mask |= (1 << E_SWALLOWTAIL)

    # Fold: during catastrophe
    if regime == Regime.R5_CATASTROPHE:
        events.fold_pressure_q15 = Q15_ONE
        events.active_mask |= (1 << E_FOLD)

    # Sheaf: catastrophe containment and recovery.
    if regime in (Regime.R5_CATASTROPHE, Regime.R6_RECOVERY):
        events.sheaf_threshold_q15 = Q15_HALF
        events.active_mask |= (1 << E_SHEAF)

    return events


# =============================================================================
# TREE MORPH CONTROL (Option A)
# =============================================================================

@dataclass
class TreeMorphState:
    """
    Tree morph state using Option A: tree_a_id, tree_b_id, tree_blend_q15.

    All tree selection happens here. Symbol Engine maintains dual buffers
    only when blend is non-zero.
    """
    tree_a_id: int = 0         # Current primary tree (0-4)
    tree_b_id: int = 0         # Target tree during morph
    blend_q15: int = 0         # 0 = pure A, Q15_ONE = pure B

    # Morph timing
    morph_rate: int = 512      # Blend increment per block
    is_morphing: bool = False


def select_tree_from_intent(intent: IntentVector, regime: Regime) -> int:
    """
    Select tree ID (0-4) based on intent vector.

    Trees:
    - 0: Balanced (uniform resolution)
    - 1: Harmonic (wider lows, detailed highs)
    - 2: Texture (high-band resolution)
    - 3: Transient (mid-band emphasis)
    - 4: Drone (low-band detail)
    """
    # Weight each tree based on intent
    scores = [0] * NUM_TREES

    # Tree 0 (Balanced): default, moderate everything
    scores[0] = Q15_HALF

    # Tree 1 (Harmonic): high tone, moderate bow
    scores[1] = (intent.tone + intent.bow) >> 1

    # Tree 2 (Texture): high air, some glitch
    scores[2] = (intent.air + (intent.glitch >> 1))

    # Tree 3 (Transient): high snap, warp
    scores[3] = (intent.snap + intent.warp) >> 1

    # Tree 4 (Drone): high calm, low snap
    scores[4] = intent.calm

    # Regime biases
    if regime == Regime.R1_ATTACK:
        scores[3] += Q15_HALF  # Transient
    elif regime == Regime.R2_STABLE:
        scores[1] += Q15_HALF  # Harmonic
    elif regime == Regime.R3_AIR:
        scores[2] += Q15_HALF  # Texture
    elif regime == Regime.R4_GLITCH:
        scores[2] += Q15_HALF // 2
        scores[3] += Q15_HALF // 2
    elif regime == Regime.R5_CATASTROPHE:
        scores[2] += Q15_HALF  # Texture for chaos
    elif regime == Regime.R6_RECOVERY:
        scores[4] += Q15_HALF  # Drone for stability

    return int(np.argmax(scores))


def update_tree_morph(state: TreeMorphState, target_tree: int) -> TreeMorphState:
    """
    Update tree morph state toward target tree.

    Returns updated state.
    """
    if target_tree == state.tree_a_id and not state.is_morphing:
        # Already at target, no change
        return state

    if target_tree != state.tree_b_id and not state.is_morphing:
        # Start new morph
        state.tree_b_id = target_tree
        state.blend_q15 = 0
        state.is_morphing = True

    if state.is_morphing:
        # Advance blend
        state.blend_q15 = min(Q15_ONE, state.blend_q15 + state.morph_rate)

        if state.blend_q15 >= Q15_ONE:
            # Morph complete: promote B to A
            state.tree_a_id = state.tree_b_id
            state.blend_q15 = 0
            state.is_morphing = False

    return state


# =============================================================================
# VOICEBANK CONTAINER (per spec Section 2)
# =============================================================================

@dataclass
class ControlMap:
    """
    Control source to gesture mapping.

    Maps MIDI CCs, aftertouch, pitch bend, etc. to GestureBasis fields.
    """
    # CC mappings: cc_number -> (gesture_field, scale_q15, offset_q15)
    cc_mappings: Dict[int, Tuple[str, int, int]] = field(default_factory=dict)

    # Pitch bend range in semitones
    pitch_bend_range: int = 2

    # Aftertouch destination
    aftertouch_dest: str = "pressure_q15"
    aftertouch_scale_q15: int = Q15_ONE

    # Mod wheel destination
    modwheel_dest: str = "chaos_q15"
    modwheel_scale_q15: int = Q15_ONE


@dataclass
class VoiceBank:
    """
    Container for instrument programs, presets, and control maps.

    Per spec Section 2: VoiceBank contains:
    - Multiple InstrumentPrograms
    - Control maps (MIDI CC to gesture)
    - Preset definitions
    - Global settings
    """
    bank_id: int = 0
    name: str = "Default Bank"

    # Programs indexed by program_id
    programs: Dict[int, 'InstrumentProgram'] = field(default_factory=dict)

    # Control map (shared or per-program)
    control_map: ControlMap = field(default_factory=ControlMap)

    # Preset definitions: preset_id -> program_id + parameter overrides
    presets: Dict[int, Dict] = field(default_factory=dict)

    # Global bank settings
    master_tune_cents: int = 0
    transpose_semitones: int = 0
    velocity_curve: str = "linear"  # linear, soft, hard, fixed

    # Polyphony settings
    max_polyphony: int = MAX_VOICES
    voice_stealing: str = "oldest"  # oldest, quietest, lowest, highest

    def get_program(self, program_id: int) -> Optional['InstrumentProgram']:
        """Get program by ID, or None if not found."""
        return self.programs.get(program_id)

    def add_program(self, program: 'InstrumentProgram') -> None:
        """Add or replace a program in the bank."""
        self.programs[program.program_id] = program

    def get_preset(self, preset_id: int) -> Optional[Dict]:
        """Get preset definition by ID."""
        return self.presets.get(preset_id)


# =============================================================================
# INSTRUMENT PROGRAMS (per spec Section 3)
# =============================================================================

@dataclass
class InstrumentProgram:
    """
    Complete instrument program specification per spec Section 3.

    Defines:
    - VoiceModel: Family policy, tree policy, event policy
    - GestureMap: Gesture to intent mapping
    - RegimeGraph: State machine with edges and arcs
    - Tail policies for output processing
    """
    program_id: int = 0
    name: str = "Default"

    # Tower level compliance (L0-L4)
    tower_level: TowerLevel = TowerLevel.L0_FULL

    # ==========================================================================
    # GESTURE MAP (per spec Section 3.2)
    # ==========================================================================
    gesture_to_intent: Callable[[GestureBasis], IntentVector] = field(
        default_factory=lambda: default_gesture_to_intent
    )

    # ==========================================================================
    # VOICE MODEL POLICIES (per spec Sections 3.3-3.5)
    # ==========================================================================

    # Family policy: how intent maps to family weights
    family_policy: FamilyPolicySpec = field(default_factory=FamilyPolicySpec)

    # Tree policy: how to select and blend trees
    tree_policy: TreePolicySpec = field(default_factory=TreePolicySpec)

    # Event policy: how to generate events per lane
    # None selects the built-in regime/arc rules. An explicit empty policy
    # disables events, so authored silence is distinct from an omitted policy.
    event_policy: Optional[EventPolicySpec] = None
    # Frozen-wire authored events; distinct from the eight semantic CMMM lanes.
    authored_events: Optional[CompiledEventProgram] = None

    # ==========================================================================
    # REGIME GRAPH (per spec Section 7)
    # ==========================================================================

    # Regime specifications (per-regime settings)
    regime_specs: Dict[Regime, RegimeSpec] = field(default_factory=dict)

    # Regime graph (edges and arcs)
    regime_graph: Optional[RegimeGraph] = None

    # Temperament parameters (CMMM transition behavior, Phase P2)
    # Default is None = neutral temperament (no modulation of EdgeSpec)
    temperament: Optional['TemperamentVector'] = None

    # Legacy: simple regime transitions (used if regime_graph is None)
    attack_regime: Regime = Regime.R1_ATTACK
    sustain_regime: Regime = Regime.R2_STABLE
    release_regime: Regime = Regime.R0_IDLE

    # ==========================================================================
    # ADSR ENVELOPE (per spec Section 5)
    # ==========================================================================
    attack_blocks: int = 8
    decay_blocks: int = 32
    sustain_level_q15: int = Q15_HALF + Q15_HALF // 2  # 0.75
    release_blocks: int = 64

    # ==========================================================================
    # TAIL POLICY (per spec Section 8)
    # ==========================================================================
    normalize_curve: str = "pink"      # flat, pink, bright
    fineclamp_cap_ratio: float = 0.5   # Max deviation ratio
    slew_max_delta: int = 512          # Max block-to-block coefficient change

    # ==========================================================================
    # LEGACY COMPATIBILITY FIELDS
    # ==========================================================================

    # Family weight modifiers (legacy - use family_policy.W_q15 instead)
    family_mods: List[int] = field(default_factory=lambda: [Q15_ONE] * NUM_FAMILIES)

    # Tree preferences (legacy - use tree_policy instead)
    preferred_tree: int = 0
    tree_morph_rate: int = 512

    # Event modifiers (legacy - use event_policy instead)
    event_probability: List[int] = field(default_factory=lambda: [Q15_HALF] * NUM_EVENTS)

    # ==========================================================================
    # EMPHASIS OVERRIDES (from preset behavior section)
    # ==========================================================================

    # Per-regime emphasis weights from the preset's behavior section.
    # Maps Regime enum value (int) → emphasis dict (e.g., {tone: 0.9, energy: 1.0}).
    # When present, emphasis_to_family_base() is used to derive family_base
    # instead of the hardcoded RegimeSpec defaults.
    per_regime_emphasis: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def get_regime_spec(self, regime: Regime) -> RegimeSpec:
        """Get spec for regime, or create default."""
        if regime in self.regime_specs:
            return self.regime_specs[regime]
        return RegimeSpec(regime=regime)


def make_glitch_violin_program() -> InstrumentProgram:
    """
    Program 1001: Glitch Violin

    Expressive string-like with glitch potential.
    High pressure = clean tone, low pressure = air/glitch.
    """
    def glitch_violin_intent(g: GestureBasis) -> IntentVector:
        intent = default_gesture_to_intent(g)
        # Boost bow response
        intent.lanes[I_BOW] = sat16(intent.lanes[I_BOW] + (g.pressure_q15 >> 1))
        # Glitch increases with chaos and extreme tilt
        intent.lanes[I_GLITCH] = sat16(intent.lanes[I_GLITCH] + abs(g.tilt_q15))
        return intent

    return InstrumentProgram(
        program_id=1001,
        name="Glitch Violin",
        gesture_to_intent=glitch_violin_intent,
        attack_regime=Regime.R1_ATTACK,
        sustain_regime=Regime.R2_STABLE,
        family_mods=[Q15_ONE, Q15_ONE, Q15_HALF, Q15_ONE, Q15_HALF, Q15_ONE, Q15_HALF, Q15_ONE],
        preferred_tree=1,  # Harmonic
        attack_blocks=4,
        decay_blocks=24,
        sustain_level_q15=int(Q15_ONE * 0.8),
        release_blocks=48,
    )


def make_shepard_random_program() -> InstrumentProgram:
    """
    Program 1002: Shepard Random

    Continuously morphing pitch illusion with random elements.
    Position controls pitch cycling, chaos adds variation.
    """
    def shepard_intent(g: GestureBasis) -> IntentVector:
        intent = default_gesture_to_intent(g)
        # Constant warp from position cycling
        intent.lanes[I_WARP] = sat16(abs(g.position_q15) + Q15_HALF // 2)
        # Air from chaos
        intent.lanes[I_AIR] = sat16(g.chaos_q15 + intent.lanes[I_AIR])
        return intent

    return InstrumentProgram(
        program_id=1002,
        name="Shepard Random",
        gesture_to_intent=shepard_intent,
        attack_regime=Regime.R3_AIR,  # Start textured
        sustain_regime=Regime.R3_AIR,
        family_mods=[Q15_ONE, Q15_HALF, Q15_ONE, Q15_HALF, Q15_ONE, Q15_HALF, Q15_ONE, Q15_HALF],
        preferred_tree=2,  # Texture
        tree_morph_rate=256,  # Slow morphs
        attack_blocks=16,
        decay_blocks=64,
        sustain_level_q15=int(Q15_ONE * 0.6),
        release_blocks=128,
    )


def make_turbulent_choir_program() -> InstrumentProgram:
    """
    Program 1003: Turbulent Choir

    Choral texture with chaotic modulation.
    Multiple implicit "voices" through coefficient structure.
    """
    def choir_intent(g: GestureBasis) -> IntentVector:
        intent = default_gesture_to_intent(g)
        # Boost glue for coherence
        intent.lanes[I_GLUE] = sat16(intent.lanes[I_GLUE] + Q15_HALF)
        # Air adds breathiness
        intent.lanes[I_AIR] = sat16(g.speed_q15 + (g.chaos_q15 >> 2))
        # Moderate glitch for texture
        intent.lanes[I_GLITCH] = sat16(g.chaos_q15 >> 1)
        return intent

    return InstrumentProgram(
        program_id=1003,
        name="Turbulent Choir",
        gesture_to_intent=choir_intent,
        attack_regime=Regime.R1_ATTACK,
        sustain_regime=Regime.R3_AIR,  # Sustain is textured
        family_mods=[Q15_ONE, Q15_ONE, Q15_ONE, Q15_ONE, Q15_HALF, Q15_ONE, Q15_ONE, Q15_HALF],
        preferred_tree=2,  # Texture
        attack_blocks=24,
        decay_blocks=48,
        sustain_level_q15=int(Q15_ONE * 0.7),
        release_blocks=96,
    )


def make_static_organ_program() -> InstrumentProgram:
    """
    Program 1004: Static Organ

    Stable organ-like tone, minimal modulation.
    Quick attack/release, constant sustain.
    """
    def organ_intent(g: GestureBasis) -> IntentVector:
        intent = IntentVector()
        # Strong tone, minimal variation
        intent.lanes[I_TONE] = Q15_ONE if g.gate else 0
        intent.lanes[I_BOW] = g.pressure_q15
        intent.lanes[I_GLUE] = int(Q15_ONE * 0.9)  # High coherence
        intent.lanes[I_CALM] = Q15_HALF
        # Minimal air/glitch
        intent.lanes[I_AIR] = g.chaos_q15 >> 3
        intent.lanes[I_GLITCH] = 0
        return intent

    return InstrumentProgram(
        program_id=1004,
        name="Static Organ",
        gesture_to_intent=organ_intent,
        attack_regime=Regime.R1_ATTACK,
        sustain_regime=Regime.R2_STABLE,
        family_mods=[Q15_ONE, Q15_ONE, Q15_HALF // 2, Q15_HALF, 0, Q15_ONE, 0, Q15_ONE],
        preferred_tree=4,  # Drone
        tree_morph_rate=1024,  # Fast morph (instant)
        attack_blocks=2,
        decay_blocks=4,
        sustain_level_q15=Q15_ONE,  # Full sustain
        release_blocks=8,
    )


# Program registry
INSTRUMENT_PROGRAMS: Dict[int, InstrumentProgram] = {
    1001: make_glitch_violin_program(),
    1002: make_shepard_random_program(),
    1003: make_turbulent_choir_program(),
    1004: make_static_organ_program(),
}


# =============================================================================
# VOICE STATE - Full Snapshotable State for Replay
# =============================================================================

class VoicePhase(IntEnum):
    """Voice lifecycle phases (internal ADSR)."""
    FREE = 0
    ATTACK = 1
    DECAY = 2
    SUSTAIN = 3
    RELEASE = 4


@dataclass
class ArcState:
    """
    Active arc modulator state.

    Tracks progress through an arc transition.
    """
    active: bool = False
    arc_id: int = 0
    progress_blocks: int = 0
    total_blocks: int = 0

    def is_complete(self) -> bool:
        return self.progress_blocks >= self.total_blocks

    def get_progress_q15(self) -> int:
        """Get arc progress as Q15 (0 = start, Q15_ONE = complete)."""
        if self.total_blocks == 0:
            return Q15_ONE
        return min(Q15_ONE, (self.progress_blocks * Q15_ONE) // self.total_blocks)

    def step(self) -> bool:
        """Advance arc by one block. Returns True if still active."""
        if not self.active:
            return False
        self.progress_blocks += 1
        if self.is_complete():
            self.active = False
            return False
        return True


def _new_transition_memory():
    from ..vm.state import CMMMemory
    return CMMMemory()


@dataclass
class VoiceState:
    """
    Complete per-voice state - FULLY SNAPSHOTABLE FOR REPLAY.

    All fields needed to reconstruct voice behavior at any block.
    This is the "VoiceState" from the CMMM specification.

    Invariant: Given VoiceState S and GestureBasis G,
               step(S, G) produces identical (VoiceState', VoiceFrame).
    """
    voice_id: int = 0

    # Lifecycle (Semi-Markov: non-geometric durations)
    phase: VoicePhase = VoicePhase.FREE
    phase_block: int = 0        # Blocks in current phase
    note_start_pending: bool = False
    release_start_q15: int = 0

    # Current gesture (Controller input)
    gesture: GestureBasis = field(default_factory=GestureBasis)

    # Computed intent (Gesture -> Intent mapping)
    intent: IntentVector = field(default_factory=IntentVector)

    # Regime FSM state
    regime: Regime = Regime.R0_IDLE
    regime_blocks: int = 0      # Blocks in current regime (dwell time)
    previous_regime: Regime = Regime.R0_IDLE  # For arc lookup

    # Arc modulator state
    arc: ArcState = field(default_factory=ArcState)

    # Family weights (output of regime + intent -> weights)
    family_weights: List[int] = field(default_factory=lambda: [0] * NUM_FAMILIES)

    # Event payload
    events: EventPayload = field(default_factory=EventPayload)

    # Tree morph state (Option A: tree_a, tree_b, blend)
    tree_morph: TreeMorphState = field(default_factory=TreeMorphState)

    # Envelope values
    gain_q15: int = 0

    # Degradation level (from CPU monitor)
    degradation: DegradationLevel = DegradationLevel.NONE

    # Tower level (from regime spec)
    tower_level: TowerLevel = TowerLevel.L0_FULL

    # Instrument program ID (program object stored separately)
    program_id: int = 0

    # Block counter (monotonic, for deterministic timing)
    block_count: int = 0

    # Deterministic RNG state (per spec Section 11)
    # All randomness derives from this; update is xorshift32
    rng_u32: int = 0x12345678

    previous_gate: bool = False
    gate_rise_pending: bool = False
    gate_fall_pending: bool = False
    authored_event_state: AuthoredEventState = field(default_factory=AuthoredEventState)
    authored_regime_state: AuthoredRegimeState = field(default_factory=AuthoredRegimeState)
    transition_memory: 'CMMMemory' = field(default_factory=_new_transition_memory)

    # ADSR envelope state (per spec Section 5)
    env_phase: VoicePhase = VoicePhase.FREE
    env_level_q15: int = 0
    env_counter: int = 0

    # Intent dynamics (derivative for smooth transitions)
    dI_q15: List[int] = field(default_factory=lambda: [0] * NUM_INTENTS)

    # Arc phase accumulators (for LFO/pattern arcs)
    arc_phase_q16: List[int] = field(default_factory=lambda: [0] * MAX_ARCS)

    # Note info
    note_u8: int = 60  # MIDI note number
    cents_i16: int = 0  # Fine tuning in cents
    detune_cents: int = 0  # Additional detune from controller (-2400 to +2400)

    # Oscillator controls (from Pico2 controller)
    eta_q15: int = Q15_HALF  # Dry/wet blend (0=dry, 32767=wet)
    harmonics_q15: int = 0  # Harmonic mix (0=fundamental only)
    master_gain_q15: int = Q15_3_4  # Master gain override (~0.75)

    # Activity level (for voice stealing)
    activity_q15: int = 0

    # Send levels (for aux buses)
    sendA_q15: int = 0
    sendB_q15: int = 0

    # Overlay state (during transitions)
    overlay_active: bool = False
    overlay_blocks_remaining: int = 0
    overlay_spec: Optional[OverlaySpec] = None

    # LFO definitions (from preset, set when voice is allocated)
    # When populated, tick_lfos() is called automatically in step_voice()
    lfos: List[LFOState] = field(default_factory=list)

    # ==========================================================================
    # Simple Mode fields (for preset-only synthesis without full CMMM)
    # ==========================================================================

    # Engine mode: SIMPLE uses preset, FULL_CMMM uses InstrumentProgram
    engine_mode: int = 1  # Default to FULL_CMMM (EngineMode.FULL_CMMM)

    # CompiledPreset reference (for Simple mode)
    # In Simple mode, sound parameters come from preset instead of InstrumentProgram
    compiled_preset: Optional['CompiledPreset'] = None

    # Beta ramping (from VoiceFrameEmitter, used in Simple mode)
    beta_current_q15: int = 0
    beta_start_q15: int = 0
    beta_target_q15: int = Q15_3_4
    beta_ramp_blocks: int = 10

    # Pitch offset from control (in cents)
    pitch_offset_cents: int = 0
    controls: VoiceControls = field(default_factory=VoiceControls)
    channel: int = 0
    unison: Optional[UnisonMember] = None

    # Velocity (Q15)
    velocity_q15: int = Q15_ONE

    # Matrix envelopes: scheduler, targets, optional velocity/key scaling/base spec.
    # Initialized from CompiledPreset.envelope_bindings in Simple mode
    envelope_matrix: Dict[str, tuple] = field(
        default_factory=dict
    )

    # Main amplitude envelope
    amp_envelope: Optional['EnvelopeScheduler'] = None
    amp_env_base_spec: Optional['ADSRSpec'] = None
    amp_key_scale_rate: float = 0.0
    amp_velocity_sens_q15: int = Q15_ONE

    # LFO Phase Storage:
    # LFO phases are stored in arc_phase_q16[arc_slot] for unified time ownership.
    # LFO definitions (LFOState objects) come from the preset and are stored in
    # the lfos field above. Phase accumulators are in arc_phase_q16.
    # This aligns with WSE-ARCHITECTURE-v1: Voice Engine owns musical time.
    #
    # Helper methods for LFO phase management:

    def tick_lfos(self, lfos: List[LFOState], block_samples: int = BLOCK_SIZE) -> None:
        """
        Advance all LFOs by one block using arc_phase_q16 storage.

        Args:
            lfos: List of LFOState objects (from preset)
            block_samples: Number of samples in block
        """
        from .lfo_clocks import advance_lfo_clocks
        advance_lfo_clocks(lfos, self.arc_phase_q16, block_samples)

    def handle_note_on_lfos(self, lfos: List[LFOState], is_legato: bool = False) -> None:
        """
        Handle note-on event for LFO sync modes.

        Args:
            lfos: List of LFOState objects (from preset)
            is_legato: True if this is a legato note transition
        """
        from .lfo_clocks import gate_lfo_clocks
        gate_lfo_clocks(lfos, self.arc_phase_q16, is_legato=is_legato)

    # Snapshot/replay support
    def snapshot(self) -> Dict:
        """Create a detached versioned checkpoint of all per-voice time state.

        CMMM programs are resolved separately by ID during restore. Output DSP
        and engine-global state require their own checkpoints.
        """
        from .snapshot_codec import snapshot_voice
        return snapshot_voice(self)

    def _snapshot_legacy(self) -> Dict:
        """Legacy CMMM representation for compatibility fixtures, not full replay."""
        return {
            'voice_id': self.voice_id,
            'phase': int(self.phase),
            'phase_block': self.phase_block,
            'note_start_pending': self.note_start_pending,
            'release_start_q15': self.release_start_q15,
            'channel': self.channel,
            'pitch_offset_cents': self.pitch_offset_cents,
            'controls': {
                'values': dict(self.controls.values),
                'detune_override': self.controls.detune_override,
                'eta_override': self.controls.eta_override,
            },
            'gesture': {
                'gate': self.gesture.gate,
                'pressure_q15': self.gesture.pressure_q15,
                'position_q15': self.gesture.position_q15,
                'speed_q15': self.gesture.speed_q15,
                'tilt_q15': self.gesture.tilt_q15,
                'chaos_q15': self.gesture.chaos_q15,
                'pitch_q15': self.gesture.pitch_q15,
                'pan_q15': self.gesture.pan_q15,
            },
            'intent': list(self.intent.lanes),
            'regime': int(self.regime),
            'regime_blocks': self.regime_blocks,
            'previous_regime': int(self.previous_regime),
            'arc': {
                'active': self.arc.active,
                'arc_id': self.arc.arc_id,
                'progress_blocks': self.arc.progress_blocks,
                'total_blocks': self.arc.total_blocks,
            },
            'family_weights': list(self.family_weights),
            'tree_morph': {
                'tree_a_id': self.tree_morph.tree_a_id,
                'tree_b_id': self.tree_morph.tree_b_id,
                'blend_q15': self.tree_morph.blend_q15,
                'morph_rate': self.tree_morph.morph_rate,
                'is_morphing': self.tree_morph.is_morphing,
            },
            'gain_q15': self.gain_q15,
            'degradation': int(self.degradation),
            'tower_level': int(self.tower_level),
            'program_id': self.program_id,
            'block_count': self.block_count,
            # New fields for full spec compliance (Section 5, 11)
            'rng_u32': self.rng_u32,
            'previous_gate': self.previous_gate,
            'gate_rise_pending': self.gate_rise_pending,
            'gate_fall_pending': self.gate_fall_pending,
            'env_phase': int(self.env_phase),
            'env_level_q15': self.env_level_q15,
            'env_counter': self.env_counter,
            'dI_q15': list(self.dI_q15),
            'arc_phase_q16': list(self.arc_phase_q16),
            'note_u8': self.note_u8,
            'cents_i16': self.cents_i16,
            'activity_q15': self.activity_q15,
            'sendA_q15': self.sendA_q15,
            'sendB_q15': self.sendB_q15,
            'overlay_active': self.overlay_active,
            'overlay_blocks_remaining': self.overlay_blocks_remaining,
            'overlay_spec': {
                'duration_blocks': self.overlay_spec.duration_blocks,
                'family_slew_mult_q15': self.overlay_spec.family_slew_mult_q15,
                'tree_blend_override_q15': self.overlay_spec.tree_blend_override_q15,
                'gain_mult_q15': self.overlay_spec.gain_mult_q15,
            } if self.overlay_spec else None,
        }

    @classmethod
    def restore(cls, snapshot: Dict,
                programs: Optional[Dict[int, InstrumentProgram]] = None) -> 'VoiceState':
        """
        Restore voice state from a snapshot.

        Enables deterministic replay from any saved state.
        All fields needed for replay are restored.
        """
        if not isinstance(snapshot, dict):
            raise ValueError('Voice snapshot must be a dictionary')
        if 'snapshot_version' in snapshot:
            from .snapshot_codec import restore_voice
            return restore_voice(cls, snapshot, programs)
        state = cls()
        state.voice_id = snapshot['voice_id']
        state.phase = VoicePhase(snapshot['phase'])
        state.phase_block = snapshot['phase_block']
        state.note_start_pending = snapshot.get('note_start_pending', False)
        state.release_start_q15 = snapshot.get('release_start_q15', snapshot['gain_q15'])
        state.channel = snapshot.get('channel', 0)
        state.pitch_offset_cents = snapshot.get('pitch_offset_cents', 0)
        controls = snapshot.get('controls', {})
        state.controls = VoiceControls(
            values={int(key): value for key, value in controls.get('values', {}).items()},
            detune_override=controls.get('detune_override'),
            eta_override=controls.get('eta_override'),
        )

        g = snapshot['gesture']
        state.gesture = GestureBasis(
            gate=g['gate'],
            pressure_q15=g['pressure_q15'],
            position_q15=g['position_q15'],
            speed_q15=g['speed_q15'],
            tilt_q15=g['tilt_q15'],
            chaos_q15=g['chaos_q15'],
            pitch_q15=g['pitch_q15'],
            pan_q15=g['pan_q15'],
        )

        state.intent = IntentVector(lanes=list(snapshot['intent']))
        state.regime = Regime(snapshot['regime'])
        state.regime_blocks = snapshot['regime_blocks']
        state.previous_regime = Regime(snapshot['previous_regime'])

        a = snapshot['arc']
        state.arc = ArcState(
            active=a['active'],
            arc_id=a['arc_id'],
            progress_blocks=a['progress_blocks'],
            total_blocks=a['total_blocks'],
        )

        state.family_weights = list(snapshot['family_weights'])

        tm = snapshot['tree_morph']
        state.tree_morph = TreeMorphState(
            tree_a_id=tm['tree_a_id'],
            tree_b_id=tm['tree_b_id'],
            blend_q15=tm['blend_q15'],
            morph_rate=tm['morph_rate'],
            is_morphing=tm['is_morphing'],
        )

        state.gain_q15 = snapshot['gain_q15']
        state.degradation = DegradationLevel(snapshot['degradation'])
        state.tower_level = TowerLevel(snapshot['tower_level'])
        state.program_id = snapshot['program_id']
        if isinstance(state, VoiceStateWithProgram):
            registry = INSTRUMENT_PROGRAMS if programs is None else programs
            if state.program_id in registry:
                state.program = registry[state.program_id]
            elif state.program_id != 0:
                raise ValueError(f'Program {state.program_id} is required to restore this voice')
        state.block_count = snapshot['block_count']

        # Restore new fields (with backwards compatibility defaults)
        state.rng_u32 = snapshot.get('rng_u32', 0x12345678)
        state.previous_gate = snapshot.get('previous_gate', state.gesture.gate)
        state.gate_rise_pending = snapshot.get('gate_rise_pending', state.note_start_pending and state.gesture.gate)
        state.gate_fall_pending = snapshot.get('gate_fall_pending', False)
        state.env_phase = VoicePhase(snapshot.get('env_phase', int(VoicePhase.FREE)))
        state.env_level_q15 = snapshot.get('env_level_q15', 0)
        state.env_counter = snapshot.get('env_counter', 0)
        state.dI_q15 = list(snapshot.get('dI_q15', [0] * NUM_INTENTS))
        state.arc_phase_q16 = list(snapshot.get('arc_phase_q16', [0] * MAX_ARCS))
        state.note_u8 = snapshot.get('note_u8', 60)
        state.cents_i16 = snapshot.get('cents_i16', 0)
        state.activity_q15 = snapshot.get('activity_q15', 0)
        state.sendA_q15 = snapshot.get('sendA_q15', 0)
        state.sendB_q15 = snapshot.get('sendB_q15', 0)
        state.overlay_active = snapshot.get('overlay_active', False)
        state.overlay_blocks_remaining = snapshot.get('overlay_blocks_remaining', 0)

        overlay_data = snapshot.get('overlay_spec')
        if overlay_data:
            state.overlay_spec = OverlaySpec(
                duration_blocks=overlay_data['duration_blocks'],
                family_slew_mult_q15=overlay_data['family_slew_mult_q15'],
                tree_blend_override_q15=overlay_data['tree_blend_override_q15'],
                gain_mult_q15=overlay_data['gain_mult_q15'],
            )
        else:
            state.overlay_spec = None

        return state


# Legacy compatibility: keep program reference in state
@dataclass
class VoiceStateWithProgram(VoiceState):
    """VoiceState with embedded program reference for legacy compatibility."""
    program: InstrumentProgram = field(default_factory=InstrumentProgram)


# =============================================================================
# VOICEFRAME OUTPUT ABI (CMMM Specification)
# =============================================================================

@dataclass
class VoiceFrameABI:
    """
    VoiceFrame output per the CMMM specification ABI.

    This is the Mealy output: what Voice Engine emits per block per voice.
    Symbol Engine and Wavelet Engine consume this as stateless input.

    ABI Layout (for FPGA/C compatibility):
    - program_id: u16
    - voice_id: u8
    - regime_id: u8
    - gain_q15: i16
    - pan_q15: i16
    - w_family_q15[8]: i16[8]
    - event_mask: u8
    - event_strengths[8]: u8[8]
    - tree_a_id: u8
    - tree_b_id: u8
    - tree_blend_q15: u16
    - degradation_level: u8
    - tower_level: u8
    - tail_policy: u8
    - flags: u8
    """
    # Identity
    program_id: int = 0          # Instrument program ID
    voice_id: int = 0            # Voice slot ID

    # Regime state
    regime_id: int = 0           # Current regime (0-6)

    # Amplitude control
    gain_q15: int = 0            # Voice gain envelope (Q1.15)
    pan_q15: int = 0             # Stereo pan (-Q15 to +Q15)

    # Family weights for Symbol Engine (8 x Q1.15)
    w_family_q15: List[int] = field(default_factory=lambda: [0] * NUM_FAMILIES)

    # Event lanes (bitmask + strengths)
    event_mask: int = 0          # Active event lanes (bitmask)
    event_strengths: List[int] = field(default_factory=lambda: [0] * NUM_EVENTS)

    # Tree control (Option A)
    tree_a_id: int = 0           # Primary tree ID (0-4)
    tree_b_id: int = 0           # Secondary tree ID (during morph)
    tree_blend_q15: int = 0      # Blend factor (0 = pure A, Q15_ONE = pure B)

    # Processing hints
    degradation_level: DegradationLevel = DegradationLevel.NONE
    tower_level: TowerLevel = TowerLevel.L0_FULL

    # Tail policy (from RegimeSpec)
    normalize_curve: str = "pink"
    fineclamp_cap_ratio: float = 0.5
    slew_max_delta: int = 512

    # Musical envelopes (derived from intent)
    tilt_env_q15: int = Q15_HALF
    density_env_q15: int = Q15_HALF

    # Lifecycle flags
    is_starting: bool = False    # First block after note-on
    is_stopping: bool = False    # Transitioning to FREE
    is_active: bool = False      # Voice is sounding

    # Arc modulator state
    arc_active: bool = False
    arc_id: int = 0
    arc_progress_q15: int = 0    # 0 = start, Q15_ONE = complete

    # Activity (for voice stealing priority)
    activity_q15: int = 0        # Current voice activity level

    # Aux send levels
    sendA_q15: int = 0           # Aux bus A send level
    sendB_q15: int = 0           # Aux bus B send level

    # Note info
    note_u8: int = 60            # MIDI note number
    cents_i16: int = 0           # Fine tuning in cents

    # Event payload (packed u64 for FPGA)
    event_payload_u64: int = 0   # Packed event data
    event_commands: Optional[WireFrameData] = None

    # Selectors for the semantic CMMM lanes (not the wire payload layout).
    impulse_band: int = 3
    noise_band: int = 5

    # Legacy compatibility aliases
    @property
    def regime(self) -> Regime:
        return Regime(self.regime_id)

    @property
    def family_weights(self) -> List[int]:
        return self.w_family_q15

    @property
    def events(self) -> EventPayload:
        """Reconstruct EventPayload from mask and strengths."""
        ep = EventPayload(active_mask=self.event_mask,
                          impulse_band=self.impulse_band, noise_band=self.noise_band)
        if self.event_mask & (1 << E_IMPULSE):
            ep.impulse_strength_q15 = self.event_strengths[E_IMPULSE]
        if self.event_mask & (1 << E_NOISE):
            ep.noise_amount_q15 = self.event_strengths[E_NOISE]
        if self.event_mask & (1 << E_PULSE):
            ep.pulse_pressure_q15 = self.event_strengths[E_PULSE]
        if self.event_mask & (1 << E_GATE):
            ep.gate_mask = self.event_strengths[E_GATE]
        if self.event_mask & (1 << E_CUSP):
            ep.cusp_amount_q15 = self.event_strengths[E_CUSP]
        if self.event_mask & (1 << E_SWALLOWTAIL):
            ep.swallowtail_bands = self.event_strengths[E_SWALLOWTAIL]
        if self.event_mask & (1 << E_FOLD):
            ep.fold_pressure_q15 = self.event_strengths[E_FOLD]
        if self.event_mask & (1 << E_SHEAF):
            ep.sheaf_threshold_q15 = self.event_strengths[E_SHEAF]
        return ep

    def to_bytes(self) -> bytes:
        """
        Serialize to bytes for FPGA/network transmission.

        Fixed 64-byte layout for deterministic handling.
        """
        import struct

        if self.event_commands is not None:
            raise ValueError('Legacy VoiceFrameABI bytes cannot carry authored events; convert to VoiceFramePacketV0b')

        # Pack family weights (8 x i16 = 16 bytes)
        family_bytes = struct.pack('<8h', *self.w_family_q15)

        # Pack event strengths (8 x u8 = 8 bytes)
        event_bytes = bytes(min(255, s >> 7) for s in self.event_strengths)

        # Flags byte
        flags = (
            (1 if self.is_starting else 0) |
            (2 if self.is_stopping else 0) |
            (4 if self.is_active else 0) |
            (8 if self.arc_active else 0)
        )

        return struct.pack(
            '<HBBhhh8s8sBBHBBBBBB',
            self.program_id,           # H (2 bytes)
            self.voice_id,             # B (1 byte)
            self.regime_id,            # B (1 byte)
            self.gain_q15,             # h (2 bytes)
            self.pan_q15,              # h (2 bytes)
            self.tilt_env_q15,         # h (2 bytes)
            family_bytes,              # 8s (8 bytes for i16, but we pack separately)
            event_bytes,               # 8s (8 bytes)
            self.event_mask,           # B (1 byte)
            self.tree_a_id,            # B (1 byte)
            self.tree_blend_q15,       # H (2 bytes)
            self.tree_b_id,            # B (1 byte)
            int(self.degradation_level),  # B (1 byte)
            int(self.tower_level),     # B (1 byte)
            self.arc_id,               # B (1 byte)
            (self.arc_progress_q15 >> 8) & 0xFF,  # B (1 byte, high bits)
            flags,                     # B (1 byte)
        ) + family_bytes  # Append full family weights


# Legacy alias
VoiceFrame = VoiceFrameABI


# =============================================================================
# VOICE ENGINE - CMMM Implementation
# =============================================================================

class VoiceEngine:
    """
    Voice Engine CMMM Implementation.

    Implements a Controlled Semi-Markov Mealy Machine:
    - Controlled: GestureBasis input steers behavior
    - Semi-Markov: Non-geometric state durations (ADSR)
    - Mealy: Outputs (VoiceFrame) depend on transitions

    Features:
    - RegimeGraph-based state machine with EdgeSpec transitions
    - ArcSpec modulators for transition effects
    - RegimeSpec policies for Family/Tree/Event/Tail
    - Full VoiceState snapshotability for replay
    - Tower level hierarchical projections
    """

    def __init__(self, max_voices: int = 12, block_size: int = BLOCK_SIZE,
                 sample_rate: float = SAMPLE_RATE,
                 macro_runtime: Optional['MacroRuntime'] = None):
        self.max_voices = max_voices
        self.block_size = block_size
        self.sample_rate = sample_rate
        self.block_time = block_size / sample_rate

        # Voice pool (uses VoiceStateWithProgram for legacy compat)
        self.voices: Dict[int, VoiceStateWithProgram] = {}
        # Sparse controls latch even when their voice slot is currently idle.
        self._voice_controls: Dict[int, VoiceControls] = {}
        self._render_frames: Dict[int, 'VoiceFrame'] = {}
        self.master_gain_q15 = Q15_ONE
        self.master_pan_q15 = 0
        self.tempo_q8_8 = 120 << 8

        # ControlFrame's editing level is independent of coefficient TowerLevel.
        self.control_mode_L = 3
        self.selected_preset: Optional['CompiledPreset'] = None
        self.selected_preset_id: Optional[int] = None
        self._control_presets: Dict[int, Tuple['CompiledPreset', int]] = {}
        self._capture_marks: List['CaptureMark'] = []

        # Global block counter
        self.block_count: int = 0

        # Regime graph (default)
        self.regime_graph = build_default_regime_graph()

        # Program registry
        self.programs = dict(INSTRUMENT_PROGRAMS)

        # Default program
        self.default_program = InstrumentProgram()

        # Global degradation level (from CPU monitor)
        self.global_degradation = DegradationLevel.NONE

        # MacroRuntime for SystemMacro integration (optional)
        # When set, macros (BRIGHTNESS, DENSITY, RISK, etc.) influence CMMM
        self.macro_runtime: Optional['MacroRuntime'] = macro_runtime

    def set_degradation_level(self, level: DegradationLevel):
        """Set global degradation level for CPU load management."""
        self.global_degradation = level

    def register_control_preset(self, preset_id: int, preset: 'CompiledPreset',
                                preset_hash32: int = 0) -> None:
        """Bind a wire ID to an already compiled preset, outside the audio loop.

        The optional hash is supplied by the preset loader; ControlFrame v0
        does not prescribe a content-hash algorithm. No filesystem access or
        compilation occurs while consuming PRESET_LOAD.
        """
        from ..library.preset_compiler import CompiledPreset
        for value in (preset_id, preset_hash32):
            if not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
                raise ValueError('Preset ID and hash must be uint32')
        if not isinstance(preset, CompiledPreset):
            raise TypeError('Expected a CompiledPreset')
        self._control_presets[preset_id] = (preset, preset_hash32)

    def select_compiled_preset(self, preset: 'CompiledPreset') -> None:
        """Select the sound for subsequent notes; existing voices keep theirs."""
        from ..library.preset_compiler import CompiledPreset
        if not isinstance(preset, CompiledPreset):
            raise TypeError('Expected a CompiledPreset')
        self.selected_preset = preset
        self.selected_preset_id = None

    def drain_capture_marks(self) -> List['CaptureMark']:
        """Take markers consumed since the last drain or step_block output."""
        marks, self._capture_marks = self._capture_marks, []
        return marks

    def set_macro_runtime(self, macro_runtime: 'MacroRuntime'):
        """
        Set MacroRuntime for SystemMacro integration.

        When set, macros (BRIGHTNESS, DENSITY, RISK, MOTION, etc.)
        influence the CMMM gesture→intent→family weight pipeline.

        Usage:
            from symcrash_python_ap.macro_runtime import MacroRuntime
            runtime = MacroRuntime()
            engine.set_macro_runtime(runtime)

            # Now macros affect synthesis
            runtime.set_macro("BRIGHT", 24576)  # Increase brightness
            runtime.tick()
        """
        self.macro_runtime = macro_runtime

    def allocate_voice(self, voice_id: Optional[int] = None,
                       program_id: int = 0) -> Optional[int]:
        """
        Allocate a voice slot with optional instrument program.
        """
        if voice_id is None:
            # Find free slot
            for i in range(self.max_voices):
                if i not in self.voices or self.voices[i].phase == VoicePhase.FREE:
                    voice_id = i
                    break
            if voice_id is None:
                return None  # Pool exhausted

        # Get program
        program = self.programs.get(program_id, self.default_program)
        if program.event_policy is not None:
            graph = program.regime_graph or self.regime_graph
            available = {pid for arc in graph.arc_specs.values() for pid in arc.event_patterns}
            required = {rule.pattern_id for lane, rule in enumerate(program.event_policy.lane_rules)
                        if program.event_policy.enable_mask & (1 << lane)
                        and rule.rule_type == LaneRuleType.PATTERN}
            if not required <= available:
                raise ValueError(f'Undefined event patterns: {sorted(required - available)}')

        # Create voice state
        state = VoiceStateWithProgram(voice_id=voice_id, program=program, program_id=program_id)
        state.tree_morph.morph_rate = program.tree_morph_rate
        state.tree_morph.tree_a_id = program.preferred_tree

        state.controls = self._voice_controls.setdefault(voice_id, VoiceControls())
        self.voices[voice_id] = state
        self._render_frames.pop(voice_id, None)
        return voice_id

    def allocate_voice_from_preset(
        self,
        preset: 'CompiledPreset',
        voice_id: Optional[int] = None,
    ) -> Optional[int]:
        """
        Allocate a voice in Simple mode using a CompiledPreset.

        Simple mode bypasses the full CMMM regime system and uses the preset's
        sound parameters directly. LFOs, envelopes, and matrix routing are
        initialized from the preset.

        Args:
            preset: CompiledPreset containing sound definition
            voice_id: Optional voice slot ID (auto-allocate if None)

        Returns:
            Allocated voice ID, or None if pool exhausted
        """
        # Import here to avoid circular imports
        from ..envelope_scheduler import EnvelopeScheduler, ADSRSpec
        from ..envelope_targets import parse_envelope_target

        if voice_id is None:
            # Find free slot
            for i in range(self.max_voices):
                if i not in self.voices or self.voices[i].phase == VoicePhase.FREE:
                    voice_id = i
                    break
            if voice_id is None:
                return None  # Pool exhausted

        # Create voice state in Simple mode (use default program as placeholder)
        state = VoiceStateWithProgram(
            voice_id=voice_id,
            program=self.default_program,
            program_id=0
        )

        # Set Simple mode flag
        state.engine_mode = int(EngineMode.SIMPLE)
        state.compiled_preset = preset
        state.sendA_q15 = getattr(preset, 'sendA_q15', 0)
        state.sendB_q15 = getattr(preset, 'sendB_q15', 0)

        # Initialize beta ramp from preset
        state.beta_start_q15 = preset.material.beta_start_q15
        state.beta_target_q15 = preset.material.beta_target_q15
        state.beta_ramp_blocks = preset.material.beta_ramp_blocks
        state.beta_current_q15 = state.beta_start_q15

        # Initialize main amplitude envelope from preset
        if preset.amp_env:
            state.amp_envelope = EnvelopeScheduler(spec=preset.amp_env)
        else:
            state.amp_envelope = EnvelopeScheduler(spec=ADSRSpec())

        # Amp dynamics (WSE-STD-PRESET-AUTHORING 3.3): velocity curve and
        # DX7-style EG key scaling. Defaults preserve legacy behavior.
        state.amp_env_base_spec = state.amp_envelope.spec
        state.amp_key_scale_rate = float(getattr(preset, 'amp_key_scale_rate', 0.0))
        state.amp_velocity_sens_q15 = int(
            getattr(preset, 'amp_velocity_sens', 1.0) * Q15_ONE)

        # Initialize LFOs from preset
        lfos = getattr(preset, 'lfos', None)
        if lfos:
            from copy import deepcopy
            state.lfos = deepcopy(lfos)  # Runtime delay/ramp counters belong to each voice.
            from .lfo_clocks import initialize_lfo_clocks
            initialize_lfo_clocks(state.lfos, state.arc_phase_q16)

        # Initialize matrix envelopes from preset
        envelope_bindings = getattr(preset, 'envelope_bindings', None)
        if envelope_bindings:
            state.envelope_matrix = {}
            from ..instrument_ir import EnvelopeBinding, ADSR
            for binding in envelope_bindings:
                adsr = binding.adsr
                spec = ADSRSpec(
                    attack_ms=adsr.a_ms,
                    decay_ms=adsr.d_ms,
                    sustain=adsr.s,
                    release_ms=adsr.r_ms,
                )
                target_ids = []
                for target_name in binding.targets:
                    try:
                        target = parse_envelope_target(target_name)
                        target_ids.append(int(target))
                    except (KeyError, ValueError):
                        pass
                if target_ids:
                    state.envelope_matrix[binding.id] = (
                        EnvelopeScheduler(spec=spec),
                        target_ids,
                        int(getattr(binding, 'velocity_sens', 0.0) * Q15_ONE),
                        float(getattr(binding, 'key_scale_rate', 0.0)),
                        spec,  # base (unscaled) spec for key scaling
                    )

        # Initialize tree morph from preset
        state.tree_morph.tree_a_id = preset.latch.tree_a_id
        state.tree_morph.tree_b_id = preset.latch.tree_b_id
        state.tree_morph.blend_q15 = preset.latch.tree_blend_q15

        # Push preset character defaults to MacroRuntime
        control_config = getattr(preset, 'control_config', None)
        if control_config is not None and self.macro_runtime is not None:
            control_program = control_config.to_control_program()
            self.macro_runtime.apply_control_defaults(control_program)

        state.controls = self._voice_controls.setdefault(voice_id, VoiceControls())
        self.voices[voice_id] = state
        self._render_frames.pop(voice_id, None)
        return voice_id

    def note_on_simple(
        self,
        voice_id: int,
        note: int = 60,
        velocity_q15: int = Q15_ONE,
        cents: int = 0,
    ) -> None:
        """
        Trigger note-on for a Simple mode voice.

        Args:
            voice_id: Voice ID (must be allocated with allocate_voice_from_preset)
            note: MIDI note number (0-127)
            velocity_q15: Velocity (0-32767)
            cents: Fine tuning in cents
        """
        if voice_id not in self.voices:
            return

        state = self.voices[voice_id]

        # Store note info
        is_legato = state.phase not in (VoicePhase.FREE, VoicePhase.RELEASE)
        state.note_u8 = note
        state.cents_i16 = cents
        state.velocity_q15 = velocity_q15

        # Reset beta ramp
        if state.compiled_preset:
            state.beta_current_q15 = state.compiled_preset.material.beta_start_q15

        # Reset block counter
        state.block_count = 0

        # Start attack phase
        state.phase = VoicePhase.ATTACK
        state.phase_block = 0
        state.note_start_pending = True

        state.authored_event_state.rise_pending = True

        state.authored_regime_state.rise_pending = True

        # DX7-style EG key scaling: rescale envelope times for this note
        # (2^(-ks * (note - 60) / 24): higher notes ring shorter)
        from ..envelope_scheduler import ADSRSpec as _ADSRSpec

        def _key_scaled(base_spec, ks):
            mult = 2.0 ** (-ks * (note - 60) / 24.0)
            return _ADSRSpec(
                attack_ms=base_spec.attack_ms * mult,
                decay_ms=base_spec.decay_ms * mult,
                sustain=base_spec.sustain,
                release_ms=base_spec.release_ms * mult,
            )

        def _apply_spec(scheduler, spec):
            # EnvelopeScheduler caches block counts in __post_init__;
            # re-run it after swapping the spec or key scaling is a no-op.
            scheduler.spec = spec
            scheduler.__post_init__()

        # Trigger main envelope
        if state.amp_envelope:
            ks_amp = getattr(state, 'amp_key_scale_rate', 0.0)
            base = getattr(state, 'amp_env_base_spec', None)
            if ks_amp and base is not None:
                _apply_spec(state.amp_envelope, _key_scaled(base, ks_amp))
            state.amp_envelope.trigger_note()

        # Trigger all matrix envelopes (with per-binding key scaling)
        for entry in state.envelope_matrix.values():
            scheduler = entry[0]
            ks = entry[3] if len(entry) > 3 else 0.0
            if ks and len(entry) > 4:
                _apply_spec(scheduler, _key_scaled(entry[4], ks))
            scheduler.trigger_note()

        # Handle LFO sync modes on note-on
        if state.lfos:
            state.handle_note_on_lfos(state.lfos, is_legato=is_legato)

        # Default category until an authored graph selects its first node.
        state.regime = Regime.R2_STABLE

    def note_off_simple(self, voice_id: int) -> None:
        """
        Trigger note-off for a Simple mode voice.

        Args:
            voice_id: Voice ID
        """
        if voice_id not in self.voices:
            return

        state = self.voices[voice_id]
        state.authored_event_state.fall_pending = True
        state.authored_regime_state.fall_pending = True

        # Start release phase
        if state.phase != VoicePhase.FREE:
            state.release_start_q15 = state.gain_q15
            state.phase = VoicePhase.RELEASE
            state.phase_block = 0

        # Release main envelope
        if state.amp_envelope:
            state.amp_envelope.release_note()

        # Release all matrix envelopes
        for scheduler, *_ in state.envelope_matrix.values():
            scheduler.release_note()

    def note_on(self, voice_id: int, gesture: Optional[GestureBasis] = None,
                velocity_q15: int = Q15_HALF, pitch_q15: int = 0, pan_q15: int = 0):
        """
        Trigger note-on for a voice with gesture input.
        """
        if voice_id not in self.voices:
            self.allocate_voice(voice_id)

        state = self.voices[voice_id]

        # Set gesture
        is_legato = state.phase not in (VoicePhase.FREE, VoicePhase.RELEASE) and state.gesture.gate
        if state.phase == VoicePhase.FREE and state.block_count == 0 and state.lfos:
            from .lfo_clocks import initialize_lfo_clocks
            initialize_lfo_clocks(state.lfos, state.arc_phase_q16)
        if gesture is not None:
            state.gesture = replace(gesture)
        else:
            state.gesture = GestureBasis.from_note_on(velocity_q15, pitch_q15, pan_q15)
        state.gesture.gate = True
        state.gate_rise_pending = True
        state.authored_event_state.rise_pending = True
        state.transition_memory.reset_for_note_on()

        # Start attack phase
        state.phase = VoicePhase.ATTACK
        state.phase_block = 0
        state.note_start_pending = True

        # Handle LFO sync modes on note-on (if LFOs are present)
        # Per WSE-ARCHITECTURE-v1, Voice Engine owns musical time including LFO phases
        if state.lfos:
            state.handle_note_on_lfos(state.lfos, is_legato)

        # Transition to attack regime with arc
        self._transition_regime(state, state.program.attack_regime, use_arc=True)

    def note_off(self, voice_id: int):
        """
        Trigger note-off for a voice.
        """
        if voice_id not in self.voices:
            return

        state = self.voices[voice_id]
        state.gate_fall_pending |= state.gesture.gate
        state.authored_event_state.fall_pending |= state.gesture.gate
        state.gesture.gate = False

        # Start release phase
        if state.phase != VoicePhase.FREE:
            state.release_start_q15 = state.gain_q15
            state.phase = VoicePhase.RELEASE
            state.phase_block = 0

    def update_gesture(self, voice_id: int, gesture: GestureBasis):
        """
        Update gesture for a voice (continuous control).
        """
        if voice_id not in self.voices:
            return

        state = self.voices[voice_id]
        gesture = replace(gesture)
        # Preserve gate state from current if new gesture doesn't specify
        if not gesture.gate and state.gesture.gate:
            gesture.gate = True
        state.gesture = gesture

    def _graph_for(self, state: VoiceStateWithProgram) -> RegimeGraph:
        return state.program.regime_graph or self.regime_graph

    def _regime_spec_for(self, state: VoiceStateWithProgram, regime=None) -> Optional[RegimeSpec]:
        regime = state.regime if regime is None else regime
        return state.program.regime_specs.get(regime) or self._graph_for(state).regime_specs.get(regime)

    def _arc_spec_for(self, state: VoiceStateWithProgram) -> Optional[ArcSpec]:
        return self._graph_for(state).arc_specs.get(state.arc.arc_id)

    def _transition_regime(self, state: VoiceStateWithProgram, new_regime: Regime,
                           use_arc: bool = True):
        """
        Execute regime transition with optional arc modulator.
        """
        graph = self._graph_for(state)
        note_attack = state.note_start_pending and new_regime == state.program.attack_regime
        allowed = (any(edge.from_regime == state.regime and edge.to_regime == new_regime for edge in graph.edges)
                   if state.program.regime_graph is not None
                   else new_regime in REGIME_TRANSITIONS.get(state.regime, set()))
        if not (note_attack or allowed):
            return  # Invalid transition

        # Record previous regime for arc lookup
        state.previous_regime = state.regime

        # Start arc if specified
        if use_arc:
            arc = graph.get_arc(state.regime, new_regime)
            if arc is None and note_attack:
                arc = graph.get_arc(Regime.R0_IDLE, new_regime)
            if arc is not None:
                state.arc = ArcState(
                    active=True,
                    arc_id=arc.arc_id,
                    progress_blocks=0,
                    total_blocks=arc.duration_blocks,
                )

        # Execute transition
        state.regime = new_regime
        state.regime_blocks = 0
        state.transition_memory.blocks_since_transition = 0

        # Update tower level from regime spec
        regime_spec = self._regime_spec_for(state, new_regime)
        if regime_spec:
            state.tower_level = regime_spec.tower_level

    def _check_regime_transition(self, state: VoiceStateWithProgram) -> Optional[Regime]:
        """
        Check if any regime transition should occur using RegimeGraph.

        Applies temperament modulation to edges if the program has temperament.
        """
        # Use RegimeGraph for intent-based transitions
        # Pass temperament from program if available
        # check_transition returns (target, new_prng_state); the voice's
        # rng_u32 is threaded through so stochastic transitions stay
        # deterministic across snapshot/replay. (Before 2026-06-11 the
        # tuple was used ununpacked: it was never None, the fallback was
        # dead code, and _transition_regime rejected it - every voice
        # was stuck in R1_ATTACK.)
        temperament = state.program.temperament
        target, state.rng_u32 = self._graph_for(state).check_transition(
            state.regime, state.intent, state.regime_blocks,
            temperament=temperament,
            memory=state.transition_memory,
            prng_state=state.rng_u32,
            gesture=state.gesture,
            gate_rise=state.gate_rise_pending or (state.gesture.gate and not state.previous_gate),
            gate_fall=state.gate_fall_pending or (not state.gesture.gate and state.previous_gate),
        )

        if target is not None:
            return target

        # Legacy fallback for attack->sustain transition
        if state.program.regime_graph is None and state.regime == Regime.R1_ATTACK:
            if state.phase_block > state.controls.values.get(ControlID.ATK_BLOCKS, state.program.attack_blocks):
                return state.program.sustain_regime

        return None

    def _apply_arc_modulation(self, state: VoiceStateWithProgram,
                              weights: List[int]) -> List[int]:
        """
        Apply arc modulator to family weights during transition.
        """
        if not state.arc.active:
            return weights

        arc = self._arc_spec_for(state)
        if arc is None:
            return weights

        # Compute arc envelope
        progress = state.arc.get_progress_q15()

        # Apply gain curve
        if arc.gain_curve == "ease_in":
            env = (progress * progress) >> 15
        elif arc.gain_curve == "ease_out":
            inv = Q15_ONE - progress
            env = Q15_ONE - ((inv * inv) >> 15)
        elif arc.gain_curve == "s_curve":
            # Smoothstep: 3x^2 - 2x^3
            x2 = (progress * progress) >> 15
            x3 = (x2 * progress) >> 15
            env = (3 * x2 - 2 * x3)
        else:  # linear
            env = progress

        # Apply family boosts scaled by envelope
        result = list(weights)
        for i in range(NUM_FAMILIES):
            boost = mul_q15_sat(arc.family_boost[i], env)
            result[i] = sat16(result[i] + boost)

        return result

    def _compute_family_weights_with_regime(self, state: VoiceStateWithProgram) -> List[int]:
        """
        Compute family weights using RegimeSpec policies.

        If the program carries per-regime emphasis (from the preset's behavior
        section), those weights are used as family_base instead of the hardcoded
        RegimeSpec defaults.  This lets preset authors control family dispatch
        using musical vocabulary (tone, body, energy, …) without touching
        tower internals.
        """
        from ..instrument_ir import emphasis_to_family_base

        intent = state.intent
        regime = state.regime
        regime_spec = self._regime_spec_for(state, regime)

        # Start with base weights — prefer preset emphasis over hardcoded defaults
        emphasis = state.program.per_regime_emphasis.get(int(regime))
        if emphasis:
            weights = emphasis_to_family_base(emphasis)
        elif regime_spec:
            weights = list(regime_spec.family_base)
        else:
            weights = [0] * NUM_FAMILIES

        # Add intent-derived weights
        weights[F_COLOR] = sat16(weights[F_COLOR] + ((intent.tone + intent.bow) >> 2))
        weights[F_TRANSPORT] = sat16(weights[F_TRANSPORT] + ((intent.warp + intent.air) >> 2))
        weights[F_LOCAL] = sat16(weights[F_LOCAL] + ((intent.bow + intent.tone) >> 2))
        weights[F_TOPOLOGY] = sat16(weights[F_TOPOLOGY] + ((intent.glitch + intent.warp) >> 2))
        weights[F_GLUE] = sat16(weights[F_GLUE] + (intent.glue >> 1))
        weights[F_EVENTS] = sat16(weights[F_EVENTS] + ((intent.snap + intent.glitch) >> 2))

        # Catastrophe only in R5
        if regime == Regime.R5_CATASTROPHE:
            weights[F_CATASTROPHE] = Q15_ONE
        else:
            weights[F_CATASTROPHE] = 0

        # Apply caps from regime spec
        if regime_spec:
            for i in range(NUM_FAMILIES):
                weights[i] = min(weights[i], regime_spec.family_cap[i])
                # Apply enable mask
                if not (regime_spec.family_enable_mask & (1 << i)):
                    weights[i] = 0

        # Apply program modifiers
        weights = [
            mul_q15_sat(weights[i], state.program.family_mods[i])
            for i in range(NUM_FAMILIES)
        ]

        # Apply arc modulation
        weights = self._apply_arc_modulation(state, weights)

        # Apply degradation
        weights = self._apply_degradation(weights, state.degradation)

        return weights

    def _apply_degradation(self, weights: List[int],
                           level: DegradationLevel) -> List[int]:
        """
        Apply degradation level to family weights.
        """
        result = list(weights)

        if level >= DegradationLevel.LIGHT:
            # Cap transport
            result[F_TRANSPORT] = min(result[F_TRANSPORT], int(Q15_ONE * 0.3))

        if level >= DegradationLevel.HEAVY:
            # Disable transport and topology
            result[F_TRANSPORT] = 0
            result[F_TOPOLOGY] = 0

        if level >= DegradationLevel.EMERGENCY:
            # Only glue (tail stages)
            for i in range(NUM_FAMILIES):
                if i != F_GLUE:
                    result[i] = 0

        return result

    def _compute_events_with_regime(self, state: VoiceStateWithProgram,
                                    allowed_mask: int = 0xFF) -> Tuple[int, List[int]]:
        """
        Compute events using RegimeSpec event policy.

        Returns (event_mask, event_strengths).
        """
        intent = state.intent
        regime = state.regime
        regime_spec = self._regime_spec_for(state, regime)

        mask = 0
        strengths = [0] * NUM_EVENTS

        enable_mask = (regime_spec.event_enable_mask if regime_spec else 0xFF) & allowed_mask

        rise = state.gate_rise_pending or (state.gesture.gate and not state.previous_gate)
        fall = state.gate_fall_pending or (not state.gesture.gate and state.previous_gate)
        state.previous_gate = state.gesture.gate
        state.gate_rise_pending = state.gate_fall_pending = False
        policy = state.program.event_policy
        if state.program.authored_events is not None:
            # The IR explicitly authors its events; built-in semantic bursts
            # must not also fire. Edge history still advances above.
            return 0, strengths
        if policy is not None:
            arc = self._arc_spec_for(state) if state.arc.active else None
            mask, strengths, _, state.rng_u32 = policy.compute(
                intent, state.gesture, state.rng_u32, allowed_mask=enable_mask,
                gate_rise=rise, gate_fall=fall,
                patterns=arc.event_patterns if arc else None,
                arc_block=state.arc.progress_blocks if arc else None,
                probabilities=regime_spec.event_probability if regime_spec else None,
            )
            return mask, strengths

        # E0: Impulse - high snap
        if (enable_mask & (1 << E_IMPULSE)) and intent.snap > Q15_HALF:
            mask |= (1 << E_IMPULSE)
            strengths[E_IMPULSE] = intent.snap

        # E1: Noise - high air
        if (enable_mask & (1 << E_NOISE)) and intent.air > Q15_HALF:
            mask |= (1 << E_NOISE)
            strengths[E_NOISE] = intent.air >> 1

        # E2: Pulse - attack regime
        if (enable_mask & (1 << E_PULSE)) and regime == Regime.R1_ATTACK:
            mask |= (1 << E_PULSE)
            strengths[E_PULSE] = Q15_HALF

        # E3: Gate - high glitch
        if (enable_mask & (1 << E_GATE)) and intent.glitch > Q15_HALF:
            mask |= (1 << E_GATE)
            strengths[E_GATE] = (state.block_count * 17) & 0xFF

        # E4: Cusp - catastrophe
        if (enable_mask & (1 << E_CUSP)) and regime == Regime.R5_CATASTROPHE:
            mask |= (1 << E_CUSP)
            strengths[E_CUSP] = Q15_HALF

        # E5: Swallowtail - glitch and catastrophe corridors.
        if (enable_mask & (1 << E_SWALLOWTAIL)) and regime in (Regime.R4_GLITCH, Regime.R5_CATASTROPHE):
            mask |= (1 << E_SWALLOWTAIL)
            strengths[E_SWALLOWTAIL] = 0x3C  # Mid bands

        # E6: Fold - catastrophe
        if (enable_mask & (1 << E_FOLD)) and regime == Regime.R5_CATASTROPHE:
            mask |= (1 << E_FOLD)
            strengths[E_FOLD] = Q15_ONE

        # E7: Sheaf - catastrophe containment and recovery.
        if (enable_mask & (1 << E_SHEAF)) and regime in (Regime.R5_CATASTROPHE, Regime.R6_RECOVERY):
            mask |= (1 << E_SHEAF)
            strengths[E_SHEAF] = Q15_HALF

        # Add arc events if active
        if state.arc.active:
            arc = self._arc_spec_for(state)
            if arc:
                previous_mask = mask
                mask |= arc.arc_events & allowed_mask
                for i in range(NUM_EVENTS):
                    if arc.arc_events & allowed_mask & (1 << i):
                        if i in (E_GATE, E_SWALLOWTAIL):
                            # These lanes carry masks, not amplitudes. An arc
                            # must not turn a valid mask into the integer 16384.
                            if not previous_mask & (1 << i):
                                strengths[i] = 0xFF if i == E_GATE else 0x3C
                        else:
                            strengths[i] = max(strengths[i], Q15_HALF)

        return mask, strengths

    def _step_envelope(self, state: VoiceStateWithProgram) -> int:
        """
        Step ADSR envelope, return gain in Q15.
        """
        program = state.program
        controls = state.controls.values
        attack = controls.get(ControlID.ATK_BLOCKS, program.attack_blocks)
        decay = controls.get(ControlID.DCY_BLOCKS, program.decay_blocks)
        sustain = controls.get(ControlID.SUS_Q15, program.sustain_level_q15)
        release = controls.get(ControlID.REL_BLOCKS, program.release_blocks)

        if state.phase == VoicePhase.FREE:
            return 0

        state.phase_block += 1

        if state.phase == VoicePhase.ATTACK:
            if state.phase_block >= attack:
                state.phase = VoicePhase.DECAY
                state.phase_block = 0
                return Q15_ONE
            else:
                # Linear attack
                return (state.phase_block * Q15_ONE) // max(1, attack)

        elif state.phase == VoicePhase.DECAY:
            if state.phase_block >= decay:
                state.phase = VoicePhase.SUSTAIN
                state.phase_block = 0
                return sustain
            else:
                # Linear decay from peak to sustain
                progress = state.phase_block / max(1, decay)
                return int(Q15_ONE * (1 - progress) + sustain * progress)

        elif state.phase == VoicePhase.SUSTAIN:
            return sustain

        elif state.phase == VoicePhase.RELEASE:
            if state.phase_block >= release:
                state.phase = VoicePhase.FREE
                return 0
            else:
                # Release from the actual level, including a note-off during
                # attack/decay or after a live sustain change.
                return (state.release_start_q15 * (release - state.phase_block)) // max(1, release)

        return 0

    def step_voice(self, voice_id: int, *, mid_due: bool = True,
                   top_due: bool = True) -> Optional[VoiceFrameABI]:
        """
        Step a single voice by one block (CMMM step function).

        Full CMMM pipeline:
        1. Update gesture -> intent (Controller input)
        2. Check regime transition via RegimeGraph (Semi-Markov FSM)
        3. Apply arc modulator if active
        4. Compute family weights from RegimeSpec + intent
        5. Compute events from RegimeSpec + intent
        6. Update tree morph
        7. Step envelope (Semi-Markov duration)
        8. Emit VoiceFrameABI (Mealy output)
        """
        if voice_id not in self.voices:
            return None

        state = self.voices[voice_id]

        if state.phase == VoicePhase.FREE:
            return None

        # Track for is_starting / is_stopping
        old_phase = state.phase

        # Apply global degradation
        state.degradation = self.global_degradation

        # 1. Gesture -> Intent (Controller input processing)
        # If macro_runtime is set, apply SystemMacros to gesture/intent pipeline
        if self.macro_runtime is not None:
            state.intent = gesture_to_intent_with_macros(
                state.gesture,
                self.macro_runtime,
                state.program.gesture_to_intent
            )
        else:
            state.intent = state.program.gesture_to_intent(state.gesture)

        # 2. Check regime transition via RegimeGraph
        old_regime = state.regime
        target_regime = self._check_regime_transition(state)
        if target_regime is not None and target_regime != state.regime:
            self._transition_regime(state, target_regime, use_arc=True)
        state.regime_blocks += 1
        intent_energy = sum(max(0, min(Q15_ONE, value)) for value in state.intent.lanes) // NUM_INTENTS
        state.transition_memory.step(intent_energy, state.regime != old_regime)

        # 3. Sample the arc at its current zero-based block. Advance only
        # after emission so a duration-N arc contributes exactly N frames.

        # 4. Compute family weights with regime policy + arc
        state.family_weights = self._compute_family_weights_with_regime(state)

        # 5. Compute events with regime policy
        from ..vm.scheduling import cmmm_emission_mask
        event_mask, event_strengths = self._compute_events_with_regime(
            state, cmmm_emission_mask(mid_due, top_due))

        # 6. Tree morph (with regime spec override)
        regime_spec = self._regime_spec_for(state)
        if regime_spec and regime_spec.tree_lock:
            # Regime locks tree
            target_tree = regime_spec.preferred_tree
        else:
            target_tree = select_tree_from_intent(state.intent, state.regime)

        # Apply arc morph rate if active
        if state.arc.active:
            arc = self._arc_spec_for(state)
            if arc and arc.morph_rate_override > 0:
                state.tree_morph.morph_rate = arc.morph_rate_override
        elif regime_spec:
            state.tree_morph.morph_rate = regime_spec.morph_rate

        state.tree_morph = update_tree_morph(state.tree_morph, target_tree)

        # 7. Step envelope (Semi-Markov duration)
        state.gain_q15 = self._step_envelope(state)

        # 7b. Tick LFOs if present (uses arc_phase_q16 for phase storage)
        # Per WSE-ARCHITECTURE-v1, Voice Engine owns musical time including LFO phases
        if state.lfos:
            state.tick_lfos(state.lfos)

        # 8. Update block counter
        state.block_count += 1

        # Detect phase transitions
        new_phase = state.phase
        is_starting = state.note_start_pending and new_phase != VoicePhase.FREE
        state.note_start_pending = False
        is_stopping = new_phase == VoicePhase.FREE and old_phase != VoicePhase.FREE

        # Compute musical envelopes from intent
        tilt_env = (state.intent.tone + state.intent.bow) >> 1
        density_env = (state.intent.air + state.intent.glitch) >> 1

        # Get tail policy from regime spec
        normalize_curve = state.program.normalize_curve
        fineclamp_cap = state.program.fineclamp_cap_ratio
        slew_delta = state.program.slew_max_delta
        if regime_spec:
            if regime_spec.normalize_curve is not None:
                normalize_curve = regime_spec.normalize_curve
            if regime_spec.fineclamp_cap_ratio is not None:
                fineclamp_cap = regime_spec.fineclamp_cap_ratio
            if regime_spec.slew_max_delta is not None:
                slew_delta = regime_spec.slew_max_delta

        # Emit VoiceFrameABI (Mealy output)
        frame = VoiceFrameABI(
            program_id=state.program_id,
            voice_id=voice_id,
            regime_id=int(state.regime),
            gain_q15=state.gain_q15,
            pan_q15=state.gesture.pan_q15,
            sendA_q15=state.sendA_q15,
            sendB_q15=state.sendB_q15,
            w_family_q15=list(state.family_weights),
            event_mask=event_mask,
            event_strengths=event_strengths,
            tree_a_id=state.tree_morph.tree_a_id,
            tree_b_id=state.tree_morph.tree_b_id,
            tree_blend_q15=state.tree_morph.blend_q15,
            degradation_level=state.degradation,
            tower_level=state.tower_level,
            normalize_curve=normalize_curve,
            fineclamp_cap_ratio=fineclamp_cap,
            slew_max_delta=slew_delta,
            tilt_env_q15=tilt_env,
            density_env_q15=density_env,
            is_starting=is_starting,
            is_stopping=is_stopping,
            is_active=state.phase != VoicePhase.FREE,
            arc_active=state.arc.active,
            arc_id=state.arc.arc_id,
            arc_progress_q15=state.arc.get_progress_q15(),
            note_u8=state.note_u8,
            cents_i16=state.cents_i16 + state.pitch_offset_cents + state.controls.pitch_offset_cents,
        )
        policy = state.program.event_policy
        if policy is not None:
            impulse_band = policy.lane_rules[E_IMPULSE].target_band
            noise_band = policy.lane_rules[E_NOISE].target_band
            if impulse_band is not None:
                frame.impulse_band = impulse_band
            if noise_band is not None:
                frame.noise_band = noise_band
        if state.program.authored_events is not None:
            program = state.program.authored_events
            template = program.regime_templates.get(frame.regime_id,
                {0: 'T_IDLE', 1: 'T_ATTACK', 2: 'T_STABLE', 3: 'T_AIR',
                 4: 'T_GLITCH', 5: 'T_CATA', 6: 'T_RECOVERY'}[frame.regime_id])
            context = event_context(gesture=state.gesture, intent=state.intent,
                velocity=state.velocity_q15, gain=state.gain_q15, block=max(0, state.block_count - 1),
                risk=state.gesture.chaos_q15)
            frame.event_commands = state.authored_event_state.emit(
                program, template, context, gate=state.gesture.gate, regime=frame.regime_id)
        state.arc.step()
        return frame

    def step_voice_simple(self, voice_id: int) -> Optional['VoiceFrame']:
        """
        Step a Simple mode voice by one block.

        Returns a VoiceFrame (not VoiceFrameABI) for compatibility with
        existing render paths that use VoiceFrameEmitter.

        Pipeline:
        1. Tick amplitude envelope
        2. Tick LFOs and apply modulation
        3. Update beta ramp
        4. Tick matrix envelopes
        5. Emit VoiceFrame

        Returns:
            VoiceFrame if voice is active, None otherwise
        """
        if voice_id not in self.voices:
            return None

        state = self.voices[voice_id]

        if state.phase == VoicePhase.FREE:
            return None

        if state.engine_mode != int(EngineMode.SIMPLE):
            # Not in Simple mode - caller should use step_voice()
            return None

        preset = state.compiled_preset
        if preset is None:
            return None
        native = getattr(preset, 'native_carrier', None)

        # Track for is_starting
        is_starting = (state.block_count == 0)

        if state.amp_envelope:
            state.controls.apply_envelope(state.amp_envelope)

        # Step amplitude envelope
        if state.amp_envelope:
            gain_q15 = state.amp_envelope.tick()
            # Check if envelope is done
            if not state.amp_envelope.is_active:
                state.phase = VoicePhase.FREE
                if getattr(preset, 'event_program', None) is None and getattr(preset, 'regime_program', None) is None:
                    return None
            else:
                state.phase = VoicePhase[state.amp_envelope.phase.name]
            state.phase_block = state.amp_envelope.phase_block
        else:
            gain_q15 = Q15_ONE if state.phase != VoicePhase.RELEASE else 0

        # Apply velocity scaling
        # Velocity -> gain through the Amp velocity curve:
        # scale = (1 - sens) + sens * velocity (sens=1.0 -> legacy raw scaling)
        amp_sens_q15 = getattr(state, 'amp_velocity_sens_q15', Q15_ONE)
        vel_scale = Q15_ONE - amp_sens_q15 + ((amp_sens_q15 * state.velocity_q15) >> 15)
        gain_q15 = (gain_q15 * vel_scale) >> 15
        state.gain_q15 = gain_q15

        # Compute beta ramp
        if state.block_count < state.beta_ramp_blocks and state.beta_ramp_blocks > 1:
            t = state.block_count / (state.beta_ramp_blocks - 1)
            state.beta_current_q15 = int(
                state.beta_start_q15 + (state.beta_target_q15 - state.beta_start_q15) * t
            )
        else:
            state.beta_current_q15 = state.beta_target_q15

        # Compute f0_hz from MIDI note + cents + pitch offset
        total_cents = state.cents_i16 + state.pitch_offset_cents + state.controls.pitch_offset_cents
        if native is not None:
            total_cents += native.detune_cents
        # Get preset PWM width default
        pwm_width_q15 = getattr(preset, 'pwm_width_q15', Q15_HALF)

        # Get vibrato macro depth scale (Q15: 32767 = 100%)
        # SystemMacro.VIBRATO controls global LFO depth modulation
        vibrato_depth_scale = Q15_ONE  # Default: full depth
        brightness_q15 = Q15_ONE  # Default: full brightness (filter open)
        if self.macro_runtime is not None:
            from ..macro_runtime import SystemMacro
            vibrato_depth_scale = self.macro_runtime.get_system_macro(SystemMacro.VIBRATO)
            brightness_q15 = self.macro_runtime.get_system_macro(SystemMacro.BRIGHTNESS)

        # Initialize filter parameters from preset defaults
        fm_modulator = None
        if native is not None and native.fm_source == 2 and native.fm_depth_q15:
            from .phase_modulation import phase_lfo_lanes
            fm_modulator = phase_lfo_lanes(state.lfos, state.arc_phase_q16, vibrato_depth_scale)
        # PM samples the start of this block; existing control-rate targets
        # retain their end-of-block sample and every clock advances once.
        if state.lfos:
            state.tick_lfos(state.lfos)

        filter_cutoff_q15 = getattr(native or preset, 'filter_cutoff_q15', Q15_ONE)
        filter_reso_q15 = getattr(native or preset, 'filter_reso_q15', 0)
        filter_mode = getattr(native or preset, 'filter_mode', 0)

        # Apply BRIGHTNESS macro to filter cutoff
        # BRIGHTNESS scales the cutoff: 0 = closed, 32767 = fully open
        # This gives the classic "brightness knob opens the filter" behavior
        filter_cutoff_q15 = RSHIFT_RNE(filter_cutoff_q15 * brightness_q15, 15)

        for lfo in state.lfos:
            lfo_val = lfo.get_lfo_value_from_arc(state.arc_phase_q16)
            # Scale LFO output by vibrato macro
            lfo_val = RSHIFT_RNE(lfo_val * vibrato_depth_scale, 15)
            for routing in lfo.routings:
                scaled = RSHIFT_RNE(lfo_val * routing.amount_q15, 15)

                if routing.target == LFOTarget.PITCH:
                    total_cents += scaled
                elif routing.target == LFOTarget.GAIN:
                    gain_q15 = mul_q15_sat(gain_q15, Q15_ONE + scaled)
                elif routing.target == LFOTarget.PWM_WIDTH:
                    pwm_width_q15 = max(0, min(Q15_ONE, pwm_width_q15 + scaled))
                elif routing.target == LFOTarget.FILTER_CUTOFF:
                    # Modulate filter cutoff (additive on log scale)
                    filter_cutoff_q15 = max(0, min(Q15_ONE, filter_cutoff_q15 + scaled))
                elif routing.target == LFOTarget.FILTER_RESO:
                    # Modulate filter resonance
                    filter_reso_q15 = max(0, min(Q15_ONE, filter_reso_q15 + scaled))

        # Recompute f0_hz if pitch was modulated
        f0_hz = 440.0 * (2.0 ** ((state.note_u8 - 69 + total_cents / 100.0) / 12.0))

        # Tick matrix envelopes and collect target values
        env_targets_q15 = None
        if state.envelope_matrix:
            env_targets_q15 = {}
            for env_id, entry in state.envelope_matrix.items():
                scheduler, target_ids = entry[0], entry[1]
                sens_q15 = entry[2] if len(entry) > 2 else 0
                value = scheduler.tick()
                if sens_q15:
                    # velocity_sens: value *= (1 - sens + sens*velocity)
                    scale = Q15_ONE - sens_q15 + ((sens_q15 * state.velocity_q15) >> 15)
                    value = (value * scale) >> 15
                for target_id in target_ids:
                    env_targets_q15[target_id] = value

        # Get CMMM defaults from preset
        cmmm = getattr(preset, 'cmmm', None)

        # Determine oscillator mode and FM6 fields
        osc_mode = getattr(preset, 'osc_mode', 0)
        fm6_config = getattr(preset, 'fm6', None)

        # Build VoiceFrame
        frame = NativeVoiceFrame(
            voice_id=voice_id,
            f0_hz=f0_hz,
            note=state.note_u8,
            cents=total_cents,
            gain_q15=gain_q15,
            sendA_q15=state.sendA_q15,
            sendB_q15=state.sendB_q15,
            beta_target_q15=state.beta_current_q15,
            partial_amps_q15=list(preset.harmonic_amps_q15),
            H_request=(min(255, len(preset.harmonic_amps_q15))
                       if native is None or native.oeis_sequence_id < 0 else 0) or 16,
            ops_enable=preset.backbone_ops(),
            # Dict keyed by operator (frames.VoiceFrame contract); the old
            # list(...) form crashed executor._get_dynamic_backbone's .get()
            families=preset.backbone_families(),
            tree_a_id=preset.latch.tree_a_id,
            tree_b_id=preset.latch.tree_b_id,
            tree_blend_q15=preset.latch.tree_blend_q15,
            is_starting=is_starting,
            is_active=state.phase != VoicePhase.FREE,
            pwm_width_q15=pwm_width_q15,
            coeff_osc_mode=osc_mode,
            fm_algorithm_id=fm6_config.algorithm_id if fm6_config else None,
            fm_op_ratios=list(fm6_config.op_ratios) if fm6_config else None,
            fm_op_levels=list(fm6_config.op_levels) if fm6_config else None,
            fm_feedback_depth=fm6_config.feedback_depth if fm6_config else None,
            fm_mod_index_q16=(
                round(fm6_config.mod_index * Q16_ONE)
                if fm6_config and getattr(fm6_config, 'mod_index', None) is not None
                else None
            ),
            fm_post_filter_enabled=(
                bool(getattr(fm6_config, 'post_filter_hz', None)) or None
                if fm6_config else None
            ),
            fm_post_filter_cutoff_hz=(
                getattr(fm6_config, 'post_filter_hz', None)
                if fm6_config else None
            ),
            env_targets_q15=env_targets_q15,
            cmmm_node_mask=cmmm.node_mask_u32 if cmmm else None,
            cmmm_mask_ramp_blocks=cmmm.mask_ramp_blocks if cmmm else 5,
            cmmm_tail_mode=cmmm.tail_mode if cmmm else None,
            cmmm_traction_q15=cmmm.traction_q15 if cmmm else Q15_ONE,
            # Filter parameters (modulated by LFO/macro)
            filter_cutoff_q15=filter_cutoff_q15,
            filter_reso_q15=filter_reso_q15,
            filter_mode=filter_mode,
        )

        frame.pan_q15 = self.master_pan_q15
        if native is not None:
            native.apply(frame)
            if native.classic_modulation is not None:
                native.classic_modulation.apply(frame, state.block_count)
        frame.fm_modulator_q15 = fm_modulator

        event_program = getattr(preset, 'event_program', None)
        regime_program = getattr(preset, 'regime_program', None)
        if event_program is not None or regime_program is not None:
            gate = state.phase not in (VoicePhase.RELEASE, VoicePhase.FREE)
            gesture = replace(state.gesture, gate=gate)
            if state.authored_event_state.rise_pending:
                gesture.pressure_q15 = state.velocity_q15
            state.gesture = gesture
            state.intent = (gesture_to_intent_with_macros(gesture, self.macro_runtime, default_gesture_to_intent)
                            if self.macro_runtime is not None else default_gesture_to_intent(gesture))
            risk = gesture.chaos_q15
            if self.macro_runtime is not None:
                from ..macro_runtime import SystemMacro
                risk = self.macro_runtime.get_system_macro(SystemMacro.RISK)
            # Authored dwell is evaluated against this voice's template clock.
            context = event_context(gesture=state.gesture, intent=state.intent,
                velocity=state.velocity_q15, gain=state.gain_q15, block=state.block_count,
                risk=risk)
            context.update({'macro.' + key: value for key, value in state.authored_event_state.macros.items()})
            event_phase = ('ATTACK' if state.authored_event_state.rise_pending else
                           'RELEASE' if state.authored_event_state.fall_pending else state.phase.name)
            if regime_program is not None:
                runtime = state.authored_regime_state
                weights, graph_context = runtime.step(regime_program, context, gate=gate)
                runtime.apply(regime_program, frame, weights, graph_context)
                node = regime_program.nodes[runtime.node]
                selected, event_regime = node.template, node.event_id
                category = {'Idle': 0, 'Transient': 1, 'StableTone': 2,
                            'NoiseTexture': 3, 'GlitchChatter': 4, 'Catastrophe': 5, 'Recovery': 6}
                state.regime = Regime(category.get(node.kind, 2))
                state.regime_blocks = runtime.blocks
            elif event_program is not None:
                selected, event_regime = phase_template(event_program, event_phase), phase_regime(event_phase)
            if event_program is not None:
                frame.event_commands = state.authored_event_state.emit(event_program,
                    selected, context, gate=gate, regime=event_regime,
                    entered=regime_program is not None and bool(graph_context['regime_enter']))
            else:
                state.authored_event_state.rise_pending = state.authored_event_state.fall_pending = False

        state.controls.apply_frame(frame)

        # Advance block counter
        if state.unison is not None:
            state.unison.apply(frame)
        state.block_count += 1

        return frame

    def step_all(self) -> List[VoiceFrameABI]:
        """
        Step all active voices by one block.
        """
        self.block_count += 1
        frames = []

        for voice_id in list(self.voices.keys()):
            frame = self.step_voice(voice_id)
            if frame is not None:
                frames.append(frame)

        return frames

    def is_voice_active(self, voice_id: int) -> bool:
        """Check if voice is still sounding."""
        if voice_id not in self.voices:
            return False
        return self.voices[voice_id].phase != VoicePhase.FREE

    def get_block_count(self) -> int:
        """Get current block count."""
        return self.block_count

    def get_active_voice_count(self) -> int:
        """Get number of active voices."""
        return sum(1 for v in self.voices.values() if v.phase != VoicePhase.FREE)

    def active_voices(self) -> List[int]:
        """Get list of active voice IDs."""
        return [vid for vid, v in self.voices.items() if v.phase != VoicePhase.FREE]

    def get_voice_regime(self, voice_id: int) -> Optional[Regime]:
        """Get current regime for a voice."""
        if voice_id not in self.voices:
            return None
        return self.voices[voice_id].regime

    def get_voice_intent(self, voice_id: int) -> Optional[IntentVector]:
        """Get current intent vector for a voice."""
        if voice_id not in self.voices:
            return None
        return self.voices[voice_id].intent

    def set_event_macro(self, voice_id: int, name: str, value_q15: int) -> None:
        """Set a declared event/graph macro without changing the shared preset."""
        state = self.voices.get(voice_id)
        if state is None:
            raise ValueError('Event macro requires an allocated voice')
        program = (getattr(state.compiled_preset, 'event_program', None) if state.compiled_preset is not None
                   else state.program.authored_events)
        if program is None and state.compiled_preset is not None:
            program = state.compiled_preset.regime_program
        if program is None:
            raise ValueError('Voice has no authored event program')
        state.authored_event_state.set_macro(program, name, value_q15)

    # =========================================================================
    # CONTROL PLANE INTEGRATION (Phase 1: API Unification)
    # =========================================================================

    def step_block(self, control_frame: Optional['ControlFrame'] = None, *,
                   schedule: Optional['ScheduleResult'] = None) -> VoiceEngineOutput:
        """
        Process one block with ControlFrame input.

        This is the canonical entry point per WSE-ARCHITECTURE-v1:
        - VoiceEngine is the sole owner of musical time
        - Consumes ControlFrame from CockpitEngine
        - Returns render_batch + ParamFrame + telemetry

        The unified execution layer (Stages 3-10) consumes the output
        to produce audio. VoiceEngine does NOT own rendering.

        Args:
            control_frame: ControlFrame from CockpitEngine (optional for backward compat)

        Returns:
            VoiceEngineOutput with:
            - render_batch: Voices to synthesize, with explicit serviced/held flags
            - param_frame: Globals for Material/Safety
            - telemetry: Scheduler health metrics
        """
        from ..frames import ParamFrame, VoiceFrame
        from ..control_frame import ControlFrame, EventType
        from ..symbol.executor_scheduling import HealthTelemetry

        # The host supplies a schedule computed at BLOCK_START. Omitting it
        # retains unlimited offline rendering; no hardware budget is guessed.
        selected = None
        if schedule is not None:
            selected = {info.voice_id: info for info in schedule.serviced}
            if (len(selected) != len(schedule.serviced) or
                    any(not 0 <= vid < self.max_voices for vid in selected)):
                raise ValueError('Schedule must contain unique, valid voice slots')

        # 1. Process ControlFrame events (NOTE_ON, NOTE_OFF, etc.)
        if control_frame is not None:
            self.consume_control_frame(control_frame)

        # 2. Extract globals from ControlFrame → ParamFrame
        param_frame = self._extract_param_frame(control_frame)

        # 3. Increment block counter
        self.block_count += 1

        # 4. Step serviced voices and retain render frames for held voices.
        render_batch: List[RenderBatchItem] = []

        for voice_id in list(self.voices.keys()):
            state = self.voices[voice_id]
            if state.phase == VoicePhase.FREE:
                self._render_frames.pop(voice_id, None)
                continue

            info = selected.get(voice_id) if selected is not None else None
            if selected is not None and info is None:
                held = self._render_frames.get(voice_id)
                if held is not None:
                    render_batch.append(RenderBatchItem(
                        voice_id, held, mid_due=False, top_due=False,
                        degrade_level=schedule.degrade_state, serviced=False))
                continue
            mid_due = info.mid_due if info is not None else True
            top_due = info.top_due if info is not None else True

            # Step the voice (CMMM or Simple mode)
            if getattr(state, 'engine_mode', 0) == int(EngineMode.SIMPLE):
                frame = self.step_voice_simple(voice_id)
            else:
                abi = self.step_voice(voice_id, mid_due=mid_due, top_due=top_due)
                if abi is not None:
                    # Convert VoiceFrameABI to VoiceFrame
                    frame = self._abi_to_voice_frame(abi, voice_id)
                else:
                    frame = None

            if frame is not None:
                self._render_frames[voice_id] = frame
                item = RenderBatchItem(
                    voice_id=voice_id,
                    frame=frame,
                    mid_due=mid_due,
                    top_due=top_due,
                    degrade_level=(schedule.degrade_state if schedule is not None
                                   else int(self.global_degradation)),
                )
                render_batch.append(item)

        # Use the host's observational telemetry without recomputing same-block D.
        telemetry = schedule.telemetry if schedule is not None else HealthTelemetry(
            voices_serviced=len(render_batch),
            voices_eligible=len([v for v in self.voices.values() if v.phase != VoicePhase.FREE]),
            gear=5 - int(self.global_degradation),
        )

        return VoiceEngineOutput(
            render_batch=render_batch,
            param_frame=param_frame,
            telemetry=telemetry,
            block_index=self.block_count,
            capture_marks=self.drain_capture_marks(),
        )

    def consume_control_frame(self, control_frame: 'ControlFrame') -> None:
        """Latch controls and consume events without advancing musical time.

        Voice IDs always address real slots, including slot zero. A cockpit
        controlling a pool emits one patch per slot via tick(voice_ids=...).
        """
        # Reject unresolved or malformed commands before applying any controls
        # or notes. The caller retains the complete event list for correction.
        self._validate_control_commands(control_frame)
        self.control_mode_L = control_frame.mode_L
        g = control_frame.globals
        if g.master_gain_q15 is not None:
            self.master_gain_q15 = max(0, min(Q15_ONE, int(g.master_gain_q15)))
        if g.master_pan_q15 is not None:
            self.master_pan_q15 = max(Q15_NEG_ONE, min(Q15_ONE, int(g.master_pan_q15)))
        if g.tempo_q8_8 is not None:
            self.tempo_q8_8 = max(0, min(65535, int(g.tempo_q8_8)))

        for patch in control_frame.voice_patches:
            if not 0 <= patch.voice_id < self.max_voices:
                continue
            controls = (self.voices[patch.voice_id].controls if patch.voice_id in self.voices
                        else self._voice_controls.setdefault(patch.voice_id, VoiceControls()))
            self._voice_controls[patch.voice_id] = controls
            controls.update(patch.items)
            if patch.voice_id in self.voices:
                self.voices[patch.voice_id].controls = controls
            if ControlID.MASTER_GAIN in controls.values:
                self.master_gain_q15 = controls.values.pop(ControlID.MASTER_GAIN)
        self._process_control_events(control_frame)
        control_frame.clear_events()

    def _validate_control_commands(self, control_frame: 'ControlFrame') -> None:
        from ..control_frame import EventType

        if not isinstance(control_frame.mode_L, int) or not 0 <= control_frame.mode_L <= 5:
            raise ValueError('ControlFrame mode_L must be in 0..5')
        for event in control_frame.events:
            try:
                kind = EventType(event.type)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'Unsupported control event: {event.type}') from exc
            if kind in (EventType.NOTE_ON, EventType.NOTE_OFF):
                continue
            for value in (event.a, event.b):
                if not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
                    raise ValueError('Command payloads must be uint32')
            if kind == EventType.PRESET_LOAD:
                if event.a not in self._control_presets:
                    raise LookupError(f'Unregistered control preset: {event.a}')
                if event.b and event.b != self._control_presets[event.a][1]:
                    raise ValueError(f'Preset hash mismatch: {event.a}')
            else:
                if event.b != 0:
                    raise ValueError('MODE_SET and CAPTURE_MARK reserve payload b as zero')
                if kind == EventType.MODE_SET and event.a > 5:
                    raise ValueError('MODE_SET must be in 0..5')

    def _process_control_events(self, control_frame: 'ControlFrame') -> None:
        """Release before allocation, preserving order within each event kind."""
        from ..control_frame import CaptureMark, EventType

        for event in sorted(control_frame.events, key=lambda e: e.type != EventType.NOTE_OFF):
            note = event.a & 0x7F
            if event.type == EventType.NOTE_OFF:
                from .unison import group_slots
                addressed = None if event.voice_id == 0xffff else group_slots(self, event.voice_id)
                for vid, state in self.voices.items():
                    if (state.phase != VoicePhase.FREE and state.note_u8 == note
                            and state.channel == event.channel
                            and (addressed is None or vid in addressed)):
                        if state.engine_mode == int(EngineMode.SIMPLE):
                            self.note_off_simple(vid)
                        else:
                            self.note_off(vid)
            elif event.type == EventType.NOTE_ON:
                from .unison import allocate_note_group
                slots = allocate_note_group(self, note, event.channel, event.voice_id)
                velocity_q15 = min(Q15_ONE, (event.a >> 8) & 0xFFFF)
                for vid in slots:
                    state = self.voices[vid]
                    state.channel = event.channel
                    if state.engine_mode == int(EngineMode.SIMPLE):
                        self.note_on_simple(vid, note, velocity_q15)
                    else:
                        state.note_u8 = note
                        self.note_on(vid, gesture=GestureBasis(gate=True, pressure_q15=velocity_q15))
            elif event.type == EventType.PRESET_LOAD:
                self.selected_preset = self._control_presets[event.a][0]
                self.selected_preset_id = event.a
            elif event.type == EventType.MODE_SET:
                self.control_mode_L = event.a
            elif event.type == EventType.CAPTURE_MARK:
                self._capture_marks.append(CaptureMark(
                    block_index=control_frame.block_index, mark_id=event.a,
                    channel=event.channel, voice_id=event.voice_id,
                ))

    def _prepare_control_voice(self, voice_id: int) -> int:
        """Apply the selected preset when a slot starts or retriggers a note."""
        state = self.voices.get(voice_id)
        if self.selected_preset is not None and (
                state is None or state.compiled_preset != self.selected_preset):
            self.allocate_voice_from_preset(self.selected_preset, voice_id)
        elif state is None:
            self.allocate_voice(voice_id)
        return voice_id

    def _find_or_allocate_voice(self, note: int, channel: int = 0) -> Optional[int]:
        """Reuse the same note/channel, then a free slot, then a release tail."""
        existing = self._find_voice_by_note(note, channel)
        if existing is not None:
            return self._prepare_control_voice(existing)
        for vid in range(self.max_voices):
            if vid not in self.voices or self.voices[vid].phase == VoicePhase.FREE:
                return self._prepare_control_voice(vid)
        releasing = [v for v in self.voices.values() if v.phase == VoicePhase.RELEASE]
        if releasing:
            vid = max(releasing, key=lambda v: (v.phase_block, -v.voice_id)).voice_id
            return self._prepare_control_voice(vid)
        return None

    def _find_voice_by_note(self, note: int, channel: int = 0) -> Optional[int]:
        for vid, state in self.voices.items():
            if (state.phase not in (VoicePhase.FREE, VoicePhase.RELEASE)
                    and state.note_u8 == note and state.channel == channel):
                if state.unison is not None:
                    leader = self.voices.get(state.unison.leader_id)
                    if (leader is not None and leader.unison is not None
                            and leader.unison.group_id == state.unison.group_id
                            and leader.phase not in (VoicePhase.FREE, VoicePhase.RELEASE)):
                        return leader.voice_id
                return vid
        return None

    def _extract_param_frame(self, control_frame: Optional['ControlFrame']) -> 'ParamFrame':
        """Return latched globals; the renderer applies master gain after FX."""
        from ..frames import ParamFrame
        return ParamFrame(gate_q15=self.master_gain_q15)

    def _abi_to_voice_frame(self, abi: 'VoiceFrameABI', voice_id: int) -> 'VoiceFrame':
        """
        Convert VoiceFrameABI to VoiceFrame for render batch.

        VoiceFrameABI is the internal CMMM output format.
        VoiceFrame is the unified format for the execution layer.
        """
        from ..frames import VoiceFrame
        from ..tables.coefficient_tail import compile_tail_policy

        state = self.voices.get(voice_id)

        frame = VoiceFrame(
            voice_id=voice_id,
            gain_q15=abi.gain_q15,
            sendA_q15=abi.sendA_q15,
            sendB_q15=abi.sendB_q15,
            pan_q15=max(Q15_NEG_ONE, min(Q15_ONE, abi.pan_q15 + self.master_pan_q15)),
            tree_a_id=abi.tree_a_id,
            tree_b_id=abi.tree_b_id,
            tree_blend_q15=abi.tree_blend_q15,
            is_active=abi.is_active,
            is_starting=abi.is_starting,
            is_stopping=abi.is_stopping,
            tilt_env_q15=abi.tilt_env_q15,
            density_env_q15=abi.density_env_q15,
            cmmm_events=abi.events,
            event_commands=abi.event_commands,
            tail_policy=compile_tail_policy(
                abi.normalize_curve, abi.fineclamp_cap_ratio, abi.slew_max_delta),
            note=abi.note_u8,
            cents=abi.cents_i16,
            f0_hz=440.0 * 2.0 ** ((abi.note_u8 - 69 + abi.cents_i16 / 100.0) / 12.0),
        )

        # Copy additional fields from state if available
        if state is not None:
            from ..primitives import (
                FAMILY_TABLE, OPERATOR_TABLE, get_operators_for_state,
                compute_timbral_mode_from_intent,
            )
            mode = compute_timbral_mode_from_intent(state.intent.lanes)
            frame.ops_enable = [OPERATOR_TABLE[op_id].name for op_id in
                                get_operators_for_state(abi.regime_id, mode)]
            frame.families = {FAMILY_TABLE[slot].name: weight / Q15_ONE
                              for slot, weight in enumerate(abi.w_family_q15)
                              if slot in FAMILY_TABLE}
            if hasattr(state, 'compiled_preset') and state.compiled_preset:
                frame.beta_target_q15 = state.compiled_preset.material.beta_target_q15
                frame.partial_amps_q15 = list(state.compiled_preset.harmonic_amps_q15)
            state.controls.apply_frame(frame)

        return frame


# =============================================================================
# DEMO
# =============================================================================

def demo_voice_engine():
    """Demonstrate Voice Engine functionality."""
    print("=" * 60)
    print("VOICE ENGINE DEMO")
    print("=" * 60)

    # Create engine with Glitch Violin program
    engine = VoiceEngine(max_voices=4)
    voice_id = engine.allocate_voice(program_id=1001)
    print(f"Allocated voice {voice_id} with Glitch Violin program")

    # Note on with expressive gesture
    gesture = GestureBasis(
        gate=True,
        pressure_q15=int(Q15_ONE * 0.7),
        speed_q15=int(Q15_ONE * 0.5),
        tilt_q15=int(Q15_ONE * 0.3),
        pan_q15=int(Q15_ONE * -0.2),
    )
    engine.note_on(voice_id, gesture=gesture)
    print("Note ON with expressive gesture")

    # Simulate lifecycle
    print("\nSimulating note lifecycle:")
    print("-" * 40)

    for block in range(100):
        frame = engine.step_voice(voice_id)

        if frame:
            # Print every 10 blocks
            if block % 10 == 0:
                regime_name = frame.regime.name if hasattr(frame.regime, 'name') else str(frame.regime)
                gain_pct = (frame.gain_q15 / Q15_ONE) * 100
                print(f"Block {block:3d}: regime={regime_name:20s} gain={gain_pct:5.1f}% "
                      f"tree_a={frame.tree_a_id} tree_b={frame.tree_b_id} blend={frame.tree_blend_q15}")

        # Update gesture mid-note
        if block == 30:
            gesture.chaos_q15 = Q15_HALF
            engine.update_gesture(voice_id, gesture)
            print("\n>>> Increased chaos <<<\n")

        # Note off at block 60
        if block == 60:
            engine.note_off(voice_id)
            print("\n>>> Note OFF <<<\n")

    print("-" * 40)
    print(f"Final active voices: {engine.get_active_voice_count()}")


# =============================================================================
# MACRO → CMMM BRIDGE (SystemMacro integration per WSE-STD-MACRO v0)
# =============================================================================

def apply_macros_to_gesture(gesture: GestureBasis, macro_runtime: 'MacroRuntime') -> GestureBasis:
    """
    Apply SystemMacro values to a GestureBasis.

    This bridges MacroRuntime (controller state) to GestureBasis (CMMM input).

    Mapping (per WSE-STD-MACRO v0 §2):
    - BRIGHTNESS (4) → tilt_q15 (spectral tilt)
    - DENSITY (5) → pressure_q15 boost (body/sustain)
    - RISK (6) → chaos_q15 (stochastic behavior)
    - MOTION (2) → speed_q15 boost (movement)
    - SPACE (8) → position_q15 spread (stereo width)
    - MATERIAL (9) → tilt_q15 secondary (wood=negative, glass=positive)

    Args:
        gesture: Base gesture from controller input
        macro_runtime: MacroRuntime with current slewed macro values

    Returns:
        Modified GestureBasis with macro influence applied
    """
    from ..macro_runtime import SystemMacro

    # Get current macro values (post-slew)
    brightness = macro_runtime.get_system_macro(SystemMacro.BRIGHTNESS)
    density = macro_runtime.get_system_macro(SystemMacro.DENSITY)
    risk = macro_runtime.get_system_macro(SystemMacro.RISK)
    motion = macro_runtime.get_system_macro(SystemMacro.MOTION)
    space = macro_runtime.get_system_macro(SystemMacro.SPACE)
    material = macro_runtime.get_system_macro(SystemMacro.MATERIAL)

    # BRIGHTNESS → tilt (0=dark, 32767=bright)
    # Map 0-32767 to -16384..+16384 tilt offset
    brightness_offset = brightness - Q15_HALF
    new_tilt = sat16(gesture.tilt_q15 + brightness_offset)

    # MATERIAL modifies tilt further (wood=low=dark, glass=high=bright)
    # 0=wood (-4096), 16384=neutral (0), 32767=glass (+4096)
    material_offset = (material - Q15_HALF) >> 2
    new_tilt = sat16(new_tilt + material_offset)

    # DENSITY → pressure boost (more body = more sustained pressure)
    # Add up to 25% pressure boost at full density
    density_boost = mul_q15_sat(density, Q15_ONE >> 2)
    new_pressure = sat16(gesture.pressure_q15 + density_boost)

    # RISK → chaos (direct mapping, additive)
    # RISK at 32767 adds full chaos
    new_chaos = sat16(gesture.chaos_q15 + (risk >> 1))

    # MOTION → speed boost (movement/modulation)
    # MOTION adds up to 50% speed
    motion_boost = mul_q15_sat(motion, Q15_HALF)
    new_speed = sat16(gesture.speed_q15 + motion_boost)

    # SPACE → position spread (subtle stereo widening)
    # At high SPACE, amplify position; at low, center
    space_factor = space  # 0=center, 32767=wide
    new_position = mul_q15_sat(gesture.position_q15, space_factor >> 1 + Q15_HALF)

    return GestureBasis(
        gate=gesture.gate,
        pressure_q15=new_pressure,
        position_q15=new_position,
        speed_q15=new_speed,
        tilt_q15=new_tilt,
        chaos_q15=new_chaos,
        pitch_q15=gesture.pitch_q15,
        pan_q15=gesture.pan_q15,
    )


def apply_macros_to_intent(intent: IntentVector, macro_runtime: 'MacroRuntime') -> IntentVector:
    """
    Apply SystemMacro values directly to IntentVector.

    This is an alternative to apply_macros_to_gesture when you want
    direct semantic control over intent lanes.

    Mapping:
    - ENERGY (1) -> I_TONE, I_BOW boost
    - FOCUS (3) -> I_TONE boost, I_AIR reduction
    - RISK (6) -> I_GLITCH boost
    - AGE (7) -> I_CALM boost, I_SNAP reduction (older voices settle)
    - UNITY (10) -> I_GLUE boost

    Args:
        intent: Base intent computed from gesture
        macro_runtime: MacroRuntime with current macro values

    Returns:
        Modified IntentVector with macro influence applied
    """
    from ..macro_runtime import SystemMacro

    # Get macro values
    energy = macro_runtime.get_system_macro(SystemMacro.ENERGY)
    focus = macro_runtime.get_system_macro(SystemMacro.FOCUS)
    risk = macro_runtime.get_system_macro(SystemMacro.RISK)
    age = macro_runtime.get_system_macro(SystemMacro.AGE)
    unity = macro_runtime.get_system_macro(SystemMacro.UNITY)

    # Create modified intent
    new_lanes = list(intent.lanes)

    # ENERGY -> boost TONE and BOW
    intent_boost = energy >> 2  # Up to 25% boost
    new_lanes[I_TONE] = sat16(new_lanes[I_TONE] + intent_boost)
    new_lanes[I_BOW] = sat16(new_lanes[I_BOW] + intent_boost)

    # FOCUS -> boost TONE, reduce AIR (concentrated vs diffuse)
    focus_offset = focus - Q15_HALF
    new_lanes[I_TONE] = sat16(new_lanes[I_TONE] + (focus_offset >> 2))
    new_lanes[I_AIR] = sat16(new_lanes[I_AIR] - (focus_offset >> 2))

    # RISK -> boost GLITCH
    new_lanes[I_GLITCH] = sat16(new_lanes[I_GLITCH] + (risk >> 1))

    # AGE -> older voices become calmer, less snappy (natural settling)
    # High AGE boosts CALM, reduces SNAP
    age_effect = age >> 3  # Up to 12.5% effect
    new_lanes[I_CALM] = sat16(new_lanes[I_CALM] + age_effect)
    new_lanes[I_SNAP] = sat16(new_lanes[I_SNAP] - (age_effect >> 1))

    # UNITY -> boost GLUE (coherence)
    new_lanes[I_GLUE] = sat16(new_lanes[I_GLUE] + (unity >> 2))

    return IntentVector(lanes=new_lanes)


def gesture_to_intent_with_macros(
    gesture: GestureBasis,
    macro_runtime: 'MacroRuntime',
    gesture_to_intent_fn: callable = None
) -> IntentVector:
    """
    Complete pipeline: Gesture + Macros → Intent.

    This is the recommended way to compute intent when using MacroRuntime.

    1. Apply macros to gesture (BRIGHTNESS, DENSITY, RISK, etc.)
    2. Compute base intent from modified gesture
    3. Apply macro refinements to intent (ENERGY_INTENT, FOCUS, UNITY)

    Args:
        gesture: Raw gesture from controller
        macro_runtime: MacroRuntime with current macro state
        gesture_to_intent_fn: Custom gesture→intent function (default: default_gesture_to_intent)

    Returns:
        IntentVector with full macro influence
    """
    if gesture_to_intent_fn is None:
        gesture_to_intent_fn = default_gesture_to_intent

    # Step 1: Apply macros to gesture
    modified_gesture = apply_macros_to_gesture(gesture, macro_runtime)

    # Step 2: Compute base intent
    base_intent = gesture_to_intent_fn(modified_gesture)

    # Step 3: Apply macro refinements to intent
    final_intent = apply_macros_to_intent(base_intent, macro_runtime)

    return final_intent


if __name__ == "__main__":
    # Run demo (tests moved to pipeline_tests/test_voice_engine.py)
    demo_voice_engine()
