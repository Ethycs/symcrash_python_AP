"""
Voice Engine Master Spec v1.0
=============================

Owns all musical time and note lifecycle. Emits VoiceFrames per block.
Implements Option A Tree Morphing and Four Compiled Instruments.

The Voice Engine is the ONLY layer that knows about:
- Note lifecycle (on/off, legato, retrigger)
- Musical timing (tempo, phase, "every N blocks")
- GestureBasis (8 physical gesture inputs)
- IntentVector (8 semantic intent lanes)
- Regime FSM (R0-R6 states)
- Tree trajectory over note lifecycle

Downstream engines (Symbol, Wavelet, Safety) receive VoiceFrames
and treat them as stateless inputs - no musical time awareness.
"""

from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np

from fixed import (
    sat16, mul_q15_sat, slew_q15,
    Q15_ONE, Q15_HALF, Q15_SHIFT
)

# =============================================================================
# CONSTANTS
# =============================================================================

BLOCK_SIZE = 64
SAMPLE_RATE = 48000.0
BLOCK_TIME = BLOCK_SIZE / SAMPLE_RATE  # ~1.33ms

NUM_INTENTS = 8
NUM_FAMILIES = 8
NUM_EVENTS = 8
NUM_TREES = 5

# =============================================================================
# REGIME SYSTEM (R0-R6)
# =============================================================================

class Regime(IntEnum):
    """Voice regime states."""
    R0_IDLE = 0              # Voice silent, no processing
    R1_ATTACK_IGNITION = 1   # Note-on transient burst
    R2_STABLE_TONE = 2       # Sustained pitched tone
    R3_GRAIN_AIR = 3         # Breathy, textured sustain
    R4_GLITCH_CHATTER = 4    # Intentional artifacts
    R5_CATASTROPHE = 5       # Extreme morphing/chaos
    R6_RECOVERY_HEAL = 6     # Return to stability


# Regime transition matrix: from_regime -> valid_to_regimes
REGIME_TRANSITIONS = {
    Regime.R0_IDLE: {Regime.R1_ATTACK_IGNITION},
    Regime.R1_ATTACK_IGNITION: {Regime.R2_STABLE_TONE, Regime.R3_GRAIN_AIR, Regime.R4_GLITCH_CHATTER},
    Regime.R2_STABLE_TONE: {Regime.R0_IDLE, Regime.R3_GRAIN_AIR, Regime.R4_GLITCH_CHATTER, Regime.R5_CATASTROPHE},
    Regime.R3_GRAIN_AIR: {Regime.R0_IDLE, Regime.R2_STABLE_TONE, Regime.R4_GLITCH_CHATTER, Regime.R5_CATASTROPHE},
    Regime.R4_GLITCH_CHATTER: {Regime.R0_IDLE, Regime.R2_STABLE_TONE, Regime.R3_GRAIN_AIR, Regime.R5_CATASTROPHE},
    Regime.R5_CATASTROPHE: {Regime.R6_RECOVERY_HEAL},
    Regime.R6_RECOVERY_HEAL: {Regime.R0_IDLE, Regime.R2_STABLE_TONE, Regime.R3_GRAIN_AIR},
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

# Intent lane indices
I_TONE = 0     # Pitched tone strength
I_BOW = 1      # Bowing/sustain pressure
I_AIR = 2      # Breathiness/noise
I_GLITCH = 3   # Glitch/artifact desire
I_WARP = 4     # Time/pitch warping
I_SNAP = 5     # Transient snap
I_GLUE = 6     # Coherence/binding
I_CALM = 7     # Recovery/simplification


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

# Family indices (match symbol_engine.py)
F_RESERVED = 0   # Reserved / identity
F_COLOR = 1      # Timbral color (Tilt, FineClamp)
F_TRANSPORT = 2  # Energy transport (DiffuseUp/Down, BandShear)
F_LOCAL = 3      # Local mixing (IntraMix, LowRankMorph)
F_TOPOLOGY = 4   # Structural (SwapBands, RotateNodes, Permute)
F_GLUE = 5       # Coherence (GluePair, NormalizeScale)
F_EVENTS = 6     # Impulses, noise (Impulse, NoiseBand, etc.)
F_CATASTROPHE = 7  # Emergency (EmergencySimplify, FoldPressure)


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
    if regime == Regime.R1_ATTACK_IGNITION:
        # Boost snap/events during attack
        weights[F_EVENTS] = sat16(weights[F_EVENTS] + Q15_HALF)
        weights[F_COLOR] = sat16(weights[F_COLOR] + Q15_HALF // 2)

    elif regime == Regime.R2_STABLE_TONE:
        # Boost tone families, reduce chaos
        weights[F_COLOR] = sat16(weights[F_COLOR] + Q15_HALF // 2)
        weights[F_LOCAL] = sat16(weights[F_LOCAL] + Q15_HALF // 2)
        weights[F_GLUE] = sat16(weights[F_GLUE] + Q15_HALF // 2)
        weights[F_TOPOLOGY] = weights[F_TOPOLOGY] >> 1

    elif regime == Regime.R3_GRAIN_AIR:
        # Boost air/transport
        weights[F_TRANSPORT] = sat16(weights[F_TRANSPORT] + Q15_HALF)
        weights[F_EVENTS] = sat16(weights[F_EVENTS] + Q15_HALF // 2)

    elif regime == Regime.R4_GLITCH_CHATTER:
        # Boost topology and events
        weights[F_TOPOLOGY] = sat16(weights[F_TOPOLOGY] + Q15_HALF)
        weights[F_EVENTS] = sat16(weights[F_EVENTS] + Q15_HALF)

    elif regime == Regime.R6_RECOVERY_HEAL:
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


@dataclass
class EventPayload:
    """Event lane payloads for Symbol Engine."""
    # E0: Impulse - band and strength
    impulse_band: int = 0
    impulse_strength_q15: int = 0

    # E1: Noise - band and amount
    noise_band: int = 0
    noise_amount_q15: int = 0

    # E2: Pulse - pressure value
    pulse_pressure_q15: int = 0

    # E3: Gate - mask bitmap
    gate_mask: int = 0xFF  # All bands open by default

    # E4: Cusp - injection amount
    cusp_amount_q15: int = 0

    # E5: Swallowtail - mask
    swallowtail_bands: int = 0

    # E6: Fold - pressure
    fold_pressure_q15: int = 0

    # E7: Sheaf - projection threshold
    sheaf_threshold_q15: int = 0

    # Which events are active this frame
    active_mask: int = 0


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
    if regime == Regime.R1_ATTACK_IGNITION:
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

    # Swallowtail: during glitch regime
    if regime == Regime.R4_GLITCH_CHATTER:
        events.swallowtail_bands = 0x3C  # Mid bands
        events.active_mask |= (1 << E_SWALLOWTAIL)

    # Fold: during catastrophe
    if regime == Regime.R5_CATASTROPHE:
        events.fold_pressure_q15 = Q15_ONE
        events.active_mask |= (1 << E_FOLD)

    # Sheaf: during recovery
    if regime == Regime.R6_RECOVERY_HEAL:
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
    if regime == Regime.R1_ATTACK_IGNITION:
        scores[3] += Q15_HALF  # Transient
    elif regime == Regime.R2_STABLE_TONE:
        scores[1] += Q15_HALF  # Harmonic
    elif regime == Regime.R3_GRAIN_AIR:
        scores[2] += Q15_HALF  # Texture
    elif regime == Regime.R4_GLITCH_CHATTER:
        scores[2] += Q15_HALF // 2
        scores[3] += Q15_HALF // 2
    elif regime == Regime.R5_CATASTROPHE:
        scores[2] += Q15_HALF  # Texture for chaos
    elif regime == Regime.R6_RECOVERY_HEAL:
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
# INSTRUMENT PROGRAMS (1001-1004)
# =============================================================================

@dataclass
class InstrumentProgram:
    """
    Compiled instrument program specification.

    Defines how gesture maps to intent, regime transitions,
    family weights, and tree selection.
    """
    program_id: int = 0
    name: str = "Default"

    # Gesture to intent mapper (can be customized per instrument)
    gesture_to_intent: Callable[[GestureBasis], IntentVector] = field(
        default_factory=lambda: default_gesture_to_intent
    )

    # Regime transition rules
    attack_regime: Regime = Regime.R1_ATTACK_IGNITION
    sustain_regime: Regime = Regime.R2_STABLE_TONE
    release_regime: Regime = Regime.R0_IDLE

    # Family weight modifiers (multipliers in Q15)
    family_mods: List[int] = field(default_factory=lambda: [Q15_ONE] * NUM_FAMILIES)

    # Tree preferences
    preferred_tree: int = 0
    tree_morph_rate: int = 512

    # ADSR timings (in blocks)
    attack_blocks: int = 8
    decay_blocks: int = 32
    sustain_level_q15: int = Q15_HALF + Q15_HALF // 2  # 0.75
    release_blocks: int = 64

    # Event modifiers
    event_probability: List[int] = field(default_factory=lambda: [Q15_HALF] * NUM_EVENTS)


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
        attack_regime=Regime.R1_ATTACK_IGNITION,
        sustain_regime=Regime.R2_STABLE_TONE,
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
        attack_regime=Regime.R3_GRAIN_AIR,  # Start textured
        sustain_regime=Regime.R3_GRAIN_AIR,
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
        attack_regime=Regime.R1_ATTACK_IGNITION,
        sustain_regime=Regime.R3_GRAIN_AIR,  # Sustain is textured
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
        attack_regime=Regime.R1_ATTACK_IGNITION,
        sustain_regime=Regime.R2_STABLE_TONE,
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
# VOICE STATE
# =============================================================================

class VoicePhase(IntEnum):
    """Voice lifecycle phases (internal ADSR)."""
    FREE = 0
    ATTACK = 1
    DECAY = 2
    SUSTAIN = 3
    RELEASE = 4


@dataclass
class VoiceState:
    """
    Complete per-voice state.
    """
    voice_id: int = 0

    # Lifecycle
    phase: VoicePhase = VoicePhase.FREE
    phase_block: int = 0        # Blocks in current phase

    # Current gesture
    gesture: GestureBasis = field(default_factory=GestureBasis)

    # Computed intent
    intent: IntentVector = field(default_factory=IntentVector)

    # Regime FSM
    regime: Regime = Regime.R0_IDLE
    regime_blocks: int = 0      # Blocks in current regime

    # Family weights
    family_weights: List[int] = field(default_factory=lambda: [0] * NUM_FAMILIES)

    # Event payload
    events: EventPayload = field(default_factory=EventPayload)

    # Tree morph
    tree_morph: TreeMorphState = field(default_factory=TreeMorphState)

    # Envelope values
    gain_q15: int = 0

    # Instrument program
    program: InstrumentProgram = field(default_factory=InstrumentProgram)

    # Block counter (for timing)
    block_count: int = 0


# =============================================================================
# VOICEFRAME OUTPUT ABI
# =============================================================================

@dataclass
class VoiceFrameABI:
    """
    VoiceFrame output per the spec's ABI.

    This is what Voice Engine emits per block per voice.
    Symbol Engine and Wavelet Engine consume this.
    """
    voice_id: int = 0

    # Tree control (Option A)
    tree_a_id: int = 0           # Primary tree ID (0-4)
    tree_b_id: int = 0           # Secondary tree ID (during morph)
    tree_blend_q15: int = 0      # Blend factor (0 = pure A)

    # Amplitude envelope
    gain_q15: int = 0

    # Pan
    pan_q15: int = 0

    # Family weights for Symbol Engine
    family_weights: List[int] = field(default_factory=lambda: [0] * NUM_FAMILIES)

    # Event payload
    events: EventPayload = field(default_factory=EventPayload)

    # Musical envelopes
    tilt_env_q15: int = Q15_HALF
    density_env_q15: int = Q15_HALF

    # Regime (for Symbol Engine degrade policy)
    regime: Regime = Regime.R0_IDLE

    # Voice state flags
    is_starting: bool = False
    is_stopping: bool = False
    is_active: bool = False

    # Material / program ID
    program_id: int = 0


# =============================================================================
# VOICE ENGINE
# =============================================================================

class VoiceEngine:
    """
    Voice Engine Master Implementation.

    Owns all musical time and emits VoiceFrameABI per block per voice.
    Implements the full spec with GestureBasis, IntentVector, Regime FSM,
    family weights, event lanes, and Option A tree morphing.
    """

    def __init__(self, max_voices: int = 12, block_size: int = BLOCK_SIZE,
                 sample_rate: float = SAMPLE_RATE):
        self.max_voices = max_voices
        self.block_size = block_size
        self.sample_rate = sample_rate
        self.block_time = block_size / sample_rate

        # Voice pool
        self.voices: Dict[int, VoiceState] = {}

        # Global block counter
        self.block_count: int = 0

        # Default program
        self.default_program = InstrumentProgram()

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
        program = INSTRUMENT_PROGRAMS.get(program_id, self.default_program)

        # Create voice state
        state = VoiceState(voice_id=voice_id, program=program)
        state.tree_morph.morph_rate = program.tree_morph_rate
        state.tree_morph.tree_a_id = program.preferred_tree

        self.voices[voice_id] = state
        return voice_id

    def note_on(self, voice_id: int, gesture: Optional[GestureBasis] = None,
                velocity_q15: int = Q15_HALF, pitch_q15: int = 0, pan_q15: int = 0):
        """
        Trigger note-on for a voice with gesture input.
        """
        if voice_id not in self.voices:
            self.allocate_voice(voice_id)

        state = self.voices[voice_id]

        # Set gesture
        if gesture is not None:
            state.gesture = gesture
        else:
            state.gesture = GestureBasis.from_note_on(velocity_q15, pitch_q15, pan_q15)

        # Start attack phase
        state.phase = VoicePhase.ATTACK
        state.phase_block = 0

        # Transition to attack regime
        self._transition_regime(state, state.program.attack_regime)

    def note_off(self, voice_id: int):
        """
        Trigger note-off for a voice.
        """
        if voice_id not in self.voices:
            return

        state = self.voices[voice_id]
        state.gesture.gate = False

        # Start release phase
        if state.phase != VoicePhase.FREE:
            state.phase = VoicePhase.RELEASE
            state.phase_block = 0

    def update_gesture(self, voice_id: int, gesture: GestureBasis):
        """
        Update gesture for a voice (continuous control).
        """
        if voice_id not in self.voices:
            return

        state = self.voices[voice_id]
        # Preserve gate state from current if new gesture doesn't specify
        if not gesture.gate and state.gesture.gate:
            gesture.gate = True
        state.gesture = gesture

    def _transition_regime(self, state: VoiceState, new_regime: Regime):
        """
        Attempt regime transition if valid.
        """
        if new_regime in REGIME_TRANSITIONS.get(state.regime, set()):
            state.regime = new_regime
            state.regime_blocks = 0

    def _compute_regime_from_intent(self, state: VoiceState) -> Regime:
        """
        Determine target regime from current intent.
        """
        intent = state.intent
        current = state.regime

        # Check for catastrophe trigger (very high glitch + warp)
        if intent.glitch > int(Q15_ONE * 0.8) and intent.warp > int(Q15_ONE * 0.6):
            if current in {Regime.R2_STABLE_TONE, Regime.R3_GRAIN_AIR, Regime.R4_GLITCH_CHATTER}:
                return Regime.R5_CATASTROPHE

        # Check for glitch regime
        if intent.glitch > int(Q15_ONE * 0.6):
            if current in REGIME_TRANSITIONS.get(current, set()) and Regime.R4_GLITCH_CHATTER in REGIME_TRANSITIONS[current]:
                return Regime.R4_GLITCH_CHATTER

        # Check for grain/air regime
        if intent.air > int(Q15_ONE * 0.6):
            if Regime.R3_GRAIN_AIR in REGIME_TRANSITIONS.get(current, set()):
                return Regime.R3_GRAIN_AIR

        # Recovery from catastrophe
        if current == Regime.R5_CATASTROPHE and state.regime_blocks > 64:
            return Regime.R6_RECOVERY_HEAL

        # Recovery to stable
        if current == Regime.R6_RECOVERY_HEAL and state.regime_blocks > 32:
            if intent.calm > Q15_HALF:
                return Regime.R2_STABLE_TONE

        # Default sustain regime
        if current == Regime.R1_ATTACK_IGNITION:
            if state.phase_block > state.program.attack_blocks:
                return state.program.sustain_regime

        return current  # No change

    def _step_envelope(self, state: VoiceState) -> int:
        """
        Step ADSR envelope, return gain in Q15.
        """
        program = state.program

        if state.phase == VoicePhase.FREE:
            return 0

        state.phase_block += 1

        if state.phase == VoicePhase.ATTACK:
            if state.phase_block >= program.attack_blocks:
                state.phase = VoicePhase.DECAY
                state.phase_block = 0
                return Q15_ONE
            else:
                # Linear attack
                return (state.phase_block * Q15_ONE) // max(1, program.attack_blocks)

        elif state.phase == VoicePhase.DECAY:
            if state.phase_block >= program.decay_blocks:
                state.phase = VoicePhase.SUSTAIN
                state.phase_block = 0
                return program.sustain_level_q15
            else:
                # Linear decay from peak to sustain
                progress = state.phase_block / max(1, program.decay_blocks)
                return int(Q15_ONE * (1 - progress) + program.sustain_level_q15 * progress)

        elif state.phase == VoicePhase.SUSTAIN:
            return program.sustain_level_q15

        elif state.phase == VoicePhase.RELEASE:
            if state.phase_block >= program.release_blocks:
                state.phase = VoicePhase.FREE
                return 0
            else:
                # Linear release from sustain to zero
                progress = state.phase_block / max(1, program.release_blocks)
                return int(program.sustain_level_q15 * (1 - progress))

        return 0

    def step_voice(self, voice_id: int) -> Optional[VoiceFrameABI]:
        """
        Step a single voice by one block.

        Full pipeline:
        1. Update gesture → intent
        2. Compute regime transition
        3. Compute family weights
        4. Compute events
        5. Update tree morph
        6. Step envelope
        7. Emit VoiceFrameABI
        """
        if voice_id not in self.voices:
            return None

        state = self.voices[voice_id]

        if state.phase == VoicePhase.FREE:
            return None

        # Track for is_starting / is_stopping
        old_phase = state.phase

        # 1. Gesture → Intent
        state.intent = state.program.gesture_to_intent(state.gesture)

        # 2. Regime transition
        target_regime = self._compute_regime_from_intent(state)
        if target_regime != state.regime:
            self._transition_regime(state, target_regime)
        state.regime_blocks += 1

        # 3. Family weights
        base_weights = compute_family_weights(state.intent, state.regime)
        # Apply program modifiers
        state.family_weights = [
            mul_q15_sat(base_weights[i], state.program.family_mods[i])
            for i in range(NUM_FAMILIES)
        ]

        # 4. Events
        state.events = compute_events(state.intent, state.regime, state.block_count)

        # 5. Tree morph
        target_tree = select_tree_from_intent(state.intent, state.regime)
        state.tree_morph = update_tree_morph(state.tree_morph, target_tree)

        # 6. Envelope
        state.gain_q15 = self._step_envelope(state)

        # 7. Update block counter
        state.block_count += 1

        # Detect phase transitions
        new_phase = state.phase
        is_starting = (old_phase == VoicePhase.FREE or old_phase == VoicePhase.ATTACK) and new_phase != VoicePhase.FREE
        is_stopping = new_phase == VoicePhase.FREE and old_phase != VoicePhase.FREE

        # Compute musical envelopes from intent
        tilt_env = (state.intent.tone + state.intent.bow) >> 1
        density_env = (state.intent.air + state.intent.glitch) >> 1

        return VoiceFrameABI(
            voice_id=voice_id,
            tree_a_id=state.tree_morph.tree_a_id,
            tree_b_id=state.tree_morph.tree_b_id,
            tree_blend_q15=state.tree_morph.blend_q15,
            gain_q15=state.gain_q15,
            pan_q15=state.gesture.pan_q15,
            family_weights=list(state.family_weights),
            events=state.events,
            tilt_env_q15=tilt_env,
            density_env_q15=density_env,
            regime=state.regime,
            is_starting=is_starting and state.block_count <= 1,
            is_stopping=is_stopping,
            is_active=state.phase != VoicePhase.FREE,
            program_id=state.program.program_id,
        )

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


# =============================================================================
# ACCEPTANCE TESTS
# =============================================================================

def test_gesture_to_intent():
    """Test: GestureBasis correctly maps to IntentVector."""
    print("Test: gesture_to_intent ... ", end="")

    # High pressure gesture
    g = GestureBasis(gate=True, pressure_q15=Q15_ONE, speed_q15=Q15_HALF)
    intent = default_gesture_to_intent(g)

    assert intent.bow == Q15_ONE, f"Expected bow={Q15_ONE}, got {intent.bow}"
    assert intent.tone > 0, "Tone should be positive with gate on"

    # High chaos gesture
    g2 = GestureBasis(gate=True, chaos_q15=Q15_ONE, pressure_q15=Q15_HALF)
    intent2 = default_gesture_to_intent(g2)

    assert intent2.glitch > 0, "Glitch should be positive with high chaos"
    assert intent2.glue < Q15_ONE, "Glue should decrease with chaos"

    print("PASS")


def test_regime_transitions():
    """Test: Regime FSM transitions correctly."""
    print("Test: regime_transitions ... ", end="")

    engine = VoiceEngine()
    engine.allocate_voice(0, program_id=1001)
    engine.note_on(0, velocity_q15=Q15_HALF)

    # Should start in attack regime
    frame = engine.step_voice(0)
    assert frame.regime == Regime.R1_ATTACK_IGNITION, f"Expected R1, got {frame.regime}"

    # Step through attack
    for _ in range(10):
        frame = engine.step_voice(0)

    # Should transition to sustain regime
    assert frame.regime in {Regime.R2_STABLE_TONE, Regime.R3_GRAIN_AIR}, \
        f"Expected sustain regime, got {frame.regime}"

    print("PASS")


def test_family_weights():
    """Test: Family weights computed correctly from intent."""
    print("Test: family_weights ... ", end="")

    intent = IntentVector()
    intent.lanes[I_TONE] = Q15_ONE
    intent.lanes[I_BOW] = Q15_HALF
    intent.lanes[I_GLUE] = Q15_ONE

    weights = compute_family_weights(intent, Regime.R2_STABLE_TONE)

    # Color should be high (tone + bow)
    assert weights[F_COLOR] > Q15_HALF, f"Expected high color, got {weights[F_COLOR]}"

    # Glue should be high
    assert weights[F_GLUE] > Q15_HALF, f"Expected high glue, got {weights[F_GLUE]}"

    # Catastrophe should be zero (not in R5)
    assert weights[F_CATASTROPHE] == 0, f"Expected zero catastrophe, got {weights[F_CATASTROPHE]}"

    print("PASS")


def test_tree_morph():
    """Test: Tree morph transitions correctly."""
    print("Test: tree_morph ... ", end="")

    state = TreeMorphState(tree_a_id=0)

    # Start morph to tree 2
    state = update_tree_morph(state, 2)
    assert state.is_morphing, "Should be morphing"
    assert state.tree_b_id == 2, f"Expected tree_b=2, got {state.tree_b_id}"

    # Step through morph
    while state.is_morphing:
        state = update_tree_morph(state, 2)

    # Should complete with A promoted to 2
    assert state.tree_a_id == 2, f"Expected tree_a=2, got {state.tree_a_id}"
    assert state.blend_q15 == 0, f"Expected blend=0, got {state.blend_q15}"

    print("PASS")


def test_instrument_programs():
    """Test: All four instrument programs load correctly."""
    print("Test: instrument_programs ... ", end="")

    for program_id in [1001, 1002, 1003, 1004]:
        engine = VoiceEngine()
        vid = engine.allocate_voice(program_id=program_id)
        assert vid is not None, f"Failed to allocate voice for program {program_id}"

        engine.note_on(vid, velocity_q15=Q15_HALF)
        frame = engine.step_voice(vid)

        assert frame is not None, f"No frame for program {program_id}"
        assert frame.program_id == program_id, f"Wrong program_id: {frame.program_id}"
        assert frame.is_active, f"Voice should be active for program {program_id}"

    print("PASS")


def test_voiceframe_abi():
    """Test: VoiceFrameABI emitted correctly."""
    print("Test: voiceframe_abi ... ", end="")

    engine = VoiceEngine()
    engine.allocate_voice(0, program_id=1001)

    gesture = GestureBasis(
        gate=True,
        pressure_q15=Q15_HALF,
        pan_q15=Q15_HALF // 2,  # Slight right
        chaos_q15=Q15_HALF // 4,
    )
    engine.note_on(0, gesture=gesture)

    frame = engine.step_voice(0)

    # Check all ABI fields are set
    assert frame.voice_id == 0
    assert 0 <= frame.tree_a_id < NUM_TREES
    assert 0 <= frame.tree_b_id < NUM_TREES
    assert 0 <= frame.tree_blend_q15 <= Q15_ONE
    assert frame.gain_q15 > 0, "Gain should be positive during attack"
    assert len(frame.family_weights) == NUM_FAMILIES
    assert frame.is_active

    print("PASS")


def run_acceptance_tests():
    """Run all Voice Engine acceptance tests."""
    print("=" * 60)
    print("VOICE ENGINE ACCEPTANCE TESTS")
    print("=" * 60)

    tests = [
        test_gesture_to_intent,
        test_regime_transitions,
        test_family_weights,
        test_tree_morph,
        test_instrument_programs,
        test_voiceframe_abi,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

    print("-" * 60)
    print(f"Results: {passed}/{len(tests)} passed")

    return failed == 0


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


if __name__ == "__main__":
    # Run acceptance tests first
    if run_acceptance_tests():
        print("\n")
        demo_voice_engine()
