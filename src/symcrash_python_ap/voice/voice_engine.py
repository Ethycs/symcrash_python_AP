"""
Voice Engine Instrument Master Implementation v3.0
===================================================

Implements a Controlled Semi-Markov Mealy Machine (CMMM) for voice control.
Conforms to Voice Engine Instrument Master Spec v1.0.

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

from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np

from ..fixed import (
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
MAX_ARCS = 8          # Maximum arc modulators per voice
MAX_VOICES = 128      # Maximum polyphony target


# =============================================================================
# DETERMINISTIC RNG (xorshift32 per spec Section 11)
# =============================================================================

def xorshift32(state: int) -> Tuple[int, int]:
    """
    Deterministic xorshift32 RNG.

    Per spec Section 11: All randomness derives from VoiceState.rng_u32.
    The RNG update is deterministic and documented.

    Returns: (new_state, random_value)
    """
    x = state & 0xFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17) & 0xFFFFFFFF
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0xFFFFFFFF, x & 0xFFFFFFFF


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

    def check(self, gesture: 'GestureBasis', intent: 'IntentVector') -> bool:
        """Check if condition is satisfied."""
        # Check gate conditions
        if self.gate_required is not None and gesture.gate != self.gate_required:
            return False
        if self.gate_rise_required and not getattr(gesture, 'gate_rise', False):
            return False
        if self.gate_fall_required and not getattr(gesture, 'gate_fall', False):
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

    def check_condition(self, intent: 'IntentVector', dwell_blocks: int,
                       gesture: Optional['GestureBasis'] = None) -> bool:
        """Check if transition condition is met."""
        if dwell_blocks < self.min_dwell_blocks:
            return False

        # Check new-style condition if present
        if gesture is not None and (self.condition.intent_thresholds or
                                    self.condition.gate_required is not None):
            return self.condition.check(gesture, intent)

        # Fall back to legacy thresholds
        for lane_id, threshold in self.intent_thresholds.items():
            if intent.lanes[lane_id] < threshold:
                return False

        return True


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

class LaneRuleType(IntEnum):
    """How an event lane is triggered."""
    DISABLED = 0       # Lane disabled
    GATE_EDGE = 1      # On gate rise/fall
    THRESHOLD = 2      # When intent exceeds threshold
    STOCHASTIC = 3     # Random with probability
    PATTERN = 4        # Pattern-driven by arc


@dataclass
class LaneRuleSpec:
    """
    Rule for a single event lane (per spec Section 8.3).
    """
    rule_type: LaneRuleType = LaneRuleType.DISABLED

    # For THRESHOLD: intent lane and threshold
    intent_lane: int = 0
    threshold_q15: int = Q15_HALF

    # For STOCHASTIC: rate (probability per block, Q15)
    rate_q15: int = 1000  # ~3% per block

    # For GATE_EDGE: which edge
    on_rise: bool = True
    on_fall: bool = False

    # For PATTERN: arc pattern ID
    pattern_id: int = 0

    # Strength modulation (Q15 multiplier)
    strength_mod_q15: int = Q15_ONE

    def check(self, intent: 'IntentVector', gesture: 'GestureBasis',
             rng_state: int) -> Tuple[bool, int, int]:
        """
        Check if event should fire.

        Returns: (fire, strength_q15, new_rng_state)
        """
        if self.rule_type == LaneRuleType.DISABLED:
            return False, 0, rng_state

        elif self.rule_type == LaneRuleType.GATE_EDGE:
            rise = getattr(gesture, 'gate_rise', False)
            fall = getattr(gesture, 'gate_fall', False)
            fire = (self.on_rise and rise) or (self.on_fall and fall)
            return fire, self.strength_mod_q15, rng_state

        elif self.rule_type == LaneRuleType.THRESHOLD:
            value = intent.lanes[self.intent_lane]
            if value >= self.threshold_q15:
                strength = mul_q15_sat(value, self.strength_mod_q15)
                return True, strength, rng_state
            return False, 0, rng_state

        elif self.rule_type == LaneRuleType.STOCHASTIC:
            new_rng, fire = rng_bool(rng_state, self.rate_q15)
            if fire:
                new_rng, strength = rng_q15(new_rng)
                strength = mul_q15_sat(strength, self.strength_mod_q15)
                return True, strength, new_rng
            return False, 0, new_rng

        return False, 0, rng_state


@dataclass
class EventPolicySpec:
    """
    Event generation policy (per spec Section 8.3).

    Output: (event_mask, event_strength_q15[8], event_payload_u64)
    """
    # Enable mask
    enable_mask: int = 0xFF

    # Per-lane rules
    lane_rules: List[LaneRuleSpec] = field(
        default_factory=lambda: [LaneRuleSpec() for _ in range(NUM_EVENTS)]
    )

    def compute(self, intent: 'IntentVector', gesture: 'GestureBasis',
               rng_state: int) -> Tuple[int, List[int], int, int]:
        """
        Compute event output.

        Returns: (event_mask, strengths[8], payload, new_rng_state)
        """
        mask = 0
        strengths = [0] * NUM_EVENTS
        payload = 0

        for i, rule in enumerate(self.lane_rules):
            if not (self.enable_mask & (1 << i)):
                continue

            fire, strength, rng_state = rule.check(intent, gesture, rng_state)
            if fire:
                mask |= (1 << i)
                strengths[i] = strength

        return mask, strengths, payload, rng_state


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

    # Tail Policy
    normalize_curve: str = "pink"  # "flat", "pink", "bright"
    fineclamp_cap_ratio: float = 0.5
    slew_max_delta: int = 512

    # Tower level for this regime
    tower_level: TowerLevel = TowerLevel.L0_FULL


# Default regime specs
REGIME_SPECS: Dict[Regime, RegimeSpec] = {
    Regime.R0_IDLE: RegimeSpec(
        regime=Regime.R0_IDLE,
        name="Idle",
        family_enable_mask=0x00,  # All disabled
        event_enable_mask=0x00,
        tower_level=TowerLevel.L4_SCALAR,
    ),
    Regime.R1_ATTACK_IGNITION: RegimeSpec(
        regime=Regime.R1_ATTACK_IGNITION,
        name="Attack",
        family_base=[0, Q15_HALF, 0, Q15_HALF // 2, 0, 0, Q15_HALF, 0],
        family_cap=[0, Q15_ONE, Q15_HALF, Q15_ONE, Q15_HALF, Q15_ONE, Q15_ONE, 0],
        preferred_tree=3,  # Transient
        event_enable_mask=0x07,  # Impulse, noise, pulse
        normalize_curve="bright",
        tower_level=TowerLevel.L1_COARSE,
    ),
    Regime.R2_STABLE_TONE: RegimeSpec(
        regime=Regime.R2_STABLE_TONE,
        name="Stable Tone",
        family_base=[0, Q15_HALF, 0, Q15_HALF, 0, Q15_HALF, 0, 0],
        family_cap=[0, Q15_ONE, Q15_HALF, Q15_ONE, Q15_HALF, Q15_ONE, Q15_HALF, 0],
        preferred_tree=1,  # Harmonic
        event_enable_mask=0x00,
        normalize_curve="pink",
        tower_level=TowerLevel.L0_FULL,
    ),
    Regime.R3_GRAIN_AIR: RegimeSpec(
        regime=Regime.R3_GRAIN_AIR,
        name="Grain/Air",
        family_base=[0, Q15_HALF // 2, Q15_HALF, Q15_HALF // 2, 0, Q15_HALF // 2, Q15_HALF // 2, 0],
        family_cap=[0, Q15_ONE, Q15_ONE, Q15_ONE, Q15_HALF, Q15_ONE, Q15_ONE, 0],
        preferred_tree=2,  # Texture
        event_enable_mask=0x02,  # Noise
        normalize_curve="flat",
        tower_level=TowerLevel.L1_COARSE,
    ),
    Regime.R4_GLITCH_CHATTER: RegimeSpec(
        regime=Regime.R4_GLITCH_CHATTER,
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
    Regime.R6_RECOVERY_HEAL: RegimeSpec(
        regime=Regime.R6_RECOVERY_HEAL,
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
                        dwell_blocks: int) -> Optional[Regime]:
        """Check if any transition should occur."""
        for edge in self.get_valid_transitions(from_regime):
            if edge.check_condition(intent, dwell_blocks):
                return edge.to_regime
        return None

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
            to_regime=Regime.R1_ATTACK_IGNITION,
            intent_thresholds={},  # Triggered by note_on
            priority=10,
            arc_id=1,  # Attack burst
        ),

        # From R1_ATTACK_IGNITION
        EdgeSpec(
            from_regime=Regime.R1_ATTACK_IGNITION,
            to_regime=Regime.R2_STABLE_TONE,
            intent_thresholds={I_TONE: Q15_HALF},
            min_dwell_blocks=4,
            priority=10,
        ),
        EdgeSpec(
            from_regime=Regime.R1_ATTACK_IGNITION,
            to_regime=Regime.R3_GRAIN_AIR,
            intent_thresholds={I_AIR: int(Q15_ONE * 0.6)},
            min_dwell_blocks=4,
            priority=8,
        ),
        EdgeSpec(
            from_regime=Regime.R1_ATTACK_IGNITION,
            to_regime=Regime.R4_GLITCH_CHATTER,
            intent_thresholds={I_GLITCH: int(Q15_ONE * 0.7)},
            min_dwell_blocks=4,
            priority=6,
        ),

        # From R2_STABLE_TONE
        # Note: R2->R0 is triggered by note_off(), not by intent conditions
        # So we don't include it here (it's handled explicitly)
        EdgeSpec(
            from_regime=Regime.R2_STABLE_TONE,
            to_regime=Regime.R3_GRAIN_AIR,
            intent_thresholds={I_AIR: int(Q15_ONE * 0.6)},
            min_dwell_blocks=8,
            priority=5,
        ),
        EdgeSpec(
            from_regime=Regime.R2_STABLE_TONE,
            to_regime=Regime.R4_GLITCH_CHATTER,
            intent_thresholds={I_GLITCH: int(Q15_ONE * 0.6)},
            min_dwell_blocks=8,
            priority=5,
        ),
        EdgeSpec(
            from_regime=Regime.R2_STABLE_TONE,
            to_regime=Regime.R5_CATASTROPHE,
            intent_thresholds={I_GLITCH: int(Q15_ONE * 0.8), I_WARP: int(Q15_ONE * 0.6)},
            min_dwell_blocks=16,
            priority=3,
            arc_id=2,  # Catastrophe entry
        ),

        # From R3_GRAIN_AIR
        # Note: R3->R0 is triggered by note_off(), not by intent conditions
        EdgeSpec(
            from_regime=Regime.R3_GRAIN_AIR,
            to_regime=Regime.R2_STABLE_TONE,
            intent_thresholds={I_TONE: int(Q15_ONE * 0.7), I_AIR: 0},
            min_dwell_blocks=8,
            priority=5,
        ),
        EdgeSpec(
            from_regime=Regime.R3_GRAIN_AIR,
            to_regime=Regime.R4_GLITCH_CHATTER,
            intent_thresholds={I_GLITCH: int(Q15_ONE * 0.6)},
            min_dwell_blocks=8,
            priority=5,
        ),
        EdgeSpec(
            from_regime=Regime.R3_GRAIN_AIR,
            to_regime=Regime.R5_CATASTROPHE,
            intent_thresholds={I_GLITCH: int(Q15_ONE * 0.8), I_WARP: int(Q15_ONE * 0.6)},
            min_dwell_blocks=16,
            priority=3,
            arc_id=2,
        ),

        # From R4_GLITCH_CHATTER
        # Note: R4->R0 is triggered by note_off(), not by intent conditions
        EdgeSpec(
            from_regime=Regime.R4_GLITCH_CHATTER,
            to_regime=Regime.R2_STABLE_TONE,
            intent_thresholds={I_CALM: int(Q15_ONE * 0.6)},
            min_dwell_blocks=8,
            priority=5,
        ),
        EdgeSpec(
            from_regime=Regime.R4_GLITCH_CHATTER,
            to_regime=Regime.R3_GRAIN_AIR,
            intent_thresholds={I_AIR: int(Q15_ONE * 0.7)},
            min_dwell_blocks=8,
            priority=5,
        ),
        EdgeSpec(
            from_regime=Regime.R4_GLITCH_CHATTER,
            to_regime=Regime.R5_CATASTROPHE,
            intent_thresholds={I_GLITCH: int(Q15_ONE * 0.9)},
            min_dwell_blocks=16,
            priority=3,
            arc_id=2,
        ),

        # From R5_CATASTROPHE
        EdgeSpec(
            from_regime=Regime.R5_CATASTROPHE,
            to_regime=Regime.R6_RECOVERY_HEAL,
            intent_thresholds={},  # Automatic after duration
            min_dwell_blocks=64,
            priority=10,
            arc_id=3,  # Recovery smooth
        ),

        # From R6_RECOVERY_HEAL
        # Note: R6->R0 is triggered by note_off(), not by intent conditions
        EdgeSpec(
            from_regime=Regime.R6_RECOVERY_HEAL,
            to_regime=Regime.R2_STABLE_TONE,
            intent_thresholds={I_CALM: int(Q15_ONE * 0.5)},
            min_dwell_blocks=32,
            priority=5,
        ),
        EdgeSpec(
            from_regime=Regime.R6_RECOVERY_HEAL,
            to_regime=Regime.R3_GRAIN_AIR,
            intent_thresholds={I_AIR: int(Q15_ONE * 0.5)},
            min_dwell_blocks=32,
            priority=5,
        ),
    ]

    return graph


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
    event_policy: EventPolicySpec = field(default_factory=EventPolicySpec)

    # ==========================================================================
    # REGIME GRAPH (per spec Section 7)
    # ==========================================================================

    # Regime specifications (per-regime settings)
    regime_specs: Dict[Regime, RegimeSpec] = field(default_factory=dict)

    # Regime graph (edges and arcs)
    regime_graph: Optional[RegimeGraph] = None

    # Legacy: simple regime transitions (used if regime_graph is None)
    attack_regime: Regime = Regime.R1_ATTACK_IGNITION
    sustain_regime: Regime = Regime.R2_STABLE_TONE
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
    normalize_curve: str = "pink"      # pink, white, brown, custom
    fineclamp_cap_ratio: float = 0.5   # Max deviation ratio
    slew_max_delta: int = 512          # Max sample-to-sample change

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

    def get_regime_spec(self, regime: Regime) -> RegimeSpec:
        """Get spec for regime, or create default."""
        if regime in self.regime_specs:
            return self.regime_specs[regime]
        return RegimeSpec(regime_id=regime)


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
    eta_q15: int = 16384  # Dry/wet blend (0=dry, 32767=wet)
    harmonics_q15: int = 0  # Harmonic mix (0=fundamental only)
    master_gain_q15: int = 24576  # Master gain override (~0.75)

    # Activity level (for voice stealing)
    activity_q15: int = 0

    # Send levels (for aux buses)
    sendA_q15: int = 0
    sendB_q15: int = 0

    # Overlay state (during transitions)
    overlay_active: bool = False
    overlay_blocks_remaining: int = 0
    overlay_spec: Optional[OverlaySpec] = None

    # Snapshot/replay support
    def snapshot(self) -> Dict:
        """
        Create a serializable snapshot of voice state.

        Returns dict that can be used with restore() for replay.
        All fields needed for deterministic replay are included.
        """
        return {
            'voice_id': self.voice_id,
            'phase': int(self.phase),
            'phase_block': self.phase_block,
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
    def restore(cls, snapshot: Dict) -> 'VoiceState':
        """
        Restore voice state from a snapshot.

        Enables deterministic replay from any saved state.
        All fields needed for replay are restored.
        """
        state = cls()
        state.voice_id = snapshot['voice_id']
        state.phase = VoicePhase(snapshot['phase'])
        state.phase_block = snapshot['phase_block']

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
        state.block_count = snapshot['block_count']

        # Restore new fields (with backwards compatibility defaults)
        state.rng_u32 = snapshot.get('rng_u32', 0x12345678)
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
        ep = EventPayload(active_mask=self.event_mask)
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

        # Global block counter
        self.block_count: int = 0

        # Regime graph (default)
        self.regime_graph = build_default_regime_graph()

        # Program registry
        self.programs = INSTRUMENT_PROGRAMS

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

        # Create voice state
        state = VoiceStateWithProgram(voice_id=voice_id, program=program, program_id=program_id)
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

        # Transition to attack regime with arc
        self._transition_regime(state, state.program.attack_regime, use_arc=True)

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

    def _transition_regime(self, state: VoiceStateWithProgram, new_regime: Regime,
                           use_arc: bool = True):
        """
        Execute regime transition with optional arc modulator.
        """
        if new_regime not in REGIME_TRANSITIONS.get(state.regime, set()):
            return  # Invalid transition

        # Record previous regime for arc lookup
        state.previous_regime = state.regime

        # Start arc if specified
        if use_arc:
            arc = self.regime_graph.get_arc(state.regime, new_regime)
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

        # Update tower level from regime spec
        regime_spec = self.regime_graph.regime_specs.get(new_regime)
        if regime_spec:
            state.tower_level = regime_spec.tower_level

    def _check_regime_transition(self, state: VoiceStateWithProgram) -> Optional[Regime]:
        """
        Check if any regime transition should occur using RegimeGraph.
        """
        # Use RegimeGraph for intent-based transitions
        target = self.regime_graph.check_transition(
            state.regime, state.intent, state.regime_blocks
        )

        if target is not None:
            return target

        # Legacy fallback for attack->sustain transition
        if state.regime == Regime.R1_ATTACK_IGNITION:
            if state.phase_block > state.program.attack_blocks:
                return state.program.sustain_regime

        return None

    def _apply_arc_modulation(self, state: VoiceStateWithProgram,
                              weights: List[int]) -> List[int]:
        """
        Apply arc modulator to family weights during transition.
        """
        if not state.arc.active:
            return weights

        arc = ARC_SPECS.get(state.arc.arc_id)
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
        """
        intent = state.intent
        regime = state.regime
        regime_spec = self.regime_graph.regime_specs.get(regime)

        # Start with base weights from regime spec
        if regime_spec:
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

    def _compute_events_with_regime(self, state: VoiceStateWithProgram) -> Tuple[int, List[int]]:
        """
        Compute events using RegimeSpec event policy.

        Returns (event_mask, event_strengths).
        """
        intent = state.intent
        regime = state.regime
        regime_spec = self.regime_graph.regime_specs.get(regime)

        mask = 0
        strengths = [0] * NUM_EVENTS

        enable_mask = regime_spec.event_enable_mask if regime_spec else 0xFF

        # E0: Impulse - high snap
        if (enable_mask & (1 << E_IMPULSE)) and intent.snap > Q15_HALF:
            mask |= (1 << E_IMPULSE)
            strengths[E_IMPULSE] = intent.snap

        # E1: Noise - high air
        if (enable_mask & (1 << E_NOISE)) and intent.air > Q15_HALF:
            mask |= (1 << E_NOISE)
            strengths[E_NOISE] = intent.air >> 1

        # E2: Pulse - attack regime
        if (enable_mask & (1 << E_PULSE)) and regime == Regime.R1_ATTACK_IGNITION:
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

        # E5: Swallowtail - glitch regime
        if (enable_mask & (1 << E_SWALLOWTAIL)) and regime == Regime.R4_GLITCH_CHATTER:
            mask |= (1 << E_SWALLOWTAIL)
            strengths[E_SWALLOWTAIL] = 0x3C  # Mid bands

        # E6: Fold - catastrophe
        if (enable_mask & (1 << E_FOLD)) and regime == Regime.R5_CATASTROPHE:
            mask |= (1 << E_FOLD)
            strengths[E_FOLD] = Q15_ONE

        # E7: Sheaf - recovery
        if (enable_mask & (1 << E_SHEAF)) and regime == Regime.R6_RECOVERY_HEAL:
            mask |= (1 << E_SHEAF)
            strengths[E_SHEAF] = Q15_HALF

        # Add arc events if active
        if state.arc.active:
            arc = ARC_SPECS.get(state.arc.arc_id)
            if arc:
                mask |= arc.arc_events
                for i in range(NUM_EVENTS):
                    if arc.arc_events & (1 << i):
                        strengths[i] = max(strengths[i], Q15_HALF)

        return mask, strengths

    def _step_envelope(self, state: VoiceStateWithProgram) -> int:
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
        target_regime = self._check_regime_transition(state)
        if target_regime is not None and target_regime != state.regime:
            self._transition_regime(state, target_regime, use_arc=True)
        state.regime_blocks += 1

        # 3. Step arc modulator
        state.arc.step()

        # 4. Compute family weights with regime policy + arc
        state.family_weights = self._compute_family_weights_with_regime(state)

        # 5. Compute events with regime policy
        event_mask, event_strengths = self._compute_events_with_regime(state)

        # 6. Tree morph (with regime spec override)
        regime_spec = self.regime_graph.regime_specs.get(state.regime)
        if regime_spec and regime_spec.tree_lock:
            # Regime locks tree
            target_tree = regime_spec.preferred_tree
        else:
            target_tree = select_tree_from_intent(state.intent, state.regime)

        # Apply arc morph rate if active
        if state.arc.active:
            arc = ARC_SPECS.get(state.arc.arc_id)
            if arc and arc.morph_rate_override > 0:
                state.tree_morph.morph_rate = arc.morph_rate_override
        elif regime_spec:
            state.tree_morph.morph_rate = regime_spec.morph_rate

        state.tree_morph = update_tree_morph(state.tree_morph, target_tree)

        # 7. Step envelope (Semi-Markov duration)
        state.gain_q15 = self._step_envelope(state)

        # 8. Update block counter
        state.block_count += 1

        # Detect phase transitions
        new_phase = state.phase
        is_starting = (old_phase == VoicePhase.FREE or old_phase == VoicePhase.ATTACK) and new_phase != VoicePhase.FREE
        is_stopping = new_phase == VoicePhase.FREE and old_phase != VoicePhase.FREE

        # Compute musical envelopes from intent
        tilt_env = (state.intent.tone + state.intent.bow) >> 1
        density_env = (state.intent.air + state.intent.glitch) >> 1

        # Get tail policy from regime spec
        normalize_curve = "pink"
        fineclamp_cap = 0.5
        slew_delta = 512
        if regime_spec:
            normalize_curve = regime_spec.normalize_curve
            fineclamp_cap = regime_spec.fineclamp_cap_ratio
            slew_delta = regime_spec.slew_max_delta

        # Emit VoiceFrameABI (Mealy output)
        return VoiceFrameABI(
            program_id=state.program_id,
            voice_id=voice_id,
            regime_id=int(state.regime),
            gain_q15=state.gain_q15,
            pan_q15=state.gesture.pan_q15,
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
            is_starting=is_starting and state.block_count <= 1,
            is_stopping=is_stopping,
            is_active=state.phase != VoicePhase.FREE,
            arc_active=state.arc.active,
            arc_id=state.arc.arc_id,
            arc_progress_q15=state.arc.get_progress_q15(),
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
    - ENERGY_INTENT (1) -> I_TONE, I_BOW boost
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
    energy_intent = macro_runtime.get_system_macro(SystemMacro.ENERGY_INTENT)
    focus = macro_runtime.get_system_macro(SystemMacro.FOCUS)
    risk = macro_runtime.get_system_macro(SystemMacro.RISK)
    age = macro_runtime.get_system_macro(SystemMacro.AGE)
    unity = macro_runtime.get_system_macro(SystemMacro.UNITY)

    # Create modified intent
    new_lanes = list(intent.lanes)

    # ENERGY_INTENT -> boost TONE and BOW
    intent_boost = energy_intent >> 2  # Up to 25% boost
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
