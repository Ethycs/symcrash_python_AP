"""
WSE-STD-030 Executor Scheduling Implementation
==============================================

Implements executor scheduling, tail ordering, and budget accounting per WSE-STD-030 v0.

This module provides:
- Budget accounting system with cost units per operator
- Deterministic drop policy (D1-D4)
- Mandatory tail stage with TAIL_SAFE predicate
- Event lane processing in strict order (E1-E3)
- op_enable_mask toggle persistence
- Reference executor conforming to the normative pipeline order

Conformance: WSE-STD-030 v0

Execution Pipeline (normative order):
1. Prepare stage - Load c_in, enable_mask
2. Backbone stage - Apply operators in author order (no reordering)
3. Event stage - Process lanes 0..7 in order
4. Tail stage - Mandatory, never dropped
5. Commit stage - Write back c_out

Per Section 3.2, Tail sequence:
1. Projection / clamp / legality (Q-family)
2. Emergency simplify (optional but deterministic trigger)
3. Normalization / scale discipline
4. State coercion / final invariants
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import IntEnum
import numpy as np

from ..fixed import sat16, mul_q15_sat, slew_q15, Q15_ONE, Q15_HALF
from ..operator_library import COEFF_LEN, BAND_SLICES, NUM_BANDS

# Import from uart_protocol for VoiceFrame ABI
from ..comm.uart_protocol import (
    VoiceFrameV0, BackboneEntry, Payload64, PayloadMode,
    EventLane, NUM_EVENT_LANES, resolve_pattern
)
from ..primitives import Q15_NEG_ONE


# =============================================================================
# CONSTANTS (per WSE-STD-030)
# =============================================================================

# Maximum coefficient magnitude (C_MAX per Section 3.3)
C_MAX = 32767  # int16 max

# Default energy maximum per traction level (E_MAX per Section 3.3)
E_MAX_DEFAULT = 0x7FFFFFFF  # Large default

# Cost unit constants (stable across builds per Section 4.1)
COST_UNIT_BASE = 100

# Tail cost (reserved outside budget per Section 3.1)
COST_TAIL_NORMALIZE = 50
COST_TAIL_CLAMP = 30
COST_TAIL_SIMPLIFY = 80
COST_TAIL_TOTAL = COST_TAIL_NORMALIZE + COST_TAIL_CLAMP + COST_TAIL_SIMPLIFY

# Event lane costs
COST_EVENT_INJECT = 40
COST_EVENT_TOGGLE = 10
COST_EVENT_PERMUTE = 60


# =============================================================================
# OPERATOR COST TABLE (per Section 4.1)
# =============================================================================

# Cost units per operator (must be stable across builds)
OPERATOR_COSTS: Dict[int, int] = {
    0: 10,      # Identity
    1: 80,      # Tilt (diagonal)
    2: 60,      # FineClamp (projection)
    3: 100,     # IntraMix (banded)
    4: 120,     # DiffuseUp (low-rank)
    5: 120,     # DiffuseDown (low-rank)
    6: 90,      # BandShear (banded)
    7: 150,     # LowRankMorph (low-rank)
    8: 50,      # GlobalPulse (inject)
    9: 140,     # SwapBands (permute)
    10: 80,     # RotateNodes (permute)
    11: 50,     # NormalizeScale (diagonal)
    12: 100,    # GluePair (low-rank)
    13: 40,     # Impulse (inject)
    14: 60,     # NoiseBand (inject)
    15: 70,     # PressureRamp (diagonal)
    16: 50,     # GateMask (diagonal)
    17: 100,    # QuasiRotate (permute)
    18: 120,    # MorphMix (mix)
    19: 80,     # EmergencySimplify (projection)
    20: 90,     # FoldPressure (banded)
    21: 50,     # CuspInjection (inject)
    22: 70,     # SwallowtailMask (diagonal)
    23: 110,    # SheafProject (projection)
}

# Event lane costs
EVENT_LANE_COSTS: Dict[int, int] = {
    EventLane.EVT_OP_FIRE: COST_EVENT_INJECT,
    EventLane.EVT_OP_TOGGLE: COST_EVENT_TOGGLE,
    EventLane.EVT_INJECT: COST_EVENT_INJECT,
    EventLane.EVT_MATERIAL: COST_EVENT_TOGGLE,
    EventLane.EVT_TREE: COST_EVENT_TOGGLE,
    EventLane.EVT_CLOUD: COST_EVENT_TOGGLE,
    EventLane.EVT_VOICE: COST_EVENT_TOGGLE,
    EventLane.EVT_DEBUG: 5,  # Nearly free
}

# Event lane drop priorities (higher = drop first per Section 4.4)
# Rule D4: cosmetic first, toggles last, never emergency
EVENT_LANE_DROP_PRIORITY: Dict[int, int] = {
    EventLane.EVT_OP_FIRE: 3,       # Permutation burst - drop first
    EventLane.EVT_INJECT: 2,        # Injection - drop second
    EventLane.EVT_MATERIAL: 1,      # Material - drop third
    EventLane.EVT_TREE: 1,          # Tree - drop third
    EventLane.EVT_CLOUD: 1,         # Cloud - drop third
    EventLane.EVT_OP_TOGGLE: 0,     # Toggle - drop last (affects future)
    EventLane.EVT_VOICE: 0,         # Voice lifecycle - don't drop
    EventLane.EVT_DEBUG: 4,         # Debug - can drop early
}


# =============================================================================
# BUDGET ACCOUNTING (per Section 4.2)
# =============================================================================

@dataclass
class Budget:
    """
    Budget accounting per WSE-STD-030 Section 4.2.

    Budget variables:
    - B_total: Total available cost units for non-tail work
    - B_tail: Reserved cost for tail (implicit, always subtracted first)
    - B_voice_max: Per-voice cap (optional)
    """
    B_total: int = 10000            # Total budget for backbone + events
    B_tail: int = COST_TAIL_TOTAL   # Reserved for tail (always runs)
    B_voice_max: int = 2000         # Per-voice cap (optional)

    # Runtime state
    remaining: int = 0
    consumed_backbone: int = 0
    consumed_events: int = 0
    ops_skipped: int = 0
    events_skipped: int = 0

    def reset_for_voice(self, voice_budget: Optional[int] = None):
        """Reset budget for a new voice."""
        if voice_budget is not None:
            self.remaining = min(voice_budget, self.B_voice_max)
        else:
            self.remaining = self.B_voice_max
        self.consumed_backbone = 0
        self.consumed_events = 0

    def can_afford(self, cost: int, reserve_for_events: int = 0) -> bool:
        """Check if we can afford an operation."""
        return self.remaining >= cost + reserve_for_events

    def consume(self, cost: int, is_event: bool = False):
        """Consume budget for an operation."""
        self.remaining -= cost
        if is_event:
            self.consumed_events += cost
        else:
            self.consumed_backbone += cost

    def reserve_for_tail(self) -> int:
        """Return reserved tail cost (always 0 since tail is outside budget)."""
        return 0  # Tail is reserved outside budget

    def reserve_for_events(self) -> int:
        """Estimate remaining event cost."""
        # Rough estimate: 8 lanes * average cost
        return NUM_EVENT_LANES * COST_EVENT_INJECT // 2


# =============================================================================
# SYMBOL STATE (per Section 1.3)
# =============================================================================

@dataclass
class SymbolState:
    """
    Per-voice symbol state per WSE-STD-030 Section 1.3.

    The executor maintains:
    - c: coefficient state vector (256 int16)
    - op_enable_mask: persistent per-op toggles
    - internal: per-op scratch, per-voice PRNG state handle
    """
    c: List[int] = field(default_factory=lambda: [0] * COEFF_LEN)
    op_enable_mask: int = 0xFFFFFFFF  # All ops enabled by default
    rng_state: int = 0x12345678       # PRNG state (per WSE-STD-040)

    # Previous state for slewing
    c_prev: List[int] = field(default_factory=lambda: [0] * COEFF_LEN)

    # Voice ID for state tracking
    voice_id: int = 0

    def reset_on_allocation(self, preset_mask: int = 0xFFFFFFFF):
        """
        Reset state on voice allocation per Section 5.2 completion rules.

        Toggle reset condition: voice allocation (per question 3 in spec).
        """
        self.c = [0] * COEFF_LEN
        self.c_prev = [0] * COEFF_LEN
        self.op_enable_mask = preset_mask


# =============================================================================
# TAIL SAFE PREDICATE (per Section 3.3)
# =============================================================================

@dataclass
class TailSafeResult:
    """Result of TAIL_SAFE predicate check."""
    is_safe: bool = True
    bounded_magnitude: bool = True      # All |c[k]| <= C_MAX
    bounded_energy: bool = True         # ENERGY(c) <= E_MAX
    no_nan_undefined: bool = True       # No overflow/wrap
    deterministic_coercion: bool = True # Legal state reached
    max_coeff: int = 0
    total_energy: int = 0


def compute_energy(c: List[int]) -> int:
    """Compute total energy of coefficient vector."""
    # Energy = sum of squared coefficients (in int64 to avoid overflow)
    return sum(int(x) * int(x) for x in c)


def check_tail_safe(c: List[int], traction_q15: int = Q15_ONE,
                    e_max: int = E_MAX_DEFAULT) -> TailSafeResult:
    """
    Check TAIL_SAFE predicate per WSE-STD-030 Section 3.3.

    Postconditions:
    1. Bounded magnitude: |c_out[k]| <= C_MAX for all k
    2. Bounded energy: ENERGY(c_out) <= E_MAX (optional)
    3. No NaN/undefined/wrap: all operations defined
    4. Deterministic coercion: illegal states coerced deterministically
    5. Idempotence-ish: reapplying tail doesn't add artifacts
    """
    result = TailSafeResult()

    # 1. Check bounded magnitude
    max_coeff = max(abs(x) for x in c)
    result.max_coeff = max_coeff
    result.bounded_magnitude = max_coeff <= C_MAX

    # 2. Check bounded energy
    energy = compute_energy(c)
    result.total_energy = energy

    # Scale E_MAX by traction (higher traction = stricter limit)
    scaled_e_max = e_max * traction_q15 // Q15_ONE
    result.bounded_energy = energy <= scaled_e_max

    # 3. No NaN/undefined - Python ints don't have NaN, check for overflow
    result.no_nan_undefined = all(Q15_NEG_ONE <= x <= Q15_ONE for x in c)

    # 4. Deterministic coercion - assume OK if we reach this point
    result.deterministic_coercion = True

    # Overall safety
    result.is_safe = (
        result.bounded_magnitude and
        result.bounded_energy and
        result.no_nan_undefined and
        result.deterministic_coercion
    )

    return result


# =============================================================================
# TAIL STAGE IMPLEMENTATION (per Section 3.2)
# =============================================================================

def apply_tail_normalize(c: List[int], target_curve: str = "pink") -> List[int]:
    """
    Tail op 3: Normalization / scale discipline.

    Normalizes coefficient energy distribution.
    """
    # Compute total energy
    energy = compute_energy(c)
    if energy == 0:
        return c

    # Target energy based on curve
    if target_curve == "flat":
        target_per_coeff = int(np.sqrt(energy / COEFF_LEN))
    elif target_curve == "pink":
        # Pink noise: more energy in low bands
        target_per_coeff = int(np.sqrt(energy / COEFF_LEN) * 0.8)
    else:  # bright
        target_per_coeff = int(np.sqrt(energy / COEFF_LEN) * 1.2)

    # Scale if needed
    max_coeff = max(abs(x) for x in c)
    if max_coeff > Q15_ONE * 0.9:  # Near saturation
        scale = int(Q15_ONE * 0.8)  # Reduce by 20%
        c = [mul_q15_sat(x, scale) for x in c]

    return c


def apply_tail_clamp(c: List[int], cap_ratio: float = 0.5) -> List[int]:
    """
    Tail op 1: Projection / clamp / legality (Q-family).

    Per-band energy capping.
    """
    # Compute per-band target energy
    target_energy_per_band = int((Q15_ONE ** 2) * cap_ratio * 8)

    for start, end in BAND_SLICES:
        band_energy = sum(int(c[i]) ** 2 for i in range(start, end))

        if band_energy > target_energy_per_band:
            # Scale down band
            scale = int(np.sqrt(target_energy_per_band / band_energy) * Q15_ONE)
            scale = min(Q15_ONE, scale)
            for i in range(start, end):
                c[i] = mul_q15_sat(c[i], scale)

    return c


def apply_tail_simplify(c: List[int], trigger: bool = False, k: int = 4) -> List[int]:
    """
    Tail op 2: Emergency simplify (optional but deterministic trigger).

    Zeros the k highest-frequency leaves when triggered.
    """
    if not trigger:
        return c

    # Zero the top k bands (finest detail)
    for band_idx in range(NUM_BANDS - 1, max(0, NUM_BANDS - k) - 1, -1):
        start, end = BAND_SLICES[band_idx]
        for i in range(start, end):
            c[i] = 0

    return c


def apply_tail_coerce(c: List[int]) -> List[int]:
    """
    Tail op 4: State coercion / final invariants.

    Ensures all coefficients are in valid int16 range (saturation).
    """
    return [sat16(x) for x in c]


def apply_mandatory_tail(c: List[int], traction_q15: int = Q15_ONE,
                         normalize_curve: str = "pink",
                         cap_ratio: float = 0.5) -> Tuple[List[int], TailSafeResult]:
    """
    Apply mandatory tail stage per WSE-STD-030 Section 3.2.

    Sequence:
    1. Projection / clamp / legality
    2. Emergency simplify (if needed)
    3. Normalization / scale discipline
    4. State coercion / final invariants

    Returns: (c_out, tail_safe_result)
    """
    # 1. Clamp (projection)
    c = apply_tail_clamp(c, cap_ratio)

    # Check if emergency simplify needed
    max_coeff = max(abs(x) for x in c)
    trigger_simplify = max_coeff > C_MAX * 0.95

    # 2. Emergency simplify (if triggered)
    c = apply_tail_simplify(c, trigger=trigger_simplify)

    # 3. Normalize
    c = apply_tail_normalize(c, normalize_curve)

    # 4. Final coercion
    c = apply_tail_coerce(c)

    # Verify TAIL_SAFE
    result = check_tail_safe(c, traction_q15)

    # If still not safe, force simplify
    if not result.is_safe:
        c = apply_tail_simplify(c, trigger=True, k=6)
        c = apply_tail_coerce(c)
        result = check_tail_safe(c, traction_q15)

    return c, result


# =============================================================================
# EVENT LANE PROCESSING (per Section 2.3)
# =============================================================================

def should_drop_lane(lane: int, budget: Budget) -> bool:
    """
    Determine if lane should be dropped per Rule D4.

    Drop order (per Section 4.4):
    1. Cosmetic/permutation bursts first
    2. Then injections
    3. Then toggles last
    4. Never drop emergency triggers
    """
    priority = EVENT_LANE_DROP_PRIORITY.get(lane, 2)

    # High priority = drop first
    if priority >= 3:
        return True

    # Medium priority = drop if budget tight
    if priority >= 2 and budget.remaining < COST_EVENT_INJECT * 2:
        return True

    # Low priority = only drop if very tight
    if priority >= 1 and budget.remaining < COST_EVENT_INJECT:
        return True

    # Priority 0 = never drop
    return False


@dataclass
class EventContext:
    """Context for event lane processing."""
    c: List[int]
    mask: int
    event_mask: int
    event_strength: List[int]
    payload: Payload64
    per_lane_payloads: List[Payload64]
    payload_mode: PayloadMode
    budget: Budget


def apply_event_lane(lane: int, ctx: EventContext) -> Tuple[List[int], int]:
    """
    Apply single event lane per WSE-STD-030 Section 2.3.

    Rule E1: Lanes processed in strictly increasing order.
    Rule E2: Events don't change already-run backbone stage.
    Rule E3: Toggle mutations persist in SymbolState.

    Returns: (updated_c, updated_mask)
    """
    c = ctx.c
    mask = ctx.mask
    strength = ctx.event_strength[lane]

    # Resolve payload (v0a/v0b per Section 2.3.3)
    if ctx.payload_mode == PayloadMode.COMMON:
        payload = ctx.payload
    elif ctx.payload_mode == PayloadMode.PER_LANE_FULL:
        payload = ctx.per_lane_payloads[lane] if lane < len(ctx.per_lane_payloads) else ctx.payload
    else:  # PER_LANE_SPARSE
        # Find payload for this lane
        payload = ctx.payload  # Default fallback
        for p in ctx.per_lane_payloads:
            if p.lane_hint == lane:
                payload = p
                break

    # Resolve target leaves and coefficients
    leaf_mask, coeff_mask = resolve_pattern(payload)

    # Apply event action based on lane type
    if lane == EventLane.EVT_OP_FIRE:
        # Fire one-shot operator using payload targeting
        c = _apply_injection(c, strength, leaf_mask, coeff_mask, payload.variant0)
        ctx.budget.consume(EVENT_LANE_COSTS[lane], is_event=True)

    elif lane == EventLane.EVT_OP_TOGGLE:
        # Toggle op_enable bits
        toggle_bits = payload.target_id  # Use target_id as bitmask
        mask ^= toggle_bits
        ctx.budget.consume(EVENT_LANE_COSTS[lane], is_event=True)

    elif lane == EventLane.EVT_INJECT:
        # Injection (impulse/noise/burst)
        c = _apply_injection(c, strength, leaf_mask, coeff_mask, payload.variant0)
        ctx.budget.consume(EVENT_LANE_COSTS[lane], is_event=True)

    elif lane == EventLane.EVT_MATERIAL:
        # Material engine override (forward downstream)
        # No coefficient change, just consume budget
        ctx.budget.consume(EVENT_LANE_COSTS[lane], is_event=True)

    elif lane == EventLane.EVT_TREE:
        # Tree morph/switch event
        # Handled by Voice Engine, just consume budget
        ctx.budget.consume(EVENT_LANE_COSTS[lane], is_event=True)

    elif lane == EventLane.EVT_CLOUD:
        # Cloud/foreground hint
        ctx.budget.consume(EVENT_LANE_COSTS[lane], is_event=True)

    elif lane == EventLane.EVT_VOICE:
        # Note lifecycle event
        ctx.budget.consume(EVENT_LANE_COSTS[lane], is_event=True)

    elif lane == EventLane.EVT_DEBUG:
        # Debug marker
        ctx.budget.consume(EVENT_LANE_COSTS[lane], is_event=True)

    return c, mask


def _apply_injection(c: List[int], strength: int, leaf_mask: int,
                     coeff_mask: int, variant: int) -> List[int]:
    """Apply injection to coefficients based on targeting."""
    # Simple injection: add scaled impulse to targeted coefficients
    coeffs_per_leaf = COEFF_LEN // 32

    for leaf in range(32):
        if not (leaf_mask & (1 << leaf)):
            continue

        start = leaf * coeffs_per_leaf
        for i in range(coeffs_per_leaf):
            if coeff_mask & (1 << (i % 8)):
                # Injection amount based on variant
                if variant == 0:  # Impulse
                    injection = strength
                elif variant == 1:  # Noise-like
                    injection = ((strength * (i * 7 + leaf * 13)) % Q15_ONE) - Q15_HALF
                else:  # Scaled
                    injection = mul_q15_sat(strength, variant * 128)

                c[start + i] = sat16(c[start + i] + injection)

    return c


# =============================================================================
# BACKBONE STAGE (per Section 2.2)
# =============================================================================

def apply_backbone_op(c: List[int], op_id: int, strength: int,
                      params: Dict, ops_engine) -> List[int]:
    """
    Apply single backbone operator.

    Uses the operator engine for actual application.
    """
    return ops_engine.apply_operator(c, op_id, strength, params)


# =============================================================================
# REFERENCE EXECUTOR (per Section 7)
# =============================================================================

@dataclass
class ExecutorStats:
    """Executor statistics."""
    backbone_ops_executed: int = 0
    backbone_ops_skipped: int = 0
    events_executed: int = 0
    events_skipped: int = 0
    tail_simplify_triggered: int = 0
    budget_drops: int = 0
    voices_processed: int = 0


class Executor030:
    """
    Reference executor per WSE-STD-030 Section 7.

    Implements normative execution order:
    1. Prepare stage
    2. Backbone stage (author order)
    3. Event stage (lane order 0..7)
    4. Tail stage (mandatory)
    5. Commit stage
    """

    def __init__(self, n_voices: int = 128, ops_engine=None):
        self.n_voices = n_voices
        self.ops_engine = ops_engine

        # Per-voice symbol state
        self.symbol_state: Dict[int, SymbolState] = {}

        # Budget configuration
        self.budget = Budget()

        # Statistics
        self.stats = ExecutorStats()

        # Tail configuration
        self.normalize_curve = "pink"
        self.cap_ratio = 0.5

    def get_symbol_state(self, voice_id: int) -> SymbolState:
        """Get or create symbol state for voice."""
        if voice_id not in self.symbol_state:
            state = SymbolState()
            state.voice_id = voice_id
            self.symbol_state[voice_id] = state
        return self.symbol_state[voice_id]

    def allocate_voice(self, voice_id: int, preset_mask: int = 0xFFFFFFFF):
        """
        Allocate voice and reset state.

        Per Section 5.2: op_enable_mask resets on voice allocation.
        """
        state = self.get_symbol_state(voice_id)
        state.reset_on_allocation(preset_mask)

    def execute_voice_block(self, voice_id: int, frame: VoiceFrameV0,
                           voice_budget: Optional[int] = None) -> List[int]:
        """
        Execute one block for one voice per WSE-STD-030 Section 7.

        This is the reference pseudocode implementation.
        """
        # Get symbol state
        sym_state = self.get_symbol_state(voice_id)

        # Reset budget for this voice
        self.budget.reset_for_voice(voice_budget)

        # =================================================================
        # PREPARE STAGE (Section 2.1)
        # =================================================================
        c_work = list(sym_state.c)
        enable_mask = sym_state.op_enable_mask

        # =================================================================
        # BACKBONE STAGE (Section 2.2)
        # =================================================================
        # Rule B1: Author order - iterate in given list order
        # Rule B2: No executor sorting
        # Rule B3: Skip disabled ops without advancing PRNG

        for entry in frame.backbone:
            op_id = entry.op_id
            strength = entry.strength_q15
            params = {"param0": entry.param0_q15, "param1": entry.param1_q15}

            # Check if op is disabled
            if not (enable_mask & (1 << (op_id % 32))):
                continue  # Rule B3: Skip without PRNG draw

            # Check budget (Rule D1: drop remainder, not reorder)
            cost = OPERATOR_COSTS.get(op_id, COST_UNIT_BASE)
            reserve = self.budget.reserve_for_events() + self.budget.reserve_for_tail()

            if not self.budget.can_afford(cost, reserve):
                # Skip remaining backbone ops
                self.stats.backbone_ops_skipped += 1
                self.stats.budget_drops += 1
                break  # Rule D1

            # Apply operator (Rule D3: atomic at block level)
            if self.ops_engine:
                c_work = apply_backbone_op(c_work, op_id, strength, params, self.ops_engine)
            else:
                # Fallback: simple scaling
                scale = strength
                c_work = [mul_q15_sat(x, scale) for x in c_work]

            self.budget.consume(cost)
            self.stats.backbone_ops_executed += 1

        # =================================================================
        # EVENT STAGE (Section 2.3)
        # =================================================================
        # Rule E1: Process lanes in strictly increasing order 0..L-1

        event_ctx = EventContext(
            c=c_work,
            mask=enable_mask,
            event_mask=frame.event_mask,
            event_strength=frame.event_strength_q15,
            payload=frame.payload_common,
            per_lane_payloads=frame.payload_per_lane,
            payload_mode=frame.payload_mode,
            budget=self.budget,
        )

        for lane in range(NUM_EVENT_LANES):
            # Check if lane is active
            if not (frame.event_mask & (1 << lane)):
                continue

            # Check budget for event
            cost = EVENT_LANE_COSTS.get(lane, COST_EVENT_INJECT)
            reserve = self.budget.reserve_for_tail()

            if not self.budget.can_afford(cost, reserve):
                # Apply drop policy (Rule D4)
                if should_drop_lane(lane, self.budget):
                    self.stats.events_skipped += 1
                    continue
                # Non-droppable lane: force no-op behavior
                pass

            # Apply event lane (Rule E2: doesn't change backbone)
            c_work, enable_mask = apply_event_lane(lane, event_ctx)
            event_ctx.c = c_work
            event_ctx.mask = enable_mask
            self.stats.events_executed += 1

        # =================================================================
        # TAIL STAGE (Section 3) - Mandatory, never dropped
        # =================================================================
        # Rule T0: Reserved overhead, outside budget
        # Rule D2: Tail is sacred - always runs

        c_out, tail_result = apply_mandatory_tail(
            c_work,
            traction_q15=frame.traction_q15,
            normalize_curve=self.normalize_curve,
            cap_ratio=self.cap_ratio
        )

        if not tail_result.is_safe:
            # This should not happen after mandatory tail
            # Log and force simplify
            self.stats.tail_simplify_triggered += 1
            c_out = apply_tail_simplify(c_out, trigger=True, k=6)
            c_out = apply_tail_coerce(c_out)

        # Assert TAIL_SAFE (Section 3.3)
        assert check_tail_safe(c_out, frame.traction_q15).is_safe, "TAIL_SAFE violated"

        # =================================================================
        # COMMIT STAGE (Section 6)
        # =================================================================
        # Rule C1: Commit atomicity - either full commit or nothing

        sym_state.c = c_out
        sym_state.c_prev = list(c_out)  # Update prev for next block slew
        sym_state.op_enable_mask = enable_mask  # Rule E3: Toggle persistence

        self.stats.voices_processed += 1

        return c_out

    def execute_block(self, frames: List[Tuple[int, VoiceFrameV0]],
                     total_budget: Optional[int] = None) -> Dict[int, List[int]]:
        """
        Execute one block for all voices.

        Returns: Dict of voice_id -> coefficient output
        """
        # Distribute budget across voices
        n_voices = len(frames)
        per_voice_budget = None
        if total_budget is not None and n_voices > 0:
            per_voice_budget = total_budget // n_voices

        results = {}
        for voice_id, frame in frames:
            c_out = self.execute_voice_block(voice_id, frame, per_voice_budget)
            results[voice_id] = c_out

        return results

    def reset_stats(self):
        """Reset statistics."""
        self.stats = ExecutorStats()

    def get_voice_mask(self, voice_id: int) -> int:
        """Get current op_enable_mask for voice."""
        state = self.get_symbol_state(voice_id)
        return state.op_enable_mask

    def set_voice_mask(self, voice_id: int, mask: int):
        """Set op_enable_mask for voice (for testing)."""
        state = self.get_symbol_state(voice_id)
        state.op_enable_mask = mask


# Tests moved to pipeline_tests/test_executor_std030.py


# =============================================================================
# BLOCK-LEVEL SCHEDULER (from scheduler.py)
# =============================================================================
#
# Implements Scheduler Spec v01 for OrangeCrab (ECP5).
# Goal: Each audio block, run the highest effective Tower level (L5->L0) the FPGA
# can afford without clicks, without divergence, and with tail always enforced.
#
# Key Components:
# - VoiceSelector: Ring cursor voice selection with eligibility
# - TileScheduler: Tile-based TOP/MID throttling with counters
# - DegradeController: Global degrade state D with hysteresis
# - HealthTelemetry: CPU/mem/stab bars and gear
# - BlockScheduler: Main entry point combining all components

from typing import Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..frames import VoiceFramePacketV0b
    from ..budgets import RegimeBudget

# Import VoiceFlags for eligibility check
from ..frames import VoiceFlags


# =============================================================================
# SCHEDULER CONSTANTS (from Scheduler Spec v01)
# =============================================================================

V_MAX = 128      # Maximum voices
TILE_SIZE = 16   # Voices per tile (power of 2)
NUM_TILES = V_MAX // TILE_SIZE  # 8 tiles

# Block size (should match primitives.BLOCK_SIZE)
SCHED_BLOCK_SIZE = 64

# Q15 scale for scheduler calculations
SCHED_Q15_SCALE = 32767
SCHED_Q15_ONE = 32767

# Eligibility threshold: activity > 0.05 in Q15
THETA_ACTIVE_Q15 = int(0.05 * SCHED_Q15_SCALE)  # ~1638

# Stability threshold for TOP suppression
THETA_STAB_RED_Q15 = int(0.3 * SCHED_Q15_SCALE)  # ~9830

# Hysteresis: blocks before D can decrease
H_DOWN = 4

# Degrade thresholds in Q15
# D=0 if p < 0.70, D=1 if 0.70 <= p < 0.85, etc.
DEGRADE_THRESHOLDS_Q15 = [
    int(0.70 * SCHED_Q15_SCALE),  # 22938 - boundary between D=0 and D=1
    int(0.85 * SCHED_Q15_SCALE),  # 27853 - boundary between D=1 and D=2
    int(1.00 * SCHED_Q15_SCALE),  # 32768 - boundary between D=2 and D=3
    int(1.15 * SCHED_Q15_SCALE),  # 37683 - boundary between D=3 and D=4
]

# Degrade -> period table (K_top, K_mid)
PERIOD_TABLE = {
    0: (1, 1),   # D=0: no throttling
    1: (2, 1),   # D=1: TOP every 2 blocks
    2: (4, 2),   # D=2: TOP every 4, MID every 2
    3: (8, 4),   # D=3: TOP every 8, MID every 4
    4: (16, 8),  # D=4: TOP every 16 (or suppressed), MID every 8
}

# Tail cost per voice (in cost units)
# FineClamp(2) + NormalizeScale(11) + SheafProject(23) = ~3N each = 9N
# Add margin for GluePair and EmergencySimplify
C_TAIL_VOICE = 12  # Cost units per voice for mandatory tail


# =============================================================================
# EMISSION GROUPS
# =============================================================================

class EmissionGroup(IntEnum):
    """Tower level emission groups."""
    TAIL = 0   # Phase C: Always runs for serviced voices
    BASE = 1   # L0-L1: Minimal injections for continuity
    MID = 2    # L2-L3: BANDED + Diffuse, LOWRANK, richer events
    TOP = 3    # L4-L5: Chaos overlays, catastrophe corridor, dense patterning


# =============================================================================
# HEALTH TELEMETRY
# =============================================================================

@dataclass
class HealthTelemetry:
    """
    Health telemetry output per block.

    These are observational outputs and MUST NOT affect decisions
    in the same block except via D computed at block start.
    """
    cpu_bar: int = 0      # Q15: used_compute / B_compute
    mem_bar: int = 0      # Q15: used_mem / B_mem
    stab_bar: int = SCHED_Q15_ONE  # Q15: derived from tail metrics (starts healthy)
    gear: int = 5         # 5 - D (higher = more headroom)

    # Tail stability counters (aggregated from serviced voices)
    clamp_hits: int = 0
    norm_gain_extreme: int = 0
    emergency_fired: int = 0

    # Service stats
    voices_serviced: int = 0
    voices_eligible: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'cpu_bar': self.cpu_bar,
            'mem_bar': self.mem_bar,
            'stab_bar': self.stab_bar,
            'gear': self.gear,
            'clamp_hits': self.clamp_hits,
            'norm_gain_extreme': self.norm_gain_extreme,
            'emergency_fired': self.emergency_fired,
            'voices_serviced': self.voices_serviced,
            'voices_eligible': self.voices_eligible,
        }


# =============================================================================
# TILE HELPERS
# =============================================================================

def tile_id(v: int) -> int:
    """Get tile ID for voice v."""
    return v >> 4  # v // TILE_SIZE (TILE_SIZE = 16 = 2^4)


def tile_off(v: int) -> int:
    """Get offset within tile for voice v."""
    return v & (TILE_SIZE - 1)  # v % TILE_SIZE


# =============================================================================
# VOICE SELECTOR (Ring Scheduler)
# =============================================================================

@dataclass
class VoiceSelector:
    """
    Ring cursor voice selection per Scheduler Spec v01 Section 6.

    Maintains cursor_fg over [0..V_MAX-1] and selects eligible foreground
    voices up to the budget-determined V_serviced limit.
    """
    cursor_fg: int = 0  # Ring cursor position
    v_max: int = V_MAX  # Total voice count

    def is_eligible(self, frame: 'VoiceFramePacketV0b') -> bool:
        """
        Check if a voice is eligible foreground.

        Eligible if:
        - reject_code == 0 AND
        - (flags.gate == 1 OR activity_q15 > theta_active)
        """
        if frame.reject_code != 0:
            return False

        gate_active = (frame.flags & VoiceFlags.GATE) != 0
        activity_above_threshold = frame.activity_q15 > THETA_ACTIVE_Q15

        return gate_active or activity_above_threshold

    def compute_v_serviced(self, b_compute: int, b_mem: int) -> int:
        """
        Compute maximum V_serviced based on tail reservation.

        V_serviced * C_TAIL_VOICE <= min(b_compute, b_mem)
        """
        budget = min(b_compute, b_mem)
        if C_TAIL_VOICE <= 0:
            return self.v_max
        return min(budget // C_TAIL_VOICE, self.v_max)

    def select_voices(
        self,
        frames: List[Optional['VoiceFramePacketV0b']],
        v_serviced_max: int,
    ) -> Tuple[List[int], int]:
        """
        Select eligible foreground voices using ring scheduler.

        Args:
            frames: List of VoiceFramePacketV0b (or None for absent voices)
            v_serviced_max: Maximum voices to service this block

        Returns:
            Tuple of (serviced_indices, eligible_count)
        """
        serviced: List[int] = []
        eligible_count = 0

        # Scan from cursor, wrapping around
        scanned = 0
        pos = self.cursor_fg

        while scanned < self.v_max and len(serviced) < v_serviced_max:
            frame = frames[pos] if pos < len(frames) else None

            if frame is not None and self.is_eligible(frame):
                eligible_count += 1
                serviced.append(pos)

            scanned += 1
            pos = (pos + 1) % self.v_max

        # Count remaining eligible voices we didn't get to
        while scanned < self.v_max:
            frame = frames[pos] if pos < len(frames) else None
            if frame is not None and self.is_eligible(frame):
                eligible_count += 1
            scanned += 1
            pos = (pos + 1) % self.v_max

        # Update cursor to next position after last scanned
        self.cursor_fg = pos

        return serviced, eligible_count


# =============================================================================
# TILE SCHEDULER (TOP/MID Throttling)
# =============================================================================

@dataclass
class TileScheduler:
    """
    Tile-based TOP/MID throttling per Scheduler Spec v01 Section 5.

    Maintains per-tile counters for MID and TOP emission groups.
    Counters only advance when at least one voice in the tile is serviced.
    """
    num_tiles: int = NUM_TILES
    ctr_mid: List[int] = field(default_factory=lambda: [0] * NUM_TILES)
    ctr_top: List[int] = field(default_factory=lambda: [0] * NUM_TILES)

    def get_periods(self, D: int) -> Tuple[int, int]:
        """Get (K_top, K_mid) periods for degrade state D."""
        return PERIOD_TABLE.get(D, (16, 8))

    def phase_mid(self, v: int, K_mid: int) -> int:
        """Compute MID phase for voice v."""
        return tile_off(v) & (K_mid - 1)

    def phase_top(self, v: int, K_top: int) -> int:
        """Compute TOP phase for voice v."""
        return tile_off(v) & (K_top - 1)

    def is_mid_due(self, v: int, D: int) -> bool:
        """
        Check if MID emission is due for voice v.

        MID_due(v) = ((ctr_mid[t] + phase_mid(v)) & (K_mid-1)) == 0
        """
        K_top, K_mid = self.get_periods(D)
        t = tile_id(v)
        phase = self.phase_mid(v, K_mid)
        return ((self.ctr_mid[t] + phase) & (K_mid - 1)) == 0

    def is_top_due(self, v: int, D: int, top_suppressed: bool = False) -> bool:
        """
        Check if TOP emission is due for voice v.

        TOP_due(v) = ((ctr_top[t] + phase_top(v)) & (K_top-1)) == 0

        Returns False if TOP is suppressed (D=4 and stab_bar < threshold).
        """
        if top_suppressed:
            return False

        K_top, K_mid = self.get_periods(D)
        t = tile_id(v)
        phase = self.phase_top(v, K_top)
        return ((self.ctr_top[t] + phase) & (K_top - 1)) == 0

    def compute_due_flags(
        self,
        serviced: List[int],
        D: int,
        top_suppressed: bool = False,
    ) -> List[Tuple[int, bool, bool]]:
        """
        Compute due flags for all serviced voices.

        Returns:
            List of (voice_id, mid_due, top_due) tuples
        """
        result = []
        for v in serviced:
            mid_due = self.is_mid_due(v, D)
            top_due = self.is_top_due(v, D, top_suppressed)
            result.append((v, mid_due, top_due))
        return result

    def advance_counters(self, serviced: List[int], D: int) -> None:
        """
        Advance tile counters for tiles with serviced voices.

        A tile counter advances only if at least one serviced voice
        in that tile exists this block.
        """
        K_top, K_mid = self.get_periods(D)

        # Find tiles with serviced voices
        serviced_tiles: Set[int] = set()
        for v in serviced:
            serviced_tiles.add(tile_id(v))

        # Advance counters for serviced tiles only
        for t in serviced_tiles:
            self.ctr_mid[t] = (self.ctr_mid[t] + 1) % K_mid
            self.ctr_top[t] = (self.ctr_top[t] + 1) % K_top


# =============================================================================
# DEGRADE CONTROLLER (Hysteresis)
# =============================================================================

@dataclass
class DegradeController:
    """
    Global degrade state D with hysteresis per Scheduler Spec v01 Section 4.

    D ranges from 0 (full budget) to 4 (maximum degradation).
    D increases immediately when pressure exceeds threshold.
    D decreases only after H_DOWN consecutive blocks below threshold.
    """
    D: int = 0  # Current degrade state [0..4]
    down_counter: int = 0  # Blocks spent below threshold

    def compute_target_D(self, p_q15: int) -> int:
        """
        Compute target D based on pressure p.

        D=0 if p < 0.70
        D=1 if 0.70 <= p < 0.85
        D=2 if 0.85 <= p < 1.00
        D=3 if 1.00 <= p < 1.15
        D=4 if p >= 1.15
        """
        if p_q15 < DEGRADE_THRESHOLDS_Q15[0]:
            return 0
        elif p_q15 < DEGRADE_THRESHOLDS_Q15[1]:
            return 1
        elif p_q15 < DEGRADE_THRESHOLDS_Q15[2]:
            return 2
        elif p_q15 < DEGRADE_THRESHOLDS_Q15[3]:
            return 3
        else:
            return 4

    def update(self, cpu_bar: int, mem_bar: int) -> int:
        """
        Update degrade state based on previous block telemetry.

        Args:
            cpu_bar: CPU usage bar (Q15)
            mem_bar: Memory usage bar (Q15)

        Returns:
            New degrade state D
        """
        # Combined pressure
        p = max(cpu_bar, mem_bar)
        target_D = self.compute_target_D(p)

        if target_D > self.D:
            # Increase immediately (but at most +1 per block)
            self.D = min(self.D + 1, 4)
            self.down_counter = 0
        elif target_D < self.D:
            # Decrease only after H_DOWN blocks below threshold
            self.down_counter += 1
            if self.down_counter >= H_DOWN:
                self.D = max(self.D - 1, 0)
                self.down_counter = 0
        else:
            # At target, reset down counter
            self.down_counter = 0

        return self.D

    def get_gear(self) -> int:
        """Get gear (5 - D). Higher gear = more headroom."""
        return 5 - self.D


# =============================================================================
# SCHEDULE RESULT
# =============================================================================

@dataclass
class VoiceScheduleInfo:
    """Schedule information for a single voice."""
    voice_id: int
    mid_due: bool
    top_due: bool

    def to_dict(self) -> dict:
        return {
            'voice_id': self.voice_id,
            'mid_due': self.mid_due,
            'top_due': self.top_due,
        }


@dataclass
class ScheduleResult:
    """Result of block scheduling."""
    serviced: List[VoiceScheduleInfo]
    degrade_state: int
    top_suppressed: bool
    telemetry: HealthTelemetry

    @property
    def serviced_ids(self) -> List[int]:
        """Get list of serviced voice IDs."""
        return [s.voice_id for s in self.serviced]

    def to_dict(self) -> dict:
        return {
            'serviced': [s.to_dict() for s in self.serviced],
            'degrade_state': self.degrade_state,
            'top_suppressed': self.top_suppressed,
            'telemetry': self.telemetry.to_dict(),
        }


# =============================================================================
# BLOCK SCHEDULER (Main Entry Point)
# =============================================================================

@dataclass
class BlockScheduler:
    """
    Main block scheduler combining all components.

    Implements Scheduler Spec v01:
    1. Update D from previous block telemetry (with hysteresis)
    2. Select voices using ring scheduler
    3. Compute due flags for TOP/MID
    4. Advance tile counters
    5. Return schedule result with telemetry
    """
    voice_selector: VoiceSelector = field(default_factory=VoiceSelector)
    tile_scheduler: TileScheduler = field(default_factory=TileScheduler)
    degrade_controller: DegradeController = field(default_factory=DegradeController)

    # Budget configuration
    b_compute: int = 1000  # Compute budget per block
    b_mem: int = 1000      # Memory budget per block

    # Previous block telemetry (for D update)
    prev_cpu_bar: int = 0
    prev_mem_bar: int = 0
    prev_stab_bar: int = SCHED_Q15_ONE

    # Accumulated costs this block
    used_compute: int = 0
    used_mem: int = 0

    def reset_block_costs(self) -> None:
        """Reset accumulated costs for new block."""
        self.used_compute = 0
        self.used_mem = 0

    def add_costs(self, compute: int, mem: int) -> None:
        """Add costs from executed operations."""
        self.used_compute += compute
        self.used_mem += mem

    def set_regime_budget(self, budget: 'RegimeBudget') -> None:
        """
        Update budgets from regime.

        Maps RegimeBudget's max_ticks_per_block to scheduler's b_compute/b_mem.

        Args:
            budget: RegimeBudget with max_ticks_per_block, etc.
        """
        # Map max_ticks_per_block to compute budget
        self.b_compute = budget.max_ticks_per_block

        # Use same value for memory budget (could be refined later)
        self.b_mem = budget.max_ticks_per_block

    def compute_health_bars(self) -> Tuple[int, int]:
        """
        Compute CPU and memory bars from used costs.

        Returns (cpu_bar, mem_bar) in Q15.
        """
        if self.b_compute > 0:
            cpu_bar = min((self.used_compute * SCHED_Q15_SCALE) // self.b_compute, 0xFFFF)
        else:
            cpu_bar = SCHED_Q15_ONE

        if self.b_mem > 0:
            mem_bar = min((self.used_mem * SCHED_Q15_SCALE) // self.b_mem, 0xFFFF)
        else:
            mem_bar = SCHED_Q15_ONE

        return cpu_bar, mem_bar

    def is_top_suppressed(self, D: int, stab_bar: int) -> bool:
        """
        Check if TOP should be suppressed.

        TOP suppressed if D=4 AND stab_bar < theta_stab_red
        """
        return D == 4 and stab_bar < THETA_STAB_RED_Q15

    def compute_stab_bar(
        self,
        clamp_hits: int,
        norm_gain_extreme: int,
        emergency_fired: int,
        voices_serviced: int,
    ) -> int:
        """
        Compute stability bar from tail metrics.

        stab_bar decreases with clamp_hits and emergency events.
        """
        if voices_serviced == 0:
            return SCHED_Q15_ONE  # No voices = stable

        # Start at full stability
        stab = SCHED_Q15_ONE

        # Decrease by 10% for each emergency
        stab -= emergency_fired * (SCHED_Q15_SCALE // 10)

        # Decrease by 5% for each extreme gain
        stab -= norm_gain_extreme * (SCHED_Q15_SCALE // 20)

        # Decrease by 1% for each clamp hit (normalized to voice count)
        clamp_penalty = (clamp_hits * SCHED_Q15_SCALE) // (voices_serviced * 100)
        stab -= clamp_penalty

        return max(0, stab)

    def schedule_block(
        self,
        frames: List[Optional['VoiceFramePacketV0b']],
        tail_stats: Optional[dict] = None,
    ) -> ScheduleResult:
        """
        Schedule a block of voices.

        Args:
            frames: List of VoiceFramePacketV0b (or None for absent voices)
            tail_stats: Optional dict with clamp_hits, norm_gain_extreme, emergency_fired

        Returns:
            ScheduleResult with serviced voices and telemetry
        """
        # Reset costs for new block
        self.reset_block_costs()

        # 1. Update D from previous block telemetry
        D = self.degrade_controller.update(self.prev_cpu_bar, self.prev_mem_bar)

        # 2. Check TOP suppression
        top_suppressed = self.is_top_suppressed(D, self.prev_stab_bar)

        # 3. Compute V_serviced based on budget
        v_serviced_max = self.voice_selector.compute_v_serviced(
            self.b_compute, self.b_mem
        )

        # 4. Select voices using ring scheduler
        serviced_ids, eligible_count = self.voice_selector.select_voices(
            frames, v_serviced_max
        )

        # 5. Compute due flags for serviced voices
        due_flags = self.tile_scheduler.compute_due_flags(
            serviced_ids, D, top_suppressed
        )

        # 6. Build schedule info
        serviced_info = [
            VoiceScheduleInfo(v, mid_due, top_due)
            for v, mid_due, top_due in due_flags
        ]

        # 7. Advance tile counters (only for tiles with serviced voices)
        self.tile_scheduler.advance_counters(serviced_ids, D)

        # 8. Process tail stats
        clamp_hits = 0
        norm_gain_extreme = 0
        emergency_fired = 0
        if tail_stats:
            clamp_hits = tail_stats.get('clamp_hits', 0)
            norm_gain_extreme = tail_stats.get('norm_gain_extreme', 0)
            emergency_fired = tail_stats.get('emergency_fired', 0)

        # 9. Build telemetry
        telemetry = HealthTelemetry(
            cpu_bar=self.prev_cpu_bar,
            mem_bar=self.prev_mem_bar,
            stab_bar=self.prev_stab_bar,
            gear=self.degrade_controller.get_gear(),
            clamp_hits=clamp_hits,
            norm_gain_extreme=norm_gain_extreme,
            emergency_fired=emergency_fired,
            voices_serviced=len(serviced_ids),
            voices_eligible=eligible_count,
        )

        return ScheduleResult(
            serviced=serviced_info,
            degrade_state=D,
            top_suppressed=top_suppressed,
            telemetry=telemetry,
        )

    def end_block(self, tail_stats: Optional[dict] = None) -> None:
        """
        Called at end of block to update telemetry for next block.

        Args:
            tail_stats: Optional dict with clamp_hits, norm_gain_extreme, emergency_fired
        """
        # Compute health bars from this block's costs
        self.prev_cpu_bar, self.prev_mem_bar = self.compute_health_bars()

        # Compute stability bar
        clamp_hits = 0
        norm_gain_extreme = 0
        emergency_fired = 0
        voices = 1  # Prevent div by zero

        if tail_stats:
            clamp_hits = tail_stats.get('clamp_hits', 0)
            norm_gain_extreme = tail_stats.get('norm_gain_extreme', 0)
            emergency_fired = tail_stats.get('emergency_fired', 0)
            voices = max(tail_stats.get('voices_serviced', 1), 1)

        self.prev_stab_bar = self.compute_stab_bar(
            clamp_hits, norm_gain_extreme, emergency_fired, voices
        )
