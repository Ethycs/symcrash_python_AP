"""Authored, block-rate CMMM event rules (Instrument Master Spec section 8.3)."""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from ..primitives import Q15_HALF, Q15_ONE, mul_q15_sat
from ..vm.state import xorshift32


class LaneRuleType(IntEnum):
    DISABLED = 0
    GATE_EDGE = 1
    THRESHOLD = 2
    STOCHASTIC = 3
    PATTERN = 4


@dataclass
class EventPatternSpec:
    """Arc-relative amplitudes; zero is a rest, one step starts per step_blocks.

    Sampling is stateless. The VoiceEngine supplies the arc's zero-based block
    position, so two voices never share a cursor and snapshots need no extra
    pattern clock. Looping is bounded by the enclosing arc's lifetime.
    """
    steps_q15: List[int] = field(default_factory=lambda: [Q15_ONE])
    step_blocks: int = 1
    loop: bool = True

    def __post_init__(self):
        if not self.steps_q15 or any(not isinstance(v, int) or not 0 <= v <= Q15_ONE for v in self.steps_q15):
            raise ValueError('Pattern steps must be nonempty Q15 amplitudes')
        if not isinstance(self.step_blocks, int) or self.step_blocks < 1:
            raise ValueError('Pattern step_blocks must be positive')

    def sample(self, arc_block: int) -> Optional[int]:
        if arc_block < 0 or arc_block % self.step_blocks:
            return None
        index = arc_block // self.step_blocks
        if not self.loop and index >= len(self.steps_q15):
            return None
        value = self.steps_q15[index % len(self.steps_q15)]
        return value if value else None


@dataclass
class LaneRuleSpec:
    """One lane trigger plus its amplitude and optional semantic selectors."""
    rule_type: LaneRuleType = LaneRuleType.DISABLED
    intent_lane: int = 0
    threshold_q15: int = Q15_HALF
    rate_q15: int = 1000
    on_rise: bool = True
    on_fall: bool = False
    pattern_id: int = 0
    strength_mod_q15: int = Q15_ONE
    target_band: Optional[int] = None
    band_mask: Optional[int] = None

    def __post_init__(self):
        self.rule_type = LaneRuleType(self.rule_type)
        if not isinstance(self.intent_lane, int) or not 0 <= self.intent_lane < 8:
            raise ValueError('Intent lane must be in 0..7')
        for value in (self.threshold_q15, self.rate_q15, self.strength_mod_q15):
            if not isinstance(value, int) or not 0 <= value <= Q15_ONE:
                raise ValueError('Event rule values must be Q15 amplitudes')
        if not isinstance(self.pattern_id, int) or not 0 <= self.pattern_id <= 65535:
            raise ValueError('Pattern ID must be uint16')
        if self.target_band is not None and (not isinstance(self.target_band, int) or not 0 <= self.target_band < 6):
            raise ValueError('Target band must be in 0..5')
        if self.band_mask is not None and (not isinstance(self.band_mask, int) or not 0 <= self.band_mask <= 255):
            raise ValueError('Band mask must be uint8')

    def check(self, intent, gesture, rng_state: int, *,
              gate_rise: Optional[bool] = None, gate_fall: Optional[bool] = None,
              pattern: Optional[EventPatternSpec] = None,
              arc_block: Optional[int] = None, rate_scale_q15: int = Q15_ONE) -> Tuple[bool, int, int]:
        """Return (fire, strength, RNG). No clock or edge state lives in a rule.

        A stochastic lane draws once for a nontrivial probability, then once
        for amplitude on a hit. Zero probability draws nothing; certainty
        needs only the amplitude draw. Disabled/masked rules do not draw.
        """
        if self.rule_type == LaneRuleType.DISABLED:
            return False, 0, rng_state
        if self.rule_type == LaneRuleType.GATE_EDGE:
            rise = getattr(gesture, 'gate_rise', False) if gate_rise is None else gate_rise
            fall = getattr(gesture, 'gate_fall', False) if gate_fall is None else gate_fall
            fire = (self.on_rise and rise) or (self.on_fall and fall)
            return fire, self.strength_mod_q15 if fire else 0, rng_state
        if self.rule_type == LaneRuleType.THRESHOLD:
            value = intent.lanes[self.intent_lane]
            if value >= self.threshold_q15:
                return True, mul_q15_sat(value, self.strength_mod_q15), rng_state
            return False, 0, rng_state
        if self.rule_type == LaneRuleType.PATTERN:
            value = pattern.sample(arc_block) if pattern is not None and arc_block is not None else None
            if value is not None:
                return True, mul_q15_sat(value, self.strength_mod_q15), rng_state
            return False, 0, rng_state
        if self.rule_type == LaneRuleType.STOCHASTIC:
            # Endpoint-preserving multiplication for probabilities.
            probability = (self.rate_q15 * rate_scale_q15 + Q15_ONE // 2) // Q15_ONE
            if probability <= 0:
                return False, 0, rng_state
            if probability < Q15_ONE:
                rng_state, value = xorshift32(rng_state)
                if (value & 0x7FFF) >= probability:
                    return False, 0, rng_state
            rng_state, value = xorshift32(rng_state)
            return True, mul_q15_sat(value & 0x7FFF, self.strength_mod_q15), rng_state
        raise ValueError(f'Unknown event rule type: {self.rule_type}')


@dataclass
class EventPolicySpec:
    """Eight authored lanes. An explicit empty policy disables all events."""
    enable_mask: int = 0xFF
    lane_rules: List[LaneRuleSpec] = field(default_factory=lambda: [LaneRuleSpec() for _ in range(8)])

    def __post_init__(self):
        if not isinstance(self.enable_mask, int) or not 0 <= self.enable_mask <= 255:
            raise ValueError('Event enable_mask must be uint8')
        if len(self.lane_rules) != 8 or any(not isinstance(rule, LaneRuleSpec) for rule in self.lane_rules):
            raise ValueError('Event policy requires eight LaneRuleSpec entries')

    def compute(self, intent, gesture, rng_state: int, *, allowed_mask: int = 0xFF,
                gate_rise: Optional[bool] = None, gate_fall: Optional[bool] = None,
                patterns: Optional[Dict[int, EventPatternSpec]] = None,
                arc_block: Optional[int] = None,
                probabilities: Optional[List[int]] = None) -> Tuple[int, List[int], int, int]:
        if probabilities is not None and (len(probabilities) != 8 or any(
                not isinstance(value, int) or not 0 <= value <= Q15_ONE for value in probabilities)):
            raise ValueError('Regime event probabilities require eight Q15 values')
        mask, values = 0, [0] * 8
        for lane, rule in enumerate(self.lane_rules):
            if not (self.enable_mask & allowed_mask & (1 << lane)):
                continue
            fire, strength, rng_state = rule.check(
                intent, gesture, rng_state, gate_rise=gate_rise, gate_fall=gate_fall,
                pattern=(patterns or {}).get(rule.pattern_id), arc_block=arc_block,
                rate_scale_q15=probabilities[lane] if probabilities is not None else Q15_ONE,
            )
            if fire:
                mask |= 1 << lane
                if lane in (3, 5):
                    values[lane] = rule.band_mask if rule.band_mask is not None else (0xFF if lane == 3 else 0x3C)
                else:
                    values[lane] = strength
        # Legacy scalar payload stays zero; typed selectors are carried by ABI.
        return mask, values, 0, rng_state
