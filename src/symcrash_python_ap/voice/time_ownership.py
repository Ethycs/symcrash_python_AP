"""
WSE-STD-060 — Time Ownership & Temporal Responsibility
=======================================================

This module implements time ownership rules and temporal responsibility
validation for all engines in the WSE pipeline.

Key Guarantees:
1. Voice Engine is the SOLE owner of musical time
2. Symbol Engine is time-blind (no counters, clocks, phase detection)
3. Wavelet Engine is purely mechanical (no timing inference)
4. Safety Engine is block-shaped only (no musical phase detection)
5. All timing is quantized to BlockTime (64 samples)

Architecture:
- GestureTime: Asynchronous, continuous (UI/MIDI/Sensors)
- VoiceTime: Discrete, block-quantized (Voice Engine)
- BlockTime: Discrete, block-exact (Symbol/Wavelet/Safety)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Any, Callable
from enum import Enum, IntEnum
import re
from ..primitives import Q15_ONE
from ..primitives import Q15_QUARTER


# =============================================================================
# CONSTANTS
# =============================================================================

# Block size in audio samples (normative per Section 2.1)
BLOCK_SIZE_SAMPLES = 64

# Standard sample rates
SAMPLE_RATE_48K = 48000
SAMPLE_RATE_44K = 44100


# =============================================================================
# TIME DOMAINS (Section 2.2)
# =============================================================================

class TimeDomain(Enum):
    """
    Time domain classification per WSE-STD-060 Section 2.2.

    | Domain      | Resolution                | Owner                     |
    | ----------- | ------------------------- | ------------------------- |
    | GestureTime | asynchronous, continuous  | UI / MIDI / Sensors       |
    | VoiceTime   | discrete, block-quantized | Voice Engine              |
    | BlockTime   | discrete, block-exact     | Symbol / Wavelet / Safety |
    """
    GESTURE_TIME = "gesture_time"  # Asynchronous, continuous
    VOICE_TIME = "voice_time"      # Discrete, block-quantized
    BLOCK_TIME = "block_time"      # Discrete, block-exact


class EngineType(Enum):
    """Engine classification for time ownership."""
    VOICE_ENGINE = "voice_engine"      # Time-aware (sole owner)
    SYMBOL_ENGINE = "symbol_engine"    # Time-blind
    WAVELET_ENGINE = "wavelet_engine"  # Purely mechanical
    SAFETY_ENGINE = "safety_engine"    # Block-shaped only


# =============================================================================
# ENGINE RESPONSIBILITIES (Section 4)
# =============================================================================

@dataclass
class EngineCapabilities:
    """
    Defines what an engine is allowed to do per WSE-STD-060 Section 4.

    Each engine type has strict rules about time-related operations.
    """
    engine_type: EngineType

    # What the engine MAY do
    may_maintain_counters: bool = False
    may_maintain_clocks: bool = False
    may_track_phase: bool = False
    may_trigger_time_events: bool = False
    may_use_block_modulo: bool = False
    may_detect_regime_progress: bool = False
    may_contain_adsr: bool = False
    may_schedule_transitions: bool = False

    # What the engine MUST do
    must_be_stateless_temporal: bool = True
    must_treat_frame_as_command: bool = True

    @classmethod
    def voice_engine(cls) -> 'EngineCapabilities':
        """Voice Engine: Time-aware, sole owner of musical time."""
        return cls(
            engine_type=EngineType.VOICE_ENGINE,
            may_maintain_counters=True,
            may_maintain_clocks=True,
            may_track_phase=True,
            may_trigger_time_events=True,
            may_use_block_modulo=True,
            may_detect_regime_progress=True,
            may_contain_adsr=True,
            may_schedule_transitions=True,
            must_be_stateless_temporal=False,
            must_treat_frame_as_command=False,
        )

    @classmethod
    def symbol_engine(cls) -> 'EngineCapabilities':
        """Symbol Engine: Time-blind, must not track time."""
        return cls(
            engine_type=EngineType.SYMBOL_ENGINE,
            may_maintain_counters=False,
            may_maintain_clocks=False,
            may_track_phase=False,
            may_trigger_time_events=False,
            may_use_block_modulo=False,
            may_detect_regime_progress=False,
            may_contain_adsr=False,
            may_schedule_transitions=False,
            must_be_stateless_temporal=True,
            must_treat_frame_as_command=True,
        )

    @classmethod
    def wavelet_engine(cls) -> 'EngineCapabilities':
        """Wavelet Engine: Purely mechanical, no timing inference."""
        return cls(
            engine_type=EngineType.WAVELET_ENGINE,
            may_maintain_counters=False,
            may_maintain_clocks=False,
            may_track_phase=False,
            may_trigger_time_events=False,
            may_use_block_modulo=False,
            may_detect_regime_progress=False,
            may_contain_adsr=False,
            may_schedule_transitions=False,
            must_be_stateless_temporal=True,
            must_treat_frame_as_command=True,
        )

    @classmethod
    def safety_engine(cls) -> 'EngineCapabilities':
        """Safety Engine: Block-shaped only, no musical phase detection."""
        return cls(
            engine_type=EngineType.SAFETY_ENGINE,
            may_maintain_counters=False,
            may_maintain_clocks=False,
            may_track_phase=False,
            may_trigger_time_events=False,
            may_use_block_modulo=False,
            may_detect_regime_progress=False,
            may_contain_adsr=False,
            may_schedule_transitions=False,
            must_be_stateless_temporal=True,
            must_treat_frame_as_command=True,
        )


# Engine capability registry
ENGINE_CAPABILITIES: Dict[EngineType, EngineCapabilities] = {
    EngineType.VOICE_ENGINE: EngineCapabilities.voice_engine(),
    EngineType.SYMBOL_ENGINE: EngineCapabilities.symbol_engine(),
    EngineType.WAVELET_ENGINE: EngineCapabilities.wavelet_engine(),
    EngineType.SAFETY_ENGINE: EngineCapabilities.safety_engine(),
}


def get_engine_capabilities(engine_type: EngineType) -> EngineCapabilities:
    """Get capabilities for an engine type."""
    return ENGINE_CAPABILITIES[engine_type]


# =============================================================================
# BLOCKTIME (Section 2.1)
# =============================================================================

@dataclass
class BlockTime:
    """
    Block time representation per WSE-STD-060 Section 2.1.

    One block corresponds to exactly 64 audio samples.
    All downstream engines operate strictly at BlockTime resolution.
    """
    block_index: int
    sample_rate: int = SAMPLE_RATE_48K

    @property
    def sample_offset(self) -> int:
        """Get absolute sample offset for this block."""
        return self.block_index * BLOCK_SIZE_SAMPLES

    @property
    def time_seconds(self) -> float:
        """Get time in seconds (informational only)."""
        return self.sample_offset / self.sample_rate

    @classmethod
    def from_sample(cls, sample: int, sample_rate: int = SAMPLE_RATE_48K) -> 'BlockTime':
        """Convert sample offset to BlockTime (quantized)."""
        return cls(
            block_index=sample // BLOCK_SIZE_SAMPLES,
            sample_rate=sample_rate,
        )

    def advance(self, blocks: int = 1) -> 'BlockTime':
        """Advance by N blocks."""
        return BlockTime(
            block_index=self.block_index + blocks,
            sample_rate=self.sample_rate,
        )


# =============================================================================
# PROHIBITED PATTERNS (Section 6)
# =============================================================================

class ProhibitedPattern(Enum):
    """
    Prohibited patterns per WSE-STD-060 Section 6.

    These patterns are EXPLICITLY FORBIDDEN outside Voice Engine.
    Violation is a spec breach, not a style issue.
    """
    BLOCK_MODULO = "block_modulo"          # if block % N == 0
    EVENT_SCHEDULING = "event_scheduling"   # Scheduling events based on time
    ADSR_LOGIC = "adsr_logic"              # ADSR envelopes
    PHASE_DETECTION = "phase_detection"     # Detecting phases
    TIME_BRANCHING = "time_branching"       # Time-based branching


@dataclass
class PatternViolation:
    """A detected prohibited pattern violation."""
    pattern: ProhibitedPattern
    engine_type: EngineType
    location: str  # File:line or description
    description: str

    def __str__(self) -> str:
        return f"[{self.pattern.value}] in {self.engine_type.value}: {self.description} @ {self.location}"


# Regex patterns for detecting prohibited code patterns
PROHIBITED_PATTERNS_REGEX = {
    ProhibitedPattern.BLOCK_MODULO: [
        r'block\s*%\s*\d+',           # block % N
        r'block_index\s*%\s*\d+',     # block_index % N
        r'local_block\s*%\s*\d+',     # local_block % N
    ],
    ProhibitedPattern.PHASE_DETECTION: [
        r'phase\s*==\s*',             # phase == X
        r'adsr_phase',                # adsr_phase reference
        r'regime_progress',           # regime_progress reference
    ],
    ProhibitedPattern.ADSR_LOGIC: [
        r'adsr\s*\(',                 # adsr( function call
        r'envelope\s*\[',             # envelope[ array access
        r'attack_time',               # attack_time reference
        r'release_time',              # release_time reference
    ],
    ProhibitedPattern.TIME_BRANCHING: [
        r'if\s+.*time.*:',            # if ... time ...:
        r'if\s+.*elapsed.*:',         # if ... elapsed ...:
        r'if\s+.*duration.*:',        # if ... duration ...:
    ],
}


def detect_prohibited_patterns(
    code: str,
    engine_type: EngineType,
) -> List[PatternViolation]:
    """
    Detect prohibited patterns in code.

    Per Section 6, these patterns are forbidden outside Voice Engine.

    Args:
        code: Source code to analyze
        engine_type: Type of engine the code belongs to

    Returns:
        List of detected violations
    """
    # Voice Engine is allowed all patterns
    if engine_type == EngineType.VOICE_ENGINE:
        return []

    violations = []

    for pattern_type, regexes in PROHIBITED_PATTERNS_REGEX.items():
        for regex in regexes:
            matches = re.finditer(regex, code, re.IGNORECASE)
            for match in matches:
                # Find line number
                line_num = code[:match.start()].count('\n') + 1
                violations.append(PatternViolation(
                    pattern=pattern_type,
                    engine_type=engine_type,
                    location=f"line {line_num}",
                    description=f"Found '{match.group()}'"
                ))

    return violations


# =============================================================================
# TIME OWNERSHIP VALIDATION (Rule 060.1)
# =============================================================================

@dataclass
class TimeOwnershipValidator:
    """
    Validates time ownership rules per WSE-STD-060.

    Rule 060.1: The Voice Engine is the sole owner of musical time.
    """

    def validate_engine(
        self,
        engine_type: EngineType,
        has_counters: bool = False,
        has_clocks: bool = False,
        has_phase_tracking: bool = False,
        has_adsr: bool = False,
        has_event_scheduling: bool = False,
    ) -> List[str]:
        """
        Validate an engine implementation against time ownership rules.

        Returns list of violations (empty if compliant).
        """
        violations = []
        caps = get_engine_capabilities(engine_type)

        if has_counters and not caps.may_maintain_counters:
            violations.append(
                f"{engine_type.value} MUST NOT maintain counters (Rule 060.1)"
            )

        if has_clocks and not caps.may_maintain_clocks:
            violations.append(
                f"{engine_type.value} MUST NOT maintain clocks (Rule 060.1)"
            )

        if has_phase_tracking and not caps.may_track_phase:
            violations.append(
                f"{engine_type.value} MUST NOT track phase (Section 4)"
            )

        if has_adsr and not caps.may_contain_adsr:
            violations.append(
                f"{engine_type.value} MUST NOT contain ADSR logic (Section 4)"
            )

        if has_event_scheduling and not caps.may_trigger_time_events:
            violations.append(
                f"{engine_type.value} MUST NOT schedule time-based events (Section 6)"
            )

        return violations

    def validate_voice_frame_completeness(
        self,
        has_timing_info: bool,
        has_event_payload: bool,
        has_regime_info: bool,
    ) -> List[str]:
        """
        Validate that VoiceFrame carries all temporal information.

        Per Rule 060.2: VoiceFrame is the only legal carrier of time-resolved intent.
        """
        violations = []

        if not has_timing_info:
            violations.append(
                "VoiceFrame MUST carry timing information (Rule 060.2)"
            )

        if not has_event_payload:
            violations.append(
                "VoiceFrame MUST carry event payload (Rule 060.2)"
            )

        if not has_regime_info:
            violations.append(
                "VoiceFrame MUST carry regime information (Rule 060.2)"
            )

        return violations


# =============================================================================
# INTENT FRAME → VOICE FRAME BOUNDARY (Section 5)
# =============================================================================

@dataclass
class IntentFrame:
    """
    IntentFrame per WSE-STD-060 Section 5.

    Represents continuous musical intent from UI/MIDI/Sensors.

    Properties:
    - May arrive asynchronously
    - Contains NO timing logic
    - Is NOT executable
    """
    # Continuous intent values (time-free)
    note: int = 60
    velocity: float = 0.0      # [0.0, 1.0]
    pitch_bend: float = 0.0    # [-1.0, 1.0]
    mod_wheel: float = 0.0     # [0.0, 1.0]
    expression: float = 1.0    # [0.0, 1.0]

    # Control intent (time-free)
    gate: bool = False
    sustain_pedal: bool = False

    # Family weights (time-free)
    family_weights: Dict[str, float] = field(default_factory=dict)

    # Regime hint (time-free)
    regime_hint: Optional[int] = None


@dataclass
class VoiceFrame:
    """
    VoiceFrame per WSE-STD-060 Section 5.

    Represents time-resolved, executable intent.

    Properties:
    - Emitted at BlockTime resolution
    - Contains all timing decisions
    - IS executable
    """
    # Identity
    voice_id: int
    block_index: int

    # Time-resolved values (quantized from IntentFrame)
    note_u8: int = 60
    cents_i16: int = 0
    velocity_q15: int = 0
    gain_q15: int = Q15_ONE
    activity_q15: int = 0
    pan_q15: int = 0

    # Time-resolved control (from Voice Engine timing)
    gate: bool = False
    gate_rise: bool = False   # Decided by Voice Engine
    gate_fall: bool = False   # Decided by Voice Engine

    # Time-resolved regime (from Voice Engine state machine)
    regime_id: int = 0
    degradation_level: int = 0

    # Time-resolved events (scheduled by Voice Engine)
    event_mask: int = 0
    event_strengths: List[int] = field(default_factory=lambda: [0] * 8)

    # Time-resolved family weights (slewed by Voice Engine)
    w_family_q15: List[int] = field(default_factory=lambda: [0] * 8)

    # Time-resolved tree selection
    tree_a_id: int = 0
    tree_b_id: int = 0
    tree_blend_q15: int = 0


def convert_intent_to_voice(
    intent: IntentFrame,
    voice_id: int,
    block_index: int,
    # These timing parameters come from Voice Engine state
    gate_rise: bool = False,
    gate_fall: bool = False,
    regime_id: int = 0,
    event_mask: int = 0,
    event_strengths: Optional[List[int]] = None,
) -> VoiceFrame:
    """
    Convert IntentFrame to VoiceFrame.

    This is the ONLY legal time domain conversion:
    GestureTime (via IntentFrame) → VoiceTime (via Voice Engine) → BlockTime (via VoiceFrame)

    The Voice Engine provides all timing decisions:
    - gate_rise/gate_fall (note lifecycle)
    - regime_id (regime state machine)
    - event_mask/event_strengths (scheduled events)
    """
    # Convert velocity to Q15
    velocity_q15 = int(intent.velocity * Q15_ONE)

    # Convert family weights to Q15 array
    w_family_q15 = [0] * 8
    family_names = ['CALM', 'TENSE', 'FOLD', 'CUSP', 'SWALLOW', 'PULSE', 'SPARSE', 'DENSE']
    for i, name in enumerate(family_names):
        if name in intent.family_weights:
            w_family_q15[i] = int(intent.family_weights[name] * Q15_ONE)

    return VoiceFrame(
        voice_id=voice_id,
        block_index=block_index,
        note_u8=intent.note,
        cents_i16=int(intent.pitch_bend * Q15_QUARTER),  # q15-literal-ok (cents scale, not Q15)
        velocity_q15=velocity_q15,
        gain_q15=int(intent.expression * Q15_ONE),
        gate=intent.gate,
        gate_rise=gate_rise,
        gate_fall=gate_fall,
        regime_id=regime_id if intent.regime_hint is None else intent.regime_hint,
        event_mask=event_mask,
        event_strengths=event_strengths or [0] * 8,
        w_family_q15=w_family_q15,
    )


# =============================================================================
# DETERMINISM GUARANTEE (Rule 060.3)
# =============================================================================

@dataclass
class DeterminismChecker:
    """
    Validates scheduling invariance per Rule 060.3.

    If VoiceFrames are identical and numeric rules are followed,
    then reordering, skipping, or parallelizing downstream engines
    MUST NOT change musical timing or structure.
    """

    def check_voice_frame_equivalence(
        self,
        frame1: VoiceFrame,
        frame2: VoiceFrame,
    ) -> bool:
        """Check if two VoiceFrames are equivalent."""
        return (
            frame1.voice_id == frame2.voice_id and
            frame1.block_index == frame2.block_index and
            frame1.note_u8 == frame2.note_u8 and
            frame1.cents_i16 == frame2.cents_i16 and
            frame1.velocity_q15 == frame2.velocity_q15 and
            frame1.gain_q15 == frame2.gain_q15 and
            frame1.gate == frame2.gate and
            frame1.gate_rise == frame2.gate_rise and
            frame1.gate_fall == frame2.gate_fall and
            frame1.regime_id == frame2.regime_id and
            frame1.event_mask == frame2.event_mask and
            frame1.event_strengths == frame2.event_strengths and
            frame1.w_family_q15 == frame2.w_family_q15
        )

    def validate_scheduling_invariance(
        self,
        original_frames: List[VoiceFrame],
        reordered_frames: List[VoiceFrame],
    ) -> bool:
        """
        Validate that reordering does not change frame content.

        Per Rule 060.3: Reordering MUST NOT change musical timing.
        """
        if len(original_frames) != len(reordered_frames):
            return False

        # Sort both by (voice_id, block_index) for comparison
        original_sorted = sorted(
            original_frames,
            key=lambda f: (f.voice_id, f.block_index)
        )
        reordered_sorted = sorted(
            reordered_frames,
            key=lambda f: (f.voice_id, f.block_index)
        )

        for f1, f2 in zip(original_sorted, reordered_sorted):
            if not self.check_voice_frame_equivalence(f1, f2):
                return False

        return True


# =============================================================================
# COMPLIANCE SUMMARY (Section 9)
# =============================================================================

@dataclass
class ComplianceReport:
    """Compliance report for WSE-STD-060."""
    engine_type: EngineType
    is_compliant: bool
    violations: List[str]
    pattern_violations: List[PatternViolation]

    def __str__(self) -> str:
        status = "COMPLIANT" if self.is_compliant else "NON-COMPLIANT"
        lines = [f"{self.engine_type.value}: {status}"]

        if self.violations:
            lines.append("  Rule violations:")
            for v in self.violations:
                lines.append(f"    - {v}")

        if self.pattern_violations:
            lines.append("  Pattern violations:")
            for pv in self.pattern_violations:
                lines.append(f"    - {pv}")

        return "\n".join(lines)


def check_compliance(
    engine_type: EngineType,
    code: Optional[str] = None,
    has_counters: bool = False,
    has_clocks: bool = False,
    has_phase_tracking: bool = False,
    has_adsr: bool = False,
    has_event_scheduling: bool = False,
) -> ComplianceReport:
    """
    Check WSE-STD-060 compliance for an engine.

    Per Section 9 summary:
    - Musical time lives ONLY in the Voice Engine
    - All other engines are time-blind
    - All timing is quantized to BlockTime
    - Violations are structural errors, not tuning mistakes
    """
    validator = TimeOwnershipValidator()

    # Check rule violations
    violations = validator.validate_engine(
        engine_type=engine_type,
        has_counters=has_counters,
        has_clocks=has_clocks,
        has_phase_tracking=has_phase_tracking,
        has_adsr=has_adsr,
        has_event_scheduling=has_event_scheduling,
    )

    # Check pattern violations in code
    pattern_violations = []
    if code:
        pattern_violations = detect_prohibited_patterns(code, engine_type)

    is_compliant = len(violations) == 0 and len(pattern_violations) == 0

    return ComplianceReport(
        engine_type=engine_type,
        is_compliant=is_compliant,
        violations=violations,
        pattern_violations=pattern_violations,
    )


# =============================================================================
# BEHAVIOR TABLES (Section 4 Summary)
# =============================================================================

def get_engine_responsibilities(engine_type: EngineType) -> Dict[str, str]:
    """
    Get engine responsibilities per Section 4.

    Returns dict of responsibility -> behavior.
    """
    if engine_type == EngineType.VOICE_ENGINE:
        return {
            'counters_clocks': 'MUST maintain',
            'timing_quantization': 'MUST quantize to BlockTime',
            'voiceframe_emission': 'MUST emit one per voice per block',
            'event_timing': 'MUST decide WHEN events occur',
            'operator_intent': 'MUST decide WHAT operators intended',
            'periodic_scheduling': 'MAY use "every N blocks"',
        }
    elif engine_type == EngineType.SYMBOL_ENGINE:
        return {
            'counters_clocks': 'MUST NOT maintain',
            'block_modulo': 'MUST NOT use',
            'phase_detection': 'MUST NOT perform',
            'event_triggering': 'MUST NOT trigger based on time',
            'frame_treatment': 'MUST treat as complete command',
            'operator_application': 'MUST apply exactly as instructed',
            'temporal_state': 'MUST remain stateless',
        }
    elif engine_type == EngineType.WAVELET_ENGINE:
        return {
            'coefficient_acceptance': 'MUST accept coefficients',
            'inverse_transform': 'MUST perform deterministically',
            'interpolation': 'MUST perform when instructed',
            'note_observation': 'MUST NOT observe on/off',
            'envelopes': 'MUST NOT contain',
            'transition_scheduling': 'MUST NOT schedule',
            'event_triggering': 'MUST NOT trigger',
        }
    elif engine_type == EngineType.SAFETY_ENGINE:
        return {
            'block_invariants': 'MUST enforce',
            'ola_declick': 'MUST apply',
            'idempotence': 'MUST be idempotent',
            'phase_detection': 'MUST NOT detect',
            'regime_alteration': 'MUST NOT alter',
            'event_introduction': 'MUST NOT introduce',
        }
    else:
        return {}


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'BLOCK_SIZE_SAMPLES',
    'SAMPLE_RATE_48K',
    'SAMPLE_RATE_44K',
    # Enums
    'TimeDomain',
    'EngineType',
    'ProhibitedPattern',
    # Capabilities
    'EngineCapabilities',
    'ENGINE_CAPABILITIES',
    'get_engine_capabilities',
    # BlockTime
    'BlockTime',
    # Pattern detection
    'PatternViolation',
    'detect_prohibited_patterns',
    # Validation
    'TimeOwnershipValidator',
    # Intent/Voice conversion
    'IntentFrame',
    'VoiceFrame',
    'convert_intent_to_voice',
    # Determinism
    'DeterminismChecker',
    # Compliance
    'ComplianceReport',
    'check_compliance',
    # Behavior tables
    'get_engine_responsibilities',
]
