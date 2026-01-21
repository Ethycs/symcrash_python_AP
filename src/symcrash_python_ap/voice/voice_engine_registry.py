"""
Voice Engine Registry (WSE-STD-VOICEENGINEREGISTRY v0)
=======================================================

Single source of truth for engine-owned parameters.

This registry defines parameters that control:
- Voice DSP configuration (oscillator modes, tree IDs, IDWPT config)
- CMMM/L5 selectors (regime graphs, program routing)
- Planner hints (K policy, tail hints, feature gates)
- Engine-owned stateful sub-engines

Separate from ControlRegistry which handles cockpit macro controls.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import IntEnum, Enum


# =============================================================================
# ENUMS
# =============================================================================

class ValueType(IntEnum):
    """Parameter value types (WSE-STD-VOICEENGINEREGISTRY §1.1)."""
    Q15U = 0   # Unsigned Q0.15 (0 to 32767)
    I16 = 1    # Signed int16 (-32768 to 32767)
    U16 = 2    # Unsigned int16 (0 to 65535)
    BOOL = 3   # Boolean (0 or 1)
    ENUM8 = 4  # Enumerated value (0 to 255)
    U32 = 5    # Unsigned int32 (0 to 4294967295) - for Q16.16 ratios etc.
    Q15S = 6   # Signed Q0.15 (-32768 to 32767)
    I32 = 7    # Signed int32 (-2147483648 to 2147483647)


class UpdateMode(IntEnum):
    """Parameter update modes (WSE-STD-VOICEENGINEREGISTRY §1.2)."""
    ABS = 0    # Set value directly
    DELTA = 1  # Add delta (signed)
    BOTH = 2   # Accept both modes


class Persistence(IntEnum):
    """Parameter persistence scope (WSE-STD-VOICEENGINEREGISTRY §1.3)."""
    PER_VOICE = 0  # Applied per-voice on allocation
    PROGRAM = 1    # Shared defaults for voices under a program
    GLOBAL = 2     # Global engine-wide parameter


class Domain(IntEnum):
    """Parameter domain classification (WSE-STD-VOICEENGINEREGISTRY §11)."""
    CORE = 0      # Core voice parameters (0-31)
    CMMM = 1      # CMMM parameters (32-63)
    CCS = 2       # Coefficient/Carrier/Synthesis parameters (64-127)
    IDWPT = 3     # IDWPT wavelet parameters (128-159)
    PLANNER = 4   # Budget planner hints (160-191)
    DEBUG = 5     # Debug parameters (240-255)


class VoiceParamID(IntEnum):
    """
    Voice parameter IDs (WSE-STD-VOICEENGINEREGISTRY §11).

    Allocation ranges:
    - CORE: 0-31
    - CMMM: 32-63
    - CCS: 64-127
    - IDWPT: 128-159
    - PLANNER: 160-191
    - DEBUG: 240-255
    """
    # CORE (0-31)
    PROGRAM_ID = 0
    REGIME_GRAPH_ID = 1
    TREE_ID = 2
    VOICE_CLASS = 3
    SEED_POLICY = 4

    # CMMM (32-63)
    ROUTE_ID = 32
    REGIME_BIAS_Q15 = 33
    DWELL_MIN_BLOCKS = 34
    TRANSITION_POLICY = 35

    # CCS (64-127) - Coefficient/Carrier/Synthesis
    # Per WSE-STD-001: Q1.15 for gains/strengths, Q0.16 for unipolar, ENUM8 for modes
    OSC_MODE = 64           # 0=A(LUT), 1=B(Rotator), 2=Sample, 3=FM6
    WAVE_ID = 65            # Waveform LUT index
    H_REQUEST = 66          # Harmonic count request
    PWM_ENABLE = 67         # PWM modulation enable
    FM_ENABLE = 68          # Mode B FM feedback enable
    WAVEFORM_TYPE = 69      # 0=sine, 1=square, 2=pulse, 3=saw, 4=saw_down, 5=tri
    PWM_WIDTH_Q15 = 70      # Duty cycle Q0.15 (16384=50%)
    SYNC_ENABLE = 71        # Hard sync enable
    SYNC_RATIO_Q16 = 72     # Master/slave ratio Q16.16 (65536=1.0)
    FM_DEPTH_Q15 = 73       # Mode B FM depth Q0.15
    FM_SOURCE = 74          # 0=off, 1=feedback, 2=LFO
    UNISON_VOICES = 75      # Unison voice count (1-8)
    UNISON_SPREAD = 76      # Detune spread in cents
    DRONE_MODE = 77         # Sustained coefficient mode
    LEGATO_MODE = 78        # Smooth pitch transitions
    SAMPLE_RATE_Q16 = 79    # Sample playback rate Q16.16

    # FM6 (80-99) - 6-operator FM synthesis (OSC_MODE=3)
    FM6_ALGORITHM = 80      # DX7 algorithm 1-32
    FM6_FEEDBACK = 81       # Feedback level 0-7 (maps to Q15 via DX7 table)
    FM6_OP0_RATIO = 82      # Op1 frequency ratio Q16.16 (65536=1.0)
    FM6_OP1_RATIO = 83      # Op2 frequency ratio Q16.16
    FM6_OP2_RATIO = 84      # Op3 frequency ratio Q16.16
    FM6_OP3_RATIO = 85      # Op4 frequency ratio Q16.16
    FM6_OP4_RATIO = 86      # Op5 frequency ratio Q16.16
    FM6_OP5_RATIO = 87      # Op6 frequency ratio Q16.16
    FM6_OP0_LEVEL = 88      # Op1 output level Q0.15 (32767=full)
    FM6_OP1_LEVEL = 89      # Op2 output level Q0.15
    FM6_OP2_LEVEL = 90      # Op3 output level Q0.15
    FM6_OP3_LEVEL = 91      # Op4 output level Q0.15
    FM6_OP4_LEVEL = 92      # Op5 output level Q0.15
    FM6_OP5_LEVEL = 93      # Op6 output level Q0.15
    FM6_TOPO_A = 94         # Source archetype 0-7 (CHAIN,STACK,DUAL,SPLIT,etc)
    FM6_TOPO_B = 95         # Target archetype 0-7
    FM6_TOPO_BLEND = 96     # Topology morph Q0.15 (0=A, 32767=B)

    # Filter (97-104) - Subtractive synthesis filter
    FILTER_ENABLE = 97          # Filter enable (0=bypass, 1=active)
    FILTER_TYPE = 98            # 0=LP, 1=HP, 2=BP, 3=Notch
    FILTER_CUTOFF_Q15 = 99      # Cutoff frequency Q0.15 (0=20Hz, 32767=20kHz log)
    FILTER_RESONANCE_Q15 = 100  # Resonance/Q Q0.15 (0=none, 32767=self-osc)
    FILTER_ENV_DEPTH_Q15 = 101  # Envelope mod depth Q1.15 (signed, -32768 to 32767)
    FILTER_KEY_TRACK_Q15 = 102  # Key tracking Q0.15 (0=none, 16384=50%, 32767=100%)
    FILTER_DRIVE_Q15 = 103      # Pre-filter drive/saturation Q0.15
    FILTER_SLOPE = 104          # Slope: 0=12dB/oct, 1=24dB/oct, 2=6dB/oct

    # IDWPT (128-159)
    K_REQUEST = 128
    TAIL_POLICY_HINT = 129
    TREE_XFADE_BLOCKS = 130
    WET_ENABLE = 131

    # PLANNER (160-191)
    MAX_TICKS_HINT = 160
    ALLOW_NONLINEAR_HINT = 161

    # DEBUG (240-255)
    BYPASS_IDWPT = 240
    FREEZE_COEFFS = 241


# Flags for VoiceParamEntry (§1.5)
FLAG_CUMULATIVE = 1 << 0  # Multiple params with same binding add deterministically
FLAG_INIT_ONLY = 1 << 1   # Only applied at voice allocation
FLAG_DEBUG_ONLY = 1 << 2  # Ignored unless debug build
FLAG_BUDGET_HINT = 1 << 3  # May be used by planner as hint


# =============================================================================
# REGISTRY ENTRY
# =============================================================================

@dataclass(frozen=True)
class VoiceParamEntry:
    """
    Single voice-engine parameter definition (WSE-STD-VOICEENGINEREGISTRY §1).
    """
    param_id: int
    name: str
    value_type: ValueType
    min_value: int
    max_value: int
    default_value: int
    update_mode: UpdateMode
    persistence: Persistence
    domain: Domain
    binding_target: str  # What this parameter affects
    flags: int = 0

    def clamp(self, value: int) -> int:
        """Clamp value to valid range."""
        return max(self.min_value, min(self.max_value, value))

    def apply_abs(self, value: int) -> int:
        """Apply absolute update (with clamping)."""
        if self.update_mode == UpdateMode.DELTA:
            raise ValueError(f"Parameter {self.name} only supports DELTA updates")
        return self.clamp(value)

    def apply_delta(self, current: int, delta: int) -> int:
        """Apply delta update (with clamping)."""
        if self.update_mode == UpdateMode.ABS:
            raise ValueError(f"Parameter {self.name} only supports ABS updates")
        return self.clamp(current + delta)


# =============================================================================
# VOICE ENGINE REGISTRY
# =============================================================================

class VoiceEngineRegistry:
    """
    Voice Engine parameter registry (WSE-STD-VOICEENGINEREGISTRY v0).

    Single source of truth for engine-owned parameters.
    """

    def __init__(self, entries: Optional[List[VoiceParamEntry]] = None):
        """
        Initialize registry.

        Args:
            entries: List of parameter entries (defaults to VOICE_ENGINE_REGISTRY_V0)
        """
        self._entries: Dict[int, VoiceParamEntry] = {}

        entries = entries or VOICE_ENGINE_REGISTRY_V0
        for entry in entries:
            self._entries[entry.param_id] = entry

    def get(self, param_id: int) -> Optional[VoiceParamEntry]:
        """Get parameter entry by ID."""
        return self._entries.get(param_id)

    def clamp(self, param_id: int, value: int) -> int:
        """Clamp value to parameter range."""
        entry = self.get(param_id)
        if entry is None:
            return value  # Unknown param, pass through
        return entry.clamp(value)

    def apply_abs(self, param_id: int, value: int) -> int:
        """Apply absolute value update (with clamping)."""
        entry = self.get(param_id)
        if entry is None:
            return value
        return entry.apply_abs(value)

    def apply_delta(self, param_id: int, current: int, delta: int) -> int:
        """Apply delta update (with clamping)."""
        entry = self.get(param_id)
        if entry is None:
            return current + delta  # Unknown param, apply unclamped delta
        return entry.apply_delta(current, delta)

    def get_default(self, param_id: int) -> int:
        """Get default value for parameter."""
        entry = self.get(param_id)
        if entry is None:
            return 0
        return entry.default_value

    def list_by_domain(self, domain: Domain) -> List[VoiceParamEntry]:
        """Get all parameters in a domain."""
        return [e for e in self._entries.values() if e.domain == domain]

    def list_by_persistence(self, persistence: Persistence) -> List[VoiceParamEntry]:
        """Get all parameters with specific persistence scope."""
        return [e for e in self._entries.values() if e.persistence == persistence]

    def validate_all(self, overrides: Dict[int, int]) -> Dict[int, int]:
        """
        Validate and clamp all override values.

        Args:
            overrides: param_id → value mapping

        Returns:
            Clamped overrides (unknown params removed)
        """
        validated = {}
        for param_id, value in overrides.items():
            if param_id in self._entries:
                validated[param_id] = self.clamp(param_id, value)
        return validated


# =============================================================================
# DEFAULT REGISTRY TABLE (WSE-STD-VOICEENGINEREGISTRY v0 §4)
# =============================================================================

VOICE_ENGINE_REGISTRY_V0: List[VoiceParamEntry] = [

    # =========================================================================
    # CORE DOMAIN (0-31)
    # =========================================================================

    VoiceParamEntry(
        param_id=VoiceParamID.PROGRAM_ID,
        name="program_id",
        value_type=ValueType.U16,
        min_value=1000,
        max_value=9999,
        default_value=1001,  # Glitch Violin
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PER_VOICE,
        domain=Domain.CORE,
        binding_target="core.program_id",
        flags=FLAG_INIT_ONLY,
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.REGIME_GRAPH_ID,
        name="regime_graph_id",
        value_type=ValueType.U16,
        min_value=0,
        max_value=255,
        default_value=0,
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CORE,
        binding_target="core.regime_graph_id",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.TREE_ID,
        name="tree_id",
        value_type=ValueType.ENUM8,
        min_value=0,
        max_value=4,  # 5 trees: 0=Balanced, 1=Harmonic, 2=Texture, 3=Impact, 4=Drone
        default_value=1,  # Harmonic
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PER_VOICE,
        domain=Domain.CORE,
        binding_target="core.tree_id",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.VOICE_CLASS,
        name="voice_class",
        value_type=ValueType.ENUM8,
        min_value=0,
        max_value=1,  # 0=MAIN, 1=RESIDUAL
        default_value=0,  # MAIN
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PER_VOICE,
        domain=Domain.CORE,
        binding_target="core.voice_class",
        flags=FLAG_INIT_ONLY,
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.SEED_POLICY,
        name="seed_policy",
        value_type=ValueType.ENUM8,
        min_value=0,
        max_value=3,  # 0=deterministic, 1=per-voice, 2=per-note, 3=random
        default_value=0,  # Deterministic
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CORE,
        binding_target="core.seed_policy",
    ),

    # =========================================================================
    # CMMM DOMAIN (32-63)
    # =========================================================================

    VoiceParamEntry(
        param_id=VoiceParamID.ROUTE_ID,
        name="route_id",
        value_type=ValueType.ENUM8,
        min_value=0,
        max_value=15,
        default_value=0,
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CMMM,
        binding_target="cmmm.route_id",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.REGIME_BIAS_Q15,
        name="regime_bias_q15",
        value_type=ValueType.Q15U,
        min_value=0,
        max_value=32767,
        default_value=0,
        update_mode=UpdateMode.BOTH,
        persistence=Persistence.PROGRAM,
        domain=Domain.CMMM,
        binding_target="cmmm.regime_bias",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.DWELL_MIN_BLOCKS,
        name="dwell_min_blocks",
        value_type=ValueType.U16,
        min_value=1,
        max_value=1000,
        default_value=8,
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CMMM,
        binding_target="cmmm.dwell_min_blocks",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.TRANSITION_POLICY,
        name="transition_policy",
        value_type=ValueType.ENUM8,
        min_value=0,
        max_value=2,  # 0=deterministic, 1=weighted, 2=adaptive
        default_value=0,
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CMMM,
        binding_target="cmmm.transition_policy",
    ),

    # =========================================================================
    # CCS DOMAIN (64-127)
    # =========================================================================

    VoiceParamEntry(
        param_id=VoiceParamID.OSC_MODE,
        name="osc_mode",
        value_type=ValueType.ENUM8,
        min_value=0,
        max_value=3,  # 0=Mode A (LUT), 1=Mode B (Rotator), 2=Sample, 3=FM6
        default_value=0,  # Mode A
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.osc_mode",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.WAVE_ID,
        name="wave_id",
        value_type=ValueType.ENUM8,
        min_value=0,
        max_value=255,
        default_value=0,  # Basic harmonic series
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.wave_id",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.H_REQUEST,
        name="h_request",
        value_type=ValueType.U16,
        min_value=1,
        max_value=16,  # Up to 16 harmonics
        default_value=8,
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.h_request",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.PWM_ENABLE,
        name="pwm_enable",
        value_type=ValueType.BOOL,
        min_value=0,
        max_value=1,
        default_value=0,
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.pwm_enable",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.FM_ENABLE,
        name="fm_enable",
        value_type=ValueType.BOOL,
        min_value=0,
        max_value=1,
        default_value=0,
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.fm_enable",
    ),

    # --- Waveform and PWM (69-72) ---

    VoiceParamEntry(
        param_id=VoiceParamID.WAVEFORM_TYPE,
        name="waveform_type",
        value_type=ValueType.ENUM8,
        min_value=0,
        max_value=5,  # 0=sine, 1=square, 2=pulse, 3=saw, 4=saw_down, 5=tri
        default_value=0,  # Sine
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.waveform_type",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.PWM_WIDTH_Q15,
        name="pwm_width_q15",
        value_type=ValueType.Q15U,
        min_value=0,
        max_value=32767,  # Q0.15: 16384 = 50% duty cycle
        default_value=16384,  # 50%
        update_mode=UpdateMode.BOTH,
        persistence=Persistence.PER_VOICE,
        domain=Domain.CCS,
        binding_target="ccs.pwm_width",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.SYNC_ENABLE,
        name="sync_enable",
        value_type=ValueType.BOOL,
        min_value=0,
        max_value=1,
        default_value=0,
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.sync_enable",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.SYNC_RATIO_Q16,
        name="sync_ratio_q16",
        value_type=ValueType.U16,
        min_value=0,
        max_value=65535,  # Q16.16 stored as U16 (high bits) - 65536 = 1.0
        default_value=65535,  # ~1.0
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.sync_ratio",
    ),

    # --- Mode B FM (73-74) ---

    VoiceParamEntry(
        param_id=VoiceParamID.FM_DEPTH_Q15,
        name="fm_depth_q15",
        value_type=ValueType.Q15U,
        min_value=0,
        max_value=32767,  # Q0.15 depth
        default_value=0,
        update_mode=UpdateMode.BOTH,
        persistence=Persistence.PER_VOICE,
        domain=Domain.CCS,
        binding_target="ccs.fm_depth",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.FM_SOURCE,
        name="fm_source",
        value_type=ValueType.ENUM8,
        min_value=0,
        max_value=2,  # 0=off, 1=feedback, 2=LFO
        default_value=0,
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.fm_source",
    ),

    # --- Unison (75-76) ---

    VoiceParamEntry(
        param_id=VoiceParamID.UNISON_VOICES,
        name="unison_voices",
        value_type=ValueType.ENUM8,
        min_value=1,
        max_value=8,  # 1-8 voices
        default_value=1,  # No unison
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.unison_voices",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.UNISON_SPREAD,
        name="unison_spread",
        value_type=ValueType.U16,
        min_value=0,
        max_value=100,  # Detune spread in cents (0-100)
        default_value=10,
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.unison_spread",
    ),

    # --- Mode flags (77-78) ---

    VoiceParamEntry(
        param_id=VoiceParamID.DRONE_MODE,
        name="drone_mode",
        value_type=ValueType.BOOL,
        min_value=0,
        max_value=1,
        default_value=0,
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.drone_mode",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.LEGATO_MODE,
        name="legato_mode",
        value_type=ValueType.BOOL,
        min_value=0,
        max_value=1,
        default_value=0,
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.legato_mode",
    ),

    # --- Sample mode (79) ---

    VoiceParamEntry(
        param_id=VoiceParamID.SAMPLE_RATE_Q16,
        name="sample_rate_q16",
        value_type=ValueType.U16,
        min_value=0,
        max_value=65535,  # Q16.16 playback rate (65536 = 1.0x)
        default_value=65535,  # ~1.0x
        update_mode=UpdateMode.BOTH,
        persistence=Persistence.PER_VOICE,
        domain=Domain.CCS,
        binding_target="ccs.sample_rate",
    ),

    # --- FM6 6-operator synthesis (80-96) ---

    VoiceParamEntry(
        param_id=VoiceParamID.FM6_ALGORITHM,
        name="fm6_algorithm",
        value_type=ValueType.ENUM8,
        min_value=1,
        max_value=32,  # DX7 algorithms 1-32
        default_value=1,  # Algorithm 1
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.fm6_algorithm",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.FM6_FEEDBACK,
        name="fm6_feedback",
        value_type=ValueType.ENUM8,
        min_value=0,
        max_value=7,  # 0-7 maps to DX7 feedback table
        default_value=0,
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.fm6_feedback",
    ),

    # Op frequency ratios Q16.16 (65536 = 1.0x carrier frequency)
    # Stored as U32 to support ratios > 1.0x (e.g., 131072 = 2.0x)
    VoiceParamEntry(
        param_id=VoiceParamID.FM6_OP0_RATIO,
        name="fm6_op0_ratio",
        value_type=ValueType.U32,
        min_value=0,
        max_value=0xFFFFFFFF,
        default_value=65536,  # 1.0x in Q16.16
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.fm6_op0_ratio",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.FM6_OP1_RATIO,
        name="fm6_op1_ratio",
        value_type=ValueType.U32,
        min_value=0,
        max_value=0xFFFFFFFF,
        default_value=65536,  # 1.0x in Q16.16
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.fm6_op1_ratio",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.FM6_OP2_RATIO,
        name="fm6_op2_ratio",
        value_type=ValueType.U32,
        min_value=0,
        max_value=0xFFFFFFFF,
        default_value=65536,  # 1.0x in Q16.16
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.fm6_op2_ratio",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.FM6_OP3_RATIO,
        name="fm6_op3_ratio",
        value_type=ValueType.U32,
        min_value=0,
        max_value=0xFFFFFFFF,
        default_value=65536,  # 1.0x in Q16.16
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.fm6_op3_ratio",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.FM6_OP4_RATIO,
        name="fm6_op4_ratio",
        value_type=ValueType.U32,
        min_value=0,
        max_value=0xFFFFFFFF,
        default_value=65536,  # 1.0x in Q16.16
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.fm6_op4_ratio",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.FM6_OP5_RATIO,
        name="fm6_op5_ratio",
        value_type=ValueType.U32,
        min_value=0,
        max_value=0xFFFFFFFF,
        default_value=65536,  # 1.0x in Q16.16
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.fm6_op5_ratio",
    ),

    # Op output levels Q0.15 (32767 = full output)
    VoiceParamEntry(
        param_id=VoiceParamID.FM6_OP0_LEVEL,
        name="fm6_op0_level",
        value_type=ValueType.Q15U,
        min_value=0,
        max_value=32767,
        default_value=32767,  # Full level
        update_mode=UpdateMode.BOTH,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.fm6_op0_level",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.FM6_OP1_LEVEL,
        name="fm6_op1_level",
        value_type=ValueType.Q15U,
        min_value=0,
        max_value=32767,
        default_value=0,
        update_mode=UpdateMode.BOTH,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.fm6_op1_level",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.FM6_OP2_LEVEL,
        name="fm6_op2_level",
        value_type=ValueType.Q15U,
        min_value=0,
        max_value=32767,
        default_value=0,
        update_mode=UpdateMode.BOTH,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.fm6_op2_level",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.FM6_OP3_LEVEL,
        name="fm6_op3_level",
        value_type=ValueType.Q15U,
        min_value=0,
        max_value=32767,
        default_value=0,
        update_mode=UpdateMode.BOTH,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.fm6_op3_level",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.FM6_OP4_LEVEL,
        name="fm6_op4_level",
        value_type=ValueType.Q15U,
        min_value=0,
        max_value=32767,
        default_value=0,
        update_mode=UpdateMode.BOTH,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.fm6_op4_level",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.FM6_OP5_LEVEL,
        name="fm6_op5_level",
        value_type=ValueType.Q15U,
        min_value=0,
        max_value=32767,
        default_value=0,
        update_mode=UpdateMode.BOTH,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.fm6_op5_level",
    ),

    # Topology morphing
    VoiceParamEntry(
        param_id=VoiceParamID.FM6_TOPO_A,
        name="fm6_topo_a",
        value_type=ValueType.ENUM8,
        min_value=0,
        max_value=7,  # Source archetype 0-7 (CHAIN, STACK, DUAL, SPLIT, etc.)
        default_value=0,  # CHAIN
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.fm6_topo_a",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.FM6_TOPO_B,
        name="fm6_topo_b",
        value_type=ValueType.ENUM8,
        min_value=0,
        max_value=7,  # Target archetype 0-7
        default_value=0,  # CHAIN
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.fm6_topo_b",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.FM6_TOPO_BLEND,
        name="fm6_topo_blend",
        value_type=ValueType.Q15U,
        min_value=0,
        max_value=32767,  # Q0.15: 0 = A, 32767 = B
        default_value=0,  # Full A
        update_mode=UpdateMode.BOTH,
        persistence=Persistence.PER_VOICE,
        domain=Domain.CCS,
        binding_target="ccs.fm6_topo_blend",
    ),

    # --- Filter (97-104) - Subtractive synthesis ---

    VoiceParamEntry(
        param_id=VoiceParamID.FILTER_ENABLE,
        name="filter_enable",
        value_type=ValueType.BOOL,
        min_value=0,
        max_value=1,
        default_value=0,  # Bypass by default
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.filter_enable",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.FILTER_TYPE,
        name="filter_type",
        value_type=ValueType.ENUM8,
        min_value=0,
        max_value=3,  # 0=LP, 1=HP, 2=BP, 3=Notch
        default_value=0,  # Lowpass
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.filter_type",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.FILTER_CUTOFF_Q15,
        name="filter_cutoff_q15",
        value_type=ValueType.Q15U,
        min_value=0,
        max_value=32767,  # Q0.15: log scale 20Hz-20kHz
        default_value=32767,  # Wide open
        update_mode=UpdateMode.BOTH,
        persistence=Persistence.PER_VOICE,
        domain=Domain.CCS,
        binding_target="ccs.filter_cutoff",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.FILTER_RESONANCE_Q15,
        name="filter_resonance_q15",
        value_type=ValueType.Q15U,
        min_value=0,
        max_value=32767,  # Q0.15: 0=flat, 32767=self-oscillation
        default_value=0,  # No resonance
        update_mode=UpdateMode.BOTH,
        persistence=Persistence.PER_VOICE,
        domain=Domain.CCS,
        binding_target="ccs.filter_resonance",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.FILTER_ENV_DEPTH_Q15,
        name="filter_env_depth_q15",
        value_type=ValueType.I16,  # SIGNED for bipolar modulation
        min_value=-32768,
        max_value=32767,  # Q1.15: negative = inverted env
        default_value=0,  # No envelope modulation
        update_mode=UpdateMode.BOTH,
        persistence=Persistence.PER_VOICE,
        domain=Domain.CCS,
        binding_target="ccs.filter_env_depth",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.FILTER_KEY_TRACK_Q15,
        name="filter_key_track_q15",
        value_type=ValueType.Q15U,
        min_value=0,
        max_value=32767,  # Q0.15: 16384=50%, 32767=100% tracking
        default_value=0,  # No key tracking
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.filter_key_track",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.FILTER_DRIVE_Q15,
        name="filter_drive_q15",
        value_type=ValueType.Q15U,
        min_value=0,
        max_value=32767,  # Q0.15: pre-filter saturation
        default_value=0,  # Clean
        update_mode=UpdateMode.BOTH,
        persistence=Persistence.PER_VOICE,
        domain=Domain.CCS,
        binding_target="ccs.filter_drive",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.FILTER_SLOPE,
        name="filter_slope",
        value_type=ValueType.ENUM8,
        min_value=0,
        max_value=2,  # 0=12dB/oct, 1=24dB/oct, 2=6dB/oct
        default_value=1,  # 24dB/oct (classic 4-pole)
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.CCS,
        binding_target="ccs.filter_slope",
    ),

    # =========================================================================
    # IDWPT DOMAIN (128-159)
    # =========================================================================

    VoiceParamEntry(
        param_id=VoiceParamID.K_REQUEST,
        name="k_request",
        value_type=ValueType.ENUM8,
        min_value=0,
        max_value=6,  # Maps to K ladder: 0=32, 1=24, 2=16, 3=12, 4=8, 5=4, 6=1
        default_value=0,  # K=32
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PER_VOICE,
        domain=Domain.IDWPT,
        binding_target="idwpt.k_request",
        flags=FLAG_BUDGET_HINT,
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.TAIL_POLICY_HINT,
        name="tail_policy_hint",
        value_type=ValueType.ENUM8,
        min_value=0,
        max_value=3,  # 0=FULL, 1=MID, 2=MIN, 3=EMERGENCY
        default_value=0,  # FULL
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.IDWPT,
        binding_target="idwpt.tail_policy_hint",
        flags=FLAG_BUDGET_HINT,
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.TREE_XFADE_BLOCKS,
        name="tree_xfade_blocks",
        value_type=ValueType.U16,
        min_value=1,
        max_value=128,
        default_value=8,
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.IDWPT,
        binding_target="idwpt.tree_xfade_blocks",
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.WET_ENABLE,
        name="wet_enable",
        value_type=ValueType.BOOL,
        min_value=0,
        max_value=1,
        default_value=1,  # Wet path enabled by default
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.IDWPT,
        binding_target="idwpt.wet_enable",
    ),

    # =========================================================================
    # PLANNER DOMAIN (160-191)
    # =========================================================================

    VoiceParamEntry(
        param_id=VoiceParamID.MAX_TICKS_HINT,
        name="max_ticks_hint",
        value_type=ValueType.U16,
        min_value=0,
        max_value=20000,
        default_value=8000,  # STABLE_TONE budget
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.PLANNER,
        binding_target="planner.max_ticks_hint",
        flags=FLAG_BUDGET_HINT,
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.ALLOW_NONLINEAR_HINT,
        name="allow_nonlinear_hint",
        value_type=ValueType.BOOL,
        min_value=0,
        max_value=1,
        default_value=1,  # Allow nonlinear by default
        update_mode=UpdateMode.ABS,
        persistence=Persistence.PROGRAM,
        domain=Domain.PLANNER,
        binding_target="planner.allow_nonlinear_hint",
        flags=FLAG_BUDGET_HINT,
    ),

    # =========================================================================
    # DEBUG DOMAIN (240-255)
    # =========================================================================

    VoiceParamEntry(
        param_id=VoiceParamID.BYPASS_IDWPT,
        name="bypass_idwpt",
        value_type=ValueType.BOOL,
        min_value=0,
        max_value=1,
        default_value=0,
        update_mode=UpdateMode.ABS,
        persistence=Persistence.GLOBAL,
        domain=Domain.DEBUG,
        binding_target="debug.bypass_idwpt",
        flags=FLAG_DEBUG_ONLY,
    ),

    VoiceParamEntry(
        param_id=VoiceParamID.FREEZE_COEFFS,
        name="freeze_coeffs",
        value_type=ValueType.BOOL,
        min_value=0,
        max_value=1,
        default_value=0,
        update_mode=UpdateMode.ABS,
        persistence=Persistence.GLOBAL,
        domain=Domain.DEBUG,
        binding_target="debug.freeze_coeffs",
        flags=FLAG_DEBUG_ONLY,
    ),
]


# =============================================================================
# MODULE LEVEL ACCESSOR
# =============================================================================

_global_registry: Optional[VoiceEngineRegistry] = None


def get_registry() -> VoiceEngineRegistry:
    """Get global registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = VoiceEngineRegistry()
    return _global_registry


def reset_registry():
    """Reset global registry to defaults (for testing)."""
    global _global_registry
    _global_registry = None


__all__ = [
    # Enums
    'ValueType', 'UpdateMode', 'Persistence', 'Domain', 'VoiceParamID',
    # Flags
    'FLAG_CUMULATIVE', 'FLAG_INIT_ONLY', 'FLAG_DEBUG_ONLY', 'FLAG_BUDGET_HINT',
    # Classes
    'VoiceParamEntry', 'VoiceEngineRegistry',
    # Registry table
    'VOICE_ENGINE_REGISTRY_V0',
    # Accessor
    'get_registry', 'reset_registry',
]
