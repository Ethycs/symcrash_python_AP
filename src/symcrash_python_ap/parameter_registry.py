"""
Parameter Registry
==================

Implements Closure Appendix v0.1 Section A: Parameter Registry (Normative).

All controllable quantities in the system are registered parameters.
All parameter updates (UART, MIDI, editor, automation) must resolve
through this registry before affecting any engine.

Key Features:
- Canonical Parameter Model with domain, type, range, defaults
- ABS and DELTA update semantics with clamping
- Per-voice, Global, and Program persistence scopes
- Fixed-point arithmetic rules (q15, q8, u8)
- Determinism guarantees for replay
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum, IntEnum, auto

# Import canonical Q15 constants from primitives (single source of truth)
from .primitives import Q15_ONE, Q15_SCALE, Q15_NEG_ONE


# =============================================================================
# CONSTANTS
# =============================================================================

Q15_MIN = Q15_NEG_ONE  # -32768
Q15_MAX = Q15_ONE      # 32767
# Q15_SCALE imported from primitives

Q8_MIN = -128
Q8_MAX = 127

U8_MIN = 0
U8_MAX = 255

U16_MIN = 0
U16_MAX = 65535


# =============================================================================
# ENUMS
# =============================================================================

class ParamDomain(Enum):
    """Parameter domain classification."""
    VOICE = "VOICE"       # Per-voice parameters
    SYMBOL = "SYMBOL"     # Symbol executor parameters
    WAVELET = "WAVELET"   # Wavelet engine parameters
    MATERIAL = "MATERIAL" # Material/physics parameters
    SAFETY = "SAFETY"     # Safety and constraint parameters


class ParamType(Enum):
    """Parameter data types."""
    Q15 = "q15"    # Signed int16, range [-1.0, +0.99997]
    Q8 = "q8"      # Signed int8, normalized
    U8 = "u8"      # Unsigned int8
    U16 = "u16"    # Unsigned int16
    BOOL = "bool"  # Boolean (0 or 1)
    ENUM = "enum"  # Enumerated value


class UpdateMode(Enum):
    """Parameter update modes."""
    ABS = "ABS"      # Absolute value update
    DELTA = "DELTA"  # Relative delta update
    BOTH = "BOTH"    # Supports both modes


class Persistence(Enum):
    """Parameter persistence scope."""
    PER_VOICE = "PER_VOICE"  # Each voice has independent state
    GLOBAL = "GLOBAL"        # Single global value
    PROGRAM = "PROGRAM"      # Per-program value


class FieldMask(IntEnum):
    """
    Semantic presence bitmap for partial parameter updates.

    FieldMask is the addressing and intent layer that enables:
    - Sparse updates without bandwidth waste ("only these changed")
    - ABS/DELTA disambiguation per parameter
    - Cross-engine routing (Voice → Wavelet → Symbol → Safety)
    - Deterministic replay and debugging
    - Editor-safe defaults ("unset" vs "set to zero")

    Bit allocation by domain:
    - Bits 0-15:  Voice domain
    - Bits 16-23: Symbol domain
    - Bits 24-31: Safety domain
    """
    # Voice Domain (bits 0-15)
    GAIN_Q15 = 0
    ACTIVITY_Q15 = 1
    PAN_Q15 = 2
    SEND_A_Q15 = 3
    SEND_B_Q15 = 4
    NOTE_U8 = 5
    CENTS_I16 = 6
    REGIME_ID = 7
    DEGRADATION = 8
    TREE_A_ID = 9
    TREE_B_ID = 10
    TREE_BLEND_Q15 = 11
    FAMILY_WEIGHTS = 12      # All 8 family weights as group
    EVENT_MASK = 13
    EVENT_STRENGTHS = 14     # All 8 lane strengths as group

    # Symbol Domain (bits 16-23)
    K_MODE = 16
    DRIVE_Q15 = 17
    TONE_Q15 = 18
    BIAS_Q15 = 19
    PRESSURE_Q15 = 20

    # Safety Domain (bits 24-31)
    ENERGY_CAP_Q15 = 24
    GEAR_STRENGTH_Q15 = 25
    TRACTION_Q15 = 26


# =============================================================================
# PARAMETER DEFINITION
# =============================================================================

@dataclass
class ParameterDef:
    """
    Parameter definition in the registry.

    All controllable quantities must be registered with this structure.
    """
    param_id: int
    name: str
    domain: ParamDomain
    dtype: ParamType
    min_value: int
    max_value: int
    default_value: int
    update_mode: UpdateMode = UpdateMode.BOTH
    persistence: Persistence = Persistence.PER_VOICE
    description: str = ""

    def validate_value(self, value: int) -> int:
        """Clamp value to valid range."""
        return max(self.min_value, min(self.max_value, value))

    def type_range(self) -> Tuple[int, int]:
        """Get type-based range."""
        if self.dtype == ParamType.Q15:
            return Q15_MIN, Q15_MAX
        elif self.dtype == ParamType.Q8:
            return Q8_MIN, Q8_MAX
        elif self.dtype == ParamType.U8:
            return U8_MIN, U8_MAX
        elif self.dtype == ParamType.U16:
            return U16_MIN, U16_MAX
        elif self.dtype == ParamType.BOOL:
            return 0, 1
        else:
            return self.min_value, self.max_value


# =============================================================================
# FIELDMASK ↔ PARAM_ID MAPPING
# =============================================================================

# FieldMask bit → param_id(s)
# For group fields (FAMILY_WEIGHTS, EVENT_STRENGTHS), maps to list of param_ids
FIELD_TO_PARAMS: Dict[int, Union[int, List[int]]] = {
    # Voice Domain
    FieldMask.GAIN_Q15: 0x0001,
    FieldMask.ACTIVITY_Q15: 0x0002,
    FieldMask.PAN_Q15: 0x0003,
    FieldMask.SEND_A_Q15: 0x0004,
    FieldMask.SEND_B_Q15: 0x0005,
    FieldMask.NOTE_U8: 0x0010,
    FieldMask.CENTS_I16: 0x0011,
    FieldMask.REGIME_ID: 0x0020,
    FieldMask.DEGRADATION: 0x0021,
    FieldMask.TREE_A_ID: 0x0030,
    FieldMask.TREE_B_ID: 0x0031,
    FieldMask.TREE_BLEND_Q15: 0x0032,
    FieldMask.FAMILY_WEIGHTS: [0x0040, 0x0041, 0x0042, 0x0043,
                                0x0044, 0x0045, 0x0046, 0x0047],
    FieldMask.EVENT_MASK: 0x0050,
    FieldMask.EVENT_STRENGTHS: [0x0051, 0x0052, 0x0053, 0x0054,
                                 0x0055, 0x0056, 0x0057, 0x0058],

    # Symbol Domain
    FieldMask.K_MODE: 0x0100,
    FieldMask.DRIVE_Q15: 0x0101,
    FieldMask.TONE_Q15: 0x0102,
    FieldMask.BIAS_Q15: 0x0103,
    FieldMask.PRESSURE_Q15: 0x0104,

    # Safety Domain
    FieldMask.ENERGY_CAP_Q15: 0x0200,
    FieldMask.GEAR_STRENGTH_Q15: 0x0201,
    FieldMask.TRACTION_Q15: 0x0202,
}

# Build reverse mapping: param_id → FieldMask bit
def _build_param_to_field() -> Dict[int, int]:
    """Build reverse mapping from param_id to FieldMask bit."""
    result: Dict[int, int] = {}
    for field_bit, param_ids in FIELD_TO_PARAMS.items():
        if isinstance(param_ids, list):
            for pid in param_ids:
                result[pid] = int(field_bit)
        else:
            result[param_ids] = int(field_bit)
    return result

PARAM_TO_FIELD: Dict[int, int] = _build_param_to_field()


# =============================================================================
# MASKED UPDATE
# =============================================================================

@dataclass
class MaskedUpdate:
    """
    A sparse parameter update with presence bitmap.

    MaskedUpdate is the transport format for partial updates:
    - field_mask indicates which fields are present
    - values contains the actual values for those fields
    - mode specifies ABS or DELTA semantics

    For group fields (FAMILY_WEIGHTS, EVENT_STRENGTHS), values should
    contain a list of 8 values.
    """
    field_mask: int = 0                           # Which fields are present
    values: Dict[int, Any] = field(default_factory=dict)  # FieldMask bit → value(s)
    mode: UpdateMode = UpdateMode.ABS             # How to apply

    def has(self, field: FieldMask) -> bool:
        """Check if field is present in this update."""
        return bool(self.field_mask & (1 << field))

    def get(self, fld: FieldMask) -> Optional[Any]:
        """Get value for field if present."""
        if self.has(fld):
            return self.values.get(int(fld))
        return None

    def set(self, fld: FieldMask, value: Any) -> None:
        """Set value for a field and update mask."""
        self.field_mask |= (1 << fld)
        self.values[int(fld)] = value

    def clear(self, fld: FieldMask) -> None:
        """Clear a field from the update."""
        self.field_mask &= ~(1 << fld)
        self.values.pop(int(fld), None)

    def fields(self) -> List[FieldMask]:
        """Return list of fields present in this update."""
        result = []
        for fm in FieldMask:
            if self.has(fm):
                result.append(fm)
        return result

    def is_empty(self) -> bool:
        """Check if no fields are set."""
        return self.field_mask == 0


# =============================================================================
# PARAMETER STATE
# =============================================================================

@dataclass
class ParameterState:
    """
    Current state of a parameter.

    Tracks initialization status for DELTA semantics:
    DELTA applied to uninitialized parameter is ignored until ABS arrives.
    """
    value: int
    initialized: bool = False


# =============================================================================
# PARAMETER REGISTRY
# =============================================================================

class ParameterRegistry:
    """
    Central registry for all system parameters.

    All parameter updates must resolve through this registry.
    """

    def __init__(self):
        self._definitions: Dict[int, ParameterDef] = {}
        self._by_name: Dict[str, int] = {}
        self._register_canonical_parameters()

    def _register_canonical_parameters(self) -> None:
        """Register all canonical parameters per Closure Appendix v0.1."""
        # =========================================================
        # VOICE DOMAIN - VoiceFramePacket fields
        # =========================================================

        # Gain/Activity/Pan/Send
        self.register(ParameterDef(
            param_id=0x0001, name="gain_q15",
            domain=ParamDomain.VOICE, dtype=ParamType.Q15,
            min_value=0, max_value=Q15_MAX,
            default_value=Q15_ONE,
            description="Voice amplitude gain"
        ))
        self.register(ParameterDef(
            param_id=0x0002, name="activity_q15",
            domain=ParamDomain.VOICE, dtype=ParamType.Q15,
            min_value=0, max_value=Q15_MAX,
            default_value=0,
            description="Voice activity level"
        ))
        self.register(ParameterDef(
            param_id=0x0003, name="pan_q15",
            domain=ParamDomain.VOICE, dtype=ParamType.Q15,
            min_value=Q15_MIN, max_value=Q15_MAX,
            default_value=0,
            description="Stereo pan position"
        ))
        self.register(ParameterDef(
            param_id=0x0004, name="sendA_q15",
            domain=ParamDomain.VOICE, dtype=ParamType.Q15,
            min_value=0, max_value=Q15_MAX,
            default_value=0,
            description="FX send A level"
        ))
        self.register(ParameterDef(
            param_id=0x0005, name="sendB_q15",
            domain=ParamDomain.VOICE, dtype=ParamType.Q15,
            min_value=0, max_value=Q15_MAX,
            default_value=0,
            description="FX send B level"
        ))

        # Note parameters
        self.register(ParameterDef(
            param_id=0x0010, name="note_u8",
            domain=ParamDomain.VOICE, dtype=ParamType.U8,
            min_value=0, max_value=127,
            default_value=60,
            description="MIDI note number"
        ))
        self.register(ParameterDef(
            param_id=0x0011, name="cents_i16",
            domain=ParamDomain.VOICE, dtype=ParamType.Q15,
            min_value=-9600, max_value=9600,
            default_value=0,
            description="Fine tuning in cents"
        ))

        # Regime and degrade
        self.register(ParameterDef(
            param_id=0x0020, name="regime_id",
            domain=ParamDomain.VOICE, dtype=ParamType.U8,
            min_value=0, max_value=15,
            default_value=2,  # REG_STABLE
            description="Current regime ID"
        ))
        self.register(ParameterDef(
            param_id=0x0021, name="degradation_level",
            domain=ParamDomain.VOICE, dtype=ParamType.U8,
            min_value=0, max_value=4,
            default_value=0,
            description="Degradation level D"
        ))

        # Tree control
        self.register(ParameterDef(
            param_id=0x0030, name="tree_a_id",
            domain=ParamDomain.VOICE, dtype=ParamType.U8,
            min_value=0, max_value=7,
            default_value=1,  # TREE_DEFAULT
            description="Tree A ID"
        ))
        self.register(ParameterDef(
            param_id=0x0031, name="tree_b_id",
            domain=ParamDomain.VOICE, dtype=ParamType.U8,
            min_value=0, max_value=7,
            default_value=1,  # TREE_DEFAULT
            description="Tree B ID"
        ))
        self.register(ParameterDef(
            param_id=0x0032, name="tree_blend_q15",
            domain=ParamDomain.VOICE, dtype=ParamType.Q15,
            min_value=0, max_value=Q15_MAX,
            default_value=0,
            description="Tree blend factor"
        ))

        # Family weights (8 families)
        for i in range(8):
            self.register(ParameterDef(
                param_id=0x0040 + i, name=f"w_family_{i}_q15",
                domain=ParamDomain.VOICE, dtype=ParamType.Q15,
                min_value=0, max_value=Q15_MAX,
                default_value=0,
                description=f"Family {i} weight"
            ))

        # Event mask and strengths
        self.register(ParameterDef(
            param_id=0x0050, name="event_mask",
            domain=ParamDomain.VOICE, dtype=ParamType.U8,
            min_value=0, max_value=255,
            default_value=0,
            description="Event lane enable mask"
        ))
        for i in range(8):
            self.register(ParameterDef(
                param_id=0x0051 + i, name=f"event_strength_{i}_q15",
                domain=ParamDomain.VOICE, dtype=ParamType.Q15,
                min_value=0, max_value=Q15_MAX,
                default_value=0,
                description=f"Event lane {i} strength"
            ))

        # =========================================================
        # SYMBOL DOMAIN - ParamFrame fields
        # =========================================================

        self.register(ParameterDef(
            param_id=0x0100, name="K_mode",
            domain=ParamDomain.SYMBOL, dtype=ParamType.U8,
            min_value=0, max_value=7,
            default_value=0,
            description="Kernel mode selector"
        ))
        self.register(ParameterDef(
            param_id=0x0101, name="drive_q15",
            domain=ParamDomain.SYMBOL, dtype=ParamType.Q15,
            min_value=0, max_value=Q15_MAX,
            default_value=Q15_MAX // 2,
            description="Drive amount"
        ))
        self.register(ParameterDef(
            param_id=0x0102, name="tone_q15",
            domain=ParamDomain.SYMBOL, dtype=ParamType.Q15,
            min_value=Q15_MIN, max_value=Q15_MAX,
            default_value=0,
            description="Tone control"
        ))
        self.register(ParameterDef(
            param_id=0x0103, name="bias_q15",
            domain=ParamDomain.SYMBOL, dtype=ParamType.Q15,
            min_value=Q15_MIN, max_value=Q15_MAX,
            default_value=0,
            description="Bias amount"
        ))
        self.register(ParameterDef(
            param_id=0x0104, name="pressure_q15",
            domain=ParamDomain.SYMBOL, dtype=ParamType.Q15,
            min_value=0, max_value=Q15_MAX,
            default_value=0,
            description="Pressure/aftertouch"
        ))

        # =========================================================
        # SAFETY DOMAIN - Constraints
        # =========================================================

        self.register(ParameterDef(
            param_id=0x0200, name="energy_cap_q15",
            domain=ParamDomain.SAFETY, dtype=ParamType.Q15,
            min_value=0, max_value=Q15_MAX,
            default_value=Q15_MAX,
            persistence=Persistence.GLOBAL,
            description="Maximum energy cap"
        ))
        self.register(ParameterDef(
            param_id=0x0201, name="gear_strength_q15",
            domain=ParamDomain.SAFETY, dtype=ParamType.Q15,
            min_value=0, max_value=Q15_MAX,
            default_value=Q15_ONE,
            persistence=Persistence.GLOBAL,
            description="Global gear strength"
        ))
        self.register(ParameterDef(
            param_id=0x0202, name="traction_q15",
            domain=ParamDomain.SAFETY, dtype=ParamType.Q15,
            min_value=0, max_value=Q15_MAX,
            default_value=Q15_MAX // 2,
            persistence=Persistence.GLOBAL,
            description="System traction"
        ))

    def register(self, param: ParameterDef) -> None:
        """Register a parameter definition."""
        if param.param_id in self._definitions:
            raise ValueError(f"Parameter ID {param.param_id} already registered")
        if param.name in self._by_name:
            raise ValueError(f"Parameter name '{param.name}' already registered")

        self._definitions[param.param_id] = param
        self._by_name[param.name] = param.param_id

    def get(self, param_id: int) -> Optional[ParameterDef]:
        """Get parameter definition by ID."""
        return self._definitions.get(param_id)

    def get_by_name(self, name: str) -> Optional[ParameterDef]:
        """Get parameter definition by name."""
        param_id = self._by_name.get(name)
        if param_id is not None:
            return self._definitions.get(param_id)
        return None

    def list_by_domain(self, domain: ParamDomain) -> List[ParameterDef]:
        """List all parameters in a domain."""
        return [p for p in self._definitions.values() if p.domain == domain]

    def list_all(self) -> List[ParameterDef]:
        """List all registered parameters."""
        return list(self._definitions.values())


# =============================================================================
# PARAMETER BANK (State Management)
# =============================================================================

class ParameterBank:
    """
    Manages parameter state for a single context (voice, global, or program).

    Implements ABS and DELTA update semantics:
    - ABS: v := clamp(value, min, max) - always initializes
    - DELTA: v := clamp(v + delta, min, max) - ignored if uninitialized
    """

    def __init__(self, registry: ParameterRegistry):
        self._registry = registry
        self._state: Dict[int, ParameterState] = {}
        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        """Initialize all parameters to their default values."""
        for param in self._registry.list_all():
            self._state[param.param_id] = ParameterState(
                value=param.default_value,
                initialized=True,
            )

    def get(self, param_id: int) -> Optional[int]:
        """Get current parameter value."""
        state = self._state.get(param_id)
        if state is not None and state.initialized:
            return state.value
        return None

    def get_by_name(self, name: str) -> Optional[int]:
        """Get current parameter value by name."""
        param = self._registry.get_by_name(name)
        if param:
            return self.get(param.param_id)
        return None

    def is_initialized(self, param_id: int) -> bool:
        """Check if parameter has been initialized."""
        state = self._state.get(param_id)
        return state is not None and state.initialized

    def apply_abs(self, param_id: int, value: int) -> int:
        """
        Apply absolute update to parameter.

        ABS update: v := clamp(value, min, max)
        Always initializes the parameter.

        Returns the clamped value that was applied.
        """
        param = self._registry.get(param_id)
        if param is None:
            raise ValueError(f"Unknown parameter ID: {param_id}")

        clamped = param.validate_value(value)

        if param_id not in self._state:
            self._state[param_id] = ParameterState(value=clamped, initialized=True)
        else:
            self._state[param_id].value = clamped
            self._state[param_id].initialized = True

        return clamped

    def apply_delta(self, param_id: int, delta: int) -> Optional[int]:
        """
        Apply delta update to parameter.

        DELTA update: v := clamp(v + delta, min, max)
        Ignored if parameter is uninitialized (returns None).

        Returns the new value, or None if ignored.
        """
        param = self._registry.get(param_id)
        if param is None:
            raise ValueError(f"Unknown parameter ID: {param_id}")

        state = self._state.get(param_id)
        if state is None or not state.initialized:
            # DELTA on uninitialized parameter is ignored
            return None

        new_value = param.validate_value(state.value + delta)
        state.value = new_value
        return new_value

    def reset_to_default(self, param_id: int) -> int:
        """Reset parameter to its default value."""
        param = self._registry.get(param_id)
        if param is None:
            raise ValueError(f"Unknown parameter ID: {param_id}")

        return self.apply_abs(param_id, param.default_value)

    def reset_all_defaults(self) -> None:
        """Reset all parameters to their default values."""
        self._initialize_defaults()

    def reset_domain(self, domain: ParamDomain) -> None:
        """Reset all parameters in a domain to defaults."""
        for param in self._registry.list_by_domain(domain):
            self.reset_to_default(param.param_id)

    def to_dict(self) -> Dict[str, int]:
        """Export current state as name -> value dict."""
        result = {}
        for param_id, state in self._state.items():
            if state.initialized:
                param = self._registry.get(param_id)
                if param:
                    result[param.name] = state.value
        return result

    # =========================================================================
    # MASKED UPDATE METHODS
    # =========================================================================

    def apply_masked(self, update: MaskedUpdate) -> Dict[int, int]:
        """
        Apply sparse update to parameter state.

        For each field in mask:
        1. Look up param_id(s) via FIELD_TO_PARAMS
        2. Apply value with ABS or DELTA per mode
        3. Return dict of param_id → final value applied

        Args:
            update: MaskedUpdate with field_mask, values, and mode

        Returns:
            Dict of param_id → final value that was applied
        """
        applied: Dict[int, int] = {}

        for fld in update.fields():
            value = update.get(fld)
            if value is None:
                continue

            param_ids = FIELD_TO_PARAMS.get(int(fld))
            if param_ids is None:
                continue

            # Handle group fields (list of param_ids)
            if isinstance(param_ids, list):
                # value should be a list of the same length
                if isinstance(value, list):
                    for i, pid in enumerate(param_ids):
                        if i < len(value):
                            final = self._apply_single(pid, value[i], update.mode)
                            if final is not None:
                                applied[pid] = final
                # Single value applied to all in group
                else:
                    for pid in param_ids:
                        final = self._apply_single(pid, value, update.mode)
                        if final is not None:
                            applied[pid] = final
            else:
                # Single param_id
                final = self._apply_single(param_ids, value, update.mode)
                if final is not None:
                    applied[param_ids] = final

        return applied

    def _apply_single(self, param_id: int, value: int, mode: UpdateMode) -> Optional[int]:
        """Apply a single value to a single param_id with given mode."""
        try:
            if mode == UpdateMode.ABS:
                return self.apply_abs(param_id, value)
            elif mode == UpdateMode.DELTA:
                return self.apply_delta(param_id, value)
            else:
                # BOTH mode defaults to ABS
                return self.apply_abs(param_id, value)
        except ValueError:
            # Unknown param_id
            return None

    def diff(self, other: 'ParameterBank') -> MaskedUpdate:
        """
        Compute sparse diff between two states.

        Returns MaskedUpdate with only changed fields.
        Fields present in self but different from other are included.

        Args:
            other: ParameterBank to compare against

        Returns:
            MaskedUpdate with changed fields and their values from self
        """
        update = MaskedUpdate(mode=UpdateMode.ABS)

        # Group param_ids that have changed, by FieldMask
        changed_by_field: Dict[int, List[Tuple[int, int]]] = {}  # field → [(param_id, value), ...]

        for param_id, state in self._state.items():
            if not state.initialized:
                continue

            other_value = other.get(param_id)
            if other_value != state.value:
                # Find which field this param belongs to
                field_bit = PARAM_TO_FIELD.get(param_id)
                if field_bit is not None:
                    if field_bit not in changed_by_field:
                        changed_by_field[field_bit] = []
                    changed_by_field[field_bit].append((param_id, state.value))

        # Build the MaskedUpdate
        for field_bit, changes in changed_by_field.items():
            param_ids = FIELD_TO_PARAMS.get(field_bit)

            if isinstance(param_ids, list):
                # Group field - extract values in order
                values = []
                for pid in param_ids:
                    val = self.get(pid)
                    values.append(val if val is not None else 0)
                update.set(FieldMask(field_bit), values)
            else:
                # Single field
                _, value = changes[0]
                update.set(FieldMask(field_bit), value)

        return update

    def to_masked_update(self, fields: Optional[List[FieldMask]] = None) -> MaskedUpdate:
        """
        Export current state as MaskedUpdate.

        Args:
            fields: Optional list of fields to include. If None, includes all
                    initialized params that have a FieldMask mapping.

        Returns:
            MaskedUpdate with current values
        """
        update = MaskedUpdate(mode=UpdateMode.ABS)

        if fields is None:
            # Include all fields that have initialized params
            for field_bit, param_ids in FIELD_TO_PARAMS.items():
                if isinstance(param_ids, list):
                    # Group field - check if any param is initialized
                    values = []
                    any_initialized = False
                    for pid in param_ids:
                        val = self.get(pid)
                        if val is not None:
                            any_initialized = True
                        values.append(val if val is not None else 0)
                    if any_initialized:
                        update.set(FieldMask(field_bit), values)
                else:
                    val = self.get(param_ids)
                    if val is not None:
                        update.set(FieldMask(field_bit), val)
        else:
            # Include only specified fields
            for fld in fields:
                param_ids = FIELD_TO_PARAMS.get(int(fld))
                if param_ids is None:
                    continue

                if isinstance(param_ids, list):
                    values = []
                    for pid in param_ids:
                        val = self.get(pid)
                        values.append(val if val is not None else 0)
                    update.set(fld, values)
                else:
                    val = self.get(param_ids)
                    if val is not None:
                        update.set(fld, val)

        return update


# =============================================================================
# ENGINE ROUTING
# =============================================================================

def route_by_domain(update: MaskedUpdate, registry: ParameterRegistry) -> Dict[ParamDomain, MaskedUpdate]:
    """
    Split a MaskedUpdate by engine domain.

    Useful for routing updates to the correct engine:
    - Voice domain → Voice Engine
    - Symbol domain → Symbol Engine
    - Safety domain → Safety Engine

    Args:
        update: MaskedUpdate to split
        registry: ParameterRegistry for domain lookup

    Returns:
        Dict mapping ParamDomain → MaskedUpdate (only domains with fields)
    """
    domain_updates: Dict[ParamDomain, MaskedUpdate] = {}

    for fld in update.fields():
        param_ids = FIELD_TO_PARAMS.get(int(fld))
        if param_ids is None:
            continue

        # Get domain from first param_id
        first_pid = param_ids[0] if isinstance(param_ids, list) else param_ids
        param_def = registry.get(first_pid)
        if param_def is None:
            continue

        domain = param_def.domain

        # Create or get domain update
        if domain not in domain_updates:
            domain_updates[domain] = MaskedUpdate(mode=update.mode)

        # Copy field to domain update
        value = update.get(fld)
        domain_updates[domain].set(fld, value)

    return domain_updates


# =============================================================================
# FIXED-POINT ARITHMETIC (Normative per Closure Appendix B)
# =============================================================================

def q15_mul(a: int, b: int) -> int:
    """
    Multiply two q15 values with proper rounding.

    Per Closure Appendix B.2:
    prod = int32(a) * int32(b)  // full precision
    temp = prod + 0x4000         // rounding bias
    out  = temp >> 15            // arithmetic right shift

    Returns clamped q15 result.
    """
    prod = a * b
    temp = prod + 0x4000  # Rounding bias for ties-to-nearest
    out = temp >> 15  # Arithmetic right shift

    # Clamp to q15 range
    return max(Q15_MIN, min(Q15_MAX, out))


def q15_add(a: int, b: int) -> int:
    """Add two q15 values with saturation."""
    result = a + b
    return max(Q15_MIN, min(Q15_MAX, result))


def q15_sub(a: int, b: int) -> int:
    """Subtract two q15 values with saturation."""
    result = a - b
    return max(Q15_MIN, min(Q15_MAX, result))


def q15_from_float(f: float) -> int:
    """Convert float [-1.0, 1.0) to q15."""
    return max(Q15_MIN, min(Q15_MAX, int(round(f * Q15_SCALE))))


def q15_to_float(q: int) -> float:
    """Convert q15 to float."""
    return q / Q15_SCALE


def q8_from_float(f: float) -> int:
    """Convert float [-1.0, 1.0) to q8."""
    return max(Q8_MIN, min(Q8_MAX, int(round(f * 128))))


def q8_to_float(q: int) -> float:
    """Convert q8 to float."""
    return q / 128.0


# =============================================================================
# CANONICAL DEFAULTS (per Closure Appendix C.2)
# =============================================================================

CANONICAL_DEFAULTS = {
    "regime_id": 2,          # REG_STABLE
    "gear_strength": Q15_ONE,  # 1.0
    "gain": Q15_ONE,          # 1.0
    "tree_id": 1,            # TREE_DEFAULT
    "tree_blend": 0,         # 0.0
    "traction": Q15_MAX // 2,  # 0.5
    "backbone": [],          # empty
    "event_mask": 0,
    "event_strength": 0,
    "event_payload": 0,
}


# =============================================================================
# WIRE PROTOCOL CONVERSION
# =============================================================================

# Wire protocol FieldMask (from uart_protocol.py) uses different bit assignments.
# This maps canonical FieldMask → wire FieldMask and vice versa.

# Wire protocol field bits (per WSE-STD-020 Section 7.2)
class WireFieldMask(IntEnum):
    """Wire protocol field mask bits (uart_protocol.py compatible)."""
    REGIME_ID = 0
    GEAR_Q15 = 1
    TREE_ID_A = 2
    TREE_ID_B = 3
    TREE_BLEND_Q15 = 4
    GAIN_Q15 = 5
    TRACTION_Q15 = 6
    BACKBONE = 7         # Not mapped to ParameterRegistry
    EVENT_MASK = 8
    EVENT_STRENGTH = 9
    PAYLOAD_MODE = 10    # Not mapped to ParameterRegistry
    PAYLOAD_COMMON = 11  # Not mapped to ParameterRegistry
    PER_LANE_PAYLOADS = 12  # Not mapped to ParameterRegistry


# Canonical FieldMask → Wire FieldMask
CANONICAL_TO_WIRE: Dict[int, int] = {
    FieldMask.REGIME_ID: WireFieldMask.REGIME_ID,
    FieldMask.GEAR_STRENGTH_Q15: WireFieldMask.GEAR_Q15,
    FieldMask.TREE_A_ID: WireFieldMask.TREE_ID_A,
    FieldMask.TREE_B_ID: WireFieldMask.TREE_ID_B,
    FieldMask.TREE_BLEND_Q15: WireFieldMask.TREE_BLEND_Q15,
    FieldMask.GAIN_Q15: WireFieldMask.GAIN_Q15,
    FieldMask.TRACTION_Q15: WireFieldMask.TRACTION_Q15,
    FieldMask.EVENT_MASK: WireFieldMask.EVENT_MASK,
    FieldMask.EVENT_STRENGTHS: WireFieldMask.EVENT_STRENGTH,
}

# Wire FieldMask → Canonical FieldMask
WIRE_TO_CANONICAL: Dict[int, int] = {v: k for k, v in CANONICAL_TO_WIRE.items()}


def to_wire_field_mask(canonical_mask: int) -> int:
    """
    Convert canonical FieldMask to wire protocol FieldMask.

    Only fields that have wire protocol mappings are included.
    Fields without mappings (e.g., ACTIVITY_Q15, PAN_Q15) are dropped.

    Args:
        canonical_mask: Bitmask of canonical FieldMask values

    Returns:
        Bitmask of WireFieldMask values
    """
    wire_mask = 0
    for canon_bit, wire_bit in CANONICAL_TO_WIRE.items():
        if canonical_mask & (1 << canon_bit):
            wire_mask |= (1 << wire_bit)
    return wire_mask


def from_wire_field_mask(wire_mask: int) -> int:
    """
    Convert wire protocol FieldMask to canonical FieldMask.

    Wire-only fields (BACKBONE, PAYLOAD_*) are dropped.

    Args:
        wire_mask: Bitmask of WireFieldMask values

    Returns:
        Bitmask of canonical FieldMask values
    """
    canonical_mask = 0
    for wire_bit, canon_bit in WIRE_TO_CANONICAL.items():
        if wire_mask & (1 << wire_bit):
            canonical_mask |= (1 << canon_bit)
    return canonical_mask


# =============================================================================
# SINGLETON REGISTRY
# =============================================================================

_global_registry: Optional[ParameterRegistry] = None


def get_registry() -> ParameterRegistry:
    """Get or create the global parameter registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ParameterRegistry()
    return _global_registry


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'Q15_MIN', 'Q15_MAX', 'Q15_SCALE', 'Q15_ONE',
    'Q8_MIN', 'Q8_MAX', 'U8_MIN', 'U8_MAX', 'U16_MIN', 'U16_MAX',
    # Enums
    'ParamDomain', 'ParamType', 'UpdateMode', 'Persistence',
    'FieldMask', 'WireFieldMask',
    # Classes
    'ParameterDef', 'ParameterState', 'ParameterRegistry', 'ParameterBank',
    'MaskedUpdate',
    # Mapping tables
    'FIELD_TO_PARAMS', 'PARAM_TO_FIELD',
    'CANONICAL_TO_WIRE', 'WIRE_TO_CANONICAL',
    # Fixed-point functions
    'q15_mul', 'q15_add', 'q15_sub', 'q15_from_float', 'q15_to_float',
    'q8_from_float', 'q8_to_float',
    # Routing/conversion functions
    'route_by_domain', 'to_wire_field_mask', 'from_wire_field_mask',
    # Defaults
    'CANONICAL_DEFAULTS',
    # Global registry
    'get_registry',
]
