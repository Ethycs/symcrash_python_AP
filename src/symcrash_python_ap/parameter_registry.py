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


# =============================================================================
# CONSTANTS
# =============================================================================

Q15_MIN = -32768
Q15_MAX = 32767
Q15_SCALE = 32768
Q15_ONE = 32767

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
    # Classes
    'ParameterDef', 'ParameterState', 'ParameterRegistry', 'ParameterBank',
    # Fixed-point functions
    'q15_mul', 'q15_add', 'q15_sub', 'q15_from_float', 'q15_to_float',
    'q8_from_float', 'q8_to_float',
    # Defaults
    'CANONICAL_DEFAULTS',
    # Global registry
    'get_registry',
]
