"""
Control Registry
=================

Single source of truth for all control IDs, ranges, defaults, slew policies,
and binding targets.

Implements WSE-STD-CONTROLREGISTRY v0 (frozen).

Usage:
    from symcrash_python_ap.control_registry import ControlRegistry, ControlID

    registry = ControlRegistry()
    entry = registry.get_entry(ControlID.TIMBRE)
    default = registry.get_default(ControlID.ATTACK)
    clamped = registry.clamp(ControlID.SUSTAIN, 40000)
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List


# =============================================================================
# ENUMS
# =============================================================================

class ValueType(IntEnum):
    """Control value type (shared with VoiceEngineRegistry)."""
    Q15U = 0      # Q15 unsigned: 0..32767
    I16 = 1       # Signed int16: -32768..32767
    U16 = 2       # Unsigned int16: 0..65535
    BOOL = 3      # Boolean: 0 or 1
    ENUM8 = 4     # Enumeration: 0..255
    U32 = 5       # Unsigned int32: 0..4294967295 (Q16.16 ratios)
    Q15S = 6      # Q15 signed: -32768..32767 (bipolar modulation)


class SlewPolicy(IntEnum):
    """Slew filter policy."""
    INSTANT = 0    # No filtering
    NBLOCK = 1     # Linear ramp over N blocks
    ONE_POLE = 2   # Exponential smoothing


class ControlID(IntEnum):
    """Frozen v0 control IDs."""
    # Timbral controls (0-4)
    TIMBRE = 0
    TEXTURE = 1
    DYNAMICS = 2
    MORPH = 3
    CHAOS = 4

    # Pitch controls (5-6)
    DETUNE = 5
    PITCH_BEND = 6

    # Reserved (7-15)
    RESERVED_7 = 7
    RESERVED_8 = 8
    MASTER_GAIN = 9

    # ADSR controls (16-19)
    ATK_BLOCKS = 16
    DCY_BLOCKS = 17
    SUS_Q15 = 18
    REL_BLOCKS = 19

    # Reserved expansion (20+)
    RESERVED_20 = 20


# =============================================================================
# CONTROL ENTRY
# =============================================================================

@dataclass
class ControlEntry:
    """Single control definition."""
    control_id: int          # ControlID enum value
    name: str                # Human-readable name
    value_type: ValueType    # Q15U, I16, U16, BOOL, ENUM8
    min_value: int           # Inclusive minimum
    max_value: int           # Inclusive maximum
    default_value: int       # Initial value
    slew_policy: SlewPolicy  # INSTANT, NBLOCK, ONE_POLE
    slew_param: int          # N for NBLOCK, alpha_q15 for ONE_POLE
    binding_target: str      # What it affects


# =============================================================================
# FROZEN REGISTRY TABLE (v0)
# =============================================================================

CONTROL_REGISTRY_V0: List[ControlEntry] = [
    ControlEntry(0, "TIMBRE", ValueType.Q15U, 0, 32767, 32767,
                 SlewPolicy.ONE_POLE, 2048, "family_scale[COLOR]"),
    ControlEntry(1, "TEXTURE", ValueType.Q15U, 0, 32767, 32767,
                 SlewPolicy.ONE_POLE, 2048, "family_scale[LOCAL]"),
    ControlEntry(2, "DYNAMICS", ValueType.Q15U, 0, 32767, 32767,
                 SlewPolicy.ONE_POLE, 2048, "gain_scale_q15"),
    ControlEntry(3, "MORPH", ValueType.Q15U, 0, 32767, 0,
                 SlewPolicy.NBLOCK, 16, "tree_blend_q15"),
    ControlEntry(4, "CHAOS", ValueType.Q15U, 0, 32767, 0,
                 SlewPolicy.NBLOCK, 32, "family_scale[CATASTROPHE]"),
    ControlEntry(5, "DETUNE", ValueType.I16, -2400, 2400, 0,
                 SlewPolicy.INSTANT, 0, "pitch_offset_cents"),
    ControlEntry(6, "PITCH_BEND", ValueType.I16, -200, 200, 0,
                 SlewPolicy.INSTANT, 0, "pitch_offset_cents"),
    ControlEntry(9, "MASTER_GAIN", ValueType.Q15U, 0, 32767, 32767,
                 SlewPolicy.ONE_POLE, 4096, "global.master_gain_q15"),
    ControlEntry(16, "ATK_BLOCKS", ValueType.U16, 1, 500, 8,
                 SlewPolicy.NBLOCK, 4, "attack_blocks"),
    ControlEntry(17, "DCY_BLOCKS", ValueType.U16, 1, 1000, 32,
                 SlewPolicy.NBLOCK, 4, "decay_blocks"),
    ControlEntry(18, "SUS_Q15", ValueType.Q15U, 0, 32767, 24576,
                 SlewPolicy.NBLOCK, 4, "sustain_level_q15"),
    ControlEntry(19, "REL_BLOCKS", ValueType.U16, 1, 1500, 64,
                 SlewPolicy.NBLOCK, 4, "release_blocks"),
]


# =============================================================================
# CONTROL REGISTRY
# =============================================================================

class ControlRegistry:
    """Single source of truth for control definitions."""

    def __init__(self, entries: List[ControlEntry] = None):
        self._entries: Dict[int, ControlEntry] = {}
        entries = entries or CONTROL_REGISTRY_V0
        for entry in entries:
            self._entries[entry.control_id] = entry

    def get_entry(self, control_id: int) -> ControlEntry:
        """Get control entry by ID. Raises KeyError if not found."""
        return self._entries[control_id]

    def get_default(self, control_id: int) -> int:
        """Get default value for control."""
        return self._entries[control_id].default_value

    def clamp(self, control_id: int, value: int) -> int:
        """Clamp value to legal range for control."""
        entry = self._entries[control_id]
        return max(entry.min_value, min(entry.max_value, value))

    def all_entries(self) -> List[ControlEntry]:
        """Get all registry entries."""
        return list(self._entries.values())

    def has_control(self, control_id: int) -> bool:
        """Check if control ID is defined."""
        return control_id in self._entries


# =============================================================================
# SLEW COMPUTATION (Integer-Only)
# =============================================================================

def slew_nblock(current: int, target: int, n: int) -> int:
    """
    Linear ramp over n blocks.

    Guaranteed to reach target in exactly n blocks (worst case).

    Args:
        current: Current value
        target: Target value
        n: Number of blocks to ramp over

    Returns:
        Next value
    """
    if current == target:
        return current

    delta = target - current
    step = (abs(delta) + n - 1) // n  # ceiling division

    if delta > 0:
        return min(current + step, target)
    else:
        return max(current - step, target)


def slew_one_pole(current: int, target: int, alpha_q15: int, min_val: int, max_val: int) -> int:
    """
    Exponential smoothing (integer-only).

    Time constant: Approximately (32767 / alpha_q15) blocks to reach ~63% of target.

    Args:
        current: Current value
        target: Target value
        alpha_q15: Smoothing coefficient (Q15)
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Next value
    """
    delta = target - current
    increment = (alpha_q15 * delta) >> 15

    # Ensure progress even for small deltas
    if increment == 0 and delta != 0:
        increment = 1 if delta > 0 else -1

    return max(min_val, min(max_val, current + increment))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ValueType',
    'SlewPolicy',
    'ControlID',
    'ControlEntry',
    'ControlRegistry',
    'CONTROL_REGISTRY_V0',
    'slew_nblock',
    'slew_one_pole',
]
