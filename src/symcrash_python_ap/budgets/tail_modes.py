"""
Tail Mode Definitions (WSE-STD-CMMM)
====================================

Defines tail operator sets for different quality/budget levels.
Tail operators guarantee "instrument-safe" output.

Tail modes:
    FULL: Maximum quality - [FineClamp, NormalizeScale, SheafProject]
    MID:  Good quality    - [FineClamp, NormalizeScale]
    MIN:  Basic safety    - [FineClamp]
    EMERGENCY: Fallback   - [EmergencySimplify]

Rules:
    - MAIN voices: Start at FULL, can degrade to MID/MIN
    - RESIDUAL voices: Start at MIN or EMERGENCY
    - If nonlinear not allowed, skip NormalizeScale (FULL→MIN or MID→MIN)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import IntEnum


class TailMode(IntEnum):
    """Tail mode levels in degradation order."""
    FULL = 0        # Full quality: [2, 11, 23]
    MID = 1         # Good quality: [2, 11]
    MIN = 2         # Basic safety: [2]
    EMERGENCY = 3   # Fallback: [19]


# Operator IDs for tail operators
OP_FINE_CLAMP = 2
OP_NORMALIZE_SCALE = 11
OP_SHEAF_PROJECT = 23
OP_EMERGENCY_SIMPLIFY = 19


# =============================================================================
# TAIL OPERATOR SETS
# =============================================================================

TAIL_OPS: Dict[TailMode, List[int]] = {
    TailMode.FULL: [OP_FINE_CLAMP, OP_NORMALIZE_SCALE, OP_SHEAF_PROJECT],
    TailMode.MID: [OP_FINE_CLAMP, OP_NORMALIZE_SCALE],
    TailMode.MIN: [OP_FINE_CLAMP],
    TailMode.EMERGENCY: [OP_EMERGENCY_SIMPLIFY],
}

# Which tail modes contain nonlinear operators
TAIL_USES_NONLINEAR: Dict[TailMode, bool] = {
    TailMode.FULL: True,   # NormalizeScale uses sqrt
    TailMode.MID: True,    # NormalizeScale uses sqrt
    TailMode.MIN: False,
    TailMode.EMERGENCY: False,
}


# =============================================================================
# TAIL MODE FUNCTIONS
# =============================================================================

def get_tail_ops(mode: TailMode) -> List[int]:
    """Get operator IDs for a tail mode."""
    return TAIL_OPS[mode].copy()


def tail_uses_nonlinear(mode: TailMode) -> bool:
    """Check if tail mode requires nonlinear operations."""
    return TAIL_USES_NONLINEAR[mode]


def get_allowed_tail_mode(requested: TailMode, allow_nonlinear: bool) -> TailMode:
    """
    Get the best allowed tail mode given constraints.

    If nonlinear is disallowed, downgrades FULL/MID to MIN.

    Args:
        requested: Desired tail mode
        allow_nonlinear: Whether nonlinear ops are allowed

    Returns:
        Best allowed tail mode
    """
    if allow_nonlinear:
        return requested

    # Nonlinear not allowed - can't use modes with NormalizeScale
    if requested in (TailMode.FULL, TailMode.MID):
        return TailMode.MIN

    return requested


def downgrade_tail_mode(current: TailMode) -> TailMode:
    """
    Get the next lower tail mode in degradation order.

    Args:
        current: Current tail mode

    Returns:
        Next lower mode (or EMERGENCY if already at MIN)
    """
    if current == TailMode.FULL:
        return TailMode.MID
    elif current == TailMode.MID:
        return TailMode.MIN
    elif current == TailMode.MIN:
        return TailMode.EMERGENCY
    else:
        return TailMode.EMERGENCY  # Already at bottom


def get_tail_mode_for_voice_class(is_residual: bool) -> TailMode:
    """
    Get default tail mode for a voice class.

    Args:
        is_residual: True for RESIDUAL/cloud voices

    Returns:
        Default tail mode (FULL for MAIN, MIN for RESIDUAL)
    """
    if is_residual:
        return TailMode.MIN
    return TailMode.FULL


@dataclass(frozen=True)
class TailConfig:
    """Configuration for tail execution."""
    mode: TailMode
    ops: Tuple[int, ...]
    uses_nonlinear: bool

    @classmethod
    def from_mode(cls, mode: TailMode) -> 'TailConfig':
        """Create TailConfig from mode."""
        return cls(
            mode=mode,
            ops=tuple(TAIL_OPS[mode]),
            uses_nonlinear=TAIL_USES_NONLINEAR[mode],
        )


__all__ = [
    'TailMode',
    'OP_FINE_CLAMP', 'OP_NORMALIZE_SCALE', 'OP_SHEAF_PROJECT', 'OP_EMERGENCY_SIMPLIFY',
    'TAIL_OPS', 'TAIL_USES_NONLINEAR',
    'get_tail_ops', 'tail_uses_nonlinear',
    'get_allowed_tail_mode', 'downgrade_tail_mode',
    'get_tail_mode_for_voice_class', 'TailConfig',
]
