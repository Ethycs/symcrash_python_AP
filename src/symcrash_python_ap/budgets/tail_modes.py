"""
Tail Mode Definitions (WSE-STD-CMMM)
====================================

Re-exports from tables/tail_modes.py (canonical source).

Provides additional budget-specific functions for tail mode management
during degradation scenarios.
"""

from typing import Tuple

# Re-export canonical definitions from tables/tail_modes.py
from ..tables.tail_modes import (
    # TailMode enum (from primitives)
    TailMode,
    # Operator constants
    OP_FINE_CLAMP, OP_NORMALIZE_SCALE, OP_SHEAF_PROJECT, OP_EMERGENCY_SIMPLIFY,
    # Core table and functions
    TAIL_OPS,
    get_tail_ops,
    can_use_tail_mode,
    downgrade_tail_mode,
    TAIL_MODE_DESCRIPTIONS,
)

# Import Operator for local use
from ..primitives import Operator


# =============================================================================
# BUDGET-SPECIFIC EXTENSIONS
# =============================================================================

# Which tail modes require nonlinear budget class (projection-class ops)
# Note: "nonlinear" here means budget category (expensive math/projection),
# not necessarily platform float. VM uses integer isqrt for determinism.
TAIL_USES_NONLINEAR = {
    TailMode.FULL: True,   # NormalizeScale (projection-class, extra math budget)
    TailMode.MID: True,    # NormalizeScale (projection-class, extra math budget)
    TailMode.MIN: False,
    TailMode.EMERGENCY: False,
}


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


class TailConfig:
    """Configuration for tail execution."""
    __slots__ = ('mode', 'ops', 'uses_nonlinear')

    def __init__(self, mode: TailMode, ops: Tuple[int, ...], uses_nonlinear: bool):
        self.mode = mode
        self.ops = ops
        self.uses_nonlinear = uses_nonlinear

    @classmethod
    def from_mode(cls, mode: TailMode) -> 'TailConfig':
        """Create TailConfig from mode."""
        return cls(
            mode=mode,
            ops=tuple(TAIL_OPS[mode]),
            uses_nonlinear=TAIL_USES_NONLINEAR[mode],
        )


__all__ = [
    # Re-exported from tables/tail_modes.py
    'TailMode',
    'OP_FINE_CLAMP', 'OP_NORMALIZE_SCALE', 'OP_SHEAF_PROJECT', 'OP_EMERGENCY_SIMPLIFY',
    'TAIL_OPS',
    'get_tail_ops',
    'can_use_tail_mode',
    'downgrade_tail_mode',
    'TAIL_MODE_DESCRIPTIONS',
    # Budget-specific extensions
    'TAIL_USES_NONLINEAR',
    'tail_uses_nonlinear',
    'get_allowed_tail_mode',
    'get_tail_mode_for_voice_class',
    'TailConfig',
]
