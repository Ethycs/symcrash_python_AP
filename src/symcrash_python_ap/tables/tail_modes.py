"""
Tail Modes (Canonical Table v0)
===============================

Tail operators are the mandatory safety operators that must run
to guarantee well-formed output. Different tail modes provide
different quality/cost tradeoffs.

Per WSE-STD-070 §10 (and InstrumentIR v0 tail table):
- FULL: High quality [FineClamp, NormalizeScale, SheafProject]
- MID:  Medium quality [FineClamp, NormalizeScale]
- MIN:  Minimum safety [FineClamp]
- EMERGENCY: [FineClamp, EmergencySimplify] - clamping is the bedrock
  guarantee and runs in every mode, including worst-case degradation.
  (An older CMMM-2 draft had carrier-only EMERGENCY; the budget tables
  and primitives.TAIL_MODE_TABLE always reserved for both ops.)
"""

from typing import List, Dict

# Import canonical TailMode from primitives (single source of truth)
from ..primitives import TailMode, Operator

# Operator IDs for tail operators (from primitives)
OP_FINE_CLAMP = Operator.FINE_CLAMP
OP_NORMALIZE_SCALE = Operator.NORMALIZE_SCALE
OP_SHEAF_PROJECT = Operator.SHEAF_PROJECT
OP_EMERGENCY_SIMPLIFY = Operator.EMERGENCY_SIMPLIFY


# Canonical tail operator sets per mode
TAIL_OPS: Dict[TailMode, List[int]] = {
    TailMode.FULL: [OP_FINE_CLAMP, OP_NORMALIZE_SCALE, OP_SHEAF_PROJECT],
    TailMode.MID: [OP_FINE_CLAMP, OP_NORMALIZE_SCALE],
    TailMode.MIN: [OP_FINE_CLAMP],
    TailMode.EMERGENCY: [OP_FINE_CLAMP, OP_EMERGENCY_SIMPLIFY],
}


def get_tail_ops(mode: TailMode) -> List[int]:
    """Get the tail operator list for a given mode."""
    return TAIL_OPS[mode].copy()


def can_use_tail_mode(mode: TailMode, allow_nonlinear: bool) -> bool:
    """
    Check if a tail mode is allowed given nonlinear constraints.

    NormalizeScale (op 11) involves sqrt-ish operations (nonlinear).
    If allow_nonlinear=False, only MIN and EMERGENCY are valid.
    """
    if allow_nonlinear:
        return True
    # Op 11 (NormalizeScale) is in FULL and MID
    return mode in (TailMode.MIN, TailMode.EMERGENCY)


def downgrade_tail_mode(mode: TailMode, allow_nonlinear: bool = True) -> TailMode:
    """
    Return the next lower tail mode in the degradation ladder.

    Returns same mode if already at EMERGENCY.
    """
    if mode == TailMode.FULL:
        return TailMode.MID if allow_nonlinear else TailMode.MIN
    elif mode == TailMode.MID:
        return TailMode.MIN
    elif mode == TailMode.MIN:
        return TailMode.EMERGENCY
    else:
        return TailMode.EMERGENCY


# =============================================================================
# TAIL MODE DESCRIPTIONS (for documentation/UI)
# =============================================================================

TAIL_MODE_DESCRIPTIONS = {
    TailMode.FULL: "Full quality: FineClamp + NormalizeScale + SheafProject",
    TailMode.MID: "Medium quality: FineClamp + NormalizeScale",
    TailMode.MIN: "Minimum safety: FineClamp only",
    TailMode.EMERGENCY: "Emergency fallback: FineClamp + EmergencySimplify",
}
