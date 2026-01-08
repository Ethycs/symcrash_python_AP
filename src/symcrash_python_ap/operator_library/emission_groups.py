"""
Emission Group Registry
=======================

Maps operators to emission groups per Scheduler Spec v01 Section 7.

Emission groups control when operators run based on scheduler decisions:
- TAIL: Phase C, always runs for serviced voices
- BASE: L0-L1, minimal injections for continuity
- MID: L2-L3, BANDED + Diffuse, LOWRANK, richer events
- TOP: L4-L5, chaos overlays, catastrophe corridor, dense patterning

Per-voice emission flags (mid_due, top_due) from the scheduler determine
which operator groups execute each block.
"""

from enum import IntEnum
from typing import Dict


class EmissionGroup(IntEnum):
    """Tower level emission groups."""
    TAIL = 0   # Phase C: Always runs for serviced voices
    BASE = 1   # L0-L1: Minimal injections for continuity
    MID = 2    # L2-L3: BANDED + Diffuse, LOWRANK, richer events
    TOP = 3    # L4-L5: Chaos overlays, catastrophe corridor, dense patterning


# Operator ID to Emission Group mapping
# Based on OP_NAME_TO_ID from operators.py and Scheduler Spec v01 Section 7
OP_EMISSION_GROUP: Dict[int, EmissionGroup] = {
    # ==========================================================================
    # TAIL (Phase C) - Always run for serviced voices
    # Mandatory tail set: FineClamp(2) -> NormalizeScale(11) -> GluePair(12)
    #                     -> SheafProject(23) -> EmergencySimplify(19)
    # ==========================================================================
    2: EmissionGroup.TAIL,    # FINE_CLAMP - Output clamping
    11: EmissionGroup.TAIL,   # NORMALIZE_SCALE - Energy normalization
    12: EmissionGroup.TAIL,   # GLUE_PAIR - Coefficient pairing (optional)
    19: EmissionGroup.TAIL,   # EMERGENCY_SIMPLIFY - Stability fallback
    23: EmissionGroup.TAIL,   # SHEAF_PROJECT - Final projection

    # ==========================================================================
    # BASE (L0-L1) - Minimal injections, always allowed
    # Core operators that maintain basic signal continuity
    # ==========================================================================
    0: EmissionGroup.BASE,    # IDENTITY - Pass-through
    1: EmissionGroup.BASE,    # TILT - Spectral tilt adjustment
    3: EmissionGroup.BASE,    # INTRA_MIX - Within-band mixing

    # ==========================================================================
    # MID (L2-L3) - Budgeted, gated by mid_due
    # BANDED operators, diffuse, LOWRANK, richer events
    # ==========================================================================
    4: EmissionGroup.MID,     # DIFFUSE_UP - Upward diffusion
    5: EmissionGroup.MID,     # DIFFUSE_DOWN - Downward diffusion
    7: EmissionGroup.MID,     # LOWRANK_MORPH - Low-rank morphing
    8: EmissionGroup.MID,     # GLOBAL_PULSE - Global pulse injection
    13: EmissionGroup.MID,    # IMPULSE - Impulse injection
    14: EmissionGroup.MID,    # NOISE_BAND - Band-limited noise

    # ==========================================================================
    # TOP (L4-L5) - Budgeted, gated by top_due
    # Chaos overlays, catastrophe corridor features, dense patterning
    # Most expensive operators, throttled under load
    # ==========================================================================
    6: EmissionGroup.TOP,     # BAND_SHEAR - Inter-band shearing
    9: EmissionGroup.TOP,     # SWAP_BANDS - Band swapping
    10: EmissionGroup.TOP,    # ROTATE_NODES - Node rotation
    15: EmissionGroup.TOP,    # PRESSURE_RAMP - Pressure modulation
    16: EmissionGroup.TOP,    # GATE_MASK - Coefficient gating
    17: EmissionGroup.TOP,    # QUASI_ROTATE - Quasi-periodic rotation
    18: EmissionGroup.TOP,    # MORPH_MIX - Morphing mixture
    20: EmissionGroup.TOP,    # FOLD_PRESSURE - Catastrophe fold
    21: EmissionGroup.TOP,    # CUSP_INJECTION - Cusp catastrophe injection
    22: EmissionGroup.TOP,    # SWALLOWTAIL_MASK - Swallowtail pattern mask
}


def get_emission_group(op_id: int) -> EmissionGroup:
    """
    Get the emission group for an operator.

    Args:
        op_id: Operator ID (0-23)

    Returns:
        EmissionGroup for the operator. Defaults to BASE for unknown IDs.
    """
    return OP_EMISSION_GROUP.get(op_id, EmissionGroup.BASE)


def is_op_allowed(op_id: int, mid_due: bool, top_due: bool) -> bool:
    """
    Check if an operator should run given emission flags.

    Args:
        op_id: Operator ID
        mid_due: True if MID group is due this block
        top_due: True if TOP group is due this block

    Returns:
        True if the operator should execute
    """
    group = get_emission_group(op_id)

    # TAIL and BASE always run
    if group == EmissionGroup.TAIL or group == EmissionGroup.BASE:
        return True

    # MID only runs when due
    if group == EmissionGroup.MID:
        return mid_due

    # TOP only runs when due
    if group == EmissionGroup.TOP:
        return top_due

    # Unknown defaults to allowed
    return True


def filter_ops_by_emission(
    op_ids: list,
    mid_due: bool,
    top_due: bool,
) -> list:
    """
    Filter a list of operator IDs by emission flags.

    Args:
        op_ids: List of operator IDs
        mid_due: True if MID group is due
        top_due: True if TOP group is due

    Returns:
        Filtered list containing only allowed operators
    """
    return [op_id for op_id in op_ids if is_op_allowed(op_id, mid_due, top_due)]


def get_ops_by_group(group: EmissionGroup) -> list:
    """
    Get all operator IDs belonging to an emission group.

    Args:
        group: EmissionGroup to query

    Returns:
        List of operator IDs in that group
    """
    return [op_id for op_id, g in OP_EMISSION_GROUP.items() if g == group]


__all__ = [
    'EmissionGroup',
    'OP_EMISSION_GROUP',
    'get_emission_group',
    'is_op_allowed',
    'filter_ops_by_emission',
    'get_ops_by_group',
]
