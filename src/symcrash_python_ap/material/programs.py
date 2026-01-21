"""
Material Engine Programs (0-5)
==============================

Complete material processing chains built from P1-P7 building blocks.

Programs:
- 0: Clean Glue (transparent with gentle glue compression)
- 1: Tape Bus (tape hysteresis saturation)
- 2: Rank Grit (median/trimmed mean nonlinearity)
- 3: Tropical Cut Mix (hard threshold/gate operations)
- 4: Catastrophe Snap (fold bifurcation)
- 5: Catastrophe Swallowtail (higher-order bifurcation with tape)

Each program provides:
- State class for per-channel processing state
- process_sample() for single-sample processing
- process_block() for block processing
"""

from dataclasses import dataclass
from typing import List, Tuple

from ..fixed import (
    sat16, mul_q15_sat,
    Q15_ONE, Q15_HALF,
)
from .building_blocks import (
    # P1
    TiltState, tilt_sample,
    # P2
    SaturatorParams, saturator_sample,
    # P3
    TapeHysteresisState, tape_hysteresis_sample,
    # P4
    OnePoleState, one_pole_lp_sample, freq_to_one_pole_coeff,
    # P5
    RankState, rank_trimmed_mean_sample,
    # P6
    tropical_gate_sample,
    # P7
    CatastropheState, catastrophe_fold_sample, catastrophe_swallowtail_sample,
)


# =============================================================================
# PROGRAM 0: CLEAN GLUE
# =============================================================================


@dataclass
class CleanGlueState:
    """
    State for Clean Glue program.

    Signal chain: Input -> Tilt -> Soft Saturator -> LP Filter -> Output

    Provides transparent processing with gentle "glue" compression that
    holds the mix together without obvious coloration.
    """
    tilt: TiltState = None
    post_lp: OnePoleState = None

    # Cached coefficient
    lp_coeff: int = 0

    def __post_init__(self):
        if self.tilt is None:
            self.tilt = TiltState()
        if self.post_lp is None:
            self.post_lp = OnePoleState()
        # Default post-filter at 12kHz
        self.lp_coeff = freq_to_one_pole_coeff(12000.0)


def clean_glue_sample(
    x: int,
    state: CleanGlueState,
    tilt_q15: int = 0,
    drive_q15: int = Q15_HALF,
    tone_q15: int = Q15_ONE
) -> int:
    """
    Single-sample Clean Glue processing.

    Args:
        x: Input sample (Q15)
        state: CleanGlueState
        tilt_q15: Pre-emphasis tilt (-32768 to 32767)
        drive_q15: Saturator drive (Q15)
        tone_q15: Post-filter brightness (0=dark, Q15_ONE=bright)

    Returns:
        Processed output (Q15)
    """
    # Step 1: Tilt EQ (subtle pre-emphasis)
    y = tilt_sample(x, state.tilt, tilt_q15 >> 2)  # Gentle tilt

    # Step 2: Soft saturator (very gentle "glue")
    # Use low drive for transparency
    sat_params = SaturatorParams(
        drive_q15=min(drive_q15, Q15_HALF + 8000),  # Cap drive for clean
        curve=1  # Soft clip - most transparent
    )
    y = saturator_sample(y, sat_params)

    # Step 3: Post-filter LP (tone control)
    # tone_q15 controls cutoff: low = dark, high = bright
    # Map tone to coefficient
    if tone_q15 < Q15_HALF:
        # Darker: use lower cutoff coefficient
        coeff = state.lp_coeff >> 1
    else:
        # Brighter: use higher coefficient (closer to bypass)
        coeff = state.lp_coeff + ((Q15_ONE - state.lp_coeff) >> 1)

    y = one_pole_lp_sample(y, state.post_lp, coeff)

    return y


def clean_glue_block(
    L: List[int],
    R: List[int],
    state_L: CleanGlueState,
    state_R: CleanGlueState,
    tilt_q15: int = 0,
    drive_q15: int = Q15_HALF,
    tone_q15: int = Q15_ONE
) -> Tuple[List[int], List[int]]:
    """
    Block-rate Clean Glue processing.

    Args:
        L, R: Input stereo block (Q15)
        state_L, state_R: Per-channel state
        tilt_q15: Pre-emphasis tilt
        drive_q15: Saturator drive
        tone_q15: Post-filter brightness

    Returns:
        Processed stereo block (Q15)
    """
    outL = []
    outR = []

    for i in range(len(L)):
        yL = clean_glue_sample(L[i], state_L, tilt_q15, drive_q15, tone_q15)
        yR = clean_glue_sample(R[i], state_R, tilt_q15, drive_q15, tone_q15)
        outL.append(yL)
        outR.append(yR)

    return outL, outR


# =============================================================================
# PROGRAM 1: TAPE BUS
# =============================================================================


@dataclass
class TapeBusState:
    """
    State for Tape Bus program.

    Signal chain: Input -> Tilt -> Tape Hysteresis -> LP Filter -> Output

    Models analog tape saturation with hysteresis, providing warm
    compression and subtle harmonic enhancement.
    """
    tilt: TiltState = None
    tape: TapeHysteresisState = None
    post_lp: OnePoleState = None

    # Cached coefficient
    lp_coeff: int = 0

    def __post_init__(self):
        if self.tilt is None:
            self.tilt = TiltState()
        if self.tape is None:
            self.tape = TapeHysteresisState()
        if self.post_lp is None:
            self.post_lp = OnePoleState()
        # Default post-filter at 10kHz (tape roll-off)
        self.lp_coeff = freq_to_one_pole_coeff(10000.0)


def tape_bus_sample(
    x: int,
    state: TapeBusState,
    tilt_q15: int = 0,
    drive_q15: int = Q15_HALF,
    bias_q15: int = 0,
    saturation_q15: int = Q15_HALF,
    tone_q15: int = Q15_ONE
) -> int:
    """
    Single-sample Tape Bus processing.

    Args:
        x: Input sample (Q15)
        state: TapeBusState
        tilt_q15: Pre-emphasis tilt (positive = brighter before saturation)
        drive_q15: Tape drive/input level (Q15)
        bias_q15: Tape bias (asymmetry) (Q15)
        saturation_q15: Saturation amount (Q15)
        tone_q15: Post-filter brightness (0=dark, Q15_ONE=bright)

    Returns:
        Processed output (Q15)
    """
    # Step 1: Pre-emphasis tilt
    # Positive tilt brightens input (classic tape "high boost before record")
    y = tilt_sample(x, state.tilt, tilt_q15)

    # Step 2: Tape hysteresis
    y = tape_hysteresis_sample(
        y,
        state.tape,
        drive_q15=drive_q15,
        bias_q15=bias_q15,
        saturation_q15=saturation_q15
    )

    # Step 3: Post-filter LP (tape head response roll-off)
    # Map tone to cutoff
    if tone_q15 < Q15_HALF:
        # Darker: more roll-off (lower coeff = more filtering)
        coeff = mul_q15_sat(state.lp_coeff, tone_q15 << 1)
    else:
        # Brighter: less roll-off
        excess = tone_q15 - Q15_HALF
        coeff = state.lp_coeff + mul_q15_sat(Q15_ONE - state.lp_coeff, excess << 1)

    y = one_pole_lp_sample(y, state.post_lp, coeff)

    return y


def tape_bus_block(
    L: List[int],
    R: List[int],
    state_L: TapeBusState,
    state_R: TapeBusState,
    tilt_q15: int = 0,
    drive_q15: int = Q15_HALF,
    bias_q15: int = 0,
    saturation_q15: int = Q15_HALF,
    tone_q15: int = Q15_ONE
) -> Tuple[List[int], List[int]]:
    """
    Block-rate Tape Bus processing.

    Args:
        L, R: Input stereo block (Q15)
        state_L, state_R: Per-channel state
        tilt_q15: Pre-emphasis tilt
        drive_q15: Tape drive
        bias_q15: Tape bias
        saturation_q15: Saturation amount
        tone_q15: Post-filter brightness

    Returns:
        Processed stereo block (Q15)
    """
    outL = []
    outR = []

    for i in range(len(L)):
        yL = tape_bus_sample(
            L[i], state_L, tilt_q15, drive_q15, bias_q15, saturation_q15, tone_q15
        )
        yR = tape_bus_sample(
            R[i], state_R, tilt_q15, drive_q15, bias_q15, saturation_q15, tone_q15
        )
        outL.append(yL)
        outR.append(yR)

    return outL, outR


# =============================================================================
# PROGRAM 2: RANK GRIT
# =============================================================================


@dataclass
class RankGritState:
    """
    State for Rank Grit program.

    Signal chain: Input -> Saturator -> Rank Filter -> LP -> Output

    Uses rank statistics (median/trimmed mean) for unique "gritty" texture
    that differs from conventional distortion.
    """
    rank: RankState = None
    post_lp: OnePoleState = None
    lp_coeff: int = 0

    def __post_init__(self):
        if self.rank is None:
            self.rank = RankState(window_size=9)
        if self.post_lp is None:
            self.post_lp = OnePoleState()
        self.lp_coeff = freq_to_one_pole_coeff(8000.0)


def rank_grit_sample(
    x: int,
    state: RankGritState,
    drive_q15: int = Q15_HALF,
    mix_q15: int = Q15_HALF
) -> int:
    """
    Single-sample Rank Grit processing.

    Args:
        x: Input sample (Q15)
        state: RankGritState
        drive_q15: Saturation drive before rank filter (Q15)
        mix_q15: Wet/dry mix (0=dry, Q15_ONE=wet)

    Returns:
        Processed output (Q15)
    """
    # Step 1: Pre-saturate
    sat_params = SaturatorParams(drive_q15=drive_q15, curve=0)
    y = saturator_sample(x, sat_params)

    # Step 2: Rank filter (trimmed mean)
    y_rank = rank_trimmed_mean_sample(y, state.rank, trim=2)

    # Step 3: Mix with original
    dry = mul_q15_sat(x, Q15_ONE - mix_q15)
    wet = mul_q15_sat(y_rank, mix_q15)
    y = sat16(dry + wet)

    # Step 4: Post LP
    y = one_pole_lp_sample(y, state.post_lp, state.lp_coeff)

    return y


def rank_grit_block(
    L: List[int],
    R: List[int],
    state_L: RankGritState,
    state_R: RankGritState,
    drive_q15: int = Q15_HALF,
    mix_q15: int = Q15_HALF
) -> Tuple[List[int], List[int]]:
    """Block-rate Rank Grit processing."""
    outL = []
    outR = []
    for i in range(len(L)):
        outL.append(rank_grit_sample(L[i], state_L, drive_q15, mix_q15))
        outR.append(rank_grit_sample(R[i], state_R, drive_q15, mix_q15))
    return outL, outR


# =============================================================================
# PROGRAM 3: TROPICAL CUT MIX
# =============================================================================


@dataclass
class TropicalCutState:
    """
    State for Tropical Cut Mix program.

    Signal chain: Input -> Tropical Gate -> Saturator -> LP -> Output

    Uses hard threshold operations for aggressive gating and shaping.
    """
    post_lp: OnePoleState = None
    lp_coeff: int = 0

    def __post_init__(self):
        if self.post_lp is None:
            self.post_lp = OnePoleState()
        self.lp_coeff = freq_to_one_pole_coeff(6000.0)


def tropical_cut_sample(
    x: int,
    state: TropicalCutState,
    threshold_q15: int = 8000,
    gate_mode: int = 0,
    drive_q15: int = Q15_HALF
) -> int:
    """
    Single-sample Tropical Cut processing.

    Args:
        x: Input sample (Q15)
        state: TropicalCutState
        threshold_q15: Gate threshold (Q15)
        gate_mode: 0=hard gate, 1=hard limit, 2=dead zone
        drive_q15: Post-gate saturation drive

    Returns:
        Processed output (Q15)
    """
    # Step 1: Tropical gate
    y = tropical_gate_sample(x, threshold_q15, gate_mode)

    # Step 2: Soft saturate
    sat_params = SaturatorParams(drive_q15=drive_q15, curve=1)
    y = saturator_sample(y, sat_params)

    # Step 3: Post LP
    y = one_pole_lp_sample(y, state.post_lp, state.lp_coeff)

    return y


def tropical_cut_block(
    L: List[int],
    R: List[int],
    state_L: TropicalCutState,
    state_R: TropicalCutState,
    threshold_q15: int = 8000,
    gate_mode: int = 0,
    drive_q15: int = Q15_HALF
) -> Tuple[List[int], List[int]]:
    """Block-rate Tropical Cut processing."""
    outL = []
    outR = []
    for i in range(len(L)):
        outL.append(tropical_cut_sample(L[i], state_L, threshold_q15, gate_mode, drive_q15))
        outR.append(tropical_cut_sample(R[i], state_R, threshold_q15, gate_mode, drive_q15))
    return outL, outR


# =============================================================================
# PROGRAM 4: CATASTROPHE SNAP
# =============================================================================


@dataclass
class CatastropheSnapState:
    """
    State for Catastrophe Snap program.

    Signal chain: Input -> Fold Catastrophe -> LP -> Output

    Uses fold catastrophe dynamics for sudden "snap" transitions
    that create unique attack transients.
    """
    cat: CatastropheState = None
    post_lp: OnePoleState = None
    lp_coeff: int = 0

    def __post_init__(self):
        if self.cat is None:
            self.cat = CatastropheState()
        if self.post_lp is None:
            self.post_lp = OnePoleState()
        self.lp_coeff = freq_to_one_pole_coeff(8000.0)


def catastrophe_snap_sample(
    x: int,
    state: CatastropheSnapState,
    pressure_q15: int = 10000,
    mix_q15: int = Q15_HALF
) -> int:
    """
    Single-sample Catastrophe Snap processing.

    Args:
        x: Input sample (Q15)
        state: CatastropheSnapState
        pressure_q15: Fold control parameter (Q15)
        mix_q15: Wet/dry mix

    Returns:
        Processed output (Q15)
    """
    # Step 1: Fold catastrophe
    y = catastrophe_fold_sample(x, state.cat, pressure_q15)

    # Step 2: Mix
    dry = mul_q15_sat(x, Q15_ONE - mix_q15)
    wet = mul_q15_sat(y, mix_q15)
    y = sat16(dry + wet)

    # Step 3: Post LP
    y = one_pole_lp_sample(y, state.post_lp, state.lp_coeff)

    return y


def catastrophe_snap_block(
    L: List[int],
    R: List[int],
    state_L: CatastropheSnapState,
    state_R: CatastropheSnapState,
    pressure_q15: int = 10000,
    mix_q15: int = Q15_HALF
) -> Tuple[List[int], List[int]]:
    """Block-rate Catastrophe Snap processing."""
    outL = []
    outR = []
    for i in range(len(L)):
        outL.append(catastrophe_snap_sample(L[i], state_L, pressure_q15, mix_q15))
        outR.append(catastrophe_snap_sample(R[i], state_R, pressure_q15, mix_q15))
    return outL, outR


# =============================================================================
# PROGRAM 5: CATASTROPHE SWALLOWTAIL
# =============================================================================


@dataclass
class CatastropheSwallowtailState:
    """
    State for Catastrophe Swallowtail program.

    Signal chain: Input -> Swallowtail Catastrophe -> Tape -> LP -> Output

    Uses higher-order catastrophe for complex bifurcations, followed by
    tape warmth for smoothing harsh transitions.
    """
    cat: CatastropheState = None
    tape: TapeHysteresisState = None
    post_lp: OnePoleState = None
    lp_coeff: int = 0

    def __post_init__(self):
        if self.cat is None:
            self.cat = CatastropheState()
        if self.tape is None:
            self.tape = TapeHysteresisState()
        if self.post_lp is None:
            self.post_lp = OnePoleState()
        self.lp_coeff = freq_to_one_pole_coeff(7000.0)


def catastrophe_swallowtail_program_sample(
    x: int,
    state: CatastropheSwallowtailState,
    a_q15: int = 5000,
    b_q15: int = 3000,
    c_q15: int = 1000,
    tape_drive_q15: int = Q15_HALF,
    mix_q15: int = Q15_HALF
) -> int:
    """
    Single-sample Catastrophe Swallowtail processing.

    Args:
        x: Input sample (Q15)
        state: CatastropheSwallowtailState
        a_q15, b_q15, c_q15: Swallowtail control parameters
        tape_drive_q15: Post-catastrophe tape drive
        mix_q15: Wet/dry mix

    Returns:
        Processed output (Q15)
    """
    # Step 1: Swallowtail catastrophe (use the building block function)
    y = catastrophe_swallowtail_sample(x, state.cat, a_q15, b_q15, c_q15)

    # Step 2: Tape smoothing
    y = tape_hysteresis_sample(y, state.tape, drive_q15=tape_drive_q15)

    # Step 3: Mix
    dry = mul_q15_sat(x, Q15_ONE - mix_q15)
    wet = mul_q15_sat(y, mix_q15)
    y = sat16(dry + wet)

    # Step 4: Post LP
    y = one_pole_lp_sample(y, state.post_lp, state.lp_coeff)

    return y


def catastrophe_swallowtail_program_block(
    L: List[int],
    R: List[int],
    state_L: CatastropheSwallowtailState,
    state_R: CatastropheSwallowtailState,
    a_q15: int = 5000,
    b_q15: int = 3000,
    c_q15: int = 1000,
    tape_drive_q15: int = Q15_HALF,
    mix_q15: int = Q15_HALF
) -> Tuple[List[int], List[int]]:
    """Block-rate Catastrophe Swallowtail processing."""
    outL = []
    outR = []
    for i in range(len(L)):
        outL.append(catastrophe_swallowtail_program_sample(
            L[i], state_L, a_q15, b_q15, c_q15, tape_drive_q15, mix_q15
        ))
        outR.append(catastrophe_swallowtail_program_sample(
            R[i], state_R, a_q15, b_q15, c_q15, tape_drive_q15, mix_q15
        ))
    return outL, outR


# =============================================================================
# PROGRAM REGISTRY
# =============================================================================


PROGRAM_INFO = {
    0: {
        'name': 'Clean Glue',
        'description': 'Transparent processing with gentle glue compression',
        'state_class': CleanGlueState,
        'sample_fn': clean_glue_sample,
        'block_fn': clean_glue_block,
    },
    1: {
        'name': 'Tape Bus',
        'description': 'Tape hysteresis saturation with warmth',
        'state_class': TapeBusState,
        'sample_fn': tape_bus_sample,
        'block_fn': tape_bus_block,
    },
    2: {
        'name': 'Rank Grit',
        'description': 'Rank statistics for unique gritty texture',
        'state_class': RankGritState,
        'sample_fn': rank_grit_sample,
        'block_fn': rank_grit_block,
    },
    3: {
        'name': 'Tropical Cut',
        'description': 'Hard threshold gating and shaping',
        'state_class': TropicalCutState,
        'sample_fn': tropical_cut_sample,
        'block_fn': tropical_cut_block,
    },
    4: {
        'name': 'Catastrophe Snap',
        'description': 'Fold catastrophe for snap transitions',
        'state_class': CatastropheSnapState,
        'sample_fn': catastrophe_snap_sample,
        'block_fn': catastrophe_snap_block,
    },
    5: {
        'name': 'Catastrophe Swallowtail',
        'description': 'Higher-order catastrophe with tape warmth',
        'state_class': CatastropheSwallowtailState,
        'sample_fn': catastrophe_swallowtail_program_sample,
        'block_fn': catastrophe_swallowtail_program_block,
    },
}


def get_program_state(program_id: int):
    """Create fresh state for the specified program."""
    if program_id not in PROGRAM_INFO:
        raise ValueError(f"Unknown program ID: {program_id}")
    return PROGRAM_INFO[program_id]['state_class']()


def get_program_name(program_id: int) -> str:
    """Get human-readable name for program."""
    if program_id not in PROGRAM_INFO:
        return f"Unknown ({program_id})"
    return PROGRAM_INFO[program_id]['name']
