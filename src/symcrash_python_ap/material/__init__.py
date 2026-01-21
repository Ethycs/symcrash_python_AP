"""Material Engine - Nonlinear signal processing.

Includes tape saturation, catastrophe bifurcation, and rank filtering.

Building Blocks (P1-P7):
- P1: Tilt EQ (pre-emphasis / de-emphasis)
- P2: Soft Saturator (tanh, soft clip, hard clip, asymmetric)
- P3: Tape Hysteresis (Jiles-Atherton style)
- P4: One-pole LP/HP filters
- P5: Rank Nonlinearity (median, trimmed mean)
- P6: Tropical min/max operations
- P7: Catastrophe shapers (fold, cusp, swallowtail)
"""

from .material_engine import (
    MaterialEngine,
    TapeState,
    CatastropheState,
    RankState,
)

from .programs import (
    # Program 0: Clean Glue
    CleanGlueState,
    clean_glue_sample,
    clean_glue_block,
    # Program 1: Tape Bus
    TapeBusState,
    tape_bus_sample,
    tape_bus_block,
    # Program 2: Rank Grit
    RankGritState,
    rank_grit_sample,
    rank_grit_block,
    # Program 3: Tropical Cut
    TropicalCutState,
    tropical_cut_sample,
    tropical_cut_block,
    # Program 4: Catastrophe Snap
    CatastropheSnapState,
    catastrophe_snap_sample,
    catastrophe_snap_block,
    # Program 5: Catastrophe Swallowtail
    CatastropheSwallowtailState,
    catastrophe_swallowtail_program_sample,
    catastrophe_swallowtail_program_block,
    # Registry
    PROGRAM_INFO,
    get_program_state,
    get_program_name,
)

from .building_blocks import (
    # P1: Tilt EQ
    TiltState,
    tilt_sample,
    # P2: Saturator
    SaturatorParams,
    saturator_sample,
    # P3: Tape Hysteresis
    TapeHysteresisState,
    tape_hysteresis_sample,
    # P4: One-pole filters
    OnePoleState,
    one_pole_lp_sample,
    one_pole_hp_sample,
    freq_to_one_pole_coeff,
    # P5: Rank nonlinearity
    RankState as RankFilterState,  # Alias to avoid conflict
    rank_median_sample,
    rank_trimmed_mean_sample,
    # P6: Tropical operations
    tropical_max,
    tropical_min,
    tropical_gate_sample,
    # P7: Catastrophe shapers
    CatastropheState as CatastropheShaperState,  # Alias
    catastrophe_fold_sample,
    catastrophe_cusp_sample,
    catastrophe_swallowtail_sample,
    # Composite
    pre_post_filter,
)

__all__ = [
    # Material Engine (legacy)
    'MaterialEngine',
    'TapeState',
    'CatastropheState',
    'RankState',
    # Programs (0-5)
    'CleanGlueState',
    'clean_glue_sample',
    'clean_glue_block',
    'TapeBusState',
    'tape_bus_sample',
    'tape_bus_block',
    'RankGritState',
    'rank_grit_sample',
    'rank_grit_block',
    'TropicalCutState',
    'tropical_cut_sample',
    'tropical_cut_block',
    'CatastropheSnapState',
    'catastrophe_snap_sample',
    'catastrophe_snap_block',
    'CatastropheSwallowtailState',
    'catastrophe_swallowtail_program_sample',
    'catastrophe_swallowtail_program_block',
    'PROGRAM_INFO',
    'get_program_state',
    'get_program_name',
    # P1: Tilt
    'TiltState',
    'tilt_sample',
    # P2: Saturator
    'SaturatorParams',
    'saturator_sample',
    # P3: Tape
    'TapeHysteresisState',
    'tape_hysteresis_sample',
    # P4: One-pole
    'OnePoleState',
    'one_pole_lp_sample',
    'one_pole_hp_sample',
    'freq_to_one_pole_coeff',
    # P5: Rank
    'RankFilterState',
    'rank_median_sample',
    'rank_trimmed_mean_sample',
    # P6: Tropical
    'tropical_max',
    'tropical_min',
    'tropical_gate_sample',
    # P7: Catastrophe
    'CatastropheShaperState',
    'catastrophe_fold_sample',
    'catastrophe_cusp_sample',
    'catastrophe_swallowtail_sample',
    # Composite
    'pre_post_filter',
]
