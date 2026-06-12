"""
Material Engine Building Blocks (P1-P7)
=======================================

Reusable DSP primitives for material programs.
All processing is Q1.15 bit-accurate.

Building Blocks:
- P1: Pre-emphasis / Tilt EQ
- P2: Soft Saturator (multiple curves)
- P3: Tape Hysteresis Model
- P4: One-pole LP/HP Filter
- P5: Rank Nonlinearity (median/trimmed)
- P6: Tropical min/max Operations
- P7: Catastrophe Shaper (fold/cusp/swallowtail)

Conforms to WSE-STD-001 (Fixed-Point Arithmetic).
"""

from dataclasses import dataclass
from typing import Tuple

from ..fixed import (
    sat16, mul_q15_sat, shift_round, slew_q15,
    tanh_q15, soft_clip_q15,
    Q15_ONE, Q15_SHIFT, Q15_HALF, RSHIFT_RNE
)
from ..primitives import Q15_3_4, Q15_NEG_ONE


# =============================================================================
# P1: PRE-EMPHASIS / TILT EQ
# =============================================================================


@dataclass
class TiltState:
    """State for tilt EQ (one-pole shelving)."""
    z1: int = 0  # Previous output (Q15)


def tilt_sample(x: int, state: TiltState, tilt_q15: int) -> int:
    """
    Single-sample tilt EQ.

    Simple one-pole shelving filter that boosts/cuts highs relative to lows.

    Args:
        x: Input sample (Q15)
        state: TiltState with filter memory
        tilt_q15: Tilt amount (-32768 to 32767)
                  Negative = low shelf boost (darker)
                  Zero = flat
                  Positive = high shelf boost (brighter)

    Returns:
        Filtered output (Q15)
    """
    # Coefficient for tilt (maps tilt to cutoff-like behavior)
    # tilt = 0 -> coeff = 0.5 (neutral)
    # tilt > 0 -> coeff > 0.5 (more high freq)
    # tilt < 0 -> coeff < 0.5 (more low freq)
    coeff = Q15_HALF + (tilt_q15 >> 2)  # Map to ~0.25 to 0.75
    coeff = max(1000, min(Q15_ONE - 1000, coeff))  # Clamp for stability

    # High-frequency path (differentiated)
    hp = x - state.z1

    # Mix based on tilt
    # Positive tilt: emphasize HP
    # Negative tilt: de-emphasize HP (emphasize LP)
    if tilt_q15 >= 0:
        # Brighten: mix in more high frequencies
        emphasis = mul_q15_sat(hp, coeff)
        output = sat16(x + emphasis)
    else:
        # Darken: mix in low frequencies (subtract highs)
        emphasis = mul_q15_sat(hp, -tilt_q15)  # Invert for subtraction
        output = sat16(x - (emphasis >> 1))

    # Update state (one-pole lowpass for z1)
    state.z1 = sat16(state.z1 + ((x - state.z1) >> 2))

    return output


# =============================================================================
# P2: SOFT SATURATOR
# =============================================================================


@dataclass
class SaturatorParams:
    """Parameters for saturator."""
    drive_q15: int = Q15_HALF  # Input drive (0.5 default)
    curve: int = 0  # 0=tanh, 1=soft_clip, 2=hard_clip, 3=asymmetric


def saturator_sample(x: int, params: SaturatorParams) -> int:
    """
    Single-sample saturator with selectable curve.

    Curves:
        0 = tanh (smooth, musical)
        1 = soft_clip (knee at threshold)
        2 = hard_clip (abrupt)
        3 = asymmetric (tube-like odd harmonics)

    Args:
        x: Input sample (Q15)
        params: SaturatorParams with drive and curve selection

    Returns:
        Saturated output (Q15)
    """
    # Apply drive
    driven = mul_q15_sat(x, params.drive_q15)
    # Extra gain for drive > 0.5
    if params.drive_q15 > Q15_HALF:
        extra = params.drive_q15 - Q15_HALF
        driven = sat16(driven + mul_q15_sat(driven, extra))

    # Apply selected curve
    if params.curve == 0:
        # Tanh: smooth, symmetric saturation
        return tanh_q15(driven)

    elif params.curve == 1:
        # Soft clip: linear below threshold, compressed above
        return soft_clip_q15(driven, threshold=Q15_3_4)

    elif params.curve == 2:
        # Hard clip: abrupt limiting
        return sat16(driven)

    else:  # curve == 3
        # Asymmetric: different positive/negative saturation
        # Creates odd harmonics like tube distortion
        if driven >= 0:
            # Positive: gentler saturation
            return soft_clip_q15(driven, threshold=28000)
        else:
            # Negative: harder saturation
            return soft_clip_q15(driven, threshold=20000)


# =============================================================================
# P3: TAPE HYSTERESIS MODEL
# =============================================================================


@dataclass
class TapeHysteresisState:
    """State for tape hysteresis model."""
    m: int = 0  # Magnetization state (Q15)
    prev_input: int = 0  # Previous input for slew detection


def tape_hysteresis_sample(
    x: int,
    state: TapeHysteresisState,
    drive_q15: int = Q15_HALF,
    bias_q15: int = 0,
    saturation_q15: int = Q15_HALF
) -> int:
    """
    Single-sample tape hysteresis model.

    Models the nonlinear magnetization curve of analog tape.
    Based on Jiles-Atherton-style hysteresis with simplified math.

    Model: m' = m + λ(tanh(drive * x + bias) - m) * (1 - ρ*m²)

    Args:
        x: Input sample (Q15)
        state: TapeHysteresisState with magnetization memory
        drive_q15: Input drive/gain (Q15)
        bias_q15: Asymmetric bias (Q15)
        saturation_q15: Saturation amount (Q15)

    Returns:
        Output through hysteresis (Q15)
    """
    # λ (feedback rate): higher drive -> faster response
    lam = min(Q15_ONE, drive_q15 + Q15_HALF)

    # ρ (saturation depth): controls compression curve
    rho = saturation_q15 >> 2

    # Apply drive and bias
    driven = mul_q15_sat(x, drive_q15)
    driven = sat16(driven + bias_q15)

    # Target magnetization (tanh of driven input)
    target = tanh_q15(driven)

    # Hysteresis damping: (1 - ρ*m²)
    m_sq = mul_q15_sat(state.m, state.m)
    damp = Q15_ONE - mul_q15_sat(rho, m_sq)
    damp = max(0, damp)

    # Update magnetization: m' = m + λ(target - m) * damp
    delta = mul_q15_sat(target - state.m, lam)
    delta = mul_q15_sat(delta, damp)

    state.m = sat16(state.m + (delta >> 1))
    state.prev_input = x

    return state.m


# =============================================================================
# P4: ONE-POLE LP/HP FILTER
# =============================================================================


@dataclass
class OnePoleState:
    """State for one-pole filter."""
    z1: int = 0  # Filter state (Q15)


def one_pole_lp_sample(x: int, state: OnePoleState, coeff_q15: int) -> int:
    """
    One-pole lowpass filter.

    y[n] = y[n-1] + coeff * (x[n] - y[n-1])

    Args:
        x: Input sample (Q15)
        state: OnePoleState with filter memory
        coeff_q15: Filter coefficient (Q15)
                   0 = no filtering (DC only)
                   Q15_ONE = bypass (no filtering)

    Returns:
        Lowpass filtered output (Q15)
    """
    diff = x - state.z1
    update = mul_q15_sat(diff, coeff_q15)
    state.z1 = sat16(state.z1 + update)
    return state.z1


def one_pole_hp_sample(x: int, state: OnePoleState, coeff_q15: int) -> int:
    """
    One-pole highpass filter.

    HP = Input - LP

    Args:
        x: Input sample (Q15)
        state: OnePoleState with filter memory
        coeff_q15: Filter coefficient (Q15)
                   Higher = more high frequencies pass

    Returns:
        Highpass filtered output (Q15)
    """
    # First get lowpass
    lp = one_pole_lp_sample(x, state, Q15_ONE - coeff_q15)
    # HP = input - lowpass
    return sat16(x - lp)


def freq_to_one_pole_coeff(freq_hz: float, sample_rate: int = 48000) -> int:
    """
    Convert frequency to one-pole coefficient.

    Args:
        freq_hz: Cutoff frequency in Hz
        sample_rate: Sample rate in Hz

    Returns:
        Filter coefficient (Q15)
    """
    import math
    # One-pole: coeff = 1 - exp(-2*pi*fc/fs)
    fc = max(1.0, min(freq_hz, sample_rate * 0.49))
    coeff = 1.0 - math.exp(-2.0 * math.pi * fc / sample_rate)
    return int(coeff * Q15_ONE)


# =============================================================================
# P5: RANK NONLINEARITY (Median/Trimmed Mean)
# =============================================================================


@dataclass
class RankState:
    """State for rank statistics filter."""
    buffer: list = None
    idx: int = 0
    window_size: int = 9

    def __post_init__(self):
        if self.buffer is None:
            self.buffer = [0] * self.window_size


def rank_median_sample(x: int, state: RankState) -> int:
    """
    Rank filter: median of sliding window.

    Good for impulse noise removal while preserving edges.

    Args:
        x: Input sample (Q15)
        state: RankState with circular buffer

    Returns:
        Median filtered output (Q15)
    """
    # Add to circular buffer
    state.buffer[state.idx] = x
    state.idx = (state.idx + 1) % state.window_size

    # Sort and return median
    sorted_vals = sorted(state.buffer)
    return sorted_vals[state.window_size // 2]


def rank_trimmed_mean_sample(x: int, state: RankState, trim: int = 2) -> int:
    """
    Rank filter: trimmed mean (drop outliers).

    More robust than mean, smoother than median.

    Args:
        x: Input sample (Q15)
        state: RankState with circular buffer
        trim: Number of samples to drop from each end

    Returns:
        Trimmed mean filtered output (Q15)
    """
    # Add to circular buffer
    state.buffer[state.idx] = x
    state.idx = (state.idx + 1) % state.window_size

    # Sort and compute trimmed mean
    sorted_vals = sorted(state.buffer)

    if state.window_size > 2 * trim:
        trimmed = sorted_vals[trim:-trim]
        result = sum(trimmed) // len(trimmed)
    else:
        # Fallback to median if window too small
        result = sorted_vals[state.window_size // 2]

    return sat16(result)


# =============================================================================
# P6: TROPICAL MIN/MAX OPERATIONS
# =============================================================================


def tropical_max(x: int, y: int) -> int:
    """Tropical algebra max (corresponds to addition in tropical semiring)."""
    return max(x, y)


def tropical_min(x: int, y: int) -> int:
    """Tropical algebra min (corresponds to multiplication in tropical semiring)."""
    return min(x, y)


def tropical_gate_sample(x: int, threshold_q15: int, mode: int = 0) -> int:
    """
    Tropical gate: hard threshold operations.

    Modes:
        0 = Hard gate (zero below threshold)
        1 = Hard limit (clamp above threshold)
        2 = Dead zone (zero in [-threshold, +threshold])

    Args:
        x: Input sample (Q15)
        threshold_q15: Threshold level (Q15)
        mode: Gate mode

    Returns:
        Gated output (Q15)
    """
    abs_x = abs(x) if x != Q15_NEG_ONE else Q15_ONE

    if mode == 0:
        # Hard gate: zero below threshold
        if abs_x < threshold_q15:
            return 0
        return x

    elif mode == 1:
        # Hard limit: clamp above threshold
        if x > threshold_q15:
            return threshold_q15
        elif x < -threshold_q15:
            return -threshold_q15
        return x

    else:  # mode == 2
        # Dead zone: zero in range
        if abs_x < threshold_q15:
            return 0
        elif x > 0:
            return x - threshold_q15
        else:
            return x + threshold_q15


# =============================================================================
# P7: CATASTROPHE SHAPER (Fold/Cusp/Swallowtail)
# =============================================================================


@dataclass
class CatastropheState:
    """State for catastrophe dynamical system."""
    s: int = 0  # Slow state variable (Q15)


def catastrophe_fold_sample(
    x: int,
    state: CatastropheState,
    pressure_q15: int = 0
) -> int:
    """
    Fold catastrophe (A2) shaper.

    Potential: V(x) = x³ + ax
    Dynamics: x' = -∂V/∂x = -3x² - a

    Creates sudden "snap" transitions as pressure crosses threshold.

    Args:
        x: Input sample (Q15)
        state: CatastropheState with slow variable
        pressure_q15: Control parameter 'a' (Q15)

    Returns:
        Shaped output (Q15)
    """
    # Scale input to state variable range
    fast = x >> 2
    s = state.s

    # Fold dynamics
    s_sq = mul_q15_sat(s, s)
    s_cu = mul_q15_sat(s_sq, s)

    # Gradient: -3s² - a (simplified to -s³ - a*s)
    grad = -s_cu - mul_q15_sat(pressure_q15, s)

    # Integration with input driving
    rate = Q15_HALF
    delta = mul_q15_sat(rate, sat16(grad + fast))

    state.s = sat16(s + (delta >> 2))

    return soft_clip_q15(state.s)


def catastrophe_cusp_sample(
    x: int,
    state: CatastropheState,
    pressure_q15: int = 0,
    bias_q15: int = 0
) -> int:
    """
    Cusp catastrophe (A3) shaper.

    Potential: V(x) = x⁴ + ax² + bx
    Two control parameters: pressure (a) and bias (b).

    Creates hysteretic transitions with bistable regions.

    Args:
        x: Input sample (Q15)
        state: CatastropheState with slow variable
        pressure_q15: Control parameter 'a' (Q15)
        bias_q15: Control parameter 'b' (Q15)

    Returns:
        Shaped output (Q15)
    """
    fast = x >> 2
    s = state.s

    # Cusp dynamics
    s_sq = mul_q15_sat(s, s)
    s_cu = mul_q15_sat(s_sq, s)

    # Gradient: -4s³ - 2as - b
    grad = -mul_q15_sat(s_cu, 4 * Q15_ONE >> 15)
    grad = sat16(grad - mul_q15_sat(pressure_q15, s << 1))
    grad = sat16(grad - bias_q15)

    # Drive with input
    rate = Q15_HALF
    delta = mul_q15_sat(rate, sat16(grad + fast))

    state.s = sat16(s + (delta >> 2))

    return soft_clip_q15(state.s)


def catastrophe_swallowtail_sample(
    x: int,
    state: CatastropheState,
    a_q15: int = 0,
    b_q15: int = 0,
    c_q15: int = 0
) -> int:
    """
    Swallowtail catastrophe (A4) shaper.

    Potential: V(x) = x⁵ + ax³ + bx² + cx
    Three control parameters for richer bifurcation structure.

    Creates more complex transition patterns than fold/cusp.

    Args:
        x: Input sample (Q15)
        state: CatastropheState with slow variable
        a_q15, b_q15, c_q15: Control parameters (Q15)

    Returns:
        Shaped output (Q15)
    """
    fast = x >> 2
    s = state.s

    # Powers of s
    s_sq = mul_q15_sat(s, s)
    s_cu = mul_q15_sat(s_sq, s)
    s_4 = mul_q15_sat(s_sq, s_sq)

    # Gradient: -5s⁴ - 3as² - 2bs - c
    grad = -mul_q15_sat(s_4, 5 * Q15_ONE >> 15)
    grad = sat16(grad - mul_q15_sat(a_q15, mul_q15_sat(s_sq, 3 * Q15_ONE >> 15)))
    grad = sat16(grad - mul_q15_sat(b_q15, s << 1))
    grad = sat16(grad - c_q15)

    # Drive
    rate = Q15_HALF
    delta = mul_q15_sat(rate, sat16(grad + fast))

    state.s = sat16(s + (delta >> 3))  # Slower integration for stability

    return soft_clip_q15(state.s)


# =============================================================================
# COMPOSITE BUILDING BLOCKS
# =============================================================================


def pre_post_filter(
    x: int,
    pre_state: OnePoleState,
    post_state: OnePoleState,
    processor_fn,
    pre_freq_hz: float = 2000.0,
    post_freq_hz: float = 8000.0,
    sample_rate: int = 48000
) -> int:
    """
    Common pattern: pre-filter -> process -> post-filter.

    Pre-filter removes low frequencies before saturation (less mud).
    Post-filter removes harsh high frequencies after saturation.

    Args:
        x: Input sample (Q15)
        pre_state: State for pre-emphasis highpass
        post_state: State for post-filter lowpass
        processor_fn: Processing function (x) -> y
        pre_freq_hz: Pre-emphasis highpass cutoff
        post_freq_hz: Post-filter lowpass cutoff
        sample_rate: Sample rate

    Returns:
        Processed and filtered output (Q15)
    """
    # Pre-emphasis (highpass to remove lows before saturation)
    pre_coeff = freq_to_one_pole_coeff(pre_freq_hz, sample_rate)
    pre_filtered = one_pole_hp_sample(x, pre_state, pre_coeff)

    # Mix back some low end
    mixed = sat16((pre_filtered >> 1) + (x >> 1))

    # Process
    processed = processor_fn(mixed)

    # Post-filter (lowpass to smooth harshness)
    post_coeff = freq_to_one_pole_coeff(post_freq_hz, sample_rate)
    return one_pole_lp_sample(processed, post_state, post_coeff)
