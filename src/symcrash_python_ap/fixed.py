"""
Fixed-Point Arithmetic Layer (bit-accurate)
============================================

Conforms to WSE-STD-001 — Fixed-Point Arithmetic & Numeric Hygiene (v0)

All audio pipeline operations depend on these primitives.
Python reference uses s64/Python int for all intermediates to ensure
bit-exact results matching FPGA implementation.

Formats (from STD-001):
- Q1.15 (s16): gains, strengths, normalized coefficients [-1, 1)
- Q8.8  (s16): slow rates, UI values, Hz coarse values [-128, 127.996)
- Q0.16 (u16): unipolar controls [0, 0.99998)
- Q16.16 (s32): long accumulators/envelopes [-32768, 32767.999)

Audio samples and coefficients typically use Q1.15.
"""

import numpy as np
from typing import List

# =============================================================================
# CONSTANTS (WSE-STD-001 Section 3)
# =============================================================================

# Q1.15 format constants
Q15_SHIFT = 15
ONE_Q15 = 0x7FFF       # ≈ 0.999969482 (canonical constant per STD-001)
Q15_ONE = ONE_Q15      # Alias for compatibility
Q15_MINUS_ONE = -32768
Q15_HALF = 16384
Q15_SCALE = 32768
ZERO = 0

# Integer bounds (STD-001 Section 2)
MIN_S16 = -32768
MAX_S16 = 32767
INT16_MIN = MIN_S16    # Alias for compatibility
INT16_MAX = MAX_S16    # Alias for compatibility
INT32_MAX = 2147483647
INT32_MIN = -2147483648


# =============================================================================
# SATURATION (WSE-STD-001 Section 5)
# =============================================================================

def SAT16(x: int) -> int:
    """
    Saturate to s16 range (WSE-STD-001 Section 5.1).

    Maps any integer x to [-32768, 32767]:
    - if x > 32767  → 32767
    - if x < -32768 → -32768
    - else          → x

    NORMATIVE: Any intermediate that might exceed container width
    MUST be saturated. Silent wraparound is FORBIDDEN.
    """
    if x > MAX_S16:
        return MAX_S16
    elif x < MIN_S16:
        return MIN_S16
    return int(x)


def SAT32(x: int) -> int:
    """
    Saturate to s32 range (WSE-STD-001 Section 5.2).

    Maps any integer x to [-2^31, 2^31-1].
    """
    if x > INT32_MAX:
        return INT32_MAX
    elif x < INT32_MIN:
        return INT32_MIN
    return int(x)


# Aliases for compatibility
sat16 = SAT16
sat32 = SAT32


# =============================================================================
# ROUNDING (WSE-STD-001 Section 6)
# =============================================================================

def RSHIFT_RNE(x: int, k: int) -> int:
    """
    Round-to-nearest, ties-to-even right shift (WSE-STD-001 Section 6.1).

    Computes (x / 2^k) rounded to nearest integer, with ties going to even.
    This is banker's rounding - the canonical rounding mode for bit-exact paths.

    Algorithm:
    - q = x >> k (arithmetic shift, sign-extended)
    - r = remainder
    - half = 1 << (k-1)
    - if r > half: round up
    - if r < half: round down (truncate)
    - if r == half: round to even (inc only if q is odd)

    For negative numbers, we apply the same logic to abs(x) and restore sign.
    This avoids Python's floor-division quirks.

    NORMATIVE: All "multiply-then-shift" reductions MUST use this rounding mode.
    """
    if k <= 0:
        return x << (-k)

    if k > 63:
        return 0  # Shift amount too large

    half = 1 << (k - 1)

    if x >= 0:
        q = x >> k
        r = x & ((1 << k) - 1)

        if r > half:
            return q + 1
        elif r < half:
            return q
        else:  # r == half, tie - round to even
            return q + 1 if (q & 1) else q
    else:
        # For negative: work with absolute value, then restore sign
        abs_x = -x
        q = abs_x >> k
        r = abs_x & ((1 << k) - 1)

        if r > half:
            return -(q + 1)
        elif r < half:
            return -q
        else:  # r == half, tie - round to even
            return -(q + 1) if (q & 1) else -q


def shift_round(x: int, shift: int) -> int:
    """
    Legacy alias for RSHIFT_RNE.

    Note: Previous implementation used simple round-to-nearest (ties round up).
    This version uses the STD-001 compliant ties-to-even rounding.
    """
    return RSHIFT_RNE(x, shift)


def shift_round_sat16(x: int, shift: int) -> int:
    """Shift with RNE rounding, then saturate to s16."""
    return SAT16(RSHIFT_RNE(x, shift))


# =============================================================================
# MULTIPLICATION (WSE-STD-001 Section 7)
# =============================================================================

def MUL_Q15(a: int, b: int) -> int:
    """
    Q1.15 multiply (WSE-STD-001 Section 7.1).

    Inputs: a, b are s16 interpreted as Q1.15
    Process:
    1. Compute wide product: p = (s64)a * (s64)b  → Q2.30
    2. Reduce to Q1.15 via RSHIFT_RNE(p, 15)
    3. Saturate to s16 via SAT16

    Formula: MUL_Q15(a,b) = SAT16(RSHIFT_RNE((s64)a*(s64)b, 15))

    NORMATIVE: Every multiply MUST be computed in s64 (Python int),
    then reduced with the rounding/saturation rules above.
    """
    # Python int handles arbitrary precision, equivalent to s64
    p = a * b  # Q2.30
    return SAT16(RSHIFT_RNE(p, 15))


def MUL_Q15_TO_Q31(a: int, b: int) -> int:
    """
    Q1.15 multiply to Q1.31 (WSE-STD-001 Section 7.2).

    For accumulating gains or when wider output is needed.

    Process:
    1. p = (s64)a * (s64)b  → Q2.30
    2. Shift left by 1 to Q1.31: p2 = p << 1
    3. Saturate to s32: SAT32(p2)
    """
    p = a * b  # Q2.30
    p2 = p << 1  # Q1.31
    return SAT32(p2)


def mac_q15(acc: int, a: int, b: int) -> int:
    """
    Multiply-accumulate in s32.

    Adds (a * b) to accumulator, saturates to s32.
    Note: Product is Q2.30, accumulator should be treated accordingly.
    """
    return SAT32(acc + a * b)


# Aliases for compatibility
mul_q15 = MUL_Q15
mul_q15_sat = MUL_Q15  # MUL_Q15 now includes saturation per STD-001


# =============================================================================
# ADD/MIX PRIMITIVES (WSE-STD-001 Section 8)
# =============================================================================

def ADD_Q15(a: int, b: int) -> int:
    """
    Add two Q1.15 values with saturation (WSE-STD-001 Section 8.1).

    Formula: ADD_Q15(a,b) = SAT16((s32)a + (s32)b)
    """
    return SAT16(a + b)


def MIX_Q15(x: int, y: int, t: int) -> int:
    """
    Weighted crossfade mix (WSE-STD-001 Section 8.2).

    Given x, y in Q1.15 and t in Q1.15 where:
    - t = 0    → returns x
    - t = ONE  → returns y

    Formula: MIX_Q15(x,y,t) = ADD_Q15(MUL_Q15(x, ONE-t), MUL_Q15(y, t))

    NORMATIVE: All tree morphs, coefficient blends, and OLA-style blends
    in bit-exact mode must use this (or an explicitly defined equivalent).
    """
    # ONE_Q15 - t, but we need to handle potential overflow
    # Since ONE_Q15 = 0x7FFF and t can be up to 0x7FFF, the difference is safe
    one_minus_t = ONE_Q15 - t
    return ADD_Q15(MUL_Q15(x, one_minus_t), MUL_Q15(y, t))


# =============================================================================
# ABSOLUTE VALUE (WSE-STD-001 Section 9)
# =============================================================================

def ABS16(x: int) -> int:
    """
    Saturating absolute value (WSE-STD-001 Section 9).

    Avoids the "abs(minint) overflows" trap:
    - If x == -32768, returns 32767 (saturating)
    - Else returns abs(x)
    """
    if x == MIN_S16:
        return MAX_S16
    return abs(x)


# =============================================================================
# SLEW RATE LIMITING
# =============================================================================

def slew_q15(cur: int, target: int, max_delta: int) -> int:
    """Slew cur toward target with maximum step size."""
    delta = target - cur
    if delta > max_delta:
        return cur + max_delta
    elif delta < -max_delta:
        return cur - max_delta
    return target


def slew_q15_asym(cur: int, target: int, attack: int, release: int) -> int:
    """Asymmetric slew with different attack/release rates."""
    delta = target - cur
    if delta > 0:
        return cur + min(delta, attack)
    else:
        return cur + max(delta, -release)


# =============================================================================
# TANH LUT (bit-accurate)
# =============================================================================

_TANH_LUT_SIZE = 256
_TANH_LUT_SHIFT = 8  # Fixed: map full int16 range to 256 entries

def _build_tanh_lut() -> List[int]:
    """Build tanh LUT covering input range [-32768, 32767] -> index [0, 255]."""
    lut = []
    for i in range(_TANH_LUT_SIZE):
        # Map index to normalized input: i=0 -> -4, i=128 -> 0, i=255 -> ~4
        x = (i - 128) / 32.0
        y = np.tanh(x)
        lut.append(int(round(y * Q15_ONE)))
    return lut

_TANH_LUT = _build_tanh_lut()


def tanh_q15(x: int) -> int:
    """
    Bit-accurate tanh via LUT.
    Input: Q1.15 in [-32768, 32767]
    Output: Q1.15 tanh, saturating at ±32767 for large inputs
    """
    # Map input range to LUT index [0, 255]
    # x = -32768 -> idx = 0
    # x = 0 -> idx = 128  
    # x = 32767 -> idx = 255
    idx = (x + 32768) >> _TANH_LUT_SHIFT
    idx = max(0, min(_TANH_LUT_SIZE - 1, idx))
    return _TANH_LUT[idx]


def tanh_q15_interp(x: int) -> int:
    """Tanh with linear interpolation."""
    x_shifted = x + 32768
    idx = x_shifted >> _TANH_LUT_SHIFT
    frac = x_shifted & ((1 << _TANH_LUT_SHIFT) - 1)
    idx = max(0, min(_TANH_LUT_SIZE - 2, idx))
    y0 = _TANH_LUT[idx]
    y1 = _TANH_LUT[idx + 1]
    delta = y1 - y0
    interp = (delta * frac) >> _TANH_LUT_SHIFT
    return sat16(y0 + interp)


# =============================================================================
# SQRT LUT (for RMS)
# =============================================================================

_SQRT_LUT_SIZE = 256

def _build_sqrt_lut() -> List[int]:
    lut = []
    for i in range(_SQRT_LUT_SIZE):
        x = i / (_SQRT_LUT_SIZE - 1) if i > 0 else 0
        y = np.sqrt(x)
        lut.append(int(round(y * Q15_ONE)))
    return lut

_SQRT_LUT = _build_sqrt_lut()


def sqrt_q15(x: int) -> int:
    """Approximate sqrt for Q1.15 positive input."""
    if x <= 0:
        return 0
    if x >= Q15_ONE:
        return Q15_ONE
    idx = (x * (_SQRT_LUT_SIZE - 1)) >> Q15_SHIFT
    idx = min(_SQRT_LUT_SIZE - 1, idx)
    return _SQRT_LUT[idx]


# =============================================================================
# SOFT CLIPPING
# =============================================================================

def soft_clip_q15(x: int, threshold: int = 24576) -> int:
    """Soft clip: linear below threshold, compressed above."""
    if abs(x) <= threshold:
        return x
    sign = 1 if x >= 0 else -1
    excess = abs(x) - threshold
    compressed = threshold + (excess >> 1)
    return sat16(sign * min(compressed, Q15_ONE))


# =============================================================================
# VECTOR OPERATIONS
# =============================================================================

def vec_mul_q15(a: List[int], b: List[int]) -> List[int]:
    """Element-wise multiply two Q1.15 vectors."""
    return [mul_q15_sat(ai, bi) for ai, bi in zip(a, b)]


def vec_add_sat(a: List[int], b: List[int]) -> List[int]:
    """Element-wise add with saturation."""
    return [sat16(ai + bi) for ai, bi in zip(a, b)]


def vec_scale_q15(v: List[int], gain: int) -> List[int]:
    """Scale vector by Q1.15 gain."""
    return [mul_q15_sat(x, gain) for x in v]


def vec_dot_q15(a: List[int], b: List[int]) -> int:
    """Dot product, accumulate in int32, return Q1.15."""
    acc = 0
    for ai, bi in zip(a, b):
        acc = mac_q15(acc, ai, bi)
    return shift_round_sat16(acc, Q15_SHIFT)


# =============================================================================
# PITCH TABLE (WSE-STD-PITCH v0)
# =============================================================================

# Precomputed phase increments for MIDI notes 0-127 at 48kHz sample rate
# phase_inc = int(round(440.0 * (2 ** ((note - 69) / 12.0)) / 48000.0 * (2**32)))
# Computed at module load time with double precision, then frozen as integers

PITCH_SAMPLE_RATE = 48000
PITCH_PHASE_BITS = 32


def _build_pitch_table() -> List[int]:
    """Build MIDI note to phase_inc table (computed once at load)."""
    table = []
    for note in range(128):
        freq = 440.0 * (2.0 ** ((note - 69) / 12.0))
        phase_inc = int(round((freq / PITCH_SAMPLE_RATE) * (2**PITCH_PHASE_BITS)))
        table.append(phase_inc)
    return table


_PITCH_TABLE = _build_pitch_table()

# Ratio for 1 cent in Q32.0: (2^(1/1200) - 1) * 2^32 ≈ 2485961
# Used for linear interpolation of cents
_CENT_RATIO_Q32 = 2485961


def midi_to_phase_inc_fixed(note: int, cents: int = 0) -> int:
    """
    Convert MIDI note + cents to phase increment using integer LUT.

    This is the bit-exact version (WSE-STD-001 compliant) that avoids
    floating-point noise. Uses precomputed table for notes and linear
    interpolation for cents with RSHIFT_RNE rounding.

    Args:
        note: MIDI note number (0-127)
        cents: Fine tuning in cents (-9999 to +9999)

    Returns:
        32-bit phase increment for oscillator at 48kHz sample rate
    """
    # Clamp note to valid range
    note = max(0, min(127, note))
    base_inc = _PITCH_TABLE[note]

    if cents == 0:
        return base_inc

    # For larger cent ranges, use semitone stepping
    if abs(cents) >= 100:
        # Handle full semitones
        semitones = cents // 100
        remaining_cents = cents % 100

        # Handle negative cents properly
        if cents < 0 and remaining_cents != 0:
            semitones -= 1
            remaining_cents += 100

        target_note = note + semitones
        if target_note < 0 or target_note > 127:
            # Out of range - clamp
            target_note = max(0, min(127, target_note))
            remaining_cents = 0

        base_inc = _PITCH_TABLE[target_note]
        cents = remaining_cents

    if cents == 0:
        return base_inc

    # Linear interpolation for remaining cents (0-99)
    # delta = base_inc * cents * (2^(1/1200) - 1)
    # Using WSE-STD-001 compliant RSHIFT_RNE for bit-exact rounding
    product = base_inc * cents * _CENT_RATIO_Q32
    delta = RSHIFT_RNE(product, 32)

    return base_inc + delta


