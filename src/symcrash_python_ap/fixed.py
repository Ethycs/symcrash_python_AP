"""
Fixed-Point Arithmetic Layer (bit-accurate)
============================================

All audio pipeline operations depend on these primitives.
Formats:
- Audio samples: Q1.15 signed int16 (-32768..32767)
- Coefficients: Q1.15 int16
- Accumulators: Q6.26 int32
- Gains/params: Q1.15 int16
"""

import numpy as np
from typing import List

# =============================================================================
# CONSTANTS
# =============================================================================

Q15_SHIFT = 15
Q15_ONE = 32767
Q15_MINUS_ONE = -32768
Q15_HALF = 16384
Q15_SCALE = 32768

INT16_MAX = 32767
INT16_MIN = -32768
INT32_MAX = 2147483647
INT32_MIN = -2147483648


# =============================================================================
# SATURATION
# =============================================================================

def sat16(x: int) -> int:
    """Saturate to int16 range."""
    if x > INT16_MAX:
        return INT16_MAX
    elif x < INT16_MIN:
        return INT16_MIN
    return int(x)


def sat32(x: int) -> int:
    """Saturate to int32 range."""
    if x > INT32_MAX:
        return INT32_MAX
    elif x < INT32_MIN:
        return INT32_MIN
    return int(x)


# =============================================================================
# ROUNDING
# =============================================================================

def shift_round(x: int, shift: int) -> int:
    """Arithmetic right shift with round-to-nearest."""
    if shift <= 0:
        return x << (-shift)
    round_bit = 1 << (shift - 1)
    return (x + round_bit) >> shift


def shift_round_sat16(x: int, shift: int) -> int:
    """Shift, round, and saturate to int16."""
    return sat16(shift_round(x, shift))


# =============================================================================
# MULTIPLICATION
# =============================================================================

def mul_q15(a: int, b: int) -> int:
    """Multiply two Q1.15 values, return Q1.15."""
    prod = a * b
    return shift_round(prod, Q15_SHIFT)


def mul_q15_sat(a: int, b: int) -> int:
    """Multiply two Q1.15 values, return saturated Q1.15."""
    return sat16(mul_q15(a, b))


def mac_q15(acc: int, a: int, b: int) -> int:
    """Multiply-accumulate in int32."""
    return sat32(acc + a * b)


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
