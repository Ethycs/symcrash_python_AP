"""Validation and scalar reference preparation for native spectral gains."""
from ..primitives import RSHIFT_RNE, Q15_ONE, Q15_SCALE


def validate_harmonic_gains(gains, bound):
    if type(bound) is not int or not Q15_SCALE <= bound <= 4 * Q15_SCALE:
        raise ValueError('Harmonic gain bound must be an integer in 32768..131072')
    if gains is None:
        if bound != Q15_SCALE:
            raise ValueError('Absent harmonic gains require the unity bound')
    elif (type(gains) not in (list, tuple) or len(gains) != 32 or
          any(type(g) is not int or not 0 <= g <= bound for g in gains)):
        raise ValueError('Harmonic gains require 32 integers between zero and their bound')


def modulated_weights(weights, count, gains, bound):
    """Apply gains to detached wide weights with constant conservative headroom.

    Both signed sine/cosine components receive the same nonnegative gain.
    The base spectrum and DC are untouched. Zero-depth callers bypass this.
    """
    validate_harmonic_gains(gains, bound)
    a = [int(v) for v in weights.a_h_q15[:count]]
    b = [int(v) for v in weights.b_h_q15[:count]]
    total = (sum(max(abs(x), abs(y)) for x, y in zip(a, b)) * bound + Q15_ONE) >> 15
    shift = 0
    while (total >> shift) > Q15_ONE:
        shift += 1
    return ([RSHIFT_RNE((x >> shift) * gains[h], 15) for h, x in enumerate(a)],
            [RSHIFT_RNE((y >> shift) * gains[h], 15) for h, y in enumerate(b)])
