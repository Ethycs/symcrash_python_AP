"""Integer TPT state-variable filter, with block-rate coefficient calculation.

The recurrence is the bounded-coefficient form in Andrew Simper's derivation:
https://cytomic.com/files/dsp/SvfLinearTrapOptimised2.pdf

Audio is Q1.15. Coefficients have 30 fractional bits; signed 32-bit integrator
states have 27 fractional bits (12 guard fractional bits over audio, with
headroom to +/-16). Products use wide accumulators, then RNE. Only state writes
and final audio are saturated. The fixed ROM covers the engine's 48 kHz rate.
"""

from dataclasses import dataclass

from ..primitives import Q15_ONE, RSHIFT_RNE, SAT16, SAT32
from ..tables.material_svf import SVF_G_Q24

COEFF_BITS = 30
STATE_GUARD_BITS = 12


def div_rne(numerator: int, denominator: int) -> int:
    """Signed integer division, nearest with ties to even; positive divisor."""
    quotient, remainder = divmod(numerator, denominator)
    twice = remainder * 2
    return quotient + int(twice > denominator or (twice == denominator and quotient & 1))


@dataclass(frozen=True)
class SVFCoefficients:
    a1: int
    a2: int
    a3: int
    k: int


def coefficients(cutoff_q15: int, reso_q15: int) -> SVFCoefficients:
    cutoff = max(0, min(Q15_ONE, int(cutoff_q15)))
    resonance = max(0, min(Q15_ONE, int(reso_q15)))
    index, fraction = divmod(cutoff * (len(SVF_G_Q24) - 1), Q15_ONE)
    g = SVF_G_Q24[index]
    if fraction:
        g += div_rne((SVF_G_Q24[index + 1] - g) * fraction, Q15_ONE)

    # Q = 0.5 + 24.5 * resonance; damping k = 1/Q, decreasing with Q.
    k = div_rne((2 * Q15_ONE) << COEFF_BITS, Q15_ONE + 49 * resonance)
    k_q24 = RSHIFT_RNE(k, 6)
    denominator_q48 = (1 << 48) + g * (g + k_q24)
    # Reduce to Q30 before the reciprocal so each intermediate fits 64 bits.
    denominator = RSHIFT_RNE(denominator_q48, 18)
    a1 = div_rne(1 << (2 * COEFF_BITS), denominator)
    a2 = RSHIFT_RNE(g * a1, 24)
    a3 = RSHIFT_RNE(g * a2, 24)
    return SVFCoefficients(a1, a2, a3, k)


def process_sample(x: int, state, coeff: SVFCoefficients, mode: int) -> int:
    x_wide = int(x) << STATE_GUARD_BITS
    v3 = x_wide - state.ic2eq
    v1 = RSHIFT_RNE(coeff.a1 * state.ic1eq + coeff.a2 * v3, COEFF_BITS)
    v2 = state.ic2eq + RSHIFT_RNE(coeff.a2 * state.ic1eq + coeff.a3 * v3, COEFF_BITS)
    hp = x_wide - RSHIFT_RNE(coeff.k * v1, COEFF_BITS) - v2

    state.ic1eq = SAT32(2 * v1 - state.ic1eq)
    state.ic2eq = SAT32(2 * v2 - state.ic2eq)

    output = (v2, hp, v1, v2 + hp)[mode]
    return SAT16(RSHIFT_RNE(output, STATE_GUARD_BITS))
