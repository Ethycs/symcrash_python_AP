"""Exact integer RMS reduction for signed-Q15 sample energies."""
from math import isqrt

from ..primitives import Q15_ONE


def rms_from_energy(sum_squares: int, sample_count: int) -> int:
    """SAT_Q15(round_even(sqrt(sum_squares / sample_count))).

    Accumulate unshifted, widened sample squares before calling. Comparing
    against the squared half-integer boundary avoids rounding the mean before
    taking its root. Empty input has RMS zero; inconsistent energies are errors.
    """
    energy, count = int(sum_squares), int(sample_count)
    if energy < 0 or count < 0 or (count == 0 and energy != 0):
        raise ValueError('RMS requires nonnegative energy and a positive sample count')
    if energy == 0:
        return 0
    if energy >= count * Q15_ONE * Q15_ONE:
        return Q15_ONE
    root = isqrt(energy // count)
    midpoint_squared = count * (2 * root + 1) ** 2
    if 4 * energy > midpoint_squared or (4 * energy == midpoint_squared and root & 1):
        root += 1
    return root
