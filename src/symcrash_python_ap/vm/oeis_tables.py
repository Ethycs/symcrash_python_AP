"""
OEIS Sequence Tables for Harmonic Partial Selection
====================================================

This module provides precomputed integer sequences from OEIS for
driving partial selection in the Harmonic Bank.

Per THEORETICAL-TECHNIQUES-ROADMAP item 2.4:
- A000040 (primes) → metallic, bell-like
- A000045 (Fibonacci) → organic, inharmonic
- A001950 (Beatty φ) → golden ratio partials
- A002024 (triangular repetition) → stepped, quantized

Usage:
    from oeis_tables import OEISSequenceID, get_partial_mask

    # Get mask for first 32 harmonics using prime sequence
    mask = get_partial_mask(OEISSequenceID.PRIMES, 32)
    # mask[i] = True if harmonic (i+1) should be included
"""

from enum import IntEnum
from typing import List
import numpy as np

# =============================================================================
# SEQUENCE DEFINITIONS
# =============================================================================

class OEISSequenceID(IntEnum):
    """
    OEIS sequence identifiers for partial selection.

    Each sequence produces a different timbral character:
    - PRIMES: metallic, bell-like (sparse upper partials)
    - FIBONACCI: organic, complex (increasingly sparse)
    - BEATTY_PHI: golden ratio spacing (quasi-harmonic)
    - TRIANGULAR: stepped, quantized feel
    - LUCAS: similar to Fibonacci, different seed
    - SQUARE_FREE: excludes multiples of perfect squares
    """
    PRIMES = 0        # A000040: 2, 3, 5, 7, 11, 13, ...
    FIBONACCI = 1     # A000045: 1, 1, 2, 3, 5, 8, 13, 21, ...
    BEATTY_PHI = 2    # A001950: floor(n * φ) = 1, 3, 4, 6, 8, 9, 11, ...
    TRIANGULAR = 3    # A000217: 1, 3, 6, 10, 15, 21, ...
    LUCAS = 4         # A000032: 2, 1, 3, 4, 7, 11, 18, ...
    SQUARE_FREE = 5   # A005117: 1, 2, 3, 5, 6, 7, 10, 11, ...
    ODD_ONLY = 6      # Simple odd harmonics: 1, 3, 5, 7, ...
    MERSENNE = 7      # A000225: 2^n - 1 = 1, 3, 7, 15, 31, ...


# =============================================================================
# PRECOMPUTED SEQUENCES (first 64 elements)
# =============================================================================

# A000040: Prime numbers
_PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
    137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
    227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311
]

# A000045: Fibonacci numbers (starting from F(1)=1)
_FIBONACCI = [
    1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987,
    1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811,
    514229, 832040, 1346269, 2178309
]

# A001950: Beatty sequence for φ = (1 + √5) / 2 ≈ 1.618
# floor(n * φ) for n = 1, 2, 3, ...
_GOLDEN_RATIO = 1.6180339887498948482
_BEATTY_PHI = [int(n * _GOLDEN_RATIO) for n in range(1, 65)]

# A000217: Triangular numbers T(n) = n(n+1)/2
_TRIANGULAR = [n * (n + 1) // 2 for n in range(1, 65)]

# A000032: Lucas numbers (2, 1, 3, 4, 7, 11, ...)
def _compute_lucas(count: int) -> List[int]:
    lucas = [2, 1]
    while len(lucas) < count:
        lucas.append(lucas[-1] + lucas[-2])
    return lucas[:count]

_LUCAS = _compute_lucas(64)

# A005117: Square-free numbers (not divisible by any perfect square > 1)
def _is_square_free(n: int) -> bool:
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 1
    return True

_SQUARE_FREE = [n for n in range(1, 200) if _is_square_free(n)][:64]

# Odd harmonics (1, 3, 5, 7, ...)
_ODD_ONLY = [2 * n + 1 for n in range(64)]

# A000225: Mersenne numbers 2^n - 1
_MERSENNE = [2**n - 1 for n in range(1, 33)]


# Sequence lookup table
_SEQUENCE_DATA = {
    OEISSequenceID.PRIMES: _PRIMES,
    OEISSequenceID.FIBONACCI: _FIBONACCI,
    OEISSequenceID.BEATTY_PHI: _BEATTY_PHI,
    OEISSequenceID.TRIANGULAR: _TRIANGULAR,
    OEISSequenceID.LUCAS: _LUCAS,
    OEISSequenceID.SQUARE_FREE: _SQUARE_FREE,
    OEISSequenceID.ODD_ONLY: _ODD_ONLY,
    OEISSequenceID.MERSENNE: _MERSENNE,
}


# =============================================================================
# PUBLIC API
# =============================================================================

def get_sequence(seq_id: OEISSequenceID, count: int = 64) -> List[int]:
    """
    Get raw sequence values.

    Args:
        seq_id: OEISSequenceID to retrieve
        count: Maximum number of elements to return

    Returns:
        List of sequence values (up to count elements)
    """
    data = _SEQUENCE_DATA.get(seq_id, _PRIMES)
    return data[:count]


def get_partial_mask(seq_id: OEISSequenceID, H_max: int = 32) -> np.ndarray:
    """
    Get boolean mask for which harmonics to include.

    Maps sequence values to harmonic indices:
    - mask[h-1] = True if harmonic h is in the sequence
    - Only includes harmonics up to H_max

    Args:
        seq_id: OEISSequenceID defining which harmonics to include
        H_max: Maximum harmonic number to consider

    Returns:
        Boolean array of length H_max where mask[i] = True if
        harmonic (i+1) should be included
    """
    sequence = get_sequence(seq_id)
    mask = np.zeros(H_max, dtype=bool)

    for val in sequence:
        if 1 <= val <= H_max:
            mask[val - 1] = True

    return mask


def get_partial_amplitudes(
    seq_id: OEISSequenceID,
    H_max: int = 32,
    rolloff_db_per_octave: float = -3.0,
    base_amplitude_q15: int = 32767,
) -> np.ndarray:
    """
    Get Q15 amplitudes for harmonics in the sequence.

    Applies natural rolloff to higher partials for more musical results.
    Partials NOT in the sequence get amplitude 0.

    Args:
        seq_id: OEISSequenceID defining which harmonics to include
        H_max: Maximum harmonic number
        rolloff_db_per_octave: Amplitude rolloff per octave (negative = decay)
        base_amplitude_q15: Amplitude for fundamental (Q15)

    Returns:
        Q15 amplitude array of length H_max
    """
    import math

    mask = get_partial_mask(seq_id, H_max)
    amps = np.zeros(H_max, dtype=np.int16)

    for h in range(1, H_max + 1):
        if mask[h - 1]:
            # Compute rolloff: octaves from fundamental = log2(h)
            octaves = math.log2(h) if h > 1 else 0
            db_reduction = rolloff_db_per_octave * octaves
            linear_scale = 10.0 ** (db_reduction / 20.0)
            amps[h - 1] = int(base_amplitude_q15 * linear_scale)

    return amps


def get_partial_list(seq_id: OEISSequenceID, H_max: int = 32) -> List[int]:
    """
    Get list of harmonic numbers to include.

    More efficient than mask when only a few partials are needed.

    Args:
        seq_id: OEISSequenceID defining which harmonics to include
        H_max: Maximum harmonic number

    Returns:
        List of harmonic numbers (1-indexed) in the sequence
    """
    sequence = get_sequence(seq_id)
    return [val for val in sequence if 1 <= val <= H_max]


# =============================================================================
# TIMBRAL CHARACTER DESCRIPTIONS
# =============================================================================

SEQUENCE_DESCRIPTIONS = {
    OEISSequenceID.PRIMES: {
        "name": "Prime Partials",
        "character": "Metallic, bell-like, sparse upper register",
        "oeis": "A000040",
        "example_harmonics": "2, 3, 5, 7, 11, 13, 17, 19, ...",
    },
    OEISSequenceID.FIBONACCI: {
        "name": "Fibonacci Partials",
        "character": "Organic, complex, increasingly sparse",
        "oeis": "A000045",
        "example_harmonics": "1, 2, 3, 5, 8, 13, 21, ...",
    },
    OEISSequenceID.BEATTY_PHI: {
        "name": "Golden Ratio Partials",
        "character": "Quasi-harmonic, iridescent, non-repeating",
        "oeis": "A001950",
        "example_harmonics": "1, 3, 4, 6, 8, 9, 11, 12, 14, ...",
    },
    OEISSequenceID.TRIANGULAR: {
        "name": "Triangular Partials",
        "character": "Stepped, quantized, staircase feel",
        "oeis": "A000217",
        "example_harmonics": "1, 3, 6, 10, 15, 21, 28, ...",
    },
    OEISSequenceID.LUCAS: {
        "name": "Lucas Partials",
        "character": "Similar to Fibonacci, brighter seed",
        "oeis": "A000032",
        "example_harmonics": "2, 1, 3, 4, 7, 11, 18, ...",
    },
    OEISSequenceID.SQUARE_FREE: {
        "name": "Square-Free Partials",
        "character": "Dense but irregular, avoids resonant stacking",
        "oeis": "A005117",
        "example_harmonics": "1, 2, 3, 5, 6, 7, 10, 11, ...",
    },
    OEISSequenceID.ODD_ONLY: {
        "name": "Odd Partials",
        "character": "Hollow, clarinet-like (square wave family)",
        "oeis": "N/A",
        "example_harmonics": "1, 3, 5, 7, 9, 11, 13, ...",
    },
    OEISSequenceID.MERSENNE: {
        "name": "Mersenne Partials",
        "character": "Exponentially sparse, digital, glitchy",
        "oeis": "A000225",
        "example_harmonics": "1, 3, 7, 15, 31, ...",
    },
}


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'OEISSequenceID',
    'get_sequence',
    'get_partial_mask',
    'get_partial_amplitudes',
    'get_partial_list',
    'SEQUENCE_DESCRIPTIONS',
]
