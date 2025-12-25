"""
Leaf Content Generator
======================

Generates 64-leaf content blocks from voice partials using soft assignment.

Each partial (harmonic) contributes to multiple leaves via Gaussian-weighted
distribution. This creates smooth spectral content that the IDWPT lift can
then shape with instrument tuning.

Key concepts:
- Partials: Voice harmonics with frequencies and amplitudes
- Leaves: 64 master wavelet packet leaves (D=6 tree)
- Soft assignment: Each partial spreads across 2-4 nearby leaves

Target: ECP5-25 FPGA @ 48kHz, N=64 block size
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import math

from .carrier_lift import (
    LeafContent, NUM_LEAVES, Q15_ONE, Q15_HALF, Q15_SCALE,
    SAMPLE_RATE, BLOCK_SIZE
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Frequency range for the 64 leaves
# Nyquist at 48kHz is 24kHz, leaves cover 0-24kHz
NYQUIST_HZ = SAMPLE_RATE // 2  # 24000 Hz

# Each leaf covers approximately NYQUIST / NUM_LEAVES = 375 Hz bandwidth
LEAF_BANDWIDTH_HZ = NYQUIST_HZ / NUM_LEAVES

# Default Gaussian sigma for partial spread (in Hz)
DEFAULT_SIGMA_HZ = 150.0


# =============================================================================
# LEAF FREQUENCY MAPPING
# =============================================================================

def compute_leaf_center_frequencies() -> List[float]:
    """
    Compute center frequency for each of the 64 leaves.

    Leaves are linearly spaced from 0 to Nyquist.
    Leaf k covers approximately [(k-0.5)*BW, (k+0.5)*BW] Hz.

    Returns:
        List of 64 center frequencies in Hz
    """
    centers = []
    for k in range(NUM_LEAVES):
        # Center of leaf k
        center = (k + 0.5) * LEAF_BANDWIDTH_HZ
        centers.append(center)
    return centers


# Precomputed leaf centers
LEAF_CENTERS_HZ = compute_leaf_center_frequencies()


def frequency_to_leaf_index(freq_hz: float) -> int:
    """
    Map frequency to nearest leaf index.

    Args:
        freq_hz: Frequency in Hz

    Returns:
        Leaf index (0-63), clamped to valid range
    """
    if freq_hz <= 0:
        return 0
    if freq_hz >= NYQUIST_HZ:
        return NUM_LEAVES - 1

    # Linear mapping
    index = int(freq_hz / LEAF_BANDWIDTH_HZ)
    return max(0, min(NUM_LEAVES - 1, index))


def leaf_index_to_frequency(index: int) -> float:
    """
    Get center frequency for a leaf index.

    Args:
        index: Leaf index (0-63)

    Returns:
        Center frequency in Hz
    """
    index = max(0, min(NUM_LEAVES - 1, index))
    return LEAF_CENTERS_HZ[index]


# =============================================================================
# GAUSSIAN WEIGHT COMPUTATION
# =============================================================================

def gaussian_weight(freq_hz: float, center_hz: float, sigma_hz: float) -> float:
    """
    Compute Gaussian weight for partial-to-leaf assignment.

    Args:
        freq_hz: Partial frequency
        center_hz: Leaf center frequency
        sigma_hz: Gaussian spread (standard deviation)

    Returns:
        Weight in range [0, 1]
    """
    if sigma_hz <= 0:
        return 1.0 if abs(freq_hz - center_hz) < 1.0 else 0.0

    diff = freq_hz - center_hz
    return math.exp(-(diff * diff) / (2 * sigma_hz * sigma_hz))


def compute_leaf_weights(freq_hz: float, sigma_hz: float = DEFAULT_SIGMA_HZ,
                         threshold: float = 0.01) -> List[Tuple[int, float]]:
    """
    Compute (leaf_index, weight) pairs for a partial frequency.

    Only returns leaves with weight above threshold.

    Args:
        freq_hz: Partial frequency in Hz
        sigma_hz: Gaussian spread
        threshold: Minimum weight to include

    Returns:
        List of (leaf_index, weight) tuples, sorted by weight descending
    """
    weights = []

    for k in range(NUM_LEAVES):
        center = LEAF_CENTERS_HZ[k]
        w = gaussian_weight(freq_hz, center, sigma_hz)
        if w >= threshold:
            weights.append((k, w))

    # Sort by weight descending
    weights.sort(key=lambda x: -x[1])

    return weights


# =============================================================================
# PARTIAL REPRESENTATION
# =============================================================================

@dataclass
class Partial:
    """
    A single partial (harmonic component) of a voice.

    Partials are the additive synthesis building blocks that get
    distributed across leaves.
    """
    frequency_hz: float = 440.0     # Partial frequency
    amplitude_q15: int = Q15_ONE    # Amplitude (0-32767)
    phase_q15: int = 0              # Phase (optional, for coherent synthesis)
    harmonic_number: int = 1        # Which harmonic (1 = fundamental)

    @classmethod
    def from_fundamental(cls, f0_hz: float, harmonic: int,
                         amplitude_q15: int = Q15_ONE) -> 'Partial':
        """Create partial from fundamental frequency and harmonic number."""
        return cls(
            frequency_hz=f0_hz * harmonic,
            amplitude_q15=amplitude_q15,
            harmonic_number=harmonic
        )


# =============================================================================
# LEAF GENERATOR
# =============================================================================

@dataclass
class LeafGenerator:
    """
    Generates 64-leaf content from voice partials.

    Uses soft Gaussian assignment so each partial contributes to
    multiple nearby leaves, creating smooth spectral content.
    """
    # Gaussian spread for partial assignment
    sigma_hz: float = DEFAULT_SIGMA_HZ

    # Minimum weight threshold
    weight_threshold: float = 0.01

    # Maximum leaves per partial (for efficiency)
    max_leaves_per_partial: int = 4

    # Per-voice state (caches for efficiency)
    _weight_cache: Dict[int, List[List[Tuple[int, float]]]] = field(
        default_factory=dict, repr=False
    )

    def generate_from_partials(self, partials: List[Partial]) -> LeafContent:
        """
        Generate leaf content from a list of partials.

        Each partial's amplitude is distributed across nearby leaves
        using Gaussian weights. The weights are normalized so total
        energy is preserved.

        Args:
            partials: List of Partial objects

        Returns:
            LeafContent with 64 Q15 coefficients
        """
        content = LeafContent()

        for partial in partials:
            if partial.amplitude_q15 == 0:
                continue

            # Get leaf weights for this frequency
            weights = compute_leaf_weights(
                partial.frequency_hz,
                self.sigma_hz,
                self.weight_threshold
            )

            # Limit to max leaves
            weights = weights[:self.max_leaves_per_partial]

            if not weights:
                continue

            # Normalize weights
            total_weight = sum(w for _, w in weights)
            if total_weight <= 0:
                continue

            # Distribute amplitude across leaves
            for leaf_idx, weight in weights:
                # Normalized weight
                norm_weight = weight / total_weight

                # Amplitude contribution in Q15
                contribution = int(partial.amplitude_q15 * norm_weight)

                # Add to leaf (with saturation)
                content.coeffs_q15[leaf_idx] = min(
                    Q15_ONE,
                    content.coeffs_q15[leaf_idx] + contribution
                )

        return content

    def generate_from_fundamental(self, f0_hz: float,
                                   harmonic_amps_q15: List[int]) -> LeafContent:
        """
        Generate leaf content from fundamental frequency and harmonic amplitudes.

        This is the common case: a voice has a fundamental and we specify
        the amplitude of each harmonic.

        Args:
            f0_hz: Fundamental frequency in Hz
            harmonic_amps_q15: Q15 amplitudes for harmonics 1, 2, 3, ...

        Returns:
            LeafContent with 64 Q15 coefficients
        """
        partials = []

        for n, amp in enumerate(harmonic_amps_q15, start=1):
            if amp > 0:
                freq = f0_hz * n

                # Skip partials above Nyquist
                if freq >= NYQUIST_HZ:
                    break

                partials.append(Partial(
                    frequency_hz=freq,
                    amplitude_q15=amp,
                    harmonic_number=n
                ))

        return self.generate_from_partials(partials)

    def generate_from_midi(self, note: int, cents: int = 0,
                           harmonic_amps_q15: Optional[List[int]] = None) -> LeafContent:
        """
        Generate leaf content from MIDI note.

        Args:
            note: MIDI note number (0-127)
            cents: Fine tuning in cents
            harmonic_amps_q15: Optional harmonic amplitudes (defaults to saw-like)

        Returns:
            LeafContent with 64 Q15 coefficients
        """
        # Convert MIDI to frequency
        f0_hz = 440.0 * (2.0 ** ((note - 69 + cents / 100.0) / 12.0))

        # Default harmonic amplitudes (saw-like 1/n falloff)
        if harmonic_amps_q15 is None:
            harmonic_amps_q15 = [Q15_ONE // n for n in range(1, 17)]

        return self.generate_from_fundamental(f0_hz, harmonic_amps_q15)


# =============================================================================
# PRESET HARMONIC PROFILES
# =============================================================================

def saw_harmonics(num_harmonics: int = 16) -> List[int]:
    """Saw wave harmonics: 1/n amplitude falloff."""
    return [Q15_ONE // n for n in range(1, num_harmonics + 1)]


def square_harmonics(num_harmonics: int = 16) -> List[int]:
    """Square wave harmonics: odd only, 1/n falloff."""
    return [Q15_ONE // n if n % 2 == 1 else 0 for n in range(1, num_harmonics + 1)]


def triangle_harmonics(num_harmonics: int = 16) -> List[int]:
    """Triangle wave harmonics: odd only, 1/n^2 falloff."""
    return [Q15_ONE // (n * n) if n % 2 == 1 else 0 for n in range(1, num_harmonics + 1)]


def sine_harmonics() -> List[int]:
    """Pure sine (fundamental only)."""
    return [Q15_ONE] + [0] * 15


def organ_harmonics() -> List[int]:
    """Organ-like harmonics (drawbar style)."""
    # 16', 8', 5-1/3', 4', 2-2/3', 2', 1-3/5', 1-1/3', 1'
    # Harmonics: 1, 2, 3, 4, 5, 6, 8
    return [
        Q15_ONE,        # 1st (16')
        int(Q15_ONE * 0.8),  # 2nd (8')
        int(Q15_ONE * 0.6),  # 3rd (5-1/3')
        int(Q15_ONE * 0.7),  # 4th (4')
        int(Q15_ONE * 0.3),  # 5th (2-2/3')
        int(Q15_ONE * 0.5),  # 6th (2')
        0,              # 7th (not used)
        int(Q15_ONE * 0.4),  # 8th (1')
    ] + [0] * 8


def brass_harmonics() -> List[int]:
    """Brass-like harmonics."""
    return [
        Q15_ONE,
        int(Q15_ONE * 0.9),
        int(Q15_ONE * 0.7),
        int(Q15_ONE * 0.5),
        int(Q15_ONE * 0.4),
        int(Q15_ONE * 0.3),
        int(Q15_ONE * 0.2),
        int(Q15_ONE * 0.15),
        int(Q15_ONE * 0.1),
        int(Q15_ONE * 0.08),
        int(Q15_ONE * 0.05),
        int(Q15_ONE * 0.03),
    ] + [0] * 4


def string_harmonics() -> List[int]:
    """Bowed string-like harmonics."""
    return [
        Q15_ONE,
        int(Q15_ONE * 0.5),
        int(Q15_ONE * 0.33),
        int(Q15_ONE * 0.25),
        int(Q15_ONE * 0.15),
        int(Q15_ONE * 0.12),
        int(Q15_ONE * 0.08),
        int(Q15_ONE * 0.06),
    ] + [0] * 8


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'NYQUIST_HZ',
    'LEAF_BANDWIDTH_HZ',
    'LEAF_CENTERS_HZ',
    'DEFAULT_SIGMA_HZ',
    # Frequency mapping
    'compute_leaf_center_frequencies',
    'frequency_to_leaf_index',
    'leaf_index_to_frequency',
    # Weights
    'gaussian_weight',
    'compute_leaf_weights',
    # Partial
    'Partial',
    # Generator
    'LeafGenerator',
    # Presets
    'saw_harmonics',
    'square_harmonics',
    'triangle_harmonics',
    'sine_harmonics',
    'organ_harmonics',
    'brass_harmonics',
    'string_harmonics',
]
