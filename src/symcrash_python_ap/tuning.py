"""
Instrument Tuning System
========================

Provides 64-leaf gain tables that define instrument character.

Tunings are slow-morphing global settings (1-5s transitions) shared by
all voices. They represent instrument character like guitar pickup
positions or filter settings, NOT per-voice topology swaps.

Key concepts:
- Each tuning = 64 Q15 gains (one per master leaf)
- Tunings morph slowly (1-5 seconds) for smooth timbral transitions
- All voices share the same tuning morph state
- Trees become "tuning presets" not per-voice topologies

Target: ECP5-25 FPGA @ 48kHz, N=64 block size
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import math

from .wavelet.carrier_lift import (
    InstrumentTuning, TuningMorphState,
    NUM_LEAVES, Q15_ONE, Q15_HALF, BLOCKS_PER_SECOND
)


# =============================================================================
# TUNING PRESETS
# =============================================================================

def _generate_harmonic_tuning() -> List[int]:
    """
    Generate harmonic-emphasizing tuning.

    Emphasizes leaves that correspond to harmonic frequencies.
    Lower leaves (fundamental region) get more gain.
    """
    gains = []
    for i in range(NUM_LEAVES):
        # Exponential falloff: 0.9^(i/8)
        gain = int(Q15_ONE * (0.9 ** (i / 8)))
        gains.append(max(1000, gain))  # Minimum floor
    return gains


def _generate_bright_tuning() -> List[int]:
    """
    Generate bright tuning (tilt toward high frequencies).

    Linear tilt from 0.5 at leaf 0 to 1.0 at leaf 63.
    """
    gains = []
    for i in range(NUM_LEAVES):
        tilt = 0.5 + 0.5 * (i / (NUM_LEAVES - 1))
        gains.append(int(Q15_ONE * tilt))
    return gains


def _generate_dark_tuning() -> List[int]:
    """
    Generate dark tuning (tilt toward low frequencies).

    Inverse of bright: 1.0 at leaf 0 to 0.3 at leaf 63.
    """
    gains = []
    for i in range(NUM_LEAVES):
        tilt = 1.0 - 0.7 * (i / (NUM_LEAVES - 1))
        gains.append(int(Q15_ONE * tilt))
    return gains


def _generate_hollow_tuning() -> List[int]:
    """
    Generate hollow tuning (odd harmonics emphasized).

    Reduces even-indexed leaves, boosts odd-indexed leaves.
    Creates clarinet-like hollow quality.
    """
    gains = []
    for i in range(NUM_LEAVES):
        if i % 2 == 0:
            gains.append(int(Q15_ONE * 0.5))  # Even: reduced
        else:
            gains.append(Q15_ONE)  # Odd: full
    return gains


def _generate_nasal_tuning() -> List[int]:
    """
    Generate nasal tuning (mid-frequency emphasis).

    Bell curve centered around leaf 24 (~9kHz).
    """
    gains = []
    center = 24
    sigma = 12
    for i in range(NUM_LEAVES):
        diff = i - center
        weight = math.exp(-(diff * diff) / (2 * sigma * sigma))
        # Mix with baseline
        gain = 0.3 + 0.7 * weight
        gains.append(int(Q15_ONE * gain))
    return gains


def _generate_formant_tuning(formant_freqs: List[float]) -> List[int]:
    """
    Generate formant tuning from formant frequencies.

    Creates peaks at specified frequencies with resonance.

    Args:
        formant_freqs: List of formant frequencies in Hz
    """
    gains = [int(Q15_ONE * 0.2)] * NUM_LEAVES  # Baseline

    # Each leaf covers ~375 Hz (24000/64)
    leaf_bandwidth = 24000 / NUM_LEAVES

    for freq in formant_freqs:
        center_leaf = int(freq / leaf_bandwidth)
        sigma = 3  # ~3 leaves = ~1100 Hz bandwidth

        for i in range(NUM_LEAVES):
            diff = i - center_leaf
            weight = math.exp(-(diff * diff) / (2 * sigma * sigma))
            contribution = int(Q15_ONE * 0.8 * weight)
            gains[i] = min(Q15_ONE, gains[i] + contribution)

    return gains


def _generate_vowel_a_tuning() -> List[int]:
    """Vowel 'A' formants (approx)."""
    return _generate_formant_tuning([700, 1200, 2600])


def _generate_vowel_e_tuning() -> List[int]:
    """Vowel 'E' formants (approx)."""
    return _generate_formant_tuning([400, 2000, 2600])


def _generate_vowel_i_tuning() -> List[int]:
    """Vowel 'I' formants (approx)."""
    return _generate_formant_tuning([300, 2200, 3000])


def _generate_vowel_o_tuning() -> List[int]:
    """Vowel 'O' formants (approx)."""
    return _generate_formant_tuning([500, 800, 2600])


def _generate_vowel_u_tuning() -> List[int]:
    """Vowel 'U' formants (approx)."""
    return _generate_formant_tuning([350, 700, 2600])


# =============================================================================
# CANONICAL TUNING PRESETS
# =============================================================================

# Flat (unity gain)
TUNING_FLAT = InstrumentTuning(
    name="flat",
    gains_q15=[Q15_ONE] * NUM_LEAVES,
    description="Unity gain across all leaves"
)

# Harmonic emphasis
TUNING_HARMONIC = InstrumentTuning(
    name="harmonic",
    gains_q15=_generate_harmonic_tuning(),
    description="Emphasizes fundamental and low harmonics"
)

# Bright tilt
TUNING_BRIGHT = InstrumentTuning(
    name="bright",
    gains_q15=_generate_bright_tuning(),
    description="Tilted toward high frequencies"
)

# Dark tilt
TUNING_DARK = InstrumentTuning(
    name="dark",
    gains_q15=_generate_dark_tuning(),
    description="Tilted toward low frequencies"
)

# Hollow (odd harmonics)
TUNING_HOLLOW = InstrumentTuning(
    name="hollow",
    gains_q15=_generate_hollow_tuning(),
    description="Odd harmonic emphasis (clarinet-like)"
)

# Nasal (mid emphasis)
TUNING_NASAL = InstrumentTuning(
    name="nasal",
    gains_q15=_generate_nasal_tuning(),
    description="Mid-frequency emphasis"
)

# Vowel formants
TUNING_VOWEL_A = InstrumentTuning(
    name="vowel_a",
    gains_q15=_generate_vowel_a_tuning(),
    description="Vowel 'A' formant structure"
)

TUNING_VOWEL_E = InstrumentTuning(
    name="vowel_e",
    gains_q15=_generate_vowel_e_tuning(),
    description="Vowel 'E' formant structure"
)

TUNING_VOWEL_I = InstrumentTuning(
    name="vowel_i",
    gains_q15=_generate_vowel_i_tuning(),
    description="Vowel 'I' formant structure"
)

TUNING_VOWEL_O = InstrumentTuning(
    name="vowel_o",
    gains_q15=_generate_vowel_o_tuning(),
    description="Vowel 'O' formant structure"
)

TUNING_VOWEL_U = InstrumentTuning(
    name="vowel_u",
    gains_q15=_generate_vowel_u_tuning(),
    description="Vowel 'U' formant structure"
)


# All presets by name
TUNING_PRESETS: Dict[str, InstrumentTuning] = {
    "flat": TUNING_FLAT,
    "harmonic": TUNING_HARMONIC,
    "bright": TUNING_BRIGHT,
    "dark": TUNING_DARK,
    "hollow": TUNING_HOLLOW,
    "nasal": TUNING_NASAL,
    "vowel_a": TUNING_VOWEL_A,
    "vowel_e": TUNING_VOWEL_E,
    "vowel_i": TUNING_VOWEL_I,
    "vowel_o": TUNING_VOWEL_O,
    "vowel_u": TUNING_VOWEL_U,
}


def get_tuning(name: str) -> InstrumentTuning:
    """
    Get tuning preset by name.

    Args:
        name: Tuning name (e.g., "harmonic", "bright")

    Returns:
        InstrumentTuning instance

    Raises:
        KeyError if name not found
    """
    return TUNING_PRESETS[name]


def list_tunings() -> List[str]:
    """Get list of available tuning preset names."""
    return list(TUNING_PRESETS.keys())


# =============================================================================
# TUNING MORPHER (Global Controller)
# =============================================================================

@dataclass
class TuningMorpher:
    """
    Global instrument tuning morph controller.

    Manages slow transitions between tunings (1-5 seconds).
    All voices share the same current tuning gains.

    Usage:
        morpher = TuningMorpher()
        morpher.set_target(TUNING_BRIGHT, morph_time_sec=2.0)

        # Each block:
        morpher.advance_block()
        gains = morpher.get_current_gains()
    """
    # Internal morph state
    _state: TuningMorphState = field(default_factory=TuningMorphState)

    # Default morph time in seconds
    default_morph_time_sec: float = 2.0

    def __post_init__(self):
        """Initialize with flat tuning."""
        self._state.tuning_a = TUNING_FLAT
        self._state.tuning_b = TUNING_FLAT

    def set_tuning_immediate(self, tuning: InstrumentTuning) -> None:
        """
        Set tuning immediately without morphing.

        Args:
            tuning: Target tuning to apply immediately
        """
        self._state.tuning_a = tuning
        self._state.tuning_b = tuning
        self._state.mu_q15 = 0
        self._state.mu_target_q15 = 0

    def set_target(self, tuning: InstrumentTuning,
                   morph_time_sec: Optional[float] = None) -> None:
        """
        Start morphing to a new tuning.

        Args:
            tuning: Target instrument tuning
            morph_time_sec: Transition time (uses default if None)
        """
        if morph_time_sec is None:
            morph_time_sec = self.default_morph_time_sec

        self._state.set_target_tuning(tuning, morph_time_sec)

    def set_target_by_name(self, name: str,
                           morph_time_sec: Optional[float] = None) -> None:
        """
        Start morphing to a tuning by name.

        Args:
            name: Tuning preset name
            morph_time_sec: Transition time
        """
        tuning = get_tuning(name)
        self.set_target(tuning, morph_time_sec)

    def advance_block(self) -> None:
        """
        Advance morph state by one block.

        Call this once per block to update the interpolation.
        """
        self._state.advance_block()

    def get_current_gains(self) -> List[int]:
        """
        Get interpolated 64-gain table for this block.

        Returns:
            List of 64 Q15 gains
        """
        return self._state.get_current_gains()

    @property
    def is_morphing(self) -> bool:
        """True if currently transitioning between tunings."""
        return self._state.is_morphing

    @property
    def current_tuning_name(self) -> str:
        """Name of current/target tuning."""
        if self._state.mu_q15 == 0:
            return self._state.tuning_a.name
        elif self._state.mu_q15 >= Q15_ONE:
            return self._state.tuning_b.name
        else:
            return f"{self._state.tuning_a.name}->{self._state.tuning_b.name}"

    @property
    def morph_progress(self) -> float:
        """Current morph progress (0.0 to 1.0)."""
        return self._state.mu_q15 / Q15_ONE


# =============================================================================
# TUNING INTERPOLATION UTILITIES
# =============================================================================

def interpolate_tunings(tuning_a: InstrumentTuning,
                        tuning_b: InstrumentTuning,
                        mu: float) -> List[int]:
    """
    Interpolate between two tunings.

    Args:
        tuning_a: First tuning
        tuning_b: Second tuning
        mu: Interpolation factor (0.0 = all A, 1.0 = all B)

    Returns:
        List of 64 Q15 gains
    """
    gains = []
    for i in range(NUM_LEAVES):
        gain_a = tuning_a.gains_q15[i] if i < len(tuning_a.gains_q15) else Q15_ONE
        gain_b = tuning_b.gains_q15[i] if i < len(tuning_b.gains_q15) else Q15_ONE
        interpolated = int(gain_a * (1 - mu) + gain_b * mu)
        gains.append(interpolated)
    return gains


def blend_tunings(tunings: List[InstrumentTuning],
                  weights: List[float]) -> List[int]:
    """
    Blend multiple tunings with weights.

    Args:
        tunings: List of tunings to blend
        weights: Weights for each tuning (should sum to 1.0)

    Returns:
        List of 64 Q15 gains
    """
    if not tunings:
        return [Q15_ONE] * NUM_LEAVES

    # Normalize weights
    total = sum(weights)
    if total <= 0:
        weights = [1.0 / len(tunings)] * len(tunings)
    else:
        weights = [w / total for w in weights]

    gains = [0] * NUM_LEAVES

    for tuning, weight in zip(tunings, weights):
        for i in range(NUM_LEAVES):
            gain = tuning.gains_q15[i] if i < len(tuning.gains_q15) else Q15_ONE
            gains[i] += int(gain * weight)

    return gains


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Preset tunings
    'TUNING_FLAT',
    'TUNING_HARMONIC',
    'TUNING_BRIGHT',
    'TUNING_DARK',
    'TUNING_HOLLOW',
    'TUNING_NASAL',
    'TUNING_VOWEL_A',
    'TUNING_VOWEL_E',
    'TUNING_VOWEL_I',
    'TUNING_VOWEL_O',
    'TUNING_VOWEL_U',
    'TUNING_PRESETS',
    # Access functions
    'get_tuning',
    'list_tunings',
    # Morpher
    'TuningMorpher',
    # Utilities
    'interpolate_tunings',
    'blend_tunings',
]
