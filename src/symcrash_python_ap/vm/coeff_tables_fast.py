"""
Fast Coefficient Tables - Minimal Startup Overhead
===================================================

Provides coefficient tables with FAST generation using FWPT directly.
Tables are generated on-demand per note (lazy loading).

This eliminates the multi-minute startup delay of full table generation
while still providing the coefficient-domain carrier architecture.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict
import logging

# Import FWPT for fast coefficient generation
from ..wavelet.fast_lifting import fwpt_jit

# Import sine_core for bit-exact sinusoid generation
from ..wavelet.sine_core import sine_q15

# Q15 constants
Q15_ONE = 32767
Q15_HALF = 16384

logger = logging.getLogger(__name__)


def midi_to_hz(note: int, cents: int = 0) -> float:
    """Convert MIDI note to Hz."""
    semitones = note - 69 + (cents / 100.0)
    return 440.0 * (2.0 ** (semitones / 12.0))


def midi_to_phase_inc(note: int, cents: int = 0, sample_rate: int = 48000) -> int:
    """Convert MIDI note to 32-bit phase increment."""
    freq_hz = midi_to_hz(note, cents)
    # phase_inc = (freq / sample_rate) * 2^32
    return int((freq_hz / sample_rate) * (1 << 32))


def generate_sinusoid_block_fast(
    freq_hz: float,
    phase_start: int,
    num_samples: int = 64,
    sample_rate: int = 48000,
) -> np.ndarray:
    """
    Generate one block of sinusoid using sine_core (bit-exact, fast).

    Args:
        freq_hz: Frequency in Hz
        phase_start: Starting phase (32-bit)
        num_samples: Block size (64)
        sample_rate: Sample rate

    Returns:
        int16 array [64] of audio samples
    """
    # Compute phase increment
    phase_inc = int((freq_hz / sample_rate) * (1 << 32))

    # Generate using sine_core
    audio = np.zeros(num_samples, dtype=np.int16)
    phase = phase_start

    for i in range(num_samples):
        audio[i] = sine_q15(phase)
        phase = (phase + phase_inc) & 0xFFFFFFFF

    return audio


def generate_coeffs_for_note_phase(note: int, phase_idx: int, phase_steps: int = 64) -> tuple:
    """
    Generate sin/cos coefficients for one (note, phase) using FWPT.

    FAST: Uses FWPT directly (no iterative solving).
    Since FWPT/IDWPT are inverses, this gives perfect reconstruction.

    Args:
        note: MIDI note (0-127)
        phase_idx: Phase step (0-63)
        phase_steps: Total phase steps (64)

    Returns:
        (c_sin, c_cos): Tuple of 64-element int16 arrays
    """
    freq_hz = midi_to_hz(note)

    # Compute phase in 32-bit format
    # phase_idx / phase_steps → [0, 1] → [0, 2^32]
    phase_32bit = int((phase_idx / phase_steps) * (1 << 32))

    # Generate reference sin block
    audio_sin = generate_sinusoid_block_fast(freq_hz, phase_32bit)

    # Generate reference cos block (90° phase shift = 0x40000000)
    phase_cos = (phase_32bit + 0x40000000) & 0xFFFFFFFF
    audio_cos = generate_sinusoid_block_fast(freq_hz, phase_cos)

    # Run FWPT to get coefficients (this is the "analytic" solution!)
    c_sin_list = fwpt_jit(audio_sin.tolist(), headroom_bits=1)
    c_cos_list = fwpt_jit(audio_cos.tolist(), headroom_bits=1)

    c_sin = np.array(c_sin_list, dtype=np.int16)
    c_cos = np.array(c_cos_list, dtype=np.int16)

    return c_sin, c_cos


@dataclass
class FastCoeffTable2D:
    """
    Fast coefficient table with on-demand generation.

    Instead of precomputing all 128×64×2 = 16,384 entries at startup,
    we generate them lazily as needed. Cached for subsequent use.
    """
    # Cached entries: {(note, phase_idx): (c_sin, c_cos)}
    cache: Dict[tuple, tuple] = field(default_factory=dict)
    sample_rate: int = 48000
    phase_steps: int = 64

    def get_coeffs(self, note: int, phase_idx: int) -> tuple:
        """
        Get sin/cos coefficients for (note, phase), generating if needed.

        Args:
            note: MIDI note (0-127)
            phase_idx: Phase step (0-63)

        Returns:
            (c_sin, c_cos): Tuple of [64] int16 arrays
        """
        key = (note, phase_idx)

        if key not in self.cache:
            # Generate on demand
            c_sin, c_cos = generate_coeffs_for_note_phase(note, phase_idx, self.phase_steps)
            self.cache[key] = (c_sin, c_cos)

        return self.cache[key]

    @property
    def sin_tables(self):
        """Property for sin table access."""
        return FastCoeffTableView(self, 'sin')

    @property
    def cos_tables(self):
        """Property for cos table access."""
        return FastCoeffTableView(self, 'cos')


@dataclass
class FastCoeffTableView:
    """View wrapper to handle sin_tables[note, phase_idx, :] indexing."""
    parent: FastCoeffTable2D
    table_type: str  # 'sin' or 'cos'

    def __getitem__(self, key):
        """Support table[note, phase_idx, :] indexing."""
        if isinstance(key, tuple) and len(key) == 3:
            note, phase_idx, coeff_slice = key
            c_sin, c_cos = self.parent.get_coeffs(note, phase_idx)

            if self.table_type == 'sin':
                return c_sin
            else:
                return c_cos

        # For partial indexing like table[note]
        return self


# Global fast table (singleton)
_FAST_TABLE: Optional[FastCoeffTable2D] = None


def get_fast_coeff_table() -> FastCoeffTable2D:
    """Get or create the fast coefficient table (singleton)."""
    global _FAST_TABLE

    if _FAST_TABLE is None:
        logger.info("Creating fast coefficient table (on-demand generation)")
        _FAST_TABLE = FastCoeffTable2D()

    return _FAST_TABLE


__all__ = [
    'FastCoeffTable2D',
    'get_fast_coeff_table',
    'generate_coeffs_for_note_phase',
]
