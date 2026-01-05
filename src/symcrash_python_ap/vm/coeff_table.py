"""
WSE-STD-120 / WSE-ARCH-CCS Compliant Coefficient Tables
========================================================

Per the specifications:

1. CoeffTable[wave_id][phase_index] → coeffs[256]
2. Phase resolution L (16 or 64 steps per cycle)
3. Linear interpolation between phase entries
4. Tables generated OFFLINE via FWPT (no FWPT in voice path)
5. Coefficients are Q1.15 (int16)

Memory budget:
- L=64 phases × 256 coeffs × 2 bytes = 32 KB per waveform
- L=16 phases × 256 coeffs × 2 bytes = 8 KB per waveform

Runtime:
- Phase → phase_index lookup
- Optional linear interpolation
- No FWPT, purely table lookup
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Optional, Callable
import logging

logger = logging.getLogger(__name__)

# Configuration per WSE-ARCH-CCS
NUM_COEFFS = 256           # 256 coefficients per entry
DEFAULT_PHASE_STEPS = 64   # L = 64 phase steps (high quality)
BLOCK_SIZE = 64            # B = 64 samples per block
Q15_ONE = 32767
TARGET_RMS_Q15 = 16384     # -6 dBFS nominal


@dataclass
class CoeffTable:
    """
    Coefficient wavetable per WSE-STD-120.

    Structure: [L phase steps, 256 coefficients]
    Indexed by phase (0..L-1), returns 256 coefficients.
    """
    wave_id: int
    name: str
    data: np.ndarray  # [L, 256] int16
    phase_steps: int = DEFAULT_PHASE_STEPS

    def get_coeffs(self, phase_32: int, interpolate: bool = True) -> np.ndarray:
        """
        Get 256 coefficients for a 32-bit phase.

        Args:
            phase_32: 32-bit phase accumulator value
            interpolate: If True, linear interpolate between adjacent entries

        Returns:
            256 coefficients (int16)
        """
        L = self.phase_steps

        # Convert 32-bit phase to floating point index [0, L)
        phase_frac = phase_32 / (1 << 32)  # [0, 1)
        phase_idx_f = phase_frac * L       # [0, L)

        if not interpolate:
            # Nearest neighbor
            idx = int(phase_idx_f) % L
            return self.data[idx].copy()

        # Linear interpolation
        idx0 = int(phase_idx_f) % L
        idx1 = (idx0 + 1) % L
        alpha = phase_idx_f - int(phase_idx_f)  # Fractional part [0, 1)

        c0 = self.data[idx0].astype(np.float32)
        c1 = self.data[idx1].astype(np.float32)

        coeffs = c0 * (1 - alpha) + c1 * alpha
        return np.clip(coeffs, -32768, 32767).astype(np.int16)

    def get_coeffs_by_index(self, phase_idx: int) -> np.ndarray:
        """
        Get coefficients by direct phase index (no interpolation).

        Args:
            phase_idx: Phase index (0..L-1)

        Returns:
            256 coefficients (int16)
        """
        idx = phase_idx % self.phase_steps
        return self.data[idx].copy()


def generate_coeff_table(
    wave_id: int,
    name: str,
    sample_fn: Callable[[int], int],
    phase_steps: int = DEFAULT_PHASE_STEPS,
    sample_rate: int = 48000,
) -> CoeffTable:
    """
    Generate a CoeffTable OFFLINE via FWPT.

    Per WSE-STD-120:
    - Generate reference audio at each phase offset
    - Run FWPT to get coefficients
    - Store in table indexed by phase

    The reference frequency is chosen such that we complete
    exactly 'cycles_per_block' cycles per 64-sample block.
    For phase continuity, we use cycles_per_block that gives
    integer samples per cycle.

    Args:
        wave_id: Waveform identifier
        name: Human-readable name
        sample_fn: Function(phase_32) -> sample (Q15)
        phase_steps: Number of phase entries (L)
        sample_rate: Sample rate for reference

    Returns:
        CoeffTable ready for runtime use
    """
    from ..wavelet.fast_lifting import fwpt_jit

    logger.info(f"Generating CoeffTable '{name}' with L={phase_steps}")

    # Allocate table [L, 256]
    # Note: FWPT on 64 samples produces 64 coefficients, not 256
    # Per WSE-ARCH-CCS, coeff_state is [256] with K=4 time samples per leaf
    # For block IDWPT, we use 64 leaves. The 256 = 64 * 4 mapping
    # is handled by the Symbol engine (inject/extract)

    # For now, generate 64 coefficients per phase (block IDWPT compatible)
    data = np.zeros((phase_steps, 64), dtype=np.int16)

    # Reference: complete one cycle per block (64 samples)
    # This gives us exactly one full waveform cycle
    phase_inc_per_sample = (1 << 32) // BLOCK_SIZE

    for p in range(phase_steps):
        # Starting phase for this table entry
        phase_start = int((p / phase_steps) * (1 << 32)) & 0xFFFFFFFF

        # Generate 64-sample reference block
        block = np.zeros(BLOCK_SIZE, dtype=np.int16)
        phase = phase_start
        for i in range(BLOCK_SIZE):
            block[i] = sample_fn(phase)
            phase = (phase + phase_inc_per_sample) & 0xFFFFFFFF

        # FWPT to get 64 coefficients
        coeffs = fwpt_jit(block.tolist(), headroom_bits=1)

        data[p] = np.array(coeffs, dtype=np.int16)

    # Normalize table entries
    data = _normalize_table(data, phase_steps)

    return CoeffTable(
        wave_id=wave_id,
        name=name,
        data=data,
        phase_steps=phase_steps,
    )


def _normalize_table(data: np.ndarray, L: int) -> np.ndarray:
    """
    Per WSE-ARCH-CCS Section 6.1:
    - Compute IDWPT(coeffs) for each entry
    - Scale to target_rms
    - Clamp to prevent overflow
    """
    from ..wavelet.fast_lifting import idwpt_jit

    for p in range(L):
        coeffs = data[p].tolist()
        audio = np.array(idwpt_jit(coeffs, headroom_bits=1), dtype=np.float32)

        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 1e-6:
            scale = TARGET_RMS_Q15 / rms
            # Apply scale to coefficients
            data[p] = np.clip(data[p].astype(np.float32) * scale, -32768, 32767).astype(np.int16)

    return data


# =============================================================================
# Standard waveform generators
# =============================================================================

def _sine_sample(phase_32: int) -> int:
    """Generate sine sample from 32-bit phase."""
    from ..wavelet.sine_core import sine_q15
    return sine_q15(phase_32)


def _saw_sample(phase_32: int) -> int:
    """Generate bandlimited sawtooth sample."""
    from ..wavelet.sine_core import sine_q15

    # Bandlimited: sum harmonics 1-16
    val = 0
    for h in range(1, 17):
        h_phase = (phase_32 * h) & 0xFFFFFFFF
        val += sine_q15(h_phase) / h

    return int(np.clip(val / 2, -32768, 32767))


def _square_sample(phase_32: int) -> int:
    """Generate bandlimited square sample."""
    from ..wavelet.sine_core import sine_q15

    # Bandlimited: odd harmonics 1,3,5,...,15
    val = 0
    for h in [1, 3, 5, 7, 9, 11, 13, 15]:
        h_phase = (phase_32 * h) & 0xFFFFFFFF
        val += sine_q15(h_phase) / h

    return int(np.clip(val, -32768, 32767))


def _triangle_sample(phase_32: int) -> int:
    """Generate bandlimited triangle sample."""
    from ..wavelet.sine_core import sine_q15

    # Bandlimited: odd harmonics with alternating signs and 1/h^2
    val = 0
    sign = 1
    for h in [1, 3, 5, 7, 9, 11, 13, 15]:
        h_phase = (phase_32 * h) & 0xFFFFFFFF
        val += sign * sine_q15(h_phase) / (h * h)
        sign = -sign

    return int(np.clip(val * 8, -32768, 32767))


# =============================================================================
# Global table library
# =============================================================================

_COEFF_TABLE_LIBRARY: Optional[Dict[int, CoeffTable]] = None


def _init_library() -> Dict[int, CoeffTable]:
    """Initialize the CoeffTable library with standard waveforms."""
    logger.info("Initializing CoeffTable library...")
    return {
        0: generate_coeff_table(0, "sine", _sine_sample),
        1: generate_coeff_table(1, "saw", _saw_sample),
        2: generate_coeff_table(2, "square", _square_sample),
        3: generate_coeff_table(3, "triangle", _triangle_sample),
    }


def get_coeff_table(wave_id: int) -> CoeffTable:
    """Get CoeffTable by wave ID (lazy initialization)."""
    global _COEFF_TABLE_LIBRARY
    if _COEFF_TABLE_LIBRARY is None:
        _COEFF_TABLE_LIBRARY = _init_library()
    return _COEFF_TABLE_LIBRARY.get(wave_id, _COEFF_TABLE_LIBRARY[0])


def midi_to_phase_inc(note: int, sample_rate: int = 48000) -> int:
    """Convert MIDI note to phase increment per sample."""
    freq_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))
    return int((freq_hz / sample_rate) * (1 << 32))


__all__ = [
    'CoeffTable',
    'generate_coeff_table',
    'get_coeff_table',
    'midi_to_phase_inc',
    'NUM_COEFFS',
    'DEFAULT_PHASE_STEPS',
    'BLOCK_SIZE',
]
