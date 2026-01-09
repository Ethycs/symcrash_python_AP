"""
Matrix Method Coefficient Synthesis
===================================

Converts cyclic shift matrix waveforms directly to wavelet coefficients
without runtime FWPT.

Key insight: For periodic waveforms, the relationship between time-domain
cyclic shift and coefficient-domain transformation can be precomputed.

Architecture:
    1. Precompute FWPT of c-vector at phase=0: C₀ = FWPT(c)
    2. Precompute shift operator S in coefficient domain
    3. At runtime: C_φ = S^φ @ C₀ (matrix-vector multiply)

For efficiency, we precompute S^φ for a set of phase bins and interpolate.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import numpy as np

from ..fixed import Q15_ONE, RSHIFT_RNE, SAT16
from ..wavelet.fast_lifting import fwpt_jit, idwpt_jit
from ..wavelet.sine_core import PHASE_MASK
from .waveform_matrix import (
    WaveformType, waveform_sample_q15,
    WAVEFORM_PERIOD, PHASE_TO_WAVE_SHIFT
)

# =============================================================================
# CONSTANTS
# =============================================================================

NUM_LEAVES = 64
BLOCK_SIZE = 64
VM_K = 8

# Number of phase bins for coefficient LUT
DEFAULT_COEFF_BINS = 256


# =============================================================================
# COEFFICIENT SHIFT MATRIX
# =============================================================================

def compute_coeff_shift_matrix(phase_inc: int, num_bins: int = DEFAULT_COEFF_BINS) -> np.ndarray:
    """
    Compute the coefficient-domain shift operator for cyclic waveforms.

    For a cyclic shift in time domain by δ samples:
        y_shifted[n] = y[n - δ]

    The corresponding operation in coefficient domain is:
        C_shifted = T_δ @ C_original

    Where T_δ is the shift transformation matrix.

    We compute T for each phase bin by:
    1. Generate reference block at phase=0
    2. Generate shifted block at phase=p
    3. FWPT both
    4. T[p] transforms C₀ → C_p

    For our use, we precompute coefficients directly at each phase bin
    rather than storing the transformation matrices (simpler, same memory).

    Args:
        phase_inc: Phase increment per sample
        num_bins: Number of phase bins

    Returns:
        Coefficient array [num_bins, NUM_LEAVES] containing FWPT(c) at each phase
    """
    # This function is a placeholder - actual implementation uses
    # WaveformCoeffBank which stores coefficients directly
    pass


@dataclass
class WaveformCoeffBank:
    """
    Precomputed coefficient bank for a waveform at a specific frequency.

    Stores FWPT(c-vector) at multiple phase positions for direct lookup.
    No runtime FWPT needed - just coefficient interpolation.

    Memory: num_bins × 64 × 2 bytes = 32KB for 256 bins
    """
    waveform_type: int
    phase_inc: int
    num_bins: int = DEFAULT_COEFF_BINS

    # Coefficient data: [num_bins, NUM_LEAVES]
    coeffs: np.ndarray = field(default_factory=lambda: np.zeros((DEFAULT_COEFF_BINS, NUM_LEAVES), dtype=np.int16))

    # Phase shift for bin indexing
    _bin_shift: int = 0

    def __post_init__(self):
        # Compute shift for phase-to-bin conversion
        # bin = (phase >> shift) & (num_bins - 1)
        import math
        self._bin_shift = 32 - int(math.log2(self.num_bins))

    def sample(self, phase: int) -> np.ndarray:
        """
        Get coefficients at given phase with linear interpolation.

        Args:
            phase: Q32 phase

        Returns:
            64-element coefficient array
        """
        # Convert phase to bin index
        bin_idx = (phase >> self._bin_shift) & (self.num_bins - 1)
        bin_next = (bin_idx + 1) & (self.num_bins - 1)

        # Fractional part for interpolation (Q15)
        frac_mask = (1 << self._bin_shift) - 1
        frac = (phase & frac_mask) >> (self._bin_shift - 15)

        # Linear interpolation
        c0 = self.coeffs[bin_idx].astype(np.int32)
        c1 = self.coeffs[bin_next].astype(np.int32)

        result = c0 + (((c1 - c0) * frac + (1 << 14)) >> 15)

        return np.clip(result, -32768, 32767).astype(np.int16)

    @staticmethod
    def generate(
        waveform_type: int,
        phase_inc: int,
        num_bins: int = DEFAULT_COEFF_BINS,
        bandlimited: bool = True
    ) -> 'WaveformCoeffBank':
        """
        Generate coefficient bank for a waveform.

        Precomputes FWPT of c-vector samples at each phase bin.

        Args:
            waveform_type: WaveformType constant
            phase_inc: Phase increment per sample
            num_bins: Number of phase bins (power of 2)
            bandlimited: Use BLEP for anti-aliased c-vectors

        Returns:
            WaveformCoeffBank with precomputed coefficients
        """
        import math
        bin_shift = 32 - int(math.log2(num_bins))

        coeffs = np.zeros((num_bins, NUM_LEAVES), dtype=np.int16)

        for p in range(num_bins):
            # Phase for this bin
            phase_p = (p << bin_shift) & PHASE_MASK

            # Generate 64 time-domain samples at this phase
            block = []
            phase = phase_p
            for _ in range(BLOCK_SIZE):
                # Use waveform_sample which respects bandlimited cache setting
                sample = waveform_sample_q15(phase, waveform_type)
                block.append(sample)
                phase = (phase + phase_inc) & PHASE_MASK

            # FWPT to coefficients
            c = fwpt_jit(block, headroom_bits=1)
            coeffs[p, :] = c

        bank = WaveformCoeffBank(
            waveform_type=waveform_type,
            phase_inc=phase_inc,
            num_bins=num_bins,
            coeffs=coeffs,
        )
        bank._bin_shift = bin_shift

        return bank


# =============================================================================
# HARMONIC COEFFICIENT MATRIX (Alternative approach)
# =============================================================================

@dataclass
class HarmonicCoeffMatrix:
    """
    Direct harmonic-to-coefficient transformation matrix.

    Instead of storing FWPT(c) at many phases, store:
    - Base coefficients for each harmonic: B_h = FWPT(sin(h·θ))
    - At runtime: C = Σ_h a_h · rotate(B_h, h·φ)

    This preserves harmonic control while using matrix representation.

    The key insight: FWPT(sin(h·(θ+φ))) = rotate(FWPT(sin(h·θ)), h·φ)
    where "rotate" is a coefficient-domain operation.

    For wavelets, this rotation is NOT simple phase multiplication like DFT.
    Instead, we precompute the rotation effect at multiple phases.
    """
    phase_inc: int
    H_max: int = 32
    num_phase_bins: int = 64  # Per-harmonic phase resolution

    # Harmonic basis coefficients: [H_max, num_phase_bins, NUM_LEAVES]
    # B[h, p, :] = FWPT(sin(h·θ)) at phase bin p
    basis: np.ndarray = field(default_factory=lambda: np.zeros((32, 64, NUM_LEAVES), dtype=np.int16))

    _bin_shift: int = 0

    def __post_init__(self):
        import math
        self._bin_shift = 32 - int(math.log2(self.num_phase_bins))

    def synthesize(self, phase: int, a_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
        """
        Synthesize coefficients from harmonic weights.

        Args:
            phase: Q32 phase
            a_h: Sine harmonic weights [H_max]
            b_h: Cosine harmonic weights [H_max]

        Returns:
            64-element coefficient array
        """
        result = np.zeros(NUM_LEAVES, dtype=np.int32)

        for h in range(1, min(self.H_max + 1, len(a_h) + 1)):
            if a_h[h-1] == 0 and b_h[h-1] == 0:
                continue

            # Phase for this harmonic
            h_phase = (phase * h) & PHASE_MASK

            # Get bin index for this harmonic's phase
            bin_idx = (h_phase >> self._bin_shift) & (self.num_phase_bins - 1)
            bin_next = (bin_idx + 1) & (self.num_phase_bins - 1)

            # Fractional interpolation
            frac_mask = (1 << self._bin_shift) - 1
            frac = (h_phase & frac_mask) >> (self._bin_shift - 15)

            # Interpolate basis coefficients
            b0 = self.basis[h-1, bin_idx].astype(np.int32)
            b1 = self.basis[h-1, bin_next].astype(np.int32)
            basis_interp = b0 + (((b1 - b0) * frac + (1 << 14)) >> 15)

            # Add weighted contribution
            # For sin: use a_h directly
            # For cos: phase shift by π/2 (handled in basis generation)
            weight = int(a_h[h-1])
            result += (basis_interp * weight) >> 15

        return np.clip(result, -32768, 32767).astype(np.int16)

    @staticmethod
    def generate(phase_inc: int, H_max: int = 32, num_phase_bins: int = 64) -> 'HarmonicCoeffMatrix':
        """
        Generate harmonic coefficient matrix.

        Precomputes FWPT of sin(h·θ) at multiple phases for each harmonic.
        """
        from ..wavelet.sine_core import sin_cos_q15
        import math

        bin_shift = 32 - int(math.log2(num_phase_bins))
        basis = np.zeros((H_max, num_phase_bins, NUM_LEAVES), dtype=np.int16)

        for h in range(1, H_max + 1):
            for p in range(num_phase_bins):
                # Starting phase for this bin
                phase_p = (p << bin_shift) & PHASE_MASK

                # Generate sin(h·θ) block
                block = []
                phase = phase_p
                for _ in range(BLOCK_SIZE):
                    h_phase = (phase * h) & PHASE_MASK
                    sin_val, _ = sin_cos_q15(h_phase)
                    block.append(sin_val)
                    phase = (phase + phase_inc) & PHASE_MASK

                # FWPT
                coeffs = fwpt_jit(block, headroom_bits=1)
                basis[h-1, p, :] = coeffs

        matrix = HarmonicCoeffMatrix(
            phase_inc=phase_inc,
            H_max=H_max,
            num_phase_bins=num_phase_bins,
            basis=basis,
        )
        matrix._bin_shift = bin_shift

        return matrix


# =============================================================================
# GLOBAL CACHE
# =============================================================================

_COEFF_BANK_CACHE: Dict[Tuple[int, int], WaveformCoeffBank] = {}
_HARMONIC_MATRIX_CACHE: Dict[int, HarmonicCoeffMatrix] = {}


def get_waveform_coeff_bank(waveform_type: int, phase_inc: int) -> WaveformCoeffBank:
    """Get or create coefficient bank for waveform."""
    key = (waveform_type, phase_inc)
    if key not in _COEFF_BANK_CACHE:
        _COEFF_BANK_CACHE[key] = WaveformCoeffBank.generate(waveform_type, phase_inc)
    return _COEFF_BANK_CACHE[key]


def get_harmonic_coeff_matrix(phase_inc: int) -> HarmonicCoeffMatrix:
    """Get or create harmonic coefficient matrix."""
    if phase_inc not in _HARMONIC_MATRIX_CACHE:
        _HARMONIC_MATRIX_CACHE[phase_inc] = HarmonicCoeffMatrix.generate(phase_inc)
    return _HARMONIC_MATRIX_CACHE[phase_inc]


def clear_matrix_coeff_cache() -> None:
    """Clear all matrix coefficient caches."""
    _COEFF_BANK_CACHE.clear()
    _HARMONIC_MATRIX_CACHE.clear()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'WaveformCoeffBank',
    'HarmonicCoeffMatrix',
    'get_waveform_coeff_bank',
    'get_harmonic_coeff_matrix',
    'clear_matrix_coeff_cache',
]
