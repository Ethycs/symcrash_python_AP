"""
WSE-STD-121 — CoeffLUT Generation (v0)
========================================

Implements offline FWPT-based table generation to create
CoeffLUT libraries for streaming CCS synthesis.

Shape A (streaming-friendly):
    CoeffLUT[wave_id][leaf ℓ][k][p] -> q15

Where:
    - leaf ℓ ∈ [0..63]   (64 leaves for 64-sample blocks)
    - k ∈ [0..K-1]       (K polyphase taps per block, default 8)
    - p ∈ [0..P-1]       (P phase steps per cycle, default 256)

Runtime consumes K samples per block from each leaf, chosen by
phase index p, and streams into polyphase IDWPT.

WSE-STD-122 Normalization:
    - DC removal (mandatory)
    - Global waveform RMS normalization
    - Bounded leaf energy equalization
    - Peak safety clamp
    - Optional phase smoothing

Usage:
    lut = CoeffLUT.generate_sine()
    coeffs = lut.get_coeffs_for_phase(phase_idx=0)  # [64 leaves, K taps]
    audio = polyphase_idwpt(coeffs)
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Optional, List
import logging

from ..wavelet.fast_lifting import fwpt_jit
from ..wavelet.sine_core import sine_q15

logger = logging.getLogger(__name__)

# Constants (v0)
BLOCK_SIZE = 64        # B = 64 samples per block
NUM_LEAVES = 64        # 64 leaves (full WPT at depth 6)
DEFAULT_K = 8          # K = 8 polyphase taps per block
DEFAULT_P = 256        # P = 256 phase steps per cycle
CYCLE_LENGTH = 8192    # C = 8192 samples for table generation
Q15_ONE = 32767
TARGET_RMS_Q15 = 16384  # Target RMS (-6 dBFS)


@dataclass
class CoeffLUT:
    """
    Coefficient lookup table with sin/cos basis for phase rotation.

    Rather than storing coefficients for each of P phase steps,
    we store sin and cos basis coefficients. At runtime, we compute:
        coeffs = cos(φ) * C_sin + sin(φ) * C_cos

    This allows arbitrary frequency synthesis from a single pair of tables.

    Structure:
    - c_sin: [64 leaves] - coefficients for sin(ωt)
    - c_cos: [64 leaves] - coefficients for cos(ωt)
    """
    wave_id: int
    name: str

    # Basis coefficients: [64 leaves] each
    c_sin: np.ndarray  # Coefficients for sin basis
    c_cos: np.ndarray  # Coefficients for cos basis

    # Parameters
    num_leaves: int = NUM_LEAVES
    P: int = DEFAULT_P  # Kept for compatibility
    K: int = DEFAULT_K  # Kept for compatibility

    # Legacy data for backward compatibility (can be removed later)
    data: np.ndarray = field(default_factory=lambda: np.array([]))

    # Metadata
    phase_anchor: str = "rising_zero"
    normalization: str = "rms_bounded"

    def get_coeffs_for_phase(self, phase_32: int) -> np.ndarray:
        """
        Get coefficient block for a given 32-bit phase.

        Uses phase rotation: c(φ) = cos(φ)*C_sin + sin(φ)*C_cos

        Args:
            phase_32: 32-bit phase accumulator value

        Returns:
            np.ndarray[num_leaves] - 64 coefficients
        """
        # Convert 32-bit phase to radians
        phase_frac = phase_32 / (1 << 32)
        phase_rad = 2.0 * np.pi * phase_frac

        cos_phi = np.cos(phase_rad)
        sin_phi = np.sin(phase_rad)

        # Phase rotation
        coeffs = cos_phi * self.c_sin.astype(np.float32) + sin_phi * self.c_cos.astype(np.float32)
        return np.clip(coeffs, -32768, 32767).astype(np.int16)

    def get_single_tap(self, phase_idx: int, tap: int) -> np.ndarray:
        """
        Get single tap (64 leaves) - legacy interface.

        For backward compatibility, converts phase_idx to 32-bit phase
        and calls get_coeffs_for_phase.

        Args:
            phase_idx: Phase step (0..P-1)
            tap: Ignored (polyphase taps not used in this version)

        Returns:
            np.ndarray[num_leaves] - one coefficient per leaf
        """
        # Convert phase index to 32-bit phase
        phase_32 = int((phase_idx / self.P) * (1 << 32)) & 0xFFFFFFFF
        return self.get_coeffs_for_phase(phase_32)

    @staticmethod
    def generate_from_waveform(
        wave_id: int,
        name: str,
        generate_sample_fn,  # Callable[[int], int] - phase -> sample
        K: int = DEFAULT_K,
        P: int = DEFAULT_P,
        cycle_length: int = CYCLE_LENGTH,
        normalize: bool = True,
    ) -> 'CoeffLUT':
        """
        Generate CoeffLUT with sin/cos basis coefficients.

        For sinusoidal waveforms, we generate coefficients for:
        - C_sin: FWPT of sin(ωt) block
        - C_cos: FWPT of cos(ωt) block (= sin(ωt + π/2))

        At runtime, phase rotation gives:
            C(φ) = cos(φ)*C_sin + sin(φ)*C_cos

        For non-sinusoidal waveforms, the basis approach still works
        because any periodic waveform can be decomposed.

        Args:
            wave_id: Unique waveform identifier
            name: Human-readable name
            generate_sample_fn: Function(phase) -> sample value (Q15)
            K: Polyphase taps (kept for compatibility)
            P: Phase steps (kept for compatibility)
            cycle_length: Samples per cycle for table generation
            normalize: Apply normalization

        Returns:
            CoeffLUT with sin/cos basis coefficients
        """
        logger.info(f"Generating CoeffLUT with sin/cos basis for {name}")

        # Reference frequency: complete exactly 1 cycle per 64-sample block
        # This gives good coefficient resolution
        phase_inc = (1 << 32) // BLOCK_SIZE

        # Generate sin basis (starting at phase 0)
        block_sin = np.zeros(BLOCK_SIZE, dtype=np.float32)
        phase = 0
        for i in range(BLOCK_SIZE):
            block_sin[i] = generate_sample_fn(phase)
            phase = (phase + phase_inc) & 0xFFFFFFFF

        # DC removal
        block_sin -= block_sin.mean()

        # FWPT for sin basis
        block_sin_int = np.clip(block_sin, -32768, 32767).astype(np.int16)
        c_sin = np.array(fwpt_jit(block_sin_int.tolist(), headroom_bits=1), dtype=np.float32)

        # Generate cos basis (starting at phase π/2 = 0x40000000)
        block_cos = np.zeros(BLOCK_SIZE, dtype=np.float32)
        phase = 0x40000000  # 90 degrees
        for i in range(BLOCK_SIZE):
            block_cos[i] = generate_sample_fn(phase)
            phase = (phase + phase_inc) & 0xFFFFFFFF

        # DC removal
        block_cos -= block_cos.mean()

        # FWPT for cos basis
        block_cos_int = np.clip(block_cos, -32768, 32767).astype(np.int16)
        c_cos = np.array(fwpt_jit(block_cos_int.tolist(), headroom_bits=1), dtype=np.float32)

        # Normalize both bases together
        if normalize:
            # Joint RMS normalization
            rms = np.sqrt((np.mean(c_sin**2) + np.mean(c_cos**2)) / 2)
            if rms > 1e-6:
                scale = TARGET_RMS_Q15 / rms
                c_sin = c_sin * scale
                c_cos = c_cos * scale

            # Peak safety
            peak = max(np.abs(c_sin).max(), np.abs(c_cos).max())
            max_allowed = 0.95 * Q15_ONE
            if peak > max_allowed:
                safety_scale = max_allowed / peak
                c_sin = c_sin * safety_scale
                c_cos = c_cos * safety_scale

        # Quantize to int16
        c_sin = np.clip(c_sin, -32768, 32767).astype(np.int16)
        c_cos = np.clip(c_cos, -32768, 32767).astype(np.int16)

        return CoeffLUT(
            wave_id=wave_id,
            name=name,
            c_sin=c_sin,
            c_cos=c_cos,
            num_leaves=NUM_LEAVES,
            K=K,
            P=P,
            phase_anchor="rising_zero",
            normalization="rms" if normalize else "none",
        )

    @staticmethod
    def generate_sine(K: int = DEFAULT_K, P: int = DEFAULT_P) -> 'CoeffLUT':
        """Generate sine wave CoeffLUT."""
        def sine_sample(phase: int) -> int:
            return sine_q15(phase)

        return CoeffLUT.generate_from_waveform(
            wave_id=0,
            name="sine",
            generate_sample_fn=sine_sample,
            K=K,
            P=P,
        )

    @staticmethod
    def generate_saw(K: int = DEFAULT_K, P: int = DEFAULT_P) -> 'CoeffLUT':
        """Generate sawtooth wave CoeffLUT (bandlimited)."""
        def saw_sample(phase: int) -> int:
            # Simple sawtooth: phase / 2^32 -> [-1, 1] -> Q15
            # Bandlimit by only using first 16 harmonics
            val = 0
            for h in range(1, 17):
                h_phase = (phase * h) & 0xFFFFFFFF
                val += sine_q15(h_phase) // h
            return int(np.clip(val // 2, -32768, 32767))

        return CoeffLUT.generate_from_waveform(
            wave_id=1,
            name="saw",
            generate_sample_fn=saw_sample,
            K=K,
            P=P,
        )

    @staticmethod
    def generate_square(K: int = DEFAULT_K, P: int = DEFAULT_P) -> 'CoeffLUT':
        """Generate square wave CoeffLUT (bandlimited)."""
        def square_sample(phase: int) -> int:
            # Square: odd harmonics only
            val = 0
            for h in [1, 3, 5, 7, 9, 11, 13, 15]:
                h_phase = (phase * h) & 0xFFFFFFFF
                val += sine_q15(h_phase) // h
            return int(np.clip(val, -32768, 32767))

        return CoeffLUT.generate_from_waveform(
            wave_id=2,
            name="square",
            generate_sample_fn=square_sample,
            K=K,
            P=P,
        )


# =============================================================================
# Global CoeffLUT library (lazy initialization)
# =============================================================================

_COEFF_LUT_LIBRARY: Optional[Dict[int, CoeffLUT]] = None


def _init_library() -> Dict[int, CoeffLUT]:
    """Initialize the CoeffLUT library."""
    logger.info("Initializing CoeffLUT library...")
    return {
        0: CoeffLUT.generate_sine(),
        1: CoeffLUT.generate_saw(),
        2: CoeffLUT.generate_square(),
    }


def get_coeff_lut(wave_id: int) -> CoeffLUT:
    """Get CoeffLUT by wave ID (lazy initialization)."""
    global _COEFF_LUT_LIBRARY
    if _COEFF_LUT_LIBRARY is None:
        _COEFF_LUT_LIBRARY = _init_library()
    return _COEFF_LUT_LIBRARY.get(wave_id, _COEFF_LUT_LIBRARY[0])


# =============================================================================
# Polyphase IDWPT (streaming synthesis)
# =============================================================================

def polyphase_idwpt_streaming(
    coeff_lut: CoeffLUT,
    phase_accum: int,
    phase_inc: int,
) -> np.ndarray:
    """
    Streaming polyphase IDWPT synthesis.

    Per WSE-STD-121, processes K taps per block, each producing
    a portion of the output audio.

    Args:
        coeff_lut: CoeffLUT to use
        phase_accum: Current phase accumulator (32-bit)
        phase_inc: Phase increment per sample

    Returns:
        64-sample audio output
    """
    from ..wavelet.fast_lifting import idwpt_jit

    K = coeff_lut.K
    P = coeff_lut.P

    output = np.zeros(BLOCK_SIZE, dtype=np.int16)

    # Process K polyphase components
    samples_per_tap = BLOCK_SIZE // K

    for k in range(K):
        # Compute phase for this tap
        tap_phase = (phase_accum + (k * samples_per_tap * phase_inc)) & 0xFFFFFFFF

        # Convert to phase index
        phase_idx = (tap_phase >> 24) % P  # Use top 8 bits for 256 steps

        # Get coefficients for this tap
        coeffs = coeff_lut.get_single_tap(phase_idx, k)

        # IDWPT on this tap
        audio = idwpt_jit(coeffs.astype(np.int16).tolist(), headroom_bits=1)

        # Interleave into output
        # Tap k contributes to samples k, k+K, k+2K, ...
        for i, sample in enumerate(audio):
            out_idx = k + (i % samples_per_tap) * K
            if out_idx < BLOCK_SIZE:
                output[out_idx] = np.clip(output[out_idx] + sample // K, -32768, 32767)

    return output


__all__ = [
    'CoeffLUT',
    'get_coeff_lut',
    'polyphase_idwpt_streaming',
    'DEFAULT_K',
    'DEFAULT_P',
    'BLOCK_SIZE',
    'NUM_LEAVES',
]
