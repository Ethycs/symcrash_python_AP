"""
Waveform Matrix Generator - Cyclic Shift Waveform Synthesis
============================================================

Generates non-smooth periodic waveforms using cyclic shift matrices.

Key insight: All periodic discrete-time waveforms can be generated via:
    y_n = c^T @ P^n @ x_0

Where:
    P = cyclic shift (permutation) matrix [m x m]
    c = one period of waveform samples [m]
    x_0 = e_0 = [1, 0, 0, ..., 0]^T (unit vector)

This gives y_n = c[n mod m], i.e., the waveform sample at time n.

For CCS synthesis, we DON'T need to explicitly compute matrix powers.
Instead, we:
1. Use ONE high-quality quarter-sine LUT for smooth oscillation
2. Store ONE PERIOD of each non-smooth waveform as a lookup vector c
3. Index via phase accumulator: y_n = c[phase >> shift]

This achieves:
- Memory efficiency: only store c vectors, not full matrices
- CPU efficiency: simple array lookup, no matrix multiply
- Quality: smooth sine from sine_core, sharp edges from c vectors
- Flexibility: arbitrary waveforms via c vector

Waveform Types:
- SINE: Uses sine_core directly (smooth)
- SQUARE: c = [+1, +1, ..., -1, -1, ...] (sharp edges)
- PULSE: c = [+1, +1, ..., -1, -1, -1, ...] (variable duty)
- SAW: c = linear ramp with discontinuity
- TRIANGLE: c = piecewise linear (corners, no discontinuity)
- IMPULSE: c = [1, 0, 0, ..., 0] (single spike per period)
- NOISE: LFSR-based pseudo-random sequence
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import numpy as np

from ..fixed import Q15_ONE, RSHIFT_RNE, SAT16
from ..wavelet.sine_core import (
    sine_q15, cosine_q15, sin_cos_q15,
    PHASE_BITS, PHASE_MASK
)

# =============================================================================
# CONSTANTS
# =============================================================================

# Waveform period resolution (samples per waveform period)
# Higher = more accurate edges, more memory
# 256 is good balance: 512 bytes per waveform, <0.4% Nyquist error at 20kHz
WAVEFORM_PERIOD = 256
WAVEFORM_BITS = 8  # log2(WAVEFORM_PERIOD)

# Phase to waveform index shift
PHASE_TO_WAVE_SHIFT = PHASE_BITS - WAVEFORM_BITS  # 32 - 8 = 24


# =============================================================================
# WAVEFORM TYPES
# =============================================================================

class WaveformType:
    """Waveform type constants."""
    SINE = 0        # Smooth sine (uses sine_core)
    SQUARE = 1      # Square wave (50% duty)
    PULSE = 2       # Pulse wave (variable duty)
    SAW = 3         # Sawtooth (ramp up, jump down)
    SAW_DOWN = 4    # Inverse sawtooth (jump up, ramp down)
    TRIANGLE = 5    # Triangle wave (smooth corners via matrix)
    IMPULSE = 6     # Single impulse per period
    NOISE = 7       # Pseudo-random (LFSR)


# =============================================================================
# BLEP (Band-Limited Step) for anti-aliased c-vectors
# =============================================================================

def _generate_blep_table(size: int = 16) -> np.ndarray:
    """
    Generate a BLEP (Band-Limited Step) residual table.

    BLEP replaces the hard step discontinuity with a smooth, bandlimited
    transition. The residual is added around step transitions.

    Args:
        size: Number of samples in the transition region (each side)

    Returns:
        BLEP residual table [2*size] in Q15
    """
    # Polynomial BLEP (PolyBLEP) residual
    # For t in [-1, 0]: blep = t^2 + 2*t + 1
    # For t in [0, 1]:  blep = t^2 - 2*t + 1
    # Normalized so blep integrates to 0 over the transition
    table = np.zeros(2 * size, dtype=np.int16)

    for i in range(size):
        # Pre-transition samples (t goes from -1 to 0)
        t = (i - size) / size  # t in [-1, 0)
        residual = t * t + 2 * t + 1  # Quadratic BLEP
        table[i] = int(residual * Q15_ONE * 0.5)  # Scale for smoother transition

    for i in range(size):
        # Post-transition samples (t goes from 0 to 1)
        t = i / size  # t in [0, 1)
        residual = t * t - 2 * t + 1  # Quadratic BLEP
        table[size + i] = int(residual * Q15_ONE * 0.5)

    return table


# Precomputed BLEP table
_BLEP_SIZE = 8
_BLEP_TABLE = _generate_blep_table(_BLEP_SIZE)


def apply_blep(c: np.ndarray, transition_idx: int, step_direction: int = 1) -> None:
    """
    Apply BLEP residual around a step transition in-place.

    Args:
        c: Waveform c-vector to modify
        transition_idx: Index where the step occurs
        step_direction: +1 for rising edge, -1 for falling edge
    """
    period = len(c)

    for i in range(_BLEP_SIZE):
        # Pre-transition
        idx = (transition_idx - _BLEP_SIZE + i) % period
        c[idx] = np.clip(
            int(c[idx]) + step_direction * int(_BLEP_TABLE[i]),
            -Q15_ONE, Q15_ONE
        ).astype(np.int16)

        # Post-transition
        idx = (transition_idx + i) % period
        c[idx] = np.clip(
            int(c[idx]) - step_direction * int(_BLEP_TABLE[_BLEP_SIZE + i]),
            -Q15_ONE, Q15_ONE
        ).astype(np.int16)


# =============================================================================
# WAVEFORM VECTOR GENERATION (c vectors)
# =============================================================================

def generate_square_vector(
    period: int = WAVEFORM_PERIOD,
    duty_q15: int = 16384,
    bandlimited: bool = False
) -> np.ndarray:
    """
    Generate square wave c vector.

    Args:
        period: Number of samples per period
        duty_q15: Duty cycle in Q15 (16384 = 50%)
        bandlimited: If True, apply BLEP for anti-aliased edges

    Returns:
        Q15 waveform vector [period]
    """
    c = np.zeros(period, dtype=np.int16)
    duty_samples = (duty_q15 * period) >> 15
    duty_samples = max(1, min(period - 1, duty_samples))

    c[:duty_samples] = Q15_ONE
    c[duty_samples:] = -Q15_ONE

    if bandlimited:
        # Apply BLEP at both transitions
        apply_blep(c, duty_samples, step_direction=-1)  # Falling edge
        apply_blep(c, 0, step_direction=1)  # Rising edge (wraps around)

    return c


def generate_pulse_vector(
    period: int = WAVEFORM_PERIOD,
    duty_q15: int = 8192,
    bandlimited: bool = False
) -> np.ndarray:
    """
    Generate pulse wave c vector with variable duty cycle.

    Args:
        period: Number of samples per period
        duty_q15: Duty cycle in Q15 (8192 = 25%)
        bandlimited: If True, apply BLEP for anti-aliased edges

    Returns:
        Q15 waveform vector [period]
    """
    return generate_square_vector(period, duty_q15, bandlimited)


def generate_saw_vector(
    period: int = WAVEFORM_PERIOD,
    invert: bool = False,
    bandlimited: bool = False
) -> np.ndarray:
    """
    Generate sawtooth wave c vector.

    Ramp from -1 to +1, then discontinuous jump back.

    Args:
        period: Number of samples per period
        invert: If True, ramp down instead of up
        bandlimited: If True, apply BLEP at the discontinuity

    Returns:
        Q15 waveform vector [period]
    """
    c = np.zeros(period, dtype=np.int16)

    for i in range(period):
        # Linear ramp: -1 at i=0, approaches +1 at i=period-1
        # value = -1 + 2 * i / (period - 1)
        # In Q15: value = -32767 + (65534 * i) / (period - 1)
        if period > 1:
            value = -Q15_ONE + (2 * Q15_ONE * i) // (period - 1)
        else:
            value = 0

        if invert:
            value = -value

        c[i] = np.clip(value, -Q15_ONE, Q15_ONE)

    if bandlimited:
        # Apply BLEP at the discontinuity (wrap-around from end to start)
        # For standard saw: rising edge wraps at index 0
        step_dir = -1 if invert else 1
        apply_blep(c, 0, step_direction=step_dir)

    return c


def generate_triangle_vector(period: int = WAVEFORM_PERIOD) -> np.ndarray:
    """
    Generate triangle wave c vector.

    Linear ramp up for half period, linear ramp down for half period.
    Has corners but no discontinuities.

    Args:
        period: Number of samples per period

    Returns:
        Q15 waveform vector [period]
    """
    c = np.zeros(period, dtype=np.int16)
    half = period // 2

    for i in range(period):
        if i < half:
            # Ascending: -1 to +1 over first half
            if half > 0:
                value = -Q15_ONE + (2 * Q15_ONE * i) // half
            else:
                value = -Q15_ONE
        else:
            # Descending: +1 to -1 over second half
            pos_in_second_half = i - half
            remaining = period - half
            if remaining > 0:
                value = Q15_ONE - (2 * Q15_ONE * pos_in_second_half) // remaining
            else:
                value = Q15_ONE

        c[i] = np.clip(value, -Q15_ONE, Q15_ONE)

    return c


def generate_impulse_vector(period: int = WAVEFORM_PERIOD) -> np.ndarray:
    """
    Generate impulse/click train c vector.

    Single spike at start of period, zeros elsewhere.

    Args:
        period: Number of samples per period

    Returns:
        Q15 waveform vector [period]
    """
    c = np.zeros(period, dtype=np.int16)
    c[0] = Q15_ONE
    return c


def generate_noise_vector(period: int = WAVEFORM_PERIOD, seed: int = 0xACE1) -> np.ndarray:
    """
    Generate pseudo-random noise c vector via LFSR.

    Uses 16-bit LFSR with taps at bits 15, 13, 12, 10 (maximal length).
    Period of LFSR is 2^16 - 1 = 65535.

    Args:
        period: Number of samples per period
        seed: LFSR seed (must be non-zero)

    Returns:
        Q15 waveform vector [period] with values in {-Q15_ONE, +Q15_ONE}
    """
    c = np.zeros(period, dtype=np.int16)
    state = seed if seed != 0 else 0xACE1

    for i in range(period):
        # Output bit -> bipolar
        bit = state & 1
        c[i] = Q15_ONE if bit else -Q15_ONE

        # LFSR step: x^16 + x^14 + x^13 + x^11 + 1
        feedback = ((state >> 0) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1
        state = ((state >> 1) | (feedback << 15)) & 0xFFFF

    return c


# =============================================================================
# WAVEFORM CACHE
# =============================================================================

@dataclass
class WaveformCache:
    """
    Cache of precomputed waveform c vectors.

    Lazy-loads waveforms on first use.
    Supports both raw (aliased) and bandlimited (BLEP) versions.
    """
    _cache: dict = field(default_factory=dict)
    period: int = WAVEFORM_PERIOD
    bandlimited: bool = True  # Use BLEP by default for smoother coefficient interpolation

    def get_vector(
        self,
        wave_type: int,
        duty_q15: int = 16384,
        bandlimited: Optional[bool] = None
    ) -> np.ndarray:
        """
        Get waveform vector, generating if needed.

        Args:
            wave_type: WaveformType constant
            duty_q15: Duty cycle for pulse/square (ignored for others)
            bandlimited: Override default BLEP setting (None = use cache default)

        Returns:
            Q15 waveform vector [period]
        """
        use_blep = self.bandlimited if bandlimited is None else bandlimited

        # Cache key includes duty for pulse waves and BLEP setting
        if wave_type in (WaveformType.SQUARE, WaveformType.PULSE):
            key = (wave_type, duty_q15, use_blep)
        else:
            key = (wave_type, 0, use_blep)

        if key not in self._cache:
            self._cache[key] = self._generate(wave_type, duty_q15, use_blep)

        return self._cache[key]

    def _generate(self, wave_type: int, duty_q15: int, bandlimited: bool) -> np.ndarray:
        """Generate waveform vector."""
        if wave_type == WaveformType.SQUARE:
            return generate_square_vector(self.period, 16384, bandlimited)
        elif wave_type == WaveformType.PULSE:
            return generate_pulse_vector(self.period, duty_q15, bandlimited)
        elif wave_type == WaveformType.SAW:
            return generate_saw_vector(self.period, invert=False, bandlimited=bandlimited)
        elif wave_type == WaveformType.SAW_DOWN:
            return generate_saw_vector(self.period, invert=True, bandlimited=bandlimited)
        elif wave_type == WaveformType.TRIANGLE:
            return generate_triangle_vector(self.period)  # Triangle has no discontinuities
        elif wave_type == WaveformType.IMPULSE:
            return generate_impulse_vector(self.period)
        elif wave_type == WaveformType.NOISE:
            return generate_noise_vector(self.period)
        else:
            # Default to zeros (SINE uses sine_core directly)
            return np.zeros(self.period, dtype=np.int16)

    def clear(self):
        """Clear cache."""
        self._cache.clear()


# Global cache instance
_WAVEFORM_CACHE = WaveformCache()


def get_waveform_cache() -> WaveformCache:
    """Get global waveform cache."""
    return _WAVEFORM_CACHE


# =============================================================================
# WAVEFORM SAMPLE FUNCTIONS
# =============================================================================

def waveform_sample_q15(
    phase_32: int,
    wave_type: int,
    duty_q15: int = 16384,
) -> int:
    """
    Get waveform sample at given phase.

    For SINE, uses high-quality sine_core.
    For others, uses cyclic shift vector lookup.

    Args:
        phase_32: 32-bit phase (0 to 2^32-1 = one full cycle)
        wave_type: WaveformType constant
        duty_q15: Duty cycle for pulse waves

    Returns:
        Q15 sample value
    """
    if wave_type == WaveformType.SINE:
        return sine_q15(phase_32)

    # Get waveform vector
    cache = get_waveform_cache()
    c = cache.get_vector(wave_type, duty_q15)

    # Convert phase to index: index = phase >> (32 - WAVEFORM_BITS)
    index = (phase_32 >> PHASE_TO_WAVE_SHIFT) & (WAVEFORM_PERIOD - 1)

    return int(c[index])


def waveform_sample_vectorized(
    phases_32: np.ndarray,
    wave_type: int,
    duty_q15: int = 16384,
) -> np.ndarray:
    """
    Vectorized waveform sampling.

    Args:
        phases_32: Array of 32-bit phases
        wave_type: WaveformType constant
        duty_q15: Duty cycle for pulse waves

    Returns:
        Array of Q15 samples
    """
    if wave_type == WaveformType.SINE:
        from ..wavelet.sine_core import sine_q15_vectorized
        # Convert Q32 to Q16 for vectorized sine
        phases_16 = ((phases_32 >> 16) & 0xFFFF).astype(np.uint16)
        return sine_q15_vectorized(phases_16)

    # Get waveform vector
    cache = get_waveform_cache()
    c = cache.get_vector(wave_type, duty_q15)

    # Convert phases to indices
    indices = (phases_32 >> PHASE_TO_WAVE_SHIFT) & (WAVEFORM_PERIOD - 1)

    return c[indices].astype(np.int32)


# =============================================================================
# HARMONIC DECOMPOSITION (Fourier coefficients of c vectors)
# =============================================================================

def compute_harmonic_coefficients(
    wave_type: int,
    H_max: int = 32,
    duty_q15: int = 16384,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Compute Fourier series coefficients for waveform.

    Decomposes c vector into: x(φ) = dc + Σ_h (b_h·cos(hφ) + a_h·sin(hφ))

    This allows using the waveform with the harmonic bank (Mode B)
    for bandlimited playback.

    Args:
        wave_type: WaveformType constant
        H_max: Maximum harmonic number
        duty_q15: Duty cycle for pulse waves

    Returns:
        (dc_q15, a_h_q15[H_max], b_h_q15[H_max])
        where a_h is sine coefficient, b_h is cosine coefficient
    """
    # Get waveform vector
    cache = get_waveform_cache()
    c = cache.get_vector(wave_type, duty_q15)
    period = len(c)

    # Compute DFT coefficients
    # c[n] = dc + Σ_h (b_h·cos(2πhn/N) + a_h·sin(2πhn/N))

    # DC component
    dc = int(np.mean(c))

    # Harmonic coefficients via correlation
    a_h = np.zeros(H_max, dtype=np.int16)
    b_h = np.zeros(H_max, dtype=np.int16)

    for h in range(1, H_max + 1):
        # Compute correlation with sin and cos at harmonic h
        sin_sum = 0
        cos_sum = 0

        for n in range(period):
            # Phase for this sample: 2πhn/N
            phase_32 = int((h * n * (1 << PHASE_BITS)) / period) & PHASE_MASK
            sin_val, cos_val = sin_cos_q15(phase_32)

            sample = int(c[n]) - dc  # Remove DC for AC coefficients
            sin_sum += sample * sin_val
            cos_sum += sample * cos_val

        # Normalize: divide by N/2 (for orthogonal basis)
        # Then convert to Q15
        a_h[h - 1] = SAT16(RSHIFT_RNE(sin_sum, WAVEFORM_BITS - 1 + 15))
        b_h[h - 1] = SAT16(RSHIFT_RNE(cos_sum, WAVEFORM_BITS - 1 + 15))

    return dc, a_h, b_h


def get_theoretical_harmonics(
    wave_type: int,
    H_max: int = 32,
    duty_q15: int = 16384,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Get theoretical harmonic coefficients for standard waveforms.

    More accurate than DFT decomposition for known waveforms.

    Args:
        wave_type: WaveformType constant
        H_max: Maximum harmonic number
        duty_q15: Duty cycle for pulse waves

    Returns:
        (dc_q15, a_h_q15[H_max], b_h_q15[H_max])
    """
    import math

    dc = 0
    a_h = np.zeros(H_max, dtype=np.int16)
    b_h = np.zeros(H_max, dtype=np.int16)

    if wave_type == WaveformType.SINE:
        # Pure sine: a_1 = 1, all else = 0
        a_h[0] = Q15_ONE

    elif wave_type == WaveformType.SAW:
        # Sawtooth: a_h = -2/πh (alternating signs for proper saw)
        # x(t) = -2/π Σ ((-1)^h / h) sin(2πht)
        for h in range(1, H_max + 1):
            # a_h = -2/(πh) * (-1)^h = 2/(πh) * (-1)^(h+1)
            sign = 1 if (h % 2 == 1) else -1
            amp = (2 * Q15_ONE) / (math.pi * h)
            a_h[h - 1] = SAT16(int(-sign * amp))

    elif wave_type == WaveformType.SQUARE:
        # Square (50% duty): a_h = 4/(πh) for odd h only
        for h in range(1, H_max + 1, 2):  # Odd harmonics only
            amp = (4 * Q15_ONE) / (math.pi * h)
            a_h[h - 1] = SAT16(int(amp))

    elif wave_type == WaveformType.PULSE:
        # Pulse with duty D: more complex spectrum
        # a_h = (2/(πh)) * sin(πhD)
        duty = duty_q15 / 32768.0  # Convert to 0-1
        for h in range(1, H_max + 1):
            amp = (2 * Q15_ONE / (math.pi * h)) * math.sin(math.pi * h * duty)
            a_h[h - 1] = SAT16(int(amp))
        # DC offset for non-50% duty
        dc = int((2 * duty - 1) * Q15_ONE)

    elif wave_type == WaveformType.TRIANGLE:
        # Triangle: a_h = 8/(π²h²) for odd h only (alternating signs)
        for h in range(1, H_max + 1, 2):  # Odd harmonics only
            sign = 1 if ((h - 1) // 2 % 2 == 0) else -1
            amp = (8 * Q15_ONE) / (math.pi * math.pi * h * h)
            a_h[h - 1] = SAT16(int(sign * amp))

    return dc, a_h, b_h


# =============================================================================
# PWM SUPPORT (Maps to Mode B)
# =============================================================================

def compute_pwm_harmonics(
    duty_q15: int,
    H_max: int = 32,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Compute harmonic coefficients for PWM at given duty cycle.

    PWM spectrum: dc + Σ_h (2/(πh) * sin(πhD)) * sin(hφ)
    where D is duty cycle (0-1).

    Args:
        duty_q15: Duty cycle in Q15 (16384 = 50%)
        H_max: Maximum harmonic number

    Returns:
        (dc_q15, a_h_q15[H_max], b_h_q15[H_max])
    """
    import math

    duty = duty_q15 / 32768.0

    # DC offset: (2D - 1) maps duty 0->-1, 0.5->0, 1->+1
    dc = int((2 * duty - 1) * Q15_ONE)

    a_h = np.zeros(H_max, dtype=np.int16)
    b_h = np.zeros(H_max, dtype=np.int16)

    for h in range(1, H_max + 1):
        # a_h = (2/(πh)) * sin(πhD)
        amp = (2 * Q15_ONE / (math.pi * h)) * math.sin(math.pi * h * duty)
        a_h[h - 1] = SAT16(int(amp))

    return dc, a_h, b_h


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'WAVEFORM_PERIOD',
    'WAVEFORM_BITS',
    'PHASE_TO_WAVE_SHIFT',
    # Waveform types
    'WaveformType',
    # Vector generation
    'generate_square_vector',
    'generate_pulse_vector',
    'generate_saw_vector',
    'generate_triangle_vector',
    'generate_impulse_vector',
    'generate_noise_vector',
    # Cache
    'WaveformCache',
    'get_waveform_cache',
    # Sampling
    'waveform_sample_q15',
    'waveform_sample_vectorized',
    # Harmonic decomposition
    'compute_harmonic_coefficients',
    'get_theoretical_harmonics',
    'compute_pwm_harmonics',
]
