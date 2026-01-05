"""
CoeffCore - Coefficient-Domain Oscillator Primitive
====================================================

Like sine_core generates audio samples from phase, coeff_core generates
wavelet coefficients directly from phase. No tables, no FWPT at runtime.

Key Insight:
------------
For a sinusoid at frequency ω passing through a linear filterbank:
    coeff[l] = A[l](ω) * sin(ω*t + φ[l](ω))

Where:
    - A[l](ω) = amplitude response of leaf l at frequency ω
    - φ[l](ω) = phase response of leaf l at frequency ω

The filterbank response is computed ONCE per waveform type (sine, saw, etc.)
and stored as compact gain/phase tables. Runtime is just:
    1. Look up gain[l] and phase_offset[l] for each leaf
    2. Compute: coeff[l] = gain[l] * sine_q15(phase + phase_offset[l])

This is O(64) sine lookups per block, same as additive synthesis but in
coefficient space. No FWPT, no per-note tables.

Memory: 64 leaves × 2 (gain + phase) × 2 bytes = 256 bytes per waveform

Architecture:
-------------
1. FilterbankResponse: Precomputed gain/phase per leaf (from offline FWPT analysis)
2. coeff_block(): Generate 64 coefficients from phase (pure function)
3. CoeffOscCore: Stateful oscillator wrapper
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict

from .sine_core import sine_q15, cosine_q15, PHASE_MASK
from .fast_lifting import fwpt_jit, idwpt_jit
from ..fixed import RSHIFT_RNE

# Constants
NUM_LEAVES = 64
BLOCK_SIZE = 64
Q15_ONE = 32767
SAMPLE_RATE = 48000


# =============================================================================
# FILTERBANK RESPONSE (Computed once per waveform)
# =============================================================================

@dataclass
class FilterbankResponse:
    """
    Precomputed filterbank amplitude/phase response for a waveform.

    For each leaf l:
        coeff[l] = gain_q15[l] * sin(phase + phase_offset[l])

    This captures how the waveform's energy distributes across subbands.
    """
    name: str
    wave_id: int

    # Per-leaf gains in Q15 (0 = silent, 32767 = full)
    gain_q15: np.ndarray  # [64] int16

    # Per-leaf phase offsets (32-bit, same scale as phase accumulator)
    phase_offset: np.ndarray  # [64] uint32

    # Reference frequency used for analysis (for scaling)
    ref_freq_hz: float = 750.0  # 1 cycle per 64 samples at 48kHz


def analyze_filterbank_response(
    generate_sample_fn,  # Callable[[int], int] - phase -> Q15 sample
    name: str = "sine",
    wave_id: int = 0,
    ref_freq_hz: float = 750.0,
) -> FilterbankResponse:
    """
    Analyze a waveform to extract filterbank gain/phase response.

    Process:
    1. Generate one cycle of the waveform at reference frequency
    2. Run FWPT to get coefficients
    3. Extract amplitude and phase for each leaf

    This is done ONCE at startup, not at runtime.

    Args:
        generate_sample_fn: Waveform generator function
        name: Waveform name
        wave_id: Waveform ID
        ref_freq_hz: Reference frequency for analysis

    Returns:
        FilterbankResponse with gain/phase per leaf
    """
    # Generate reference waveform at phase 0 (sin basis)
    phase_inc = int((ref_freq_hz / SAMPLE_RATE) * (1 << 32))

    audio_sin = np.zeros(BLOCK_SIZE, dtype=np.int16)
    phase = 0
    for i in range(BLOCK_SIZE):
        audio_sin[i] = generate_sample_fn(phase)
        phase = (phase + phase_inc) & 0xFFFFFFFF

    # Generate reference waveform at phase 90° (cos basis)
    audio_cos = np.zeros(BLOCK_SIZE, dtype=np.int16)
    phase = 0x40000000  # 90 degrees
    for i in range(BLOCK_SIZE):
        audio_cos[i] = generate_sample_fn(phase)
        phase = (phase + phase_inc) & 0xFFFFFFFF

    # FWPT to get coefficient bases
    coeffs_sin = np.array(fwpt_jit(audio_sin.tolist(), headroom_bits=1), dtype=np.float32)
    coeffs_cos = np.array(fwpt_jit(audio_cos.tolist(), headroom_bits=1), dtype=np.float32)

    # Extract amplitude and phase per leaf
    # For each leaf: coeff = A * sin(ωt + φ)
    # From sin/cos basis: A = sqrt(c_sin² + c_cos²), φ = atan2(c_cos, c_sin)
    gain_q15 = np.zeros(NUM_LEAVES, dtype=np.int16)
    phase_offset = np.zeros(NUM_LEAVES, dtype=np.uint32)

    for l in range(NUM_LEAVES):
        c_sin = coeffs_sin[l]
        c_cos = coeffs_cos[l]

        # Amplitude (magnitude of phasor)
        amplitude = math.sqrt(c_sin * c_sin + c_cos * c_cos)

        # Clamp to Q15 range
        gain_q15[l] = int(np.clip(amplitude, 0, Q15_ONE))

        # Phase offset: atan2(c_cos, c_sin) converted to 32-bit phase
        if amplitude > 1e-6:
            phase_rad = math.atan2(c_cos, c_sin)
            # Convert radians [-π, π] to 32-bit phase [0, 2^32)
            phase_32 = int((phase_rad / (2 * math.pi)) * (1 << 32)) & 0xFFFFFFFF
            phase_offset[l] = phase_32
        else:
            phase_offset[l] = 0

    return FilterbankResponse(
        name=name,
        wave_id=wave_id,
        gain_q15=gain_q15,
        phase_offset=phase_offset,
        ref_freq_hz=ref_freq_hz,
    )


# =============================================================================
# COEFFICIENT GENERATION (Pure function, no state)
# =============================================================================

def coeff_block_q15(
    response: FilterbankResponse,
    phase: int,
    output: np.ndarray,
) -> None:
    """
    Generate 64 coefficients from phase using filterbank response.

    This is the core primitive - pure function, no state.

    For each leaf:
        output[l] = gain[l] * sin(phase + phase_offset[l])

    Args:
        response: Precomputed filterbank response
        phase: 32-bit phase accumulator value
        output: np.ndarray[64] int16 output buffer (modified in place)
    """
    for l in range(NUM_LEAVES):
        gain = response.gain_q15[l]
        offset = response.phase_offset[l]

        # Compute leaf phase
        leaf_phase = (phase + offset) & 0xFFFFFFFF

        # Sine lookup
        sin_val = sine_q15(leaf_phase)

        # Scale by gain (Q15 * Q15 >> 15)
        coeff = RSHIFT_RNE(int(gain) * int(sin_val), 15)

        # Saturate
        output[l] = int(np.clip(coeff, -32768, 32767))


def coeff_block_q15_vectorized(
    response: FilterbankResponse,
    phase: int,
) -> np.ndarray:
    """
    Vectorized coefficient generation (faster for Python).

    Returns:
        np.ndarray[64] int16 coefficients
    """
    from .sine_core import sine_q15_vectorized

    # Compute all leaf phases
    leaf_phases = ((phase + response.phase_offset.astype(np.int64)) & 0xFFFFFFFF).astype(np.uint32)

    # Convert to 16-bit for fast lookup
    phases_16 = (leaf_phases >> 16).astype(np.uint16)

    # Vectorized sine lookup
    sin_vals = sine_q15_vectorized(phases_16)

    # Scale by gains
    coeffs = (response.gain_q15.astype(np.int32) * sin_vals) >> 15

    return np.clip(coeffs, -32768, 32767).astype(np.int16)


# =============================================================================
# COEFFICIENT OSCILLATOR STATE
# =============================================================================

@dataclass
class CoeffOscCore:
    """
    Coefficient oscillator state.

    Like sine_core.OscillatorState but generates coefficients instead of audio.
    """
    phase: int = 0
    phase_inc: int = 0
    target_phase_inc: int = 0
    slew_rate: int = 0

    def reset(self, phase: int = 0) -> None:
        """Hard reset phase."""
        self.phase = phase & PHASE_MASK

    def set_frequency(self, freq_hz: float, sample_rate: int = SAMPLE_RATE) -> None:
        """Set frequency immediately."""
        self.phase_inc = int((freq_hz / sample_rate) * (1 << 32)) & PHASE_MASK
        self.target_phase_inc = self.phase_inc
        self.slew_rate = 0

    def set_midi_note(self, note: int, cents: int = 0, sample_rate: int = SAMPLE_RATE) -> None:
        """Set frequency from MIDI note."""
        freq_hz = 440.0 * (2.0 ** ((note - 69 + cents / 100.0) / 12.0))
        self.set_frequency(freq_hz, sample_rate)

    def advance_block(self, block_size: int = BLOCK_SIZE) -> int:
        """
        Advance phase by one block, return starting phase.

        Returns:
            Phase at start of block (for coefficient generation)
        """
        current_phase = self.phase

        # Apply pitch slew
        if self.slew_rate != 0:
            if self.slew_rate > 0:
                self.phase_inc = min(self.target_phase_inc,
                                     self.phase_inc + self.slew_rate * block_size)
            else:
                self.phase_inc = max(self.target_phase_inc,
                                     self.phase_inc + self.slew_rate * block_size)
            if self.phase_inc == self.target_phase_inc:
                self.slew_rate = 0

        # Advance phase
        self.phase = (self.phase + self.phase_inc * block_size) & PHASE_MASK

        return current_phase


def render_coeff_block(
    osc: CoeffOscCore,
    response: FilterbankResponse,
    output: np.ndarray,
) -> None:
    """
    Render one block of coefficients.

    Args:
        osc: Oscillator state (modified in place)
        response: Filterbank response for waveform
        output: np.ndarray[64] int16 output buffer
    """
    phase = osc.advance_block()
    coeff_block_q15(response, phase, output)


# =============================================================================
# STANDARD WAVEFORM RESPONSES (Precomputed at module load)
# =============================================================================

_RESPONSES: Optional[Dict[int, FilterbankResponse]] = None


def _init_responses() -> Dict[int, FilterbankResponse]:
    """Initialize standard waveform responses."""
    responses = {}

    # Sine wave
    responses[0] = analyze_filterbank_response(
        generate_sample_fn=sine_q15,
        name="sine",
        wave_id=0,
    )

    # Sawtooth (bandlimited)
    def saw_sample(phase: int) -> int:
        val = 0
        for h in range(1, 17):
            h_phase = (phase * h) & 0xFFFFFFFF
            val += sine_q15(h_phase) // h
        return int(np.clip(val // 2, -32768, 32767))

    responses[1] = analyze_filterbank_response(
        generate_sample_fn=saw_sample,
        name="saw",
        wave_id=1,
    )

    # Square (bandlimited)
    def square_sample(phase: int) -> int:
        val = 0
        for h in [1, 3, 5, 7, 9, 11, 13, 15]:
            h_phase = (phase * h) & 0xFFFFFFFF
            val += sine_q15(h_phase) // h
        return int(np.clip(val, -32768, 32767))

    responses[2] = analyze_filterbank_response(
        generate_sample_fn=square_sample,
        name="square",
        wave_id=2,
    )

    # Triangle (bandlimited)
    def triangle_sample(phase: int) -> int:
        val = 0
        sign = 1
        for h in [1, 3, 5, 7, 9, 11, 13, 15]:
            h_phase = (phase * h) & 0xFFFFFFFF
            val += sign * sine_q15(h_phase) // (h * h)
            sign = -sign
        return int(np.clip(val * 2, -32768, 32767))

    responses[3] = analyze_filterbank_response(
        generate_sample_fn=triangle_sample,
        name="triangle",
        wave_id=3,
    )

    return responses


def get_response(wave_id: int) -> FilterbankResponse:
    """Get filterbank response by wave ID."""
    global _RESPONSES
    if _RESPONSES is None:
        _RESPONSES = _init_responses()
    return _RESPONSES.get(wave_id, _RESPONSES[0])


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'FilterbankResponse',
    'analyze_filterbank_response',
    'coeff_block_q15',
    'coeff_block_q15_vectorized',
    'CoeffOscCore',
    'render_coeff_block',
    'get_response',
    'NUM_LEAVES',
    'BLOCK_SIZE',
]
