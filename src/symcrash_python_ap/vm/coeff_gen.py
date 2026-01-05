"""
Runtime coefficient generation for CCS.
======================================

This module provides real-time coefficient generation for the
coefficient-domain carrier synthesis pipeline.

Key insight: For perfect reconstruction, each block's coefficients
must come from FWPT of audio generated at the correct phase.
Phase must be continuous across blocks.

Since FWPT/IDWPT are perfect inverses, this approach:
1. Generates reference audio at the target frequency (continuous phase)
2. Runs FWPT to get coefficients
3. Passes coefficients to operators
4. Reconstructs via IDWPT

The result is click-free synthesis with proper frequency accuracy.
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional

from ..wavelet.fast_lifting import fwpt_jit, idwpt_jit
from ..wavelet.sine_core import sine_q15


BLOCK_SIZE = 64


def midi_to_phase_inc(note: int, sample_rate: int = 48000) -> int:
    """Convert MIDI note to phase increment."""
    freq_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))
    return int((freq_hz / sample_rate) * (1 << 32))


def midi_to_freq(note: int) -> float:
    """Convert MIDI note to frequency in Hz."""
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


@dataclass
class CoeffGenState:
    """
    State for coefficient generation.

    Tracks phase accumulator for continuous phase across blocks.
    """
    phase: int = 0          # 32-bit phase accumulator
    phase_inc: int = 0      # Phase increment per sample
    note: int = 60          # Current MIDI note
    waveform: int = 0       # Waveform ID (0=sine, 1=saw, 2=square)

    def set_note(self, note: int, sample_rate: int = 48000):
        """Set the note and update phase increment."""
        self.note = note
        self.phase_inc = midi_to_phase_inc(note, sample_rate)

    def reset(self, note: int = 60, sample_rate: int = 48000):
        """Reset state for a new voice."""
        self.phase = 0
        self.set_note(note, sample_rate)
        self.waveform = 0


def generate_sine_block(state: CoeffGenState) -> np.ndarray:
    """
    Generate a 64-sample sine block at the current phase.

    Updates state.phase for continuity.

    Args:
        state: CoeffGenState with current phase

    Returns:
        64-sample audio block (int16)
    """
    audio = np.zeros(BLOCK_SIZE, dtype=np.int16)

    for i in range(BLOCK_SIZE):
        audio[i] = sine_q15(state.phase)
        state.phase = (state.phase + state.phase_inc) & 0xFFFFFFFF

    return audio


def generate_saw_block(state: CoeffGenState) -> np.ndarray:
    """Generate bandlimited sawtooth block."""
    audio = np.zeros(BLOCK_SIZE, dtype=np.float32)

    for i in range(BLOCK_SIZE):
        # Bandlimited sawtooth: sum of harmonics 1-16
        val = 0.0
        for h in range(1, 17):
            h_phase = (state.phase * h) & 0xFFFFFFFF
            val += sine_q15(h_phase) / h

        audio[i] = val / 2  # Scale down to prevent clipping
        state.phase = (state.phase + state.phase_inc) & 0xFFFFFFFF

    return np.clip(audio, -32768, 32767).astype(np.int16)


def generate_square_block(state: CoeffGenState) -> np.ndarray:
    """Generate bandlimited square block."""
    audio = np.zeros(BLOCK_SIZE, dtype=np.float32)

    for i in range(BLOCK_SIZE):
        # Bandlimited square: odd harmonics 1,3,5,...,15
        val = 0.0
        for h in [1, 3, 5, 7, 9, 11, 13, 15]:
            h_phase = (state.phase * h) & 0xFFFFFFFF
            val += sine_q15(h_phase) / h

        audio[i] = val
        state.phase = (state.phase + state.phase_inc) & 0xFFFFFFFF

    return np.clip(audio, -32768, 32767).astype(np.int16)


def generate_coeffs(state: CoeffGenState, headroom_bits: int = 1) -> np.ndarray:
    """
    Generate 64 coefficients for the current block.

    This is the core CCS function:
    1. Generate reference audio at current phase
    2. FWPT to get coefficients
    3. Return coefficients for operator pipeline

    Args:
        state: CoeffGenState (modified in place)
        headroom_bits: Headroom for FWPT/IDWPT

    Returns:
        64 coefficients (int16)
    """
    # Generate reference audio
    if state.waveform == 0:
        audio = generate_sine_block(state)
    elif state.waveform == 1:
        audio = generate_saw_block(state)
    elif state.waveform == 2:
        audio = generate_square_block(state)
    else:
        audio = generate_sine_block(state)

    # FWPT to get coefficients
    coeffs = fwpt_jit(audio.tolist(), headroom_bits=headroom_bits)

    return np.array(coeffs, dtype=np.int16)


def reconstruct_audio(coeffs: np.ndarray, headroom_bits: int = 1) -> np.ndarray:
    """
    Reconstruct audio from coefficients.

    Args:
        coeffs: 64 coefficients (int16)
        headroom_bits: Headroom for IDWPT

    Returns:
        64 audio samples (int16)
    """
    audio = idwpt_jit(coeffs.tolist(), headroom_bits=headroom_bits)
    return np.array(audio, dtype=np.int16)


def ccs_render_block(
    state: CoeffGenState,
    apply_operators=None,  # Optional: function(coeffs) -> coeffs
    headroom_bits: int = 1,
) -> np.ndarray:
    """
    Complete CCS render: generate → operators → reconstruct.

    This is the main entry point for CCS synthesis.

    Args:
        state: CoeffGenState (modified in place)
        apply_operators: Optional function to apply coefficient operators
        headroom_bits: Headroom for transforms

    Returns:
        64 audio samples (int16)
    """
    # Step 1: Generate coefficients
    coeffs = generate_coeffs(state, headroom_bits)

    # Step 2: Apply operators (if any)
    if apply_operators is not None:
        coeffs = apply_operators(coeffs)

    # Step 3: Reconstruct audio
    audio = reconstruct_audio(coeffs, headroom_bits)

    return audio


__all__ = [
    'CoeffGenState',
    'generate_coeffs',
    'reconstruct_audio',
    'ccs_render_block',
    'midi_to_phase_inc',
    'midi_to_freq',
]
