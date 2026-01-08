"""
Sample Playback Source (Phase 3 - Coefficient-Native Sample Playback)
======================================================================

Provides sample/rompler playback in the coefficient domain.

Per WSE-STD-210 Architecture:
  Audio samples → FWPT (once at load) → coefficient buffer
  Playback: coefficient interpolation → replicate to 512 → coeff_state

Key Features:
  - Pre-computed coefficient mode: FWPT at load time, not in hot path
  - Block interpolation for pitch shifting: stays in coefficient domain
  - No audio-mode fallback needed - CCS handles all pitch ratios
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from ..wavelet.fast_lifting import fwpt_jit

# Constants
VM_K = 8          # K-slots per block
NUM_LEAVES = 64   # Leaves in wavelet tree
COEFF_LEN = 512   # 64 leaves × 8 k-slots
BLOCK_SIZE = 64   # Audio samples per block


@dataclass
class SampleOscState:
    """
    Sample playback oscillator state (coefficient-native).

    Maintains playback position and loop parameters for sample-based synthesis.
    All playback happens in the coefficient domain with block interpolation.

    Modes:
    1. Coeff mode (use_coeff_buffer=True): Pre-computed coeffs with block interpolation
       - Supports pitch shifting via coefficient interpolation
       - No FWPT in hot path
    2. Audio mode (use_coeff_buffer=False): Legacy FWPT gateway (fallback)
    """
    # Sample buffer (loaded externally, int16)
    buffer: Optional[np.ndarray] = None

    # Pre-computed coefficient buffer (for coeff-domain playback)
    # Shape: [num_blocks, 64] where num_blocks = len(buffer) // BLOCK_SIZE
    coeff_buffer: Optional[np.ndarray] = None
    use_coeff_buffer: bool = False  # Use pre-computed coefficients

    # Playback position (Q16 fixed-point for sub-sample interpolation)
    # Used only in audio mode fallback
    position_q16: int = 0

    # Block position (Q16 fixed-point for sub-block interpolation)
    # Integer part = block index, fractional part = interpolation weight
    block_position_q16: int = 0

    # Playback rate (Q16: 65536 = 1.0x, 131072 = 2.0x, etc.)
    # Works in both coeff mode (block interpolation) and audio mode
    rate_q16: int = 65536

    # Loop parameters (in block indices when use_coeff_buffer=True)
    loop_start: int = 0      # Loop start block index
    loop_end: int = 0        # Loop end block index (0 = use buffer length)
    loop_enabled: bool = False

    # Playback state
    is_playing: bool = False
    is_finished: bool = False

    def reset(self, start_position: int = 0) -> None:
        """Reset playback to start position."""
        self.position_q16 = start_position << 16
        # For coeff mode, convert sample position to block position
        if self.use_coeff_buffer:
            block_idx = start_position // BLOCK_SIZE
            self.block_position_q16 = block_idx << 16
        else:
            self.block_position_q16 = 0
        self.is_playing = True
        self.is_finished = False

    def set_buffer(self, buffer: np.ndarray, loop_start: int = 0,
                   loop_end: int = 0, loop_enabled: bool = False,
                   precompute_coeffs: bool = False) -> None:
        """
        Load a sample buffer for playback.

        Args:
            buffer: Audio samples (int16)
            loop_start: Loop start sample index
            loop_end: Loop end sample index (0 = buffer length)
            loop_enabled: Enable looping
            precompute_coeffs: Pre-transform to coefficients (faster loops, no pitch shift)
        """
        self.buffer = buffer.astype(np.int16)
        self.loop_start = loop_start
        self.loop_end = loop_end if loop_end > 0 else len(buffer)
        self.loop_enabled = loop_enabled

        if precompute_coeffs:
            self._precompute_coefficients()
        else:
            self.coeff_buffer = None
            self.use_coeff_buffer = False

        self.reset()

    def _precompute_coefficients(self) -> None:
        """
        Pre-transform entire sample to coefficient domain.

        This eliminates FWPT from the hot path for looping samples.
        Trade-off: No pitch shifting (rate must be 1.0x).
        """
        if self.buffer is None:
            return

        num_blocks = len(self.buffer) // BLOCK_SIZE
        if num_blocks == 0:
            return

        # Allocate coefficient buffer: [num_blocks, 64 leaves]
        self.coeff_buffer = np.zeros((num_blocks, NUM_LEAVES), dtype=np.int16)

        # Transform each block
        for b in range(num_blocks):
            start = b * BLOCK_SIZE
            audio_block = self.buffer[start:start + BLOCK_SIZE]
            coeffs = fwpt_jit(audio_block.tolist(), headroom_bits=1)
            self.coeff_buffer[b, :] = np.clip(coeffs, -32768, 32767)

        self.use_coeff_buffer = True
        # Convert loop points to block indices
        self.loop_start = self.loop_start // BLOCK_SIZE
        self.loop_end = min(num_blocks, (self.loop_end + BLOCK_SIZE - 1) // BLOCK_SIZE)

    def set_pitch_ratio(self, ratio: float) -> None:
        """
        Set playback rate as pitch ratio (1.0 = original pitch).

        Pitch shifting works in coefficient mode via block interpolation.
        No fallback to audio mode needed.
        """
        self.rate_q16 = int(ratio * 65536)

    def set_pitch_semitones(self, semitones: float) -> None:
        """Set playback pitch shift in semitones."""
        ratio = 2.0 ** (semitones / 12.0)
        self.set_pitch_ratio(ratio)


def read_coeff_block(state: SampleOscState, output_512: np.ndarray) -> None:
    """
    Read coefficients with interpolation for pitch shifting (coefficient-native).

    Uses Q16 fixed-point block position for sub-block precision.
    Linearly interpolates between adjacent coefficient blocks.

    Args:
        state: SampleOscState with coeff_buffer loaded
        output_512: Output array [512] to fill with coefficients
    """
    if state.coeff_buffer is None or not state.is_playing:
        output_512[:] = 0
        return

    num_blocks = len(state.coeff_buffer)
    loop_end = state.loop_end if state.loop_end > 0 else num_blocks

    # Extract integer and fractional parts of block position
    block_int = state.block_position_q16 >> 16
    block_frac = state.block_position_q16 & 0xFFFF  # 0-65535 maps to 0.0-1.0

    # Handle looping/bounds for current block
    if block_int >= loop_end:
        if state.loop_enabled:
            loop_len = loop_end - state.loop_start
            if loop_len > 0:
                block_int = state.loop_start + ((block_int - state.loop_start) % loop_len)
                state.block_position_q16 = (block_int << 16) | block_frac
            else:
                state.is_finished = True
                state.is_playing = False
                output_512[:] = 0
                return
        else:
            state.is_finished = True
            state.is_playing = False
            output_512[:] = 0
            return

    # Get next block index for interpolation (with loop wrapping)
    block_next = block_int + 1
    if block_next >= loop_end:
        if state.loop_enabled:
            loop_len = loop_end - state.loop_start
            if loop_len > 0:
                block_next = state.loop_start + ((block_next - state.loop_start) % loop_len)
            else:
                block_next = block_int  # No interpolation at end
        else:
            block_next = block_int  # Clamp at end

    # Read coefficients for current and next blocks
    coeffs_0 = state.coeff_buffer[block_int]
    coeffs_1 = state.coeff_buffer[block_next]

    # Interpolate and replicate across K time slots
    # out = coeffs_0 + (coeffs_1 - coeffs_0) * frac / 65536
    for leaf in range(NUM_LEAVES):
        c0 = int(coeffs_0[leaf])
        c1 = int(coeffs_1[leaf])
        interp = c0 + (((c1 - c0) * block_frac) >> 16)
        interp = max(-32768, min(32767, interp))  # Saturate
        for k in range(VM_K):
            output_512[leaf * VM_K + k] = interp

    # Advance block position by rate
    state.block_position_q16 += state.rate_q16


def read_sample_block(state: SampleOscState, num_samples: int = BLOCK_SIZE) -> np.ndarray:
    """
    Read a block of samples with linear interpolation and loop handling.

    Args:
        state: SampleOscState with buffer and position
        num_samples: Number of samples to read (default 64)

    Returns:
        audio_block: [num_samples] int16 array
    """
    output = np.zeros(num_samples, dtype=np.int16)

    if state.buffer is None or not state.is_playing:
        return output

    buf_len = len(state.buffer)
    loop_end = state.loop_end if state.loop_end > 0 else buf_len

    for i in range(num_samples):
        # Extract integer and fractional parts of position
        pos_int = state.position_q16 >> 16
        pos_frac = state.position_q16 & 0xFFFF

        # Check bounds
        if pos_int >= loop_end:
            if state.loop_enabled:
                # Wrap to loop start
                loop_len = loop_end - state.loop_start
                if loop_len > 0:
                    pos_int = state.loop_start + ((pos_int - state.loop_start) % loop_len)
                    state.position_q16 = (pos_int << 16) | pos_frac
                else:
                    state.is_finished = True
                    state.is_playing = False
                    break
            else:
                # One-shot: stop at end
                state.is_finished = True
                state.is_playing = False
                break

        if pos_int >= buf_len - 1:
            # At end of buffer
            output[i] = state.buffer[min(pos_int, buf_len - 1)]
        else:
            # Linear interpolation between samples
            s0 = int(state.buffer[pos_int])
            s1 = int(state.buffer[pos_int + 1])
            # Interpolate: out = s0 + (s1 - s0) * frac / 65536
            interp = s0 + (((s1 - s0) * pos_frac) >> 16)
            output[i] = np.clip(interp, -32768, 32767)

        # Advance position
        state.position_q16 += state.rate_q16

    return output


def sample_generate_coeffs(state: SampleOscState, output_512: np.ndarray) -> None:
    """
    Generate coefficients from sample playback (coefficient-native).

    Primary path (coeff mode):
    - Pre-computed coefficients with block interpolation
    - Supports pitch shifting via coefficient interpolation
    - No FWPT in hot path

    Fallback path (audio mode):
    - FWPT gateway for non-precomputed samples
    - FWPT transform every block

    Args:
        state: SampleOscState with active sample buffer
        output_512: Output array [512] to fill with coefficients
    """
    # Primary path: coefficient-domain playback with interpolation
    if state.use_coeff_buffer and state.coeff_buffer is not None:
        read_coeff_block(state, output_512)
        return

    # Fallback: Audio path with FWPT gateway
    # Read 64 audio samples with pitch-shifting interpolation
    audio_block = read_sample_block(state, BLOCK_SIZE)

    # Transform to coefficients via FWPT (analysis transform)
    coeffs_64 = fwpt_jit(audio_block.tolist(), headroom_bits=1)

    # Replicate across K time slots
    for k in range(VM_K):
        for leaf in range(NUM_LEAVES):
            output_512[leaf * VM_K + k] = np.clip(coeffs_64[leaf], -32768, 32767)


def sample_generate_coeffs_windowed(
    state: SampleOscState,
    output_512: np.ndarray,
    window: Optional[np.ndarray] = None
) -> None:
    """
    Generate coefficients with optional windowing for smoother transitions.

    Args:
        state: SampleOscState with active sample buffer
        output_512: Output array [512] to fill with coefficients
        window: Optional [64] window function (default: None = no window)
    """
    # Read 64 audio samples
    audio_block = read_sample_block(state, BLOCK_SIZE)

    # Apply window if provided
    if window is not None:
        audio_block = (audio_block.astype(np.int32) * window.astype(np.int32)) >> 15
        audio_block = np.clip(audio_block, -32768, 32767).astype(np.int16)

    # Transform via FWPT
    coeffs_64 = fwpt_jit(audio_block.tolist(), headroom_bits=1)

    # Replicate across K time slots
    for k in range(VM_K):
        for leaf in range(NUM_LEAVES):
            output_512[leaf * VM_K + k] = np.clip(coeffs_64[leaf], -32768, 32767)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_hann_window(size: int = BLOCK_SIZE) -> np.ndarray:
    """Create a Hann window for sample transitions."""
    n = np.arange(size)
    window = 0.5 * (1 - np.cos(2 * np.pi * n / (size - 1)))
    return (window * 32767).astype(np.int16)


def compute_playback_rate(original_note: int, target_note: int) -> int:
    """
    Compute playback rate (Q16) to pitch-shift sample.

    Args:
        original_note: MIDI note of original sample
        target_note: MIDI note to play at

    Returns:
        rate_q16: Playback rate in Q16 format
    """
    semitones = target_note - original_note
    ratio = 2.0 ** (semitones / 12.0)
    return int(ratio * 65536)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SampleOscState',
    'read_coeff_block',
    'read_sample_block',
    'sample_generate_coeffs',
    'sample_generate_coeffs_windowed',
    'create_hann_window',
    'compute_playback_rate',
    # Constants
    'VM_K',
    'NUM_LEAVES',
    'COEFF_LEN',
    'BLOCK_SIZE',
]
