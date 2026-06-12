"""
Granular Synthesis Engine (Coefficient-Domain)
==============================================

Provides granular synthesis in the coefficient domain for CCS.

Architecture:
- Grains are extracted as coefficient blocks from a sample buffer
- Multiple simultaneous grains are summed in coefficient space
- Grain scheduling uses random or regular patterns
- Pitch shifting via coefficient interpolation

Key Features:
- Coefficient-domain operation (no audio-rate FWPT in hot path)
- Up to 8 simultaneous grains per voice
- Spray/scatter for position randomization
- Configurable grain size, density, and overlap
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from ..wavelet.fast_lifting import fwpt_jit
from .sample_source import SampleOscState, VM_K, NUM_LEAVES, COEFF_LEN

# Import canonical constants from primitives (single source of truth)
from ..primitives import Q15_ONE, Q15_HALF, BLOCK_SIZE
from ..primitives import Q15_NEG_ONE, Q15_SCALE, Q16_ONE


# =============================================================================
# CONSTANTS
# =============================================================================

MAX_GRAINS = 8           # Maximum simultaneous grains
MIN_GRAIN_BLOCKS = 2     # Minimum grain size (2 blocks = 128 samples)
MAX_GRAIN_BLOCKS = 64    # Maximum grain size (64 blocks = 4096 samples)
# Q15_ONE and Q15_HALF imported from primitives


# =============================================================================
# GRAIN STATE
# =============================================================================

@dataclass
class GrainState:
    """
    State for a single grain.

    Grains read from a pre-computed coefficient buffer with optional
    pitch shifting and amplitude envelope.
    """
    # Source position in coefficient buffer
    start_block: int = 0
    current_block_q16: int = 0   # Q16 for sub-block precision

    # Grain parameters
    length_blocks: int = 16      # Duration in blocks
    rate_q16: int = Q16_ONE        # Playback rate (65536 = 1.0x)
    pan_q15: int = Q15_HALF      # Stereo pan (0 = left, 32767 = right)

    # Amplitude envelope (attack/release ramps)
    attack_blocks: int = 2
    release_blocks: int = 2
    amplitude_q15: int = Q15_ONE

    # State
    blocks_played: int = 0
    is_active: bool = False

    def trigger(
        self,
        start_block: int,
        length_blocks: int,
        rate_q16: int = Q16_ONE,
        amplitude_q15: int = Q15_ONE,
    ) -> None:
        """Trigger a new grain."""
        self.start_block = start_block
        self.current_block_q16 = start_block << 16
        self.length_blocks = max(MIN_GRAIN_BLOCKS, min(MAX_GRAIN_BLOCKS, length_blocks))
        self.rate_q16 = rate_q16
        self.amplitude_q15 = amplitude_q15
        self.blocks_played = 0
        self.is_active = True

        # Set envelope ramps (10% of length for attack/release)
        ramp_blocks = max(1, length_blocks // 10)
        self.attack_blocks = ramp_blocks
        self.release_blocks = ramp_blocks

    def get_envelope_q15(self) -> int:
        """Get current envelope amplitude (Q15)."""
        if not self.is_active:
            return 0

        # Attack ramp (starts at 1/attack_blocks, not 0)
        if self.blocks_played < self.attack_blocks:
            return ((self.blocks_played + 1) * Q15_ONE) // (self.attack_blocks + 1)

        # Release ramp
        blocks_remaining = self.length_blocks - self.blocks_played
        if blocks_remaining <= self.release_blocks:
            if self.release_blocks == 0:
                return 0
            return (blocks_remaining * Q15_ONE) // self.release_blocks

        # Sustain
        return Q15_ONE

    def advance(self) -> bool:
        """
        Advance grain by one block.

        Returns True if grain is still active.
        """
        if not self.is_active:
            return False

        self.current_block_q16 += self.rate_q16
        self.blocks_played += 1

        if self.blocks_played >= self.length_blocks:
            self.is_active = False

        return self.is_active


# =============================================================================
# GRANULAR ENGINE STATE
# =============================================================================

@dataclass
class GranularState:
    """
    State for granular synthesis engine.

    Manages multiple simultaneous grains with configurable scheduling.
    """
    # Grain pool
    grains: List[GrainState] = field(default_factory=lambda: [GrainState() for _ in range(MAX_GRAINS)])

    # Source buffer (pre-computed coefficients)
    # Shape: [num_blocks, 64] where num_blocks = len(audio) // BLOCK_SIZE
    coeff_buffer: Optional[np.ndarray] = None
    num_source_blocks: int = 0

    # Scan position (where to spawn new grains)
    scan_position: int = 0          # In blocks (integer part)
    scan_position_q16: int = 0      # Q16 accumulator for fractional advancement
    scan_rate_q16: int = Q16_ONE      # Rate of scan movement (65536 = 1 block/block)

    # Grain scheduling parameters
    density_q15: int = Q15_HALF     # Probability of spawning grain per block
    grain_length: int = 16          # Default grain length in blocks
    grain_rate_q16: int = Q16_ONE     # Playback rate for new grains
    spray_q15: int = 0              # Random scatter around scan position

    # Output mixing
    mix_gain_q15: int = Q15_ONE     # Overall output gain

    # PRNG for grain scheduling
    prng_state: int = 0x12345678

    # Output buffer
    _output_512: np.ndarray = field(
        default_factory=lambda: np.zeros(COEFF_LEN, dtype=np.int16)
    )

    def set_source(self, audio_buffer: np.ndarray, precompute: bool = True) -> None:
        """
        Set the source audio for granular synthesis.

        Args:
            audio_buffer: Audio samples (int16)
            precompute: If True, pre-compute coefficient buffer
        """
        if precompute:
            self._precompute_coefficients(audio_buffer)
        else:
            # Store raw audio and compute on demand (slower)
            self.coeff_buffer = None
            self.num_source_blocks = 0

    def _precompute_coefficients(self, audio_buffer: np.ndarray) -> None:
        """Pre-transform entire sample to coefficient domain."""
        audio_buffer = audio_buffer.astype(np.int16)
        num_blocks = len(audio_buffer) // BLOCK_SIZE

        if num_blocks == 0:
            return

        self.coeff_buffer = np.zeros((num_blocks, NUM_LEAVES), dtype=np.int16)

        for b in range(num_blocks):
            start = b * BLOCK_SIZE
            audio_block = audio_buffer[start:start + BLOCK_SIZE]
            coeffs = fwpt_jit(audio_block.tolist(), headroom_bits=1)
            self.coeff_buffer[b, :] = np.clip(coeffs, Q15_NEG_ONE, Q15_ONE)

        self.num_source_blocks = num_blocks

    def reset(self) -> None:
        """Reset granular engine state."""
        for grain in self.grains:
            grain.is_active = False
        self.scan_position = 0
        self.scan_position_q16 = 0
        self._output_512.fill(0)


# =============================================================================
# PRNG
# =============================================================================

def _prng_step(state: int) -> tuple:
    """xorshift32 PRNG step. Returns (new_state, random_value)."""
    x = state & 0xFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17)
    x ^= (x << 5) & 0xFFFFFFFF
    return x, x & 0xFFFF  # Return 16-bit random value


# =============================================================================
# GRAIN SCHEDULING
# =============================================================================

def _should_spawn_grain(state: GranularState) -> bool:
    """Determine if a new grain should be spawned this block."""
    state.prng_state, rand = _prng_step(state.prng_state)
    return rand < (state.density_q15 >> 1)  # Scale to 0-32767 range


def _find_free_grain(state: GranularState) -> Optional[GrainState]:
    """Find an inactive grain slot."""
    for grain in state.grains:
        if not grain.is_active:
            return grain
    return None


def _compute_spawn_position(state: GranularState) -> int:
    """Compute position for new grain with optional spray."""
    base_pos = state.scan_position

    if state.spray_q15 > 0:
        # Add random scatter
        state.prng_state, rand = _prng_step(state.prng_state)
        spray_range = (state.spray_q15 * state.num_source_blocks) >> 15
        offset = ((rand - Q15_SCALE) * spray_range) >> 15
        base_pos += offset

    # Clamp to valid range
    return max(0, min(base_pos, state.num_source_blocks - 1))


# =============================================================================
# GRAIN RENDERING
# =============================================================================

def _read_grain_coeffs(
    grain: GrainState,
    coeff_buffer: np.ndarray,
    num_blocks: int,
    output_64: np.ndarray
) -> None:
    """
    Read coefficients for a single grain with interpolation.

    Args:
        grain: Grain state
        coeff_buffer: [num_blocks, 64] coefficient buffer
        num_blocks: Number of blocks in buffer
        output_64: Output array [64] to fill
    """
    # Get current position
    block_int = grain.current_block_q16 >> 16
    block_frac = grain.current_block_q16 & 0xFFFF

    # Clamp to buffer bounds
    block_int = max(0, min(block_int, num_blocks - 1))
    block_next = min(block_int + 1, num_blocks - 1)

    # Get coefficients for interpolation
    coeffs_0 = coeff_buffer[block_int]
    coeffs_1 = coeff_buffer[block_next]

    # Apply envelope
    env = grain.get_envelope_q15()
    amp = (grain.amplitude_q15 * env) >> 15

    # Interpolate with amplitude
    for i in range(NUM_LEAVES):
        c0 = int(coeffs_0[i])
        c1 = int(coeffs_1[i])
        interp = c0 + (((c1 - c0) * block_frac) >> 16)
        output_64[i] = (interp * amp) >> 15


# =============================================================================
# MAIN GRANULAR SYNTHESIS FUNCTION
# =============================================================================

def granular_generate_coeffs(state: GranularState, output_512: np.ndarray) -> None:
    """
    Generate coefficients from granular synthesis.

    Processes all active grains, spawns new grains as needed,
    and advances the scan position.

    Args:
        state: GranularState with source buffer loaded
        output_512: Output array [512] to fill with coefficients
    """
    if state.coeff_buffer is None or state.num_source_blocks == 0:
        output_512[:] = 0
        return

    # Accumulator for mixing (int32 to prevent overflow)
    mix_acc = np.zeros(NUM_LEAVES, dtype=np.int32)
    grain_coeffs = np.zeros(NUM_LEAVES, dtype=np.int16)
    active_count = 0

    # Process active grains
    for grain in state.grains:
        if not grain.is_active:
            continue

        # Read grain coefficients
        _read_grain_coeffs(
            grain,
            state.coeff_buffer,
            state.num_source_blocks,
            grain_coeffs
        )

        # Add to mix
        mix_acc += grain_coeffs.astype(np.int32)
        active_count += 1

        # Advance grain
        grain.advance()

    # Try to spawn new grain
    if _should_spawn_grain(state):
        free_grain = _find_free_grain(state)
        if free_grain is not None:
            spawn_pos = _compute_spawn_position(state)
            free_grain.trigger(
                start_block=spawn_pos,
                length_blocks=state.grain_length,
                rate_q16=state.grain_rate_q16,
            )

    # Advance scan position (using Q16 accumulator for fractional rates)
    state.scan_position_q16 += state.scan_rate_q16
    state.scan_position = state.scan_position_q16 >> 16
    if state.scan_position >= state.num_source_blocks:
        state.scan_position = 0  # Loop back
        state.scan_position_q16 = 0

    # Apply mix gain and normalize by active grain count
    if active_count > 0:
        # Divide by sqrt(active_count) for energy-preserving mix
        # Approximate: divide by active_count, scale up slightly
        mix_acc = (mix_acc * state.mix_gain_q15) >> 15
        mix_acc = mix_acc // max(1, active_count)

    # Saturate and replicate to 512 coefficients
    mix_sat = np.clip(mix_acc, Q15_NEG_ONE, Q15_ONE).astype(np.int16)

    for leaf in range(NUM_LEAVES):
        val = int(mix_sat[leaf])
        for k in range(VM_K):
            output_512[leaf * VM_K + k] = val


def granular_set_parameters(
    state: GranularState,
    density: float = 0.5,
    grain_size_ms: float = 50.0,
    spray: float = 0.0,
    pitch_ratio: float = 1.0,
    scan_rate: float = 1.0,
) -> None:
    """
    Set granular synthesis parameters.

    Args:
        state: GranularState to configure
        density: Grain spawn probability (0-1)
        grain_size_ms: Grain duration in milliseconds
        spray: Random scatter factor (0-1)
        pitch_ratio: Grain playback rate (1.0 = original)
        scan_rate: Scan speed (1.0 = real-time, 0.5 = half speed)
    """
    # Block time in ms
    block_time_ms = (BLOCK_SIZE / 48000) * 1000  # ~1.33ms at 48kHz

    state.density_q15 = int(min(1.0, max(0.0, density)) * Q15_ONE)
    state.grain_length = max(MIN_GRAIN_BLOCKS, int(grain_size_ms / block_time_ms))
    state.spray_q15 = int(min(1.0, max(0.0, spray)) * Q15_ONE)
    state.grain_rate_q16 = int(pitch_ratio * Q16_ONE)
    state.scan_rate_q16 = int(scan_rate * Q16_ONE)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_granular_from_sample(
    audio: np.ndarray,
    density: float = 0.3,
    grain_size_ms: float = 80.0,
    spray: float = 0.2,
) -> GranularState:
    """
    Create a GranularState from an audio sample.

    Args:
        audio: Audio samples (int16)
        density: Grain spawn probability
        grain_size_ms: Grain duration
        spray: Random scatter

    Returns:
        Configured GranularState
    """
    state = GranularState()
    state.set_source(audio)
    granular_set_parameters(state, density=density, grain_size_ms=grain_size_ms, spray=spray)
    return state


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'GrainState',
    'GranularState',
    'granular_generate_coeffs',
    'granular_set_parameters',
    'create_granular_from_sample',
    # Constants
    'MAX_GRAINS',
    'MIN_GRAIN_BLOCKS',
    'MAX_GRAIN_BLOCKS',
]
