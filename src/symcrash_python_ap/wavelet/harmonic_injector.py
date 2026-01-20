"""
Harmonic Target Injector v0 (Baby Fourier)
==========================================

Generates coefficient-space targets for classic wave types (sine, saw, pulse, triangle).
Maps partials to wavelet leaves and provides sparse injection vectors.

This module implements the spec from:
  docs/08 - Addendum Harmonic Target Injector v0 (Baby Fourier).md

Key design:
- Treats classic wave types as coefficient-space targets, NOT parallel audio synth
- Uses existing INJECT lane in Symbol Engine
- Maps harmonic partials to wavelet leaves via frequency bucketing
- No FFT required - just deterministic frequency-to-leaf mapping
"""

from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np

from ..fixed import sat16, mul_q15_sat, Q15_ONE, Q15_SHIFT

# Import from primitives (single source of truth)
from ..primitives import NUM_LEAVES, SAMPLE_RATE_HZ

# Constants
LEAF_COUNT = NUM_LEAVES   # D=6 master tree: 64 leaves
NODE_SIZE = 4             # 256 / 64 = 4 coefficients per leaf
COEFF_LEN = LEAF_COUNT * NODE_SIZE  # 256
SAMPLE_RATE = SAMPLE_RATE_HZ  # 48000
NYQUIST = SAMPLE_RATE // 2


# =============================================================================
# WAVE TYPES
# =============================================================================

class WaveType(IntEnum):
    """Wave type enum for harmonic target generation."""
    SINE = 0
    SAW = 1
    PULSE = 2      # 50% duty (square-like, odd harmonics only)
    TRIANGLE = 3
    NOISE = 4      # Reserved - handled by NoiseBand operator


# =============================================================================
# LEAF FREQUENCY MAPPING
# =============================================================================

def build_leaf_freq_table(sample_rate: int = SAMPLE_RATE) -> Tuple[List[float], List[float]]:
    """
    Build frequency mapping tables for 32 leaves.

    Returns:
        leaf_center_hz: Center frequency of each leaf
        leaf_edges_hz: Edge frequencies (33 values, leaf i covers [edges[i], edges[i+1]])
    """
    nyquist = sample_rate / 2

    # Uniform leaf distribution (for balanced tree)
    # Each leaf covers nyquist/32 Hz
    band_width = nyquist / LEAF_COUNT

    leaf_edges_hz = [i * band_width for i in range(LEAF_COUNT + 1)]
    leaf_center_hz = [(leaf_edges_hz[i] + leaf_edges_hz[i + 1]) / 2 for i in range(LEAF_COUNT)]

    return leaf_center_hz, leaf_edges_hz


# Precompute tables
LEAF_CENTER_HZ, LEAF_EDGES_HZ = build_leaf_freq_table()


def freq_to_leaf(freq_hz: float) -> int:
    """
    Map frequency to leaf index using threshold table.

    Uses simple bucket lookup - FPGA-friendly (compare chain or binary search).
    """
    if freq_hz < 0:
        return 0
    if freq_hz >= NYQUIST:
        return LEAF_COUNT - 1

    # Simple linear search (could be binary for FPGA)
    for i in range(LEAF_COUNT):
        if freq_hz < LEAF_EDGES_HZ[i + 1]:
            return i

    return LEAF_COUNT - 1


def freq_to_leaf_and_coeff(freq_hz: float) -> Tuple[int, int]:
    """
    Map frequency to (leaf_index, coeff_index) for precise frequency encoding.

    Each leaf covers a frequency band, and the 8 coefficients within encode
    position within that band:
    - coeff 0 = low end of band
    - coeff 7 = high end of band

    This allows sub-band frequency resolution, which is critical for
    proper pitch reproduction in synthesis.

    Returns:
        (leaf_idx, coeff_idx) where coeff_idx is 0-7
    """
    if freq_hz < 0:
        return 0, 0
    if freq_hz >= NYQUIST:
        return LEAF_COUNT - 1, NODE_SIZE - 1

    # Find leaf
    leaf_idx = freq_to_leaf(freq_hz)

    # Compute position within leaf's frequency band
    band_low = LEAF_EDGES_HZ[leaf_idx]
    band_high = LEAF_EDGES_HZ[leaf_idx + 1]
    band_width = band_high - band_low

    if band_width <= 0:
        return leaf_idx, 0

    # Normalize position in band: 0 = low edge, 1 = high edge
    pos_in_band = (freq_hz - band_low) / band_width
    pos_in_band = max(0.0, min(1.0, pos_in_band))

    # Map to coefficient index (0-7)
    # Use floor to get stable mapping
    coeff_idx = int(pos_in_band * NODE_SIZE)
    coeff_idx = max(0, min(NODE_SIZE - 1, coeff_idx))

    return leaf_idx, coeff_idx


# =============================================================================
# NOTE TO FREQUENCY CONVERSION
# =============================================================================

def note_to_freq(note_u8: int, cents_i16: int = 0) -> float:
    """
    Convert MIDI note + cents to frequency.

    Args:
        note_u8: MIDI note number (0-127, 69 = A4 = 440 Hz)
        cents_i16: Cents offset (-8192 to +8192 typical, 100 cents = 1 semitone)

    Returns:
        Frequency in Hz
    """
    # A4 = 440 Hz at MIDI note 69
    # Each semitone is 100 cents
    semitones = (note_u8 - 69) + cents_i16 / 100.0
    return 440.0 * (2.0 ** (semitones / 12.0))


# =============================================================================
# AMPLITUDE LAWS PER WAVE TYPE
# =============================================================================

def get_partial_amplitude(wave_type: WaveType, k: int, brightness_q15: int = Q15_ONE) -> int:
    """
    Get amplitude for partial k based on wave type.

    Args:
        wave_type: Wave type enum
        k: Partial number (1 = fundamental, 2 = 2nd harmonic, etc.)
        brightness_q15: Brightness/tilt control (0 = dark, Q15_ONE = full brightness)

    Returns:
        Amplitude in Q15 format
    """
    if k < 1:
        return 0

    # Base amplitude law
    if wave_type == WaveType.SINE:
        # Only fundamental
        base_amp = Q15_ONE if k == 1 else 0

    elif wave_type == WaveType.SAW:
        # a_k = 1/k
        base_amp = Q15_ONE // k

    elif wave_type == WaveType.PULSE:
        # Odd harmonics only with a_k = 1/k (square wave approximation)
        if k % 2 == 1:  # Odd
            base_amp = Q15_ONE // k
        else:
            base_amp = 0

    elif wave_type == WaveType.TRIANGLE:
        # Odd harmonics only with a_k = 1/k^2
        if k % 2 == 1:  # Odd
            base_amp = Q15_ONE // (k * k)
        else:
            base_amp = 0
    else:
        base_amp = 0

    # Apply brightness rolloff
    # brightness_q15 controls high partial attenuation
    # At full brightness, no rolloff. At 0, strong rolloff.
    if k > 1 and base_amp > 0:
        # Exponential rolloff based on partial number and brightness
        # rolloff = brightness^(k-1) approximately
        # Simplified: attenuate higher partials when brightness is low
        inv_brightness = Q15_ONE - brightness_q15
        rolloff_factor = inv_brightness * (k - 1) // 16  # Tuned for musical results
        rolloff_factor = min(rolloff_factor, Q15_ONE)

        # Apply rolloff
        attenuation = Q15_ONE - rolloff_factor
        base_amp = mul_q15_sat(base_amp, attenuation)

    return base_amp


# =============================================================================
# HARMONIC TARGET GENERATOR
# =============================================================================

@dataclass
class HarmonicTarget:
    """
    Sparse representation of harmonic target.

    Contains only the active leaves and their target amplitudes.
    """
    # Active leaves (indices 0-31)
    leaf_indices: List[int] = field(default_factory=list)

    # Target amplitude per leaf (Q15)
    leaf_amplitudes: List[int] = field(default_factory=list)

    # Which coefficient within each leaf to target (0-7)
    # v0: use coeff 0, later can add phase encoding
    coeff_indices: List[int] = field(default_factory=list)

    # Full dense coefficient target (optional, for debugging)
    dense_target: Optional[List[int]] = None


@dataclass
class HarmonicInjectorState:
    """Per-voice state for harmonic injection."""
    # Current note
    note_u8: int = 60  # Middle C
    cents_i16: int = 0

    # Wave parameters
    wave_type: WaveType = WaveType.SAW
    partials_n: int = 8
    brightness_q15: int = Q15_ONE

    # Injection control
    harmonic_dial_q15: int = 0  # η: how much harmonic target participates
    pull_strength_q15: int = 0x2000  # α: convergence speed per block

    # Optional phase state for v0+ (analog drift)
    phase_acc_u32: int = 0
    drift_u16: int = 0


class HarmonicTargetGenerator:
    """
    Generates coefficient-space targets for classic wave types.

    Maps harmonic partials to wavelet leaves and produces sparse injection vectors.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.nyquist = sample_rate // 2

        # Precompute leaf frequency tables
        self.leaf_center_hz, self.leaf_edges_hz = build_leaf_freq_table(sample_rate)

        # Per-voice state
        self.voice_state: Dict[int, HarmonicInjectorState] = {}

    def get_voice_state(self, voice_id: int) -> HarmonicInjectorState:
        """Get or create state for a voice."""
        if voice_id not in self.voice_state:
            self.voice_state[voice_id] = HarmonicInjectorState()
        return self.voice_state[voice_id]

    def set_note(self, voice_id: int, note_u8: int, cents_i16: int = 0):
        """Set the note for a voice."""
        state = self.get_voice_state(voice_id)
        state.note_u8 = note_u8
        state.cents_i16 = cents_i16

    def set_wave_params(self, voice_id: int,
                        wave_type: WaveType = WaveType.SAW,
                        partials_n: int = 8,
                        brightness_q15: int = Q15_ONE):
        """Set wave generation parameters."""
        state = self.get_voice_state(voice_id)
        state.wave_type = wave_type
        state.partials_n = min(partials_n, 16)  # Cap at 16 partials
        state.brightness_q15 = brightness_q15

    def set_injection_params(self, voice_id: int,
                             harmonic_dial_q15: int = Q15_ONE // 2,
                             pull_strength_q15: int = 0x2000):
        """Set injection control parameters."""
        state = self.get_voice_state(voice_id)
        state.harmonic_dial_q15 = harmonic_dial_q15
        state.pull_strength_q15 = pull_strength_q15

    def generate_target(self, voice_id: int,
                        include_dense: bool = False) -> HarmonicTarget:
        """
        Generate harmonic target for a voice.

        Uses freq_to_leaf_and_coeff for precise frequency encoding within
        each leaf's 8-coefficient structure. This is critical for proper
        pitch reproduction: the coefficient index encodes the position
        within the frequency band, allowing the synthesizer to reconstruct
        the correct frequency.

        Args:
            voice_id: Voice identifier
            include_dense: If True, also generate full 256-coeff dense target

        Returns:
            HarmonicTarget with sparse leaf/amplitude mapping
        """
        state = self.get_voice_state(voice_id)

        # Get fundamental frequency
        f0 = note_to_freq(state.note_u8, state.cents_i16)

        # Build sparse target with precise frequency mapping
        leaf_indices = []
        leaf_amplitudes = []
        coeff_indices = []

        # Track per-(leaf, coeff) energy for proper accumulation
        # Key: (leaf_idx, coeff_idx), Value: accumulated amplitude
        coeff_energy: Dict[Tuple[int, int], int] = {}

        for k in range(1, state.partials_n + 1):
            # Partial frequency
            f_k = k * f0

            # Skip if above Nyquist
            if f_k >= self.nyquist * 0.95:
                break

            # Get amplitude for this partial
            amp_k = get_partial_amplitude(
                state.wave_type, k, state.brightness_q15
            )

            if amp_k < 10:
                continue

            # Map to (leaf, coeff) for precise frequency encoding
            leaf_idx, coeff_idx = freq_to_leaf_and_coeff(f_k)

            # Accumulate energy at this specific (leaf, coeff)
            key = (leaf_idx, coeff_idx)
            if key in coeff_energy:
                coeff_energy[key] = sat16(coeff_energy[key] + amp_k)
            else:
                coeff_energy[key] = amp_k

        # Convert to sorted lists
        for (leaf_idx, coeff_idx) in sorted(coeff_energy.keys()):
            leaf_indices.append(leaf_idx)
            leaf_amplitudes.append(coeff_energy[(leaf_idx, coeff_idx)])
            coeff_indices.append(coeff_idx)

        target = HarmonicTarget(
            leaf_indices=leaf_indices,
            leaf_amplitudes=leaf_amplitudes,
            coeff_indices=coeff_indices
        )

        # Optionally generate dense target
        if include_dense:
            dense = [0] * COEFF_LEN
            for leaf_idx, amp, coeff_idx in zip(
                leaf_indices, leaf_amplitudes, coeff_indices
            ):
                idx = leaf_idx * NODE_SIZE + coeff_idx
                dense[idx] = sat16(dense[idx] + amp)
            target.dense_target = dense

        return target

    def apply_injection(self, voice_id: int,
                        current_coeffs: List[int],
                        target: Optional[HarmonicTarget] = None) -> List[int]:
        """
        Apply harmonic injection to coefficient vector.

        Implements: b_t += α * η * (c_harm - c_t) for targeted coeffs

        Args:
            voice_id: Voice identifier
            current_coeffs: Current coefficient state (256 values)
            target: Pre-computed target (if None, generates new one)

        Returns:
            Updated coefficient vector
        """
        state = self.get_voice_state(voice_id)

        # Skip if harmonic dial is off
        if state.harmonic_dial_q15 < 10:
            return current_coeffs

        # Generate target if not provided
        if target is None:
            target = self.generate_target(voice_id)

        # Make a copy to modify
        result = list(current_coeffs)

        # Combined strength = α * η
        combined_strength = mul_q15_sat(
            state.pull_strength_q15,
            state.harmonic_dial_q15
        )

        if combined_strength < 10:
            return result

        # Apply sparse injection
        for leaf_idx, target_amp, coeff_idx in zip(
            target.leaf_indices,
            target.leaf_amplitudes,
            target.coeff_indices
        ):
            idx = leaf_idx * NODE_SIZE + coeff_idx
            if idx >= COEFF_LEN:
                continue

            # Delta = target - current
            delta = target_amp - result[idx]

            # Update: c += α * η * delta
            update = mul_q15_sat(delta, combined_strength)
            result[idx] = sat16(result[idx] + update)

        return result

    def generate_dense_target(self, voice_id: int) -> List[int]:
        """
        Generate full dense 256-coefficient target vector.

        This spreads energy across all 8 coefficients per leaf for richer synthesis.
        """
        state = self.get_voice_state(voice_id)

        # Get fundamental frequency
        f0 = note_to_freq(state.note_u8, state.cents_i16)

        # Initialize dense target
        dense = [0] * COEFF_LEN

        for k in range(1, state.partials_n + 1):
            # Partial frequency
            f_k = k * f0

            # Skip if above Nyquist
            if f_k >= self.nyquist * 0.95:
                break

            # Get amplitude for this partial
            amp_k = get_partial_amplitude(
                state.wave_type, k, state.brightness_q15
            )

            if amp_k < 10:
                continue

            # Map to leaf
            leaf_idx = freq_to_leaf(f_k)

            # Calculate position within leaf band for frequency encoding
            leaf_low = self.leaf_edges_hz[leaf_idx]
            leaf_high = self.leaf_edges_hz[leaf_idx + 1]
            leaf_span = leaf_high - leaf_low

            # Position within band (0-1)
            if leaf_span > 0:
                pos_in_band = (f_k - leaf_low) / leaf_span
            else:
                pos_in_band = 0.5

            # Map to coefficient index (0-7)
            # This encodes the frequency position within the band
            coeff_idx = int(pos_in_band * NODE_SIZE)
            coeff_idx = max(0, min(NODE_SIZE - 1, coeff_idx))

            # Also spread to adjacent coefficients for smoother synthesis
            node_start = leaf_idx * NODE_SIZE

            # Primary coefficient gets full amplitude
            dense[node_start + coeff_idx] = sat16(
                dense[node_start + coeff_idx] + amp_k
            )

            # Adjacent coefficients get partial amplitude (spread for richness)
            spread_amp = amp_k // 4
            if coeff_idx > 0:
                dense[node_start + coeff_idx - 1] = sat16(
                    dense[node_start + coeff_idx - 1] + spread_amp
                )
            if coeff_idx < NODE_SIZE - 1:
                dense[node_start + coeff_idx + 1] = sat16(
                    dense[node_start + coeff_idx + 1] + spread_amp
                )

        return dense


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_harmonic_coeffs(frequency: float = 220.0,
                           amplitude: int = 0x5000,
                           wave_type: WaveType = WaveType.SAW,
                           partials_n: int = 12,
                           brightness_q15: int = Q15_ONE,
                           spread_to_neighbors: bool = False) -> List[int]:
    """
    Create harmonic coefficient pattern for a given frequency and wave type.

    This is a simplified standalone function for quick coefficient generation.
    For per-voice stateful injection, use HarmonicTargetGenerator.

    Args:
        frequency: Fundamental frequency in Hz
        amplitude: Base amplitude (Q15)
        wave_type: Wave type enum
        partials_n: Number of partials to generate
        brightness_q15: Brightness control (affects high partial rolloff)
        spread_to_neighbors: If True, spread energy to adjacent coefficients
            for richer/fatter sound. If False (default), place energy only at
            the exact coefficient for cleaner frequency targeting.

    Returns:
        256-coefficient vector with harmonic energy placed in appropriate leaves
    """
    coeffs = [0] * COEFF_LEN
    nyquist = SAMPLE_RATE / 2

    for k in range(1, partials_n + 1):
        f_k = frequency * k

        if f_k >= nyquist * 0.95:
            break

        # Get amplitude for this partial
        base_amp = get_partial_amplitude(wave_type, k, brightness_q15)
        amp_k = mul_q15_sat(base_amp, amplitude)

        if amp_k < 10:
            continue

        # Map to leaf
        leaf_idx = freq_to_leaf(f_k)

        # Calculate position within leaf for frequency encoding
        leaf_low = LEAF_EDGES_HZ[leaf_idx]
        leaf_high = LEAF_EDGES_HZ[leaf_idx + 1]
        leaf_span = leaf_high - leaf_low

        if leaf_span > 0:
            pos_in_band = (f_k - leaf_low) / leaf_span
        else:
            pos_in_band = 0.5

        # Map to coefficient index
        coeff_idx = int(pos_in_band * NODE_SIZE)
        coeff_idx = max(0, min(NODE_SIZE - 1, coeff_idx))

        node_start = leaf_idx * NODE_SIZE

        # Place energy at target coefficient
        coeffs[node_start + coeff_idx] = sat16(
            coeffs[node_start + coeff_idx] + amp_k
        )

        # Optionally spread to neighbors for richer sound
        if spread_to_neighbors:
            spread_amp = amp_k // 4
            if coeff_idx > 0:
                coeffs[node_start + coeff_idx - 1] = sat16(
                    coeffs[node_start + coeff_idx - 1] + spread_amp
                )
            if coeff_idx < NODE_SIZE - 1:
                coeffs[node_start + coeff_idx + 1] = sat16(
                    coeffs[node_start + coeff_idx + 1] + spread_amp
                )

    return coeffs


def create_harmonic_coeffs_fixed(
    note: int = 60,
    cents: int = 0,
    amplitude: int = 0x5000,
    wave_type: WaveType = WaveType.SAW,
    partials_n: int = 12,
    brightness_q15: int = Q15_ONE,
    spread_to_neighbors: bool = False
) -> List[int]:
    """
    Create harmonic coefficient pattern using fixed-point note/cents.

    This is the bit-exact version that avoids floating-point noise.
    Uses integer millihertz frequency lookup for deterministic coefficient mapping.

    Args:
        note: MIDI note number (0-127, 60 = middle C)
        cents: Fine tuning in cents (-9999 to +9999)
        amplitude: Base amplitude (Q15)
        wave_type: Wave type enum
        partials_n: Number of partials to generate
        brightness_q15: Brightness control (affects high partial rolloff)
        spread_to_neighbors: If True, spread energy to adjacent coefficients

    Returns:
        256-coefficient vector with harmonic energy placed in appropriate leaves
    """
    from ..fixed import midi_to_freq_mhz

    coeffs = [0] * COEFF_LEN
    nyquist_mhz = SAMPLE_RATE * 500  # Nyquist in millihertz (24000 * 1000)

    # Get fundamental frequency in millihertz (integer)
    f0_mhz = midi_to_freq_mhz(note, cents)

    for k in range(1, partials_n + 1):
        # Partial frequency in millihertz (integer math)
        f_k_mhz = f0_mhz * k

        # Skip if above Nyquist (with 5% margin)
        if f_k_mhz >= (nyquist_mhz * 95) // 100:
            break

        # Get amplitude for this partial
        base_amp = get_partial_amplitude(wave_type, k, brightness_q15)
        amp_k = mul_q15_sat(base_amp, amplitude)

        if amp_k < 10:
            continue

        # Map frequency to leaf (convert mHz to Hz for lookup)
        f_k_hz = f_k_mhz // 1000
        leaf_idx = freq_to_leaf(f_k_hz)

        # Calculate position within leaf using integer math
        # leaf_edges are in Hz, f_k is in mHz for precision
        leaf_low_mhz = int(LEAF_EDGES_HZ[leaf_idx] * 1000)
        leaf_high_mhz = int(LEAF_EDGES_HZ[leaf_idx + 1] * 1000)
        leaf_span_mhz = leaf_high_mhz - leaf_low_mhz

        if leaf_span_mhz > 0:
            # pos_in_band as integer 0-1000 (milliunits)
            pos_milli = ((f_k_mhz - leaf_low_mhz) * 1000) // leaf_span_mhz
            pos_milli = max(0, min(999, pos_milli))
        else:
            pos_milli = 500  # 0.5

        # Map to coefficient index (0 to NODE_SIZE-1)
        coeff_idx = (pos_milli * NODE_SIZE) // 1000
        coeff_idx = max(0, min(NODE_SIZE - 1, coeff_idx))

        node_start = leaf_idx * NODE_SIZE

        # Place energy at target coefficient
        coeffs[node_start + coeff_idx] = sat16(
            coeffs[node_start + coeff_idx] + amp_k
        )

        # Optionally spread to neighbors for richer sound
        if spread_to_neighbors:
            spread_amp = amp_k // 4
            if coeff_idx > 0:
                coeffs[node_start + coeff_idx - 1] = sat16(
                    coeffs[node_start + coeff_idx - 1] + spread_amp
                )
            if coeff_idx < NODE_SIZE - 1:
                coeffs[node_start + coeff_idx + 1] = sat16(
                    coeffs[node_start + coeff_idx + 1] + spread_amp
                )

    return coeffs


def synthesize_harmonic_audio(frequency: float = 220.0,
                              amplitude: int = 0x5000,
                              wave_type: WaveType = WaveType.SAW,
                              partials_n: int = 12,
                              brightness_q15: int = Q15_ONE,
                              block_size: int = 64,
                              sample_rate: int = SAMPLE_RATE,
                              phase_state: Optional[List[float]] = None) -> Tuple[List[int], List[float]]:
    """
    Directly synthesize audio for harmonic content (Baby Fourier additive synthesis).

    This bypasses the wavelet coefficient encoding and directly generates
    audio samples at the exact harmonic frequencies. Use this when you need
    precise pitch rather than wavelet-domain processing.

    Args:
        frequency: Fundamental frequency in Hz
        amplitude: Base amplitude (Q15)
        wave_type: Wave type enum
        partials_n: Number of partials to generate
        brightness_q15: Brightness control
        block_size: Samples per block (default 64)
        sample_rate: Sample rate in Hz (default 48000)
        phase_state: Optional phase accumulator state (one per partial)

    Returns:
        Tuple of (audio samples, updated phase state)
    """
    import numpy as np

    output = [0] * block_size
    nyquist = sample_rate / 2

    # Initialize phase state if not provided
    if phase_state is None:
        phase_state = [0.0] * partials_n
    elif len(phase_state) < partials_n:
        phase_state = list(phase_state) + [0.0] * (partials_n - len(phase_state))

    new_phase_state = list(phase_state)

    for k in range(1, partials_n + 1):
        f_k = frequency * k

        if f_k >= nyquist * 0.95:
            break

        # Get amplitude for this partial
        base_amp = get_partial_amplitude(wave_type, k, brightness_q15)
        amp_k = mul_q15_sat(base_amp, amplitude)

        if amp_k < 10:
            continue

        # Angular frequency (radians per sample)
        omega = 2 * np.pi * f_k / sample_rate

        # Get current phase for this partial
        phase = phase_state[k - 1]

        # Generate samples
        for i in range(block_size):
            sample = int(amp_k * np.sin(omega * i + phase))
            output[i] = sat16(output[i] + sample)

        # Update phase for next block (continuous phase)
        new_phase_state[k - 1] = (phase + omega * block_size) % (2 * np.pi)

    return output, new_phase_state


# Tests moved to pipeline_tests/test_harmonic_injector_conformance.py
