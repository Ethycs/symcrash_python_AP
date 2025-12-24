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

from fixed import sat16, mul_q15_sat, Q15_ONE, Q15_SHIFT

# Constants
LEAF_COUNT = 64           # D=6 master tree: 64 leaves
NODE_SIZE = 4             # 256 / 64 = 4 coefficients per leaf
COEFF_LEN = 256
SAMPLE_RATE = 48000
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


# =============================================================================
# TESTS
# =============================================================================

def test_harmonic_injector():
    """Test the harmonic target generator."""
    print("=" * 60)
    print("HARMONIC TARGET INJECTOR TESTS")
    print("=" * 60)

    generator = HarmonicTargetGenerator()

    # Test 1: Note to frequency
    print("\nTest 1: Note to frequency conversion")
    test_notes = [(69, 0, 440.0), (60, 0, 261.6), (48, 0, 130.8), (81, 0, 880.0)]
    for note, cents, expected in test_notes:
        freq = note_to_freq(note, cents)
        print(f"  Note {note} + {cents} cents -> {freq:.1f} Hz (expected ~{expected})")

    # Test 2: Frequency to leaf mapping
    print("\nTest 2: Frequency to leaf mapping")
    test_freqs = [100, 220, 440, 880, 1760, 5000, 10000]
    for freq in test_freqs:
        leaf = freq_to_leaf(freq)
        low = LEAF_EDGES_HZ[leaf]
        high = LEAF_EDGES_HZ[leaf + 1]
        print(f"  {freq} Hz -> leaf {leaf} ({low:.0f}-{high:.0f} Hz)")

    # Test 3: Wave type amplitude laws
    print("\nTest 3: Amplitude laws (first 8 partials)")
    for wt in [WaveType.SINE, WaveType.SAW, WaveType.PULSE, WaveType.TRIANGLE]:
        amps = [get_partial_amplitude(wt, k) for k in range(1, 9)]
        amps_norm = [a / Q15_ONE for a in amps]
        print(f"  {wt.name}: {[f'{a:.3f}' for a in amps_norm]}")

    # Test 4: Generate sparse target
    print("\nTest 4: Sparse target for A3 (220 Hz) saw wave")
    generator.set_note(0, 57)  # A3 = MIDI 57
    generator.set_wave_params(0, WaveType.SAW, partials_n=8)
    target = generator.generate_target(0, include_dense=True)

    print(f"  Active leaves: {target.leaf_indices}")
    print(f"  Amplitudes: {target.leaf_amplitudes}")

    # Show leaf frequencies
    for leaf, amp in zip(target.leaf_indices, target.leaf_amplitudes):
        center = LEAF_CENTER_HZ[leaf]
        print(f"    Leaf {leaf}: {center:.0f} Hz center, amp={amp}")

    # Test 5: Dense target
    print("\nTest 5: Dense target analysis")
    dense = target.dense_target
    non_zero = sum(1 for c in dense if c != 0)
    max_amp = max(dense)
    print(f"  Non-zero coefficients: {non_zero}")
    print(f"  Max amplitude: {max_amp}")

    # Test 6: Injection
    print("\nTest 6: Injection application")
    current = [0] * COEFF_LEN
    generator.set_injection_params(0, harmonic_dial_q15=Q15_ONE, pull_strength_q15=Q15_ONE)
    result = generator.apply_injection(0, current, target)
    non_zero_result = sum(1 for c in result if c != 0)
    print(f"  Applied injection: {non_zero_result} non-zero coefficients")

    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_harmonic_injector()
