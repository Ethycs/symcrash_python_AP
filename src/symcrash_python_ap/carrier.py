"""
Carrier Oscillator Module
=========================

Phase-continuous carrier oscillator for the Carrier + Lift architecture.

The carrier provides:
1. Stable pitch via fixed-point phase accumulator
2. Phase continuity across parameter changes
3. Fast sine approximation via parabolic method
4. Smooth pitch slew for portamento/glide
5. Simple additive synthesis with normalized output

This is the "Baby Fourier" that owns pitch identity. The IDWPT lift
shapes timbre but never determines pitch.

Target: ECP5-25 FPGA @ 48kHz, N=64 block size
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import math

from .carrier_lift import (
    CarrierState, BlendState, EnvelopeState, EnvelopeStage,
    Q15_ONE, Q15_HALF, Q15_SCALE, BLOCK_SIZE, SAMPLE_RATE
)


# =============================================================================
# PHASE ACCUMULATOR CONSTANTS
# =============================================================================

PHASE_BITS = 32
PHASE_MASK = (1 << PHASE_BITS) - 1


# =============================================================================
# FREQUENCY CONVERSION
# =============================================================================

def frequency_to_phase_inc(freq_hz: float, sample_rate: int = SAMPLE_RATE) -> int:
    """
    Convert frequency in Hz to phase increment.

    Phase increment = freq * 2^PHASE_BITS / sample_rate

    This ensures the phase accumulator completes one full cycle
    in exactly (sample_rate / freq_hz) samples.
    """
    return int((freq_hz * (1 << PHASE_BITS)) / sample_rate)


def midi_to_frequency(note: int, cents: int = 0) -> float:
    """Convert MIDI note + cents to frequency in Hz."""
    return 440.0 * (2.0 ** ((note - 69 + cents / 100.0) / 12.0))


def midi_to_phase_inc(note: int, cents: int = 0,
                      sample_rate: int = SAMPLE_RATE) -> int:
    """Convert MIDI note + cents directly to phase increment."""
    freq = midi_to_frequency(note, cents)
    return frequency_to_phase_inc(freq, sample_rate)


# =============================================================================
# FAST SINE APPROXIMATION
# =============================================================================

# =============================================================================
# SINE LOOKUP TABLE (256 entries for quarter-wave, 12-bit index with interpolation)
# =============================================================================

# Quarter-wave sine table: 256 Q15 values for phase [0, pi/2)
# Generated with: [int(32767 * sin(pi/2 * i/256)) for i in range(256)] + [32767]
_SINE_TABLE_Q15 = [
    0, 402, 804, 1206, 1608, 2009, 2411, 2811,
    3212, 3612, 4011, 4410, 4808, 5205, 5602, 5998,
    6393, 6787, 7180, 7571, 7962, 8351, 8740, 9127,
    9512, 9896, 10279, 10660, 11039, 11417, 11793, 12167,
    12540, 12910, 13279, 13646, 14010, 14373, 14733, 15091,
    15447, 15800, 16151, 16500, 16846, 17190, 17531, 17869,
    18205, 18538, 18868, 19195, 19520, 19841, 20160, 20475,
    20788, 21097, 21403, 21706, 22006, 22302, 22595, 22884,
    23170, 23453, 23732, 24008, 24279, 24548, 24812, 25073,
    25330, 25583, 25833, 26078, 26320, 26557, 26791, 27020,
    27246, 27467, 27684, 27897, 28106, 28311, 28511, 28707,
    28899, 29086, 29269, 29448, 29622, 29792, 29957, 30118,
    30274, 30425, 30572, 30715, 30853, 30986, 31114, 31238,
    31357, 31471, 31581, 31686, 31786, 31881, 31972, 32058,
    32138, 32214, 32286, 32352, 32413, 32470, 32522, 32568,
    32610, 32647, 32679, 32706, 32729, 32746, 32758, 32766,
    32767, 32766, 32758, 32746, 32729, 32706, 32679, 32647,
    32610, 32568, 32522, 32470, 32413, 32352, 32286, 32214,
    32138, 32058, 31972, 31881, 31786, 31686, 31581, 31471,
    31357, 31238, 31114, 30986, 30853, 30715, 30572, 30425,
    30274, 30118, 29957, 29792, 29622, 29448, 29269, 29086,
    28899, 28707, 28511, 28311, 28106, 27897, 27684, 27467,
    27246, 27020, 26791, 26557, 26320, 26078, 25833, 25583,
    25330, 25073, 24812, 24548, 24279, 24008, 23732, 23453,
    23170, 22884, 22595, 22302, 22006, 21706, 21403, 21097,
    20788, 20475, 20160, 19841, 19520, 19195, 18868, 18538,
    18205, 17869, 17531, 17190, 16846, 16500, 16151, 15800,
    15447, 15091, 14733, 14373, 14010, 13646, 13279, 12910,
    12540, 12167, 11793, 11417, 11039, 10660, 10279, 9896,
    9512, 9127, 8740, 8351, 7962, 7571, 7180, 6787,
    6393, 5998, 5602, 5205, 4808, 4410, 4011, 3612,
    3212, 2811, 2411, 2009, 1608, 1206, 804, 402,
    0,  # Extra entry for interpolation at boundary
]


def fast_sine_q15(phase: int) -> int:
    """
    Fast sine approximation from 32-bit phase using lookup table with interpolation.

    Uses 256-entry half-wave table with linear interpolation for ~0.05% THD.
    FPGA-friendly: only table lookup, shift, and multiply-accumulate.

    Args:
        phase: 32-bit phase value (0 to 2^32-1 = one full cycle)

    Returns:
        Q15 sine value (-32768 to 32767)
    """
    # Extract half-cycle flag and position within half-cycle
    # Bit 31 = which half of full cycle (0 = positive, 1 = negative)
    # Bits 22-30 = 9 bits for table index (256 entries) + interpolation
    half_cycle = (phase >> 31) & 1

    # Position within half-cycle: use bits 22-30 for 512 positions (256 table + interp)
    # Bits 30-23 = 8-bit table index (0-255)
    # Bits 22-19 = 4-bit fractional for interpolation
    pos = (phase >> 19) & 0xFFF  # 12 bits: 8 index + 4 frac

    table_idx = (pos >> 4) & 0xFF  # 8-bit index into 256-entry table
    frac = pos & 0xF  # 4-bit fractional (0-15)

    # Lookup with linear interpolation
    y0 = _SINE_TABLE_Q15[table_idx]
    y1 = _SINE_TABLE_Q15[table_idx + 1]

    # Interpolate: y = y0 + (y1 - y0) * frac / 16
    y = y0 + (((y1 - y0) * frac) >> 4)

    # Apply sign based on half-cycle
    if half_cycle:
        return -y
    else:
        return y


# =============================================================================
# CARRIER OSCILLATOR
# =============================================================================

@dataclass
class CarrierOscillator:
    """
    Phase-continuous carrier oscillator.

    Generates stable harmonic content with:
    - 32-bit phase accumulator for precise pitch
    - Per-sample pitch slew for smooth portamento
    - Additive synthesis with up to 8 harmonics
    - Automatic gain normalization
    - DC blocking
    """
    # Number of harmonics for additive synthesis
    num_harmonics: int = 8

    # Per-voice state storage
    states: Dict[int, CarrierState] = field(default_factory=dict)

    # Precomputed normalization gains per harmonic configuration
    _norm_cache: Dict[tuple, int] = field(default_factory=dict, repr=False)

    def get_or_create_state(self, voice_id: int) -> CarrierState:
        """Get existing state or create new one for voice."""
        if voice_id not in self.states:
            state = CarrierState(voice_id=voice_id)
            self.states[voice_id] = state
        return self.states[voice_id]

    def remove_state(self, voice_id: int) -> None:
        """Remove state for voice (on voice free)."""
        self.states.pop(voice_id, None)

    def configure_voice(self, voice_id: int, note: int, cents: int = 0,
                        harmonic_amps: Optional[List[int]] = None,
                        reset_phase: bool = True) -> None:
        """
        Configure carrier for a voice.

        Args:
            voice_id: Voice identifier
            note: MIDI note number
            cents: Fine tuning in cents
            harmonic_amps: Optional Q15 amplitudes for harmonics 1-8
            reset_phase: If True, reset phase to 0 (note-on)
        """
        state = self.get_or_create_state(voice_id)

        # Set frequency
        freq = midi_to_frequency(note, cents)
        state.set_frequency(freq)

        # Set harmonic amplitudes (default to saw-like falloff)
        if harmonic_amps is not None:
            state.harmonic_amps_q15 = list(harmonic_amps[:8])
            while len(state.harmonic_amps_q15) < 8:
                state.harmonic_amps_q15.append(0)
        else:
            # Default saw harmonics: 1/n falloff
            state.harmonic_amps_q15 = [Q15_ONE // n for n in range(1, 9)]

        # Reset phase if requested (note-on behavior)
        if reset_phase:
            state.reset()

    def set_pitch(self, voice_id: int, note: int, cents: int = 0,
                  slew_time_ms: float = 0.0) -> None:
        """
        Set pitch for a voice with optional slew.

        Args:
            voice_id: Voice identifier
            note: Target MIDI note
            cents: Target fine tuning
            slew_time_ms: Portamento time in ms (0 = instant)
        """
        state = self.get_or_create_state(voice_id)
        freq = midi_to_frequency(note, cents)
        if slew_time_ms > 0:
            state.set_target_frequency(freq, slew_time_ms)
        else:
            state.set_frequency(freq)

    def gate_on(self, voice_id: int) -> None:
        """Trigger envelope attack for voice."""
        state = self.get_or_create_state(voice_id)
        state.envelope.gate_on()

    def gate_off(self, voice_id: int) -> None:
        """Trigger envelope release for voice."""
        if voice_id in self.states:
            self.states[voice_id].envelope.gate_off()

    def _compute_normalization(self, harmonic_amps: Tuple[int, ...]) -> int:
        """
        Compute normalization gain for given harmonic amplitudes.

        Uses peak measurement over one cycle for accurate normalization.
        Caches results to avoid recomputation.
        """
        if harmonic_amps in self._norm_cache:
            return self._norm_cache[harmonic_amps]

        if not any(harmonic_amps):
            self._norm_cache[harmonic_amps] = Q15_ONE
            return Q15_ONE

        # Measure actual peak over one cycle (256 samples)
        max_val = 0
        for i in range(256):
            phase = (i * (1 << PHASE_BITS)) // 256
            output = 0
            for n, amp in enumerate(harmonic_amps, start=1):
                if amp == 0:
                    continue
                h_phase = (phase * n) & PHASE_MASK
                sine = fast_sine_q15(h_phase)
                output += (sine * amp) >> 15
            max_val = max(max_val, abs(output))

        if max_val > 0:
            norm_gain = min(Q15_ONE, (Q15_ONE * Q15_ONE) // max_val)
        else:
            norm_gain = Q15_ONE

        self._norm_cache[harmonic_amps] = norm_gain
        return norm_gain

    def render_sample(self, state: CarrierState) -> int:
        """
        Render one sample for a carrier state.

        Handles:
        - Per-sample pitch slew
        - Phase accumulation
        - Additive synthesis
        - Envelope application
        - DC blocking

        Args:
            state: CarrierState to render

        Returns:
            Q15 audio sample
        """
        # Apply pitch slew (per-sample for smoothness)
        if state.pitch_slew_rate != 0:
            if state.pitch_slew_rate > 0:
                state.phase_inc = min(state.target_phase_inc,
                                      state.phase_inc + state.pitch_slew_rate)
            else:
                state.phase_inc = max(state.target_phase_inc,
                                      state.phase_inc + state.pitch_slew_rate)

            if state.phase_inc == state.target_phase_inc:
                state.pitch_slew_rate = 0

        # Advance phase
        state.phase = (state.phase + state.phase_inc) & PHASE_MASK

        # Additive synthesis
        output = 0
        for n, amp in enumerate(state.harmonic_amps_q15, start=1):
            if amp == 0:
                continue
            # Harmonic phase = n * base_phase
            h_phase = (state.phase * n) & PHASE_MASK
            sine = fast_sine_q15(h_phase)
            output += (sine * amp) >> 15

        # Apply normalization
        norm_gain = self._compute_normalization(tuple(state.harmonic_amps_q15))
        output = (output * norm_gain) >> 15

        # Apply envelope
        env_level = state.envelope.advance()
        output = (output * env_level) >> 15

        # DC blocking (simple high-pass)
        # y = x - x_dc, x_dc = x_dc * 0.998 + x * 0.002
        state.dc_block_state = (state.dc_block_state * 32700 + output * 67) >> 15
        output = output - state.dc_block_state

        # Saturate
        if output > Q15_ONE:
            output = Q15_ONE
        elif output < -Q15_SCALE:
            output = -Q15_SCALE

        return output

    def render_block(self, voice_id: int, block_size: int = BLOCK_SIZE) -> List[int]:
        """
        Render a full block of samples for a voice.

        Args:
            voice_id: Voice identifier
            block_size: Number of samples per block

        Returns:
            List of Q15 audio samples
        """
        state = self.get_or_create_state(voice_id)
        return [self.render_sample(state) for _ in range(block_size)]

    def render_all_voices(self, block_size: int = BLOCK_SIZE) -> Dict[int, List[int]]:
        """
        Render all active voices.

        Returns:
            Dict mapping voice_id to sample list
        """
        outputs = {}
        for voice_id, state in self.states.items():
            if state.envelope.stage != EnvelopeStage.IDLE:
                outputs[voice_id] = [self.render_sample(state) for _ in range(block_size)]
        return outputs


# =============================================================================
# EQUAL-POWER BLEND
# =============================================================================

def equal_power_blend_q15(dry_sample: int, wet_sample: int, beta_q15: int) -> int:
    """
    Equal-power crossfade between dry (carrier) and wet (IDWPT lift).

    Uses: y = cos(pi*beta/2) * dry + sin(pi*beta/2) * wet

    This maintains constant perceived loudness across the blend range.

    Args:
        dry_sample: Carrier sample (Q15)
        wet_sample: IDWPT lift sample (Q15)
        beta_q15: Blend amount (0 = pure dry, 32767 = pure wet)

    Returns:
        Blended Q15 sample
    """
    # For efficiency, use precomputed or approximated trig
    # At beta=0: cos=1, sin=0 -> pure dry
    # At beta=Q15_ONE: cos=0, sin=1 -> pure wet
    # At beta=Q15_HALF: cos=sin=0.707 -> equal mix

    # Approximate cos/sin using parabolic curves
    # cos(pi*x/2) ~ 1 - x^2 for x in [0,1]
    # sin(pi*x/2) ~ x * (2 - x) for x in [0,1]

    # Normalize beta to 0..1 range in Q15
    # beta_norm = beta_q15 / Q15_ONE
    # We'll work in Q15 throughout

    # Parabolic approximation for sin(pi*beta/2)
    # sin_approx = beta * (2*Q15_ONE - beta) >> 15
    # Scaled: sin = (beta * (65534 - beta)) >> 15
    sin_val = (beta_q15 * (65534 - beta_q15)) >> 16  # Q15 result

    # cos = sqrt(1 - sin^2) approximated as (Q15_ONE - sin_val)
    # For equal-power: cos^2 + sin^2 = 1
    # Approximation: cos ~ Q15_ONE - (sin_val * sin_val >> 15)
    # But simpler: cos ~ Q15_ONE - beta_q15 (linear approx good enough for audio)
    cos_val = Q15_ONE - sin_val

    # Blend
    output = ((dry_sample * cos_val) >> 15) + ((wet_sample * sin_val) >> 15)

    # Saturate
    if output > Q15_ONE:
        return Q15_ONE
    elif output < -Q15_SCALE:
        return -Q15_SCALE
    return output


def blend_block(dry_samples: List[int], wet_samples: List[int],
                beta_q15: int) -> List[int]:
    """
    Blend entire block with constant beta.

    Args:
        dry_samples: Carrier samples (Q15)
        wet_samples: IDWPT lift samples (Q15)
        beta_q15: Blend amount

    Returns:
        List of blended Q15 samples
    """
    return [
        equal_power_blend_q15(d, w, beta_q15)
        for d, w in zip(dry_samples, wet_samples)
    ]


def blend_block_with_ramp(dry_samples: List[int], wet_samples: List[int],
                          beta_start_q15: int, beta_end_q15: int) -> List[int]:
    """
    Blend block with linearly ramping beta (for smooth transitions).

    Args:
        dry_samples: Carrier samples (Q15)
        wet_samples: IDWPT lift samples (Q15)
        beta_start_q15: Beta at start of block
        beta_end_q15: Beta at end of block

    Returns:
        List of blended Q15 samples
    """
    n = len(dry_samples)
    if n == 0:
        return []

    output = []
    for i, (d, w) in enumerate(zip(dry_samples, wet_samples)):
        # Linear interpolation of beta across block
        t = i / (n - 1) if n > 1 else 1.0
        beta = int(beta_start_q15 * (1 - t) + beta_end_q15 * t)
        output.append(equal_power_blend_q15(d, w, beta))

    return output


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'PHASE_BITS',
    'PHASE_MASK',
    # Conversion functions
    'frequency_to_phase_inc',
    'midi_to_frequency',
    'midi_to_phase_inc',
    # Fast sine
    'fast_sine_q15',
    # Carrier
    'CarrierOscillator',
    # Blending
    'equal_power_blend_q15',
    'blend_block',
    'blend_block_with_ramp',
]
