"""
Envelope Suite
==============

Unified envelope generation and following system for the symcrash synthesis engine.

This module provides a cohesive set of envelope generators and followers:

1. **ADSR** - Classic attack/decay/sustain/release (note-triggered)
2. **EnvelopeFollower** - Tracks input amplitude with attack/release
3. **LFO** - Low-frequency oscillator with multiple waveforms
4. **Slew** - Smoothing filter for control signals
5. **AR** - Simplified attack/release (no sustain)
6. **Decay** - One-shot exponential decay

All generators operate at block rate (750 Hz) and output Q15 values.
All state is integer-only for deterministic fixed-point operation.

Per WSE-STD-ENVELOPE-MATRIX v0, envelope outputs route to targets via
the EnvelopeTarget registry (FM operators, gain, filters, etc.).

Usage:
    from symcrash_python_ap.envelope_suite import (
        ADSR, EnvelopeFollower, LFO, Slew, AR, Decay
    )

    # ADSR for note-triggered amplitude
    amp_env = ADSR(attack_ms=10, decay_ms=100, sustain=0.7, release_ms=200)
    amp_env.trigger()
    for block in range(1000):
        gain = amp_env.tick()
        if block == 500:
            amp_env.release()

    # Envelope follower for sidechain
    follower = EnvelopeFollower(attack_ms=5, release_ms=100)
    for block in range(1000):
        level = follower.tick(input_signal_q15)

    # LFO for vibrato
    lfo = LFO(rate_hz=5.0, shape=LFOShape.SINE)
    for block in range(1000):
        mod = lfo.tick()
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List, Tuple
import math

from .primitives import Q15_ONE, BLOCK_RATE_HZ


# =============================================================================
# CONSTANTS
# =============================================================================

BLOCKS_PER_SECOND = BLOCK_RATE_HZ  # 750 blocks/sec at 48kHz/64


def ms_to_blocks(ms: float) -> int:
    """Convert milliseconds to block count (minimum 1)."""
    blocks = int((ms / 1000.0) * BLOCKS_PER_SECOND)
    return max(1, blocks)


def hz_to_phase_inc(hz: float) -> int:
    """
    Convert Hz to Q16 phase increment per block.

    At 750 blocks/sec, 1 Hz = 65536 / 750 ≈ 87 phase units per block.
    """
    if hz <= 0:
        return 0
    # phase_inc = (hz / BLOCKS_PER_SECOND) * 65536
    return int((hz * 65536) / BLOCKS_PER_SECOND)


def _sat16(x: int) -> int:
    """Saturate to Q15 range."""
    if x > 32767:
        return 32767
    if x < -32768:
        return -32768
    return x


def _rshift_rne(val: int, shift: int) -> int:
    """Right shift with round-to-nearest-even."""
    if shift <= 0:
        return val
    half = 1 << (shift - 1)
    result = (val + half) >> shift
    if (val & ((1 << shift) - 1)) == half:
        result = result & ~1
    return result


# =============================================================================
# ENVELOPE PHASE
# =============================================================================

class EnvelopePhase(IntEnum):
    """Envelope state machine phases."""
    IDLE = 0        # Not active
    ATTACK = 1      # Rising
    DECAY = 2       # Falling to sustain
    SUSTAIN = 3     # Holding
    RELEASE = 4     # Falling to zero


# =============================================================================
# ADSR ENVELOPE
# =============================================================================

@dataclass
class ADSR:
    """
    Classic ADSR envelope generator.

    Operates at block rate (750 Hz). All times in milliseconds.
    Output is Q15 (0 to 32767).

    Attributes:
        attack_ms: Time to rise from 0 to peak (1-10000)
        decay_ms: Time to fall from peak to sustain (1-10000)
        sustain: Sustain level as fraction (0.0 to 1.0)
        release_ms: Time to fall from current to 0 (1-10000)
        curve: Envelope curve shape (0=linear, positive=exp, negative=log)
    """
    attack_ms: float = 10.0
    decay_ms: float = 100.0
    sustain: float = 0.5
    release_ms: float = 100.0
    curve: float = 0.0  # Reserved for future curve shaping

    # Internal state (all integers for determinism)
    _phase: EnvelopePhase = field(default=EnvelopePhase.IDLE, repr=False)
    _phase_block: int = field(default=0, repr=False)
    _current_q15: int = field(default=0, repr=False)
    _release_level_q15: int = field(default=0, repr=False)

    # Cached block counts
    _attack_blocks: int = field(default=0, repr=False)
    _decay_blocks: int = field(default=0, repr=False)
    _release_blocks: int = field(default=0, repr=False)
    _sustain_q15: int = field(default=0, repr=False)

    def __post_init__(self):
        self._update_cache()

    def _update_cache(self):
        """Recompute cached values from parameters."""
        self._attack_blocks = ms_to_blocks(self.attack_ms)
        self._decay_blocks = ms_to_blocks(self.decay_ms)
        self._release_blocks = ms_to_blocks(self.release_ms)
        self._sustain_q15 = int(max(0.0, min(1.0, self.sustain)) * Q15_ONE)

    def trigger(self) -> None:
        """Start envelope (note-on). Allows retriggering from current level."""
        self._phase = EnvelopePhase.ATTACK
        self._phase_block = 0

    def release(self) -> None:
        """Enter release phase (note-off)."""
        if self._phase in (EnvelopePhase.ATTACK, EnvelopePhase.DECAY, EnvelopePhase.SUSTAIN):
            self._release_level_q15 = self._current_q15
            self._phase = EnvelopePhase.RELEASE
            self._phase_block = 0

    def tick(self) -> int:
        """
        Advance envelope by one block.

        Returns:
            Q15 envelope value (0 to 32767)
        """
        if self._phase == EnvelopePhase.IDLE:
            self._current_q15 = 0
            return 0

        elif self._phase == EnvelopePhase.ATTACK:
            # Linear rise from current to Q15_ONE
            if self._attack_blocks <= 1:
                t_q15 = Q15_ONE
            else:
                t_q15 = (self._phase_block * Q15_ONE) // (self._attack_blocks - 1)
            self._current_q15 = min(Q15_ONE, t_q15)

            self._phase_block += 1
            if self._phase_block >= self._attack_blocks:
                self._current_q15 = Q15_ONE
                self._phase = EnvelopePhase.DECAY
                self._phase_block = 0

        elif self._phase == EnvelopePhase.DECAY:
            # Linear fall from Q15_ONE to sustain
            if self._decay_blocks <= 1:
                t_q15 = Q15_ONE
            else:
                t_q15 = (self._phase_block * Q15_ONE) // (self._decay_blocks - 1)
            decay_range = Q15_ONE - self._sustain_q15
            self._current_q15 = Q15_ONE - _rshift_rne(decay_range * t_q15, 15)

            self._phase_block += 1
            if self._phase_block >= self._decay_blocks:
                self._current_q15 = self._sustain_q15
                self._phase = EnvelopePhase.SUSTAIN
                self._phase_block = 0

        elif self._phase == EnvelopePhase.SUSTAIN:
            self._current_q15 = self._sustain_q15
            self._phase_block += 1

        elif self._phase == EnvelopePhase.RELEASE:
            # Linear fall from release_level to 0
            if self._release_blocks <= 1:
                t_q15 = Q15_ONE
            else:
                t_q15 = (self._phase_block * Q15_ONE) // (self._release_blocks - 1)
            self._current_q15 = self._release_level_q15 - _rshift_rne(
                self._release_level_q15 * t_q15, 15
            )

            self._phase_block += 1
            if self._phase_block >= self._release_blocks:
                self._phase = EnvelopePhase.IDLE
                self._current_q15 = 0
                self._phase_block = 0

        return max(0, min(Q15_ONE, self._current_q15))

    @property
    def is_active(self) -> bool:
        """True if envelope is producing non-zero output."""
        return self._phase != EnvelopePhase.IDLE

    @property
    def is_releasing(self) -> bool:
        """True if in release phase."""
        return self._phase == EnvelopePhase.RELEASE

    @property
    def phase(self) -> EnvelopePhase:
        """Current envelope phase."""
        return self._phase

    @property
    def value(self) -> int:
        """Current envelope value (Q15)."""
        return self._current_q15

    def reset(self) -> None:
        """Reset envelope to idle state."""
        self._phase = EnvelopePhase.IDLE
        self._phase_block = 0
        self._current_q15 = 0
        self._release_level_q15 = 0


# =============================================================================
# AR ENVELOPE (Attack-Release, no sustain)
# =============================================================================

@dataclass
class AR:
    """
    Attack-Release envelope (no sustain phase).

    Useful for percussive sounds or one-shot triggers.
    After attack reaches peak, immediately enters release.

    Attributes:
        attack_ms: Time to rise from 0 to peak
        release_ms: Time to fall from peak to 0
    """
    attack_ms: float = 10.0
    release_ms: float = 200.0

    _phase: EnvelopePhase = field(default=EnvelopePhase.IDLE, repr=False)
    _phase_block: int = field(default=0, repr=False)
    _current_q15: int = field(default=0, repr=False)
    _attack_blocks: int = field(default=0, repr=False)
    _release_blocks: int = field(default=0, repr=False)

    def __post_init__(self):
        self._attack_blocks = ms_to_blocks(self.attack_ms)
        self._release_blocks = ms_to_blocks(self.release_ms)

    def trigger(self) -> None:
        """Start envelope."""
        self._phase = EnvelopePhase.ATTACK
        self._phase_block = 0

    def tick(self) -> int:
        """Advance by one block, return Q15 value."""
        if self._phase == EnvelopePhase.IDLE:
            return 0

        elif self._phase == EnvelopePhase.ATTACK:
            if self._attack_blocks <= 1:
                t_q15 = Q15_ONE
            else:
                t_q15 = (self._phase_block * Q15_ONE) // (self._attack_blocks - 1)
            self._current_q15 = min(Q15_ONE, t_q15)

            self._phase_block += 1
            if self._phase_block >= self._attack_blocks:
                self._current_q15 = Q15_ONE
                self._phase = EnvelopePhase.RELEASE
                self._phase_block = 0

        elif self._phase == EnvelopePhase.RELEASE:
            if self._release_blocks <= 1:
                t_q15 = Q15_ONE
            else:
                t_q15 = (self._phase_block * Q15_ONE) // (self._release_blocks - 1)
            self._current_q15 = Q15_ONE - _rshift_rne(Q15_ONE * t_q15, 15)

            self._phase_block += 1
            if self._phase_block >= self._release_blocks:
                self._phase = EnvelopePhase.IDLE
                self._current_q15 = 0

        return max(0, min(Q15_ONE, self._current_q15))

    @property
    def is_active(self) -> bool:
        return self._phase != EnvelopePhase.IDLE

    def reset(self) -> None:
        self._phase = EnvelopePhase.IDLE
        self._phase_block = 0
        self._current_q15 = 0


# =============================================================================
# DECAY ENVELOPE (One-shot exponential decay)
# =============================================================================

@dataclass
class Decay:
    """
    One-shot exponential decay envelope.

    Starts at peak and decays toward zero with exponential curve.
    Useful for hi-hats, percussion, FM modulator envelopes.

    Attributes:
        decay_ms: Time constant (time to reach ~37% of initial)
        floor: Minimum output level (0.0 to 1.0)
    """
    decay_ms: float = 100.0
    floor: float = 0.0

    _level_q15: int = field(default=0, repr=False)
    _decay_coeff: int = field(default=0, repr=False)
    _floor_q15: int = field(default=0, repr=False)
    _active: bool = field(default=False, repr=False)

    def __post_init__(self):
        self._update_cache()

    def _update_cache(self):
        """Compute decay coefficient from time constant."""
        # Exponential decay: y[n] = y[n-1] * coeff
        # After decay_ms, level should be ~37% (1/e)
        # coeff = exp(-1 / (decay_ms * blocks_per_ms))
        blocks = ms_to_blocks(self.decay_ms)
        if blocks <= 1:
            self._decay_coeff = 0
        else:
            # coeff = exp(-1/blocks) ≈ 1 - 1/blocks for large blocks
            # In Q15: coeff_q15 = 32767 * (1 - 1/blocks)
            self._decay_coeff = Q15_ONE - (Q15_ONE // blocks)
        self._floor_q15 = int(max(0.0, min(1.0, self.floor)) * Q15_ONE)

    def trigger(self) -> None:
        """Start decay from peak."""
        self._level_q15 = Q15_ONE
        self._active = True

    def tick(self) -> int:
        """Advance by one block, return Q15 value."""
        if not self._active:
            return self._floor_q15

        # Exponential decay
        self._level_q15 = _rshift_rne(self._level_q15 * self._decay_coeff, 15)

        # Check if decayed below floor
        if self._level_q15 <= self._floor_q15:
            self._level_q15 = self._floor_q15
            self._active = False

        return self._level_q15

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def value(self) -> int:
        return self._level_q15

    def reset(self) -> None:
        self._level_q15 = 0
        self._active = False


# =============================================================================
# ENVELOPE FOLLOWER
# =============================================================================

@dataclass
class EnvelopeFollower:
    """
    Amplitude envelope follower with asymmetric attack/release.

    Tracks the amplitude of an input signal, producing a smooth
    control signal suitable for:
    - Sidechain compression/ducking
    - Dynamic parameter modulation
    - Auto-wah / envelope filter
    - Level-dependent effects

    Operates at block rate. Input and output are Q15.

    Attributes:
        attack_ms: Rise time when input increases (1-1000)
        release_ms: Fall time when input decreases (10-5000)
        mode: Detection mode (PEAK or RMS)
    """
    attack_ms: float = 5.0
    release_ms: float = 100.0
    mode: str = "PEAK"  # "PEAK" or "RMS"

    _level_q15: int = field(default=0, repr=False)
    _attack_coeff: int = field(default=0, repr=False)
    _release_coeff: int = field(default=0, repr=False)
    _rms_acc: int = field(default=0, repr=False)  # For RMS mode

    def __post_init__(self):
        self._update_cache()

    def _update_cache(self):
        """Compute filter coefficients from time constants."""
        # One-pole filter coefficients
        # Time constant in blocks: tc = ms * blocks_per_ms
        # Coefficient: alpha = 1 - exp(-1/tc) ≈ 1/tc for large tc
        attack_blocks = max(1, ms_to_blocks(self.attack_ms))
        release_blocks = max(1, ms_to_blocks(self.release_ms))

        # Attack: higher coefficient = faster response
        # coeff = Q15_ONE / blocks (but at least 1)
        self._attack_coeff = max(1, Q15_ONE // attack_blocks)
        self._release_coeff = max(1, Q15_ONE // release_blocks)

    def tick(self, input_q15: int) -> int:
        """
        Process one block of input, return envelope level.

        Args:
            input_q15: Input signal level (Q15, typically abs value)

        Returns:
            Envelope level (Q15, 0 to 32767)
        """
        # Rectify input (take absolute value)
        mag = abs(input_q15)
        if mag > 32767:
            mag = 32767

        if self.mode == "RMS":
            # RMS: accumulate squared values with decay
            mag_sq = _rshift_rne(mag * mag, 15)
            # Leaky integrator for RMS
            coeff = self._release_coeff
            self._rms_acc = self._rms_acc + _rshift_rne((mag_sq - self._rms_acc) * coeff, 15)
            # Approximate sqrt via linear interpolation
            mag = _approx_sqrt_q15(self._rms_acc)

        # Asymmetric follower
        if mag > self._level_q15:
            # Attack: fast rise
            delta = mag - self._level_q15
            step = _rshift_rne(delta * self._attack_coeff, 15)
            self._level_q15 += max(1, step)  # At least 1 to ensure progress
            if self._level_q15 > mag:
                self._level_q15 = mag
        else:
            # Release: slow decay
            delta = self._level_q15 - mag
            step = _rshift_rne(delta * self._release_coeff, 15)
            self._level_q15 -= max(1, step) if delta > 0 else 0
            if self._level_q15 < mag:
                self._level_q15 = mag

        return max(0, min(Q15_ONE, self._level_q15))

    def tick_block(self, samples: List[int]) -> int:
        """
        Process a block of samples, return envelope level.

        Takes the peak or RMS of the block, then applies follower.

        Args:
            samples: List of Q15 samples

        Returns:
            Envelope level (Q15)
        """
        if not samples:
            return self.tick(0)

        if self.mode == "RMS":
            # RMS of block
            sum_sq = sum(s * s for s in samples)
            mean_sq = sum_sq // len(samples)
            mag = _approx_sqrt_q15(min(Q15_ONE * Q15_ONE, mean_sq))
        else:
            # Peak of block
            mag = max(abs(s) for s in samples)

        return self.tick(mag)

    @property
    def value(self) -> int:
        """Current envelope level (Q15)."""
        return self._level_q15

    def reset(self) -> None:
        """Reset follower to zero."""
        self._level_q15 = 0
        self._rms_acc = 0


def _approx_sqrt_q15(x: int) -> int:
    """
    Approximate integer square root for Q15 values.

    Uses Newton-Raphson iteration.
    """
    if x <= 0:
        return 0
    if x >= Q15_ONE * Q15_ONE:
        return Q15_ONE

    # Initial guess: x >> 8 (good for typical audio levels)
    guess = max(1, x >> 8)

    # 3 iterations of Newton-Raphson: guess = (guess + x/guess) / 2
    for _ in range(3):
        if guess == 0:
            guess = 1
        guess = (guess + x // guess) >> 1

    return min(Q15_ONE, guess)


# =============================================================================
# SLEW LIMITER / SMOOTHER
# =============================================================================

@dataclass
class Slew:
    """
    Slew rate limiter / one-pole smoother for control signals.

    Provides smooth transitions between values, preventing clicks
    and zipper noise. Can use linear slew or exponential smoothing.

    Attributes:
        rise_ms: Time to rise from 0 to full scale
        fall_ms: Time to fall from full scale to 0
        mode: "LINEAR" for constant slew rate, "EXP" for one-pole filter
    """
    rise_ms: float = 10.0
    fall_ms: float = 10.0
    mode: str = "LINEAR"  # "LINEAR" or "EXP"

    _current_q15: int = field(default=0, repr=False)
    _rise_rate: int = field(default=0, repr=False)
    _fall_rate: int = field(default=0, repr=False)
    _rise_coeff: int = field(default=0, repr=False)
    _fall_coeff: int = field(default=0, repr=False)

    def __post_init__(self):
        self._update_cache()

    def _update_cache(self):
        """Compute slew rates from time constants."""
        rise_blocks = max(1, ms_to_blocks(self.rise_ms))
        fall_blocks = max(1, ms_to_blocks(self.fall_ms))

        # Linear mode: rate = full_scale / blocks
        self._rise_rate = max(1, Q15_ONE // rise_blocks)
        self._fall_rate = max(1, Q15_ONE // fall_blocks)

        # Exponential mode: coefficient = 1/blocks
        self._rise_coeff = max(1, Q15_ONE // rise_blocks)
        self._fall_coeff = max(1, Q15_ONE // fall_blocks)

    def tick(self, target_q15: int) -> int:
        """
        Slew toward target value.

        Args:
            target_q15: Target value (Q15)

        Returns:
            Slewed value (Q15)
        """
        target_q15 = max(-32768, min(32767, target_q15))
        delta = target_q15 - self._current_q15

        if self.mode == "EXP":
            # Exponential smoothing
            if delta > 0:
                step = _rshift_rne(delta * self._rise_coeff, 15)
            else:
                step = _rshift_rne(delta * self._fall_coeff, 15)
            # Ensure at least 1 step toward target
            if step == 0 and delta != 0:
                step = 1 if delta > 0 else -1
            self._current_q15 += step
        else:
            # Linear slew
            if delta > 0:
                self._current_q15 += min(delta, self._rise_rate)
            else:
                self._current_q15 += max(delta, -self._fall_rate)

        return self._current_q15

    def set_immediate(self, value_q15: int) -> None:
        """Set value immediately (no slew)."""
        self._current_q15 = max(-32768, min(32767, value_q15))

    @property
    def value(self) -> int:
        """Current slewed value (Q15)."""
        return self._current_q15

    def reset(self) -> None:
        """Reset to zero."""
        self._current_q15 = 0


# =============================================================================
# LFO (Low Frequency Oscillator)
# =============================================================================

class LFOShape(IntEnum):
    """LFO waveform shapes."""
    SINE = 0
    TRIANGLE = 1
    SAW_UP = 2
    SAW_DOWN = 3
    SQUARE = 4
    SAMPLE_HOLD = 5


# Precomputed sine table (256 entries, Q15)
_SINE_TABLE_SIZE = 256
_SINE_TABLE: List[int] = []

def _build_sine_table():
    global _SINE_TABLE
    if not _SINE_TABLE:
        _SINE_TABLE = [
            int(math.sin(2 * math.pi * i / _SINE_TABLE_SIZE) * Q15_ONE)
            for i in range(_SINE_TABLE_SIZE)
        ]

_build_sine_table()


@dataclass
class LFO:
    """
    Low-frequency oscillator for modulation.

    Generates periodic control signals at sub-audio rates.
    Output is bipolar Q15 (-32768 to +32767) or unipolar (0 to 32767).

    Attributes:
        rate_hz: Oscillation frequency (0.01 to 50 Hz)
        shape: Waveform shape (SINE, TRIANGLE, SAW_UP, SAW_DOWN, SQUARE, SAMPLE_HOLD)
        unipolar: If True, output is 0 to 32767; if False, -32768 to +32767
        phase_offset: Initial phase offset (0.0 to 1.0)
    """
    rate_hz: float = 1.0
    shape: LFOShape = LFOShape.SINE
    unipolar: bool = False
    phase_offset: float = 0.0

    _phase_q16: int = field(default=0, repr=False)
    _phase_inc: int = field(default=0, repr=False)
    _sh_value: int = field(default=0, repr=False)  # Sample-hold current value
    _sh_seed: int = field(default=12345, repr=False)  # PRNG state for S&H

    def __post_init__(self):
        self._update_cache()
        # Apply phase offset
        self._phase_q16 = int(self.phase_offset * 65536) & 0xFFFF

    def _update_cache(self):
        """Compute phase increment from rate."""
        self._phase_inc = hz_to_phase_inc(self.rate_hz)

    def set_rate(self, hz: float) -> None:
        """Update LFO rate."""
        self.rate_hz = hz
        self._phase_inc = hz_to_phase_inc(hz)

    def tick(self) -> int:
        """
        Advance LFO by one block, return output value.

        Returns:
            Q15 value (bipolar or unipolar based on settings)
        """
        # Generate waveform from phase
        phase = self._phase_q16

        if self.shape == LFOShape.SINE:
            # Table lookup with linear interpolation
            idx = (phase >> 8) & 0xFF
            frac = phase & 0xFF
            y0 = _SINE_TABLE[idx]
            y1 = _SINE_TABLE[(idx + 1) & 0xFF]
            output = y0 + ((y1 - y0) * frac >> 8)

        elif self.shape == LFOShape.TRIANGLE:
            # Triangle: -1 at 0, +1 at 0.5, -1 at 1.0
            if phase < 32768:
                output = (phase * 2) - Q15_ONE
            else:
                output = Q15_ONE - ((phase - 32768) * 2)
            output = _sat16(output)

        elif self.shape == LFOShape.SAW_UP:
            # Sawtooth up: -1 to +1 over full cycle
            output = phase - 32768

        elif self.shape == LFOShape.SAW_DOWN:
            # Sawtooth down: +1 to -1 over full cycle
            output = 32767 - phase

        elif self.shape == LFOShape.SQUARE:
            # Square wave
            output = Q15_ONE if phase < 32768 else -Q15_ONE

        elif self.shape == LFOShape.SAMPLE_HOLD:
            # Sample & hold: new random value at each cycle start
            old_phase = self._phase_q16
            new_phase = (old_phase + self._phase_inc) & 0xFFFF
            # Check for wrap (cycle completion) or first time (sh_value is 0 and we haven't started)
            if new_phase < old_phase or (self._sh_value == 0 and old_phase == int(self.phase_offset * 65536) & 0xFFFF):
                # Crossed zero or first tick, generate new value
                self._sh_seed = (self._sh_seed * 1103515245 + 12345) & 0x7FFFFFFF
                self._sh_value = ((self._sh_seed >> 15) & 0xFFFF) - 32768
                self._sh_value = max(-Q15_ONE, min(Q15_ONE, self._sh_value))
            output = self._sh_value

        else:
            output = 0

        # Advance phase
        self._phase_q16 = (self._phase_q16 + self._phase_inc) & 0xFFFF

        # Convert to unipolar if requested
        if self.unipolar:
            output = (output + Q15_ONE) >> 1

        return _sat16(output)

    def sync(self, phase: float = 0.0) -> None:
        """Reset phase to specified position (0.0 to 1.0)."""
        self._phase_q16 = int(phase * 65536) & 0xFFFF

    @property
    def phase(self) -> float:
        """Current phase as fraction (0.0 to 1.0)."""
        return self._phase_q16 / 65536.0

    def reset(self) -> None:
        """Reset LFO to initial state."""
        self._phase_q16 = int(self.phase_offset * 65536) & 0xFFFF
        self._sh_value = 0


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'BLOCKS_PER_SECOND',
    'ms_to_blocks',
    'hz_to_phase_inc',

    # Enums
    'EnvelopePhase',
    'LFOShape',

    # Envelope generators
    'ADSR',
    'AR',
    'Decay',

    # Followers and smoothers
    'EnvelopeFollower',
    'Slew',

    # Modulators
    'LFO',
]
