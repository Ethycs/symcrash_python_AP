"""
Carrier + IDWPT Lift Architecture - Core Data Structures
=========================================================

Implements the separation of concerns:
- Carrier (Baby Fourier): Owns pitch + phase continuity
- IDWPT Lift: Shapes timbre, acts as an effect/body
- Beta blend: Audio-domain dry/wet with latency handling

Key invariants:
1. Pitch identity lives in carrier - wavelet never determines pitch
2. Master leaf space fixed - D=6, 64 leaves, forever
3. Trees = slow global tunings - 1-5s morph, not per-voice swaps
4. All discrete changes -> continuous morph - slew/crossfade everything
5. Per-voice state persists - carrier phase, blend state, lift state

Target: ECP5-25 FPGA @ 48kHz, N=64 block size
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import IntEnum


# =============================================================================
# CONSTANTS
# =============================================================================

# Q15 fixed-point constants
Q15_ONE = 32767
Q15_HALF = 16384
Q15_SCALE = 32768

# Block/sample rate constants
BLOCK_SIZE = 64
SAMPLE_RATE = 48000
BLOCKS_PER_SECOND = SAMPLE_RATE // BLOCK_SIZE  # 750 Hz

# D=6 master tree (64 leaves, fixed forever)
MASTER_DEPTH = 6
NUM_LEAVES = 1 << MASTER_DEPTH  # 64 leaves

# Latency constants
IDWPT_LATENCY_BLOCKS = 5  # 5-block wavelet latency (~6.7ms @ 48k/64)

# Bloom timing (wet fade-in after latency)
DEFAULT_BLOOM_BLOCKS = 40  # ~53ms bloom time after wet ready

# Default blend target (~75% wet)
DEFAULT_BETA_TARGET_Q15 = 24576  # Q15 0.75


# =============================================================================
# ENVELOPE STATE
# =============================================================================

class EnvelopeStage(IntEnum):
    """ADSR envelope stages."""
    IDLE = 0
    ATTACK = 1
    DECAY = 2
    SUSTAIN = 3
    RELEASE = 4


@dataclass
class EnvelopeState:
    """
    Simple ADSR envelope state.

    Uses sample-rate timing for smooth transitions.
    """
    stage: EnvelopeStage = EnvelopeStage.IDLE
    level_q15: int = 0              # Current envelope level (0..32767)

    # ADSR times in samples
    attack_samples: int = 480       # 10ms @ 48kHz
    decay_samples: int = 2400       # 50ms @ 48kHz
    sustain_q15: int = 24576        # 0.75 sustain level
    release_samples: int = 4800     # 100ms @ 48kHz

    # Internal counters
    stage_sample: int = 0           # Current sample within stage
    gate: bool = False              # Gate state

    def gate_on(self) -> None:
        """Trigger note on."""
        self.gate = True
        self.stage = EnvelopeStage.ATTACK
        self.stage_sample = 0

    def gate_off(self) -> None:
        """Trigger note off."""
        self.gate = False
        if self.stage != EnvelopeStage.IDLE:
            self.stage = EnvelopeStage.RELEASE
            self.stage_sample = 0

    def advance(self) -> int:
        """
        Advance envelope by one sample, return current level.

        Returns:
            Current envelope level in Q15 (0..32767)
        """
        if self.stage == EnvelopeStage.IDLE:
            return 0

        elif self.stage == EnvelopeStage.ATTACK:
            if self.attack_samples > 0:
                self.level_q15 = (self.stage_sample * Q15_ONE) // self.attack_samples
            else:
                self.level_q15 = Q15_ONE
            self.stage_sample += 1
            if self.stage_sample >= self.attack_samples:
                self.stage = EnvelopeStage.DECAY
                self.stage_sample = 0
                self.level_q15 = Q15_ONE

        elif self.stage == EnvelopeStage.DECAY:
            if self.decay_samples > 0:
                # Linear decay from 1.0 to sustain
                decay_range = Q15_ONE - self.sustain_q15
                progress = self.stage_sample / self.decay_samples
                self.level_q15 = Q15_ONE - int(decay_range * progress)
            else:
                self.level_q15 = self.sustain_q15
            self.stage_sample += 1
            if self.stage_sample >= self.decay_samples:
                self.stage = EnvelopeStage.SUSTAIN
                self.level_q15 = self.sustain_q15

        elif self.stage == EnvelopeStage.SUSTAIN:
            self.level_q15 = self.sustain_q15
            # Stay in sustain until gate off

        elif self.stage == EnvelopeStage.RELEASE:
            if self.release_samples > 0:
                start_level = self.sustain_q15 if self.stage_sample == 0 else self.level_q15
                if self.stage_sample == 0:
                    self._release_start_level = self.level_q15
                progress = self.stage_sample / self.release_samples
                self.level_q15 = max(0, self._release_start_level - int(self._release_start_level * progress))
            else:
                self.level_q15 = 0
            self.stage_sample += 1
            if self.stage_sample >= self.release_samples:
                self.stage = EnvelopeStage.IDLE
                self.level_q15 = 0

        return self.level_q15

    # Private attribute for release tracking
    _release_start_level: int = field(default=0, repr=False)


# =============================================================================
# CARRIER STATE (Per-Voice)
# =============================================================================

@dataclass
class CarrierState:
    """
    Per-voice carrier oscillator state.

    The carrier provides phase-continuous pitch and stable fundamental.
    It's the "Baby Fourier" / additive component that guarantees pitch integrity.
    """
    voice_id: int = 0

    # Phase accumulator (32-bit fixed-point, 16.16 format)
    phase: int = 0                  # Current phase (0..2^32-1)
    phase_inc: int = 0              # Phase increment per sample (frequency)
    target_phase_inc: int = 0       # Target for pitch slew

    # Pitch slew rate (per-sample delta for smooth portamento)
    pitch_slew_rate: int = 0        # Phase increment delta per sample

    # Envelope
    envelope: EnvelopeState = field(default_factory=EnvelopeState)

    # Harmonic content (8 harmonics for additive synthesis)
    # These define the carrier's timbral character
    harmonic_amps_q15: List[int] = field(
        default_factory=lambda: [Q15_ONE] + [0] * 7
    )

    # DC blocking state
    dc_block_state: int = 0

    def set_frequency(self, freq_hz: float, sample_rate: int = SAMPLE_RATE) -> None:
        """Set carrier frequency immediately."""
        self.phase_inc = int((freq_hz * (1 << 32)) / sample_rate)
        self.target_phase_inc = self.phase_inc

    def set_target_frequency(self, freq_hz: float, slew_time_ms: float = 10.0,
                             sample_rate: int = SAMPLE_RATE) -> None:
        """Set target frequency for smooth pitch slew."""
        self.target_phase_inc = int((freq_hz * (1 << 32)) / sample_rate)
        if slew_time_ms > 0:
            slew_samples = int(slew_time_ms * sample_rate / 1000)
            delta = self.target_phase_inc - self.phase_inc
            self.pitch_slew_rate = delta // max(1, slew_samples)
        else:
            self.phase_inc = self.target_phase_inc
            self.pitch_slew_rate = 0

    def midi_to_phase_inc(self, note: int, cents: int = 0,
                          sample_rate: int = SAMPLE_RATE) -> int:
        """Convert MIDI note to phase increment."""
        freq = 440.0 * (2.0 ** ((note - 69 + cents / 100.0) / 12.0))
        return int((freq * (1 << 32)) / sample_rate)

    def reset(self) -> None:
        """Reset carrier state for new note."""
        self.phase = 0
        self.dc_block_state = 0


# =============================================================================
# BLEND STATE (Per-Voice)
# =============================================================================

@dataclass
class BlendState:
    """
    Per-voice carrier/lift blend state.

    Implements the audio-domain dry/wet blend (beta) with latency handling.
    Beta ramps only after wavelet wet signal is ready (after IDWPT latency).
    """
    voice_id: int = 0

    # Current blend (0 = pure carrier/dry, 32767 = pure wavelet/wet)
    beta_q15: int = 0

    # Target blend
    beta_target_q15: int = DEFAULT_BETA_TARGET_Q15

    # Latency tracking
    wet_ready_counter: int = IDWPT_LATENCY_BLOCKS  # Blocks until wet available

    # Bloom timing (how fast beta ramps after wet ready)
    bloom_blocks: int = DEFAULT_BLOOM_BLOCKS

    # Per-block beta increment (computed from bloom_blocks)
    # Computed as: Q15_ONE / bloom_blocks per block
    _beta_inc_per_block: int = field(default=0, repr=False)

    def reset_for_note_on(self) -> None:
        """Reset blend state for new note."""
        self.beta_q15 = 0
        self.wet_ready_counter = IDWPT_LATENCY_BLOCKS
        self._compute_beta_inc()

    def _compute_beta_inc(self) -> None:
        """Compute per-block beta increment."""
        if self.bloom_blocks > 0:
            self._beta_inc_per_block = self.beta_target_q15 // self.bloom_blocks
        else:
            self._beta_inc_per_block = self.beta_target_q15

    def set_beta_target(self, target_q15: int, bloom_time_ms: float = 53.0) -> None:
        """Set new beta target with bloom time."""
        self.beta_target_q15 = max(0, min(Q15_ONE, target_q15))
        self.bloom_blocks = int(bloom_time_ms * BLOCKS_PER_SECOND / 1000)
        self._compute_beta_inc()

    def advance_block(self) -> bool:
        """
        Advance blend state by one block.

        Returns:
            True if wet signal is ready (can use beta > 0)
        """
        if self.wet_ready_counter > 0:
            self.wet_ready_counter -= 1
            # Return True when counter hits 0 (wet just became ready)
            if self.wet_ready_counter == 0:
                self._compute_beta_inc()  # Recompute in case target changed
                return True
            return False

        # Wet is ready, ramp beta toward target
        if self.beta_q15 < self.beta_target_q15:
            self.beta_q15 = min(
                self.beta_target_q15,
                self.beta_q15 + self._beta_inc_per_block
            )
        elif self.beta_q15 > self.beta_target_q15:
            self.beta_q15 = max(
                self.beta_target_q15,
                self.beta_q15 - self._beta_inc_per_block
            )

        return True


# =============================================================================
# LEAF CONTENT (Per-Voice, Per-Block)
# =============================================================================

@dataclass
class LeafContent:
    """
    Per-voice 64-leaf coefficient block.

    Contains the amplitude for each of the 64 master leaves.
    Generated by the LeafGenerator from voice partials.
    """
    # 64 Q15 leaf amplitudes (one per master leaf)
    coeffs_q15: List[int] = field(default_factory=lambda: [0] * NUM_LEAVES)

    # Optional phase information per leaf (for phase-coherent synthesis)
    phases_q15: Optional[List[int]] = None

    def clear(self) -> None:
        """Zero all leaf coefficients."""
        self.coeffs_q15 = [0] * NUM_LEAVES
        if self.phases_q15 is not None:
            self.phases_q15 = [0] * NUM_LEAVES

    def apply_gain(self, gain_q15: int) -> None:
        """Apply overall gain to all leaves."""
        for i in range(NUM_LEAVES):
            self.coeffs_q15[i] = (self.coeffs_q15[i] * gain_q15) >> 15

    def apply_per_leaf_gains(self, gains_q15: List[int]) -> None:
        """Apply per-leaf gain (e.g., from instrument tuning)."""
        for i in range(min(NUM_LEAVES, len(gains_q15))):
            self.coeffs_q15[i] = (self.coeffs_q15[i] * gains_q15[i]) >> 15


# =============================================================================
# INSTRUMENT TUNING (Global)
# =============================================================================

@dataclass
class InstrumentTuning:
    """
    Global instrument tuning - 64 leaf gains.

    Tunings are slow-morphing global settings (1-5s transitions).
    They represent instrument character, NOT per-voice topology.

    Think of tunings like guitar pickup positions or filter settings
    that apply to all voices simultaneously.
    """
    name: str = "default"

    # 64 Q15 gains per leaf (defines spectral shape)
    gains_q15: List[int] = field(default_factory=lambda: [Q15_ONE] * NUM_LEAVES)

    # Optional metadata
    description: str = ""

    @classmethod
    def flat(cls, name: str = "flat") -> 'InstrumentTuning':
        """Create flat tuning (all gains = 1.0)."""
        return cls(name=name, gains_q15=[Q15_ONE] * NUM_LEAVES)

    @classmethod
    def harmonic(cls) -> 'InstrumentTuning':
        """Create harmonic-emphasizing tuning."""
        # Emphasize leaves that correspond to harmonic frequencies
        gains = []
        for i in range(NUM_LEAVES):
            # Lower leaves (fundamental region) get more gain
            # Exponential falloff toward high leaves
            gain = int(Q15_ONE * (0.9 ** (i / 8)))
            gains.append(max(1000, gain))  # Minimum floor
        return cls(name="harmonic", gains_q15=gains,
                   description="Emphasizes fundamental and low harmonics")

    @classmethod
    def bright(cls) -> 'InstrumentTuning':
        """Create bright tuning (tilt toward high frequencies)."""
        gains = []
        for i in range(NUM_LEAVES):
            # Tilt toward higher leaves
            tilt = 0.5 + 0.5 * (i / NUM_LEAVES)
            gains.append(int(Q15_ONE * tilt))
        return cls(name="bright", gains_q15=gains,
                   description="Tilted toward high frequencies")

    @classmethod
    def hollow(cls) -> 'InstrumentTuning':
        """Create hollow tuning (odd harmonics emphasized)."""
        gains = []
        for i in range(NUM_LEAVES):
            # Boost odd-indexed leaves, reduce even
            if i % 2 == 0:
                gains.append(int(Q15_ONE * 0.6))
            else:
                gains.append(Q15_ONE)
        return cls(name="hollow", gains_q15=gains,
                   description="Odd harmonic emphasis (clarinet-like)")


# =============================================================================
# TUNING MORPH STATE (Global)
# =============================================================================

@dataclass
class TuningMorphState:
    """
    Global tuning morph state.

    Handles slow transitions between instrument tunings (1-5 seconds).
    All voices share the same tuning morph position.
    """
    # Current and target tunings
    tuning_a: InstrumentTuning = field(default_factory=InstrumentTuning.flat)
    tuning_b: InstrumentTuning = field(default_factory=InstrumentTuning.flat)

    # Morph position: 0 = all A, 32767 = all B
    mu_q15: int = 0
    mu_target_q15: int = 0

    # Morph time in blocks (~2s default = 1500 blocks @ 750 Hz)
    tau_blocks: int = 1500

    # Per-block morph increment
    _mu_inc_per_block: int = field(default=0, repr=False)

    def set_target_tuning(self, tuning: InstrumentTuning,
                          morph_time_sec: float = 2.0) -> None:
        """
        Start morphing to a new tuning.

        Args:
            tuning: Target instrument tuning
            morph_time_sec: Transition time in seconds (1-5s typical)
        """
        # Shift current interpolated tuning to A
        current_gains = self.get_current_gains()
        self.tuning_a = InstrumentTuning(
            name=f"{self.tuning_a.name}→{tuning.name}",
            gains_q15=current_gains
        )

        # Set B to new target
        self.tuning_b = tuning

        # Reset morph position and set target
        self.mu_q15 = 0
        self.mu_target_q15 = Q15_ONE

        # Compute blocks and increment
        self.tau_blocks = int(morph_time_sec * BLOCKS_PER_SECOND)
        if self.tau_blocks > 0:
            self._mu_inc_per_block = Q15_ONE // self.tau_blocks
        else:
            self._mu_inc_per_block = Q15_ONE

    def advance_block(self) -> None:
        """Advance morph state by one block."""
        if self.mu_q15 < self.mu_target_q15:
            self.mu_q15 = min(
                self.mu_target_q15,
                self.mu_q15 + self._mu_inc_per_block
            )
        elif self.mu_q15 > self.mu_target_q15:
            self.mu_q15 = max(
                self.mu_target_q15,
                self.mu_q15 - self._mu_inc_per_block
            )

    def get_current_gains(self) -> List[int]:
        """
        Get interpolated 64-gain table for this block.

        Returns:
            List of 64 Q15 gains, interpolated between tuning_a and tuning_b
        """
        gains = []
        inv_mu = Q15_ONE - self.mu_q15

        for i in range(NUM_LEAVES):
            gain_a = self.tuning_a.gains_q15[i] if i < len(self.tuning_a.gains_q15) else Q15_ONE
            gain_b = self.tuning_b.gains_q15[i] if i < len(self.tuning_b.gains_q15) else Q15_ONE

            # Linear interpolation: gain = gain_a * (1-mu) + gain_b * mu
            interpolated = ((gain_a * inv_mu) >> 15) + ((gain_b * self.mu_q15) >> 15)
            gains.append(interpolated)

        return gains

    @property
    def is_morphing(self) -> bool:
        """True if currently transitioning between tunings."""
        return self.mu_q15 != self.mu_target_q15


# =============================================================================
# COMBINED VOICE STATE
# =============================================================================

@dataclass
class CarrierLiftVoiceState:
    """
    Complete per-voice state for Carrier + Lift architecture.

    Combines carrier, blend, and leaf content state.
    """
    voice_id: int = 0

    # Component states
    carrier: CarrierState = field(default_factory=CarrierState)
    blend: BlendState = field(default_factory=BlendState)
    leaf_content: LeafContent = field(default_factory=LeafContent)

    # Voice activity
    is_active: bool = False
    is_starting: bool = False
    is_stopping: bool = False

    # Current MIDI note info
    note: int = 60
    cents: int = 0

    # Partial amplitudes (fed to leaf generator)
    # 8 harmonics by default, can be modulated by symbol engine
    partial_amps_q15: List[int] = field(
        default_factory=lambda: [Q15_ONE, Q15_HALF, Q15_HALF // 2, Q15_HALF // 4,
                                  Q15_HALF // 8, Q15_HALF // 16, Q15_HALF // 32, 0]
    )

    def note_on(self, note: int, cents: int = 0, velocity_q15: int = Q15_ONE) -> None:
        """Start a new note."""
        self.note = note
        self.cents = cents
        self.is_active = True
        self.is_starting = True
        self.is_stopping = False

        # Reset carrier
        self.carrier.reset()
        self.carrier.voice_id = self.voice_id
        freq = 440.0 * (2.0 ** ((note - 69 + cents / 100.0) / 12.0))
        self.carrier.set_frequency(freq)
        self.carrier.envelope.gate_on()

        # Reset blend (carrier-only until wet ready)
        self.blend.voice_id = self.voice_id
        self.blend.reset_for_note_on()

        # Clear leaf content
        self.leaf_content.clear()

    def note_off(self) -> None:
        """Release current note."""
        self.is_stopping = True
        self.carrier.envelope.gate_off()

    def after_first_block(self) -> None:
        """Call after first block to clear is_starting flag."""
        self.is_starting = False


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'Q15_ONE',
    'Q15_HALF',
    'Q15_SCALE',
    'BLOCK_SIZE',
    'SAMPLE_RATE',
    'BLOCKS_PER_SECOND',
    'MASTER_DEPTH',
    'NUM_LEAVES',
    'IDWPT_LATENCY_BLOCKS',
    'DEFAULT_BLOOM_BLOCKS',
    'DEFAULT_BETA_TARGET_Q15',
    # Envelope
    'EnvelopeStage',
    'EnvelopeState',
    # Carrier
    'CarrierState',
    # Blend
    'BlendState',
    # Leaf
    'LeafContent',
    # Tuning
    'InstrumentTuning',
    'TuningMorphState',
    # Combined
    'CarrierLiftVoiceState',
]
