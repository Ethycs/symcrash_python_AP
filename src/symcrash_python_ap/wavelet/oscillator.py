"""
WPO v1 — Wavelet Packet Oscillator
===================================

Phase-coherent oscillator with bandlimited edges and two-stream output.

Key Features:
1. Phase accumulator with periodic invariants
2. Tree archetypes (HARMONIC, BRIGHT, HOLLOW, NOISY, TRANSIENT)
3. BLEP kernel for bandlimited discontinuities
4. Two-stream output: harmonic scaffold + packet texture with η blend
5. Controlled analogisms: drift, slop, hysteresis

FPGA Budget (16 voices):
- Phase accumulators: ~800 LUTs
- BLEP (shared): ~100 LUTs + 1KB BRAM
- Two-stream blend: ~480 LUTs
- Tree archetypes: ~20 LUTs
- Drift: ~30 LUTs
- Total: ~1,430 LUTs (6% of 24K)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum, IntEnum
import math

from ..prng import PCG32, VoicePRNG
from ..primitives import Q15_ONE, Q15_HALF, Q15_SCALE, BLOCK_SIZE, SAMPLE_RATE_HZ


# =============================================================================
# CONSTANTS
# =============================================================================

# Phase accumulator (32-bit fixed-point, 16.16 format)
PHASE_BITS = 32
PHASE_FRAC_BITS = 16
PHASE_ONE = 1 << PHASE_FRAC_BITS  # 1.0 in fixed-point
PHASE_MASK = (1 << PHASE_BITS) - 1

# Sample rate (for frequency calculations)
SAMPLE_RATE = SAMPLE_RATE_HZ

# BLEP kernel size
BLEP_KERNEL_SIZE = 64
BLEP_OVERSAMPLING = 8  # 8 fractional phase offsets

# Drift constants
DRIFT_SCALE = 256  # Drift magnitude (very small)
DRIFT_DECAY_Q15 = 32700  # ~0.998 decay per block


# =============================================================================
# TREE ARCHETYPES (Section 4)
# =============================================================================

class TreeArchetype(IntEnum):
    """
    Oscillator-native tree archetypes.

    These define musical behaviors, not analysis properties:
    - HARMONIC: Low leakage, harmonic-aligned (stable pitch)
    - BRIGHT: Tilt toward high leaves (brighter timbre)
    - HOLLOW: Odd harmonic emphasis (clarinet-like)
    - NOISY: Controlled residual in high leaves (bowed/breathy)
    - TRANSIENT: Edge-friendly, percussive attacks
    """
    HARMONIC = 0
    BRIGHT = 1
    HOLLOW = 2
    NOISY = 3
    TRANSIENT = 4


class TransitionKind(IntEnum):
    """
    Voice transition types for click-free note changes.

    - RETRIGGER: New articulation with controlled transient. Allows onset
                 operators, no phase continuity required. Use for staccato,
                 chord hits, new bow strokes.

    - GLIDE: Continuous pitch/target motion. Forbids onset operators,
             requires phase continuity and bounded parameter deltas.
             Use for portamento, MPE pitch slides, legato.

    - BRIDGE: Short transition regime (2-8 blocks) for structural changes.
              Temporarily increases traction, folds to body, applies change,
              then re-grows detail. Use for tree_id changes, permutation
              swaps, regime shifts.
    """
    RETRIGGER = 0  # New articulation (allow transient)
    GLIDE = 1      # Continuous motion (phase-continuous)
    BRIDGE = 2     # Structural change (fold→switch→regrow)


# Bridge regime defaults (WSE-STD-0xx)
BRIDGE_BLOCKS = 4           # Default bridge duration (~5.3ms @ 48k/64)
BRIDGE_TRACTION_HI = 26214  # Q15 0.8 - high traction during bridge
BRIDGE_TRACTION_NOM = 9830  # Q15 0.3 - nominal traction
GLIDE_THRESHOLD_CENTS = 30  # Below this, use param slew not bridge


@dataclass
class TreeArchetypeConfig:
    """Configuration for a tree archetype."""
    name: str
    tilt_q15: int           # Spectral tilt (-32768 = dark, +32767 = bright)
    odd_emphasis_q15: int   # Odd harmonic boost (0 = none, 32767 = full)
    noise_inject_q15: int   # High-band noise injection (0 = none)
    transient_q15: int      # Transient sharpness (0 = soft, 32767 = hard)
    harmonic_lock: bool     # Lock to harmonic series


# Canonical archetype configurations
TREE_ARCHETYPES: Dict[TreeArchetype, TreeArchetypeConfig] = {
    TreeArchetype.HARMONIC: TreeArchetypeConfig(
        name="Harmonic",
        tilt_q15=0,
        odd_emphasis_q15=0,
        noise_inject_q15=0,
        transient_q15=0,
        harmonic_lock=True,
    ),
    TreeArchetype.BRIGHT: TreeArchetypeConfig(
        name="Bright",
        tilt_q15=16384,  # +0.5 tilt
        odd_emphasis_q15=0,
        noise_inject_q15=0,
        transient_q15=0,
        harmonic_lock=True,
    ),
    TreeArchetype.HOLLOW: TreeArchetypeConfig(
        name="Hollow",
        tilt_q15=0,
        odd_emphasis_q15=24576,  # Strong odd emphasis
        noise_inject_q15=0,
        transient_q15=0,
        harmonic_lock=True,
    ),
    TreeArchetype.NOISY: TreeArchetypeConfig(
        name="Noisy",
        tilt_q15=8192,
        odd_emphasis_q15=0,
        noise_inject_q15=8192,  # Moderate noise
        transient_q15=0,
        harmonic_lock=False,
    ),
    TreeArchetype.TRANSIENT: TreeArchetypeConfig(
        name="Transient",
        tilt_q15=0,
        odd_emphasis_q15=0,
        noise_inject_q15=4096,
        transient_q15=24576,  # Sharp transients
        harmonic_lock=False,
    ),
}


def get_archetype_config(archetype: TreeArchetype) -> TreeArchetypeConfig:
    """Get configuration for a tree archetype."""
    return TREE_ARCHETYPES[archetype]


# =============================================================================
# BLEP KERNEL (Section 2)
# =============================================================================

def generate_blep_kernel(size: int = BLEP_KERNEL_SIZE) -> List[int]:
    """
    Generate a MinBLEP kernel for bandlimited discontinuities.

    This is a precomputed integrated sinc function that removes
    aliasing from hard edges (saw/pulse transitions).

    Returns Q15 samples.
    """
    kernel = []
    half = size // 2

    for i in range(size):
        # Centered position
        t = (i - half) / half

        if abs(t) < 1e-6:
            # Center point
            value = 0.5
        else:
            # Integrated sinc with Blackman window
            sinc = math.sin(math.pi * t * 2) / (math.pi * t * 2)
            # Blackman window
            window = 0.42 - 0.5 * math.cos(math.pi * (i / size + 0.5)) + \
                     0.08 * math.cos(2 * math.pi * (i / size + 0.5))
            # Integrate (cumulative sum approximation)
            value = 0.5 + 0.5 * sinc * window

        kernel.append(int(value * Q15_ONE))

    return kernel


# Precomputed BLEP kernel (shared across all voices)
BLEP_KERNEL = generate_blep_kernel()


def generate_blep_table(oversampling: int = BLEP_OVERSAMPLING) -> List[List[int]]:
    """
    Generate oversampled BLEP table for fractional phase offsets.

    Returns a table[phase_offset][sample_index] of Q15 values.
    """
    table = []
    for phase in range(oversampling):
        # Offset the kernel by fractional phase
        offset = phase / oversampling
        kernel = []
        for i in range(BLEP_KERNEL_SIZE):
            # Linear interpolation between base kernel samples
            pos = i + offset
            idx = int(pos)
            frac = pos - idx
            if idx + 1 < len(BLEP_KERNEL):
                value = int(BLEP_KERNEL[idx] * (1 - frac) + BLEP_KERNEL[idx + 1] * frac)
            else:
                value = BLEP_KERNEL[idx] if idx < len(BLEP_KERNEL) else Q15_ONE
            kernel.append(value)
        table.append(kernel)
    return table


# Precomputed BLEP table
BLEP_TABLE = generate_blep_table()


@dataclass
class BLEPState:
    """
    BLEP residual buffer state.

    When a discontinuity occurs, we inject the BLEP kernel
    and accumulate it into the output over subsequent samples.
    """
    buffer: List[int] = field(default_factory=lambda: [0] * BLEP_KERNEL_SIZE)
    write_pos: int = 0
    active: bool = False

    def inject_edge(self, amplitude_q15: int, phase_frac: int) -> None:
        """
        Inject a bandlimited edge at the given fractional phase.

        Args:
            amplitude_q15: Edge amplitude (positive = rising, negative = falling)
            phase_frac: Fractional phase offset (0-7 for 8x oversampling)
        """
        kernel = BLEP_TABLE[phase_frac % BLEP_OVERSAMPLING]

        for i, k in enumerate(kernel):
            pos = (self.write_pos + i) % BLEP_KERNEL_SIZE
            # Naive BLEP: subtract the kernel (residual correction)
            delta = (amplitude_q15 * (k - Q15_HALF)) >> 15
            self.buffer[pos] = self._saturate(self.buffer[pos] + delta)

        self.active = True

    def read_and_advance(self) -> int:
        """Read current sample and advance buffer."""
        if not self.active:
            return 0

        sample = self.buffer[self.write_pos]
        self.buffer[self.write_pos] = 0
        self.write_pos = (self.write_pos + 1) % BLEP_KERNEL_SIZE

        # Check if buffer is empty
        if all(s == 0 for s in self.buffer):
            self.active = False

        return sample

    @staticmethod
    def _saturate(value: int) -> int:
        """Saturate to Q15 range."""
        if value > Q15_ONE:
            return Q15_ONE
        if value < -Q15_SCALE:
            return -Q15_SCALE
        return value


# =============================================================================
# PHASE ACCUMULATOR (Section 1)
# =============================================================================

def frequency_to_phase_inc(freq_hz: float, sample_rate: int = SAMPLE_RATE) -> int:
    """
    Convert frequency in Hz to phase increment.

    Phase increment = freq * 2^PHASE_BITS / sample_rate

    This ensures the phase accumulator completes one full cycle (0 to 2^32)
    in exactly (sample_rate / freq_hz) samples.
    """
    # Use full 32-bit range, not just the fractional bits
    return int((freq_hz * (1 << PHASE_BITS)) / sample_rate)


def midi_to_frequency(note: int, cents: int = 0) -> float:
    """Convert MIDI note + cents to frequency in Hz."""
    return 440.0 * (2.0 ** ((note - 69 + cents / 100.0) / 12.0))


def midi_to_phase_inc(note: int, cents: int = 0, sample_rate: int = SAMPLE_RATE) -> int:
    """Convert MIDI note + cents directly to phase increment."""
    freq = midi_to_frequency(note, cents)
    return frequency_to_phase_inc(freq, sample_rate)


@dataclass
class PhaseAccumulator:
    """
    Per-voice phase accumulator for periodic waveforms.

    Provides:
    - Stable pitch via fixed-point phase tracking
    - Cycle boundary detection for periodic invariants
    - Sub-sample phase for BLEP fractional offset
    """
    phase: int = 0              # Current phase (32-bit fixed-point)
    phase_inc: int = 0          # Phase increment per sample
    cycle_count: int = 0        # Number of completed cycles

    def set_frequency(self, freq_hz: float, sample_rate: int = SAMPLE_RATE) -> None:
        """Set oscillator frequency."""
        self.phase_inc = frequency_to_phase_inc(freq_hz, sample_rate)

    def set_midi_note(self, note: int, cents: int = 0, sample_rate: int = SAMPLE_RATE) -> None:
        """Set oscillator frequency from MIDI note."""
        self.phase_inc = midi_to_phase_inc(note, cents, sample_rate)

    def advance(self) -> Tuple[int, bool, int]:
        """
        Advance phase by one sample.

        Returns:
            (new_phase, wrapped, fractional_offset)
            - new_phase: Current phase after advance
            - wrapped: True if cycle boundary crossed
            - fractional_offset: Sub-sample offset for BLEP (0-7)
        """
        old_phase = self.phase
        self.phase = (self.phase + self.phase_inc) & PHASE_MASK

        # Detect wrap (cycle boundary)
        wrapped = self.phase < old_phase
        if wrapped:
            self.cycle_count += 1

        # Fractional offset for BLEP (high bits of phase increment)
        frac_offset = (self.phase >> (PHASE_FRAC_BITS - 3)) & 0x7

        return self.phase, wrapped, frac_offset

    def advance_block(self, block_size: int = BLOCK_SIZE) -> List[Tuple[int, bool, int]]:
        """Advance phase for a full block, returning per-sample results."""
        results = []
        for _ in range(block_size):
            results.append(self.advance())
        return results

    def get_normalized_phase(self) -> float:
        """Get phase as normalized float [0, 1)."""
        return self.phase / (1 << PHASE_BITS)

    def reset(self) -> None:
        """Reset phase to zero."""
        self.phase = 0
        self.cycle_count = 0


# =============================================================================
# TWO-STREAM OUTPUT (Section 3)
# =============================================================================

@dataclass
class HarmonicScaffold:
    """
    Harmonic scaffold generator (tonal component).

    Generates a stable harmonic series that provides pitch integrity.
    This is the "baby Fourier" / additive component.

    Includes automatic gain normalization so all waveforms have equal peak amplitude.
    """
    num_harmonics: int = 8
    harmonic_amps_q15: List[int] = field(default_factory=lambda: [Q15_ONE] + [0] * 7)
    norm_gain_q15: int = Q15_ONE  # Normalization gain (Q15)

    def _compute_normalization(self) -> None:
        """
        Compute normalization gain so peak output reaches Q15_ONE.

        Uses empirical peak measurement over one cycle for accurate normalization.
        """
        if not any(self.harmonic_amps_q15):
            self.norm_gain_q15 = Q15_ONE
            return

        # Measure actual peak over one cycle (256 samples is enough)
        max_val = 0
        for i in range(256):
            phase = (i * PHASE_ONE) // 256
            output = 0
            for n, amp in enumerate(self.harmonic_amps_q15, start=1):
                if amp == 0:
                    continue
                h_phase = (phase * n) & PHASE_MASK
                sine = self._fast_sine(h_phase)
                output += (sine * amp) >> 15
            max_val = max(max_val, abs(output))

        if max_val > 0:
            # Scale so max_val maps to Q15_ONE
            self.norm_gain_q15 = min(Q15_ONE, (Q15_ONE * Q15_ONE) // max_val)
        else:
            self.norm_gain_q15 = Q15_ONE

    def set_saw_harmonics(self, num: int = 8) -> None:
        """Set amplitudes for saw wave (1/n falloff)."""
        self.num_harmonics = num
        self.harmonic_amps_q15 = []
        for n in range(1, num + 1):
            amp = Q15_ONE // n
            self.harmonic_amps_q15.append(amp)
        self._compute_normalization()

    def set_square_harmonics(self, num: int = 8) -> None:
        """Set amplitudes for square wave (odd harmonics only, 1/n falloff)."""
        self.num_harmonics = num
        self.harmonic_amps_q15 = []
        for n in range(1, num + 1):
            if n % 2 == 1:  # Odd only
                amp = Q15_ONE // n
            else:
                amp = 0
            self.harmonic_amps_q15.append(amp)
        self._compute_normalization()

    def set_triangle_harmonics(self, num: int = 8) -> None:
        """Set amplitudes for triangle wave (odd harmonics, 1/n² falloff)."""
        self.num_harmonics = num
        self.harmonic_amps_q15 = []
        for n in range(1, num + 1):
            if n % 2 == 1:  # Odd only
                amp = Q15_ONE // (n * n)
            else:
                amp = 0
            self.harmonic_amps_q15.append(amp)
        self._compute_normalization()

    def render_sample(self, phase: int) -> int:
        """
        Render one sample of the harmonic scaffold.

        Uses phase to compute each harmonic's contribution.
        Output is normalized for consistent peak amplitude across waveforms.
        """
        output = 0
        for n, amp in enumerate(self.harmonic_amps_q15, start=1):
            if amp == 0:
                continue
            # Harmonic phase = n * base_phase
            h_phase = (phase * n) & PHASE_MASK
            # Sine lookup (approximate with polynomial or table)
            sine = self._fast_sine(h_phase)
            output += (sine * amp) >> 15

        # Apply normalization gain
        output = (output * self.norm_gain_q15) >> 15

        # Saturate (should rarely hit after normalization)
        if output > Q15_ONE:
            output = Q15_ONE
        elif output < -Q15_SCALE:
            output = -Q15_SCALE

        return output

    @staticmethod
    def _fast_sine(phase: int) -> int:
        """
        Fast sine approximation from phase.

        Uses parabolic approximation for FPGA-friendliness.
        This is a Bhaskara I style approximation that produces smooth curves
        instead of triangle wave corners.

        The parabola 4x(1-x) maps [0,1] -> [0,1] with smooth peaks.
        We apply this per half-cycle for C1-continuous sine approximation.
        """
        # Normalize to [0, 2) representing half-cycles
        # Use 17 bits: bit 16 = half-cycle, bits 0-15 = position in half
        normalized = (phase >> (PHASE_BITS - 17)) & 0x1FFFF
        half_cycle = (normalized >> 16) & 1
        x = normalized & 0xFFFF  # 0 to 65535 within half-cycle

        # Parabolic approximation: 4x(1-x) scaled for Q15
        # x is 0..65535, we need x_norm = x/65535 in [0,1]
        # parabola = 4 * x_norm * (1 - x_norm)
        # = 4 * (x/65535) * (1 - x/65535)
        # = 4 * x * (65535 - x) / 65535^2
        #
        # To avoid overflow, compute in steps:
        # (x >> 8) * ((65535 - x) >> 8) gives us (x * (65535-x)) >> 16
        # Then multiply by 4 and scale for Q15

        x_hi = x >> 8  # 0..255
        inv_x_hi = (0xFFFF - x) >> 8  # 0..255

        # parabola in range 0..16256 (when x = 32768)
        parabola = x_hi * inv_x_hi

        # Scale to Q15 peak (32767)
        # max parabola value is 127*128 = 16256 at x=32768
        # We want this to map to Q15_ONE (32767)
        # Scale factor: 32767 / 16256 ≈ 2.016
        # Use (parabola * 2 + parabola >> 4) ≈ parabola * 2.0625
        y = (parabola << 1) + (parabola >> 4)

        # Clamp to Q15 (handles rounding edge cases)
        if y > Q15_ONE:
            y = Q15_ONE

        # Apply sign based on half-cycle
        if half_cycle:
            return -y
        else:
            return y


@dataclass
class PacketTexture:
    """
    Packet texture generator (wavelet component).

    This wraps the existing wavelet packet synthesis to provide
    the "organic" texture component of two-stream output.
    """
    coefficients_q15: List[int] = field(default_factory=lambda: [0] * 64)
    archetype: TreeArchetype = TreeArchetype.HARMONIC

    def apply_archetype(self, archetype: TreeArchetype) -> None:
        """Apply tree archetype configuration to coefficients."""
        self.archetype = archetype
        config = get_archetype_config(archetype)

        # Apply tilt
        if config.tilt_q15 != 0:
            self._apply_tilt(config.tilt_q15)

        # Apply odd emphasis
        if config.odd_emphasis_q15 != 0:
            self._apply_odd_emphasis(config.odd_emphasis_q15)

    def _apply_tilt(self, tilt_q15: int) -> None:
        """Apply spectral tilt to coefficients."""
        n = len(self.coefficients_q15)
        for i in range(n):
            # Linear tilt: low indices darker, high indices brighter
            tilt_factor = ((i * 2 - n) * tilt_q15) // n
            scale = Q15_ONE + tilt_factor
            self.coefficients_q15[i] = (self.coefficients_q15[i] * scale) >> 15

    def _apply_odd_emphasis(self, emphasis_q15: int) -> None:
        """Boost odd-indexed coefficients."""
        for i in range(len(self.coefficients_q15)):
            if i % 2 == 1:  # Odd index
                boost = Q15_ONE + (emphasis_q15 >> 1)
                self.coefficients_q15[i] = (self.coefficients_q15[i] * boost) >> 15


def blend_two_stream(
    scaffold_q15: int,
    texture_q15: int,
    eta_q15: int,
) -> int:
    """
    Blend harmonic scaffold and packet texture.

    Args:
        scaffold_q15: Sample from harmonic scaffold
        texture_q15: Sample from packet texture
        eta_q15: Blend factor (0 = pure scaffold, 32767 = pure texture)

    Returns:
        Blended sample in Q15
    """
    # output = scaffold * (1 - η) + texture * η
    inv_eta = Q15_ONE - eta_q15
    output = ((scaffold_q15 * inv_eta) >> 15) + ((texture_q15 * eta_q15) >> 15)

    # Saturate
    if output > Q15_ONE:
        return Q15_ONE
    if output < -Q15_SCALE:
        return -Q15_SCALE
    return output


# =============================================================================
# DRIFT / ANALOGISMS (Section 6)
# =============================================================================

@dataclass
class AnalogDrift:
    """
    Controlled analog-style drift and instabilities.

    Provides:
    - Phase drift (tiny random walk)
    - Amplitude drift (slow wander)
    - Detune spread (for unison)

    All drift is deterministic via per-voice PRNG.
    """
    phase_drift: int = 0        # Accumulated phase drift
    amp_drift_q15: int = 0      # Amplitude drift offset
    detune_cents: int = 0       # Static detune for unison

    # Drift parameters
    phase_drift_amount: int = DRIFT_SCALE
    amp_drift_amount: int = 64  # Very subtle

    def update(self, prng: VoicePRNG) -> None:
        """
        Update drift state using PRNG.

        Called once per block.
        """
        # Phase drift: random walk with decay
        drift_delta = prng.draw_q15() >> 8  # Small random value
        self.phase_drift = (
            ((self.phase_drift * DRIFT_DECAY_Q15) >> 15) +
            ((drift_delta * self.phase_drift_amount) >> 15)
        )

        # Amplitude drift: slower random walk
        amp_delta = prng.draw_q15() >> 10  # Even smaller
        self.amp_drift_q15 = (
            ((self.amp_drift_q15 * DRIFT_DECAY_Q15) >> 15) +
            ((amp_delta * self.amp_drift_amount) >> 15)
        )

    def apply_phase_drift(self, phase_inc: int) -> int:
        """Apply phase drift to phase increment."""
        return phase_inc + self.phase_drift

    def apply_amp_drift(self, amp_q15: int) -> int:
        """Apply amplitude drift to output."""
        return amp_q15 + self.amp_drift_q15

    def set_unison_detune(self, voice_index: int, spread_cents: int, num_voices: int) -> None:
        """
        Set static detune for unison voices.

        Spreads voices evenly around center pitch.
        """
        if num_voices <= 1:
            self.detune_cents = 0
        else:
            # Spread from -spread/2 to +spread/2
            position = (voice_index - (num_voices - 1) / 2) / ((num_voices - 1) / 2)
            self.detune_cents = int(position * spread_cents / 2)


# =============================================================================
# COMPLETE OSCILLATOR
# =============================================================================

@dataclass
class WPOState:
    """
    Complete WPO v1 oscillator state.

    Combines all components:
    - Phase accumulator
    - BLEP for bandlimited edges
    - Two-stream output (scaffold + texture)
    - Analog drift
    """
    voice_id: int = 0

    # Phase
    phase: PhaseAccumulator = field(default_factory=PhaseAccumulator)

    # BLEP
    blep: BLEPState = field(default_factory=BLEPState)

    # Two-stream
    scaffold: HarmonicScaffold = field(default_factory=HarmonicScaffold)
    texture: PacketTexture = field(default_factory=PacketTexture)
    eta_q15: int = Q15_HALF  # 50% blend by default

    # Tree
    archetype_a: TreeArchetype = TreeArchetype.HARMONIC
    archetype_b: TreeArchetype = TreeArchetype.HARMONIC
    archetype_blend_q15: int = 0

    # Drift
    drift: AnalogDrift = field(default_factory=AnalogDrift)

    # Output state
    last_output_q15: int = 0
    dc_block_state: int = 0
    first_cycle: bool = True  # Skip BLEP on first cycle to avoid click

    # Transition state (for click-free note changes)
    transition_kind: TransitionKind = TransitionKind.RETRIGGER
    bridge_blocks_remaining: int = 0  # Blocks left in bridge regime
    bridge_phase: int = 0  # 0=fold, 1=switch, 2=regrow

    # Glide state (for smooth pitch transitions)
    # Per-sample interpolation to avoid stair-stepping
    glide_time_blocks: int = 0      # Total glide time in blocks
    target_phase_inc: int = 0       # Target phase increment for glide
    start_phase_inc: int = 0        # Starting phase increment (for lerp)
    glide_samples_remaining: int = 0  # Samples left in glide (not blocks!)
    glide_samples_total: int = 0    # Total samples in glide (for envelope)
    glide_inc_delta: int = 0        # Per-sample increment delta (fixed-point, Q16)
    glide_gain_q15: int = Q15_ONE   # Amplitude envelope during glide

    # Traction (for bridge regime stability)
    traction_q15: int = BRIDGE_TRACTION_NOM

    # BLEP control - only needed for waveforms with discontinuities (saw, square)
    needs_blep: bool = True

    # Seam continuity state
    # Seam hints indicate known discontinuity points (note changes, parameter jumps)
    # where we need C¹-continuous bridging (value + slope match)
    seam_hints: List[int] = field(default_factory=list)  # Sample indices within next block
    seam_bridge_len: int = 4  # Samples for Hermite bridge (half-width) - TIGHT

    # History buffer for slope computation at seam points
    history_buffer: List[int] = field(default_factory=list)
    history_max_len: int = 4  # Keep last 4 samples for slope computation

    # Pending seam state (for block-crossing seams)
    pending_seam: bool = False  # Seam at block boundary
    pending_seam_old_samples: List[int] = field(default_factory=list)  # Tail of previous block

    # Pending pitch change (for automatic dual-render)
    pending_pitch_change: bool = False
    pending_new_note: Optional[int] = None
    pending_new_cents: int = 0
    pending_seam_sample: int = 0  # Sample index for the pitch change (default = start of block)

    def add_seam_hint(self, sample_index: int) -> None:
        """
        Add a seam hint at a specific sample index within the next block.

        A seam is a known discontinuity point where we need C¹-continuous
        bridging. Use this when you know a parameter change will occur.

        Args:
            sample_index: Sample index within the next render_block() call
                         (0 = first sample, BLOCK_SIZE-1 = last sample)
        """
        if sample_index not in self.seam_hints:
            self.seam_hints.append(sample_index)

    def add_seam_at_block_boundary(self) -> None:
        """
        Signal that a seam occurs at the next block boundary.

        Use this before calling render_block() when you're about to change
        parameters (note, waveform, etc.) that would cause a discontinuity.
        """
        self.pending_seam = True

    def clear_seam_hints(self) -> None:
        """Clear all seam hints (called automatically after render_block)."""
        self.seam_hints.clear()
        self.pending_seam = False
        self.pending_seam_old_samples.clear()

    def configure(
        self,
        note: int,
        cents: int = 0,
        archetype: TreeArchetype = TreeArchetype.HARMONIC,
        eta_q15: int = Q15_HALF,
        waveform: str = "saw",
        reset_phase: bool = True,
    ) -> None:
        """
        Configure oscillator for playback.

        Args:
            note: MIDI note number
            cents: Fine tuning in cents
            archetype: Tree archetype
            eta_q15: Two-stream blend (0 = scaffold, 32767 = texture)
            waveform: Scaffold waveform ("saw", "square", "triangle", "sine")
            reset_phase: If True, reset phase to 0 (note-on). If False, keep
                         phase continuous (legato/pitch change).
        """
        # Set frequency
        self.phase.set_midi_note(note, cents)

        # Phase reset behavior:
        # - reset_phase=True: New note, reset to 0 and skip first BLEP
        # - reset_phase=False: Legato/pitch change, keep phase continuous
        if reset_phase:
            self.phase.reset()
            self.first_cycle = True  # Skip first BLEP to avoid click

        # Set archetype
        self.archetype_a = archetype
        self.texture.apply_archetype(archetype)

        # Set blend
        self.eta_q15 = eta_q15

        # Configure scaffold based on waveform
        # Note: Scaffold uses additive synthesis (sum of sine harmonics), which is
        # inherently bandlimited. BLEP is only needed for naive/direct waveforms.
        # Since we use harmonic synthesis, we disable BLEP by default.
        self.needs_blep = False  # Harmonic synthesis is already bandlimited

        if waveform == "saw":
            self.scaffold.set_saw_harmonics()
        elif waveform == "square":
            self.scaffold.set_square_harmonics()
        elif waveform == "triangle":
            self.scaffold.set_triangle_harmonics()
        else:  # sine
            self.scaffold.harmonic_amps_q15 = [Q15_ONE] + [0] * 7

        # Apply archetype modifications to scaffold harmonics
        self._apply_archetype_to_scaffold(archetype)

        # Recompute normalization after archetype modifications
        self.scaffold._compute_normalization()

    def set_pitch(self, note: int, cents: int = 0, seam_sample: int = 0) -> None:
        """
        Change pitch without resetting phase (phase-continuous).

        Uses dual-render crossfade by default for click-free transitions.
        The pitch change is queued and applied during the next render_block()
        with automatic crossfade at the seam point.

        Args:
            note: MIDI note number
            cents: Fine tuning in cents
            seam_sample: Sample index within the block where the change occurs
                        (default 0 = start of block)
        """
        self.pending_pitch_change = True
        self.pending_new_note = note
        self.pending_new_cents = cents
        self.pending_seam_sample = seam_sample

    def set_pitch_immediate(self, note: int, cents: int = 0) -> None:
        """
        Change pitch immediately without dual-render crossfade.

        Use this only when you know a click won't occur (e.g., voice is silent)
        or when you're manually handling the crossfade yourself.
        """
        self.phase.set_midi_note(note, cents)

    def glide_to(self, note: int, cents: int = 0, glide_blocks: int = 8) -> None:
        """
        Glide to a new pitch over N blocks (phase-continuous).

        Uses per-sample linear interpolation for smooth portamento.
        Includes amplitude envelope to mask any discontinuities.

        Args:
            note: Target MIDI note
            cents: Target fine tuning
            glide_blocks: Number of blocks for the glide (0 = instant)
        """
        self.transition_kind = TransitionKind.GLIDE
        self.target_phase_inc = midi_to_phase_inc(note, cents)
        self.glide_time_blocks = glide_blocks

        if glide_blocks <= 0:
            # Instant change
            self.phase.phase_inc = self.target_phase_inc
            self.glide_samples_remaining = 0
            self.glide_samples_total = 0
            self.glide_inc_delta = 0
            self.start_phase_inc = self.target_phase_inc
            self.glide_gain_q15 = Q15_ONE
        else:
            # Capture start position for accurate lerp
            self.start_phase_inc = self.phase.phase_inc
            total_samples = glide_blocks * BLOCK_SIZE
            self.glide_samples_remaining = total_samples
            self.glide_samples_total = total_samples
            # Compute delta in Q16 for sub-integer precision
            delta = self.target_phase_inc - self.start_phase_inc
            # Q16 fixed-point delta: (delta << 16) / total_samples
            self.glide_inc_delta = (delta << 16) // total_samples
            self.glide_gain_q15 = Q15_ONE  # Will be updated per-sample

    def start_bridge(self, blocks: int = BRIDGE_BLOCKS) -> None:
        """
        Start a bridge regime for structural transitions.

        Use before changing tree_id, permutation, or regime.
        The bridge temporarily stabilizes state during the change.

        Phases:
        - Phase 0 (fold): Increase traction, diffuse up
        - Phase 1 (switch): Apply structural change
        - Phase 2 (regrow): Reduce traction, re-enable detail
        """
        self.transition_kind = TransitionKind.BRIDGE
        self.bridge_blocks_remaining = blocks
        self.bridge_phase = 0
        self.traction_q15 = BRIDGE_TRACTION_HI

    def retrigger(self, note: int, cents: int = 0) -> None:
        """
        Retrigger with a new note (allows transient, resets phase).

        Use for staccato, new articulations, chord hits.
        """
        self.transition_kind = TransitionKind.RETRIGGER
        self.phase.set_midi_note(note, cents)
        self.phase.reset()
        self.first_cycle = True
        self.bridge_blocks_remaining = 0
        self.glide_samples_remaining = 0
        self.glide_samples_total = 0
        self.glide_inc_delta = 0
        self.glide_gain_q15 = Q15_ONE

    def _update_glide_sample(self) -> None:
        """Update phase increment and amplitude envelope for glide (per-sample)."""
        if self.glide_samples_remaining > 0:
            self.glide_samples_remaining -= 1

            # Compute exact lerp position to avoid accumulated error
            # progress goes from 0.0 (start) to 1.0 (end)
            if self.glide_samples_total > 0:
                samples_elapsed = self.glide_samples_total - self.glide_samples_remaining
                # Use Q16 fixed-point lerp for precision
                # lerp = start + (target - start) * progress
                # = start + delta * (samples_elapsed / total_samples)
                # = start + (delta * samples_elapsed) / total_samples
                delta = self.target_phase_inc - self.start_phase_inc
                interpolated = self.start_phase_inc + (delta * samples_elapsed) // self.glide_samples_total
                self.phase.phase_inc = interpolated

                # Compute amplitude envelope: "smile" curve that dips in the middle
                # This masks any discontinuity during the pitch transition
                # t goes from 0.0 -> 1.0 as we progress through the glide
                t = samples_elapsed / self.glide_samples_total
                # Parabolic dip: 1 - depth * 4 * t * (1-t)
                # At t=0.5 (middle), dip is maximum
                # depth = 0.15 (15% dip at center)
                depth_q15 = 4915  # Q15 0.15
                dip = int(depth_q15 * 4 * t * (1 - t))
                self.glide_gain_q15 = Q15_ONE - dip

            if self.glide_samples_remaining == 0:
                # Snap to exact target at end to avoid any rounding drift
                self.phase.phase_inc = self.target_phase_inc
                self.glide_inc_delta = 0
                self.glide_gain_q15 = Q15_ONE

    def _update_bridge(self) -> None:
        """Update bridge regime state (called once per block)."""
        if self.bridge_blocks_remaining > 0:
            blocks_per_phase = max(1, BRIDGE_BLOCKS // 3)

            if self.bridge_phase == 0:
                # Phase 0: Fold to body - high traction
                self.traction_q15 = BRIDGE_TRACTION_HI
                if self.bridge_blocks_remaining <= BRIDGE_BLOCKS - blocks_per_phase:
                    self.bridge_phase = 1

            elif self.bridge_phase == 1:
                # Phase 1: Switch point - structural changes applied externally
                if self.bridge_blocks_remaining <= BRIDGE_BLOCKS - 2 * blocks_per_phase:
                    self.bridge_phase = 2

            elif self.bridge_phase == 2:
                # Phase 2: Regrow detail - reduce traction gradually
                progress = (BRIDGE_BLOCKS - self.bridge_blocks_remaining) / BRIDGE_BLOCKS
                self.traction_q15 = int(
                    BRIDGE_TRACTION_HI * (1 - progress) +
                    BRIDGE_TRACTION_NOM * progress
                )

            self.bridge_blocks_remaining -= 1

            if self.bridge_blocks_remaining == 0:
                # Bridge complete
                self.traction_q15 = BRIDGE_TRACTION_NOM
                self.transition_kind = TransitionKind.RETRIGGER

    def _apply_archetype_to_scaffold(self, archetype: TreeArchetype) -> None:
        """
        Apply archetype-specific modifications to the harmonic scaffold.

        This makes each archetype sound distinctly different.
        """
        config = get_archetype_config(archetype)

        # BRIGHT: Boost high harmonics (positive tilt)
        if config.tilt_q15 > 0:
            for i in range(len(self.scaffold.harmonic_amps_q15)):
                # Higher harmonics get boosted
                boost = (config.tilt_q15 * (i + 1)) // (len(self.scaffold.harmonic_amps_q15) * 2)
                self.scaffold.harmonic_amps_q15[i] = min(
                    Q15_ONE,
                    self.scaffold.harmonic_amps_q15[i] + boost
                )

        # HOLLOW: Emphasize odd harmonics, reduce even harmonics
        if config.odd_emphasis_q15 > 0:
            for i in range(len(self.scaffold.harmonic_amps_q15)):
                harmonic_num = i + 1
                if harmonic_num % 2 == 0:  # Even harmonic
                    # Reduce even harmonics
                    reduction = (self.scaffold.harmonic_amps_q15[i] * config.odd_emphasis_q15) >> 16
                    self.scaffold.harmonic_amps_q15[i] = max(0, self.scaffold.harmonic_amps_q15[i] - reduction)
                else:  # Odd harmonic
                    # Boost odd harmonics slightly
                    boost = (config.odd_emphasis_q15 * self.scaffold.harmonic_amps_q15[i]) >> 17
                    self.scaffold.harmonic_amps_q15[i] = min(Q15_ONE, self.scaffold.harmonic_amps_q15[i] + boost)

        # TRANSIENT: Boost higher harmonics for sharper attack
        if config.transient_q15 > 0:
            for i in range(len(self.scaffold.harmonic_amps_q15)):
                # Scale boost by harmonic number - higher harmonics get more boost
                boost = (config.transient_q15 * (i + 1)) // (len(self.scaffold.harmonic_amps_q15) * 3)
                self.scaffold.harmonic_amps_q15[i] = min(
                    Q15_ONE,
                    self.scaffold.harmonic_amps_q15[i] + boost
                )

    def render_sample(self, prng: Optional[VoicePRNG] = None) -> int:
        """
        Render one sample of oscillator output.

        Args:
            prng: Optional PRNG for noise injection

        Returns:
            Q15 sample
        """
        # Update glide per-sample (before phase advance for smoothness)
        self._update_glide_sample()

        # Advance phase
        phase, wrapped, frac_offset = self.phase.advance()

        # Apply drift to phase
        if self.drift.phase_drift != 0:
            phase = (phase + self.drift.phase_drift) & PHASE_MASK

        # Generate scaffold sample
        scaffold_sample = self.scaffold.render_sample(phase)

        # Generate texture sample (placeholder - integrate with wavelet engine)
        texture_sample = self._simple_texture(phase, prng)

        # Blend two streams
        output = blend_two_stream(scaffold_sample, texture_sample, self.eta_q15)

        # Add BLEP correction for edges (only for saw/square waveforms)
        # Triangle/sine are continuous and don't need BLEP
        if self.needs_blep:
            if wrapped:
                if self.first_cycle:
                    self.first_cycle = False
                else:
                    # Inject BLEP at cycle boundary
                    edge_amp = Q15_ONE >> 1  # Half amplitude edge
                    self.blep.inject_edge(edge_amp, frac_offset)

            output += self.blep.read_and_advance()

        # Apply glide amplitude envelope (masks transition artifacts)
        if self.glide_gain_q15 < Q15_ONE:
            output = (output * self.glide_gain_q15) >> 15

        # Apply drift
        output = self.drift.apply_amp_drift(output)

        # DC blocking (simple high-pass)
        self.dc_block_state = (self.dc_block_state * 32700 + output * 67) >> 15
        output = output - self.dc_block_state

        # Saturate
        if output > Q15_ONE:
            output = Q15_ONE
        elif output < -Q15_SCALE:
            output = -Q15_SCALE

        self.last_output_q15 = output
        return output

    def render_block(
        self,
        block_size: int = BLOCK_SIZE,
        prng: Optional[VoicePRNG] = None,
    ) -> List[int]:
        """
        Render a full block of samples with automatic dual-render for pitch changes.

        If a pitch change is pending (via set_pitch()), automatically uses
        dual-render crossfade for click-free transitions. Otherwise renders
        normally with C¹-continuous seam bridging for any seam hints.

        Args:
            block_size: Number of samples
            prng: Optional PRNG for noise/drift

        Returns:
            List of Q15 samples
        """
        # Check for pending pitch change - use dual-render automatically
        if self.pending_pitch_change and self.pending_new_note is not None:
            samples = self.render_with_seam_at(
                block_size=block_size,
                seam_index=self.pending_seam_sample,
                new_note=self.pending_new_note,
                new_cents=self.pending_new_cents,
                prng=prng,
            )
            # Clear the pending pitch change
            self.pending_pitch_change = False
            self.pending_new_note = None
            self.pending_new_cents = 0
            self.pending_seam_sample = 0
            return samples

        # Normal render path (no pitch change pending)
        # Update bridge transition state once per block
        # Note: glide is updated per-sample in render_sample() for smoothness
        self._update_bridge()

        # Update drift once per block
        if prng:
            self.drift.update(prng)

        # Render raw samples
        samples = [self.render_sample(prng) for _ in range(block_size)]

        # Apply C¹-continuous bridging at seam points
        if self.seam_hints or self.pending_seam:
            samples = self._apply_seam_bridges(samples)

        # Update history buffer for slope computation at future seams
        self._update_history(samples)

        # Clear seam hints after processing
        self.clear_seam_hints()

        return samples

    def _update_history(self, samples: List[int]) -> None:
        """Update history buffer with tail samples for slope computation."""
        # Keep last N samples for slope estimation at block boundaries
        tail = samples[-self.history_max_len:] if len(samples) >= self.history_max_len else samples
        self.history_buffer = list(tail)

    def _apply_seam_bridges(self, samples: List[int]) -> List[int]:
        """
        Apply C¹-continuous Hermite bridges at all seam points.

        The Hermite bridge guarantees both value AND slope continuity:
            H(t) = (2t³ - 3t² + 1)p₀ + (t³ - 2t² + t)m₀ + (-2t³ + 3t²)p₁ + (t³ - t²)m₁

        where p₀, p₁ are endpoint values and m₀, m₁ are endpoint slopes.
        """
        out = list(samples)

        # Handle block boundary seam first
        if self.pending_seam and self.pending_seam_old_samples:
            out = self._apply_boundary_seam_bridge(out)

        # Handle interior seam hints
        for seam_idx in sorted(self.seam_hints):
            if self.seam_bridge_len <= seam_idx < len(out) - self.seam_bridge_len:
                out = self._apply_hermite_bridge(out, seam_idx)

        return out

    def _apply_boundary_seam_bridge(self, samples: List[int]) -> List[int]:
        """
        Apply C¹-continuous bridge at block boundary.

        This handles the case where we have limited old samples (from history buffer)
        and need to smoothly connect to the new waveform. Uses pure Hermite spline
        to guarantee both value and slope continuity at the endpoints.
        """
        out = list(samples)
        old = self.pending_seam_old_samples

        if len(old) < 2 or len(samples) < 4:
            return out

        # Use smaller bridge width since we have limited old samples
        M = min(len(old), 4, len(samples))

        # Compute slopes at boundaries
        old_slope = old[-1] - old[-2] if len(old) >= 2 else 0
        new_slope = samples[1] - samples[0] if len(samples) >= 2 else 0

        # Endpoint values for Hermite
        p0 = old[-1]      # Last old sample (we match this exactly at t=0)
        p1 = samples[M]   # First sample AFTER bridge (we match this exactly at t=1)
        m0 = old_slope    # Slope coming out of old waveform
        m1 = new_slope    # Slope going into new waveform

        # Scale tangents by bridge width for proper Hermite
        m0_scaled = m0 * M
        m1_scaled = m1 * M

        for i in range(M):
            # t goes from 0 to 1 over the bridge region
            # At t=0: output = p0 with slope m0
            # At t=1: output = p1 with slope m1
            t = (i + 1) / (M + 1)  # Offset so we don't overwrite sample 0 with p0

            # Cubic Hermite basis functions
            t2 = t * t
            t3 = t2 * t
            h00 = 2*t3 - 3*t2 + 1      # value at p0
            h10 = t3 - 2*t2 + t         # tangent at p0
            h01 = -2*t3 + 3*t2          # value at p1
            h11 = t3 - t2               # tangent at p1

            # Hermite interpolation
            blended = int(h00 * p0 + h10 * m0_scaled + h01 * p1 + h11 * m1_scaled)

            # Saturate
            if blended > Q15_ONE:
                blended = Q15_ONE
            elif blended < -Q15_SCALE:
                blended = -Q15_SCALE

            out[i] = blended

        return out

    def _apply_hermite_bridge(self, samples: List[int], seam_idx: int) -> List[int]:
        """
        Apply smoothstep crossfade at an interior seam point.

        For pitch changes in additive synthesis, the waveform shape changes
        discontinuously even with phase continuity. This crossfade blends
        the pre-change and post-change waveforms smoothly.

        Args:
            samples: Audio samples (contains both pre- and post-change data)
            seam_idx: Index of the seam point where parameters changed

        Returns:
            Samples with crossfade applied around seam_idx
        """
        out = list(samples)
        fade_half = 4  # 4 samples each side = 8 sample total crossfade

        # Boundary checks
        if seam_idx < fade_half or seam_idx >= len(samples) - fade_half:
            return out

        # Get slopes on each side of the seam
        # Left side (pre-change): samples[seam_idx - fade_half] to samples[seam_idx - 1]
        left_slope = samples[seam_idx - 1] - samples[seam_idx - 2] if seam_idx >= 2 else 0

        # Right side (post-change): samples[seam_idx] to samples[seam_idx + fade_half]
        right_slope = samples[seam_idx + 1] - samples[seam_idx] if seam_idx + 1 < len(samples) else 0

        # Apply smoothstep crossfade across the seam region
        for i in range(-fade_half, fade_half + 1):
            idx = seam_idx + i
            if idx < 0 or idx >= len(out):
                continue

            # Smoothstep blend: 0 at left edge, 1 at right edge
            t = (i + fade_half) / (2 * fade_half)
            blend = 3 * t * t - 2 * t * t * t

            if i < 0:
                # Before seam: extrapolate pre-change trajectory
                # Original sample is from pre-change, keep it but start blending
                pre_val = samples[idx]
                # Project post-change waveform backwards
                post_projected = samples[seam_idx] + right_slope * i
                post_projected = max(-Q15_SCALE, min(Q15_ONE, post_projected))
                out[idx] = int(pre_val * (1 - blend) + post_projected * blend)
            else:
                # After seam: blend from extrapolated pre-change to actual post-change
                # Project pre-change waveform forward
                pre_projected = samples[seam_idx - 1] + left_slope * (i + 1)
                pre_projected = max(-Q15_SCALE, min(Q15_ONE, pre_projected))
                post_val = samples[idx]
                out[idx] = int(pre_projected * (1 - blend) + post_val * blend)

            # Saturate
            if out[idx] > Q15_ONE:
                out[idx] = Q15_ONE
            elif out[idx] < -Q15_SCALE:
                out[idx] = -Q15_SCALE

        return out

    def prepare_for_parameter_change(self, samples_before_change: List[int] = None) -> None:
        """
        Prepare for a parameter change that would cause discontinuity.

        Call this BEFORE changing parameters (note, waveform, etc.) and provide
        the tail samples from the previous render. The next render_block() will
        apply a Hermite bridge at the boundary.

        Args:
            samples_before_change: Last few samples before the change (optional,
                                   uses history_buffer if not provided)
        """
        self.pending_seam = True
        if samples_before_change:
            self.pending_seam_old_samples = list(samples_before_change[-self.seam_bridge_len:])
        elif self.history_buffer:
            self.pending_seam_old_samples = list(self.history_buffer)

    def render_with_seam_at(
        self,
        block_size: int,
        seam_index: int,
        new_note: int = None,
        new_cents: int = 0,
        prng: Optional[VoicePRNG] = None,
    ) -> List[int]:
        """
        Render a block with a known seam point where parameters change.

        Uses dual-render crossfade: renders the overlap region with BOTH
        old and new parameters, then crossfades. This is the canonical
        regularity operator for parameter changes.

        Args:
            block_size: Number of samples
            seam_index: Sample index where parameter change occurs
            new_note: New MIDI note (if changing pitch)
            new_cents: New fine tuning
            prng: Optional PRNG

        Returns:
            List of Q15 samples with smooth seam
        """
        # Update bridge/drift once per block
        self._update_bridge()
        if prng:
            self.drift.update(prng)

        # Seam width for crossfade (M samples)
        M = 8

        # Clamp seam_index to valid range with room for crossfade
        seam_index = max(M, min(block_size - M, seam_index))

        # === Phase 1: Render up to seam with old parameters ===
        samples_before = [self.render_sample(prng) for _ in range(seam_index)]

        # === Phase 2: Save state at seam point ===
        # Save phase state so we can render overlap with both old and new params
        saved_phase = self.phase.phase
        saved_phase_inc = self.phase.phase_inc
        saved_dc_state = self.dc_block_state
        saved_last_output = self.last_output_q15

        # === Phase 3: Render overlap region with OLD parameters ===
        old_overlap = [self.render_sample(prng) for _ in range(M)]

        # === Phase 4: Restore state and change parameters ===
        self.phase.phase = saved_phase
        self.dc_block_state = saved_dc_state
        self.last_output_q15 = saved_last_output

        if new_note is not None:
            self.set_pitch(new_note, new_cents)

        # === Phase 5: Render overlap region with NEW parameters ===
        new_overlap = [self.render_sample(prng) for _ in range(M)]

        # === Phase 6: Hermite crossfade with C¹ continuity ===
        # For smooth seams we need BOTH value AND slope to match.
        # Use cubic Hermite interpolation between the two waveforms.
        #
        # At t=0: output = old[0], slope = old_slope
        # At t=1: output = new[M-1], slope = new_slope
        # In between: Hermite spline smoothly interpolates both.

        # Compute slopes at boundaries
        old_slope = old_overlap[1] - old_overlap[0] if M > 1 else 0
        new_slope = new_overlap[-1] - new_overlap[-2] if M > 1 else 0

        # Endpoint values
        p0 = old_overlap[0]  # Start value (from old)
        p1 = new_overlap[-1]  # End value (from new)
        m0 = old_slope * M  # Start tangent (scaled by interval)
        m1 = new_slope * M  # End tangent (scaled by interval)

        crossfaded = []
        for i in range(M):
            t = i / (M - 1) if M > 1 else 1.0

            # Cubic Hermite basis functions
            h00 = 2*t*t*t - 3*t*t + 1      # value at p0
            h10 = t*t*t - 2*t*t + t         # tangent at p0
            h01 = -2*t*t*t + 3*t*t          # value at p1
            h11 = t*t*t - t*t               # tangent at p1

            # Hermite interpolation
            hermite_val = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

            # Blend Hermite curve with actual waveforms for texture
            # At edges, use pure waveform; in middle, use Hermite
            # This preserves waveform character while ensuring smooth endpoints
            edge_blend = 4 * t * (1 - t)  # Peaks at 0.5, zero at edges

            # Mix: at t=0 use old, at t=1 use new, in middle blend with Hermite
            if t < 0.5:
                # First half: blend old toward Hermite
                wave_val = old_overlap[i]
                blend_t = t * 2  # 0->1 over first half
            else:
                # Second half: blend Hermite toward new
                wave_val = new_overlap[i]
                blend_t = (t - 0.5) * 2  # 0->1 over second half

            # Smoothstep blend factor
            smooth_t = blend_t * blend_t * (3 - 2 * blend_t)

            if t < 0.5:
                blended = int(wave_val * (1 - smooth_t) + hermite_val * smooth_t)
            else:
                blended = int(hermite_val * (1 - smooth_t) + wave_val * smooth_t)

            # Saturate
            if blended > Q15_ONE:
                blended = Q15_ONE
            elif blended < -Q15_SCALE:
                blended = -Q15_SCALE
            crossfaded.append(blended)

        # === Phase 7: Render remainder with new parameters ===
        remaining = block_size - seam_index - M
        samples_after = [self.render_sample(prng) for _ in range(remaining)]

        # === Phase 8: Concatenate ===
        samples = samples_before + crossfaded + samples_after

        # Update history
        self._update_history(samples)

        return samples

    def _simple_texture(self, phase: int, prng: Optional[VoicePRNG]) -> int:
        """
        Simple texture placeholder.

        In full implementation, this would use the wavelet packet tree.
        """
        config = get_archetype_config(self.archetype_a)

        # Base: slightly detuned copy of fundamental
        base = self.scaffold._fast_sine((phase * 1001) >> 10)  # ~0.2% detune

        # Add noise if configured (NOISY archetype)
        if config.noise_inject_q15 > 0:
            if prng:
                noise = prng.draw_q15() >> 2
            else:
                # Deterministic noise from phase for reproducibility
                noise = ((phase * 1103515245 + 12345) >> 16) & 0x7FFF
                noise = noise - 16384  # Center around 0
            noise_scaled = (noise * config.noise_inject_q15) >> 15
            base = ((base * (Q15_ONE - config.noise_inject_q15)) >> 15) + noise_scaled

        # Add higher harmonic content for TRANSIENT character
        if config.transient_q15 > 0:
            # Add some higher frequency content
            high_content = self.scaffold._fast_sine((phase * 3) & PHASE_MASK)
            high_scaled = (high_content * config.transient_q15) >> 17
            base = base + high_scaled

        return base


# =============================================================================
# OSCILLATOR MANAGER
# =============================================================================

@dataclass
class OscillatorManager:
    """
    Manages multiple WPO instances.

    Handles voice allocation, parameter updates, and block rendering.
    """
    max_voices: int = 16
    oscillators: Dict[int, WPOState] = field(default_factory=dict)

    def allocate(self, voice_id: int) -> WPOState:
        """Allocate an oscillator for a voice."""
        osc = WPOState(voice_id=voice_id)
        self.oscillators[voice_id] = osc
        return osc

    def deallocate(self, voice_id: int) -> None:
        """Deallocate an oscillator."""
        self.oscillators.pop(voice_id, None)

    def get(self, voice_id: int) -> Optional[WPOState]:
        """Get oscillator for a voice."""
        return self.oscillators.get(voice_id)

    def configure_voice(
        self,
        voice_id: int,
        note: int,
        cents: int = 0,
        archetype: TreeArchetype = TreeArchetype.HARMONIC,
        eta_q15: int = Q15_HALF,
        waveform: str = "saw",
    ) -> None:
        """Configure an oscillator."""
        osc = self.get(voice_id)
        if osc:
            osc.configure(note, cents, archetype, eta_q15, waveform)

    def render_all(
        self,
        block_size: int = BLOCK_SIZE,
        prngs: Optional[Dict[int, VoicePRNG]] = None,
    ) -> Dict[int, List[int]]:
        """
        Render all active oscillators.

        Returns dict of voice_id -> samples.
        """
        outputs = {}
        for voice_id, osc in self.oscillators.items():
            prng = prngs.get(voice_id) if prngs else None
            outputs[voice_id] = osc.render_block(block_size, prng)
        return outputs


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'PHASE_BITS',
    'PHASE_FRAC_BITS',
    'PHASE_ONE',
    'BLOCK_SIZE',
    'SAMPLE_RATE',
    'Q15_ONE',
    'Q15_HALF',
    'BLEP_KERNEL_SIZE',
    # Tree archetypes
    'TreeArchetype',
    'TreeArchetypeConfig',
    'TREE_ARCHETYPES',
    'get_archetype_config',
    # Transitions
    'TransitionKind',
    'BRIDGE_BLOCKS',
    'BRIDGE_TRACTION_HI',
    'BRIDGE_TRACTION_NOM',
    'GLIDE_THRESHOLD_CENTS',
    # BLEP
    'BLEP_KERNEL',
    'BLEP_TABLE',
    'BLEPState',
    'generate_blep_kernel',
    # Phase
    'PhaseAccumulator',
    'frequency_to_phase_inc',
    'midi_to_frequency',
    'midi_to_phase_inc',
    # Two-stream
    'HarmonicScaffold',
    'PacketTexture',
    'blend_two_stream',
    # Drift
    'AnalogDrift',
    # Oscillator
    'WPOState',
    'OscillatorManager',
]
