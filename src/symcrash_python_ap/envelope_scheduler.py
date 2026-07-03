"""
Envelope Scheduler
==================

Block-rate ADSR envelope generator for the preset compiler.

Implements envelope scheduling at 750 blocks/sec (48kHz / 64 samples).
All output values are Q15 (0-32767).

Per WSE-STD-PRESET v0, envelope updates happen at block boundaries only.

NOTE: This module provides a legacy EnvelopeScheduler implementation.
For new code, prefer using envelope_suite.py which provides a unified
suite of envelope generators and processors:

    from symcrash_python_ap.envelope_suite import (
        ADSR,              # Standard ADSR envelope (replaces EnvelopeScheduler)
        AR,                # Attack-Release (percussive)
        Decay,             # One-shot exponential decay
        EnvelopeFollower,  # Tracks input amplitude
        Slew,              # Smoothing filter for control signals
        LFO,               # Low-frequency oscillator
    )

See WSE-STD-ENVELOPE-SUITE-v0 for the full specification.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from .primitives import BLOCK_SIZE, SAMPLE_RATE_HZ, Q15_ONE, BLOCK_RATE_HZ


# =============================================================================
# CONSTANTS
# =============================================================================

SAMPLE_RATE = SAMPLE_RATE_HZ
BLOCKS_PER_SECOND = BLOCK_RATE_HZ  # 750 blocks/sec


# =============================================================================
# ADSR SPEC
# =============================================================================

@dataclass
class ADSRSpec:
    """ADSR envelope specification in milliseconds."""
    attack_ms: float = 10.0
    decay_ms: float = 100.0
    sustain: float = 0.5        # 0.0 to 1.0
    release_ms: float = 100.0

    @property
    def sustain_q15(self) -> int:
        """Sustain level as Q15."""
        return int(self.sustain * Q15_ONE)


# Canonical conversion (single source in primitives; kept in __all__)
from .primitives import ms_to_blocks


# =============================================================================
# ENVELOPE PHASE
# =============================================================================

class EnvelopePhase(IntEnum):
    """ADSR envelope phase."""
    IDLE = 0        # Not playing
    ATTACK = 1      # Rising from 0 to peak
    DECAY = 2       # Falling from peak to sustain
    SUSTAIN = 3     # Holding at sustain level
    RELEASE = 4     # Falling from current to 0


# =============================================================================
# ENVELOPE SCHEDULER
# =============================================================================

@dataclass
class EnvelopeScheduler:
    """
    Block-rate ADSR envelope generator.

    Usage:
        env = EnvelopeScheduler(ADSRSpec(attack_ms=5, decay_ms=100))
        env.trigger_note()

        for block in range(total_blocks):
            gain = env.tick()
            if block == note_off_block:
                env.release_note()
    """
    spec: ADSRSpec

    # Internal state
    phase: EnvelopePhase = EnvelopePhase.IDLE
    phase_block: int = 0                # Blocks in current phase
    current_q15: int = 0                # Current envelope value
    release_level_q15: int = 0          # Level when release started

    # Cached block counts
    _attack_blocks: int = 0
    _decay_blocks: int = 0
    _release_blocks: int = 0

    def __post_init__(self):
        """Cache block counts from spec."""
        self._attack_blocks = ms_to_blocks(self.spec.attack_ms)
        self._decay_blocks = ms_to_blocks(self.spec.decay_ms)
        self._release_blocks = ms_to_blocks(self.spec.release_ms)

    def trigger_note(self) -> None:
        """
        Start envelope at attack phase.

        Called on note-on event.
        """
        self.phase = EnvelopePhase.ATTACK
        self.phase_block = 0
        # Don't reset current_q15 - allows retriggering from current level

    def release_note(self) -> None:
        """
        Transition to release phase.

        Called on note-off event.
        """
        if self.phase in (EnvelopePhase.ATTACK, EnvelopePhase.DECAY, EnvelopePhase.SUSTAIN):
            self.release_level_q15 = self.current_q15
            self.phase = EnvelopePhase.RELEASE
            self.phase_block = 0

    def tick(self) -> int:
        """
        Advance envelope by one block and return gain_q15.

        Returns:
            Q15 gain value (0 to 32767)
        """
        if self.phase == EnvelopePhase.IDLE:
            self.current_q15 = 0
            return 0

        elif self.phase == EnvelopePhase.ATTACK:
            # Linear rise from 0 to Q15_ONE
            if self._attack_blocks <= 1:
                t = 1.0
            else:
                t = self.phase_block / (self._attack_blocks - 1)
            self.current_q15 = int(Q15_ONE * min(1.0, t))

            self.phase_block += 1
            if self.phase_block >= self._attack_blocks:
                self.phase = EnvelopePhase.DECAY
                self.phase_block = 0

        elif self.phase == EnvelopePhase.DECAY:
            # Linear fall from Q15_ONE to sustain
            sustain_q15 = self.spec.sustain_q15
            if self._decay_blocks <= 1:
                t = 1.0
            else:
                t = self.phase_block / (self._decay_blocks - 1)
            self.current_q15 = int(Q15_ONE - (Q15_ONE - sustain_q15) * min(1.0, t))

            self.phase_block += 1
            if self.phase_block >= self._decay_blocks:
                self.phase = EnvelopePhase.SUSTAIN
                self.phase_block = 0

        elif self.phase == EnvelopePhase.SUSTAIN:
            # Hold at sustain level
            self.current_q15 = self.spec.sustain_q15
            self.phase_block += 1

        elif self.phase == EnvelopePhase.RELEASE:
            # Linear fall from release_level to 0
            if self._release_blocks <= 1:
                t = 1.0
            else:
                t = self.phase_block / (self._release_blocks - 1)
            self.current_q15 = int(self.release_level_q15 * (1.0 - min(1.0, t)))

            self.phase_block += 1
            if self.phase_block >= self._release_blocks:
                self.phase = EnvelopePhase.IDLE
                self.current_q15 = 0
                self.phase_block = 0

        # Clamp to valid range
        return max(0, min(Q15_ONE, self.current_q15))

    @property
    def is_active(self) -> bool:
        """True if envelope is producing non-zero output."""
        return self.phase != EnvelopePhase.IDLE

    @property
    def is_releasing(self) -> bool:
        """True if in release phase."""
        return self.phase == EnvelopePhase.RELEASE

    def reset(self) -> None:
        """Reset envelope to idle state."""
        self.phase = EnvelopePhase.IDLE
        self.phase_block = 0
        self.current_q15 = 0
        self.release_level_q15 = 0


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_envelope(
    attack_ms: float = 10.0,
    decay_ms: float = 100.0,
    sustain: float = 0.5,
    release_ms: float = 100.0
) -> EnvelopeScheduler:
    """Create an envelope scheduler with the given ADSR parameters."""
    spec = ADSRSpec(
        attack_ms=attack_ms,
        decay_ms=decay_ms,
        sustain=sustain,
        release_ms=release_ms
    )
    return EnvelopeScheduler(spec=spec)


# =============================================================================
# CLOSED-FORM BLOCK-INDEXED ADSR (offline/streaming law)
# =============================================================================
# Two envelope laws coexist deliberately:
#   1. The stateful block-rate EnvelopeScheduler above (live voice_engine).
#   2. This closed-form value(block_idx) evaluation, used by the offline
#      renderer and workbench streaming, where the whole timeline is known
#      (or the gate drives sustain_until_block).
# They are NOT interchangeable; consolidating them would change ear-tuned
# output. This module is the single home for both.

# Sentinel for "gate still held": sustain lasts until further notice.
GATE_HELD = 1 << 30


def adsr_value_q15(
    attack_blocks: int,
    decay_blocks: int,
    sustain_q15: int,
    release_blocks: int,
    block_idx: int,
    sustain_until_block: int,
    peak_q15: int = Q15_ONE,
    release_from_q15: Optional[int] = None,
) -> int:
    """
    Closed-form ADSR value (Q15) at a block index.

    sustain_until_block: block where release begins (offline law:
        attack_blocks + sustain_blocks; live law: note-off block, or
        GATE_HELD while the gate is down).
    peak_q15: attack peak (velocity-scaled attack level for amp envelopes).
    release_from_q15: level release decays from; None = sustain level
        (offline law). Streaming passes the latched gate-release level.

    Float operation order matches the historical offline implementation
    bit-for-bit (t computed first, then scaled).
    """
    if block_idx < attack_blocks:
        v = (int(peak_q15 * (block_idx / attack_blocks))
             if attack_blocks > 0 else peak_q15)
    elif block_idx < attack_blocks + decay_blocks:
        t = ((block_idx - attack_blocks) / decay_blocks
             if decay_blocks > 0 else 1)
        v = int(peak_q15 - (peak_q15 - sustain_q15) * t)
    elif block_idx < sustain_until_block:
        v = sustain_q15
    else:
        start = sustain_q15 if release_from_q15 is None else release_from_q15
        t = ((block_idx - sustain_until_block) / release_blocks
             if release_blocks > 0 else 1)
        v = int(start * (1.0 - t))
    return max(0, min(Q15_ONE, v))


class BindingEnvelope:
    """
    Closed-form ADSR state for one matrix-envelope binding
    (WSE-STD-ENVELOPE-MATRIX), with DX7-style key-scale-rate and velocity
    sensitivity. Single owner; formerly duplicated as _EnvelopeState in
    tools/render/render_preset.py, symcrash_workbench/streaming.py and
    symcrash_imgui_controls/workbench_render.py.
    """

    def __init__(self, binding, note: int = 60):
        from .envelope_targets import EnvelopeTarget

        adsr = binding.adsr
        # DX7-style EG key scaling: higher notes get shorter envelopes
        ks = float(getattr(binding, 'key_scale_rate', 0.0))
        mult = 2.0 ** (-ks * (note - 60) / 24.0) if ks else 1.0
        self.attack_blocks = ms_to_blocks(adsr.a_ms * mult)
        self.decay_blocks = ms_to_blocks(adsr.d_ms * mult)
        self.sustain = adsr.s
        self.release_blocks = ms_to_blocks(adsr.r_ms * mult)
        self.target_ids = []
        for name in binding.targets:
            try:
                self.target_ids.append(int(EnvelopeTarget[name]))
            except KeyError:
                pass
        self.velocity_sens_q15 = int(
            getattr(binding, 'velocity_sens', 0.0) * Q15_ONE)

    def value_q15(self, block_idx: int, sustain_blocks: int) -> int:
        return adsr_value_q15(
            self.attack_blocks,
            self.decay_blocks,
            int(self.sustain * Q15_ONE),
            self.release_blocks,
            block_idx,
            sustain_blocks,
        )

    # Back-compat alias (imgui workbench used get_value)
    get_value = value_q15


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SAMPLE_RATE',
    'BLOCK_SIZE',
    'BLOCKS_PER_SECOND',
    'Q15_ONE',
    'ADSRSpec',
    'ms_to_blocks',
    'EnvelopePhase',
    'EnvelopeScheduler',
    'create_envelope',
    'GATE_HELD',
    'adsr_value_q15',
    'BindingEnvelope',
]
