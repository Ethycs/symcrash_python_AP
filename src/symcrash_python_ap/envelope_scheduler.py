"""
Envelope Scheduler
==================

Block-rate ADSR envelope generator for the preset compiler.

Implements envelope scheduling at 750 blocks/sec (48kHz / 64 samples).
All output values are Q15 (0-32767).

Per WSE-STD-PRESET v0, envelope updates happen at block boundaries only.
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


def ms_to_blocks(ms: float) -> int:
    """Convert milliseconds to block count."""
    blocks = int((ms / 1000.0) * BLOCKS_PER_SECOND)
    return max(1, blocks)  # At least 1 block


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
]
