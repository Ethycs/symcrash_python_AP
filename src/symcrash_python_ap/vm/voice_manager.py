"""
Voice Manager - 32-Voice Polyphony Support
==========================================

Implements voice allocation, stealing, and management for CCS.

Per CCS Conformance Plan:
- 32 simultaneous voices (polyphony)
- Each voice has its own CCSVoiceState
- Voice allocation: find free slot or steal oldest
- Sum voice outputs before final output
- Shared basis frames (per-note cache, not per-voice)

Voice stealing: Oldest note first (simplest, predictable)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from .state import VMState, EnvelopePhase
from ..primitives import BLOCK_SIZE
from ..primitives import Q15_NEG_ONE, Q15_ONE


# =============================================================================
# CONSTANTS
# =============================================================================

MAX_VOICES = 32          # CCS polyphony target


# =============================================================================
# VOICE SLOT
# =============================================================================

@dataclass
class VoiceSlot:
    """
    A single voice slot in the pool.

    Tracks allocation time for oldest-first voice stealing.
    """
    voice_id: int
    state: VMState
    allocation_time: int = 0      # Block number when allocated (for stealing)
    note: int = -1                # Current MIDI note (-1 = free)
    channel: int = 0              # MIDI channel (for mono mode)
    is_active: bool = False       # True if voice is producing sound

    def is_free(self) -> bool:
        """Check if voice slot is available for allocation."""
        return not self.is_active


# =============================================================================
# VOICE POOL
# =============================================================================

class VoicePool:
    """
    32-voice polyphony manager with oldest-first voice stealing.

    Manages VMState instances for each voice slot and provides
    allocation, release, and rendering coordination.
    """

    def __init__(self, num_voices: int = MAX_VOICES):
        """
        Initialize voice pool.

        Args:
            num_voices: Number of voice slots (default: 32)
        """
        self.num_voices = num_voices
        self.block_counter = 0

        # Create voice slots
        self.slots: List[VoiceSlot] = []
        for i in range(num_voices):
            state = VMState.create(voice_id=i)
            slot = VoiceSlot(
                voice_id=i,
                state=state,
                allocation_time=0,
                note=-1,
                is_active=False,
            )
            self.slots.append(slot)

        # Output buffer for mixed audio
        self._mix_buffer = np.zeros(BLOCK_SIZE, dtype=np.int32)

    def allocate_voice(self, note: int, channel: int = 0) -> Optional[VoiceSlot]:
        """
        Allocate a voice slot for a new note.

        Uses oldest-first voice stealing when no free slots available.

        Args:
            note: MIDI note number
            channel: MIDI channel (for mono mode)

        Returns:
            VoiceSlot for the allocated voice, or None if allocation failed
        """
        self.block_counter += 1

        # First pass: find a free slot
        for slot in self.slots:
            if slot.is_free():
                self._activate_slot(slot, note, channel)
                return slot

        # Second pass: steal oldest voice
        oldest_slot = None
        oldest_time = float('inf')

        for slot in self.slots:
            # Prefer to steal released voices first
            if slot.state.envelope.phase == EnvelopePhase.RELEASE:
                if slot.allocation_time < oldest_time:
                    oldest_time = slot.allocation_time
                    oldest_slot = slot

        # If no released voices, steal oldest active voice
        if oldest_slot is None:
            for slot in self.slots:
                if slot.allocation_time < oldest_time:
                    oldest_time = slot.allocation_time
                    oldest_slot = slot

        if oldest_slot is not None:
            self._activate_slot(oldest_slot, note, channel)
            return oldest_slot

        return None

    def _activate_slot(self, slot: VoiceSlot, note: int, channel: int) -> None:
        """Activate a voice slot with a new note."""
        slot.allocation_time = self.block_counter
        slot.note = note
        slot.channel = channel
        slot.is_active = True
        # Note: Caller should reset_for_note_on() on the state

    def release_note(self, note: int, channel: int = 0) -> List[VoiceSlot]:
        """
        Release all voices playing a specific note.

        Args:
            note: MIDI note number
            channel: MIDI channel

        Returns:
            List of released VoiceSlots
        """
        released = []
        for slot in self.slots:
            if slot.is_active and slot.note == note and slot.channel == channel:
                slot.state.trigger_release()
                released.append(slot)
        return released

    def release_all(self) -> None:
        """Release all active voices."""
        for slot in self.slots:
            if slot.is_active:
                slot.state.trigger_release()

    def kill_all(self) -> None:
        """Immediately stop all voices (hard reset)."""
        for slot in self.slots:
            slot.is_active = False
            slot.note = -1
            slot.state.envelope.phase = EnvelopePhase.FREE
            slot.state.coeff_state.fill(0)

    def get_active_voices(self) -> List[VoiceSlot]:
        """Get list of all active voice slots."""
        return [slot for slot in self.slots if slot.is_active]

    def get_free_count(self) -> int:
        """Get number of free voice slots."""
        return sum(1 for slot in self.slots if slot.is_free())

    def get_active_count(self) -> int:
        """Get number of active voice slots."""
        return sum(1 for slot in self.slots if slot.is_active)

    def update_slot_states(self) -> None:
        """
        Update slot states after rendering.

        Marks voices as free when they finish their release phase.
        """
        for slot in self.slots:
            if slot.is_active:
                if slot.state.envelope.phase == EnvelopePhase.FREE:
                    slot.is_active = False
                    slot.note = -1

    def mix_outputs(self, voice_outputs: List[Tuple[int, np.ndarray]]) -> np.ndarray:
        """
        Mix multiple voice outputs into a single buffer.

        Args:
            voice_outputs: List of (voice_id, audio_buffer) tuples

        Returns:
            Mixed audio buffer (int16)
        """
        self._mix_buffer.fill(0)

        for voice_id, audio in voice_outputs:
            self._mix_buffer += audio.astype(np.int32)

        # Soft clip to prevent harsh clipping
        clipped = np.clip(self._mix_buffer, Q15_NEG_ONE, Q15_ONE)
        return clipped.astype(np.int16)

    def find_voice_for_note(self, note: int, channel: int = 0) -> Optional[VoiceSlot]:
        """
        Find an active voice playing a specific note.

        Args:
            note: MIDI note number
            channel: MIDI channel

        Returns:
            VoiceSlot if found, None otherwise
        """
        for slot in self.slots:
            if slot.is_active and slot.note == note and slot.channel == channel:
                return slot
        return None


# =============================================================================
# MONO MODE SUPPORT
# =============================================================================

class MonoVoicePool(VoicePool):
    """
    Voice pool with mono mode support (single voice per channel).

    Useful for lead synth sounds where only one note should play at a time.
    """

    def __init__(self, num_channels: int = 16, legato: bool = True):
        """
        Initialize mono voice pool.

        Args:
            num_channels: Number of MIDI channels (default: 16)
            legato: Enable legato mode (preserve phase on pitch change)
        """
        super().__init__(num_voices=num_channels)
        self.legato = legato

    def allocate_voice(self, note: int, channel: int = 0) -> Optional[VoiceSlot]:
        """
        Allocate voice in mono mode.

        If channel already has an active voice, retrigger it with new note.

        Args:
            note: MIDI note number
            channel: MIDI channel

        Returns:
            VoiceSlot for the allocated voice
        """
        # Find existing voice for this channel
        slot_idx = channel % self.num_voices
        slot = self.slots[slot_idx]

        if slot.is_active and self.legato:
            # Legato: just change the pitch
            slot.note = note
            # Caller should update pitch without resetting envelope
        else:
            # Retrigger: reset the voice
            self._activate_slot(slot, note, channel)

        return slot


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'MAX_VOICES',
    'BLOCK_SIZE',
    'VoiceSlot',
    'VoicePool',
    'MonoVoicePool',
]
