"""
VoiceFrame Emitter
==================

Emits VoiceFrames from compiled preset + note events.

The emitter is the bridge between the preset (static configuration) and
the VM (which consumes VoiceFrames). It handles:
- Envelope scheduling (block-rate ADSR)
- Beta (dry/wet) ramping
- Note-on/note-off transitions
- MIDI note to Hz conversion

Usage:
    from preset_compiler import PresetCompiler
    from voiceframe_emitter import VoiceFrameEmitter

    compiler = PresetCompiler()
    preset = compiler.compile_file("brass_stab_v0.yaml")

    emitter = VoiceFrameEmitter(preset)
    emitter.note_on(note=60, velocity_q15=32767)

    for block in range(total_blocks):
        if block == note_off_block:
            emitter.note_off()
        frame = emitter.emit_frame()
        # Pass frame to VM
"""

from dataclasses import dataclass, field
from typing import Optional, List
import math

from .preset_compiler import CompiledPreset, Q15_ONE
from .envelope_scheduler import EnvelopeScheduler, ADSRSpec
from .frames import VoiceFrame


# =============================================================================
# MIDI UTILITIES
# =============================================================================

def midi_to_hz(note: int, cents: int = 0) -> float:
    """
    Convert MIDI note number to frequency in Hz.

    Uses A4 = 440 Hz (MIDI note 69).

    Args:
        note: MIDI note number (0-127)
        cents: Fine tuning in cents (-100 to +100 typical)

    Returns:
        Frequency in Hz
    """
    # A4 = 440 Hz = MIDI note 69
    # f = 440 * 2^((n - 69) / 12)
    semitones = note - 69 + (cents / 100.0)
    return 440.0 * (2.0 ** (semitones / 12.0))


# =============================================================================
# VOICEFRAME EMITTER
# =============================================================================

@dataclass
class VoiceFrameEmitter:
    """
    Emits VoiceFrames from compiled preset + note events.

    Manages per-voice state including:
    - Envelope (ADSR)
    - Beta ramp (dry/wet blend)
    - Note state (pitch, velocity)
    """
    preset: CompiledPreset

    # Voice state
    voice_id: int = 0
    note: int = 60
    cents: int = 0
    velocity_q15: int = Q15_ONE

    # Block counter
    block_index: int = 0

    # Envelope state
    _envelope: Optional[EnvelopeScheduler] = field(default=None, repr=False)

    # Beta ramp state
    _beta_current_q15: int = 0

    # Note state
    _is_note_on: bool = False

    def __post_init__(self):
        """Initialize envelope from preset."""
        if self.preset.amp_env:
            self._envelope = EnvelopeScheduler(spec=self.preset.amp_env)
        else:
            # Default envelope if none specified
            self._envelope = EnvelopeScheduler(spec=ADSRSpec())

        # Initialize beta to start value
        self._beta_current_q15 = self.preset.material.beta_start_q15

    def note_on(self, note: int, velocity_q15: int = Q15_ONE, cents: int = 0) -> None:
        """
        Trigger note-on event.

        Args:
            note: MIDI note number (0-127)
            velocity_q15: Velocity (0-32767)
            cents: Fine tuning in cents
        """
        self.note = note
        self.velocity_q15 = velocity_q15
        self.cents = cents
        self.block_index = 0
        self._is_note_on = True

        # Reset beta ramp
        self._beta_current_q15 = self.preset.material.beta_start_q15

        # Trigger envelope
        if self._envelope:
            self._envelope.trigger_note()

    def note_off(self) -> None:
        """Trigger note-off event (start release phase)."""
        self._is_note_on = False

        if self._envelope:
            self._envelope.release_note()

    def emit_frame(self) -> VoiceFrame:
        """
        Emit next VoiceFrame for this block.

        Call this once per block to get the VoiceFrame.

        Returns:
            VoiceFrame with all fields populated from preset + state
        """
        # Get envelope gain
        if self._envelope:
            gain_q15 = self._envelope.tick()
        else:
            gain_q15 = Q15_ONE if self._is_note_on else 0

        # Apply velocity scaling
        gain_q15 = (gain_q15 * self.velocity_q15) >> 15

        # Compute beta ramp
        beta_target = self.preset.material.beta_target_q15
        ramp_blocks = self.preset.material.beta_ramp_blocks

        if self.block_index < ramp_blocks and ramp_blocks > 1:
            # Linear ramp from start to target
            beta_start = self.preset.material.beta_start_q15
            t = self.block_index / (ramp_blocks - 1)
            self._beta_current_q15 = int(beta_start + (beta_target - beta_start) * t)
        else:
            self._beta_current_q15 = beta_target

        # Compute f0_hz from MIDI note
        f0_hz = midi_to_hz(self.note, self.cents)

        # Build VoiceFrame
        frame = VoiceFrame(
            voice_id=self.voice_id,
            f0_hz=f0_hz,
            note=self.note,
            cents=self.cents,
            gain_q15=gain_q15,
            beta_target_q15=self._beta_current_q15,
            partial_amps_q15=list(self.preset.harmonic_amps_q15),
            ops_enable=self.preset.backbone_ops(),
            families=self.preset.backbone_families(),
            tree_a_id=self.preset.latch.tree_a_id,
            tree_b_id=self.preset.latch.tree_b_id,
            tree_blend_q15=self.preset.latch.tree_blend_q15,
            is_starting=(self.block_index == 0),
            is_active=self.is_active,
        )

        # Advance block counter
        self.block_index += 1

        return frame

    @property
    def is_active(self) -> bool:
        """True if voice should continue producing output."""
        if self._envelope:
            return self._envelope.is_active
        return self._is_note_on

    @property
    def is_releasing(self) -> bool:
        """True if in release phase."""
        if self._envelope:
            return self._envelope.is_releasing
        return not self._is_note_on

    def reset(self) -> None:
        """Reset emitter to initial state."""
        self.block_index = 0
        self._is_note_on = False
        self._beta_current_q15 = self.preset.material.beta_start_q15
        if self._envelope:
            self._envelope.reset()


# =============================================================================
# MULTI-VOICE EMITTER
# =============================================================================

@dataclass
class MultiVoiceEmitter:
    """
    Manages multiple VoiceFrameEmitters for polyphonic playback.

    Usage:
        multi = MultiVoiceEmitter(preset, max_voices=8)
        multi.note_on(note=60, velocity_q15=32767)
        multi.note_on(note=64, velocity_q15=32767)  # Different voice

        for block in range(total_blocks):
            frames = multi.emit_frames()
            for frame in frames:
                # Process with VM
    """
    preset: CompiledPreset
    max_voices: int = 8

    # Voice pool
    _voices: List[VoiceFrameEmitter] = field(default_factory=list, repr=False)
    _next_voice_id: int = 0

    def __post_init__(self):
        """Initialize voice pool."""
        self._voices = []
        for i in range(self.max_voices):
            emitter = VoiceFrameEmitter(preset=self.preset, voice_id=i)
            self._voices.append(emitter)

    def note_on(self, note: int, velocity_q15: int = Q15_ONE, cents: int = 0) -> int:
        """
        Trigger note-on, allocating a voice.

        Returns:
            Voice ID that was allocated, or -1 if no voice available
        """
        # Find free voice (prefer inactive, then oldest releasing)
        voice = self._find_free_voice()
        if voice is None:
            return -1

        voice.note_on(note, velocity_q15, cents)
        return voice.voice_id

    def note_off(self, note: int) -> None:
        """
        Trigger note-off for all voices playing this note.
        """
        for voice in self._voices:
            if voice._is_note_on and voice.note == note:
                voice.note_off()

    def emit_frames(self) -> List[VoiceFrame]:
        """
        Emit VoiceFrames for all active voices.

        Returns:
            List of VoiceFrames (one per active voice)
        """
        frames = []
        for voice in self._voices:
            if voice.is_active:
                frames.append(voice.emit_frame())
        return frames

    def _find_free_voice(self) -> Optional[VoiceFrameEmitter]:
        """Find a free voice for allocation."""
        # First pass: find inactive voice
        for voice in self._voices:
            if not voice.is_active:
                return voice

        # Second pass: find releasing voice (voice stealing)
        for voice in self._voices:
            if voice.is_releasing:
                return voice

        # No voice available
        return None

    @property
    def active_voice_count(self) -> int:
        """Number of currently active voices."""
        return sum(1 for v in self._voices if v.is_active)

    def reset(self) -> None:
        """Reset all voices."""
        for voice in self._voices:
            voice.reset()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'midi_to_hz',
    'VoiceFrameEmitter',
    'MultiVoiceEmitter',
]
