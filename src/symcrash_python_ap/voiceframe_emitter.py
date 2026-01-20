"""
VoiceFrame Emitter
==================

DEPRECATED: Use VoiceEngine directly with Simple mode.

This module provides a compatibility wrapper around VoiceEngine for code
that still uses the VoiceFrameEmitter API. New code should use VoiceEngine
directly:

    # NEW WAY (recommended):
    from symcrash_python_ap.voice import VoiceEngine, EngineMode

    engine = VoiceEngine()
    voice_id = engine.allocate_voice_from_preset(preset)
    engine.note_on_simple(voice_id, note=60, velocity_q15=32767)
    frame = engine.step_voice_simple(voice_id)

    # OLD WAY (deprecated):
    emitter = VoiceFrameEmitter(preset)
    emitter.note_on(note=60, velocity_q15=32767)
    frame = emitter.emit_frame()

The wrapper maintains full API compatibility while delegating all work
to VoiceEngine's Simple mode implementation.
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

from .library.preset_compiler import CompiledPreset, Q15_ONE
from .frames import VoiceFrame
from .control_frame import ControlFrame, EventType
from .control_registry import ControlID


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
# VOICEFRAME EMITTER (DEPRECATED - WRAPPER AROUND VOICEENGINE)
# =============================================================================

@dataclass
class VoiceFrameEmitter:
    """
    DEPRECATED: Use VoiceEngine directly with Simple mode.

    This class is a compatibility wrapper that creates a VoiceEngine
    in Simple mode. All state and computation is delegated to VoiceEngine.

    The wrapper will be removed in a future version. Migrate to:

        engine = VoiceEngine()
        voice_id = engine.allocate_voice_from_preset(preset)
        engine.note_on_simple(voice_id, note, velocity_q15, cents)
        frame = engine.step_voice_simple(voice_id)
    """
    preset: CompiledPreset

    # Voice ID for this emitter
    voice_id: int = 0

    # Current note state (for consume_control_frame compatibility)
    note: int = 60
    cents: int = 0
    velocity_q15: int = Q15_ONE

    # Block counter (exposed for backward compatibility)
    block_index: int = 0

    # Control modifiers from ControlFrame (for backward compatibility)
    _pitch_offset_cents: int = 0
    _gain_scale_q15: int = Q15_ONE
    _tree_blend_q15: int = 0
    _family_scale_q15: List[int] = field(default_factory=lambda: [Q15_ONE] * 8)
    _master_gain_q15: int = Q15_ONE

    # ADSR overrides from ControlFrame
    _attack_blocks: Optional[int] = None
    _decay_blocks: Optional[int] = None
    _sustain_level_q15: Optional[int] = None
    _release_blocks: Optional[int] = None

    # Manual overrides (for set_eta, set_detune)
    _eta_override: Optional[int] = None
    _detune_override: Optional[int] = None

    # Internal: wrapped VoiceEngine in Simple mode
    _engine: 'VoiceEngine' = field(init=False, repr=False)
    _voice_id_internal: int = field(init=False, repr=False)
    _is_note_on: bool = field(default=False, init=False)

    def __post_init__(self):
        """Initialize VoiceEngine wrapper."""
        # Lazy import to avoid circular dependencies
        from .voice import VoiceEngine

        # Create VoiceEngine (Simple mode is set per-voice, not engine-wide)
        self._engine = VoiceEngine(max_voices=1)

        # Allocate voice from preset (this sets Simple mode on the voice)
        self._voice_id_internal = self._engine.allocate_voice_from_preset(
            self.preset,
            voice_id=0
        )

        if self._voice_id_internal is None:
            raise RuntimeError("Failed to allocate voice from preset")

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

        # Delegate to VoiceEngine
        self._engine.note_on_simple(
            self._voice_id_internal,
            note=note,
            velocity_q15=velocity_q15,
            cents=cents
        )

    def note_off(self) -> None:
        """Trigger note-off event (start release phase)."""
        self._is_note_on = False

        # Delegate to VoiceEngine
        self._engine.note_off_simple(self._voice_id_internal)

    def consume_control_frame(self, frame: ControlFrame) -> None:
        """
        Consume a ControlFrame and update emitter state.

        Note: In Simple mode, control frame consumption is handled locally
        since VoiceEngine doesn't expose gesture updates for Simple mode.
        The control modifiers are applied during emit_frame().

        Args:
            frame: ControlFrame from CockpitEngine
        """
        # Process events
        for event in frame.events:
            if event.type == EventType.NOTE_ON and (event.voice_id == 0xFFFF or event.voice_id == self.voice_id):
                note = event.a & 0x7F
                velocity_q15 = (event.a >> 8) & 0xFFFF
                self.note_on(note, velocity_q15)
            elif event.type == EventType.NOTE_OFF and (event.voice_id == 0xFFFF or event.voice_id == self.voice_id):
                note = event.a & 0x7F
                if note == self.note:
                    self.note_off()

        # Process voice patches
        for patch in frame.voice_patches:
            if patch.voice_id != 0 and patch.voice_id != self.voice_id:
                continue  # Not for this voice

            for item in patch.items:
                # Apply control modifiers
                if item.control_id == ControlID.DYNAMICS:
                    self._gain_scale_q15 = item.value
                elif item.control_id == ControlID.MORPH:
                    self._tree_blend_q15 = item.value
                elif item.control_id == ControlID.DETUNE:
                    self._pitch_offset_cents = item.value
                elif item.control_id == ControlID.PITCH_BEND:
                    self._pitch_offset_cents += item.value  # Cumulative
                elif item.control_id == ControlID.TIMBRE:
                    self._family_scale_q15[0] = item.value  # FamilyID.COLOR
                elif item.control_id == ControlID.TEXTURE:
                    self._family_scale_q15[1] = item.value  # FamilyID.LOCAL
                elif item.control_id == ControlID.CHAOS:
                    self._family_scale_q15[5] = item.value  # FamilyID.CATASTROPHE
                # ADSR overrides
                elif item.control_id == ControlID.ATK_BLOCKS:
                    self._attack_blocks = item.value
                elif item.control_id == ControlID.DCY_BLOCKS:
                    self._decay_blocks = item.value
                elif item.control_id == ControlID.SUS_Q15:
                    self._sustain_level_q15 = item.value
                elif item.control_id == ControlID.REL_BLOCKS:
                    self._release_blocks = item.value

        # Update global controls
        if frame.globals.master_gain_q15 is not None:
            self._master_gain_q15 = frame.globals.master_gain_q15

        # Update VoiceState with pitch offset
        state = self._engine.voices.get(self._voice_id_internal)
        if state:
            state.pitch_offset_cents = self._pitch_offset_cents

    def emit_frame(self) -> VoiceFrame:
        """
        Emit next VoiceFrame for this block.

        Call this once per block to get the VoiceFrame.
        Delegates to VoiceEngine.step_voice_simple() and applies
        control frame modifiers.

        Returns:
            VoiceFrame with all fields populated from preset + control modifiers
        """
        # Get frame from VoiceEngine
        frame = self._engine.step_voice_simple(self._voice_id_internal)

        if frame is None:
            # Voice is inactive, return a minimal frame
            frame = VoiceFrame(
                voice_id=self.voice_id,
                f0_hz=midi_to_hz(self.note, self.cents),
                note=self.note,
                cents=self.cents,
                gain_q15=0,
                is_active=False,
            )
        else:
            # Apply control frame modifiers that VoiceEngine doesn't handle
            # Apply DYNAMICS control (gain_scale_q15)
            frame.gain_q15 = (frame.gain_q15 * self._gain_scale_q15) >> 15

            # Apply master gain
            frame.gain_q15 = (frame.gain_q15 * self._master_gain_q15) >> 15

            # Apply MORPH control if set
            if self._tree_blend_q15 != 0:
                frame.tree_blend_q15 = self._tree_blend_q15

            # Apply eta override if set
            if self._eta_override is not None:
                frame.beta_target_q15 = self._eta_override

            # Update voice_id to match emitter's ID
            frame.voice_id = self.voice_id

        # Advance block counter
        self.block_index += 1

        return frame

    @property
    def is_active(self) -> bool:
        """True if voice should continue producing output."""
        from .voice.voice_engine import VoicePhase

        state = self._engine.voices.get(self._voice_id_internal)
        if state:
            return state.phase != VoicePhase.FREE
        return False

    @property
    def is_releasing(self) -> bool:
        """True if in release phase."""
        from .voice.voice_engine import VoicePhase

        state = self._engine.voices.get(self._voice_id_internal)
        if state:
            return state.phase == VoicePhase.RELEASE
        return False

    def reset(self) -> None:
        """Reset emitter to initial state."""
        self.block_index = 0
        self._is_note_on = False
        self._pitch_offset_cents = 0
        self._gain_scale_q15 = Q15_ONE
        self._tree_blend_q15 = 0
        self._family_scale_q15 = [Q15_ONE] * 8
        self._master_gain_q15 = Q15_ONE
        self._attack_blocks = None
        self._decay_blocks = None
        self._sustain_level_q15 = None
        self._release_blocks = None
        self._eta_override = None
        self._detune_override = None

        # Re-allocate voice from preset to reset VoiceEngine state
        self._voice_id_internal = self._engine.allocate_voice_from_preset(
            self.preset,
            voice_id=0
        )

    def set_eta(self, eta_q15: Optional[int]) -> None:
        """
        Override the beta (eta) value for dry/wet mixing.

        Args:
            eta_q15: Beta value in Q15 format (0-32767), or None to use preset default
        """
        self._eta_override = eta_q15
        # Update VoiceState
        state = self._engine.voices.get(self._voice_id_internal)
        if state and eta_q15 is not None:
            state.beta_current_q15 = eta_q15
            state.beta_target_q15 = eta_q15

    def set_detune(self, detune_cents: Optional[int]) -> None:
        """
        Override the detune in cents.

        Args:
            detune_cents: Detune in cents, or None to use preset default
        """
        self._detune_override = detune_cents
        self._pitch_offset_cents = detune_cents if detune_cents is not None else 0
        # Update VoiceState
        state = self._engine.voices.get(self._voice_id_internal)
        if state:
            state.pitch_offset_cents = self._pitch_offset_cents

    def get_eta(self) -> Optional[int]:
        """
        Get the current eta (beta) override value.

        Returns:
            Current eta override in Q15 format, or None if using preset default
        """
        return self._eta_override

    def get_detune(self) -> Optional[int]:
        """
        Get the current detune override value.

        Returns:
            Current detune override in cents, or None if using preset default
        """
        return self._detune_override

    # Legacy property for backward compatibility
    @property
    def arc_phase(self) -> List[int]:
        """
        Get arc_phase_q16 storage for LFO phases.

        DEPRECATED: LFO phase is now managed by VoiceEngine.
        """
        state = self._engine.voices.get(self._voice_id_internal)
        if state:
            return state.arc_phase_q16
        return [0] * 8

    # Legacy property for backward compatibility
    @property
    def voice_state(self):
        """
        Get the underlying VoiceState.

        DEPRECATED: Use VoiceEngine directly.
        """
        return self._engine.voices.get(self._voice_id_internal)


# =============================================================================
# MULTI-VOICE EMITTER (DEPRECATED)
# =============================================================================

@dataclass
class MultiVoiceEmitter:
    """
    DEPRECATED: Use VoiceEngine directly with multiple voices.

    Manages multiple VoiceFrameEmitters for polyphonic playback.
    This is a compatibility wrapper; new code should use VoiceEngine directly.
    """
    preset: CompiledPreset
    max_voices: int = 8

    # Voice pool (now uses VoiceEngine internally)
    _engine: 'VoiceEngine' = field(init=False, repr=False)
    _voice_ids: List[int] = field(default_factory=list, repr=False)
    _note_to_voice: Dict[int, int] = field(default_factory=dict, repr=False)

    # Control modifiers (shared across all voices)
    _master_gain_q15: int = Q15_ONE

    def __post_init__(self):
        """Initialize voice pool using VoiceEngine."""
        from .voice import VoiceEngine

        # Create VoiceEngine with capacity for all voices
        self._engine = VoiceEngine(max_voices=self.max_voices)

        # Allocate all voices from preset
        self._voice_ids = []
        for i in range(self.max_voices):
            voice_id = self._engine.allocate_voice_from_preset(self.preset, voice_id=i)
            if voice_id is not None:
                self._voice_ids.append(voice_id)

    def consume_control_frame(self, frame: ControlFrame) -> None:
        """
        Consume a ControlFrame and update all voices.

        Args:
            frame: ControlFrame from CockpitEngine
        """
        # Process note-on/note-off events
        for event in frame.events:
            if event.type == EventType.NOTE_ON:
                note = event.a & 0x7F
                velocity_q15 = (event.a >> 8) & 0xFFFF
                self.note_on(note, velocity_q15)
            elif event.type == EventType.NOTE_OFF:
                note = event.a & 0x7F
                self.note_off(note)

        # Update global controls
        if frame.globals.master_gain_q15 is not None:
            self._master_gain_q15 = frame.globals.master_gain_q15

    def note_on(self, note: int, velocity_q15: int = Q15_ONE, cents: int = 0) -> int:
        """
        Trigger note-on, allocating a voice.

        Returns:
            Voice ID that was allocated, or -1 if no voice available
        """
        from .voice.voice_engine import VoicePhase

        # Find free voice (prefer inactive, then oldest releasing)
        voice_id = self._find_free_voice()
        if voice_id is None:
            return -1

        # Note-on on the voice
        self._engine.note_on_simple(voice_id, note, velocity_q15, cents)

        # Track note-to-voice mapping
        self._note_to_voice[note] = voice_id

        return voice_id

    def note_off(self, note: int) -> None:
        """
        Trigger note-off for voices playing this note.
        """
        if note in self._note_to_voice:
            voice_id = self._note_to_voice[note]
            self._engine.note_off_simple(voice_id)
            del self._note_to_voice[note]

    def emit_frames(self) -> List[VoiceFrame]:
        """
        Emit VoiceFrames for all active voices.

        Returns:
            List of VoiceFrames (one per active voice)
        """
        from .voice.voice_engine import VoicePhase

        frames = []
        for voice_id in self._voice_ids:
            state = self._engine.voices.get(voice_id)
            if state and state.phase != VoicePhase.FREE:
                frame = self._engine.step_voice_simple(voice_id)
                if frame:
                    # Apply master gain
                    frame.gain_q15 = (frame.gain_q15 * self._master_gain_q15) >> 15
                    frames.append(frame)
        return frames

    def _find_free_voice(self) -> Optional[int]:
        """Find a free voice for allocation."""
        from .voice.voice_engine import VoicePhase

        # First pass: find inactive voice
        for voice_id in self._voice_ids:
            state = self._engine.voices.get(voice_id)
            if state and state.phase == VoicePhase.FREE:
                return voice_id

        # Second pass: find releasing voice (voice stealing)
        for voice_id in self._voice_ids:
            state = self._engine.voices.get(voice_id)
            if state and state.phase == VoicePhase.RELEASE:
                return voice_id

        # No voice available
        return None

    @property
    def active_voice_count(self) -> int:
        """Number of currently active voices."""
        from .voice.voice_engine import VoicePhase

        count = 0
        for voice_id in self._voice_ids:
            state = self._engine.voices.get(voice_id)
            if state and state.phase != VoicePhase.FREE:
                count += 1
        return count

    def reset(self) -> None:
        """Reset all voices."""
        self._note_to_voice.clear()
        # Re-allocate all voices
        for i, voice_id in enumerate(self._voice_ids):
            self._engine.allocate_voice_from_preset(self.preset, voice_id=voice_id)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'midi_to_hz',
    'VoiceFrameEmitter',
    'MultiVoiceEmitter',
]
