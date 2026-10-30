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

from dataclasses import dataclass, field, replace
from typing import Optional, List

from .library.preset_compiler import CompiledPreset, Q15_ONE
from .frames import VoiceFrame
from .control_frame import ControlFrame, EventType


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
        frame = ControlFrame()
        frame.add_note_on(note, velocity_q15)
        frame.events[-1].voice_id = self._voice_id_internal
        self._engine.consume_control_frame(frame)
        self.voice_state.cents_i16 = cents

    def note_off(self) -> None:
        """Trigger note-off event (start release phase)."""
        self._is_note_on = False

        # Delegate to VoiceEngine
        self._engine.note_off_simple(self._voice_id_internal)

    def consume_control_frame(self, frame: ControlFrame) -> None:
        """Route this emitter's external voice ID to its internal slot."""
        events = [replace(event, voice_id=0) if event.type in (EventType.NOTE_ON, EventType.NOTE_OFF)
                  else replace(event) for event in frame.events
                  if event.voice_id in (0xFFFF, self.voice_id)]
        patches = [replace(patch, voice_id=0) for patch in frame.voice_patches
                   if patch.voice_id == self.voice_id]
        self._engine.consume_control_frame(replace(frame, events=events, voice_patches=patches))
        frame.events[:] = [event for event in frame.events
                           if event.voice_id not in (0xFFFF, self.voice_id)]
        state = self.voice_state
        self.note, self.cents, self.velocity_q15 = state.note_u8, state.cents_i16, state.velocity_q15

    def emit_frame(self, *, apply_master_gain: bool = True) -> VoiceFrame:
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
            # This compatibility API has no ParamFrame output, so apply the
            # latched master gain here. The canonical block API applies it
            # once downstream through ParamFrame.gate_q15.
            if apply_master_gain:
                frame.gain_q15 = (frame.gain_q15 * self._engine.master_gain_q15) >> 15

            # Update voice_id to match emitter's ID
            frame.voice_id = self.voice_id

        # Advance block counter
        self.block_index += 1

        return frame

    def emit_output_block(self):
        """Emit one voice block and globals for a renderer with post-FX gain.

        The legacy emit_frame() API keeps its gain-folding behavior. New live
        output applies ParamFrame.gate_q15 once after all voices/effect returns.
        """
        return self.emit_frame(apply_master_gain=False), self._engine._extract_param_frame(None)

    @property
    def output_preset(self):
        """Selected global output defaults, including cockpit preset changes."""
        return self._engine.selected_preset or self.preset

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
        """Reset voice, globals and all manual/control overrides."""
        self.block_index = 0
        self._is_note_on = False
        self.__post_init__()

    def set_eta(self, eta_q15: Optional[int]) -> None:
        """Override dry/wet mix, or restore the running preset ramp with None."""
        self.voice_state.controls.eta_override = (
            None if eta_q15 is None else max(0, min(Q15_ONE, int(eta_q15))))

    def set_detune(self, detune_cents: Optional[int]) -> None:
        """Override detune independently of pitch bend; None restores controls."""
        self.voice_state.controls.detune_override = (
            None if detune_cents is None else max(-2400, min(2400, int(detune_cents))))

    def get_eta(self) -> Optional[int]:
        return self.voice_state.controls.eta_override

    def get_detune(self) -> Optional[int]:
        return self.voice_state.controls.detune_override

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
        """Apply sparse per-voice patches and channel-aware note events."""
        self._engine.consume_control_frame(frame)

    def note_on(self, note: int, velocity_q15: int = Q15_ONE, cents: int = 0) -> int:
        """Trigger or retrigger a note on channel zero, returning its slot."""
        frame = ControlFrame()
        frame.add_note_on(note, velocity_q15)
        self._engine.consume_control_frame(frame)
        voice_id = self._engine._find_voice_by_note(note, channel=0)
        if voice_id is None:
            return -1
        from .voice.unison import group_slots
        for vid in group_slots(self._engine, voice_id):
            self._engine.voices[vid].cents_i16 = cents
        return voice_id

    def note_off(self, note: int) -> None:
        """Release channel-zero voices playing this note."""
        frame = ControlFrame()
        frame.add_note_off(note)
        self._engine.consume_control_frame(frame)

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
                    frame.gain_q15 = (frame.gain_q15 * self._engine.master_gain_q15) >> 15
                    frames.append(frame)
        return frames

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
        """Reset the complete voice pool, globals and control latches."""
        self.__post_init__()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'midi_to_hz',
    'VoiceFrameEmitter',
    'MultiVoiceEmitter',
]
