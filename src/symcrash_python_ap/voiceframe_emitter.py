"""
VoiceFrame Emitter
==================

Emits VoiceFrames from compiled preset + ControlFrame.

The emitter is the bridge between the preset (static configuration),
ControlFrame (control intent), and the VM (which consumes VoiceFrames).

It handles:
- Consuming ControlFrames from CockpitEngine
- Envelope scheduling (block-rate ADSR)
- Beta (dry/wet) ramping
- Note-on/note-off transitions
- MIDI note to Hz conversion
- Applying control modifiers (family scales, pitch offset, gain, etc.)

Usage:
    from preset_compiler import PresetCompiler
    from voiceframe_emitter import VoiceFrameEmitter
    from cockpit_engine import CockpitEngine

    compiler = PresetCompiler()
    preset = compiler.compile_file("brass_stab_v0.yaml")

    cockpit = CockpitEngine()
    emitter = VoiceFrameEmitter(preset)

    # Each block:
    control_frame = cockpit.tick()
    emitter.consume_control_frame(control_frame)
    frame = emitter.emit_frame()
    # Pass frame to VM
"""

from dataclasses import dataclass, field
from typing import Optional, List
import math

from .library.preset_compiler import CompiledPreset, Q15_ONE
from .envelope_scheduler import EnvelopeScheduler, ADSRSpec
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
# VOICEFRAME EMITTER
# =============================================================================

@dataclass
class VoiceFrameEmitter:
    """
    Emits VoiceFrames from compiled preset + ControlFrame.

    Manages per-voice state including:
    - Envelope (ADSR)
    - Beta ramp (dry/wet blend)
    - Note state (pitch, velocity)
    - Control modifiers (pitch offset, gain scale, family scales, etc.)
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

    # Control modifiers (from ControlFrame)
    _pitch_offset_cents: int = 0       # DETUNE + PITCH_BEND
    _gain_scale_q15: int = Q15_ONE     # DYNAMICS control
    _tree_blend_q15: int = 0           # MORPH control
    _family_scale_q15: List[int] = field(default_factory=lambda: [Q15_ONE] * 8)  # 8 families
    _master_gain_q15: int = Q15_ONE    # Global master gain

    # ADSR overrides (from ControlFrame)
    _attack_blocks: Optional[int] = None
    _decay_blocks: Optional[int] = None
    _sustain_level_q15: Optional[int] = None
    _release_blocks: Optional[int] = None

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

    def consume_control_frame(self, frame: ControlFrame) -> None:
        """
        Consume a ControlFrame and update emitter state.

        Extracts control modifiers and events for this voice.

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

    def emit_frame(self) -> VoiceFrame:
        """
        Emit next VoiceFrame for this block.

        Call this once per block to get the VoiceFrame.

        Returns:
            VoiceFrame with all fields populated from preset + control modifiers
        """
        # Update envelope ADSR if overridden by controls
        if self._envelope and any([
            self._attack_blocks is not None,
            self._decay_blocks is not None,
            self._sustain_level_q15 is not None,
            self._release_blocks is not None,
        ]):
            # Create new ADSR spec with overrides
            adsr = ADSRSpec(
                attack_blocks=self._attack_blocks or self._envelope.spec.attack_blocks,
                decay_blocks=self._decay_blocks or self._envelope.spec.decay_blocks,
                sustain_level_q15=self._sustain_level_q15 or self._envelope.spec.sustain_level_q15,
                release_blocks=self._release_blocks or self._envelope.spec.release_blocks,
            )
            self._envelope.spec = adsr

        # Get envelope gain
        if self._envelope:
            gain_q15 = self._envelope.tick()
        else:
            gain_q15 = Q15_ONE if self._is_note_on else 0

        # Apply velocity scaling
        gain_q15 = (gain_q15 * self.velocity_q15) >> 15

        # Apply DYNAMICS control (gain_scale_q15)
        gain_q15 = (gain_q15 * self._gain_scale_q15) >> 15

        # Apply master gain
        gain_q15 = (gain_q15 * self._master_gain_q15) >> 15

        # Compute beta ramp (preset-driven)
        beta_target = self.preset.material.beta_target_q15
        ramp_blocks = self.preset.material.beta_ramp_blocks

        if self.block_index < ramp_blocks and ramp_blocks > 1:
            # Linear ramp from start to target
            beta_start = self.preset.material.beta_start_q15
            t = self.block_index / (ramp_blocks - 1)
            self._beta_current_q15 = int(beta_start + (beta_target - beta_start) * t)
        else:
            self._beta_current_q15 = beta_target

        # Compute f0_hz from MIDI note + cents + pitch offset
        total_cents = self.cents + self._pitch_offset_cents
        f0_hz = midi_to_hz(self.note, total_cents)

        # Apply family scales to preset families
        scaled_families = []
        for i, family_weight in enumerate(self.preset.backbone_families()):
            if i < len(self._family_scale_q15):
                scaled_weight = (family_weight * self._family_scale_q15[i]) >> 15
                scaled_families.append(scaled_weight)
            else:
                scaled_families.append(family_weight)

        # Use control-driven tree_blend_q15 if available
        tree_blend = self._tree_blend_q15 if self._tree_blend_q15 != 0 else self.preset.latch.tree_blend_q15

        # Determine oscillator mode and FM6 fields from preset
        osc_mode = getattr(self.preset, 'osc_mode', 0)
        fm6_config = getattr(self.preset, 'fm6', None)

        # Build VoiceFrame
        frame = VoiceFrame(
            voice_id=self.voice_id,
            f0_hz=f0_hz,
            note=self.note,
            cents=total_cents,
            gain_q15=gain_q15,
            beta_target_q15=self._beta_current_q15,
            partial_amps_q15=list(self.preset.harmonic_amps_q15),
            ops_enable=self.preset.backbone_ops(),
            families=scaled_families,
            tree_a_id=self.preset.latch.tree_a_id,
            tree_b_id=self.preset.latch.tree_b_id,
            tree_blend_q15=tree_blend,
            is_starting=(self.block_index == 0),
            is_active=self.is_active,
            # FM6 fields (from preset if available)
            coeff_osc_mode=osc_mode,
            fm_algorithm_id=fm6_config.algorithm_id if fm6_config else None,
            fm_op_ratios=list(fm6_config.op_ratios) if fm6_config else None,
            fm_op_levels=list(fm6_config.op_levels) if fm6_config else None,
            fm_feedback_depth=fm6_config.feedback_depth if fm6_config else None,
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
        self._pitch_offset_cents = 0
        self._gain_scale_q15 = Q15_ONE
        self._tree_blend_q15 = 0
        self._family_scale_q15 = [Q15_ONE] * 8
        self._master_gain_q15 = Q15_ONE
        self._attack_blocks = None
        self._decay_blocks = None
        self._sustain_level_q15 = None
        self._release_blocks = None
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
        from cockpit_engine import CockpitEngine

        cockpit = CockpitEngine()
        multi = MultiVoiceEmitter(preset, max_voices=8)

        # Each block:
        control_frame = cockpit.tick()
        multi.consume_control_frame(control_frame)
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

    def consume_control_frame(self, frame: ControlFrame) -> None:
        """
        Consume a ControlFrame and update all voices.

        Args:
            frame: ControlFrame from CockpitEngine
        """
        for voice in self._voices:
            voice.consume_control_frame(frame)

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
