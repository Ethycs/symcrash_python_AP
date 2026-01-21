"""
Cockpit Engine
==============

Block-rate control surface that converts async inputs (pots, MIDI, deltas)
into deterministic ControlFrames.

Implements WSE-STD-COCKPIT-ENGINE v0 (frozen).

The CockpitEngine is the unified control surface layer that:
- Receives async inputs (pot changes, MIDI events, parameter deltas)
- Applies slew filtering (INSTANT, NBLOCK, ONE_POLE)
- Samples to block-rate (fly-by-wire principle)
- Emits deterministic ControlFrames for VoiceFrameEmitter

Usage:
    from symcrash_python_ap.cockpit_engine import CockpitEngine
    from symcrash_python_ap.control_registry import ControlID

    engine = CockpitEngine()
    engine.set_control(ControlID.TIMBRE, 16384)  # 50%
    engine.midi_note_on(note=60, velocity_q15=32767)

    # Each block:
    frame = engine.tick()
    # Pass frame to VoiceFrameEmitter
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

from .library.central_registry import get_central_registry
from .control_registry import (
    ControlID,
    ValueType,
    SlewPolicy,
    slew_nblock,
    slew_one_pole,
)
from .control_frame import (
    ControlFrame,
    ControlItem,
    VoiceControlPatch,
    GlobalControls,
)

if TYPE_CHECKING:
    from .voice.scheduler import HealthTelemetry


# =============================================================================
# FAMILY ID
# =============================================================================

class FamilyID:
    """Frozen v0 family IDs (matches WSE-STD-020 VoiceFrame)."""
    COLOR = 0
    LOCAL = 1
    TRANSPORT = 2
    GLUE = 3
    TOPOLOGY = 4
    CATASTROPHE = 5
    INJECT = 6
    TAIL = 7

    NUM_FAMILIES = 8


# =============================================================================
# CONTROL STATE
# =============================================================================

@dataclass
class ControlState:
    """Per-control state with slew filtering."""
    control_id: int
    target_value: int          # Target value (from input)
    current_value: int         # Current value (after slew)
    dirty: bool = False        # True if changed since last tick


# =============================================================================
# COCKPIT ENGINE
# =============================================================================

class CockpitEngine:
    """
    Block-rate control surface.

    Converts async inputs into deterministic ControlFrames.
    """

    def __init__(self):
        self.registry = get_central_registry().controls
        self.block_index = 0

        # Control state (all controls, indexed by ControlID)
        self._controls: Dict[int, ControlState] = {}
        self._init_controls()

        # Event queue (cleared after each tick)
        self._event_queue: List = []

        # Global state
        self._master_gain_q15 = 32767

        # Health telemetry from scheduler (for display/logging)
        self._health_telemetry: Optional['HealthTelemetry'] = None

    def _init_controls(self) -> None:
        """Initialize all controls to defaults."""
        for entry in self.registry.all_entries():
            self._controls[entry.control_id] = ControlState(
                control_id=entry.control_id,
                target_value=entry.default_value,
                current_value=entry.default_value,
                dirty=False,
            )

    # =========================================================================
    # INPUT API (Async)
    # =========================================================================

    def set_control(self, control_id: int, value: int) -> None:
        """
        Set control target value (async input).

        Value will be clamped and slewed according to registry policy.

        Args:
            control_id: ControlID enum value
            value: Target value (will be clamped)
        """
        if control_id not in self._controls:
            # Unknown control, ignore
            return

        entry = self.registry.get_entry(control_id)
        clamped = self.registry.clamp(control_id, value)

        state = self._controls[control_id]
        if state.target_value != clamped:
            state.target_value = clamped
            state.dirty = True

    def set_pot(self, pot_id: int, value_q15: int) -> None:
        """
        Set pot value (async input).

        Pots map to controls via fixed mapping (matches original Pico2 layout):
        - Pot 0: ATK_BLOCKS (Attack, mapped from 0-32767 to 1..500 blocks)
        - Pot 1: DCY_BLOCKS (Decay, mapped from 0-32767 to 1..1000 blocks)
        - Pot 2: SUS_Q15 (Sustain level)
        - Pot 3: REL_BLOCKS (Release, mapped from 0-32767 to 1..1500 blocks)
        - Pot 4: CHAOS (Cutoff equivalent)
        - Pot 5: TEXTURE (Resonance equivalent)
        - Pot 6: RESERVED_7 (Harmonics - future)
        - Pot 7: MORPH (Eta/dry-wet blend)
        - Pot 8: DETUNE (mapped from 0-32767 to -2400..+2400 cents)
        - Pot 9: MASTER_GAIN

        Args:
            pot_id: Pot index (0-9)
            value_q15: Q15 value (0-32767)
        """
        # Clamp pot value
        value_q15 = max(0, min(32767, value_q15))

        if pot_id == 0:
            # Attack: map 0-32767 to 1..500 blocks
            blocks = max(1, (value_q15 * 500) // 32767)
            self.set_control(ControlID.ATK_BLOCKS, blocks)
        elif pot_id == 1:
            # Decay: map 0-32767 to 1..1000 blocks
            blocks = max(1, (value_q15 * 1000) // 32767)
            self.set_control(ControlID.DCY_BLOCKS, blocks)
        elif pot_id == 2:
            # Sustain: direct Q15
            self.set_control(ControlID.SUS_Q15, value_q15)
        elif pot_id == 3:
            # Release: map 0-32767 to 1..1500 blocks
            blocks = max(1, (value_q15 * 1500) // 32767)
            self.set_control(ControlID.REL_BLOCKS, blocks)
        elif pot_id == 4:
            # Cutoff -> CHAOS
            self.set_control(ControlID.CHAOS, value_q15)
        elif pot_id == 5:
            # Resonance -> TEXTURE
            self.set_control(ControlID.TEXTURE, value_q15)
        elif pot_id == 6:
            # Harmonics -> RESERVED_7 (not yet implemented in standard)
            self.set_control(ControlID.RESERVED_7, value_q15)
        elif pot_id == 7:
            # Eta (dry/wet) -> MORPH
            self.set_control(ControlID.MORPH, value_q15)
        elif pot_id == 8:
            # Detune: map 0-32767 to -2400..+2400 cents
            cents = ((value_q15 - 16384) * 2400) // 16384
            self.set_control(ControlID.DETUNE, cents)
        elif pot_id == 9:
            # Master gain
            self.set_control(ControlID.MASTER_GAIN, value_q15)

    def midi_note_on(self, note: int, velocity: int = 127, channel: int = 0) -> None:
        """
        Queue MIDI note-on event (async input).

        Args:
            note: MIDI note (0-127)
            velocity: MIDI velocity (0-127)
            channel: MIDI channel (0-15)
        """
        # Convert MIDI velocity (0-127) to Q15 (0-32767)
        velocity_q15 = (velocity * 32767) // 127

        self._event_queue.append(('note_on', note, velocity_q15, channel))

    def midi_note_off(self, note: int, channel: int = 0) -> None:
        """
        Queue MIDI note-off event (async input).

        Args:
            note: MIDI note (0-127)
            channel: MIDI channel (0-15)
        """
        self._event_queue.append(('note_off', note, channel))

    def midi_pitch_bend(self, value: int, channel: int = 0) -> None:
        """
        Set MIDI pitch bend (async input).

        Args:
            value: MIDI pitch bend (0-16383, center=8192)
            channel: MIDI channel (0-15)
        """
        # Map 0-16383 to -200..+200 cents
        cents = ((value - 8192) * 200) // 8192
        self.set_control(ControlID.PITCH_BEND, cents)

    # =========================================================================
    # TICK (Block-rate)
    # =========================================================================

    def tick(self) -> ControlFrame:
        """
        Tick the engine (once per block).

        Applies slew filtering and produces a ControlFrame.

        Returns:
            ControlFrame for this block
        """
        # Apply slew filtering to all controls
        self._apply_slew()

        # Build ControlFrame
        frame = ControlFrame(
            block_index=self.block_index,
            mode_L=3,  # Default to L3
            flags=0,
            globals=GlobalControls(
                master_gain_q15=self._controls[ControlID.MASTER_GAIN].current_value
            ),
        )

        # Add dirty controls as patches (for voice 0, global voice)
        dirty_items = []
        for control_id, state in self._controls.items():
            if state.dirty:
                entry = self.registry.get_entry(control_id)
                dirty_items.append(ControlItem(
                    control_id=control_id,
                    value_type=entry.value_type,
                    value=state.current_value,
                ))
                state.dirty = False  # Clear dirty flag

        if dirty_items:
            frame.voice_patches.append(VoiceControlPatch(
                voice_id=0,  # Global voice
                items=dirty_items,
            ))

        # Add events from queue
        for event_data in self._event_queue:
            if event_data[0] == 'note_on':
                _, note, velocity_q15, channel = event_data
                frame.add_note_on(note, velocity_q15, channel)
            elif event_data[0] == 'note_off':
                _, note, channel = event_data
                frame.add_note_off(note, channel)

        # Clear event queue
        self._event_queue.clear()

        # Increment block index
        self.block_index += 1

        return frame

    def _apply_slew(self) -> None:
        """Apply slew filtering to all controls."""
        for control_id, state in self._controls.items():
            if state.current_value == state.target_value:
                continue

            entry = self.registry.get_entry(control_id)

            if entry.slew_policy == SlewPolicy.INSTANT:
                state.current_value = state.target_value
                state.dirty = True

            elif entry.slew_policy == SlewPolicy.NBLOCK:
                new_value = slew_nblock(
                    state.current_value,
                    state.target_value,
                    entry.slew_param,
                )
                if new_value != state.current_value:
                    state.current_value = new_value
                    state.dirty = True

            elif entry.slew_policy == SlewPolicy.ONE_POLE:
                new_value = slew_one_pole(
                    state.current_value,
                    state.target_value,
                    entry.slew_param,
                    entry.min_value,
                    entry.max_value,
                )
                if new_value != state.current_value:
                    state.current_value = new_value
                    state.dirty = True

    # =========================================================================
    # QUERY API
    # =========================================================================

    def get_control_current(self, control_id: int) -> int:
        """Get current (slewed) control value."""
        if control_id not in self._controls:
            return 0
        return self._controls[control_id].current_value

    def get_control_target(self, control_id: int) -> int:
        """Get target (unslewed) control value."""
        if control_id not in self._controls:
            return 0
        return self._controls[control_id].target_value

    def get_family_scale(self, family_id: int) -> int:
        """
        Get family scale value (Q15).

        Maps control bindings to family scales:
        - FamilyID.COLOR -> TIMBRE
        - FamilyID.LOCAL -> TEXTURE
        - FamilyID.CATASTROPHE -> CHAOS

        Args:
            family_id: FamilyID value (0-7)

        Returns:
            Q15 scale value (0-32767)
        """
        if family_id == FamilyID.COLOR:
            return self.get_control_current(ControlID.TIMBRE)
        elif family_id == FamilyID.LOCAL:
            return self.get_control_current(ControlID.TEXTURE)
        elif family_id == FamilyID.CATASTROPHE:
            return self.get_control_current(ControlID.CHAOS)
        else:
            return 32767  # Default to full scale

    def get_pitch_offset_cents(self) -> int:
        """
        Get cumulative pitch offset in cents.

        Sum of DETUNE + PITCH_BEND.

        Returns:
            Pitch offset in cents
        """
        detune = self.get_control_current(ControlID.DETUNE)
        pitch_bend = self.get_control_current(ControlID.PITCH_BEND)
        return detune + pitch_bend

    def get_adsr(self) -> Dict[str, int]:
        """
        Get current ADSR parameters.

        Returns:
            Dict with attack_blocks, decay_blocks, sustain_level_q15, release_blocks
        """
        return {
            'attack_blocks': self.get_control_current(ControlID.ATK_BLOCKS),
            'decay_blocks': self.get_control_current(ControlID.DCY_BLOCKS),
            'sustain_level_q15': self.get_control_current(ControlID.SUS_Q15),
            'release_blocks': self.get_control_current(ControlID.REL_BLOCKS),
        }

    # =========================================================================
    # TELEMETRY API
    # =========================================================================

    def receive_health_telemetry(self, telemetry: 'HealthTelemetry') -> None:
        """
        Receive scheduler health telemetry for display.

        Called by orchestrator/pipeline after each block to provide CPU, memory,
        stability bars and gear state for UI display.

        Args:
            telemetry: HealthTelemetry from BlockScheduler
        """
        self._health_telemetry = telemetry

    def get_health_telemetry(self) -> Optional['HealthTelemetry']:
        """
        Get last received health telemetry.

        Returns:
            HealthTelemetry if available, None otherwise
        """
        return self._health_telemetry

    def get_health_summary(self) -> Dict[str, int]:
        """
        Get health telemetry summary for display.

        Returns:
            Dict with cpu_bar, mem_bar, stab_bar, gear (or zeros if no telemetry)
        """
        if self._health_telemetry is None:
            return {
                'cpu_bar': 0,
                'mem_bar': 0,
                'stab_bar': 32767,  # Full stability as default
                'gear': 0,
            }
        return {
            'cpu_bar': self._health_telemetry.cpu_bar,
            'mem_bar': self._health_telemetry.mem_bar,
            'stab_bar': self._health_telemetry.stab_bar,
            'gear': self._health_telemetry.gear,
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'FamilyID',
    'CockpitEngine',
]
