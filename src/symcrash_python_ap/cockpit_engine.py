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
from .primitives import Q15_HALF, Q15_ONE

if TYPE_CHECKING:
    from .symbol.executor_scheduling import HealthTelemetry


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
    slew_blocks_remaining: int = 0


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
        self.mode_L = 3

        # Control state (all controls, indexed by ControlID)
        self._controls: Dict[int, ControlState] = {}
        self._init_controls()

        # Event queue (cleared after each tick)
        self._event_queue: List = []

        # Global state
        self._master_gain_q15 = Q15_ONE
        self._patched_controls = set()
        self._patch_targets = set()
        self._touched_controls = set()

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
        if control_id not in self._touched_controls:
            state.dirty = True  # An explicit default/zero can override a preset.
            self._touched_controls.add(control_id)
        if state.target_value != clamped:
            state.target_value = clamped
            if entry.slew_policy == SlewPolicy.NBLOCK:
                state.slew_blocks_remaining = max(1, entry.slew_param)
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
        value_q15 = max(0, min(Q15_ONE, value_q15))

        if pot_id == 0:
            # Attack: map 0-32767 to 1..500 blocks
            blocks = max(1, (value_q15 * 500) // Q15_ONE)
            self.set_control(ControlID.ATK_BLOCKS, blocks)
        elif pot_id == 1:
            # Decay: map 0-32767 to 1..1000 blocks
            blocks = max(1, (value_q15 * 1000) // Q15_ONE)
            self.set_control(ControlID.DCY_BLOCKS, blocks)
        elif pot_id == 2:
            # Sustain: direct Q15
            self.set_control(ControlID.SUS_Q15, value_q15)
        elif pot_id == 3:
            # Release: map 0-32767 to 1..1500 blocks
            blocks = max(1, (value_q15 * 1500) // Q15_ONE)
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
            cents = ((value_q15 - Q15_HALF) * 2400) // Q15_HALF
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
        note = max(0, min(127, int(note)))
        velocity = max(0, min(127, int(velocity)))
        channel = max(0, min(15, int(channel)))
        if velocity == 0:
            self.midi_note_off(note, channel)
            return
        # Convert MIDI velocity (0-127) to Q15 (0-32767)
        velocity_q15 = (velocity * Q15_ONE) // 127

        self._event_queue.append(('note_on', note, velocity_q15, channel))

    def midi_note_off(self, note: int, channel: int = 0) -> None:
        """
        Queue MIDI note-off event (async input).

        Args:
            note: MIDI note (0-127)
            channel: MIDI channel (0-15)
        """
        self._event_queue.append(('note_off', max(0, min(127, int(note))),
                                  max(0, min(15, int(channel)))))

    def midi_pitch_bend(self, value: int, channel: int = 0) -> None:
        """
        Set MIDI pitch bend (async input).

        Args:
            value: MIDI pitch bend (0-16383, center=8192)
            channel: MIDI channel (0-15)
        """
        # Map 0-16383 to -200..+200 cents
        cents = ((value - 8192) * 200) // 8192  # q15-literal-ok (MIDI pitch-bend center, 2^13)
        self.set_control(ControlID.PITCH_BEND, cents)

    def load_preset(self, preset_id: int, preset_hash32: int = 0) -> None:
        """Queue selection of a preset registered with the receiving engine."""
        for value in (preset_id, preset_hash32):
            if not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
                raise ValueError('Preset ID and hash must be uint32')
        self._event_queue.append(('preset_load', preset_id, preset_hash32))

    def set_mode(self, mode_L: int) -> None:
        """Queue the editing/performance level for the next block (L0..L5)."""
        if not isinstance(mode_L, int) or not 0 <= mode_L <= 5:
            raise ValueError('Mode must be in 0..5')
        self._event_queue.append(('mode_set', mode_L))

    def capture_mark(self, mark_id: int) -> None:
        """Queue a recorder/debug marker at the next block boundary."""
        if not isinstance(mark_id, int) or not 0 <= mark_id <= 0xFFFFFFFF:
            raise ValueError('Capture mark ID must be uint32')
        self._event_queue.append(('capture_mark', mark_id))

    # =========================================================================
    # TICK (Block-rate)
    # =========================================================================

    def tick(self, voice_ids=None) -> ControlFrame:
        """
        Tick the engine (once per block).

        Applies slew filtering and produces a ControlFrame. voice_ids selects
        the controlled pool (default: slot 0); every patch has a real voice ID.

        Returns:
            ControlFrame for this block
        """
        # Apply slew filtering to all controls
        self._apply_slew()

        # Build ControlFrame
        frame = ControlFrame(
            block_index=self.block_index,
            mode_L=self.mode_L,
            flags=0,
            globals=GlobalControls(
                master_gain_q15=self._controls[ControlID.MASTER_GAIN].current_value
            ),
        )

        # Fan out explicitly: slot zero is an ordinary voice, not a broadcast.
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

        self._patched_controls.update(item.control_id for item in dirty_items)
        targets = tuple(dict.fromkeys((0,) if voice_ids is None else voice_ids))
        for voice_id in targets:
            items = dirty_items
            if voice_id not in self._patch_targets:
                items = [ControlItem(cid, self.registry.get_entry(cid).value_type,
                                     self._controls[cid].current_value)
                         for cid in sorted(self._patched_controls)]
            if items:
                frame.voice_patches.append(VoiceControlPatch(voice_id, list(items)))
        self._patch_targets = set(targets)

        # Add events from queue
        for event_data in self._event_queue:
            if event_data[0] == 'note_on':
                _, note, velocity_q15, channel = event_data
                frame.add_note_on(note, velocity_q15, channel)
            elif event_data[0] == 'note_off':
                _, note, channel = event_data
                frame.add_note_off(note, channel)
            elif event_data[0] == 'preset_load':
                _, preset_id, preset_hash32 = event_data
                frame.add_preset_load(preset_id, preset_hash32)
            elif event_data[0] == 'mode_set':
                self.mode_L = event_data[1]
                frame.add_mode_set(self.mode_L)
            elif event_data[0] == 'capture_mark':
                frame.add_capture_mark(event_data[1])

        frame.mode_L = self.mode_L

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
                    max(1, state.slew_blocks_remaining),
                )
                state.slew_blocks_remaining = max(0, state.slew_blocks_remaining - 1)
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
            return Q15_ONE  # Default to full scale

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
                'stab_bar': Q15_ONE,  # Full stability as default
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
