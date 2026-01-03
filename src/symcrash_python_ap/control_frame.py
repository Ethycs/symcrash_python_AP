"""
ControlFrame
============

Block-rate control snapshot (ControlFrame_v0 ABI).

Implements WSE-STD-CONTROLFRAME v0 (frozen).

Produced by CockpitEngine, consumed by VoiceFrameEmitter.

Usage:
    from symcrash_python_ap.control_frame import ControlFrame, ControlEvent, EventType

    frame = ControlFrame(block_index=42)
    frame.add_note_on(note=60, velocity_q15=32767)

    # Process events
    for event in frame.events:
        if event.type == EventType.NOTE_ON:
            note = event.a & 0x7F
            velocity_q15 = (event.a >> 8)

    frame.clear_events()
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Dict

from .control_registry import ValueType


# =============================================================================
# ENUMS
# =============================================================================

class EventType(IntEnum):
    """Event types (consumed once)."""
    NOTE_ON = 0
    NOTE_OFF = 1
    PRESET_LOAD = 2
    MODE_SET = 3
    CAPTURE_MARK = 4


# =============================================================================
# CONTROL ITEM
# =============================================================================

@dataclass
class ControlItem:
    """Single control update."""
    control_id: int
    value_type: ValueType
    value: int


# =============================================================================
# VOICE CONTROL PATCH
# =============================================================================

@dataclass
class VoiceControlPatch:
    """Sparse control updates for one voice."""
    voice_id: int
    items: List[ControlItem] = field(default_factory=list)


# =============================================================================
# CONTROL EVENT
# =============================================================================

@dataclass
class ControlEvent:
    """Discrete event (consumed once)."""
    type: EventType
    channel: int = 0
    voice_id: int = 0xFFFF  # 0xFFFF = allocator decides
    a: int = 0
    b: int = 0


# =============================================================================
# GLOBAL CONTROLS
# =============================================================================

@dataclass
class GlobalControls:
    """Global control state."""
    master_gain_q15: Optional[int] = None
    master_pan_q15: Optional[int] = None
    tempo_q8_8: Optional[int] = None


# =============================================================================
# CONTROL FRAME
# =============================================================================

@dataclass
class ControlFrame:
    """
    Block-rate control snapshot.

    Produced by CockpitEngine, consumed by VoiceFrameEmitter.
    """
    block_index: int = 0
    mode_L: int = 3  # Default to L3 (performance mode)
    flags: int = 0

    globals: GlobalControls = field(default_factory=GlobalControls)
    voice_patches: List[VoiceControlPatch] = field(default_factory=list)
    events: List[ControlEvent] = field(default_factory=list)

    def clear_events(self) -> None:
        """Clear events after processing."""
        self.events.clear()

    def add_note_on(self, note: int, velocity_q15: int, channel: int = 0) -> None:
        """Add note-on event."""
        self.events.append(ControlEvent(
            type=EventType.NOTE_ON,
            channel=channel,
            voice_id=0xFFFF,
            a=note | (velocity_q15 << 8),
            b=0
        ))

    def add_note_off(self, note: int, channel: int = 0) -> None:
        """Add note-off event."""
        self.events.append(ControlEvent(
            type=EventType.NOTE_OFF,
            channel=channel,
            voice_id=0xFFFF,
            a=note,
            b=0
        ))

    def add_preset_load(self, preset_id: int, preset_hash32: int = 0) -> None:
        """Add preset-load event."""
        self.events.append(ControlEvent(
            type=EventType.PRESET_LOAD,
            channel=0,
            voice_id=0xFFFF,
            a=preset_id,
            b=preset_hash32
        ))

    def add_mode_set(self, mode_L: int) -> None:
        """Add mode-set event."""
        self.events.append(ControlEvent(
            type=EventType.MODE_SET,
            channel=0,
            voice_id=0xFFFF,
            a=mode_L,
            b=0
        ))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'EventType',
    'ControlItem',
    'VoiceControlPatch',
    'ControlEvent',
    'GlobalControls',
    'ControlFrame',
]
