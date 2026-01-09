"""
WSE-STD-CONTROLFRAME Conformance Tests
======================================

Tests for ControlFrame conformance with frozen v0 specification.

Per WSE-STD-CONTROLFRAME v0, the frame MUST:
- Contain block_index for timing
- Support discrete events (NOTE_ON, NOTE_OFF, PRESET_LOAD, MODE_SET, CAPTURE_MARK)
- Support sparse voice patches
- Support global control updates
- Be producible by CockpitEngine and consumable by VoiceFrameEmitter
"""

import pytest


# =============================================================================
# FROZEN SPEC VALUES (from WSE-STD-CONTROLFRAME v0)
# =============================================================================

FROZEN_EVENT_TYPES = {
    "NOTE_ON": 0,
    "NOTE_OFF": 1,
    "PRESET_LOAD": 2,
    "MODE_SET": 3,
    "CAPTURE_MARK": 4,
}


# =============================================================================
# CONFORMANCE: EVENT TYPES
# =============================================================================

class TestEventTypeEnum:
    """Verify EventType enum values match frozen spec."""

    def test_note_on_value(self):
        """NOTE_ON MUST be 0."""
        from symcrash_python_ap.control_frame import EventType
        assert EventType.NOTE_ON == 0

    def test_note_off_value(self):
        """NOTE_OFF MUST be 1."""
        from symcrash_python_ap.control_frame import EventType
        assert EventType.NOTE_OFF == 1

    def test_preset_load_value(self):
        """PRESET_LOAD MUST be 2."""
        from symcrash_python_ap.control_frame import EventType
        assert EventType.PRESET_LOAD == 2

    def test_mode_set_value(self):
        """MODE_SET MUST be 3."""
        from symcrash_python_ap.control_frame import EventType
        assert EventType.MODE_SET == 3

    def test_capture_mark_value(self):
        """CAPTURE_MARK MUST be 4."""
        from symcrash_python_ap.control_frame import EventType
        assert EventType.CAPTURE_MARK == 4


# =============================================================================
# CONFORMANCE: CONTROL FRAME STRUCTURE
# =============================================================================

class TestControlFrameStructure:
    """Verify ControlFrame has required structure."""

    def test_has_block_index(self):
        """ControlFrame MUST have block_index field."""
        from symcrash_python_ap.control_frame import ControlFrame

        frame = ControlFrame(block_index=42)
        assert hasattr(frame, 'block_index')
        assert frame.block_index == 42

    def test_has_mode_l(self):
        """ControlFrame MUST have mode_L field."""
        from symcrash_python_ap.control_frame import ControlFrame

        frame = ControlFrame()
        assert hasattr(frame, 'mode_L')

    def test_has_flags(self):
        """ControlFrame MUST have flags field."""
        from symcrash_python_ap.control_frame import ControlFrame

        frame = ControlFrame()
        assert hasattr(frame, 'flags')

    def test_has_globals(self):
        """ControlFrame MUST have globals field."""
        from symcrash_python_ap.control_frame import ControlFrame

        frame = ControlFrame()
        assert hasattr(frame, 'globals')

    def test_has_voice_patches(self):
        """ControlFrame MUST have voice_patches field."""
        from symcrash_python_ap.control_frame import ControlFrame

        frame = ControlFrame()
        assert hasattr(frame, 'voice_patches')
        assert isinstance(frame.voice_patches, list)

    def test_has_events(self):
        """ControlFrame MUST have events field."""
        from symcrash_python_ap.control_frame import ControlFrame

        frame = ControlFrame()
        assert hasattr(frame, 'events')
        assert isinstance(frame.events, list)


# =============================================================================
# CONFORMANCE: GLOBAL CONTROLS
# =============================================================================

class TestGlobalControls:
    """Verify GlobalControls has required fields."""

    def test_has_master_gain(self):
        """GlobalControls MUST have master_gain_q15 field."""
        from symcrash_python_ap.control_frame import GlobalControls

        gc = GlobalControls()
        assert hasattr(gc, 'master_gain_q15')

    def test_has_master_pan(self):
        """GlobalControls MUST have master_pan_q15 field."""
        from symcrash_python_ap.control_frame import GlobalControls

        gc = GlobalControls()
        assert hasattr(gc, 'master_pan_q15')

    def test_has_tempo(self):
        """GlobalControls MUST have tempo_q8_8 field."""
        from symcrash_python_ap.control_frame import GlobalControls

        gc = GlobalControls()
        assert hasattr(gc, 'tempo_q8_8')


# =============================================================================
# CONFORMANCE: CONTROL EVENT
# =============================================================================

class TestControlEvent:
    """Verify ControlEvent has required structure."""

    def test_event_has_type(self):
        """ControlEvent MUST have type field."""
        from symcrash_python_ap.control_frame import ControlEvent, EventType

        event = ControlEvent(type=EventType.NOTE_ON)
        assert hasattr(event, 'type')

    def test_event_has_channel(self):
        """ControlEvent MUST have channel field."""
        from symcrash_python_ap.control_frame import ControlEvent, EventType

        event = ControlEvent(type=EventType.NOTE_ON, channel=5)
        assert hasattr(event, 'channel')
        assert event.channel == 5

    def test_event_has_voice_id(self):
        """ControlEvent MUST have voice_id field."""
        from symcrash_python_ap.control_frame import ControlEvent, EventType

        event = ControlEvent(type=EventType.NOTE_ON)
        assert hasattr(event, 'voice_id')

    def test_event_has_ab_payload(self):
        """ControlEvent MUST have a and b payload fields."""
        from symcrash_python_ap.control_frame import ControlEvent, EventType

        event = ControlEvent(type=EventType.NOTE_ON, a=123, b=456)
        assert hasattr(event, 'a')
        assert hasattr(event, 'b')
        assert event.a == 123
        assert event.b == 456


# =============================================================================
# CONFORMANCE: NOTE ON EVENT
# =============================================================================

class TestNoteOnEvent:
    """Verify NOTE_ON event format."""

    def test_add_note_on(self):
        """add_note_on MUST create proper event."""
        from symcrash_python_ap.control_frame import ControlFrame, EventType

        frame = ControlFrame()
        frame.add_note_on(note=60, velocity_q15=32767)

        assert len(frame.events) == 1
        event = frame.events[0]
        assert event.type == EventType.NOTE_ON

    def test_note_on_packing(self):
        """NOTE_ON a field MUST pack note and velocity."""
        from symcrash_python_ap.control_frame import ControlFrame

        frame = ControlFrame()
        frame.add_note_on(note=60, velocity_q15=16384)

        event = frame.events[0]
        # Note is in lower bits, velocity in upper bits
        note = event.a & 0x7F
        assert note == 60

    def test_note_on_channel(self):
        """NOTE_ON MUST support channel."""
        from symcrash_python_ap.control_frame import ControlFrame

        frame = ControlFrame()
        frame.add_note_on(note=60, velocity_q15=32767, channel=3)

        event = frame.events[0]
        assert event.channel == 3


# =============================================================================
# CONFORMANCE: NOTE OFF EVENT
# =============================================================================

class TestNoteOffEvent:
    """Verify NOTE_OFF event format."""

    def test_add_note_off(self):
        """add_note_off MUST create proper event."""
        from symcrash_python_ap.control_frame import ControlFrame, EventType

        frame = ControlFrame()
        frame.add_note_off(note=60)

        assert len(frame.events) == 1
        event = frame.events[0]
        assert event.type == EventType.NOTE_OFF

    def test_note_off_contains_note(self):
        """NOTE_OFF a field MUST contain note number."""
        from symcrash_python_ap.control_frame import ControlFrame

        frame = ControlFrame()
        frame.add_note_off(note=72)

        event = frame.events[0]
        assert event.a == 72


# =============================================================================
# CONFORMANCE: PRESET LOAD EVENT
# =============================================================================

class TestPresetLoadEvent:
    """Verify PRESET_LOAD event format."""

    def test_add_preset_load(self):
        """add_preset_load MUST create proper event."""
        from symcrash_python_ap.control_frame import ControlFrame, EventType

        frame = ControlFrame()
        frame.add_preset_load(preset_id=42, preset_hash32=0xDEADBEEF)

        assert len(frame.events) == 1
        event = frame.events[0]
        assert event.type == EventType.PRESET_LOAD

    def test_preset_load_contains_id(self):
        """PRESET_LOAD MUST contain preset_id in a field."""
        from symcrash_python_ap.control_frame import ControlFrame

        frame = ControlFrame()
        frame.add_preset_load(preset_id=42, preset_hash32=0xABCD)

        event = frame.events[0]
        assert event.a == 42
        assert event.b == 0xABCD


# =============================================================================
# CONFORMANCE: MODE SET EVENT
# =============================================================================

class TestModeSetEvent:
    """Verify MODE_SET event format."""

    def test_add_mode_set(self):
        """add_mode_set MUST create proper event."""
        from symcrash_python_ap.control_frame import ControlFrame, EventType

        frame = ControlFrame()
        frame.add_mode_set(mode_L=2)

        assert len(frame.events) == 1
        event = frame.events[0]
        assert event.type == EventType.MODE_SET


# =============================================================================
# CONFORMANCE: EVENT CLEARING
# =============================================================================

class TestEventClearing:
    """Verify event clearing behavior."""

    def test_clear_events(self):
        """clear_events MUST remove all events."""
        from symcrash_python_ap.control_frame import ControlFrame

        frame = ControlFrame()
        frame.add_note_on(note=60, velocity_q15=32767)
        frame.add_note_on(note=64, velocity_q15=32767)
        frame.add_note_off(note=60)

        assert len(frame.events) == 3

        frame.clear_events()

        assert len(frame.events) == 0


# =============================================================================
# CONFORMANCE: VOICE PATCHES
# =============================================================================

class TestVoicePatches:
    """Verify voice patch structure."""

    def test_voice_control_patch_structure(self):
        """VoiceControlPatch MUST have voice_id and items."""
        from symcrash_python_ap.control_frame import VoiceControlPatch

        patch = VoiceControlPatch(voice_id=3)
        assert hasattr(patch, 'voice_id')
        assert hasattr(patch, 'items')
        assert patch.voice_id == 3
        assert isinstance(patch.items, list)

    def test_control_item_structure(self):
        """ControlItem MUST have control_id, value_type, value."""
        from symcrash_python_ap.control_frame import ControlItem
        from symcrash_python_ap.control_registry import ValueType

        item = ControlItem(control_id=0, value_type=ValueType.Q15U, value=16384)
        assert hasattr(item, 'control_id')
        assert hasattr(item, 'value_type')
        assert hasattr(item, 'value')


# =============================================================================
# CONFORMANCE: DEFAULT VALUES
# =============================================================================

class TestDefaultValues:
    """Verify default values are appropriate."""

    def test_default_block_index_zero(self):
        """Default block_index MUST be 0."""
        from symcrash_python_ap.control_frame import ControlFrame

        frame = ControlFrame()
        assert frame.block_index == 0

    def test_default_mode_l3(self):
        """Default mode_L MUST be 3 (performance mode)."""
        from symcrash_python_ap.control_frame import ControlFrame

        frame = ControlFrame()
        assert frame.mode_L == 3

    def test_default_voice_id_allocator(self):
        """Default voice_id MUST be 0xFFFF (allocator decides)."""
        from symcrash_python_ap.control_frame import ControlEvent, EventType

        event = ControlEvent(type=EventType.NOTE_ON)
        assert event.voice_id == 0xFFFF


# =============================================================================
# CONFORMANCE: ABI STABILITY
# =============================================================================

class TestABIStability:
    """Verify ControlFrame is ABI-stable."""

    def test_frame_instantiation(self):
        """ControlFrame MUST be instantiable with defaults."""
        from symcrash_python_ap.control_frame import ControlFrame

        frame = ControlFrame()
        assert frame is not None

    def test_frame_instantiation_with_args(self):
        """ControlFrame MUST accept block_index at construction."""
        from symcrash_python_ap.control_frame import ControlFrame

        frame = ControlFrame(block_index=100)
        assert frame.block_index == 100

    def test_event_type_count_stable(self):
        """EventType MUST have at least 5 types (frozen v0)."""
        from symcrash_python_ap.control_frame import EventType

        # Count all event types
        event_types = [e for e in EventType]
        assert len(event_types) >= 5
