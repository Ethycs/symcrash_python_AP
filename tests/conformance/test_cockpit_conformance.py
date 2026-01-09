"""
WSE-STD-COCKPIT-ENGINE Conformance Tests
=========================================

Tests for CockpitEngine conformance with frozen v0 specification.

Per WSE-STD-COCKPIT-ENGINE v0, the engine MUST:
- Accept async inputs (pots, MIDI, control deltas)
- Apply slew filtering (INSTANT, NBLOCK, ONE_POLE)
- Sample to block-rate (fly-by-wire principle)
- Emit deterministic ControlFrames
- Track block_index continuously
- Support inject_macro() for external control sources
"""

import pytest


# =============================================================================
# CONFORMANCE: ENGINE INSTANTIATION
# =============================================================================

class TestCockpitEngineInstantiation:
    """Verify engine instantiation behavior."""

    def test_engine_creates(self):
        """CockpitEngine MUST be instantiable."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine

        engine = CockpitEngine()
        assert engine is not None

    def test_engine_has_block_index(self):
        """CockpitEngine MUST have block_index."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine

        engine = CockpitEngine()
        assert hasattr(engine, 'block_index')
        assert engine.block_index == 0

    def test_engine_has_registry(self):
        """CockpitEngine MUST have a ControlRegistry."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine

        engine = CockpitEngine()
        assert hasattr(engine, 'registry')


# =============================================================================
# CONFORMANCE: CONTROL INPUT
# =============================================================================

class TestControlInput:
    """Verify control input behavior."""

    def test_set_control_clamps(self):
        """set_control MUST clamp values to valid range."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine
        from symcrash_python_ap.control_registry import ControlID

        engine = CockpitEngine()

        # Set value above max
        engine.set_control(ControlID.TIMBRE, 50000)
        current = engine.get_control_current(ControlID.TIMBRE)
        assert current <= 32767

    def test_set_control_marks_dirty(self):
        """set_control MUST mark control as dirty when value changes."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine
        from symcrash_python_ap.control_registry import ControlID

        engine = CockpitEngine()
        engine.set_control(ControlID.TIMBRE, 16384)

        # Tick to collect dirty controls
        frame = engine.tick()

        # Should have a voice patch with the changed control
        assert len(frame.voice_patches) > 0

    def test_unknown_control_ignored(self):
        """set_control MUST ignore unknown control IDs."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine

        engine = CockpitEngine()

        # Should not raise
        engine.set_control(999, 16384)

        # Tick should still work
        frame = engine.tick()
        assert frame is not None


# =============================================================================
# CONFORMANCE: POT INPUT
# =============================================================================

class TestPotInput:
    """Verify pot input behavior."""

    def test_pot_attack(self):
        """Pot 0 MUST map to ATK_BLOCKS."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine
        from symcrash_python_ap.control_registry import ControlID

        engine = CockpitEngine()
        engine.set_pot(0, 16384)  # 50%

        # Should map to attack blocks
        value = engine.get_control_target(ControlID.ATK_BLOCKS)
        assert value > 0

    def test_pot_sustain(self):
        """Pot 2 MUST map to SUS_Q15."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine
        from symcrash_python_ap.control_registry import ControlID

        engine = CockpitEngine()
        engine.set_pot(2, 24576)  # ~75%

        value = engine.get_control_target(ControlID.SUS_Q15)
        assert value == 24576

    def test_pot_detune(self):
        """Pot 8 MUST map to DETUNE with bipolar range."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine
        from symcrash_python_ap.control_registry import ControlID

        engine = CockpitEngine()

        # Center position = 0 cents
        engine.set_pot(8, 16384)
        value = engine.get_control_target(ControlID.DETUNE)
        assert abs(value) < 100  # Near zero

        # Full up = positive cents
        engine.set_pot(8, 32767)
        value = engine.get_control_target(ControlID.DETUNE)
        assert value > 0

        # Full down = negative cents
        engine.set_pot(8, 0)
        value = engine.get_control_target(ControlID.DETUNE)
        assert value < 0


# =============================================================================
# CONFORMANCE: MIDI INPUT
# =============================================================================

class TestMidiInput:
    """Verify MIDI input behavior."""

    def test_midi_note_on(self):
        """midi_note_on MUST queue NOTE_ON event."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine
        from symcrash_python_ap.control_frame import EventType

        engine = CockpitEngine()
        engine.midi_note_on(note=60, velocity=100)

        frame = engine.tick()

        assert len(frame.events) == 1
        assert frame.events[0].type == EventType.NOTE_ON

    def test_midi_note_off(self):
        """midi_note_off MUST queue NOTE_OFF event."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine
        from symcrash_python_ap.control_frame import EventType

        engine = CockpitEngine()
        engine.midi_note_off(note=60)

        frame = engine.tick()

        assert len(frame.events) == 1
        assert frame.events[0].type == EventType.NOTE_OFF

    def test_midi_pitch_bend(self):
        """midi_pitch_bend MUST update PITCH_BEND control."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine
        from symcrash_python_ap.control_registry import ControlID

        engine = CockpitEngine()

        # Center = 8192
        engine.midi_pitch_bend(8192)
        value = engine.get_control_target(ControlID.PITCH_BEND)
        assert abs(value) < 10  # Near zero

        # Full up
        engine.midi_pitch_bend(16383)
        value = engine.get_control_target(ControlID.PITCH_BEND)
        assert value > 0

    def test_midi_velocity_conversion(self):
        """MIDI velocity MUST be converted to Q15."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine

        engine = CockpitEngine()
        engine.midi_note_on(note=60, velocity=127)

        frame = engine.tick()
        event = frame.events[0]

        # Velocity should be near max Q15
        velocity_q15 = event.a >> 8
        assert velocity_q15 > 30000


# =============================================================================
# CONFORMANCE: TICK BEHAVIOR
# =============================================================================

class TestTickBehavior:
    """Verify tick() behavior."""

    def test_tick_returns_control_frame(self):
        """tick() MUST return a ControlFrame."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine
        from symcrash_python_ap.control_frame import ControlFrame

        engine = CockpitEngine()
        frame = engine.tick()

        assert isinstance(frame, ControlFrame)

    def test_tick_increments_block_index(self):
        """tick() MUST increment block_index."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine

        engine = CockpitEngine()
        assert engine.block_index == 0

        engine.tick()
        assert engine.block_index == 1

        engine.tick()
        assert engine.block_index == 2

    def test_tick_frame_has_block_index(self):
        """tick() MUST set block_index in ControlFrame."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine

        engine = CockpitEngine()

        frame1 = engine.tick()
        assert frame1.block_index == 0

        frame2 = engine.tick()
        assert frame2.block_index == 1

    def test_tick_clears_event_queue(self):
        """tick() MUST clear event queue after processing."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine

        engine = CockpitEngine()
        engine.midi_note_on(note=60, velocity=100)

        frame1 = engine.tick()
        assert len(frame1.events) == 1

        # Second tick should have no events
        frame2 = engine.tick()
        assert len(frame2.events) == 0


# =============================================================================
# CONFORMANCE: SLEW FILTERING
# =============================================================================

class TestSlewFiltering:
    """Verify slew filtering behavior."""

    def test_instant_slew(self):
        """INSTANT slew MUST apply immediately."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine
        from symcrash_python_ap.control_registry import ControlID

        engine = CockpitEngine()

        # PITCH_BEND uses INSTANT slew
        engine.set_control(ControlID.PITCH_BEND, 100)
        engine.tick()

        current = engine.get_control_current(ControlID.PITCH_BEND)
        target = engine.get_control_target(ControlID.PITCH_BEND)

        assert current == target == 100

    def test_nblock_slew_gradual(self):
        """NBLOCK slew MUST move gradually toward target."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine
        from symcrash_python_ap.control_registry import ControlID

        engine = CockpitEngine()

        # MORPH uses NBLOCK slew
        initial = engine.get_control_current(ControlID.MORPH)
        engine.set_control(ControlID.MORPH, 32767)

        # First tick
        engine.tick()
        after_one = engine.get_control_current(ControlID.MORPH)

        # Should move toward target but not reach it
        if initial != 32767:
            assert after_one > initial
            assert after_one < 32767

    def test_one_pole_slew_gradual(self):
        """ONE_POLE slew MUST move gradually toward target."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine
        from symcrash_python_ap.control_registry import ControlID

        engine = CockpitEngine()

        # TIMBRE uses ONE_POLE slew
        engine.set_control(ControlID.TIMBRE, 0)

        # Multiple ticks
        for _ in range(10):
            engine.tick()

        after_ten = engine.get_control_current(ControlID.TIMBRE)

        # Should move toward 0 but may not reach it yet
        assert after_ten < 32767


# =============================================================================
# CONFORMANCE: GLOBALS
# =============================================================================

class TestGlobals:
    """Verify global control handling."""

    def test_master_gain_in_globals(self):
        """tick() MUST include master_gain in globals."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine
        from symcrash_python_ap.control_registry import ControlID

        engine = CockpitEngine()
        engine.set_control(ControlID.MASTER_GAIN, 16384)

        # Tick several times to apply slew
        for _ in range(100):
            frame = engine.tick()

        # Globals should have master gain
        assert frame.globals.master_gain_q15 is not None


# =============================================================================
# CONFORMANCE: DETERMINISM
# =============================================================================

class TestCockpitDeterminism:
    """Verify deterministic behavior."""

    def test_same_inputs_same_output(self):
        """Same inputs MUST produce same ControlFrame."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine
        from symcrash_python_ap.control_registry import ControlID

        def run_sequence():
            engine = CockpitEngine()
            engine.set_control(ControlID.TIMBRE, 16384)
            engine.set_control(ControlID.MORPH, 8192)
            engine.midi_note_on(note=60, velocity=100)
            return engine.tick()

        frame1 = run_sequence()
        frame2 = run_sequence()

        assert frame1.block_index == frame2.block_index
        assert len(frame1.events) == len(frame2.events)

    def test_tick_order_matters(self):
        """Tick order MUST affect block_index."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine

        engine = CockpitEngine()

        frames = []
        for i in range(5):
            frames.append(engine.tick())

        for i, frame in enumerate(frames):
            assert frame.block_index == i


# =============================================================================
# CONFORMANCE: QUERY API
# =============================================================================

class TestQueryAPI:
    """Verify query API behavior."""

    def test_get_control_current(self):
        """get_control_current MUST return slewed value."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine
        from symcrash_python_ap.control_registry import ControlID

        engine = CockpitEngine()
        value = engine.get_control_current(ControlID.TIMBRE)

        # Should return the default value
        assert value == 32767

    def test_get_control_target(self):
        """get_control_target MUST return unslewed value."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine
        from symcrash_python_ap.control_registry import ControlID

        engine = CockpitEngine()
        engine.set_control(ControlID.TIMBRE, 0)

        # Target should be 0 immediately
        target = engine.get_control_target(ControlID.TIMBRE)
        assert target == 0

        # Current may still be slewing
        current = engine.get_control_current(ControlID.TIMBRE)
        # After set, current hasn't ticked yet, so may still be default
        assert current >= 0

    def test_unknown_control_returns_zero(self):
        """get_control_* MUST return 0 for unknown control IDs."""
        from symcrash_python_ap.cockpit_engine import CockpitEngine

        engine = CockpitEngine()

        current = engine.get_control_current(999)
        target = engine.get_control_target(999)

        assert current == 0
        assert target == 0
