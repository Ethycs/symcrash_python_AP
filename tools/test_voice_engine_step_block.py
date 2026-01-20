"""Test VoiceEngine.step_block() API."""

from symcrash_python_ap.voice.voice_engine import (
    VoiceEngine,
    VoiceEngineOutput,
    RenderBatchItem,
    GestureBasis,
    Q15_ONE,
)
from symcrash_python_ap.control_frame import ControlFrame, EventType
from symcrash_python_ap.frames import ParamFrame


def test_step_block_without_control_frame():
    """Test step_block with no ControlFrame (backward compat)."""
    engine = VoiceEngine(max_voices=4)

    # Allocate and trigger a voice
    vid = engine.allocate_voice()
    print(f"Allocated voice {vid}")

    gesture = GestureBasis(gate=True, pressure_q15=Q15_ONE)
    engine.note_on(vid, gesture=gesture)

    # Step without ControlFrame
    output = engine.step_block()

    assert isinstance(output, VoiceEngineOutput)
    print(f"Output type: {type(output).__name__}")
    print(f"Render batch size: {len(output.render_batch)}")
    print(f"Block index: {output.block_index}")
    print(f"ParamFrame type: {type(output.param_frame).__name__}")
    print(f"Telemetry voices_serviced: {output.telemetry.voices_serviced}")

    assert len(output.render_batch) == 1
    assert output.block_index == 1
    assert output.telemetry.voices_serviced == 1


def test_step_block_with_control_frame():
    """Test step_block with ControlFrame."""
    engine = VoiceEngine(max_voices=4)

    # Pre-allocate a voice
    vid = engine.allocate_voice()
    gesture = GestureBasis(gate=True, pressure_q15=Q15_ONE)
    engine.note_on(vid, gesture=gesture)

    # Create ControlFrame
    cf = ControlFrame(block_index=1)

    # Step with ControlFrame
    output = engine.step_block(cf)

    print(f"With ControlFrame - batch size: {len(output.render_batch)}")
    assert len(output.render_batch) == 1


def test_step_block_note_on_event():
    """Test that NOTE_ON events in ControlFrame allocate voices."""
    engine = VoiceEngine(max_voices=4)

    # Create ControlFrame with NOTE_ON event
    cf = ControlFrame(block_index=0)
    cf.add_note_on(note=60, velocity_q15=Q15_ONE)

    # Before step - no voices
    assert engine.get_active_voice_count() == 0

    # Step with ControlFrame containing NOTE_ON
    output = engine.step_block(cf)

    print(f"After NOTE_ON event - batch size: {len(output.render_batch)}")
    print(f"Active voices: {engine.get_active_voice_count()}")

    # Voice should be allocated and playing
    assert engine.get_active_voice_count() == 1
    assert len(output.render_batch) == 1


def test_step_block_note_off_event():
    """Test that NOTE_OFF events in ControlFrame release voices."""
    engine = VoiceEngine(max_voices=4)

    # First, trigger a note via event
    cf1 = ControlFrame(block_index=0)
    cf1.add_note_on(note=60, velocity_q15=Q15_ONE)
    output1 = engine.step_block(cf1)

    print(f"After NOTE_ON: {len(output1.render_batch)} voices")
    assert len(output1.render_batch) == 1

    # Now send NOTE_OFF
    cf2 = ControlFrame(block_index=1)
    cf2.add_note_off(note=60)
    output2 = engine.step_block(cf2)

    print(f"After NOTE_OFF: {len(output2.render_batch)} voices")
    # Voice enters release phase but is still active
    # (it will eventually become FREE)


def test_render_batch_item_fields():
    """Test RenderBatchItem has correct fields."""
    engine = VoiceEngine(max_voices=4)

    vid = engine.allocate_voice()
    gesture = GestureBasis(gate=True, pressure_q15=Q15_ONE)
    engine.note_on(vid, gesture=gesture)

    output = engine.step_block()

    assert len(output.render_batch) == 1
    item = output.render_batch[0]

    print(f"RenderBatchItem fields:")
    print(f"  voice_id: {item.voice_id}")
    print(f"  mid_due: {item.mid_due}")
    print(f"  top_due: {item.top_due}")
    print(f"  degrade_level: {item.degrade_level}")
    print(f"  frame type: {type(item.frame).__name__}")

    assert item.voice_id == vid
    assert item.mid_due == True
    assert item.top_due == True
    assert item.degrade_level == 0


if __name__ == "__main__":
    print("=" * 60)
    print("Testing VoiceEngine.step_block() API")
    print("=" * 60)

    print("\n1. Test step_block without ControlFrame")
    print("-" * 40)
    test_step_block_without_control_frame()
    print("PASS")

    print("\n2. Test step_block with ControlFrame")
    print("-" * 40)
    test_step_block_with_control_frame()
    print("PASS")

    print("\n3. Test NOTE_ON event")
    print("-" * 40)
    test_step_block_note_on_event()
    print("PASS")

    print("\n4. Test NOTE_OFF event")
    print("-" * 40)
    test_step_block_note_off_event()
    print("PASS")

    print("\n5. Test RenderBatchItem fields")
    print("-" * 40)
    test_render_batch_item_fields()
    print("PASS")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
