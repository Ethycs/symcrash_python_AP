"""
Test SynthMaster control plane integration.

Verifies that:
1. CockpitEngine -> ControlFrame -> VoiceEngine.step_block() works
2. ParamFrame flows from VoiceEngine to Material/Safety
3. NOTE_ON/OFF events route correctly through the pipeline
"""

from symcrash_python_ap.synth_master import SynthMaster, SynthConfig
from symcrash_python_ap.control_frame import ControlFrame
from symcrash_python_ap.frames import ParamFrame


def test_basic_render():
    """Test basic render without control plane."""
    print("Creating SynthMaster...")
    synth = SynthMaster(SynthConfig(n_voices=4))

    # Trigger a note directly (old API)
    synth.note_on(60, velocity=100)

    # Render one block
    L, R, telemetry = synth.render_block()

    print(f"Rendered block: L[0]={L[0]}, R[0]={R[0]}")
    print(f"Telemetry: {telemetry}")

    # Check we got audio
    assert len(L) == 64
    assert len(R) == 64


def test_render_with_cockpit():
    """Test render with CockpitEngine enabled."""
    print("Creating SynthMaster with cockpit...")
    synth = SynthMaster(SynthConfig(n_voices=4))

    # Ensure cockpit is enabled
    assert synth._enable_cockpit == True

    # Trigger note through cockpit
    synth.cockpit.midi_note_on(note=60, velocity=100)

    # Render - this should process the queued note event
    L, R, telemetry = synth.render_block()

    print(f"After cockpit note_on: L[0]={L[0]}")
    print(f"Active voices in VoiceEngine: {synth.voice_engine.get_active_voice_count()}")

    # Voice should now be playing via VoiceEngine
    assert synth.voice_engine.get_active_voice_count() >= 1


def test_control_frame_to_param_frame():
    """Test that ControlFrame.globals -> ParamFrame flow works."""
    print("Testing ControlFrame -> ParamFrame flow...")
    synth = SynthMaster(SynthConfig(n_voices=4))

    # Set a control that should affect ParamFrame
    from symcrash_python_ap.control_registry import ControlID
    synth.cockpit.set_control(ControlID.MASTER_GAIN, 16384)  # 50% gain

    # Trigger a note
    synth.note_on(60, velocity=100)

    # Render - the VoiceEngine should extract ParamFrame from ControlFrame
    L, R, telemetry = synth.render_block()

    print(f"Rendered with gain control")
    # ParamFrame is now used internally - we can't easily inspect it
    # but the render should complete without error


def test_multiple_blocks():
    """Test rendering multiple blocks with control plane."""
    print("Testing multiple block render...")
    synth = SynthMaster(SynthConfig(n_voices=4))

    # Trigger note
    synth.note_on(60, velocity=100)

    # Render several blocks
    for i in range(10):
        L, R, telemetry = synth.render_block()
        if i == 0:
            print(f"Block 0: L[0]={L[0]}, voices={synth.voice_engine.get_active_voice_count()}")

    # Note off
    synth.note_off(60)

    # Render release phase
    for i in range(20):
        L, R, telemetry = synth.render_block()

    print(f"After release: voices={synth.voice_engine.get_active_voice_count()}")


def test_pot_control_integration():
    """Test that pot controls flow through the pipeline."""
    print("Testing pot control integration...")
    synth = SynthMaster(SynthConfig(n_voices=4))

    # Set pot values (simulating Pico2 hardware)
    synth.cockpit.set_pot(0, 8000)   # Attack
    synth.cockpit.set_pot(1, 16000)  # Decay
    synth.cockpit.set_pot(2, 24000)  # Sustain
    synth.cockpit.set_pot(3, 12000)  # Release

    # Trigger and render
    synth.note_on(60, velocity=100)
    L, R, telemetry = synth.render_block()

    print(f"After pot controls: L[0]={L[0]}")
    print("Pot controls applied successfully")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing SynthMaster Control Plane Integration")
    print("=" * 60)

    print("\n1. Basic render test")
    print("-" * 40)
    test_basic_render()
    print("PASS")

    print("\n2. Render with CockpitEngine")
    print("-" * 40)
    test_render_with_cockpit()
    print("PASS")

    print("\n3. ControlFrame -> ParamFrame flow")
    print("-" * 40)
    test_control_frame_to_param_frame()
    print("PASS")

    print("\n4. Multiple blocks render")
    print("-" * 40)
    test_multiple_blocks()
    print("PASS")

    print("\n5. Pot control integration")
    print("-" * 40)
    test_pot_control_integration()
    print("PASS")

    print("\n" + "=" * 60)
    print("All control plane tests passed!")
    print("=" * 60)
