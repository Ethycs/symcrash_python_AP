"""
Integration Test: CockpitEngine → ControlFrame → VoiceFrameEmitter
===================================================================

Tests the control flow from CockpitEngine through ControlFrame.
"""

from symcrash_python_ap.cockpit_engine import CockpitEngine
from symcrash_python_ap.control_registry import ControlID


def test_basic_control_frame():
    """Test basic ControlFrame emission."""
    print("=" * 70)
    print("Test 1: Basic ControlFrame Emission")
    print("=" * 70)

    # Create CockpitEngine
    cockpit = CockpitEngine()
    print("[OK] CockpitEngine created")

    # Set some controls
    cockpit.set_control(ControlID.TIMBRE, 16384)  # 50%
    cockpit.set_control(ControlID.DYNAMICS, 24576)  # 75%
    cockpit.set_control(ControlID.CHAOS, 8192)  # 25%
    print("[OK] Controls set: TIMBRE=50%, DYNAMICS=75%, CHAOS=25%")

    # Send a note-on event
    cockpit.midi_note_on(note=60, velocity=100)
    print("[OK] MIDI note-on: C4 (60), velocity=100")

    # Tick the cockpit
    control_frame = cockpit.tick()
    print(f"[OK] ControlFrame emitted: block_index={control_frame.block_index}")
    print(f"    - Events: {len(control_frame.events)}")
    print(f"    - Voice patches: {len(control_frame.voice_patches)}")

    # Verify the control frame
    assert len(control_frame.events) == 1, "Expected 1 event (note-on)"
    assert len(control_frame.voice_patches) >= 1, "Expected at least 1 voice patch"
    print("[OK] ControlFrame structure validated")

    print()


def test_control_slew():
    """Test control slew filtering."""
    print("=" * 70)
    print("Test 2: Control Slew Filtering")
    print("=" * 70)

    cockpit = CockpitEngine()

    # Set TIMBRE (ONE_POLE slew, alpha=2048 → ~16 blocks to settle)
    cockpit.set_control(ControlID.TIMBRE, 32767)  # Max
    print(f"[OK] TIMBRE set to 32767 (max)")

    # Tick 20 times and observe slew
    print("[OK] Ticking 20 blocks to observe ONE_POLE slew...")
    for i in range(20):
        frame = cockpit.tick()
        current = cockpit.get_control_current(ControlID.TIMBRE)
        target = cockpit.get_control_target(ControlID.TIMBRE)
        print(f"    Block {i:2d}: current={current:5d}, target={target:5d}")
        if current == target:
            print(f"    [OK] Settled at block {i}")
            break

    print()


def test_adsr_controls():
    """Test ADSR parameter controls."""
    print("=" * 70)
    print("Test 3: ADSR Controls")
    print("=" * 70)

    cockpit = CockpitEngine()

    # Get default ADSR
    default_adsr = cockpit.get_adsr()
    print(f"[OK] Default ADSR:")
    print(f"    - attack_blocks: {default_adsr['attack_blocks']}")
    print(f"    - decay_blocks: {default_adsr['decay_blocks']}")
    print(f"    - sustain_level_q15: {default_adsr['sustain_level_q15']}")
    print(f"    - release_blocks: {default_adsr['release_blocks']}")

    # Override attack
    cockpit.set_control(ControlID.ATK_BLOCKS, 100)  # Slow attack
    print(f"[OK] ATK_BLOCKS set to 100")

    # Tick multiple times to let slew settle (NBLOCK policy with param=4)
    # With step size ceiling((100-8)/4) = 23, should converge in ~4 steps
    for i in range(20):
        control_frame = cockpit.tick()
        updated_adsr = cockpit.get_adsr()
        if updated_adsr['attack_blocks'] == 100:
            print(f"[OK] ADSR slewed to target in {i+1} blocks")
            break

    # Check if ADSR reached target
    final_adsr = cockpit.get_adsr()
    print(f"[OK] Final ADSR:")
    print(f"    - attack_blocks: {final_adsr['attack_blocks']}")

    assert final_adsr['attack_blocks'] == 100, f"ADSR didn't reach target (got {final_adsr['attack_blocks']})"
    print(f"[OK] ADSR control confirmed")

    print()


def test_pitch_offset():
    """Test pitch offset (DETUNE + PITCH_BEND)."""
    print("=" * 70)
    print("Test 4: Pitch Offset (Cumulative)")
    print("=" * 70)

    cockpit = CockpitEngine()

    # Set DETUNE to +100 cents (1 semitone)
    cockpit.set_control(ControlID.DETUNE, 100)
    print(f"[OK] DETUNE set to +100 cents")

    # Set PITCH_BEND to +50 cents
    cockpit.set_control(ControlID.PITCH_BEND, 50)
    print(f"[OK] PITCH_BEND set to +50 cents")

    # Tick to apply controls (both are INSTANT slew so should be immediate)
    cockpit.tick()

    # Check cumulative offset
    offset = cockpit.get_pitch_offset_cents()
    print(f"[OK] Total pitch offset: {offset} cents")
    assert offset == 150, f"Expected 150 cents, got {offset}"

    print()


def test_family_controls():
    """Test family control bindings."""
    print("=" * 70)
    print("Test 5: Family Controls")
    print("=" * 70)

    cockpit = CockpitEngine()

    # Set TIMBRE to 50% (binds to COLOR family)
    cockpit.set_control(ControlID.TIMBRE, 16384)
    print(f"[OK] TIMBRE set to 50% (16384)")

    # Tick to apply (ONE_POLE slew with alpha=2048, ~16 block time constant)
    # Going from 32767 to 16384 is a large delta, may take >30 blocks
    for i in range(100):
        cockpit.tick()
        color_scale = cockpit.get_family_scale(0)  # FamilyID.COLOR = 0
        if color_scale == 16384:
            print(f"[OK] COLOR family scale reached target in {i+1} blocks")
            break

    # Get final family scale
    color_scale = cockpit.get_family_scale(0)  # FamilyID.COLOR = 0
    print(f"[OK] COLOR family scale: {color_scale}")

    # Allow tolerance due to one-pole slew quantization (integer arithmetic)
    assert abs(color_scale - 16384) <= 20, f"Expected ~16384 (±20), got {color_scale}"
    print(f"[OK] Family control binding confirmed (within tolerance)")

    print()


def main():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("COCKPIT ENGINE INTEGRATION TESTS")
    print("=" * 70 + "\n")

    try:
        test_basic_control_frame()
        test_control_slew()
        test_adsr_controls()
        test_pitch_offset()
        test_family_controls()

        print("=" * 70)
        print("ALL TESTS PASSED")
        print("=" * 70)

    except Exception as e:
        print(f"\n[FAIL] Test failed with error:")
        print(f"    {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
