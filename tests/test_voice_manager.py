"""
CCS Conformance Test: Voice Manager (32-Voice Polyphony)
========================================================

Validates the voice allocation, stealing, and mixing logic.

Tests:
1. Basic voice allocation
2. Voice stealing (oldest-first)
3. Note release
4. Mix output
5. Mono mode
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from symcrash_python_ap.vm.voice_manager import (
    VoicePool, MonoVoicePool, VoiceSlot, MAX_VOICES
)
from symcrash_python_ap.vm.state import EnvelopePhase


def test_basic_allocation():
    """Test basic voice allocation."""
    print("\n=== Basic Allocation Test ===\n")

    pool = VoicePool(num_voices=8)

    # Allocate 4 voices
    slots = []
    for note in [60, 62, 64, 65]:
        slot = pool.allocate_voice(note)
        assert slot is not None, f"Failed to allocate voice for note {note}"
        slots.append(slot)

    # Check allocation
    assert pool.get_active_count() == 4
    assert pool.get_free_count() == 4

    # Check notes
    for i, note in enumerate([60, 62, 64, 65]):
        assert slots[i].note == note, f"Slot {i} has wrong note"

    print(f"Allocated 4 voices: {[s.note for s in slots]}")
    print(f"Active: {pool.get_active_count()}, Free: {pool.get_free_count()}")
    print("[PASS] Basic allocation works")
    return True


def test_voice_stealing():
    """Test oldest-first voice stealing."""
    print("\n=== Voice Stealing Test ===\n")

    pool = VoicePool(num_voices=4)

    # Fill all slots
    for note in [60, 62, 64, 65]:
        slot = pool.allocate_voice(note)
        slot.state.reset_for_note_on(note)
        slot.state.envelope.phase = EnvelopePhase.SUSTAIN
        pool.block_counter += 10  # Simulate time passing

    print(f"Filled pool with notes: {[s.note for s in pool.slots]}")
    print(f"Allocation times: {[s.allocation_time for s in pool.slots]}")

    # Allocate one more - should steal oldest (note 60)
    new_slot = pool.allocate_voice(67)
    assert new_slot is not None, "Failed to allocate voice via stealing"
    assert new_slot.note == 67, f"Stolen slot has wrong note: {new_slot.note}"

    # The oldest voice (note 60) should have been stolen
    print(f"After stealing: {[s.note for s in pool.slots]}")
    assert 60 not in [s.note for s in pool.slots], "Oldest voice was not stolen"

    print("[PASS] Voice stealing works (oldest-first)")
    return True


def test_release_note():
    """Test note release."""
    print("\n=== Note Release Test ===\n")

    pool = VoicePool(num_voices=8)

    # Allocate voices
    for note in [60, 62, 64]:
        slot = pool.allocate_voice(note)
        slot.state.reset_for_note_on(note)

    # Release one note
    released = pool.release_note(62)
    assert len(released) == 1, f"Expected 1 released voice, got {len(released)}"
    assert released[0].note == 62

    # Check release state
    assert released[0].state.envelope.phase == EnvelopePhase.RELEASE

    print(f"Released note 62, envelope phase: {released[0].state.envelope.phase.name}")
    print("[PASS] Note release works")
    return True


def test_mix_outputs():
    """Test output mixing."""
    print("\n=== Mix Outputs Test ===\n")

    pool = VoicePool(num_voices=8)

    # Create test audio buffers
    audio1 = np.full(64, 10000, dtype=np.int16)
    audio2 = np.full(64, 15000, dtype=np.int16)
    audio3 = np.full(64, 5000, dtype=np.int16)

    # Mix
    outputs = [
        (0, audio1),
        (1, audio2),
        (2, audio3),
    ]
    mixed = pool.mix_outputs(outputs)

    # Check sum
    expected_sum = 10000 + 15000 + 5000
    assert np.all(mixed == expected_sum), f"Mix error: expected {expected_sum}, got {mixed[0]}"

    print(f"Mixed 3 voices: {mixed[0]} (expected {expected_sum})")
    print("[PASS] Output mixing works")
    return True


def test_mix_clipping():
    """Test that mixing clips properly."""
    print("\n=== Mix Clipping Test ===\n")

    pool = VoicePool(num_voices=8)

    # Create loud audio that would overflow
    audio1 = np.full(64, 30000, dtype=np.int16)
    audio2 = np.full(64, 30000, dtype=np.int16)

    outputs = [
        (0, audio1),
        (1, audio2),
    ]
    mixed = pool.mix_outputs(outputs)

    # Should be clipped to 32767
    assert mixed[0] == 32767, f"Expected clipping to 32767, got {mixed[0]}"

    print(f"Mixed 2 loud voices: {mixed[0]} (clipped from {30000 + 30000})")
    print("[PASS] Clipping works")
    return True


def test_mono_mode():
    """Test mono mode voice allocation."""
    print("\n=== Mono Mode Test ===\n")

    pool = MonoVoicePool(num_channels=16, legato=True)

    # Allocate on channel 0
    slot1 = pool.allocate_voice(60, channel=0)
    slot1.state.reset_for_note_on(60)
    slot1.state.envelope.phase = EnvelopePhase.SUSTAIN
    assert slot1 is not None
    assert slot1.note == 60

    # Allocate again on channel 0 (should reuse same slot)
    slot2 = pool.allocate_voice(64, channel=0)
    assert slot2 is not None
    assert slot2.voice_id == slot1.voice_id, "Mono mode should reuse same slot"
    assert slot2.note == 64, "Note should be updated"

    print(f"Channel 0: first note 60, second note 64")
    print(f"Same slot reused: {slot1.voice_id == slot2.voice_id}")
    print("[PASS] Mono mode works")
    return True


def test_full_polyphony():
    """Test full 32-voice polyphony."""
    print("\n=== Full Polyphony Test ===\n")

    pool = VoicePool(num_voices=32)

    # Allocate all 32 voices
    for i in range(32):
        note = 36 + i  # C2 to G4
        slot = pool.allocate_voice(note)
        assert slot is not None, f"Failed to allocate voice {i}"
        slot.state.reset_for_note_on(note)
        slot.state.envelope.phase = EnvelopePhase.SUSTAIN

    assert pool.get_active_count() == 32
    assert pool.get_free_count() == 0

    print(f"Allocated all 32 voices")
    print(f"Notes: {[s.note for s in pool.slots[:8]]}... (first 8)")

    # Steal one more
    slot33 = pool.allocate_voice(100)
    assert slot33 is not None, "Voice stealing failed at 33rd voice"
    assert pool.get_active_count() == 32, "Should still have 32 voices"

    print(f"Stole oldest voice for note 100")
    print("[PASS] Full 32-voice polyphony works")
    return True


def test_update_slot_states():
    """Test slot state updates after rendering."""
    print("\n=== Slot State Update Test ===\n")

    pool = VoicePool(num_voices=4)

    # Allocate and set to FREE
    slot = pool.allocate_voice(60)
    slot.state.envelope.phase = EnvelopePhase.FREE

    assert pool.get_active_count() == 1, "Voice should be active before update"

    # Update states
    pool.update_slot_states()

    assert pool.get_active_count() == 0, "Voice should be freed after update"
    assert slot.is_active == False
    assert slot.note == -1

    print("[PASS] Slot state updates work")
    return True


def main():
    print("=" * 60)
    print("CCS Conformance: Voice Manager (32-Voice Polyphony)")
    print("=" * 60)

    results = []

    results.append(("Basic Allocation", test_basic_allocation()))
    results.append(("Voice Stealing", test_voice_stealing()))
    results.append(("Note Release", test_release_note()))
    results.append(("Mix Outputs", test_mix_outputs()))
    results.append(("Mix Clipping", test_mix_clipping()))
    results.append(("Mono Mode", test_mono_mode()))
    results.append(("Full Polyphony", test_full_polyphony()))
    results.append(("Slot State Update", test_update_slot_states()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        all_passed = all_passed and passed

    print()
    if all_passed:
        print("[PASS] All voice manager tests passed!")
        return 0
    else:
        print("[FAIL] Some voice manager tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
