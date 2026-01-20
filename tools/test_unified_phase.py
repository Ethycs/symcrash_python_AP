#!/usr/bin/env python3
"""
Test unified phase storage between VoiceFrameEmitter and VoiceState.

Verifies that when an emitter is created with create_emitter_with_state,
the arc_phase property correctly delegates to VoiceState.arc_phase_q16.
"""

from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def main():
    print("Testing unified phase storage...")

    # Test imports
    from symcrash_python_ap.library.preset_compiler import PresetCompiler
    from symcrash_python_ap.voice import VoiceState, create_emitter_with_state, create_standalone_emitter
    print("  Imports successful")

    # Compile preset with LFOs
    compiler = PresetCompiler()
    preset_path = Path(__file__).parent.parent / "presets" / "instruments" / "pwm_string_v0.yaml"
    preset = compiler.compile_file(str(preset_path))
    print(f"  Compiled preset with {len(preset.lfos)} LFOs")

    # --- Test 1: Standalone emitter uses internal storage ---
    print("\n--- Test 1: Standalone emitter ---")
    standalone = create_standalone_emitter(preset)
    standalone.note_on(note=60, velocity_q15=32767)

    # Emit a few frames
    for _ in range(10):
        standalone.emit_frame()

    # Check that internal storage has changed
    has_nonzero = any(p != 0 for p in standalone._arc_phase_q16)
    print(f"  Internal _arc_phase_q16 has non-zero values: {has_nonzero}")
    assert has_nonzero, "Standalone emitter should use internal storage"
    print("  PASS: Standalone emitter uses internal storage")

    # --- Test 2: Emitter with VoiceState uses shared storage ---
    print("\n--- Test 2: Emitter with VoiceState ---")
    voice_state = VoiceState(voice_id=0)

    # Verify initial state
    initial_phases = list(voice_state.arc_phase_q16)
    print(f"  Initial VoiceState.arc_phase_q16: {initial_phases[:4]}...")
    assert all(p == 0 for p in initial_phases), "Initial phases should be zero"

    # Create emitter sharing phase storage
    emitter = create_emitter_with_state(preset, voice_state)
    print(f"  Created emitter with voice_state reference")

    # Verify delegation
    assert emitter.arc_phase is voice_state.arc_phase_q16, \
        "arc_phase should delegate to voice_state.arc_phase_q16"
    print("  PASS: arc_phase delegates to voice_state.arc_phase_q16")

    # Trigger note and emit frames
    emitter.note_on(note=60, velocity_q15=32767)
    for _ in range(10):
        emitter.emit_frame()

    # Check that VoiceState storage was modified
    final_phases = list(voice_state.arc_phase_q16)
    print(f"  Final VoiceState.arc_phase_q16: {final_phases[:4]}...")
    has_changed = final_phases != initial_phases
    print(f"  VoiceState phases changed: {has_changed}")
    assert has_changed, "VoiceState phases should change when emitter ticks LFOs"

    # Check that emitter's internal storage was NOT used
    internal_unchanged = all(p == 0 for p in emitter._arc_phase_q16)
    print(f"  Emitter internal storage unchanged: {internal_unchanged}")
    assert internal_unchanged, "Emitter should NOT use internal storage when voice_state is provided"
    print("  PASS: Emitter uses VoiceState storage, not internal")

    # --- Test 3: VoiceState.lfos field ---
    print("\n--- Test 3: VoiceState.lfos field ---")
    voice_state2 = VoiceState(voice_id=1)
    print(f"  New VoiceState has lfos field: {hasattr(voice_state2, 'lfos')}")
    print(f"  Default lfos is empty list: {voice_state2.lfos == []}")
    assert voice_state2.lfos == [], "Default lfos should be empty list"

    # Set LFOs on VoiceState
    voice_state2.lfos = preset.lfos
    print(f"  Set lfos from preset: {len(voice_state2.lfos)} LFOs")

    # Tick using VoiceState method
    initial_phases2 = list(voice_state2.arc_phase_q16)
    voice_state2.tick_lfos(voice_state2.lfos)
    final_phases2 = list(voice_state2.arc_phase_q16)
    phases_changed = final_phases2 != initial_phases2
    print(f"  tick_lfos() modified phases: {phases_changed}")
    assert phases_changed, "tick_lfos should modify arc_phase_q16"
    print("  PASS: VoiceState.tick_lfos() works correctly")

    print("\n=== All unified phase storage tests passed! ===")

if __name__ == "__main__":
    main()
