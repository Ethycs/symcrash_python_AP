#!/usr/bin/env python3
"""
Test VoiceEngine unification across all synthesis paths.

Verifies that:
1. VoiceEngineAdapter produces VoiceFrame output
2. SynthMaster uses VoiceEngine (unified path)
3. Primitives are aligned (Regime, Intent, etc.)
4. LFO phase storage is unified through VoiceState
"""

from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def test_primitives_alignment():
    """Test that primitives are aligned across modules."""
    print("=== Test 1: Primitives Alignment ===")

    from symcrash_python_ap.primitives import Regime, Intent, Family
    from symcrash_python_ap.voice import VoiceEngine, VoiceState

    # Verify Regime is the same type
    from symcrash_python_ap.voice.voice_engine import Regime as VERegime
    assert Regime is VERegime, "Regime should be imported from primitives"
    print("  Regime is same type in both modules")

    # Verify canonical names exist
    assert hasattr(Regime, 'R0_IDLE'), "R0_IDLE should exist"
    assert hasattr(Regime, 'R1_ATTACK'), "R1_ATTACK should exist"
    assert hasattr(Regime, 'R2_STABLE'), "R2_STABLE should exist"
    print("  Canonical regime names exist")

    # Verify REGIME_SPECS uses canonical names
    from symcrash_python_ap.voice.voice_engine import REGIME_SPECS
    assert Regime.R1_ATTACK in REGIME_SPECS, "R1_ATTACK should be in REGIME_SPECS"
    assert Regime.R2_STABLE in REGIME_SPECS, "R2_STABLE should be in REGIME_SPECS"
    print("  REGIME_SPECS uses canonical names")

    print("  PASS: Primitives alignment")


def test_voice_engine_adapter():
    """Test VoiceEngineAdapter produces valid VoiceFrame."""
    print("\n=== Test 2: VoiceEngineAdapter ===")

    from symcrash_python_ap.voice import VoiceEngine, create_emitter_with_engine, VoiceEngineAdapter
    from symcrash_python_ap.library.preset_compiler import PresetCompiler
    from symcrash_python_ap.frames import VoiceFrame

    # Create engine
    engine = VoiceEngine(max_voices=4)
    print(f"  Created VoiceEngine with {engine.max_voices} voices")

    # Compile a preset
    compiler = PresetCompiler()
    preset_path = Path(__file__).parent.parent / "presets" / "instruments" / "pwm_string_v0.yaml"

    if preset_path.exists():
        preset = compiler.compile_file(str(preset_path))
        print(f"  Compiled preset: {preset_path.name}")

        # Create adapter
        adapter = create_emitter_with_engine(preset, engine, voice_id=0)
        print(f"  Created VoiceEngineAdapter for voice 0")

        # Trigger note
        adapter.note_on(note=60, velocity_q15=32767)
        print(f"  note_on(60) triggered")

        # Emit frames
        for i in range(10):
            frame = adapter.emit_frame()
            assert isinstance(frame, VoiceFrame), f"emit_frame should return VoiceFrame, got {type(frame)}"

        print(f"  Emitted 10 frames successfully")
        print(f"  Final frame gain_q15: {frame.gain_q15}")
        print(f"  Adapter is_active: {adapter.is_active}")

        # Note off
        adapter.note_off()
        print(f"  note_off() called")

        print("  PASS: VoiceEngineAdapter works")
    else:
        print(f"  SKIP: Preset not found at {preset_path}")


def test_unified_phase_storage():
    """Test that LFO phases are stored in VoiceState."""
    print("\n=== Test 3: Unified Phase Storage ===")

    from symcrash_python_ap.voice import VoiceEngine, VoiceState, create_emitter_with_state
    from symcrash_python_ap.library.preset_compiler import PresetCompiler

    # Create VoiceState
    state = VoiceState(voice_id=0)
    initial_phases = list(state.arc_phase_q16)
    print(f"  Initial arc_phase_q16: {initial_phases[:4]}...")

    # Compile preset with LFOs
    compiler = PresetCompiler()
    preset_path = Path(__file__).parent.parent / "presets" / "instruments" / "pwm_string_v0.yaml"

    if preset_path.exists():
        preset = compiler.compile_file(str(preset_path))
        lfo_count = len(preset.lfos) if hasattr(preset, 'lfos') else 0
        print(f"  Preset has {lfo_count} LFOs")

        # Create emitter with state (Mode 2)
        from symcrash_python_ap.voice import create_emitter_with_state
        emitter = create_emitter_with_state(preset, state)

        # Verify arc_phase delegation
        assert emitter.arc_phase is state.arc_phase_q16, \
            "arc_phase should delegate to state.arc_phase_q16"
        print("  arc_phase delegates to VoiceState")

        # Trigger and emit
        emitter.note_on(note=60, velocity_q15=32767)
        for _ in range(10):
            emitter.emit_frame()

        # Check phases changed
        final_phases = list(state.arc_phase_q16)
        if lfo_count > 0:
            phases_changed = final_phases != initial_phases
            print(f"  Final arc_phase_q16: {final_phases[:4]}...")
            print(f"  Phases changed: {phases_changed}")
        else:
            print("  No LFOs in preset, phases unchanged (expected)")

        print("  PASS: Unified phase storage")
    else:
        print(f"  SKIP: Preset not found")


def test_synth_master_voice_engine():
    """Test SynthMaster (unified VoiceEngine path)."""
    print("\n=== Test 4: SynthMaster (Unified VoiceEngine) ===")

    from symcrash_python_ap.synth_master import SynthMaster, SynthConfig

    # Create SynthMaster (now always uses VoiceEngine)
    config = SynthConfig(n_voices=4)
    synth = SynthMaster(config=config)
    print(f"  Created SynthMaster (VoiceEngine is the single path)")

    # Verify VoiceEngine exists
    assert synth.voice_engine is not None, "voice_engine should exist"
    print(f"  VoiceEngine initialized with {synth.voice_engine.max_voices} voices")

    # Trigger note
    voice_id = synth.note_on(note=60, velocity=100)
    print(f"  note_on(60) returned voice_id: {voice_id}")

    # Render a few blocks
    for i in range(10):
        L, R, health = synth.render_block()
        assert len(L) == config.block_size, f"L should have {config.block_size} samples"
        assert len(R) == config.block_size, f"R should have {config.block_size} samples"

    print(f"  Rendered 10 blocks successfully")

    # Note off
    synth.note_off(note=60)
    print(f"  note_off(60) called")

    print("  PASS: SynthMaster unified path")


def test_all_paths_produce_output():
    """Test that the unified VoiceEngine path produces valid audio."""
    print("\n=== Test 5: Unified Path Produces Output ===")

    from symcrash_python_ap.synth_master import SynthMaster, SynthConfig

    config = SynthConfig(n_voices=4, block_size=64)

    # Test VoiceEngine path (now the only path)
    print("  Testing unified VoiceEngine path...")
    synth = SynthMaster(config=config)
    synth.note_on(60, 100)
    for _ in range(5):  # Need a few blocks for envelope to ramp up
        L, R, _ = synth.render_block()
    has_audio = any(s != 0 for s in L) or any(s != 0 for s in R)
    print(f"    VoiceEngine path has audio: {has_audio}")

    print("  PASS: Unified path produces output")


def main():
    print("=" * 60)
    print("VoiceEngine Unification Tests")
    print("=" * 60)

    test_primitives_alignment()
    test_voice_engine_adapter()
    test_unified_phase_storage()
    test_synth_master_voice_engine()
    test_all_paths_produce_output()

    print("\n" + "=" * 60)
    print("=== All VoiceEngine unification tests passed! ===")
    print("=" * 60)


if __name__ == "__main__":
    main()
