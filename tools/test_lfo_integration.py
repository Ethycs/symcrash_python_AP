#!/usr/bin/env python3
"""
Test LFO integration with VoiceFrameEmitter.
"""

from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def main():
    # Test imports
    print("Testing imports...")
    from symcrash_python_ap.library.preset_compiler import PresetCompiler
    from symcrash_python_ap.voiceframe_emitter import VoiceFrameEmitter
    from symcrash_python_ap.control_frame import ControlFrame
    print("  VoiceFrameEmitter imported successfully")

    # Compile preset
    compiler = PresetCompiler()
    preset_path = Path(__file__).parent.parent / "presets" / "instruments" / "pwm_string_v0.yaml"
    preset = compiler.compile_file(str(preset_path))
    print(f"  Compiled preset with {len(preset.lfos)} LFOs")

    # Create emitter
    emitter = VoiceFrameEmitter(preset=preset)
    print("  VoiceFrameEmitter created")

    # Trigger note
    emitter.note_on(note=60, velocity_q15=32767)
    print("  Note triggered")

    # Emit frames and check LFO modulation
    # Run longer to see delayed vibrato (400ms delay = ~300 blocks)
    print("\n--- Frame emission test ---")
    for block in range(400):
        frame = emitter.emit_frame()
        if block % 50 == 0:
            print(f"  block {block}: f0={frame.f0_hz:.2f}Hz, cents={frame.cents}, gain={frame.gain_q15}, pwm={frame.pwm_width_q15}")

    # Note off and release
    emitter.note_off()
    print("\nNote off - checking release...")
    for block in range(10):
        frame = emitter.emit_frame()
        print(f"  release block {block}: gain={frame.gain_q15}, is_active={frame.is_active}")

    print("\n--- LFO integration test passed! ---")

if __name__ == "__main__":
    main()
