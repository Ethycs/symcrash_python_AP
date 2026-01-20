#!/usr/bin/env python3
"""
Test LFO parsing and modulation.
"""

from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from symcrash_python_ap.library.preset_compiler import PresetCompiler

def main():
    # Test LFO parsing
    compiler = PresetCompiler()
    preset_path = Path(__file__).parent.parent / "presets" / "instruments" / "pwm_string_v0.yaml"

    print(f"Compiling: {preset_path}")
    preset = compiler.compile_file(str(preset_path))

    print(f"LFOs found: {len(preset.lfos)}")

    for lfo in preset.lfos:
        print(f"  - {lfo.id}: {lfo.waveform.name}, {lfo.rate_hz:.1f}Hz, depth={lfo.depth_q15}")
        print(f"    sync_mode={lfo.sync_mode.name}, delay={lfo.delay_samples} samples, ramp={lfo.ramp_samples} samples")
        for routing in lfo.routings:
            print(f"    -> {routing.target.name}: amount={routing.amount_q15}")

    print(f"\nPWM width: {preset.pwm_width_q15}")
    print(f"Osc mode: {preset.osc_mode}")

    # Test LFO ticking
    print("\n--- LFO tick test ---")
    for lfo in preset.lfos:
        print(f"\n{lfo.id}:")
        for block in range(10):
            val = lfo.get_lfo_value()
            lfo.advance_block()
            print(f"  block {block}: lfo_value = {val}")

    print("\n--- LFO test passed! ---")

if __name__ == "__main__":
    main()
