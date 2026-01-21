#!/usr/bin/env python3
"""Test FM preset compilation to identify int16 overflow issues."""

from symcrash_python_ap.library.compiler import CapsuleCompiler
from symcrash_python_ap.library.preset_data import load_preset
import traceback

# List FM presets
fm_presets = [
    'fm_bell_v0',
    'fm_brass_v0',
    'fm_horn_v0',
    'fm_trumpet_v0',
]

compiler = CapsuleCompiler()

for preset_name in fm_presets:
    try:
        ir = load_preset(preset_name)
        compiled = compiler.compile(ir)
        print(f'{preset_name}: OK')
    except Exception as e:
        print(f'{preset_name}: FAILED - {e}')
        traceback.print_exc()
