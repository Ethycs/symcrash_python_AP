#!/usr/bin/env python3
"""Test Karplus-Strong material program."""

from symcrash_python_ap.material.material_engine import MaterialEngine, KarplusStrongState
from symcrash_python_ap.frames import ParamFrame

# Quick smoke test
engine = MaterialEngine()
L = [1000] * 64
R = [1000] * 64
params = ParamFrame()

# Test each material_id
for mid in range(11):
    try:
        outL, outR = engine.step_block(L, R, params, material_id=mid)
        print(f'material_id={mid}: OK (len={len(outL)})')
    except Exception as e:
        print(f'material_id={mid}: FAILED - {e}')

# Test KS pitch and excite
engine.set_karplus_pitch(440.0)
print(f'KS delay for 440Hz: {engine.ks_L.delay_samples} samples')
engine.excite_karplus(16000)
print('KS excite: OK')
