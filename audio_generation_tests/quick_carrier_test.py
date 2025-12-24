"""Quick carrier lift test to verify fixes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "symcrash_python_ap"))

import numpy as np
from test_carrier_lift_audio import *

# Quick test - just run the steady C4 test
pipeline = CarrierLiftPipeline()
pipeline.voice_on(0, note=60)

if 0 in pipeline.carrier.states:
    pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

samples = []
for _ in range(int(1.0 * BLOCKS_PER_SEC)):
    frame = VoiceFrame(
        voice_id=0,
        note=60,
        f0_hz=midi_to_freq(60),
        gain_q15=Q15_HALF,
        beta_target_q15=0,  # DRY only
    )
    L, R = pipeline.render_block(frame)
    samples.extend(L)

pipeline.voice_off(0)

arr = np.array(samples, dtype=np.float64)
deltas = np.diff(arr)
d2 = np.diff(deltas)

print('Carrier Lift Test (C4, 1 second, pure sine):')
print(f'  Samples: {len(samples)}')
print(f'  Peak: {int(np.max(np.abs(arr)))}')
print(f'  Max delta: {int(np.max(np.abs(deltas)))}')
print(f'  Max d2: {int(np.max(np.abs(d2)))}')
print(f'  Spikes (d2>500): {np.sum(np.abs(d2) > 500)}')
print(f'  Spikes (d2>1000): {np.sum(np.abs(d2) > 1000)}')
print(f'  Spikes (d2>3000): {np.sum(np.abs(d2) > 3000)}')
