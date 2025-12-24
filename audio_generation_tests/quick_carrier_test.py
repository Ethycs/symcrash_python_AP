"""Quick carrier lift test to verify fixes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "symcrash_python_ap"))

import numpy as np
from test_carrier_lift_audio import *


def test_config(name, note, beta, duration_sec=1.0):
    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=note)

    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    samples = []
    for _ in range(int(duration_sec * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=note,
            f0_hz=midi_to_freq(note),
            gain_q15=Q15_HALF,
            beta_target_q15=beta,
        )
        L, R = pipeline.render_block(frame)
        samples.extend(L)

    pipeline.voice_off(0)

    arr = np.array(samples, dtype=np.float64)
    deltas = np.diff(arr)
    d2 = np.diff(deltas)

    print(f'{name}:')
    print(f'  Peak: {int(np.max(np.abs(arr)))}, Max d2: {int(np.max(np.abs(d2)))}')
    print(f'  Spikes: >500={np.sum(np.abs(d2) > 500)}, >1k={np.sum(np.abs(d2) > 1000)}, >3k={np.sum(np.abs(d2) > 3000)}')


print('=' * 60)
print('CARRIER LIFT SPIKE TEST')
print('=' * 60)

# Test DRY only (beta=0)
test_config('C4 DRY (beta=0)', 60, 0)
test_config('A1 DRY (beta=0)', 33, 0)
test_config('C7 DRY (beta=0)', 96, 0)

# Test with wet blend (beta=75%)
test_config('C4 WET (beta=75%)', 60, 24576)
test_config('A1 WET (beta=75%)', 33, 24576)
test_config('C7 WET (beta=75%)', 96, 24576)
