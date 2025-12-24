"""Test carrier directly."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "symcrash_python_ap"))

import numpy as np
from symcrash_python_ap.carrier import CarrierOscillator
from symcrash_python_ap.carrier_lift import Q15_ONE, Q15_HALF, BLOCK_SIZE

carrier = CarrierOscillator()
voice_id = 0
carrier.configure_voice(voice_id, note=60)

# Set pure sine
state = carrier.get_or_create_state(voice_id)
state.harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

samples = []
for _ in range(375):
    block = carrier.render_block(voice_id, BLOCK_SIZE)
    samples.extend(block)

arr = np.array(samples, dtype=np.float64)
d2 = np.diff(np.diff(arr))
print(f"Peak: {int(np.max(np.abs(arr)))}, Max d2: {int(np.max(np.abs(d2)))}")
print(f"Spikes: >500={np.sum(np.abs(d2) > 500)}, >1k={np.sum(np.abs(d2) > 1000)}")
