import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.frames import VoiceFrame
import numpy as np

print('Testing CCS with different base_mix values\n')

for base_mix in [0, 16384, 32767]:
    print(f'Testing base_mix_q15 = {base_mix}...')

    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=60, cents=0)

    frame = VoiceFrame(
        voice_id=0,
        note=60,
        cents=0,
        gain_q15=32767,  # Full volume
        beta_target_q15=base_mix,  # Used as base_mix
    )

    state, output = vm_step_block(state, frame, DEFAULT_TABLES)

    print(f'  Output range: {output.min()} to {output.max()}')
    print(f'  Output[0:5]: {output[0:5]}')
    print(f'  RMS: {np.sqrt(np.mean(output.astype(np.float32)**2)):.1f}')
    print()
