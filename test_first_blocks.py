import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.frames import VoiceFrame
import numpy as np

print('Testing first 5 blocks in detail\n')

state = VMState.create(voice_id=0)
state.reset_for_note_on(note=60, cents=0)

frame = VoiceFrame(
    voice_id=0,
    note=60,
    gain_q15=32767,
    beta_target_q15=32767,
)

for block in range(5):
    print(f'Block {block} (blocks_since_start={state.blocks_since_start}):')

    # Check coeff_state before
    print(f'  coeff_state before: min={state.coeff_state.min()}, max={state.coeff_state.max()}, mean={state.coeff_state.mean():.1f}')

    state, output = vm_step_block(state, frame, DEFAULT_TABLES)

    # Check output
    print(f'  Output: min={output.min()}, max={output.max()}, mean={output.mean():.1f}')
    print(f'  First 5 samples: {output[:5]}')
    print()
