import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.vm.decode import decode_frame, DecodedPlan
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.vm.executor import vm_step_block_raw
from symcrash_python_ap.vm.coeff_osc import coeff_osc_generate_coeffs
from symcrash_python_ap.vm.coeff_tables_fast import get_fast_coeff_table
from symcrash_python_ap.vm.lanes import inject_64_to_256, extract_64_leaves, lane_tail
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.frames import VoiceFrame
import numpy as np

print('Testing coefficient evolution WITHOUT operators\n')

state = VMState.create(0)
state.reset_for_note_on(60, 0)

table = get_fast_coeff_table()
carrier_64 = np.zeros(64, dtype=np.int16)

for block in range(5):
    print(f'Block {block}:')

    # Generate carrier
    state.coeff_osc.table_id = 60
    state.coeff_osc.base_phase_inc = int((261.63 / 48000) * (1 << 32))
    coeff_osc_generate_coeffs(state.coeff_osc, table, carrier_64)

    print(f'  Carrier: min={carrier_64.min()}, max={carrier_64.max()}')

    # Inject (full replacement for now)
    inject_64_to_256(carrier_64, state.coeff_state)

    print(f'  After inject: min={state.coeff_state.min()}, max={state.coeff_state.max()}')

    # NO OPERATORS - just extract and IDWPT
    coeffs_64 = extract_64_leaves(state.coeff_state)
    audio = idwpt_jit(coeffs_64, headroom_bits=1)
    audio_arr = np.array(audio, dtype=np.int16)

    print(f'  Output: min={audio_arr.min()}, max={audio_arr.max()}, RMS={np.sqrt(np.mean(audio_arr.astype(np.float32)**2)):.1f}')
    print(f'  First 5: {audio_arr[:5]}')

    state.blocks_since_start += 1
    print()
