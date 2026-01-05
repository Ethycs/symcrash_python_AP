import sys
sys.path.insert(0, 'src')

import numpy as np

# Direct test bypassing full VM
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.vm.coeff_osc import CoeffOscState, coeff_osc_generate_coeffs
from symcrash_python_ap.vm.coeff_tables_fast import get_fast_coeff_table
from symcrash_python_ap.vm.lanes import extract_64_leaves, inject_64_to_256
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit

print('Direct CCS test (bypassing operators)\n')

# Create minimal state
state = VMState.create(0)
state.reset_for_note_on(60, 0)

# Get table
table = get_fast_coeff_table()

# Generate carrier
carrier_64 = np.zeros(64, dtype=np.int16)
coeff_osc_generate_coeffs(state.coeff_osc, table, carrier_64)

print(f'Carrier coeffs: range {carrier_64.min()} to {carrier_64.max()}')

# Inject with base_mix = 32767 (full carrier, no blending)
inject_64_to_256(carrier_64, state.coeff_state)

print(f'After inject: coeff_state range {state.coeff_state.min()} to {state.coeff_state.max()}')

# Extract (should be same as carrier)
extracted = extract_64_leaves(state.coeff_state)
print(f'After extract: {extracted[0:5]}')
print(f'Original carrier: {carrier_64.tolist()[0:5]}')

# IDWPT
audio = idwpt_jit(extracted, headroom_bits=1)
audio_arr = np.array(audio, dtype=np.int16)

print(f'\nIDWPT output:')
print(f'  Range: {audio_arr.min()} to {audio_arr.max()}')
print(f'  RMS: {np.sqrt(np.mean(audio_arr.astype(np.float32)**2)):.1f}')
print(f'  First 10: {audio_arr[:10]}')
