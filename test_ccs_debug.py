import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.vm.coeff_osc import CoeffOscState, coeff_osc_generate_coeffs
from symcrash_python_ap.vm.coeff_tables_fast import get_fast_coeff_table
from symcrash_python_ap.vm.lanes import extract_64_leaves, inject_64_to_256
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
import numpy as np

print('Debugging CCS coefficient flow\n')

# Step 1: Generate carrier coefficients
state = CoeffOscState()
state.table_id = 60
state.base_phase_inc = int((261.63 / 48000) * (1 << 32))

table = get_fast_coeff_table()
carrier_64 = np.zeros(64, dtype=np.int16)

print('Step 1: Generate carrier coefficients')
coeff_osc_generate_coeffs(state, table, carrier_64)
print(f'  carrier_64 range: {carrier_64.min()} to {carrier_64.max()}')
print(f'  carrier_64[0:5]: {carrier_64[0:5]}')
print(f'  Non-zero: {np.count_nonzero(carrier_64)}/64')

# Step 2: Inject into 256 state
coeff_state = np.zeros(256, dtype=np.int16)
inject_64_to_256(carrier_64, coeff_state)

print('\nStep 2: After injection (64 -> 256)')
print(f'  coeff_state range: {coeff_state.min()} to {coeff_state.max()}')
print(f'  coeff_state[0:8]: {coeff_state[0:8]}')
print(f'  Non-zero: {np.count_nonzero(coeff_state)}/256')

# Step 3: Extract back
coeffs_64_extracted = extract_64_leaves(coeff_state)

print('\nStep 3: After extraction (256 -> 64)')
print(f'  Extracted[0:5]: {coeffs_64_extracted[0:5]}')

# Step 4: IDWPT
audio = idwpt_jit(coeffs_64_extracted, headroom_bits=1)
audio_arr = np.array(audio, dtype=np.int16)

print('\nStep 4: IDWPT reconstruction')
print(f'  Audio range: {audio_arr.min()} to {audio_arr.max()}')
print(f'  Audio[0:5]: {audio_arr[0:5]}')

# Compare: Direct IDWPT without inject/extract cycle
audio_direct = idwpt_jit(carrier_64.tolist(), headroom_bits=1)
audio_direct_arr = np.array(audio_direct, dtype=np.int16)

print('\nComparison: Direct IDWPT (no inject/extract)')
print(f'  Audio range: {audio_direct_arr.min()} to {audio_direct_arr.max()}')
print(f'  Audio[0:5]: {audio_direct_arr[0:5]}')
