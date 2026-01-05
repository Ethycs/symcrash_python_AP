import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.vm.decode import decode_frame, DecodedPlan
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.vm.lanes import lane_tail, extract_64_leaves
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.vm.coeff_osc import coeff_osc_generate_coeffs
from symcrash_python_ap.vm.coeff_tables_fast import get_fast_coeff_table
from symcrash_python_ap.vm.lanes import inject_64_to_256
import numpy as np

print('Testing tail operator effect on amplitude\n')

# Setup
state = VMState.create(0)
state.reset_for_note_on(60, 0)

# Generate carrier
table = get_fast_coeff_table()
carrier_64 = np.zeros(64, dtype=np.int16)

state.coeff_osc.table_id = 60
state.coeff_osc.base_phase_inc = int((261.63 / 48000) * (1 << 32))
coeff_osc_generate_coeffs(state.coeff_osc, table, carrier_64)

print(f'Carrier range: {carrier_64.min()} to {carrier_64.max()}')

# Inject
inject_64_to_256(carrier_64, state.coeff_state)
print(f'After inject: {state.coeff_state.min()} to {state.coeff_state.max()}')

# BEFORE tail
coeffs_before = extract_64_leaves(state.coeff_state)
audio_before = idwpt_jit(coeffs_before, headroom_bits=1)
rms_before = np.sqrt(np.mean(np.array(audio_before, dtype=np.float32)**2))
print(f'Before tail: RMS={rms_before:.1f}, range={min(audio_before)} to {max(audio_before)}')

# Decode frame to get config
frame = VoiceFrame(voice_id=0, note=60, gain_q15=32767, beta_target_q15=32767)
cfg = decode_frame(frame, DEFAULT_TABLES, 0, 0)

# Apply tail
lane_tail(state, cfg)
print(f'After tail: {state.coeff_state.min()} to {state.coeff_state.max()}')

# AFTER tail
coeffs_after = extract_64_leaves(state.coeff_state)
audio_after = idwpt_jit(coeffs_after, headroom_bits=1)
rms_after = np.sqrt(np.mean(np.array(audio_after, dtype=np.float32)**2))
print(f'After tail:  RMS={rms_after:.1f}, range={min(audio_after)} to {max(audio_after)}')

print(f'\nAmplitude change: {rms_before:.1f} -> {rms_after:.1f} ({rms_after/rms_before*100:.1f}%)')
