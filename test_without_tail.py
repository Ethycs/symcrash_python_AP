import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm.decode import decode_frame
import numpy as np

print('Checking what tail ops are applied\n')

state = VMState.create(0)
state.reset_for_note_on(60, 0)

frame = VoiceFrame(voice_id=0, note=60, gain_q15=32767, beta_target_q15=32767)

# Decode to see what tail ops would be applied
cfg = decode_frame(frame, DEFAULT_TABLES, 0, 0)

print(f'Tail ops: {cfg.tail_ops}')
print(f'Tail mode: {cfg.tail_mode}')
print(f'Backbone ops: {cfg.backbone_instrs}')
print(f'use_coeff_osc: {cfg.use_coeff_osc}')
print(f'base_mix: {cfg.coeff_osc_params.base_mix_q15}')
