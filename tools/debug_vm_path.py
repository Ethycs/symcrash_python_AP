"""Debug which VM path is being used."""
import numpy as np
from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.frames import VoiceFrame

# Create a simple frame
frame = VoiceFrame(
    voice_id=0,
    note=60,
    cents=0,
    gain_q15=32767,
    pan_q15=0,
    tree_a_id=0,
    tree_b_id=0,
    tree_blend_q15=0,
    is_active=True,
    beta_target_q15=16384,  # 50% wet
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
)

# Check use_coeff_osc attribute
print(f"Frame has use_coeff_osc: {hasattr(frame, 'use_coeff_osc')}")
if hasattr(frame, 'use_coeff_osc'):
    print(f"  use_coeff_osc = {frame.use_coeff_osc}")

# Create VM state and check its use_coeff_osc
vm_state = VMState.create(voice_id=0)
vm_state.reset_for_note_on(note=60, cents=0)
print(f"\nVMState.use_coeff_osc = {vm_state.use_coeff_osc}")

# Run one block and check the decoded plan
from symcrash_python_ap.vm.decode import decode_frame
cfg = decode_frame(frame, DEFAULT_TABLES, prev_eta_q15=0, prev_gain_q15=32767)
print(f"\nDecodedPlan.use_coeff_osc = {cfg.use_coeff_osc}")
print(f"DecodedPlan.carrier_role = {cfg.carrier_role}")
print(f"DecodedPlan.eta_prev_q15 = {cfg.eta_prev_q15}")
print(f"DecodedPlan.eta_curr_q15 = {cfg.eta_curr_q15}")
print(f"DecodedPlan.idwpt_enable = {cfg.idwpt_enable}")

# Check if CCS path condition is met
ccs_path_taken = cfg.use_coeff_osc and vm_state.use_coeff_osc
print(f"\nCCS path condition: cfg.use_coeff_osc={cfg.use_coeff_osc} AND state.use_coeff_osc={vm_state.use_coeff_osc}")
print(f"CCS path would be taken: {ccs_path_taken}")

# Run the block
vm_state, output = vm_step_block(vm_state, frame, DEFAULT_TABLES)
print(f"\nOutput max: {np.abs(output).max()}")
print(f"First 8 samples: {output[:8].tolist()}")
