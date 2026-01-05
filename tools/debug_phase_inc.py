"""Debug phase increment mismatch."""

import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.vm import VMState
from symcrash_python_ap.vm.decode import decode_frame
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.fixed import midi_to_phase_inc_fixed

frame = VoiceFrame(
    voice_id=0,
    note=69,
    cents=0,
    gain_q15=32767,
    pan_q15=0,
    tree_a_id=0,
    tree_b_id=0,
    tree_blend_q15=0,
    is_active=True,
    beta_target_q15=0,
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
)

state = VMState.create(voice_id=0)
state.reset_for_note_on(note=69, cents=0)

cfg = decode_frame(frame, DEFAULT_TABLES)

expected_phase_inc = midi_to_phase_inc_fixed(69, 0)
print(f"midi_to_phase_inc_fixed(69, 0) = {expected_phase_inc}")
print(f"state.ccs_state.phase_inc = {state.ccs_state.phase_inc}")
print(f"cfg.coeff_osc_params.phase_inc = {cfg.coeff_osc_params.phase_inc}")
print()
print(f"Match state.ccs_state to cfg: {state.ccs_state.phase_inc == cfg.coeff_osc_params.phase_inc}")
