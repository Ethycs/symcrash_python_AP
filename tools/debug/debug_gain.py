"""Debug gain ramp issue."""

import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.vm import VMState
from symcrash_python_ap.vm.decode import decode_frame
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.frames import VoiceFrame

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

print(f"state.gain_prev_q15 after reset: {state.gain_prev_q15}")

cfg = decode_frame(
    frame,
    DEFAULT_TABLES,
    prev_eta_q15=state.eta_prev_q15,
    prev_gain_q15=state.gain_prev_q15,
)

print(f"cfg.gain_prev_q15: {cfg.gain_prev_q15}")
print(f"cfg.gain_curr_q15: {cfg.gain_curr_q15}")
print()
print(f"Will gain ramp be applied? {cfg.gain_curr_q15 != 32767 or cfg.gain_prev_q15 != 32767}")
