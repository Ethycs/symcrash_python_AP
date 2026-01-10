"""Check initial state values."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.vm.decode import decode_frame

state = VMState.create(voice_id=0)
state.reset_for_note_on(note=60, cents=0)

print("Initial state after reset_for_note_on:")
print(f"  gain_prev_q15: {state.gain_prev_q15}")
print(f"  eta_prev_q15: {state.eta_prev_q15}")
print(f"  blocks_since_start: {state.blocks_since_start}")

frame = VoiceFrame(
    voice_id=0,
    note=60,
    cents=0,
    gain_q15=32767,
    beta_target_q15=32767,
)

# Check what decode_frame produces for block 0
cfg = decode_frame(frame, DEFAULT_TABLES,
                   prev_eta_q15=state.eta_prev_q15,
                   prev_gain_q15=state.gain_prev_q15)

print("\nDecoded config for block 0:")
print(f"  gain_curr_q15: {cfg.gain_curr_q15}")
print(f"  gain_prev_q15: {cfg.gain_prev_q15}")
print(f"  eta_curr_q15: {cfg.eta_curr_q15}")
print(f"  eta_prev_q15: {cfg.eta_prev_q15}")

# Check if gain is being applied
if cfg.gain_curr_q15 != 32767 or cfg.gain_prev_q15 != 32767:
    print("\n>>> GAIN IS BEING APPLIED!")
    print("  This is why block 0 output is different!")
