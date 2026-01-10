"""Test ETA initialization - what happens before/after first ETA change."""
import numpy as np
from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.vm.decode import decode_frame


class Frame:
    def __init__(self, eta=None):
        self.voice_id = 0
        self.note = 69
        self.cents = 0
        self.f0_hz = 440.0
        self.gain_q15 = 32767
        # Key: what happens if beta_target_q15 is not set?
        if eta is not None:
            self.beta_target_q15 = eta
        # Note: NOT setting beta_target_q15 if eta is None
        self.ops_enable = []
        self.families = {}
        self.pattern_id = 0
        self.tail_mode = 0
        self.tree_a_id = 0
        self.tree_b_id = 0
        self.tree_blend_q15 = 0
        self.partial_amps_q15 = [32767] + [0] * 15
        self.event_mask = 0
        self.event_strength_q15 = [0] * 8


print("=" * 60)
print("Test 1: Frame WITHOUT beta_target_q15 attribute")
print("=" * 60)

state1 = VMState.create(voice_id=0)
state1.reset_for_note_on(note=69, cents=0)

# Frame without eta set
frame_no_eta = Frame(eta=None)
print(f"Frame has beta_target_q15? {hasattr(frame_no_eta, 'beta_target_q15')}")

for i in range(3):
    cfg = decode_frame(frame_no_eta, DEFAULT_TABLES,
                       prev_eta_q15=state1.eta_prev_q15,
                       prev_gain_q15=state1.gain_prev_q15)
    print(f"Block {i}: eta_prev={cfg.eta_prev_q15}, eta_curr={cfg.eta_curr_q15}")
    state1, out = vm_step_block(state1, frame_no_eta, DEFAULT_TABLES)
    print(f"         output max={out.max()}, min={out.min()}")

print("\n" + "=" * 60)
print("Test 2: Frame WITH beta_target_q15=0")
print("=" * 60)

state2 = VMState.create(voice_id=0)
state2.reset_for_note_on(note=69, cents=0)

frame_eta_0 = Frame(eta=0)
print(f"Frame beta_target_q15 = {frame_eta_0.beta_target_q15}")

for i in range(3):
    cfg = decode_frame(frame_eta_0, DEFAULT_TABLES,
                       prev_eta_q15=state2.eta_prev_q15,
                       prev_gain_q15=state2.gain_prev_q15)
    print(f"Block {i}: eta_prev={cfg.eta_prev_q15}, eta_curr={cfg.eta_curr_q15}")
    state2, out = vm_step_block(state2, frame_eta_0, DEFAULT_TABLES)
    print(f"         output max={out.max()}, min={out.min()}")

print("\n" + "=" * 60)
print("Test 3: Transition from no-eta to eta=16384")
print("=" * 60)

state3 = VMState.create(voice_id=0)
state3.reset_for_note_on(note=69, cents=0)

# First few blocks without eta
for i in range(3):
    frame = Frame(eta=None)
    state3, out = vm_step_block(state3, frame, DEFAULT_TABLES)
    print(f"Block {i} (no eta): output max={out.max()}, min={out.min()}")

# Now touch eta
print("\n--- ETA TOUCHED ---")
for i in range(3, 6):
    frame = Frame(eta=16384)
    cfg = decode_frame(frame, DEFAULT_TABLES,
                       prev_eta_q15=state3.eta_prev_q15,
                       prev_gain_q15=state3.gain_prev_q15)
    print(f"Block {i}: eta_prev={state3.eta_prev_q15}, eta_curr={cfg.eta_curr_q15}")
    state3, out = vm_step_block(state3, frame, DEFAULT_TABLES)
    print(f"         output max={out.max()}, min={out.min()}")

print("\n" + "=" * 60)
print("Test 4: Check default eta value in decode")
print("=" * 60)

# Look at what decode_frame does with missing beta_target_q15
frame_missing = Frame(eta=None)
cfg = decode_frame(frame_missing, DEFAULT_TABLES, prev_eta_q15=0, prev_gain_q15=32767)
print(f"beta_target from frame: {getattr(frame_missing, 'beta_target_q15', 'NOT SET')}")
print(f"Decoded eta_curr_q15: {cfg.eta_curr_q15}")
print(f"Decoded eta_prev_q15: {cfg.eta_prev_q15}")
