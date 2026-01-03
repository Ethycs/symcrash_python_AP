"""Test if ETA has any effect in CCS path."""
import numpy as np
from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES


class Frame:
    def __init__(self, eta):
        self.voice_id = 0
        self.note = 69
        self.cents = 0
        self.f0_hz = 440.0
        self.gain_q15 = 32767
        self.beta_target_q15 = eta
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


# Test with ETA=0
state0 = VMState.create(voice_id=0)
state0.reset_for_note_on(note=69, cents=0)
frame0 = Frame(eta=0)
for _ in range(10):
    state0, _ = vm_step_block(state0, frame0, DEFAULT_TABLES)
state0, out0 = vm_step_block(state0, frame0, DEFAULT_TABLES)

# Test with ETA=32767
state1 = VMState.create(voice_id=0)
state1.reset_for_note_on(note=69, cents=0)
frame1 = Frame(eta=32767)
for _ in range(10):
    state1, _ = vm_step_block(state1, frame1, DEFAULT_TABLES)
state1, out1 = vm_step_block(state1, frame1, DEFAULT_TABLES)

print(f"ETA=0:     max={out0.max()}, min={out0.min()}, first8={out0[:8].tolist()}")
print(f"ETA=32767: max={out1.max()}, min={out1.min()}, first8={out1[:8].tolist()}")
print(f"Difference: {np.abs(out0.astype(int) - out1.astype(int)).max()}")

if np.array_equal(out0, out1):
    print("\n✓ ETA has NO effect in CCS path (outputs are identical)")
else:
    print("\n✗ ETA DOES affect output in CCS path")
