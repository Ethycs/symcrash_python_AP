"""Test toggling ETA rapidly in CCS path."""
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


print("Test: Toggle ETA every block and check for discontinuities")
print("=" * 60)

state = VMState.create(voice_id=0)
state.reset_for_note_on(note=69, cents=0)

all_outputs = []
prev_end = None
discontinuities = []

for i in range(20):
    # Toggle ETA every block
    eta = 32767 if (i % 2) == 0 else 0
    frame = Frame(eta=eta)

    state, output = vm_step_block(state, frame, DEFAULT_TABLES)
    all_outputs.append(output.copy())

    # Check for discontinuity at block boundary
    if prev_end is not None:
        diff = abs(int(output[0]) - int(prev_end))
        if diff > 1000:
            discontinuities.append((i, diff))
            print(f"Block {i}: ETA={eta}, boundary diff={diff}")

    prev_end = output[-1]

    # Show first few samples
    if i < 5:
        print(f"Block {i}: ETA={eta}, first4={output[:4].tolist()}, last={output[-1]}")

print(f"\nTotal discontinuities (diff > 1000): {len(discontinuities)}")

# Compute overall statistics
all_samples = np.concatenate(all_outputs)
diffs = np.abs(np.diff(all_samples.astype(int)))
print(f"Max sample-to-sample diff: {diffs.max()}")
print(f"Samples with diff > 1000: {np.sum(diffs > 1000)}")
