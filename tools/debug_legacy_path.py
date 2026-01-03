"""Debug the legacy carrier->FWPT->IDWPT path."""
import numpy as np
from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.wavelet.fast_lifting import fwpt_balanced

class SimpleFrame:
    """Simple test frame for legacy path."""
    def __init__(self, f0_hz=440.0, gain_q15=24000, beta_target_q15=0):
        self.voice_id = 0
        self.note = 69  # A4
        self.cents = 0
        self.f0_hz = f0_hz
        self.gain_q15 = gain_q15
        self.beta_target_q15 = beta_target_q15
        self.ops_enable = []
        self.families = {}
        self.pattern_id = 0
        self.tail_mode = 0
        self.tree_a_id = 0
        self.tree_b_id = 0
        self.tree_blend_q15 = 0
        self.partial_amps_q15 = [32767] + [0] * 15  # Pure sine
        self.event_mask = 0
        self.event_strength_q15 = [0] * 8
        # Disable CCS to use legacy path
        self.use_coeff_osc = False


# Generate carrier using legacy path (use_coeff_osc=False)
state = VMState.create(voice_id=0)
state.use_coeff_osc = False  # Force legacy path
state.reset_for_note_on(note=69, cents=0)

frame = SimpleFrame(f0_hz=440.0, beta_target_q15=0)

# Generate a few blocks
print("Legacy path output (carrier only, beta=0):")
for block_idx in range(5):
    state, output = vm_step_block(state, frame, DEFAULT_TABLES)
    print(f"Block {block_idx}: samples[:8] = {output[:8].tolist()}")
    print(f"           samples[56:] = {output[56:].tolist()}")

# Now check what FWPT produces for a sine wave carrier
print("\n--- FWPT of sine wave carrier ---")
carrier = [int(10000 * np.sin(2 * np.pi * 440 * i / 48000)) for i in range(64)]
print(f"Carrier (first 8): {carrier[:8]}")

leaves = fwpt_balanced(carrier, depth=6)
# Each leaf is a list with one coefficient
coeffs = [leaf[0] if leaf else 0 for leaf in leaves]
print(f"FWPT coefficients (first 16): {coeffs[:16]}")
print(f"FWPT coefficients (16-32): {coeffs[16:32]}")
print(f"Non-zero coeffs: {[(i, c) for i, c in enumerate(coeffs) if abs(c) > 10]}")
