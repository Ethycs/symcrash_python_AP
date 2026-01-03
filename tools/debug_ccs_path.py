"""Debug the CCS path - trace gain application."""
import numpy as np
from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES, BLOCK_SAMPLES
from symcrash_python_ap.vm.decode import decode_frame
from symcrash_python_ap.vm.lanes import lane_ccs_render, lane_inject, dispatch_coeff_op, lane_tail


class SimpleFrame:
    """Simple test frame."""
    def __init__(self, f0_hz=440.0, gain_q15=32767, beta_target_q15=32767):
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


print("=" * 60)
print("Trace gain application with gain=24000")
print("=" * 60)

state = VMState.create(voice_id=0)
state.reset_for_note_on(note=69, cents=0)
frame = SimpleFrame(gain_q15=24000)

for block_idx in range(3):
    print(f"\n--- Block {block_idx} ---")

    cfg = decode_frame(frame, DEFAULT_TABLES,
                       prev_eta_q15=state.eta_prev_q15,
                       prev_gain_q15=state.gain_prev_q15)

    print(f"gain_prev={cfg.gain_prev_q15}, gain_curr={cfg.gain_curr_q15}")

    state.blocks_since_start += 1
    if state.blocks_since_start >= 5:
        state.wet_ready = True

    lane_inject(state, cfg)
    for instr in cfg.backbone_instrs:
        dispatch_coeff_op(state, instr, cfg.mask_u32)
    lane_tail(state, cfg)

    output = lane_ccs_render(state, cfg)
    print(f"Before gain: max={output.max()}, min={output.min()}, output[0]={output[0]}")

    # Apply gain (same as executor)
    gain_curr = cfg.gain_curr_q15
    gain_prev = cfg.gain_prev_q15
    print(f"  Condition: gain_curr={gain_curr} != 32767 is {gain_curr != 32767}, gain_prev={gain_prev} != 32767 is {gain_prev != 32767}")
    if gain_curr != 32767 or gain_prev != 32767:
        for i in range(BLOCK_SAMPLES):
            alpha = (i * 32767) // BLOCK_SAMPLES
            delta = gain_curr - gain_prev
            scaled_delta = (delta * alpha) >> 15
            gain = gain_prev + scaled_delta
            if i == 0:
                print(f"  Sample 0: alpha={alpha}, delta={delta}, scaled_delta={scaled_delta}, gain={gain}")
                print(f"           before={output[0]}, after={(int(output[0]) * gain) >> 15}")
            output[i] = (int(output[i]) * gain) >> 15

    print(f"After gain: max={output.max()}, min={output.min()}, output[0]={output[0]}")

    state.eta_prev_q15 = cfg.eta_curr_q15
    state.gain_prev_q15 = cfg.gain_curr_q15


print("\n" + "=" * 60)
print("Test 2: Call lanes manually (mimicking vm_step_block)")
print("=" * 60)

state2 = VMState.create(voice_id=0)
state2.reset_for_note_on(note=69, cents=0)
frame2 = SimpleFrame()

for block_idx in range(3):
    print(f"\n--- Block {block_idx} ---")
    cfg = decode_frame(frame2, DEFAULT_TABLES,
                       prev_eta_q15=state2.eta_prev_q15,
                       prev_gain_q15=state2.gain_prev_q15)

    print(f"Before: phase={state2.coeff_osc.node_phases[0]}")
    print(f"cfg.use_coeff_osc={cfg.use_coeff_osc}, state.use_coeff_osc={state2.use_coeff_osc}")

    # Track blocks
    state2.blocks_since_start += 1
    if state2.blocks_since_start >= 5:
        state2.wet_ready = True

    # CCS path
    lane_inject(state2, cfg)
    for instr in cfg.backbone_instrs:
        dispatch_coeff_op(state2, instr, cfg.mask_u32)
    lane_tail(state2, cfg)

    output = lane_ccs_render(state2, cfg)

    print(f"After: phase={state2.coeff_osc.node_phases[0]}")
    print(f"Output: max={output.max()}, min={output.min()}")

    # Update state for next block
    state2.eta_prev_q15 = cfg.eta_curr_q15
    state2.gain_prev_q15 = cfg.gain_curr_q15
