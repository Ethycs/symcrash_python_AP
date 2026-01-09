"""Compare manual trace with actual vm_step_block output."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.vm.coeff_lut_streaming import (
    CCSState, ccs_generate_coeffs_polyphase, NUM_LEAVES, VM_K, WAVE_SINE
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm.tables import DEFAULT_TABLES, COEFF_LEN, BLOCK_SAMPLES
from symcrash_python_ap.vm import vm_step_block

print("=== Comparing vm_step_block with manual trace ===\n")

# Create two identical states
state_vm = VMState.create(voice_id=0)
state_vm.reset_for_note_on(note=60, cents=0)

state_manual = VMState.create(voice_id=1)
state_manual.reset_for_note_on(note=60, cents=0)

frame = VoiceFrame(
    voice_id=0,
    note=60,
    cents=0,
    gain_q15=32767,
    beta_target_q15=32767,
)

# Run vm_step_block for 5 blocks
vm_outputs = []
for i in range(5):
    state_vm, output = vm_step_block(state_vm, frame, DEFAULT_TABLES)
    vm_outputs.append(output.copy())

# Run manual trace for 5 blocks (same as debug_vm_multiblock.py)
from symcrash_python_ap.vm.decode import decode_frame
from symcrash_python_ap.vm.lanes import lane_inject, lane_tail, lane_ccs_render
from symcrash_python_ap.fixed import RSHIFT_RNE

manual_outputs = []
for block in range(5):
    cfg = decode_frame(frame, DEFAULT_TABLES,
                       prev_eta_q15=state_manual.eta_prev_q15,
                       prev_gain_q15=state_manual.gain_prev_q15)

    params = cfg.coeff_osc_params

    pitch_changed = state_manual.last_note != params.note or state_manual.last_cents != params.cents
    if pitch_changed:
        state_manual.ccs_state.note = params.note
        state_manual.ccs_state.wave_id = params.table_id
        state_manual.last_note = params.note
        state_manual.last_cents = params.cents

    if state_manual.ccs_state.phase_inc != params.phase_inc:
        state_manual.ccs_state.set_pitch(params.phase_inc, slew_blocks=8)

    carrier_512 = np.zeros(COEFF_LEN, dtype=np.int16)
    ccs_generate_coeffs_polyphase(state_manual.ccs_state, carrier_512)

    base_mix = params.base_mix_q15
    for idx in range(COEFF_LEN):
        carrier_val = int(carrier_512[idx])
        old_val = int(state_manual.coeff_state[idx])
        inv_mix = 32767 - base_mix
        blended = RSHIFT_RNE(old_val * inv_mix + carrier_val * base_mix, 15)
        state_manual.coeff_state[idx] = np.clip(blended, -32768, 32767)

    lane_inject(state_manual, cfg)
    lane_tail(state_manual, cfg)
    output = lane_ccs_render(state_manual, cfg)

    manual_outputs.append(output.copy())

    state_manual.ccs_state.advance_block()
    state_manual.blocks_since_start += 1
    state_manual.eta_prev_q15 = cfg.eta_curr_q15
    state_manual.gain_prev_q15 = cfg.gain_curr_q15

# Compare
print("Block-by-block comparison:")
for i in range(5):
    vm_out = vm_outputs[i]
    manual_out = manual_outputs[i]
    match = np.array_equal(vm_out, manual_out)
    diff = np.abs(vm_out.astype(int) - manual_out.astype(int))
    print(f"\nBlock {i}:")
    print(f"  vm_step_block[:8]: {vm_out[:8]}")
    print(f"  manual[:8]:        {manual_out[:8]}")
    print(f"  Match: {match}, Max diff: {diff.max()}")

    if not match:
        print(f"  vm_step_block[-8:]: {vm_out[-8:]}")
        print(f"  manual[-8:]:        {manual_out[-8:]}")

# Show block boundary jumps for each
print("\n" + "="*60)
print("BLOCK BOUNDARY JUMPS")
print("="*60)

vm_audio = np.concatenate(vm_outputs)
manual_audio = np.concatenate(manual_outputs)

print("\nvm_step_block:")
for i in range(1, 5):
    idx = i * 64
    jump = abs(int(vm_audio[idx]) - int(vm_audio[idx - 1]))
    print(f"  Block {i-1} -> {i}: last={vm_audio[idx-1]}, first={vm_audio[idx]}, jump={jump}")

print("\nManual trace:")
for i in range(1, 5):
    idx = i * 64
    jump = abs(int(manual_audio[idx]) - int(manual_audio[idx - 1]))
    print(f"  Block {i-1} -> {i}: last={manual_audio[idx-1]}, first={manual_audio[idx]}, jump={jump}")
