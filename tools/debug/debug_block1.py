"""Debug what's happening in block 1."""

import numpy as np
import sys
sys.path.insert(0, 'src')

# Clear cache
import symcrash_python_ap.vm.coeff_lut_streaming as lut_mod
lut_mod._NOTE_LUT_CACHE.clear()

from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm.coeff_lut_streaming import (
    get_lut_for_note, BLOCK_SIZE, NUM_LEAVES, VM_K
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.fixed import midi_to_phase_inc_fixed

print("=== BLOCK 1 DEBUG ===")
print()

# Setup
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

# Check CCS state before first block
print("Before block 1:")
print(f"  ccs_state.phase = {state.ccs_state.phase}")
print(f"  ccs_state.phase_inc = {state.ccs_state.phase_inc}")
print(f"  ccs_state.note = {state.ccs_state.note}")
print()

# Run VM path
state, vm_block = vm_step_block(state, frame, DEFAULT_TABLES)

print("After block 1:")
print(f"  ccs_state.phase = {state.ccs_state.phase}")
print(f"  ccs_state.phase_inc = {state.ccs_state.phase_inc}")
print()

# Run direct path at phase 0
lut = get_lut_for_note(wave_id=0, midi_note=69)
subband_output = np.zeros((NUM_LEAVES, 8), dtype=np.int16)
lut.get_subband_samples(0, subband_output)

idwpt_outputs = []
for k in range(8):
    coeffs_k = subband_output[:, k].tolist()
    audio_k = idwpt_jit(coeffs_k, headroom_bits=1)
    idwpt_outputs.append(audio_k)

direct_block = np.zeros(BLOCK_SIZE, dtype=np.int16)
for k in range(8):
    for m in range(BLOCK_SIZE // 8):
        direct_block[k + m * 8] = idwpt_outputs[k][m * 8]

print("First 16 samples:")
print(f"  VM:     {list(vm_block[:16])}")
print(f"  Direct: {list(direct_block[:16])}")

diff = vm_block.astype(np.int32) - direct_block.astype(np.int32)
print(f"  Diff:   {list(diff[:16])}")
print()
print(f"Max diff: {np.abs(diff).max()}")
print(f"Are they equal? {np.array_equal(vm_block, direct_block)}")
