"""Debug block 6 (after latency period)."""

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
    get_lut_for_note, BLOCK_SIZE, NUM_LEAVES
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit

print("=== BLOCK 6 DEBUG (after latency period) ===")
print()

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

# Run 5 blocks to get past latency period and gain ramp
for _ in range(5):
    state, _ = vm_step_block(state, frame, DEFAULT_TABLES)

# Now block 6
print("Before block 6:")
print(f"  ccs_state.phase = {state.ccs_state.phase}")
print(f"  gain_prev_q15 = {state.gain_prev_q15}")
print()

state, vm_block = vm_step_block(state, frame, DEFAULT_TABLES)

print("After block 6:")
print(f"  ccs_state.phase = {state.ccs_state.phase}")
print()

# Run direct path at same phase (5 blocks worth)
lut = get_lut_for_note(wave_id=0, midi_note=69)
phase_inc = state.ccs_state.phase_inc
phase_5_blocks = (phase_inc * BLOCK_SIZE * 5) & 0xFFFFFFFF

print(f"Direct path phase: {phase_5_blocks}")

subband_output = np.zeros((NUM_LEAVES, 8), dtype=np.int16)
lut.get_subband_samples(phase_5_blocks, subband_output)

idwpt_outputs = []
for k in range(8):
    coeffs_k = subband_output[:, k].tolist()
    audio_k = idwpt_jit(coeffs_k, headroom_bits=1)
    idwpt_outputs.append(audio_k)

direct_block = np.zeros(BLOCK_SIZE, dtype=np.int16)
for k in range(8):
    for m in range(BLOCK_SIZE // 8):
        direct_block[k + m * 8] = idwpt_outputs[k][m * 8]

print()
print("First 16 samples:")
print(f"  VM:     {list(vm_block[:16])}")
print(f"  Direct: {list(direct_block[:16])}")

diff = vm_block.astype(np.int32) - direct_block.astype(np.int32)
print(f"  Diff:   {list(diff[:16])}")
print()
print(f"Max diff: {np.abs(diff).max()}")
print(f"VM RMS: {np.sqrt(np.mean(vm_block.astype(float)**2)):.1f}")
print(f"Direct RMS: {np.sqrt(np.mean(direct_block.astype(float)**2)):.1f}")
