"""Debug why VM path produces different output than direct path."""

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
    get_lut_for_note, BLOCK_SIZE, NUM_LEAVES, VM_K,
    ccs_generate_coeffs_polyphase, CCSState
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.fixed import RSHIFT_RNE

SAMPLE_RATE = 48000
f0 = 440.0
K = 8

print("=== DEBUGGING VM PATH ===")
print()

# Setup
lut = get_lut_for_note(wave_id=0, midi_note=69)
phase_inc = int((f0 / SAMPLE_RATE) * (1 << 32))

# Create CCS state (same as VM would do)
ccs_state = CCSState()
ccs_state.note = 69
ccs_state.wave_id = 0
ccs_state.phase = 0
ccs_state.phase_inc = phase_inc

print(f"Phase inc: {phase_inc}")
print()

# === TEST 1: Direct LUT path ===
print("=== DIRECT LUT PATH ===")
subband_output = np.zeros((NUM_LEAVES, K), dtype=np.int16)
lut.get_subband_samples(0, subband_output)  # Phase 0

# Show first few coefficients
print(f"LUT coefficients at phase 0:")
print(f"  Leaf 0: {list(subband_output[0, :])}")
print(f"  Leaf 1: {list(subband_output[1, :])}")
print(f"  Leaf 3: {list(subband_output[3, :])}")
print()

# Run IDWPT directly
idwpt_outputs = []
for k in range(K):
    coeffs_k = subband_output[:, k].tolist()
    audio_k = idwpt_jit(coeffs_k, headroom_bits=1)
    idwpt_outputs.append(audio_k)

# Interleave
direct_block = np.zeros(BLOCK_SIZE, dtype=np.int16)
for k in range(K):
    for m in range(BLOCK_SIZE // K):
        direct_block[k + m * K] = idwpt_outputs[k][m * K]

print(f"Direct path output: {list(direct_block[:16])}")
print(f"Direct RMS: {np.sqrt(np.mean(direct_block.astype(float)**2)):.1f}")
print()

# === TEST 2: ccs_generate_coeffs_polyphase path ===
print("=== ccs_generate_coeffs_polyphase PATH ===")

carrier_512 = np.zeros(512, dtype=np.int16)
ccs_generate_coeffs_polyphase(ccs_state, carrier_512)

# Check layout: [leaf * K + k]
print(f"carrier_512 coefficients:")
print(f"  Leaf 0 k0..k7: {list(carrier_512[0*8:0*8+8])}")
print(f"  Leaf 1 k0..k7: {list(carrier_512[1*8:1*8+8])}")
print(f"  Leaf 3 k0..k7: {list(carrier_512[3*8:3*8+8])}")
print()

# Reshape and extract
coeff_2d = carrier_512.reshape(64, K)
print(f"coeff_2d[:, 0] (all leaves, k=0): first 5 = {list(coeff_2d[:5, 0])}")
print(f"coeff_2d[:, 1] (all leaves, k=1): first 5 = {list(coeff_2d[:5, 1])}")
print()

# Compare to direct LUT
print("Comparing carrier_512 to subband_output:")
for l in range(3):
    for k in range(K):
        c512_val = carrier_512[l * K + k]
        lut_val = subband_output[l, k]
        if c512_val != lut_val:
            print(f"  MISMATCH at leaf {l}, k={k}: carrier_512={c512_val}, lut={lut_val}")
        else:
            print(f"  leaf {l}, k={k}: {c512_val} OK")

print()

# === TEST 3: VM blending ===
print("=== VM BLENDING ===")

# Simulate blending with base_mix = 32767 (full carrier)
base_mix = 32767
coeff_state = np.zeros(512, dtype=np.int16)  # Initial state = zeros

blended = np.zeros(512, dtype=np.int16)
for idx in range(512):
    carrier_val = int(carrier_512[idx])
    old_val = int(coeff_state[idx])
    inv_mix = 32767 - base_mix
    blended_val = RSHIFT_RNE(old_val * inv_mix + carrier_val * base_mix, 15)
    blended[idx] = np.clip(blended_val, -32768, 32767)

print(f"With base_mix={base_mix}:")
print(f"  blended leaf 0 k0..k7: {list(blended[0*8:0*8+8])}")
print(f"  carrier leaf 0 k0..k7: {list(carrier_512[0*8:0*8+8])}")
print()

# Are they the same?
diff = np.abs(blended.astype(np.int32) - carrier_512.astype(np.int32))
print(f"  Max diff between blended and carrier: {diff.max()}")
print()

# === TEST 4: Full VM path ===
print("=== FULL VM PATH ===")

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

vm_state = VMState.create(voice_id=0)
vm_state.reset_for_note_on(note=69, cents=0)

# Check initial coeff_state
print(f"Initial coeff_state[0:8]: {list(vm_state.coeff_state[0:8])}")
print(f"vm_state.use_coeff_osc: {vm_state.use_coeff_osc}")
print()

# Run one block
vm_state, vm_block = vm_step_block(vm_state, frame, DEFAULT_TABLES)

print(f"VM output: {list(vm_block[:16])}")
print(f"VM RMS: {np.sqrt(np.mean(vm_block.astype(float)**2)):.1f}")
print()

# Check coeff_state after first block
print(f"coeff_state after block 1 [0:8]: {list(vm_state.coeff_state[0:8])}")
print()

# === COMPARE ===
print("=== COMPARISON ===")
print(f"Direct: {list(direct_block[:16])}")
print(f"VM:     {list(vm_block[:16])}")
diff_arr = vm_block.astype(np.int32) - direct_block.astype(np.int32)
print(f"Diff:   {list(diff_arr[:16])}")
print()
print(f"Max diff: {np.abs(diff_arr).max()}")
print(f"Direct RMS: {np.sqrt(np.mean(direct_block.astype(float)**2)):.1f}")
print(f"VM RMS: {np.sqrt(np.mean(vm_block.astype(float)**2)):.1f}")
