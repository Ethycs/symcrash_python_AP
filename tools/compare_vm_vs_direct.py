"""Compare VM CCS output vs direct CCS (diagnose_spurs.py method)."""

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

SAMPLE_RATE = 48000
f0 = 440.0
K = 8

print("=== COMPARING VM vs DIRECT CCS ===")
print()

# Check what LUT the VM will use
lut = get_lut_for_note(wave_id=0, midi_note=69)
print(f"LUT shape: {lut.data.shape}")
print(f"LUT P: {lut.P}")
print(f"LUT K: {lut.K}")
print()

# Generate via VM
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

# Generate via direct path (like diagnose_spurs.py)
phase_inc = int((f0 / SAMPLE_RATE) * (1 << 32))

# Generate first 10 blocks via both methods
n_blocks = 10
vm_output = []
direct_output = []

phase_32 = 0

for block_idx in range(n_blocks):
    # VM path
    vm_state, vm_block = vm_step_block(vm_state, frame, DEFAULT_TABLES)
    vm_output.extend(vm_block.tolist())

    # Direct path (same as diagnose_spurs.py)
    subband_output = np.zeros((NUM_LEAVES, K), dtype=np.int16)
    lut.get_subband_samples(phase_32, subband_output)

    idwpt_outputs = []
    for k in range(K):
        coeffs_k = [subband_output[l, k] for l in range(NUM_LEAVES)]
        audio_k = idwpt_jit(coeffs_k, headroom_bits=1)
        idwpt_outputs.append(audio_k)

    block_start = block_idx * BLOCK_SIZE
    direct_block = np.zeros(BLOCK_SIZE, dtype=np.int16)
    for k in range(K):
        for m in range(BLOCK_SIZE // K):
            direct_block[k + m * K] = idwpt_outputs[k][m * K]

    direct_output.extend(direct_block.tolist())

    phase_32 = (phase_32 + phase_inc * BLOCK_SIZE) & 0xFFFFFFFF

vm_output = np.array(vm_output, dtype=np.int16)
direct_output = np.array(direct_output, dtype=np.int16)

print(f"VM output: {len(vm_output)} samples, RMS={np.sqrt(np.mean(vm_output.astype(float)**2)):.1f}")
print(f"Direct output: {len(direct_output)} samples, RMS={np.sqrt(np.mean(direct_output.astype(float)**2)):.1f}")
print()

# Compare
diff = vm_output.astype(np.int32) - direct_output.astype(np.int32)
print(f"Difference (VM - Direct):")
print(f"  Max: {np.abs(diff).max()}")
print(f"  RMS: {np.sqrt(np.mean(diff**2)):.1f}")
print()

# If they differ, show first few samples
if np.abs(diff).max() > 0:
    print("First 16 samples:")
    print(f"  VM:     {list(vm_output[:16])}")
    print(f"  Direct: {list(direct_output[:16])}")
    print(f"  Diff:   {list(diff[:16])}")
    print()

    # Check boundary discontinuity for both
    print("Boundary discontinuity check:")
    for name, data in [("VM", vm_output), ("Direct", direct_output)]:
        delta_0 = []
        for b in range(1, n_blocks):
            boundary = b * BLOCK_SIZE
            d0 = int(data[boundary]) - int(data[boundary - 1])
            delta_0.append(d0)
        delta_0 = np.array(delta_0)
        print(f"  {name}: value jump RMS = {np.sqrt(np.mean(delta_0**2)):.1f}")
