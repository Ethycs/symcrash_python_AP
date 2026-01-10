"""Trace VM step to find where coefficients diverge."""

import numpy as np
import sys
sys.path.insert(0, 'src')

# Clear cache
import symcrash_python_ap.vm.coeff_lut_streaming as lut_mod
lut_mod._NOTE_LUT_CACHE.clear()

from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.vm.decode import decode_frame
from symcrash_python_ap.vm.tables import DEFAULT_TABLES, BLOCK_SAMPLES, COEFF_LEN
from symcrash_python_ap.vm.lanes import (
    lane_inject, dispatch_coeff_op, lane_tail, lane_ccs_render, VM_K
)
from symcrash_python_ap.vm.coeff_lut_streaming import (
    get_lut_for_note, ccs_generate_coeffs_polyphase, CCSState, BLOCK_SIZE, NUM_LEAVES
)
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.fixed import RSHIFT_RNE
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit

SAMPLE_RATE = 48000
f0 = 440.0
K = 8

print("=== TRACING VM STEP ===")
print()

# Setup frame
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

# Create state
state = VMState.create(voice_id=0)
state.reset_for_note_on(note=69, cents=0)

# Decode frame
cfg = decode_frame(
    frame,
    DEFAULT_TABLES,
    prev_eta_q15=state.eta_prev_q15,
    prev_gain_q15=state.gain_prev_q15,
)

print(f"cfg.use_coeff_osc: {cfg.use_coeff_osc}")
print(f"state.use_coeff_osc: {state.use_coeff_osc}")
print(f"cfg.coeff_osc_params.base_mix_q15: {cfg.coeff_osc_params.base_mix_q15}")
print(f"cfg.backbone_instrs: {cfg.backbone_instrs}")
print(f"cfg.event_instrs: {cfg.event_instrs}")
print(f"cfg.tail_ops: {cfg.tail_ops}")
print()

# Track blocks
state.blocks_since_start += 1
print(f"blocks_since_start: {state.blocks_since_start}")
print(f"IDWPT_LATENCY_BLOCKS: 5")

# During latency period, eta is forced to 0
if state.blocks_since_start < 5:
    print("!! Latency period - eta forced to 0")
    cfg.eta_curr_q15 = 0
    cfg.idwpt_enable = False

print()

# === STEP 0: Generate carrier coefficients ===
print("=== STEP 0: GENERATE CARRIER ===")

params = cfg.coeff_osc_params
if state.last_note != params.note or state.last_cents != params.cents:
    state.ccs_state.note = params.note
    state.ccs_state.wave_id = 0
    state.last_note = params.note
    state.last_cents = params.cents

if state.ccs_state.phase_inc != params.phase_inc:
    state.ccs_state.set_pitch(params.phase_inc, slew_blocks=8)

carrier_512 = np.zeros(COEFF_LEN, dtype=np.int16)
ccs_generate_coeffs_polyphase(state.ccs_state, carrier_512)

print(f"carrier_512 (leaf 0): {list(carrier_512[:8])}")
print(f"carrier_512 RMS: {np.sqrt(np.mean(carrier_512.astype(float)**2)):.1f}")

# === STEP 1: BLEND ===
print()
print("=== STEP 1: BLEND ===")

base_mix = params.base_mix_q15
print(f"base_mix: {base_mix}")
print(f"coeff_state before (leaf 0): {list(state.coeff_state[:8])}")

for idx in range(COEFF_LEN):
    carrier_val = int(carrier_512[idx])
    old_val = int(state.coeff_state[idx])
    inv_mix = 32767 - base_mix
    blended = RSHIFT_RNE(old_val * inv_mix + carrier_val * base_mix, 15)
    state.coeff_state[idx] = np.clip(blended, -32768, 32767)

print(f"coeff_state after (leaf 0): {list(state.coeff_state[:8])}")
print(f"coeff_state RMS: {np.sqrt(np.mean(state.coeff_state.astype(float)**2)):.1f}")

# === STEP 2: LANE_INJECT ===
print()
print("=== STEP 2: LANE_INJECT ===")
coeff_before = state.coeff_state.copy()
lane_inject(state, cfg)
diff = state.coeff_state.astype(np.int32) - coeff_before.astype(np.int32)
print(f"Coefficients changed: {np.count_nonzero(diff)}")
if np.count_nonzero(diff) > 0:
    changed_idx = np.where(diff != 0)[0]
    print(f"Changed indices: {changed_idx[:10]}...")
    print(f"Changes: {diff[changed_idx[:10]]}")

# === STEP 3: BACKBONE OPS ===
print()
print("=== STEP 3: BACKBONE OPS ===")
coeff_before = state.coeff_state.copy()
for instr in cfg.backbone_instrs:
    dispatch_coeff_op(state, instr, cfg.mask_u32)
diff = state.coeff_state.astype(np.int32) - coeff_before.astype(np.int32)
print(f"Coefficients changed: {np.count_nonzero(diff)}")
if np.count_nonzero(diff) > 0:
    changed_idx = np.where(diff != 0)[0]
    print(f"Changed indices: {changed_idx[:10]}...")
    print(f"Changes: {diff[changed_idx[:10]]}")

# === STEP 4: LANE_TAIL ===
print()
print("=== STEP 4: LANE_TAIL ===")
coeff_before = state.coeff_state.copy()
lane_tail(state, cfg)
diff = state.coeff_state.astype(np.int32) - coeff_before.astype(np.int32)
print(f"Coefficients changed: {np.count_nonzero(diff)}")
print(f"coeff_state after tail (leaf 0): {list(state.coeff_state[:8])}")
if np.count_nonzero(diff) > 0:
    changed_idx = np.where(diff != 0)[0]
    print(f"Changed indices: {changed_idx[:10]}...")
    print(f"Max diff: {np.abs(diff).max()}")

# === STEP 5: LANE_CCS_RENDER ===
print()
print("=== STEP 5: LANE_CCS_RENDER ===")

output = lane_ccs_render(state, cfg)
print(f"output: {list(output[:16])}")
print(f"output RMS: {np.sqrt(np.mean(output.astype(float)**2)):.1f}")

# === COMPARE TO DIRECT PATH ===
print()
print("=== DIRECT PATH COMPARISON ===")

# Reset and do direct path
lut = get_lut_for_note(wave_id=0, midi_note=69)
subband_output = np.zeros((NUM_LEAVES, K), dtype=np.int16)
lut.get_subband_samples(0, subband_output)

idwpt_outputs = []
for k in range(K):
    coeffs_k = subband_output[:, k].tolist()
    audio_k = idwpt_jit(coeffs_k, headroom_bits=1)
    idwpt_outputs.append(audio_k)

direct_block = np.zeros(BLOCK_SIZE, dtype=np.int16)
for k in range(K):
    for m in range(BLOCK_SIZE // K):
        direct_block[k + m * K] = idwpt_outputs[k][m * K]

print(f"direct: {list(direct_block[:16])}")
print(f"direct RMS: {np.sqrt(np.mean(direct_block.astype(float)**2)):.1f}")

print()
print(f"VM/Direct RMS ratio: {np.sqrt(np.mean(output.astype(float)**2)) / np.sqrt(np.mean(direct_block.astype(float)**2)):.3f}")
