"""Trace VM step block to find corruption."""
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
from symcrash_python_ap.vm.tables import DEFAULT_TABLES, COEFF_LEN
from symcrash_python_ap.vm.decode import decode_frame
from symcrash_python_ap.vm.lanes import lane_inject, lane_tail, lane_ccs_render
from symcrash_python_ap.fixed import RSHIFT_RNE

print("=== Tracing VM step block execution ===\n")

# Create state
state = VMState.create(voice_id=0)
state.reset_for_note_on(note=60, cents=0)

frame = VoiceFrame(
    voice_id=0,
    note=60,
    cents=0,
    gain_q15=32767,
    beta_target_q15=32767,
)

# Decode
cfg = decode_frame(frame, DEFAULT_TABLES, prev_eta_q15=0, prev_gain_q15=32767)

print("Decoded config:")
print(f"  use_coeff_osc: {cfg.use_coeff_osc}")
print(f"  osc_mode: {cfg.coeff_osc_params.osc_mode}")
print(f"  phase_inc: {cfg.coeff_osc_params.phase_inc}")
print(f"  base_mix_q15: {cfg.coeff_osc_params.base_mix_q15}")
print(f"  note: {cfg.coeff_osc_params.note}")
print(f"  table_id: {cfg.coeff_osc_params.table_id}")
print(f"  tail_ops: {cfg.tail_ops}")
print(f"  gain_curr_q15: {cfg.gain_curr_q15}")
print(f"  gain_prev_q15: {cfg.gain_prev_q15}")

# ========== STEP BY STEP ==========

print("\n--- Step 0: Initial state ---")
print(f"state.ccs_state.phase: {state.ccs_state.phase}")
print(f"state.ccs_state.phase_inc: {state.ccs_state.phase_inc}")
print(f"state.coeff_state[:8]: {state.coeff_state[:8]}")

# Step 1: Generate carrier
print("\n--- Step 1: Generate carrier ---")
carrier_512 = np.zeros(COEFF_LEN, dtype=np.int16)

# Update CCS state from params (this is what executor.py does)
params = cfg.coeff_osc_params
state.ccs_state.note = params.note
state.ccs_state.wave_id = params.table_id

# Handle pitch change
if state.ccs_state.phase_inc != params.phase_inc:
    state.ccs_state.set_pitch(params.phase_inc, slew_blocks=8)
    print(f"Set pitch: phase_inc={params.phase_inc}")

print(f"CCS state before generate:")
print(f"  phase: {state.ccs_state.phase}")
print(f"  phase_inc: {state.ccs_state.phase_inc}")
print(f"  note: {state.ccs_state.note}")
print(f"  wave_id: {state.ccs_state.wave_id}")

ccs_generate_coeffs_polyphase(state.ccs_state, carrier_512)
print(f"carrier_512[:8]: {carrier_512[:8]}")
print(f"carrier_512 sum: {carrier_512.sum()}")

# Step 2: Blend with coeff_state
print("\n--- Step 2: Blend carrier into coeff_state ---")
base_mix = params.base_mix_q15
print(f"base_mix_q15: {base_mix}")
print(f"coeff_state before blend[:8]: {state.coeff_state[:8]}")

for idx in range(COEFF_LEN):
    carrier_val = int(carrier_512[idx])
    old_val = int(state.coeff_state[idx])
    inv_mix = 32767 - base_mix
    blended = RSHIFT_RNE(old_val * inv_mix + carrier_val * base_mix, 15)
    state.coeff_state[idx] = np.clip(blended, -32768, 32767)

print(f"coeff_state after blend[:8]: {state.coeff_state[:8]}")

# Step 3: lane_inject
print("\n--- Step 3: lane_inject ---")
lane_inject(state, cfg)
print(f"coeff_state after inject[:8]: {state.coeff_state[:8]}")

# Step 4: backbone (should be empty)
print("\n--- Step 4: backbone ops ---")
print(f"backbone_instrs: {cfg.backbone_instrs}")
# No ops in backbone

# Step 5: lane_tail
print("\n--- Step 5: lane_tail ---")
print(f"tail_ops: {cfg.tail_ops}")
print(f"cfg.use_coeff_osc: {cfg.use_coeff_osc}")
lane_tail(state, cfg)
print(f"coeff_state after tail[:8]: {state.coeff_state[:8]}")

# Step 6: lane_ccs_render
print("\n--- Step 6: lane_ccs_render ---")
output = lane_ccs_render(state, cfg)
print(f"output[:8]: {output[:8]}")
print(f"output sum: {output.sum()}")
print(f"output range: [{output.min()}, {output.max()}]")

# Step 7: Apply gain
print("\n--- Step 7: Apply gain ---")
print(f"gain_curr_q15: {cfg.gain_curr_q15}")
print(f"gain_prev_q15: {cfg.gain_prev_q15}")
BLOCK_SAMPLES = 64
gain_curr = cfg.gain_curr_q15
gain_prev = cfg.gain_prev_q15
if gain_curr != 32767 or gain_prev != 32767:
    print("Applying gain interpolation...")
    for i in range(BLOCK_SAMPLES):
        alpha = (i * 32767 + (BLOCK_SAMPLES >> 1)) // BLOCK_SAMPLES
        gain = gain_prev + RSHIFT_RNE((gain_curr - gain_prev) * alpha, 15)
        output[i] = RSHIFT_RNE(int(output[i]) * gain, 15)
    print(f"output after gain[:8]: {output[:8]}")
else:
    print("No gain adjustment needed (gain=32767)")

# Step 8: Advance phase
print("\n--- Step 8: Advance phase ---")
old_phase = state.ccs_state.phase
state.ccs_state.advance_block()
new_phase = state.ccs_state.phase
print(f"Phase: {old_phase} -> {new_phase} ({(old_phase/2**32)*360:.1f}° -> {(new_phase/2**32)*360:.1f}°)")

# Compare with expected
print("\n=== Compare with direct IDWPT ===")
coeffs_k0 = state.coeff_state[:COEFF_LEN].reshape(64, VM_K)[:, 0].tolist()
direct_output = idwpt_jit(coeffs_k0, headroom_bits=1)
print(f"Direct IDWPT from coeff_state[:8]: {direct_output[:8]}")
print(f"lane_ccs_render output[:8]: {output[:8]}")

# Check if coeffs were corrupted between steps
print("\n=== Coeffs at each step ===")
# Re-generate to compare
test_carrier = np.zeros(COEFF_LEN, dtype=np.int16)
test_ccs = CCSState()
test_ccs.note = 60
test_ccs.wave_id = WAVE_SINE
test_ccs.phase_inc = params.phase_inc
test_ccs.phase = 0
ccs_generate_coeffs_polyphase(test_ccs, test_carrier)

print(f"Fresh carrier[:8]: {test_carrier[:8]}")
print(f"coeff_state[:8]:   {state.coeff_state[:8]}")

# The coefficients should match after blend with base_mix=32767
test_coeff_2d = test_carrier.reshape(64, VM_K)
expected_audio = idwpt_jit(test_coeff_2d[:, 0].tolist(), headroom_bits=1)
print(f"\nExpected audio (from fresh carrier)[:8]: {expected_audio[:8]}")
print(f"Actual output[:8]: {output[:8]}")
