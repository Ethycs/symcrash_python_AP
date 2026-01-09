"""Trace VM step block across multiple blocks to find corruption."""
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
from symcrash_python_ap.vm.decode import decode_frame
from symcrash_python_ap.vm.lanes import lane_inject, lane_tail, lane_ccs_render
from symcrash_python_ap.fixed import RSHIFT_RNE

print("=== Multi-block trace ===\n")

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

# Also create a "reference" CCS for expected values
ref_ccs = CCSState()
ref_ccs.note = 60
ref_ccs.wave_id = WAVE_SINE
ref_ccs.phase = 0

all_output = []

for block in range(5):
    print(f"\n{'='*60}")
    print(f"BLOCK {block}")
    print(f"{'='*60}")

    # Decode (fresh each block to match vm_step_block behavior)
    cfg = decode_frame(frame, DEFAULT_TABLES,
                       prev_eta_q15=state.eta_prev_q15,
                       prev_gain_q15=state.gain_prev_q15)

    params = cfg.coeff_osc_params

    # Set ref_ccs phase_inc from params (first block only)
    if block == 0:
        ref_ccs.phase_inc = params.phase_inc

    print(f"\nState before block:")
    print(f"  ccs_state.phase: {state.ccs_state.phase} ({(state.ccs_state.phase/2**32)*360:.1f}°)")
    print(f"  ref_ccs.phase:   {ref_ccs.phase} ({(ref_ccs.phase/2**32)*360:.1f}°)")
    print(f"  coeff_state[:8]: {state.coeff_state[:8]}")

    # Update state from params (executor.py lines 114-119)
    pitch_changed = state.last_note != params.note or state.last_cents != params.cents
    if pitch_changed:
        state.ccs_state.note = params.note
        state.ccs_state.wave_id = params.table_id
        state.last_note = params.note
        state.last_cents = params.cents

    if state.ccs_state.phase_inc != params.phase_inc:
        state.ccs_state.set_pitch(params.phase_inc, slew_blocks=8)

    # Generate carrier (executor.py lines 197-198)
    carrier_512 = np.zeros(COEFF_LEN, dtype=np.int16)
    ccs_generate_coeffs_polyphase(state.ccs_state, carrier_512)

    # Also generate expected from reference
    ref_carrier = np.zeros(COEFF_LEN, dtype=np.int16)
    ccs_generate_coeffs_polyphase(ref_ccs, ref_carrier)

    print(f"\nCarrier comparison:")
    print(f"  state carrier[:8]: {carrier_512[:8]}")
    print(f"  ref carrier[:8]:   {ref_carrier[:8]}")
    carrier_match = np.array_equal(carrier_512, ref_carrier)
    print(f"  Match: {carrier_match}")

    # Blend (executor.py lines 212-223)
    base_mix = params.base_mix_q15
    for idx in range(COEFF_LEN):
        carrier_val = int(carrier_512[idx])
        old_val = int(state.coeff_state[idx])
        inv_mix = 32767 - base_mix
        blended = RSHIFT_RNE(old_val * inv_mix + carrier_val * base_mix, 15)
        state.coeff_state[idx] = np.clip(blended, -32768, 32767)

    print(f"\nCoeff state after blend[:8]: {state.coeff_state[:8]}")

    # lane_inject (no events)
    lane_inject(state, cfg)

    # lane_tail
    lane_tail(state, cfg)
    print(f"Coeff state after tail[:8]: {state.coeff_state[:8]}")

    # lane_ccs_render
    output = lane_ccs_render(state, cfg)
    print(f"\nOutput[:8]: {output[:8]}")
    print(f"Output[-8:]: {output[-8:]}")

    # Expected output from ref
    ref_coeffs = ref_carrier.reshape(64, VM_K)[:, 0].tolist()
    ref_output = idwpt_jit(ref_coeffs, headroom_bits=1)
    print(f"Expected[:8]: {ref_output[:8]}")
    print(f"Expected[-8:]: {ref_output[-8:]}")

    output_match = all(output[i] == ref_output[i] for i in range(64))
    print(f"Output match: {output_match}")

    all_output.append(output.copy())

    # Advance phases
    state.ccs_state.advance_block()
    ref_ccs.advance_block()

    # Update state for next block (from executor.py)
    state.blocks_since_start += 1
    state.eta_prev_q15 = cfg.eta_curr_q15
    state.gain_prev_q15 = cfg.gain_curr_q15

# Check block boundary jumps
print(f"\n{'='*60}")
print("BLOCK BOUNDARY ANALYSIS")
print(f"{'='*60}")

audio = np.concatenate(all_output)
for i in range(1, 5):
    idx = i * 64
    prev_val = audio[idx - 1]
    curr_val = audio[idx]
    jump = abs(int(curr_val) - int(prev_val))
    print(f"Block {i-1} -> {i}: last={prev_val}, first={curr_val}, jump={jump}")
