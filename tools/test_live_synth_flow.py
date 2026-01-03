"""Simulate pico2_live_synth flow to debug ETA noise."""
import numpy as np
from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.frames import VoiceFrame


def create_silent_frame():
    """Frame with no active voice (gain=0)."""
    return VoiceFrame(
        voice_id=0,
        note=60,
        cents=0,
        gain_q15=0,  # Silent
        pan_q15=0,
        tree_a_id=0,
        tree_b_id=0,
        tree_blend_q15=0,
        is_active=False,
    )


def create_active_frame(note=69, gain=32767, eta=24576, harmonics=0):
    """Frame with active voice."""
    # Compute partial amps from harmonics
    base_amps = [32767, 16384, 10923, 8192, 6554, 5461, 4681, 4096]
    partial_amps = [base_amps[0]]
    for amp in base_amps[1:]:
        partial_amps.append((amp * harmonics) >> 15)

    return VoiceFrame(
        voice_id=0,
        note=note,
        cents=0,
        gain_q15=gain,
        pan_q15=0,
        tree_a_id=0,
        tree_b_id=0,
        tree_blend_q15=0,
        is_active=True,
        beta_target_q15=eta,
        partial_amps_q15=partial_amps,
    )


print("=" * 60)
print("Simulating pico2_live_synth flow")
print("=" * 60)

# Create VM state
vm_state = VMState.create(voice_id=0)

print("\n--- Phase 1: Silent (no note playing) ---")
for i in range(5):
    frame = create_silent_frame()
    vm_state, output = vm_step_block(vm_state, frame, DEFAULT_TABLES)
    max_val = np.abs(output).max()
    print(f"Block {i}: max_abs={max_val}")

print("\n--- Phase 2: Note ON (with default eta from VoiceState = 16384) ---")
# Reset for note on
vm_state.reset_for_note_on(note=69, cents=0)
for i in range(5):
    # eta defaults to 16384 in VoiceState
    frame = create_active_frame(note=69, gain=32767, eta=16384)
    vm_state, output = vm_step_block(vm_state, frame, DEFAULT_TABLES)
    max_val = np.abs(output).max()
    print(f"Block {i}: max_abs={max_val}, first4={output[:4].tolist()}")

print("\n--- Phase 3: ETA pot touched (changes to 0) ---")
for i in range(5):
    frame = create_active_frame(note=69, gain=32767, eta=0)
    vm_state, output = vm_step_block(vm_state, frame, DEFAULT_TABLES)
    max_val = np.abs(output).max()
    print(f"Block {i}: max_abs={max_val}, first4={output[:4].tolist()}")

print("\n--- Phase 4: ETA pot moved to max (32767) ---")
for i in range(5):
    frame = create_active_frame(note=69, gain=32767, eta=32767)
    vm_state, output = vm_step_block(vm_state, frame, DEFAULT_TABLES)
    max_val = np.abs(output).max()
    print(f"Block {i}: max_abs={max_val}, first4={output[:4].tolist()}")

print("\n--- Phase 5: Compare outputs with different eta values ---")
# Reset state
vm_state_a = VMState.create(voice_id=0)
vm_state_a.reset_for_note_on(note=69, cents=0)
vm_state_b = VMState.create(voice_id=0)
vm_state_b.reset_for_note_on(note=69, cents=0)

# Warm up both
for _ in range(10):
    frame_a = create_active_frame(eta=0)
    frame_b = create_active_frame(eta=32767)
    vm_state_a, _ = vm_step_block(vm_state_a, frame_a, DEFAULT_TABLES)
    vm_state_b, _ = vm_step_block(vm_state_b, frame_b, DEFAULT_TABLES)

# Compare
frame_a = create_active_frame(eta=0)
frame_b = create_active_frame(eta=32767)
vm_state_a, out_a = vm_step_block(vm_state_a, frame_a, DEFAULT_TABLES)
vm_state_b, out_b = vm_step_block(vm_state_b, frame_b, DEFAULT_TABLES)

print(f"ETA=0:     {out_a[:8].tolist()}")
print(f"ETA=32767: {out_b[:8].tolist()}")
print(f"Difference: {np.abs(out_a.astype(int) - out_b.astype(int)).max()}")

if np.array_equal(out_a, out_b):
    print("\nOutputs are IDENTICAL - ETA has no effect in CCS path")
else:
    print("\nOutputs DIFFER - something is wrong")
