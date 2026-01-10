"""Debug CCS state comparison."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.vm.coeff_lut_streaming import CCSState, ccs_generate_coeffs_polyphase, NUM_LEAVES, VM_K
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit

# Check VM state's CCS
vm_state = VMState.create(voice_id=0)
vm_state.reset_for_note_on(note=60, cents=0)
print('VM CCSState (after reset_for_note_on):')
print(f'  phase: {vm_state.ccs_state.phase}')
print(f'  phase_inc: {vm_state.ccs_state.phase_inc}')
print(f'  note: {vm_state.ccs_state.note}')
print(f'  wave_id: {vm_state.ccs_state.wave_id}')

# Check direct CCSState with same phase_inc
direct_ccs = CCSState()
direct_ccs.note = 60
direct_ccs.wave_id = 0  # WAVE_SINE
direct_ccs.phase_inc = vm_state.ccs_state.phase_inc  # Use same phase_inc
direct_ccs.phase = 0
print()
print('Direct CCSState:')
print(f'  phase: {direct_ccs.phase}')
print(f'  phase_inc: {direct_ccs.phase_inc}')
print(f'  note: {direct_ccs.note}')
print(f'  wave_id: {direct_ccs.wave_id}')

# Generate 5 blocks from each and compare audio
print("\n=== Audio Comparison ===")

def gen_audio_from_ccs(ccs, num_blocks=5):
    """Generate audio from CCSState."""
    all_audio = []
    for _ in range(num_blocks):
        carrier = np.zeros(NUM_LEAVES * VM_K, dtype=np.int16)
        ccs_generate_coeffs_polyphase(ccs, carrier)
        coeff_2d = carrier.reshape(NUM_LEAVES, VM_K)
        coeffs = coeff_2d[:, 0].tolist()
        audio = idwpt_jit(coeffs, headroom_bits=1)
        all_audio.append(np.array(audio, dtype=np.int16))
        ccs.advance_block()
    return np.concatenate(all_audio)

# Reset both to phase=0
vm_state.ccs_state.phase = 0
direct_ccs.phase = 0

# Generate from VM CCS state
audio_vm = gen_audio_from_ccs(vm_state.ccs_state)
print(f"VM CCS audio: min={audio_vm.min()}, max={audio_vm.max()}, mean={audio_vm.mean():.1f}")

# Reset and generate from direct CCS state
direct_ccs.phase = 0
audio_direct = gen_audio_from_ccs(direct_ccs)
print(f"Direct CCS audio: min={audio_direct.min()}, max={audio_direct.max()}, mean={audio_direct.mean():.1f}")

# Compare
diff = np.abs(audio_vm.astype(int) - audio_direct.astype(int))
print(f"Difference: max={diff.max()}, mean={diff.mean():.1f}")

# Check block boundary jumps
print("\n=== Block Boundary Jumps ===")
for name, audio in [("VM CCS", audio_vm), ("Direct", audio_direct)]:
    jumps = []
    for i in range(1, 5):
        idx = i * 64
        jump = abs(int(audio[idx]) - int(audio[idx - 1]))
        jumps.append(jump)
    print(f"{name}: jumps={jumps}, max={max(jumps)}, avg={np.mean(jumps):.0f}")

# Now compare against what vm_step_block produces
print("\n=== VM Step Block Comparison ===")
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm import vm_step_block
from symcrash_python_ap.vm.tables import DEFAULT_TABLES

# Create fresh state
state = VMState.create(voice_id=0)
state.reset_for_note_on(note=60, cents=0)

frame = VoiceFrame(
    voice_id=0,
    note=60,
    cents=0,
    gain_q15=32767,
    beta_target_q15=32767,
)

# Run 5 blocks
all_output = []
for i in range(5):
    state, output = vm_step_block(state, frame, DEFAULT_TABLES)
    all_output.append(output)

audio_step = np.concatenate(all_output)
print(f"vm_step_block audio: min={audio_step.min()}, max={audio_step.max()}, mean={audio_step.mean():.1f}")

# Check boundary jumps
jumps = []
for i in range(1, 5):
    idx = i * 64
    jump = abs(int(audio_step[idx]) - int(audio_step[idx - 1]))
    jumps.append(jump)
print(f"vm_step_block jumps: {jumps}, max={max(jumps)}, avg={np.mean(jumps):.0f}")

# Compare first block of each
print("\n=== First Block Comparison ===")
print(f"Direct audio[:8]: {audio_direct[:8]}")
print(f"VM step audio[:8]: {audio_step[:8]}")

# Check phase after each block
print("\n=== Phase Progression ===")
state2 = VMState.create(voice_id=0)
state2.reset_for_note_on(note=60, cents=0)
print(f"Initial phase: {state2.ccs_state.phase}")

for i in range(5):
    old_phase = state2.ccs_state.phase
    state2, _ = vm_step_block(state2, frame, DEFAULT_TABLES)
    new_phase = state2.ccs_state.phase
    phase_deg_before = (old_phase / (2**32)) * 360
    phase_deg_after = (new_phase / (2**32)) * 360
    print(f"Block {i}: phase {phase_deg_before:.1f}° -> {phase_deg_after:.1f}°")
