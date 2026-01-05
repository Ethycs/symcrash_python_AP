import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.frames import VoiceFrame
import time
import numpy as np

print('Testing CCS path execution (should not hang)...\n')

# Create VM state
state = VMState.create(voice_id=0)
state.reset_for_note_on(note=60, cents=0)

# Create VoiceFrame
frame = VoiceFrame(
    voice_id=0,
    note=60,
    cents=0,
    gain_q15=16384,  # Half volume
    beta_target_q15=16384,
    # use_coeff_osc determined by decode_frame, defaults to True
)

print(f'Testing single block render...')
start = time.time()
state, output = vm_step_block(state, frame, DEFAULT_TABLES)
elapsed = time.time() - start

print(f'  Rendered in {elapsed*1000:.2f} ms')
print(f'  Output shape: {output.shape}')
print(f'  Output range: {output.min()} to {output.max()}')
print(f'  Output[0:5]: {output[0:5]}')

if elapsed < 2.0:
    print('\n[OK] CCS path executes quickly (no hang)')
else:
    print(f'\n[WARN] CCS path slow: {elapsed*1000:.0f} ms')

# Test 10 blocks to check continuity
print(f'\nTesting 10 blocks for continuity...')
all_audio = []
for block_idx in range(10):
    start = time.time()
    state, output = vm_step_block(state, frame, DEFAULT_TABLES)
    elapsed = time.time() - start

    all_audio.append(output)

    if block_idx == 0:
        print(f'  Block 0: {elapsed*1000:.2f} ms (first access, may cache tables)')
    elif block_idx == 1:
        print(f'  Block 1: {elapsed*1000:.2f} ms (cached)')

# Check continuity
all_audio_arr = np.concatenate(all_audio)
print(f'\nContinuity check ({len(all_audio_arr)} samples):')
for i in range(1, 10):
    jump = abs(int(all_audio_arr[i*64]) - int(all_audio_arr[i*64-1]))
    if jump > 1000:
        print(f'  Block {i-1}->{i}: jump={jump} [WARN]')

print('\n[DONE] CCS path test complete')
