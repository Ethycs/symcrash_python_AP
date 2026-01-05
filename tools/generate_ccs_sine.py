"""Generate a CCS sine wave audio file to verify the implementation."""

import numpy as np
import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import write_audio

# Generate 2 seconds at 440 Hz (note 69)
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

print("Generating 2 seconds of 440 Hz sine via CCS path...")

samples = []
num_blocks = int(2.0 * 48000 / 64)  # 2 seconds

for i in range(num_blocks):
    state, output = vm_step_block(state, frame, DEFAULT_TABLES)
    samples.extend(output.tolist())

samples = np.array(samples, dtype=np.int16)
print(f"Generated {len(samples)} samples")
print(f"RMS: {np.sqrt(np.mean(samples.astype(float)**2)):.1f}")
print(f"Peak: {np.max(np.abs(samples))}")

# Write as stereo file (same on both channels)
meta = write_audio('ccs_sine_440hz', samples, samples)
print(f"Audio file written: {meta.filepath}")
