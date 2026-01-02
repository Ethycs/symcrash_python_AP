"""Generate test audio to hear the spikes/jaggies."""
import numpy as np
import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.audio_io import write_audio

Q15_ONE = 32767
BLOCK_SIZE = 64
SAMPLE_RATE = 48000

# Create a pure sine VoiceFrame (harm=0)
frame = VoiceFrame(
    voice_id=0,
    note=60,  # Middle C
    cents=0,
    f0_hz=261.63,
    gain_q15=24576,
    pan_q15=0,  # Center
    tree_a_id=0,
    tree_b_id=0,
    tree_blend_q15=0,
    is_active=True,
    beta_target_q15=0,  # No wet/wavelet
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],  # Pure sine
)

# Create output stage
stage = UnifiedOutputStage(block_size=BLOCK_SIZE)
stage.voice_on(0, frame.note, frame.cents)

# Render 2 seconds of audio
duration_sec = 2.0
num_blocks = int(duration_sec * SAMPLE_RATE / BLOCK_SIZE)
all_L = []
all_R = []

print(f"Rendering {num_blocks} blocks ({duration_sec}s at {SAMPLE_RATE}Hz)...")

for i in range(num_blocks):
    L, R = stage.render_block(frame)
    all_L.extend(L)
    all_R.extend(R)

# Convert to numpy arrays
L_np = np.array(all_L, dtype=np.int16)
R_np = np.array(all_R, dtype=np.int16)

# Write to file
output_path = write_audio("test_pipeline_sine", L_np, R_np)
print(f"Wrote audio to: {output_path}")
