"""Test live rendering pattern (like pico2_live_synth)."""
import numpy as np
import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.audio_io import write_audio

Q15_ONE = 32767
BLOCK_SIZE = 64
SAMPLE_RATE = 48000

# Create output stage (persists across blocks like in pico2_live_synth)
stage = UnifiedOutputStage(block_size=BLOCK_SIZE)
stage.voice_on(0, 60, 0)

# Render 2 seconds of audio - creating new VoiceFrame each block
duration_sec = 2.0
num_blocks = int(duration_sec * SAMPLE_RATE / BLOCK_SIZE)
all_L = []
all_R = []

print(f"Rendering {num_blocks} blocks ({duration_sec}s)...")
print("Creating new VoiceFrame each block (like pico2_live_synth)...")

for i in range(num_blocks):
    # Create new VoiceFrame each block (like pico2_live_synth does)
    frame = VoiceFrame(
        voice_id=0,
        note=60,
        cents=0,
        f0_hz=261.63,
        gain_q15=24576,
        pan_q15=0,
        tree_a_id=0,
        tree_b_id=0,
        tree_blend_q15=0,
        is_active=True,
        beta_target_q15=0,  # No wet/wavelet (pure carrier)
        partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],  # Pure sine
    )

    L, R = stage.render_block(frame)
    all_L.extend(L)
    all_R.extend(R)

# Convert to numpy arrays
L_np = np.array(all_L, dtype=np.int16)
R_np = np.array(all_R, dtype=np.int16)

# Check for spikes
d2 = np.diff(np.diff(L_np.astype(np.int32)))
threshold = d2.std() * 3
spikes = np.where(np.abs(d2) > threshold)[0]
print(f"\nSpike analysis: {len(spikes)} spikes detected (threshold={threshold:.0f})")
if len(spikes) > 0:
    print(f"Max d2: {d2.max()}, Min d2: {d2.min()}")
else:
    print("No spikes - output is clean!")

# Write to file
output_path = write_audio("test_live_render_sine", L_np, R_np)
print(f"\nWrote audio to: {output_path}")
