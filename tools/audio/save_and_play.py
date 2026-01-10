#!/usr/bin/env python3
"""
Save synthesized audio to WAV file and play it with system player.
This bypasses sounddevice to test if synthesis is working.
"""

import sys
import subprocess

sys.path.insert(0, str(__file__).rsplit('tools', 1)[0] + 'src')

import numpy as np
from symcrash_python_ap.voice import VoiceEngine, GestureBasis
from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import write_audio

SAMPLE_RATE = 48000
BLOCK_SIZE = 64
Q15_ONE = 32767
DURATION_SEC = 2.0

print("Creating VoiceEngine and UnifiedOutputStage...")
engine = VoiceEngine(max_voices=4)
stage = UnifiedOutputStage(block_size=BLOCK_SIZE)

# Allocate a voice
note = 60  # Middle C
print(f"Allocating voice for note {note}...")
voice_id = engine.allocate_voice(program_id=1001)

# Trigger note on
gesture = GestureBasis(gate=True, pressure_q15=Q15_ONE // 2)
engine.note_on(voice_id, gesture=gesture)
stage.voice_on(voice_id, note)

# Pre-render all audio
total_blocks = int(DURATION_SEC * SAMPLE_RATE / BLOCK_SIZE)
print(f"Pre-rendering {total_blocks} blocks...")

all_left = []
all_right = []

for i in range(total_blocks):
    voice_frames = engine.step_all()

    active_frame = None
    for vf in voice_frames:
        if vf.is_active:
            active_frame = vf
            break

    if active_frame:
        f0_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))
        frame = VoiceFrame(
            voice_id=active_frame.voice_id,
            note=note,
            f0_hz=f0_hz,
            gain_q15=active_frame.gain_q15,
            is_active=True,
        )

        left, right = stage.render_block(frame)
        all_left.extend(left)
        all_right.extend(right)
    else:
        # Silence
        all_left.extend([0] * BLOCK_SIZE)
        all_right.extend([0] * BLOCK_SIZE)

print(f"Rendered {len(all_left)} stereo samples")

# Convert to numpy arrays
L = np.array(all_left, dtype=np.int16)
R = np.array(all_right, dtype=np.int16)

# Save using centralized audio_io
meta = write_audio("test_synth_output", L, R, timestamp=False, subdirectory="tools")

# Check audio content
print(f"Audio range: {L.min()} to {L.max()}")
print(f"Non-zero samples: {np.count_nonzero(L)}")

# Play with Windows media player
print(f"\nPlaying {meta.filepath} with system player...")
subprocess.run(['cmd', '/c', 'start', '', meta.filepath], shell=True)

print("Done! Check if you hear the audio.")
