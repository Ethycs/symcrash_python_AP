#!/usr/bin/env python3
"""
Test synthesis with device 4 (Realtek Digital Output) - the user's working device.
"""

import sys
import numpy as np

sys.path.insert(0, str(__file__).rsplit('tools', 1)[0] + 'src')

import sounddevice as sd
from symcrash_python_ap.voice import VoiceEngine, GestureBasis
from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.frames import VoiceFrame

SAMPLE_RATE = 44100  # Use 44100 like the working example
BLOCK_SIZE = 64
Q15_ONE = 32767
DURATION_SEC = 3.0

# Use device 4 - the user's working device
sd.default.device = 4
print(f"Using device 4: {sd.query_devices(4)['name']}")

# Create engine and stage
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
print(f"Pre-rendering {total_blocks} blocks ({DURATION_SEC}s of audio)...")

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

        left_arr = np.array(left, dtype=np.float32) / Q15_ONE
        right_arr = np.array(right, dtype=np.float32) / Q15_ONE

        all_left.append(left_arr)
        all_right.append(right_arr)
    else:
        all_left.append(np.zeros(BLOCK_SIZE, dtype=np.float32))
        all_right.append(np.zeros(BLOCK_SIZE, dtype=np.float32))

# Concatenate
audio_left = np.concatenate(all_left)
audio_right = np.concatenate(all_right)

print(f"Rendered {len(audio_left)} samples")
print(f"Audio range: {audio_left.min():.3f} to {audio_left.max():.3f}")

# Play with sounddevice using the same pattern as user's working code
print("\nPlaying...")
sd.play(audio_left, SAMPLE_RATE)  # Mono for simplicity like user's example
sd.wait()

print("Done!")

# Note off
engine.note_off(voice_id)
stage.voice_off(voice_id)
