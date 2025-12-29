#!/usr/bin/env python3
"""
Save synthesized audio to WAV file and play it with system player.
This bypasses sounddevice to test if synthesis is working.
"""

import sys
import wave
import struct
import subprocess

sys.path.insert(0, str(__file__).rsplit('tools', 1)[0] + 'src')

import numpy as np
from symcrash_python_ap.voice import VoiceEngine, GestureBasis
from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.frames import VoiceFrame

SAMPLE_RATE = 48000
BLOCK_SIZE = 64
Q15_ONE = 32767
DURATION_SEC = 2.0
OUTPUT_FILE = "test_synth_output.wav"

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

all_samples = []

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

        # Interleave stereo samples
        for l, r in zip(left, right):
            all_samples.append(int(l))
            all_samples.append(int(r))
    else:
        # Silence
        for _ in range(BLOCK_SIZE):
            all_samples.append(0)
            all_samples.append(0)

print(f"Rendered {len(all_samples) // 2} stereo samples")

# Save to WAV (16-bit signed)
print(f"Saving to {OUTPUT_FILE}...")
with wave.open(OUTPUT_FILE, 'w') as wav:
    wav.setnchannels(2)
    wav.setsampwidth(2)  # 16-bit
    wav.setframerate(SAMPLE_RATE)

    # Clamp to 16-bit range
    for i, s in enumerate(all_samples):
        all_samples[i] = max(-32768, min(32767, s))

    wav.writeframes(struct.pack(f'<{len(all_samples)}h', *all_samples))

print(f"Saved! File size: {len(all_samples) * 2} bytes")

# Check audio content
samples_arr = np.array(all_samples)
print(f"Audio range: {samples_arr.min()} to {samples_arr.max()}")
print(f"Non-zero samples: {np.count_nonzero(samples_arr)}")

# Play with Windows media player
print(f"\nPlaying {OUTPUT_FILE} with system player...")
subprocess.run(['cmd', '/c', 'start', '', OUTPUT_FILE], shell=True)

print("Done! Check if you hear the audio.")
