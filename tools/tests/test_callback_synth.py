#!/usr/bin/env python3
"""
Test callback-based synth - plays for 3 seconds and exits.
Uses the proven sd.OutputStream callback pattern.
"""

import sys
sys.path.insert(0, str(__file__).rsplit('tools', 1)[0] + 'src')

import numpy as np
import sounddevice as sd
from symcrash_python_ap.voice import VoiceEngine, GestureBasis
from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.frames import VoiceFrame

SAMPLE_RATE = 44100
BLOCK_SIZE = 64
Q15_ONE = 32767

# Use device 4 (Realtek Digital Output - proven working)
sd.default.device = 4
print(f"Using device 4: {sd.query_devices(4)['name']}")

# Create engine and stage
engine = VoiceEngine(max_voices=4)
stage = UnifiedOutputStage(block_size=BLOCK_SIZE)

# Allocate and trigger a note
note = 60  # Middle C
voice_id = engine.allocate_voice(program_id=1001)
gesture = GestureBasis(gate=True, pressure_q15=Q15_ONE // 2)
engine.note_on(voice_id, gesture=gesture)
stage.voice_on(voice_id, note)

f0_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))
blocks_rendered = 0

def audio_callback(outdata, frames, time_info, status):
    """Audio callback - called by sounddevice to fill output buffer."""
    global blocks_rendered

    if status:
        print(f"Audio status: {status}")

    num_blocks = frames // BLOCK_SIZE
    remainder = frames % BLOCK_SIZE
    output_idx = 0

    for _ in range(num_blocks):
        voice_frames = engine.step_all()

        # Find first active voice
        active_frame = None
        for vf in voice_frames:
            if vf.is_active and vf.gain_q15 > 0:
                active_frame = vf
                break

        if active_frame is not None:
            frame = VoiceFrame(
                voice_id=active_frame.voice_id,
                note=note,
                f0_hz=f0_hz,
                gain_q15=active_frame.gain_q15,
                is_active=True,
            )

            # Use render_voice() directly (faster, mono output)
            samples = stage.render_voice(frame)
            # samples is a list - use numpy slice assignment (faster than loop)
            outdata[output_idx:output_idx + BLOCK_SIZE, 0] = np.array(samples, dtype=np.float32) / Q15_ONE
        else:
            outdata[output_idx:output_idx + BLOCK_SIZE, 0] = 0.0

        output_idx += BLOCK_SIZE
        blocks_rendered += 1

    # Fill remainder with silence
    if remainder > 0:
        outdata[output_idx:output_idx + remainder, 0] = 0.0


print("Starting audio stream (3 seconds)...")
with sd.OutputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    callback=audio_callback,
    blocksize=BLOCK_SIZE * 16,  # Larger buffer for more render time
    latency='high',  # Request high latency mode for stability
):
    sd.sleep(3000)

print(f"Done! Rendered {blocks_rendered} blocks")
