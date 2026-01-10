#!/usr/bin/env python3
"""
Simple Standalone Synthesis Test - Pre-renders then plays.
No threading, just renders a note and plays it.
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

# Use device 4 (Realtek Digital Output - the one that works)
sd.default.device = 4
print(f"Using device 4: {sd.query_devices(4)['name']}")

engine = VoiceEngine(max_voices=4)
stage = UnifiedOutputStage(block_size=BLOCK_SIZE)

def render_note(note: int, duration_sec: float) -> np.ndarray:
    """Render a note for the given duration."""
    voice_id = engine.allocate_voice(program_id=1001)
    gesture = GestureBasis(gate=True, pressure_q15=Q15_ONE // 2)
    engine.note_on(voice_id, gesture=gesture)
    stage.voice_on(voice_id, note)

    total_blocks = int(duration_sec * SAMPLE_RATE / BLOCK_SIZE)
    samples = []

    f0_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))

    for _ in range(total_blocks):
        voice_frames = engine.step_all()
        active_frame = None
        for vf in voice_frames:
            if vf.is_active:
                active_frame = vf
                break

        if active_frame:
            frame = VoiceFrame(
                voice_id=active_frame.voice_id,
                note=note,
                f0_hz=f0_hz,
                gain_q15=active_frame.gain_q15,
                is_active=True,
            )
            left, right = stage.render_block(frame)
            left_float = np.array(left, dtype=np.float32) / Q15_ONE
            samples.extend(left_float)
        else:
            samples.extend([0.0] * BLOCK_SIZE)

    engine.note_off(voice_id)
    stage.voice_off(voice_id)

    return np.array(samples, dtype=np.float32)


print("Rendering note 60 (Middle C) for 2 seconds...")
audio = render_note(60, 2.0)
print(f"Rendered {len(audio)} samples, range: {audio.min():.3f} to {audio.max():.3f}")

print("Playing...")
sd.play(audio, SAMPLE_RATE)
sd.wait()
print("Done!")

print("\nRendering note 72 (C5) for 2 seconds...")
audio = render_note(72, 2.0)
print(f"Rendered {len(audio)} samples")

print("Playing...")
sd.play(audio, SAMPLE_RATE)
sd.wait()
print("Done!")

print("\nRendering note 48 (C3) for 2 seconds...")
audio = render_note(48, 2.0)
print(f"Rendered {len(audio)} samples")

print("Playing...")
sd.play(audio, SAMPLE_RATE)
sd.wait()
print("Done!")
