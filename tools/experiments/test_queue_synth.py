#!/usr/bin/env python3
"""
Test queue-based synth - renders in background, plays from queue.
Runs for 5 seconds automatically.
"""

import sys
sys.path.insert(0, str(__file__).rsplit('tools', 1)[0] + 'src')

import threading
import queue
import numpy as np
import sounddevice as sd
from symcrash_python_ap.voice import VoiceEngine, GestureBasis
from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.frames import VoiceFrame

SAMPLE_RATE = 44100
BLOCK_SIZE = 64
Q15_ONE = 32767
CHUNK_BLOCKS = 50  # ~73ms per chunk

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

# Shared state
audio_queue = queue.Queue(maxsize=4)
running = True
blocks_rendered = 0


def render_thread():
    """Background thread that renders audio chunks."""
    global blocks_rendered
    while running:
        chunk_samples = []

        for _ in range(CHUNK_BLOCKS):
            voice_frames = engine.step_all()

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

                left, right = stage.render_block(frame)
                left_float = np.array(left, dtype=np.float32) / Q15_ONE
                chunk_samples.extend(left_float)
            else:
                chunk_samples.extend([0.0] * BLOCK_SIZE)

            blocks_rendered += 1

        if chunk_samples:
            audio = np.array(chunk_samples, dtype=np.float32)
            try:
                audio_queue.put(audio, timeout=0.5)
            except queue.Full:
                pass


def play_thread():
    """Background thread that plays audio from queue."""
    while running:
        try:
            audio = audio_queue.get(timeout=0.1)
            sd.play(audio, SAMPLE_RATE)
            sd.wait()
        except queue.Empty:
            pass


print("Starting render and play threads...")
render_t = threading.Thread(target=render_thread, daemon=True)
play_t = threading.Thread(target=play_thread, daemon=True)

render_t.start()
play_t.start()

print("Playing for 5 seconds...")
sd.sleep(5000)

running = False
render_t.join(timeout=1.0)
play_t.join(timeout=1.0)

print(f"Done! Rendered {blocks_rendered} blocks, queue size: {audio_queue.qsize()}")
