#!/usr/bin/env python3
"""
Test the synthesis pipeline directly - no Pico2 hardware needed.
Plays a note using VoiceEngine + UnifiedOutputStage.
"""

import sys
import time
import numpy as np

sys.path.insert(0, str(__file__).rsplit('tools', 1)[0] + 'src')

import sounddevice as sd
from symcrash_python_ap.voice import VoiceEngine, GestureBasis
from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.frames import VoiceFrame

SAMPLE_RATE = 48000
BLOCK_SIZE = 64
Q15_ONE = 32767

# Find speakers
def find_speakers():
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        name = dev['name']
        if 'Speakers' in name and dev['max_output_channels'] >= 2:
            if 'Realtek' in name and 'WASAPI' in name:
                return i
    for i, dev in enumerate(devices):
        if 'Speakers' in dev['name'] and dev['max_output_channels'] >= 2:
            return i
    return sd.default.device[1]

output_device = find_speakers()
print(f"Using audio device: {sd.query_devices(output_device)['name']}")

# Create engine and stage
print("Creating VoiceEngine and UnifiedOutputStage...")
engine = VoiceEngine(max_voices=4)
stage = UnifiedOutputStage(block_size=BLOCK_SIZE)

# Allocate a voice
note = 60  # Middle C
print(f"Allocating voice for note {note}...")
voice_id = engine.allocate_voice(program_id=1001)
print(f"  Voice ID: {voice_id}")

# Trigger note on
gesture = GestureBasis(gate=True, pressure_q15=Q15_ONE // 2)
engine.note_on(voice_id, gesture=gesture)
stage.voice_on(voice_id, note)

# Render a few blocks to check output
print("\nRendering test blocks...")
for i in range(10):
    voice_frames = engine.step_all()

    # Find active frame
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
            cents=0,
            f0_hz=f0_hz,
            gain_q15=active_frame.gain_q15,
            pan_q15=0,
            tree_a_id=active_frame.tree_a_id,
            tree_b_id=active_frame.tree_b_id,
            tree_blend_q15=active_frame.tree_blend_q15,
            is_active=True,
        )

        left, right = stage.render_block(frame)

        # Check output
        left_arr = np.array(left, dtype=np.float32)
        right_arr = np.array(right, dtype=np.float32)
        max_val = max(np.abs(left_arr).max(), np.abs(right_arr).max())

        print(f"  Block {i}: gain_q15={active_frame.gain_q15} max_output={max_val:.1f} "
              f"(range: {left_arr.min():.0f} to {left_arr.max():.0f})")
    else:
        print(f"  Block {i}: No active voice!")

# If we got here, try real-time playback
print("\n" + "="*60)
print("Playing note for 3 seconds...")
print("="*60)

import queue
audio_queue = queue.Queue(maxsize=16)
blocks_rendered = 0

callback_count = 0
underruns = 0

def audio_callback(outdata, frames, time_info, status):
    global blocks_rendered, callback_count, underruns
    callback_count += 1
    if status:
        print(f"Audio status: {status}")

    # The callback might request multiple blocks worth of samples
    # We need to handle the actual requested frame count
    if callback_count <= 3:
        print(f"  Callback {callback_count}: frames={frames}, outdata.shape={outdata.shape}")

    samples_written = 0
    while samples_written < frames:
        try:
            left, right = audio_queue.get_nowait()
            block_len = len(left)
            end_idx = min(samples_written + block_len, frames)
            actual_len = end_idx - samples_written

            left_float = np.array(left[:actual_len], dtype=np.float32) / Q15_ONE
            right_float = np.array(right[:actual_len], dtype=np.float32) / Q15_ONE

            outdata[samples_written:end_idx, 0] = left_float
            outdata[samples_written:end_idx, 1] = right_float
            samples_written = end_idx
        except queue.Empty:
            underruns += 1
            outdata[samples_written:, :] = 0
            break

    if callback_count <= 5 and samples_written > 0:
        print(f"  Callback {callback_count}: max={np.abs(outdata).max():.3f}")

# Pre-fill buffer
print("Pre-filling audio buffer...")
for _ in range(8):
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
        audio_queue.put((left, right))
        blocks_rendered += 1

print(f"Buffer filled with {blocks_rendered} blocks")

# Start stream - use larger buffer to avoid underruns
with sd.OutputStream(
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    channels=2,
    dtype='float32',
    callback=audio_callback,
    device=output_device,
    latency='high',  # Use larger buffer
) as stream:
    print("Stream started!")

    start_time = time.time()
    inactive_count = 0
    while time.time() - start_time < 3.0:
        # Render more blocks - keep filling aggressively
        while audio_queue.qsize() < 12:
            voice_frames = engine.step_all()
            active_frame = None
            for vf in voice_frames:
                if vf.is_active:
                    active_frame = vf
                    break

            if active_frame:
                inactive_count = 0
                f0_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))
                frame = VoiceFrame(
                    voice_id=active_frame.voice_id,
                    note=note,
                    f0_hz=f0_hz,
                    gain_q15=active_frame.gain_q15,
                    is_active=True,
                )
                left, right = stage.render_block(frame)
                try:
                    audio_queue.put_nowait((left, right))
                    blocks_rendered += 1
                except queue.Full:
                    break
            else:
                inactive_count += 1
                if inactive_count == 1:
                    print(f"Voice became inactive at block {blocks_rendered}")
                break

        # No sleep - render as fast as possible

    print(f"Done! Rendered {blocks_rendered} blocks, {callback_count} callbacks, {underruns} underruns.")

# Note off
engine.note_off(voice_id)
stage.voice_off(voice_id)
print("Note released.")
