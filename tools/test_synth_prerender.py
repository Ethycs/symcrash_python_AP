#!/usr/bin/env python3
"""
Test synthesis by pre-rendering audio and then playing it.
This eliminates the real-time rendering constraint to verify the synthesis works.
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
DURATION_SEC = 3.0

# Find speakers - MUST use actual speakers, not digital output
def find_speakers():
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    # Build hostapi name lookup
    def get_hostapi_name(dev):
        return hostapis[dev['hostapi']]['name']

    # First try WASAPI Realtek Speakers (best quality, lowest latency)
    for i, dev in enumerate(devices):
        name = dev['name']
        hostapi = get_hostapi_name(dev)
        if 'Speakers' in name and 'Realtek' in name and 'WASAPI' in hostapi:
            if dev['max_output_channels'] >= 2:
                print(f"Found WASAPI Realtek Speakers at device {i}")
                return i

    # Then try any WASAPI Speakers
    for i, dev in enumerate(devices):
        name = dev['name']
        hostapi = get_hostapi_name(dev)
        if 'Speakers' in name and 'WASAPI' in hostapi:
            if dev['max_output_channels'] >= 2 and 'Digital Output' not in name:
                print(f"Found WASAPI Speakers at device {i}")
                return i

    # Then try any Realtek Speakers (MME/DirectSound)
    for i, dev in enumerate(devices):
        name = dev['name']
        if 'Speakers' in name and 'Realtek' in name:
            if dev['max_output_channels'] >= 2:
                print(f"Found Realtek Speakers at device {i}")
                return i

    # Then any speakers
    for i, dev in enumerate(devices):
        if 'Speakers' in dev['name'] and dev['max_output_channels'] >= 2:
            print(f"Found Speakers at device {i}")
            return i

    # Last resort - but warn user
    print("WARNING: Could not find speakers, using default (may be digital output)")
    return sd.default.device[1]

output_device = find_speakers()
print(f"Using audio device: {sd.query_devices(output_device)['name']}")
print(f"NOTE: Your Windows default is 'Realtek Digital Output' - change it in Sound Settings!")

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

# Pre-render all audio
total_blocks = int(DURATION_SEC * SAMPLE_RATE / BLOCK_SIZE)
print(f"\nPre-rendering {total_blocks} blocks ({DURATION_SEC}s of audio)...")

all_left = []
all_right = []
blocks_rendered = 0
max_gain = 0
max_output = 0

start_render = time.time()

for i in range(total_blocks):
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

        left_arr = np.array(left, dtype=np.float32) / Q15_ONE
        right_arr = np.array(right, dtype=np.float32) / Q15_ONE

        all_left.append(left_arr)
        all_right.append(right_arr)
        blocks_rendered += 1

        max_gain = max(max_gain, active_frame.gain_q15)
        max_output = max(max_output, np.abs(left_arr).max(), np.abs(right_arr).max())
    else:
        # Voice inactive - fill with silence
        all_left.append(np.zeros(BLOCK_SIZE, dtype=np.float32))
        all_right.append(np.zeros(BLOCK_SIZE, dtype=np.float32))

render_time = time.time() - start_render
print(f"Rendered {blocks_rendered} active blocks in {render_time:.2f}s")
print(f"Max gain_q15: {max_gain}, Max output: {max_output:.3f}")

# Concatenate into single arrays
audio_left = np.concatenate(all_left)
audio_right = np.concatenate(all_right)
audio_stereo = np.column_stack([audio_left, audio_right])

print(f"Total samples: {len(audio_left)}")
print(f"Audio duration: {len(audio_left) / SAMPLE_RATE:.2f}s")
print(f"Audio range: {audio_stereo.min():.3f} to {audio_stereo.max():.3f}")

# Save to WAV and play with system player (more reliable than sounddevice on Windows)
import subprocess
from symcrash_python_ap.audio_io import write_audio

print("\n" + "="*60)
print("Saving audio using audio_io...")
print("="*60)

# Convert to 16-bit integer samples
L_int16 = (audio_left * 32767).astype(np.int16)
R_int16 = (audio_right * 32767).astype(np.int16)

meta = write_audio("synth_output", L_int16, R_int16, timestamp=False, subdirectory="tools")

print(f"Saved! Playing with system player...")

# Play with Windows default player
subprocess.Popen(['cmd', '/c', 'start', '', meta.filepath], shell=True)

print("Done!")

# Note off
engine.note_off(voice_id)
stage.voice_off(voice_id)
print("Note released.")
