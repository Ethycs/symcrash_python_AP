#!/usr/bin/env python3
"""
Minimal audio test - bypasses all synthesis complexity.
Just generates a pure sine wave to verify sounddevice works.
"""

import sys
import time
import numpy as np

try:
    import sounddevice as sd
    print(f"sounddevice version: {sd.__version__}")
except ImportError:
    print("ERROR: sounddevice not installed. Run: pip install sounddevice")
    sys.exit(1)

SAMPLE_RATE = 48000
BLOCK_SIZE = 64
FREQUENCY = 440.0  # A4

print(f"Sample rate: {SAMPLE_RATE}")
print(f"Block size: {BLOCK_SIZE}")
print(f"Frequency: {FREQUENCY} Hz")

# List audio devices
print("\nAvailable audio devices:")
print(sd.query_devices())
print(f"\nDefault output device: {sd.default.device[1]}")

# Phase accumulator
phase = 0.0
phase_inc = 2.0 * np.pi * FREQUENCY / SAMPLE_RATE

def audio_callback(outdata, frames, time_info, status):
    global phase
    if status:
        print(f"Audio status: {status}")

    # Generate sine wave
    t = np.arange(frames) * phase_inc + phase
    sine = np.sin(t).astype(np.float32) * 0.3  # 30% volume

    # Update phase (wrap to avoid precision issues)
    phase = (phase + frames * phase_inc) % (2.0 * np.pi)

    # Stereo output
    outdata[:, 0] = sine
    outdata[:, 1] = sine

# Try to find a good output device (prefer Speakers over Digital Output)
output_device = None
devices = sd.query_devices()
for i, dev in enumerate(devices):
    name = dev['name']
    if 'Speakers' in name and dev['max_output_channels'] >= 2:
        if 'Realtek' in name and 'WASAPI' in name:
            output_device = i
            print(f"Found Realtek WASAPI Speakers at device {i}")
            break

if output_device is None:
    # Fall back to any speakers
    for i, dev in enumerate(devices):
        if 'Speakers' in dev['name'] and dev['max_output_channels'] >= 2:
            output_device = i
            print(f"Found Speakers at device {i}")
            break

if output_device is None:
    output_device = sd.default.device[1]
    print(f"Using default device {output_device}")

print(f"\nUsing output device {output_device}: {sd.query_devices(output_device)['name']}")
print("\nStarting audio stream...")
print("You should hear a 440 Hz sine wave for 3 seconds.")
print("If you don't hear anything, check your audio device settings.\n")

try:
    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=2,
        dtype='float32',
        callback=audio_callback,
        device=output_device,
    ) as stream:
        print("Stream started. Playing...")
        time.sleep(3.0)
        print("Done.")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
