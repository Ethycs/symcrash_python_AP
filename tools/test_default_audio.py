#!/usr/bin/env python3
"""Test audio on system default device."""

import sounddevice as sd
import numpy as np

print("Available output devices:")
devices = sd.query_devices()
for i, dev in enumerate(devices):
    if dev['max_output_channels'] > 0:
        marker = " <-- DEFAULT" if i == sd.default.device[1] else ""
        print(f"  {i}: {dev['name']} ({dev['max_output_channels']} ch){marker}")

print(f"\nUsing system default output: device {sd.default.device[1]}")

# Generate 2 second 440Hz tone at high volume
duration = 2
sample_rate = 44100  # Standard sample rate
t = np.linspace(0, duration, int(duration * sample_rate), dtype=np.float32)
audio = np.sin(2 * np.pi * 440 * t) * 0.8  # 80% volume - LOUD

print(f"Audio: {len(audio)} samples, {duration}s, 440Hz, 80% volume")
print("Playing NOW - you should hear a loud tone...")

# Use default device (don't specify device parameter)
sd.play(audio, samplerate=sample_rate)
sd.wait()
print("Done playing")
