#!/usr/bin/env python3
"""Test audio on WASAPI device directly."""

import sounddevice as sd
import numpy as np

# Test with WASAPI device explicitly
device = 20  # Speakers (Realtek(R) Audio), Windows WASAPI
print(f"Testing device {device}: {sd.query_devices(device)['name']}")

# Generate 2 second 440Hz tone
duration = 2
sample_rate = 48000
t = np.linspace(0, duration, int(duration * sample_rate), dtype=np.float32)
audio = np.sin(2 * np.pi * 440 * t) * 0.5  # 50% volume

print(f"Audio shape: {audio.shape}, min: {audio.min():.2f}, max: {audio.max():.2f}")
print("Playing 440Hz tone for 2 seconds...")

sd.play(audio, samplerate=sample_rate, device=device)
sd.wait()
print("Done")
