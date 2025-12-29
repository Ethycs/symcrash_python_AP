#!/usr/bin/env python3
"""Simple beep test - plays a loud beep on WASAPI device 20."""

import sounddevice as sd
import numpy as np

DEVICE = 20
SAMPLE_RATE = 48000
DURATION = 2.0

print(f"Device {DEVICE}: {sd.query_devices(DEVICE)['name']}")
print(f"Playing {DURATION}s beep at 80% volume...")
print("YOU SHOULD HEAR THIS NOW!")

# Generate loud sine wave
t = np.linspace(0, DURATION, int(DURATION * SAMPLE_RATE), dtype=np.float32)
audio = np.sin(2 * np.pi * 440 * t) * 0.8  # 80% volume

# Stereo
stereo = np.column_stack([audio, audio])

sd.play(stereo, samplerate=SAMPLE_RATE, device=DEVICE)
sd.wait()

print("Done.")
