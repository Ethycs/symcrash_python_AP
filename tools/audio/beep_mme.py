#!/usr/bin/env python3
"""Simple beep test - plays on MME device 5 (not WASAPI)."""

import sounddevice as sd
import numpy as np

# Use MME device instead of WASAPI
DEVICE = 5  # MME Speakers
SAMPLE_RATE = 48000
DURATION = 2.0

print(f"Device {DEVICE}: {sd.query_devices(DEVICE)['name']}")
print(f"Playing {DURATION}s beep at 80% volume on MME device...")

# Generate loud sine wave
t = np.linspace(0, DURATION, int(DURATION * SAMPLE_RATE), dtype=np.float32)
audio = np.sin(2 * np.pi * 440 * t) * 0.8

# Stereo
stereo = np.column_stack([audio, audio])

sd.play(stereo, samplerate=SAMPLE_RATE, device=DEVICE)
sd.wait()

print("Done.")
