#!/usr/bin/env python3
"""Test sounddevice with fresh initialization."""

import sounddevice as sd
import numpy as np
import time

# Force reset
sd._terminate()
sd._initialize()

print("Sounddevice reset.")
print(f"Default output: {sd.default.device[1]}")

# Use DirectSound instead of WASAPI (more compatible)
DEVICE = 15  # DirectSound Speakers
print(f"Using device {DEVICE}: {sd.query_devices(DEVICE)['name']}")

# Simple blocking play
print("Playing 2 second tone...")
t = np.linspace(0, 2, 96000, dtype=np.float32)
audio = np.sin(2 * np.pi * 440 * t) * 0.5

sd.play(audio, samplerate=48000, device=DEVICE, blocking=True)
print("Done.")
