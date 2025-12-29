#!/usr/bin/env python3
"""
Test callback pattern with simple sine wave.
Confirms the callback pattern works on this system.
"""

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100
FREQ = 440.0

# Use device 4 (Realtek Digital Output - proven working)
sd.default.device = 4
print(f"Using device 4: {sd.query_devices(4)['name']}")

phase = 0.0

def audio_callback(outdata, frames, time_info, status):
    global phase
    if status:
        print(f"Audio status: {status}")

    t = (np.arange(frames) + phase) / SAMPLE_RATE
    outdata[:, 0] = 0.3 * np.sin(2 * np.pi * FREQ * t)
    phase += frames


print("Playing 440Hz sine wave for 2 seconds...")
with sd.OutputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    callback=audio_callback,
):
    sd.sleep(2000)

print("Done!")
