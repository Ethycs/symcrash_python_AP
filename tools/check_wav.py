#!/usr/bin/env python3
import wave
import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else "audio_out/test_pipeline_sine-20260102-094932.wav"
with wave.open(filepath, 'r') as f:
    print("File:", filepath)
    print("Channels:", f.getnchannels())
    print("Sample width:", f.getsampwidth())
    print("Frame rate:", f.getframerate())
    print("Frames:", f.getnframes())
    print("Duration:", f.getnframes() / f.getframerate(), "seconds")
