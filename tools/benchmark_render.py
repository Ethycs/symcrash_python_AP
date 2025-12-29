#!/usr/bin/env python3
"""
Benchmark render_block performance.
"""

import sys
sys.path.insert(0, str(__file__).rsplit('tools', 1)[0] + 'src')

import time
import numpy as np
from symcrash_python_ap.voice import VoiceEngine, GestureBasis
from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.frames import VoiceFrame

BLOCK_SIZE = 64
Q15_ONE = 32767
NUM_BLOCKS = 1000

# Create engine and stage
engine = VoiceEngine(max_voices=4)
stage = UnifiedOutputStage(block_size=BLOCK_SIZE)

# Allocate and trigger a note
note = 60
voice_id = engine.allocate_voice(program_id=1001)
gesture = GestureBasis(gate=True, pressure_q15=Q15_ONE // 2)
engine.note_on(voice_id, gesture=gesture)
stage.voice_on(voice_id, note)

f0_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))

# Benchmark engine.step_all()
print(f"Benchmarking engine.step_all() for {NUM_BLOCKS} blocks...")
start = time.perf_counter()
for _ in range(NUM_BLOCKS):
    voice_frames = engine.step_all()
step_time = time.perf_counter() - start
print(f"  Total: {step_time*1000:.2f}ms, Per block: {step_time/NUM_BLOCKS*1000:.3f}ms")

# Reset for next test
engine.note_off(voice_id)
stage.voice_off(voice_id)
voice_id = engine.allocate_voice(program_id=1001)
engine.note_on(voice_id, gesture=gesture)
stage.voice_on(voice_id, note)

# Benchmark stage.render_block()
print(f"\nBenchmarking stage.render_block() for {NUM_BLOCKS} blocks...")
start = time.perf_counter()
for _ in range(NUM_BLOCKS):
    voice_frames = engine.step_all()
    active_frame = None
    for vf in voice_frames:
        if vf.is_active and vf.gain_q15 > 0:
            active_frame = vf
            break

    if active_frame:
        frame = VoiceFrame(
            voice_id=active_frame.voice_id,
            note=note,
            f0_hz=f0_hz,
            gain_q15=active_frame.gain_q15,
            is_active=True,
        )
        left, right = stage.render_block(frame)
render_time = time.perf_counter() - start
print(f"  Total: {render_time*1000:.2f}ms, Per block: {render_time/NUM_BLOCKS*1000:.3f}ms")

# Calculate real-time requirements
samples_per_block = BLOCK_SIZE
sample_rate = 44100.0
block_duration_ms = (samples_per_block / sample_rate) * 1000
print(f"\nReal-time requirement: {block_duration_ms:.3f}ms per block")
print(f"Actual render time: {render_time/NUM_BLOCKS*1000:.3f}ms per block")

if render_time/NUM_BLOCKS*1000 < block_duration_ms:
    print("=> Rendering is FASTER than real-time!")
else:
    ratio = (render_time/NUM_BLOCKS*1000) / block_duration_ms
    print(f"=> Rendering is {ratio:.1f}x SLOWER than real-time!")
