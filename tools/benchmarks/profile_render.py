#!/usr/bin/env python3
"""
Profile render_block to find hotspots.
"""

import sys
sys.path.insert(0, str(__file__).rsplit('tools', 1)[0] + 'src')

import time
import cProfile
import pstats
from io import StringIO

from symcrash_python_ap.voice import VoiceEngine, GestureBasis
from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.frames import VoiceFrame

BLOCK_SIZE = 64
Q15_ONE = 32767
NUM_BLOCKS = 500

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

def run_render():
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

# Profile
print(f"Profiling {NUM_BLOCKS} blocks...")
profiler = cProfile.Profile()
profiler.enable()
run_render()
profiler.disable()

# Print stats
s = StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
ps.print_stats(30)
print(s.getvalue())
