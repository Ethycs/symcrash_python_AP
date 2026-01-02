#!/usr/bin/env python3
"""Benchmark the full VM step_block with fast IDWPT."""

import time
import cProfile
import pstats
import io
import numpy as np

# Import VM components
from symcrash_python_ap.vm import vm_step_block, VMState, Tables
from symcrash_python_ap.frames import VoiceFrame

# Create test frame
frame = VoiceFrame(
    voice_id=1,
    note=60,
    cents=0,
    f0_hz=261.63,  # Middle C
    gain_q15=24576,
    beta_target_q15=16384,  # 50% wet
    partial_amps_q15=[32767, 16384, 10923, 8192, 6554, 5461, 4681, 4096],
    is_active=True,
)

# Create VM state
tables = Tables()
state = VMState.create(1)
state.reset_for_note_on(f0_hz=frame.f0_hz)

# Warmup
for _ in range(100):
    state, output = vm_step_block(state, frame, tables)

# Benchmark
N = 1000
start = time.perf_counter()
for _ in range(N):
    state, output = vm_step_block(state, frame, tables)
elapsed = time.perf_counter() - start

us_per_block = elapsed * 1e6 / N
blocks_per_sec = N / elapsed

print(f"VM step_block: {us_per_block:.1f} us/block")
print(f"Blocks/sec: {blocks_per_sec:.0f}")
print(f"Realtime budget: 1333 us/block")
print(f"Headroom: {1333 - us_per_block:.1f} us ({(1333 - us_per_block) / 1333 * 100:.0f}%)")

# Check output
print(f"\nOutput: {len(output)} samples, max={abs(output).max()}")

# Profile to find bottlenecks
print("\n" + "="*60)
print("Profiling VM step_block...")
print("="*60)

pr = cProfile.Profile()
pr.enable()
for _ in range(100):
    state, output = vm_step_block(state, frame, tables)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats(20)
print(s.getvalue())
