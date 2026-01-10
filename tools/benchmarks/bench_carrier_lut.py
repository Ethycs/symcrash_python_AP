#!/usr/bin/env python3
"""Benchmark carrier render with 64K LUT."""

import time
import numpy as np
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.vm.decode import DecodedPlan, CarrierParams, CarrierRole

# Setup
state = VMState()
state.phase = 0
state.phase_inc = 89478485  # ~440Hz at 48kHz
state.target_phase_inc = state.phase_inc
state.pitch_slew_rate = 0

params = CarrierParams()
params.phase_inc = state.phase_inc
params.harmonic_amps_q15 = [32767, 16384, 8192, 4096, 2048, 1024, 512, 256]

cfg = DecodedPlan()
cfg.carrier_role = CarrierRole.AUDIBLE
cfg.carrier_params = params

from symcrash_python_ap.vm.lanes import lane_carrier_render

# Warmup
for _ in range(100):
    state.phase = 0
    lane_carrier_render(state, cfg)

# Benchmark
N = 1000
start = time.perf_counter()
for _ in range(N):
    state.phase = 0
    lane_carrier_render(state, cfg)
elapsed = time.perf_counter() - start

us_per_block = elapsed * 1e6 / N
print(f"lane_carrier_render (64K LUT): {us_per_block:.1f} us/block")

# Check output quality
state.phase = 0
output = lane_carrier_render(state, cfg)
print(f"Output range: [{output.min()}, {output.max()}]")
print(f"First 8 samples: {output[:8].tolist()}")

# Verify expected fundamental frequency content
# 440 Hz at 48kHz = period of ~109 samples
# With 64 samples we should see ~0.58 cycles
print(f"Sample 32 (quarter wave expected): {output[32]}")
