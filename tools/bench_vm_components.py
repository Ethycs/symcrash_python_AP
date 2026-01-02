#!/usr/bin/env python3
"""Benchmark individual VM components to find exact time breakdown."""

import time
import numpy as np

from symcrash_python_ap.vm import VMState, Tables
from symcrash_python_ap.vm.decode import decode_frame, CarrierRole
from symcrash_python_ap.vm.lanes import (
    lane_carrier_render,
    lane_inject,
    dispatch_coeff_op,
    lane_tail,
    lane_idwpt_render,
    lane_mix_gain,
)
from symcrash_python_ap.frames import VoiceFrame

# Create test frame
frame = VoiceFrame(
    voice_id=1,
    note=60,
    cents=0,
    f0_hz=261.63,
    gain_q15=24576,
    beta_target_q15=16384,
    partial_amps_q15=[32767, 16384, 10923, 8192, 6554, 5461, 4681, 4096],
    is_active=True,
)

# Create state
tables = Tables()
state = VMState.create(1)
state.reset_for_note_on(f0_hz=frame.f0_hz)
state.blocks_since_start = 10  # Past latency period
state.wet_ready = True

# Decode frame once
cfg = decode_frame(frame, tables, prev_eta_q15=0, prev_gain_q15=0)
cfg.idwpt_enable = True

# Warmup
N = 1000
for _ in range(100):
    carrier = lane_carrier_render(state, cfg)
    lane_inject(state, cfg)
    for instr in cfg.backbone_instrs:
        dispatch_coeff_op(state, instr, cfg.mask_u32)
    lane_tail(state, cfg)
    texture = lane_idwpt_render(state, cfg, carrier)
    output = lane_mix_gain(cfg, carrier, texture)

print("VM Component Timing Breakdown")
print("=" * 50)

# 1. Decode
start = time.perf_counter()
for _ in range(N):
    cfg = decode_frame(frame, tables, prev_eta_q15=0, prev_gain_q15=0)
elapsed = time.perf_counter() - start
decode_us = elapsed * 1e6 / N
print(f"decode_frame:       {decode_us:7.1f} us")

# 2. Carrier render
start = time.perf_counter()
for _ in range(N):
    carrier = lane_carrier_render(state, cfg)
elapsed = time.perf_counter() - start
carrier_us = elapsed * 1e6 / N
print(f"lane_carrier_render: {carrier_us:7.1f} us")

# 3. Inject
start = time.perf_counter()
for _ in range(N):
    lane_inject(state, cfg)
elapsed = time.perf_counter() - start
inject_us = elapsed * 1e6 / N
print(f"lane_inject:        {inject_us:7.1f} us")

# 4. Backbone ops
start = time.perf_counter()
for _ in range(N):
    for instr in cfg.backbone_instrs:
        dispatch_coeff_op(state, instr, cfg.mask_u32)
elapsed = time.perf_counter() - start
backbone_us = elapsed * 1e6 / N
print(f"backbone ops:       {backbone_us:7.1f} us ({len(cfg.backbone_instrs)} ops)")

# 5. Tail
start = time.perf_counter()
for _ in range(N):
    lane_tail(state, cfg)
elapsed = time.perf_counter() - start
tail_us = elapsed * 1e6 / N
print(f"lane_tail:          {tail_us:7.1f} us")

# 6. IDWPT render
cfg.idwpt_enable = True
start = time.perf_counter()
for _ in range(N):
    texture = lane_idwpt_render(state, cfg, carrier)
elapsed = time.perf_counter() - start
idwpt_us = elapsed * 1e6 / N
print(f"lane_idwpt_render:  {idwpt_us:7.1f} us")

# 7. Mix + Gain
start = time.perf_counter()
for _ in range(N):
    output = lane_mix_gain(cfg, carrier, texture)
elapsed = time.perf_counter() - start
mix_us = elapsed * 1e6 / N
print(f"lane_mix_gain:      {mix_us:7.1f} us")

print("-" * 50)
total = decode_us + carrier_us + inject_us + backbone_us + tail_us + idwpt_us + mix_us
print(f"TOTAL:              {total:7.1f} us")
print(f"Budget:             1333.0 us")
print(f"Headroom:           {1333 - total:7.1f} us ({(1333 - total) / 1333 * 100:.0f}%)")

# Without IDWPT
print("\n" + "=" * 50)
print("WITHOUT IDWPT (carrier-only mode):")
print("=" * 50)
total_no_idwpt = decode_us + carrier_us + inject_us + backbone_us + tail_us + mix_us
print(f"TOTAL:              {total_no_idwpt:7.1f} us")
print(f"Budget:             1333.0 us")
print(f"Headroom:           {1333 - total_no_idwpt:7.1f} us ({(1333 - total_no_idwpt) / 1333 * 100:.0f}%)")
