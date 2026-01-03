"""Debug the CoeffOsc phase accumulator."""
from symcrash_python_ap.vm.coeff_osc import CoeffOscState, get_coeff_table, coeff_osc_render
from symcrash_python_ap.fixed import midi_to_phase_inc_fixed
from symcrash_python_ap.wavelet.sine_core import sine_q15 as _sine_q15
import numpy as np

# Test phase increment calculation
phase_inc = midi_to_phase_inc_fixed(69, 0)  # A4 = 440 Hz
print(f"Phase inc for A4: {phase_inc} (0x{phase_inc:08X})")

# This is for 48kHz sample rate: phase wraps every 2^32
# At 440 Hz, we need 440/48000 = 0.00917 of full circle per sample
# 0.00917 * 2^32 = 39,370,534 - correct!
print(f"Expected phase_inc: {int(440/48000 * 2**32)}")

# The problem: stride is Q16 where 16384 = 1.0x
# delta = phase_inc * stride >> 14
# With stride=16384: delta = phase_inc * 16384 >> 14 = phase_inc * 1 = phase_inc
# But we're doing: delta = ((phase_inc >> 8) * stride) >> 6
# = (phase_inc >> 8) * 16384 >> 6
# = phase_inc * 16384 >> 14
# = phase_inc
# So this should be correct...

stride = 16384
# Current calculation:
delta1 = ((phase_inc >> 8) * stride) >> 6
print(f"Delta method 1 (current): {delta1} (should be ~{phase_inc})")

# But wait - we're advancing per BLOCK, not per sample!
# For 64 samples per block at 440 Hz:
# delta_per_block = 64 * phase_inc
delta_per_block = 64 * phase_inc
print(f"Delta per block (64 samples): {delta_per_block} (0x{delta_per_block:08X})")

# Check what we're actually doing in render:
# We iterate 64 times but only use one accumulator per leaf
# Each iteration: state.node_phases[node] += delta
# So if we add delta 64 times, we get 64*delta
print(f"64 * delta1 = {64 * delta1}")

# The issue: we're using the same node phase for multiple samples within a block
# But we output 64 coefficients, not 64 samples
# The coefficient is constant across the block, not varying per sample

# Let me trace through more carefully
print("\n--- Tracing render ---")
state = CoeffOscState()
state.reset(phase_inc=phase_inc, table_id=0)
table = get_coeff_table(0)

for leaf in range(64):
    node = leaf // 2
    mag = table.mag_q15[leaf]
    stride = table.stride_q16[leaf]

    if mag > 0:
        delta = ((phase_inc >> 8) * stride) >> 6
        print(f"Leaf {leaf}: node={node}, mag={mag}, stride={stride}, delta={delta}")

        # After one block:
        new_phase = state.node_phases[node] + delta
        print(f"  Phase after 1 iter: 0x{new_phase:08X}")
        sin_val = _sine_q15(new_phase)
        print(f"  sin({new_phase:08X}) = {sin_val}")
        coeff = (sin_val * mag) >> 15
        print(f"  coeff = {coeff}")
