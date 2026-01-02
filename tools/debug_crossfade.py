#!/usr/bin/env python3
"""Debug the crossfade scaling issue."""

import numpy as np

# Debug the scaling
eta = 32767
phase = (eta * 0x40000000) >> 15
print(f"eta={eta}")
print(f"phase = {phase} = 0x{phase:08X}")
print(f"expected phase for 90 deg = 0x40000000 = {0x40000000}")

# The scaling in _sincos_batch_q15
scaled = phase >> 18
print(f"\nscaled = phase >> 18 = {scaled}")
print(f"expected scaled for full table = 1024 << 12 = {1024 << 12}")

indices = scaled >> 12
fracs = scaled & 0xFFF
print(f"indices = {indices} (expected ~1024)")
print(f"fracs = {fracs}")

# The problem: phase is only going to 0x3FFF8000, not 0x40000000
# because (32767 * 0x40000000) >> 15 loses precision
print("\n--- The bug ---")
print(f"32767 * 0x40000000 = {32767 * 0x40000000}")
print(f"(32767 * 0x40000000) >> 15 = {(32767 * 0x40000000) >> 15}")
print(f"0x40000000 = {0x40000000}")

# Let's check what phase we're actually getting
print(f"\nActual phase / target = {phase / 0x40000000:.6f}")
print(f"This means we're only reaching {phase / 0x40000000 * 100:.2f}% of 90 degrees!")

# The problem is the shift amount - let's fix it
print("\n--- Correct calculation ---")
# For 0x40000000 input, we want scaled = 1024 << 12 = 4194304
# 0x40000000 = 1073741824
# We need: 0x40000000 >> shift = 4194304
# shift = log2(1073741824 / 4194304) = log2(256) = 8
# But we're using shift=18, so we get 1073741824 >> 18 = 4096
# which gives index = 4096 >> 12 = 0 !

correct_shift = 8
scaled_correct = 0x40000000 >> correct_shift
print(f"0x40000000 >> 8 = {scaled_correct}")
print(f"index = {scaled_correct >> 12}")
print(f"frac = {scaled_correct & 0xFFF}")

# But wait - the actual max phase we get is less than 0x40000000
# because of the eta * 0x40000000 >> 15 calculation
# Let's see what shift gives us index=1024 at max eta
actual_max_phase = (32767 * 0x40000000) >> 15
print(f"\nActual max phase = {actual_max_phase}")
print(f"Actual max phase >> 8 = {actual_max_phase >> 8}")
print(f"Index at shift 8 = {(actual_max_phase >> 8) >> 12}")

# The phase mapping is wrong! We're computing:
# phase = (eta * 0x40000000) >> 15
# But this gives phase = 0x3FFF8000 for eta = 32767
# when we want phase = 0x40000000

# Better approach: just map eta directly to table index
# eta in [0, 32767] -> index in [0, 1024]
# index = (eta * 1024) >> 15
eta = 32767
direct_index = (eta * 1024) >> 15
print(f"\nDirect index mapping: (32767 * 1024) >> 15 = {direct_index}")
