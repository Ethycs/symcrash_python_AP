"""Analyze the pattern in CCS spurs - extended search."""

import numpy as np

# From FFT analysis - spurs that didn't match simple pattern
problem_spurs = [12310, 18310, 6310, 4810, 310, 23560, 10810]

f0 = 440.0
block_rate = 750.0
k_rate = 93.75
cycle_len = 109  # samples
cycle_freq = 48000 / cycle_len  # 440.367 Hz

print("=== EXTENDED SPUR PATTERN ANALYSIS ===")
print(f"f0 target = {f0} Hz")
print(f"f0 actual (from cycle_len) = {cycle_freq:.3f} Hz")
print(f"Block rate = {block_rate} Hz")
print()

for spur in problem_spurs:
    print(f"\nSpur at {spur} Hz:")

    # Try f_actual instead of f0
    for h in range(0, 60):
        for m in range(-40, 41):
            expected = h * cycle_freq + m * block_rate
            if abs(spur - expected) < 3:
                print(f"  {h}×{cycle_freq:.1f} + {m}×750 = {expected:.1f}")

    # Try block_rate multiples
    for m in range(0, 40):
        if abs(spur - m * block_rate) < 3:
            print(f"  {m}×750 = {m * block_rate:.1f}")
        if abs(spur - (m * block_rate + f0)) < 3:
            print(f"  {m}×750 + 440 = {m * block_rate + f0:.1f}")
        if abs(spur - (m * block_rate - f0)) < 3:
            print(f"  {m}×750 - 440 = {m * block_rate - f0:.1f}")

    # Check 310 Hz spacing (440 - 130 = 310, might be related to cycle_len)
    # 109 samples at 48kHz = 440.367 Hz
    # 109 samples - 64 samples (block) = 45 samples
    # 48000 / 45 = 1066.67 Hz... not it

    # What about 440 - 130 = 310?
    # 130 Hz = ?
    # 48000 / 130 = 369 samples... not obvious

    # Let's check: is 310 related to K=8 and block=64?
    # 48000 / (64/8) = 48000 / 8 = 6000... no
    # 48000 / (109 - 64) = 48000 / 45 = 1066.67

    # Check if it's (n * block_rate) + or - some constant offset
    for n in range(0, 35):
        base = n * block_rate
        diff = abs(spur - base)
        if diff < 350 and diff > 100:
            print(f"  {n}×750 = {base}, diff = {diff:.1f}")

print()
print("=== KEY INSIGHT ===")
print()

# The offset is 310 Hz for many spurs
# 310 = 440 - 130
# 130 = ?

# Actually let's compute: 6310 - 6000 = 310
#                        12310 - 12000 = 310
#                        18310 - 18000 = 310

print("6310 - 8×750 = 6310 - 6000 =", 6310 - 8 * 750)
print("12310 - 16×750 = 12310 - 12000 =", 12310 - 16 * 750)
print("18310 - 24×750 = 18310 - 18000 =", 18310 - 24 * 750)
print()
print("So many spurs are at: n×750 + 310")
print()
print("What is 310 Hz?")
print(f"  440 - 130 = 310")
print(f"  130 = 440 - 310")
print()
print("130 Hz = 48000 / 369 samples")
print("Or: 130 = 440 × (109-79)/109 = 440 × 30/109 = 121... close")
print()
print("Actually: 310 = 750 - 440 = beat frequency between block_rate and f0")
print("This is the key: spurs appear at n×750 ± 440, which simplifies to")
print("(n-1)×750 + (750-440) = (n-1)×750 + 310")
