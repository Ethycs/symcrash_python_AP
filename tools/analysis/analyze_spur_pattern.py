"""Analyze the pattern in CCS spurs."""

import numpy as np

# From FFT analysis - top spurs
spurs = [
    (12310.00, -19.5),
    (18310.00, -19.9),
    (6310.00, -20.3),
    (1190.00, -21.1),
    (7190.00, -21.1),
    (23690.00, -21.8),
    (4810.00, -22.3),
    (310.00, -22.6),
    (13190.00, -22.7),
    (18440.00, -24.1),
    (12440.00, -24.5),
    (23560.00, -24.7),
    (17690.00, -25.4),
    (10810.00, -25.5),
    (20690.00, -25.8),
    (6440.00, -25.9),
]

f0 = 440.0
block_rate = 750.0  # 48000 / 64
k_rate = 93.75      # 750 / 8

print("=== SPUR PATTERN ANALYSIS ===")
print(f"f0 = {f0} Hz")
print(f"Block rate = {block_rate} Hz")
print(f"K-rate = {k_rate} Hz")
print()

print("Checking if spurs are f0 harmonics ± block_rate harmonics:")
print(f"{'Spur (Hz)':>10} {'dB':>8} {'Pattern':>40}")
print("-" * 60)

for freq, db in spurs:
    found_pattern = None

    # Check: h * f0 ± m * block_rate
    for h in range(0, 60):
        for m in range(-40, 41):
            expected = h * f0 + m * block_rate
            if abs(freq - expected) < 5:
                found_pattern = f"{h}×f0 + {m}×750 = {expected:.0f}"
                break
        if found_pattern:
            break

    if not found_pattern:
        # Check: n * k_rate
        for n in range(-300, 301):
            expected = n * k_rate
            if abs(freq - expected) < 5:
                found_pattern = f"{n}×93.75 = {expected:.0f}"
                break

    if not found_pattern:
        found_pattern = "NO PATTERN FOUND"

    print(f"{freq:10.1f} {db:8.1f} {found_pattern:>40}")

print()
print("=== INTERPRETATION ===")
print()
print("The spurs follow the pattern: h×440 ± m×750")
print()
print("This is classic aliasing/phase quantization:")
print("- P=64 phase steps creates sidebands at ±750 Hz, ±1500 Hz, etc.")
print("- These interact with f0 harmonics")
print()
print("Solution options:")
print("1. Increase P from 64 to 256+ (reduces phase quantization)")
print("2. Use phase interpolation with higher precision")
print("3. Add coefficient-domain smoothing between phase steps")
