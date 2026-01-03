"""Test if wavelet roundtrip introduces phase delay."""
import numpy as np
from symcrash_python_ap.wavelet.fast_lifting import effect_mode_jit

# Generate a sine wave block
freq = 440
sr = 48000
n = 64

# Test at different phase offsets
for phase_deg in [0, 45, 90, 135, 180]:
    phase_rad = np.deg2rad(phase_deg)
    carrier = [int(10000 * np.sin(2 * np.pi * freq * i / sr + phase_rad)) for i in range(n)]

    # Roundtrip through wavelet (no mask = identity)
    wet = effect_mode_jit(carrier, None)

    # Find phase of carrier and wet by looking at zero-crossings or peaks
    carrier_arr = np.array(carrier)
    wet_arr = np.array(wet)

    # Cross-correlation to find delay
    corr = np.correlate(carrier_arr, wet_arr, mode='full')
    delay_samples = np.argmax(corr) - (n - 1)

    # RMS difference
    diff = carrier_arr - wet_arr
    rms_diff = np.sqrt(np.mean(diff**2))

    print(f"Phase {phase_deg:3d}°: delay={delay_samples:+2d} samples, RMS diff={rms_diff:.1f}")

# Now test with a mask that attenuates half the spectrum
print("\n--- With half-spectrum mask ---")
# Mask that zeros out leaves 32-63 (high frequency half)
half_mask = [32767] * 32 + [0] * 32

for phase_deg in [0, 45, 90, 135, 180]:
    phase_rad = np.deg2rad(phase_deg)
    carrier = [int(10000 * np.sin(2 * np.pi * freq * i / sr + phase_rad)) for i in range(n)]

    wet = effect_mode_jit(carrier, half_mask)

    carrier_arr = np.array(carrier)
    wet_arr = np.array(wet)

    corr = np.correlate(carrier_arr, wet_arr, mode='full')
    delay_samples = np.argmax(corr) - (n - 1)

    rms_diff = np.sqrt(np.mean((carrier_arr - wet_arr)**2))
    rms_wet = np.sqrt(np.mean(wet_arr**2))

    print(f"Phase {phase_deg:3d}°: delay={delay_samples:+2d} samples, RMS wet={rms_wet:.1f}, RMS diff={rms_diff:.1f}")
