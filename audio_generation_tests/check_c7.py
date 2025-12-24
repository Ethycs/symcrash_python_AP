"""Check C7 theoretical d2."""
import math

# C7 = MIDI 96 = 2093.00 Hz
freq = 440 * (2 ** ((96 - 69) / 12))
print(f"C7 frequency: {freq:.2f} Hz")

# At 48kHz
fs = 48000
samples_per_period = fs / freq
print(f"Samples per period: {samples_per_period:.1f}")

# For a pure sine at amplitude A, max d2 is proportional to A * omega^2
# omega = 2*pi*freq/fs
omega = 2 * math.pi * freq / fs
amplitude = 32767  # Q15 max
max_d2_theoretical = amplitude * omega * omega
print(f"Theoretical max d2: {max_d2_theoretical:.0f}")

# So 862 is expected for a pure high-frequency sine!
# The "spikes" are just the natural curvature of a high-frequency sine.

# Threshold should scale with frequency
print(f"\nFor C4 (261.63 Hz):")
freq_c4 = 261.63
omega_c4 = 2 * math.pi * freq_c4 / fs
max_d2_c4 = amplitude * omega_c4 * omega_c4
print(f"  Max d2: {max_d2_c4:.0f}")
