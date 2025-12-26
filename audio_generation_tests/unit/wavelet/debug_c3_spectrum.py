"""Debug C3 WET spectrum issue - why is energy spread across frequencies?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "symcrash_python_ap"))

import numpy as np
from symcrash_python_ap.wavelet.harmonic_injector import (
    create_harmonic_coeffs, freq_to_leaf_and_coeff, WaveType,
    LEAF_EDGES_HZ, NODE_SIZE, LEAF_COUNT
)
from symcrash_python_ap.wavelet.carrier_lift import Q15_ONE, Q15_HALF

# C3 = MIDI 48 = 130.81 Hz
f0_hz = 440.0 * (2.0 ** ((48 - 69) / 12.0))
print(f"C3 frequency: {f0_hz:.2f} Hz")

# Check where C3 maps to in leaf space
leaf_idx, coeff_idx = freq_to_leaf_and_coeff(f0_hz)
print(f"\nC3 maps to: leaf {leaf_idx}, coeff {coeff_idx}")
print(f"  Leaf {leaf_idx} covers: {LEAF_EDGES_HZ[leaf_idx]:.1f} - {LEAF_EDGES_HZ[leaf_idx+1]:.1f} Hz")

# Generate coefficients for C3 pure sine
coeffs = create_harmonic_coeffs(
    frequency=f0_hz,
    amplitude=Q15_HALF,
    wave_type=WaveType.SINE,
    partials_n=1
)

# Show non-zero coefficients
print(f"\nNon-zero coefficients from create_harmonic_coeffs:")
for i, c in enumerate(coeffs):
    if c != 0:
        leaf = i // NODE_SIZE
        coeff = i % NODE_SIZE
        # Calculate what frequency this coefficient synthesizes
        band_low = LEAF_EDGES_HZ[leaf]
        band_high = LEAF_EDGES_HZ[leaf + 1]
        band_span = band_high - band_low
        sub_pos = (coeff + 0.5) / NODE_SIZE
        synth_freq = band_low + sub_pos * band_span
        print(f"  coeffs[{i}] (leaf {leaf}, coeff {coeff}) = {c}")
        print(f"    -> synthesizes at {synth_freq:.1f} Hz")

# Check the synthesis path
print(f"\n" + "=" * 60)
print("Checking wavelet synthesis frequency mapping:")
print("=" * 60)

# The synthesizer uses this formula:
# sub_freq_hz = freq_low_hz + sub_pos * freq_span
# where sub_pos = (coeff_idx + 0.5) / NODE_SIZE

sample_rate = 48000.0
nyquist = sample_rate / 2
band_width = nyquist / LEAF_COUNT

print(f"\nLeaf bandwidth: {band_width:.1f} Hz")
print(f"NODE_SIZE: {NODE_SIZE}")

# For leaf 0 (which covers 0-375 Hz at 48kHz, 64 leaves)
leaf0_low = 0
leaf0_high = band_width
print(f"\nLeaf 0 covers: {leaf0_low:.1f} - {leaf0_high:.1f} Hz")
print("  Coefficient positions:")
for c in range(NODE_SIZE):
    sub_pos = (c + 0.5) / NODE_SIZE
    freq = leaf0_low + sub_pos * (leaf0_high - leaf0_low)
    print(f"    coeff {c}: sub_pos={sub_pos:.3f} -> {freq:.1f} Hz")

print(f"\n130.81 Hz should map to:")
pos_in_band = (f0_hz - leaf0_low) / (leaf0_high - leaf0_low)
expected_coeff = int(pos_in_band * NODE_SIZE)
print(f"  pos_in_band = {pos_in_band:.3f}")
print(f"  expected_coeff = {expected_coeff}")
