"""Explore raw coefficient patterns for square wave synthesis."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.fixed import Q15_ONE

print("=" * 70)
print("RAW COEFFICIENT EXPLORATION")
print("=" * 70)

# Generate ideal square wave (non-band-limited) and see its coefficients
print("\n1. What coefficients represent a perfect square wave?")

# Perfect square: first half +MAX, second half -MAX
perfect_square = [Q15_ONE] * 32 + [-Q15_ONE] * 32
coeffs_square = fwpt_jit(perfect_square, headroom_bits=1)

print(f"   Perfect square -> FWPT coefficients:")
print(f"   First 16: {coeffs_square[:16]}")
print(f"   Last 16:  {coeffs_square[-16:]}")

# Check reconstruction
reconstructed = idwpt_jit(coeffs_square, headroom_bits=1)
print(f"\n   Reconstructed samples 0-7:   {reconstructed[:8]}")
print(f"   Reconstructed samples 28-35: {reconstructed[28:36]}")

# What's the coefficient pattern?
print("\n2. Coefficient magnitude distribution:")
mags = np.abs(coeffs_square)
print(f"   Max coeff: {np.max(mags)}")
print(f"   Non-zero coeffs: {np.sum(mags > 100)}")

# Group by scale (wavelet decomposition levels)
# For 64 coefficients: 1 DC, 1 level-1, 2 level-2, 4 level-3, 8 level-4, 16 level-5, 32 level-6
print("\n3. Energy by wavelet scale:")
scales = [
    ("DC (avg)", coeffs_square[0:1]),
    ("Level 1", coeffs_square[1:2]),
    ("Level 2", coeffs_square[2:4]),
    ("Level 3", coeffs_square[4:8]),
    ("Level 4", coeffs_square[8:16]),
    ("Level 5", coeffs_square[16:32]),
    ("Level 6 (finest)", coeffs_square[32:64]),
]

for name, c in scales:
    energy = np.sum(np.array(c, dtype=np.float64)**2)
    print(f"   {name:20s}: energy={energy:.0f}, coeffs={list(c)[:8]}{'...' if len(c) > 8 else ''}")

# Now try: what if we zero out the finest scale?
print("\n4. Zero out finest scale (level 6) - removes highest frequencies:")
coeffs_smooth = list(coeffs_square)
for i in range(32, 64):
    coeffs_smooth[i] = 0

smooth_wave = idwpt_jit(coeffs_smooth, headroom_bits=1)
print(f"   Samples 0-7:   {smooth_wave[:8]}")
print(f"   Samples 28-35: {smooth_wave[28:36]}")
print(f"   Flat region std: {np.std(smooth_wave[:16]):.1f}")

# Try zeroing more scales
print("\n5. Progressive smoothing by zeroing scales:")
for zero_from in [48, 32, 16, 8]:
    coeffs_test = list(coeffs_square)
    for i in range(zero_from, 64):
        coeffs_test[i] = 0
    wave = idwpt_jit(coeffs_test, headroom_bits=1)
    flat_std = np.std(wave[:16])
    print(f"   Zero from {zero_from:2d}: flat_std={flat_std:.1f}, samples[0:4]={wave[:4]}")

# Compare to band-limited harmonic synthesis
print("\n6. Compare to Mode B 32-harmonic square:")
from symcrash_python_ap.wavelet.sine_core import sin_cos_q15, freq_to_phase_inc

freq_hz = 261.63
phase_inc = freq_to_phase_inc(freq_hz, 48000)

harmonic_samples = []
phase = 0
for i in range(64):
    val = 0
    for h in range(1, 33, 2):  # odd harmonics
        h_phase = (phase * h) & 0xFFFFFFFF
        s, _ = sin_cos_q15(h_phase)
        val += s // h
    harmonic_samples.append(np.clip(val, -32768, 32767))
    phase = (phase + phase_inc) & 0xFFFFFFFF

coeffs_harmonic = fwpt_jit(harmonic_samples, headroom_bits=1)

print(f"   Harmonic square coefficients:")
for name, (start, end) in [("DC", (0,1)), ("L1", (1,2)), ("L2", (2,4)), ("L3", (4,8)),
                            ("L4", (8,16)), ("L5", (16,32)), ("L6", (32,64))]:
    c = coeffs_harmonic[start:end]
    energy = np.sum(np.array(c, dtype=np.float64)**2)
    print(f"   {name}: energy={energy:.0f}")

print("\n7. Key insight:")
print("   Perfect square has energy at ALL scales (including finest)")
print("   Band-limited square has less energy at finest scales")
print("   The 'ripple' IS the missing fine-scale energy manifesting as overshoot")
