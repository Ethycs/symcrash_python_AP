"""Test with an ideal cycle length that divides evenly by P."""

import numpy as np
import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.wavelet.sine_core import sine_q15

# Parameters
P = 64
K = 8
B = 64
SAMPLE_RATE = 48000

print("=== TESTING CYCLE LENGTH ALIGNMENT ===")
print()

# Test 1: Non-aligned (109 samples for 440 Hz)
print("--- Test 1: Non-aligned cycle (109 samples, 440 Hz) ---")
cycle_len_440 = 109
energies_440 = []
for p in range(P):
    start = int(round(p * cycle_len_440 / P))
    # Generate one block of coefficients
    waveform = np.array([sine_q15(int((n / cycle_len_440) * (1 << 32)) & 0xFFFFFFFF)
                         for n in range(cycle_len_440 + B)], dtype=np.int16)
    block = waveform[start:start + B]
    coeffs = fwpt_jit(block.tolist(), headroom_bits=1)
    energy = sum(c * c for c in coeffs)
    energies_440.append(energy)

energies_440 = np.array(energies_440, dtype=np.float64)
print(f"Energy ratio: {energies_440.max() / energies_440.min():.4f}")
print(f"Energy std/mean: {energies_440.std() / energies_440.mean():.4f}")

# Test 2: Aligned (128 samples = 2*P, 375 Hz)
print()
print("--- Test 2: Aligned cycle (128 samples = 2×P, 375 Hz) ---")
cycle_len_375 = 128  # 48000/128 = 375 Hz
energies_375 = []
for p in range(P):
    start = int(round(p * cycle_len_375 / P))
    waveform = np.array([sine_q15(int((n / cycle_len_375) * (1 << 32)) & 0xFFFFFFFF)
                         for n in range(cycle_len_375 + B)], dtype=np.int16)
    block = waveform[start:start + B]
    coeffs = fwpt_jit(block.tolist(), headroom_bits=1)
    energy = sum(c * c for c in coeffs)
    energies_375.append(energy)

energies_375 = np.array(energies_375, dtype=np.float64)
print(f"Energy ratio: {energies_375.max() / energies_375.min():.4f}")
print(f"Energy std/mean: {energies_375.std() / energies_375.mean():.4f}")

# Test 3: Use P that divides cycle_len (109 doesn't divide nicely, but let's try P=109)
print()
print("--- Test 3: P=109 (matches cycle length) ---")
P_109 = 109
energies_109 = []
for p in range(P_109):
    start = int(round(p * cycle_len_440 / P_109))  # = p (since cycle_len = P)
    waveform = np.array([sine_q15(int((n / cycle_len_440) * (1 << 32)) & 0xFFFFFFFF)
                         for n in range(cycle_len_440 + B)], dtype=np.int16)
    block = waveform[start:start + B]
    coeffs = fwpt_jit(block.tolist(), headroom_bits=1)
    energy = sum(c * c for c in coeffs)
    energies_109.append(energy)

energies_109 = np.array(energies_109, dtype=np.float64)
print(f"Energy ratio: {energies_109.max() / energies_109.min():.4f}")
print(f"Energy std/mean: {energies_109.std() / energies_109.mean():.4f}")

# Test 4: Higher P (256)
print()
print("--- Test 4: P=256 (higher resolution) ---")
P_256 = 256
energies_256 = []
for p in range(P_256):
    start = int(round(p * cycle_len_440 / P_256))
    waveform = np.array([sine_q15(int((n / cycle_len_440) * (1 << 32)) & 0xFFFFFFFF)
                         for n in range(cycle_len_440 + B)], dtype=np.int16)
    block = waveform[start:start + B]
    coeffs = fwpt_jit(block.tolist(), headroom_bits=1)
    energy = sum(c * c for c in coeffs)
    energies_256.append(energy)

energies_256 = np.array(energies_256, dtype=np.float64)
print(f"Energy ratio: {energies_256.max() / energies_256.min():.4f}")
print(f"Energy std/mean: {energies_256.std() / energies_256.mean():.4f}")

print()
print("=== CONCLUSION ===")
print("The energy variation comes from phase quantization.")
print("Solutions:")
print("1. Use P that matches cycle_len (per-note P)")
print("2. Use much higher P (256+) to reduce quantization")
print("3. Use interpolation in coefficient space, not just phase")
