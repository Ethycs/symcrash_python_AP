"""Test if FWPT energy depends on signal phase alignment."""

import numpy as np
import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit

print("=== WAVELET ENERGY vs PHASE ALIGNMENT ===")
print()

# Generate a pure sine wave, much longer than one block
f0 = 440
fs = 48000
cycle_samples = fs / f0  # 109.09 samples per cycle
duration_samples = 1000

t = np.arange(duration_samples) / fs
sine = (np.sin(2 * np.pi * f0 * t) * 16000).astype(np.int16)

print(f"Sine wave: {f0} Hz, {duration_samples} samples")
print(f"Cycle length: {cycle_samples:.2f} samples")
print()

# Extract blocks at different starting positions and measure coefficient energy
B = 64
energies = []
for start in range(0, 200):
    block = sine[start:start + B]
    if len(block) < B:
        break
    coeffs = fwpt_jit(block.tolist(), headroom_bits=1)
    energy = sum(c * c for c in coeffs)
    energies.append(energy)

energies = np.array(energies, dtype=np.float64)
print(f"Coefficient energy across {len(energies)} starting positions:")
print(f"  Min: {energies.min():.0f}")
print(f"  Max: {energies.max():.0f}")
print(f"  Ratio: {energies.max() / energies.min():.4f}")
print(f"  Std/Mean: {energies.std() / energies.mean():.4f}")

# Check input energy
input_energies = []
for start in range(0, 200):
    block = sine[start:start + B]
    if len(block) < B:
        break
    energy = np.sum(block.astype(np.float64) ** 2)
    input_energies.append(energy)

input_energies = np.array(input_energies, dtype=np.float64)
print()
print(f"Input block energy across {len(input_energies)} starting positions:")
print(f"  Min: {input_energies.min():.0f}")
print(f"  Max: {input_energies.max():.0f}")
print(f"  Ratio: {input_energies.max() / input_energies.min():.4f}")

# Check if coefficient energy tracks input energy
print()
print("Does coefficient energy track input energy?")
correlation = np.corrcoef(input_energies, energies[:len(input_energies)])[0, 1]
print(f"  Correlation: {correlation:.4f}")

# Check reconstruction energy
print()
print("=== RECONSTRUCTION ENERGY ===")
recon_energies = []
for start in range(0, 200):
    block = sine[start:start + B]
    if len(block) < B:
        break
    coeffs = fwpt_jit(block.tolist(), headroom_bits=1)
    recon = idwpt_jit(coeffs, headroom_bits=1)
    recon_energy = sum(r * r for r in recon)
    recon_energies.append(recon_energy)

recon_energies = np.array(recon_energies, dtype=np.float64)
print(f"Reconstruction energy:")
print(f"  Min: {recon_energies.min():.0f}")
print(f"  Max: {recon_energies.max():.0f}")
print(f"  Ratio: {recon_energies.max() / recon_energies.min():.4f}")

# Does reconstruction match input?
print()
print("Reconstruction accuracy:")
errors = []
for start in range(0, 100):
    block = sine[start:start + B]
    coeffs = fwpt_jit(block.tolist(), headroom_bits=1)
    recon = np.array(idwpt_jit(coeffs, headroom_bits=1), dtype=np.int16)
    error = np.max(np.abs(block.astype(np.int32) - recon.astype(np.int32)))
    errors.append(error)
print(f"  Max reconstruction error: {max(errors)}")
print(f"  Mean reconstruction error: {np.mean(errors):.1f}")

# Key insight: the transform is invertible, but the *coefficient energy*
# varies with input alignment, even for constant-power input
