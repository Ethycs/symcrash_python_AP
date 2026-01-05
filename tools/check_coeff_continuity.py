"""Check coefficient continuity across phase steps in the LUT."""

import numpy as np
import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.vm.coeff_lut_streaming import get_lut_for_note

print("=== COEFFICIENT CONTINUITY CHECK ===")
lut = get_lut_for_note(wave_id=0, midi_note=69)

print(f"LUT shape: {lut.data.shape}  (leaves, K, P)")
print(f"P = {lut.P}, K = {lut.K}")
print()

# Check continuity across P dimension for a single leaf
print("=== LEAF 0 CONTINUITY (across phase) ===")
leaf = 0
for k in range(lut.K):
    coeffs = lut.data[leaf, k, :]  # All P values for this (leaf, k)
    diffs = np.diff(coeffs)
    max_diff = np.max(np.abs(diffs))
    wrap_diff = abs(int(coeffs[-1]) - int(coeffs[0]))  # Wrap-around
    print(f"  k={k}: max_diff={max_diff:5d}, wrap_diff={wrap_diff:5d}")

print()
print("=== LEAF 3 CONTINUITY (highest energy leaf) ===")
leaf = 3
for k in range(lut.K):
    coeffs = lut.data[leaf, k, :]
    diffs = np.diff(coeffs)
    max_diff = np.max(np.abs(diffs))
    wrap_diff = abs(int(coeffs[-1]) - int(coeffs[0]))
    print(f"  k={k}: max_diff={max_diff:5d}, wrap_diff={wrap_diff:5d}, range=[{coeffs.min():6d}, {coeffs.max():6d}]")

# Check if the coefficients from p to p+1 change smoothly
print()
print("=== PHASE STEP TRANSITIONS ===")
print("Checking if coeff[leaf, k, p] -> coeff[leaf, k, p+1] is smooth")
print()

# For leaf 3 (highest energy), show a few transitions
leaf = 3
k = 0
print(f"Leaf {leaf}, k={k}:")
print(f"  p:   coeff[p]  coeff[p+1]  diff")
for p in range(0, 10):
    c0 = lut.data[leaf, k, p]
    c1 = lut.data[leaf, k, (p + 1) % lut.P]
    print(f"  {p:2d}: {c0:8d}  {c1:8d}  {c1 - c0:+6d}")

print(f"  ...")
for p in range(lut.P - 3, lut.P):
    c0 = lut.data[leaf, k, p]
    c1 = lut.data[leaf, k, (p + 1) % lut.P]
    print(f"  {p:2d}: {c0:8d}  {c1:8d}  {c1 - c0:+6d}")

# Now check the K dimension continuity
print()
print("=== K DIMENSION CONTINUITY ===")
print("Checking if coeff[leaf, k, p] -> coeff[leaf, k+1, p] is smooth")
print("(This matters for polyphase interleaving)")
print()

leaf = 3
p = 0
print(f"Leaf {leaf}, p={p}:")
print(f"  k:  coeff[k]  coeff[k+1]  diff")
for k in range(lut.K - 1):
    c0 = lut.data[leaf, k, p]
    c1 = lut.data[leaf, k + 1, p]
    print(f"  {k}: {c0:8d}  {c1:8d}  {c1 - c0:+6d}")

# Check total energy variation
print()
print("=== TOTAL ENERGY VARIATION ACROSS P ===")
energies = []
for p in range(lut.P):
    energy = np.sum(lut.data[:, :, p].astype(np.float64) ** 2)
    energies.append(energy)
energies = np.array(energies)
print(f"Energy: min={energies.min():.0f}, max={energies.max():.0f}")
print(f"Energy ratio (max/min): {energies.max() / energies.min():.4f}")
print(f"This should be ~1.0 for a consistent sine wave")

# Check sum across k for each (leaf, p) - should be smooth
print()
print("=== SUM ACROSS K (per leaf, per phase) ===")
leaf = 3
sums = []
for p in range(lut.P):
    s = np.sum(lut.data[leaf, :, p].astype(np.int32))
    sums.append(s)
sums = np.array(sums)
print(f"Leaf {leaf} sum across K: min={sums.min()}, max={sums.max()}, range={sums.max()-sums.min()}")
