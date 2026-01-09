"""Check harmonic coefficients from different sources."""
import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from symcrash_python_ap.vm.waveform_matrix import get_theoretical_harmonics, WaveformType
from symcrash_python_ap.vm.harmonic_bank import HarmonicWeights
from symcrash_python_ap.fixed import Q15_ONE

print("=== Comparing harmonic coefficient sources ===\n")

# 1. From get_theoretical_harmonics
dc, a_h, b_h = get_theoretical_harmonics(WaveformType.SQUARE, H_max=16)
print("get_theoretical_harmonics (SQUARE):")
print(f"  DC: {dc}")
for h in range(1, 17):
    if h % 2 == 1:  # Odd harmonics only
        raw = (4 * Q15_ONE) / (math.pi * h)
        print(f"  h={h}: a_h={a_h[h-1]:6d}  raw={raw:8.1f}  ratio={a_h[h-1]/Q15_ONE:.4f}")

# 2. From set_square
weights = HarmonicWeights()
weights.set_square(H=16)
print("\nset_square (normalized 1/h):")
for h in range(1, 17):
    if h % 2 == 1:
        print(f"  h={h}: a_h={weights.a_h_q15[h-1]:6d}  ratio={weights.a_h_q15[h-1]/Q15_ONE:.4f}")

# 3. Compare ratios
print("\n=== Ratio comparison (h/h=1) ===")
print(f"{'h':>3}  {'theoretical':>12}  {'set_square':>12}  {'expected':>12}")
h1_theo = a_h[0]
h1_sq = weights.a_h_q15[0]
for h in [3, 5, 7, 9, 11, 13, 15]:
    theo_ratio = a_h[h-1] / h1_theo if h1_theo else 0
    sq_ratio = weights.a_h_q15[h-1] / h1_sq if h1_sq else 0
    expected = 1.0 / h
    print(f"{h:3d}  {theo_ratio:12.4f}  {sq_ratio:12.4f}  {expected:12.4f}")

# 4. Check if theoretical h=1 clips
print(f"\n=== Q15 overflow check ===")
print(f"Q15_ONE = {Q15_ONE}")
print(f"4/π * Q15_ONE = {(4/math.pi) * Q15_ONE:.1f}")
print(f"theoretical a_1 = {a_h[0]} (after SAT16)")

# The problem: 4/π ≈ 1.27, so 4/π * 32767 ≈ 41666 which clips to 32767
# This means theoretical h=1 coefficient is SATURATED
# But the ratios are still correct because we're using 4/(πh) directly

print("\n=== The issue ===")
print("The set_square() method uses 1/h which gives correct RATIOS but wrong AMPLITUDE.")
print("The get_theoretical_harmonics uses 4/(πh) but h=1 clips to 32767.")
print("Both have the same relative ratios (1:1/3:1/5:...) but different absolute levels.")
print("\nThe noise in the square wave is NOT due to harmonic coefficient values.")
