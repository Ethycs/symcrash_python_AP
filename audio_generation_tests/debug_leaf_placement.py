"""Debug where LeafGenerator places energy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "symcrash_python_ap"))

from symcrash_python_ap.leaf_generator import LeafGenerator
from symcrash_python_ap.wavelet_packet_trees import LEAF_COUNT, NODE_SIZE
from symcrash_python_ap.carrier_lift import Q15_ONE

leaf_gen = LeafGenerator()

# C4 = 261.63 Hz
f0_hz = 261.63
harmonic_amps = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

leaf_content = leaf_gen.generate_from_fundamental(f0_hz, harmonic_amps)

print(f"Leaf content for C4 ({f0_hz} Hz):")
print(f"  Total coeffs: {len(leaf_content.coeffs_q15)}")

# Find non-zero leaves
for leaf in range(min(LEAF_COUNT, len(leaf_content.coeffs_q15) // NODE_SIZE)):
    start = leaf * NODE_SIZE
    end = start + NODE_SIZE
    node = leaf_content.coeffs_q15[start:end]
    energy = sum(abs(c) for c in node)
    if energy > 100:
        print(f"  Leaf {leaf}: energy={energy}, coeffs={node}")

# Check band frequencies
nyquist = 24000
band_width = nyquist / 32  # 32 leaves = 750 Hz per leaf
print(f"\nBand frequencies (32 leaves):")
print(f"  Leaf 0: 0 - {band_width} Hz")
print(f"  Leaf 1: {band_width} - {2*band_width} Hz")
print(f"  C4 (261.63 Hz) should be in leaf 0")
