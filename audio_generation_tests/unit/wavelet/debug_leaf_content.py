"""Debug leaf content structure."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "symcrash_python_ap"))

from symcrash_python_ap.wavelet.leaf_generator import LeafGenerator
from symcrash_python_ap.wavelet.carrier_lift import Q15_ONE, NUM_LEAVES

leaf_gen = LeafGenerator()

# Generate leaf content for C4 (pure sine)
f0_hz = 261.63
harmonic_amps = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]
leaf_content = leaf_gen.generate_from_fundamental(f0_hz, harmonic_amps)

print(f"LeafContent.coeffs_q15 length: {len(leaf_content.coeffs_q15)}")
print(f"Expected NUM_LEAVES: {NUM_LEAVES}")
print(f"\nNon-zero leaves:")
for i, coeff in enumerate(leaf_content.coeffs_q15):
    if coeff != 0:
        print(f"  leaf[{i}] = {coeff}")
