"""Debug SineCore quadrant transition."""
import sys
from pathlib import Path
import math

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "symcrash_python_ap"))

from sine_core import (
    sine_q15, Q_SCALE, PHASE_BITS, TABLE_SIZE, TABLE_BITS, FRAC_BITS,
    QUAD_SHIFT, INDEX_MASK, FRAC_MASK, _get_lut
)

lut = _get_lut()

print("Checking Q0->Q1 transition (around 90 degrees)")
print("=" * 70)

# Just before 90 degrees (end of Q0)
# and just after 90 degrees (start of Q1)

for offset in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
    # 90 degrees = 0x40000000
    phase = (0x40000000 + offset * 0x100000) & 0xFFFFFFFF

    # Expected
    frac = phase / (1 << PHASE_BITS)
    theta = 2 * math.pi * frac
    expected = Q_SCALE * math.sin(theta)
    actual = sine_q15(phase)
    error = actual - expected

    quadrant = (phase >> QUAD_SHIFT) & 0x3
    pos_in_quad = phase & 0x3FFFFFFF
    scaled = pos_in_quad >> (30 - TABLE_BITS - FRAC_BITS)
    index = (scaled >> FRAC_BITS) & INDEX_MASK
    frac_bits = scaled & FRAC_MASK

    print(f"phase=0x{phase:08X} Q{quadrant} idx={index:4d} frac={frac_bits:4d} | "
          f"actual={actual:6d} expected={expected:8.1f} error={error:+6.1f}")

print()
print("LUT values at boundary:")
print(f"  LUT[1022] = {lut[1022]}")
print(f"  LUT[1023] = {lut[1023]}")
print(f"  LUT[1024] = {lut[1024]}")

print()
print("Testing specific problematic phase (frac=0.375 = 135 degrees)")
phase = int(0.375 * (1 << PHASE_BITS))
print(f"phase = 0x{phase:08X}")
quadrant = (phase >> QUAD_SHIFT) & 0x3
print(f"quadrant = {quadrant}")
pos_in_quad = phase & 0x3FFFFFFF
print(f"pos_in_quad = 0x{pos_in_quad:08X} = {pos_in_quad}")
scaled = pos_in_quad >> (30 - TABLE_BITS - FRAC_BITS)
print(f"scaled = {scaled}")
index = (scaled >> FRAC_BITS) & INDEX_MASK
frac_bits = scaled & FRAC_MASK
print(f"index = {index}, frac = {frac_bits}")

# For Q1 (quadrant=1), we mirror
if quadrant & 1:
    mirror_index = TABLE_SIZE - 1 - index
    mirror_frac = FRAC_MASK - frac_bits
    print(f"mirror_index = {mirror_index}, mirror_frac = {mirror_frac}")
    y0 = lut[mirror_index + 1]
    y1 = lut[mirror_index]
    print(f"y0 = LUT[{mirror_index + 1}] = {y0}")
    print(f"y1 = LUT[{mirror_index}] = {y1}")
    delta = y1 - y0
    print(f"delta = {delta}")
    y = y0 + ((delta * mirror_frac) >> FRAC_BITS)
    print(f"y = {y}")

# What SHOULD it be?
# 135 degrees = pi*3/4 radians
# sin(135) = sin(180 - 135) = sin(45) = sqrt(2)/2 = 0.7071
# Expected = 32767 * 0.7071 = 23170
print(f"\nExpected: {Q_SCALE * math.sin(math.pi * 0.75):.1f}")
