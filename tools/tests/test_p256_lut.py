"""Test that LUT is generated with P=256."""

import sys
sys.path.insert(0, 'src')

# Clear the cache by reimporting
import symcrash_python_ap.vm.coeff_lut_streaming as lut_mod

# Clear cache
lut_mod._NOTE_LUT_CACHE.clear()
print("Cache cleared")

# Check default P
print(f"DEFAULT_P = {lut_mod.DEFAULT_P}")

# Generate a new LUT
lut = lut_mod.get_lut_for_note(wave_id=0, midi_note=69)
print(f"Generated LUT for note 69:")
print(f"  P = {lut.P}")
print(f"  K = {lut.K}")
print(f"  shape = {lut.data.shape}")

# Verify P=256 is being used
assert lut.P == 256, f"Expected P=256, got P={lut.P}"
print("OK: LUT has P=256")
