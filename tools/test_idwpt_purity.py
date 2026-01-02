#!/usr/bin/env python3
"""Test IDWPT round-trip error with different input signals."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from symcrash_python_ap.wavelet.fast_lifting import effect_mode_jit

def test_roundtrip(name, samples):
    """Test FWPT->IDWPT round-trip error."""
    carrier_list = [int(s) for s in samples]
    wet = effect_mode_jit(carrier_list, None)
    wet_arr = np.array(wet, dtype=np.int32)
    samples_arr = np.array(samples, dtype=np.int32)

    diff = samples_arr - wet_arr
    max_err = np.max(np.abs(diff))
    rms_err = np.sqrt(np.mean(diff**2))

    print(f"{name}:")
    print(f"  Input range: [{samples_arr.min()}, {samples_arr.max()}]")
    print(f"  Error: max={max_err}, rms={rms_err:.2f}")
    return max_err, rms_err

# Test 1: Pure 440Hz sine (centered at 0)
samples_pure = [int(32767 * np.sin(2 * np.pi * 440 * i / 48000)) for i in range(64)]
test_roundtrip("Pure 440Hz sine", samples_pure)

# Test 2: Same amplitude but offset (simulating DC)
samples_offset = [int(s + 5000) for s in samples_pure]
test_roundtrip("440Hz sine + 5000 offset", samples_offset)

# Test 3: Smaller amplitude
samples_small = [int(16384 * np.sin(2 * np.pi * 440 * i / 48000)) for i in range(64)]
test_roundtrip("440Hz sine (half amplitude)", samples_small)

# Test 4: Very small amplitude
samples_tiny = [int(1000 * np.sin(2 * np.pi * 440 * i / 48000)) for i in range(64)]
test_roundtrip("440Hz sine (tiny amplitude)", samples_tiny)

# Test 5: DC only
samples_dc = [10000] * 64
test_roundtrip("Pure DC = 10000", samples_dc)

# Test 6: Zero
samples_zero = [0] * 64
test_roundtrip("All zeros", samples_zero)

# Test 7: Actual carrier-like values from debug output
samples_real = [0, 1779, 3553, 5310, 7042, 8750, 10424, 12059, 13646, 15186,
                16663, 18072, 19413, 20684, 21875, 22983, 24005, 24940, 25783, 26533,
                27188, 27745, 28201, 28557, 28812, 28964, 29013, 28958, 28800, 28540,
                28179, 27718, 27159, 26504, 25757, 24919, 23996, 22991, 21907, 20751,
                19527, 18241, 16897, 15502, 14060, 12579, 11063, 9519, 7954, 6373,
                4782, 3187, 1594, 8, -1566, -3124, -4663, -6180, -7670, -9132,
                -10561, -11955, -13309, -14622]
test_roundtrip("Actual carrier values", samples_real)

# Test 8: Actual values with explicit headroom bits check
print("\n" + "=" * 60)
print("Checking headroom_bits effect:")
from symcrash_python_ap.wavelet.fast_lifting import _get_jit_processor

processor = _get_jit_processor()
for headroom in [0, 1, 2, 3, 4]:
    wet = processor.effect_mode(samples_real, None, headroom)
    wet_arr = np.array(wet, dtype=np.int32)
    samples_arr = np.array(samples_real, dtype=np.int32)
    diff = samples_arr - wet_arr
    print(f"  headroom={headroom}: max_err={np.max(np.abs(diff))}, rms={np.sqrt(np.mean(diff**2)):.2f}")
