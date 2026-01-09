import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
import numpy as np

print('Testing IDWPT with simple coefficient patterns\n')

# Test 1: Single non-zero coefficient
print('Test 1: Single coeff[0] = 10000')
coeffs = [10000] + [0] * 63
audio = idwpt_jit(coeffs, headroom_bits=1)
audio_arr = np.array(audio)
print(f'  Audio: min={audio_arr.min()}, max={audio_arr.max()}, mean={audio_arr.mean():.1f}')
print(f'  First 10: {audio_arr[:10]}')
print(f'  Unique values: {len(np.unique(audio_arr))}')

# Test 2: Try coefficient at position 1
print('\nTest 2: Single coeff[1] = 10000')
coeffs = [0, 10000] + [0] * 62
audio = idwpt_jit(coeffs, headroom_bits=1)
audio_arr = np.array(audio)
print(f'  Audio: min={audio_arr.min()}, max={audio_arr.max()}, mean={audio_arr.mean():.1f}')
print(f'  First 10: {audio_arr[:10]}')

# Test 3: Oscillating coefficient[0]
print('\nTest 3: Varying coeff[0] over multiple blocks')
for val in [0, 10000, 20000, 10000, 0, -10000, -20000, -10000]:
    coeffs = [val] + [0] * 63
    audio = idwpt_jit(coeffs, headroom_bits=1)
    print(f'  coeff[0]={val:6d} → audio[0]={audio[0]:6d}, audio[32]={audio[32]:6d}')
