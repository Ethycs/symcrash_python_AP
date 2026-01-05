import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.vm.coeff_tables_fast import generate_sinusoid_block_fast
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit, idwpt_polyphase_streaming
import numpy as np

print('Testing streaming vs standard IDWPT\n')

# Generate reference sinusoid
freq = 261.63
audio_ref = generate_sinusoid_block_fast(freq, 0, 64)

# Get coefficients via FWPT
coeffs = fwpt_jit(audio_ref.tolist(), headroom_bits=1)
coeffs_arr = np.array(coeffs, dtype=np.int16)

print(f'Reference audio: {audio_ref[:5]}')
print(f'Coefficients: {coeffs_arr[:5]}')

# Test 1: Standard IDWPT (current working method)
audio_standard = idwpt_jit(coeffs, headroom_bits=1)
audio_std_arr = np.array(audio_standard, dtype=np.int16)

print(f'\nStandard IDWPT:')
print(f'  Output: {audio_std_arr[:10]}')
print(f'  Range: {audio_std_arr.min()} to {audio_std_arr.max()}')
print(f'  RMS: {np.sqrt(np.mean(audio_std_arr.astype(np.float32)**2)):.1f}')

# Test 2: Create time-series by replicating coefficients
coeff_state_256 = np.zeros(256, dtype=np.int16)
for i in range(64):
    for j in range(4):
        coeff_state_256[i*4 + j] = coeffs_arr[i]

print(f'\nCoeff state (replicated): {coeff_state_256[:8]}')

# Test 3: Streaming polyphase IDWPT
audio_streaming = idwpt_polyphase_streaming(coeff_state_256, headroom_bits=1)

print(f'\nStreaming polyphase IDWPT:')
print(f'  Output: {audio_streaming[:10]}')
print(f'  Range: {audio_streaming.min()} to {audio_streaming.max()}')
print(f'  RMS: {np.sqrt(np.mean(audio_streaming.astype(np.float32)**2)):.1f}')

# Compare
error = np.mean(np.abs(audio_std_arr - audio_streaming))
print(f'\nDifference: {error:.1f} samples average error')

if error < 10:
    print('[OK] Streaming IDWPT matches standard')
else:
    print(f'[WARN] Large difference: {error:.1f}')
