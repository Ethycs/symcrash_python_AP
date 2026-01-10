"""Test which interleave indexing is correct for shifted-block FWPT."""

import numpy as np
import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit

# Create a simple test waveform - ramp for easy verification
K = 8
B = 64  # block size

# Generate a simple ramp: 0, 1, 2, ..., 127
waveform = np.arange(128, dtype=np.int16) * 100  # Scale for visibility

print('=== SHIFTED-BLOCK FWPT TEST ===')
print(f'Waveform[0:10]: {waveform[0:10].tolist()}')
print()

# For phase p=0, start=0
# Build coefficients for each k
start = 0
coeffs_by_k = []
for k in range(K):
    # Block starts at (start + k), length 64
    block = np.array([waveform[(start + k + n) % len(waveform)] for n in range(B)], dtype=np.int16)
    print(f'k={k}: block[0:5] = {block[0:5].tolist()} (starts at waveform[{start+k}])')
    coeffs = fwpt_jit(block.tolist(), headroom_bits=1)
    coeffs_by_k.append(coeffs)

print()
print('=== IDWPT RECONSTRUCTION ===')

# Now reconstruct each k and check what we get
for k in range(K):
    y_k = idwpt_jit(coeffs_by_k[k], headroom_bits=1)
    print(f'k={k}: y_k[0:8] = {y_k[0:8]}')
    print(f'      Expected (waveform[{start+k}:{start+k+8}]): {waveform[start+k:start+k+8].tolist()}')
    print()

print('=== INTERLEAVE OPTIONS ===')
print()

# Option A: out[k::K] = y_k[0::K]
out_A = np.zeros(B, dtype=np.int16)
for k in range(K):
    y_k = np.array(idwpt_jit(coeffs_by_k[k], headroom_bits=1), dtype=np.int16)
    out_A[k::K] = y_k[0::K]

print('Option A: out[k::K] = y_k[0::K]')
print(f'  out[0:16] = {out_A[0:16].tolist()}')
print(f'  Expected  = {waveform[start:start+16].tolist()}')
print(f'  Match: {np.allclose(out_A, waveform[start:start+B], atol=10)}')
err_A = np.abs(out_A.astype(np.int32) - waveform[start:start+B].astype(np.int32))
print(f'  Max error: {err_A.max()}')
print()

# Option B: out[n] = y_{n%K}[n]
out_B = np.zeros(B, dtype=np.int16)
for n in range(B):
    k = n % K
    y_k = idwpt_jit(coeffs_by_k[k], headroom_bits=1)
    out_B[n] = y_k[n]

print('Option B: out[n] = y_{n%K}[n]')
print(f'  out[0:16] = {out_B[0:16].tolist()}')
print(f'  Expected  = {waveform[start:start+16].tolist()}')
print(f'  Match: {np.allclose(out_B, waveform[start:start+B], atol=10)}')
err_B = np.abs(out_B.astype(np.int32) - waveform[start:start+B].astype(np.int32))
print(f'  Max error: {err_B.max()}')
print()

# Option C: out[k::K] = y_k[k::K]
out_C = np.zeros(B, dtype=np.int16)
for k in range(K):
    y_k = np.array(idwpt_jit(coeffs_by_k[k], headroom_bits=1), dtype=np.int16)
    out_C[k::K] = y_k[k::K]

print('Option C: out[k::K] = y_k[k::K]')
print(f'  out[0:16] = {out_C[0:16].tolist()}')
print(f'  Expected  = {waveform[start:start+16].tolist()}')
print(f'  Match: {np.allclose(out_C, waveform[start:start+B], atol=10)}')
err_C = np.abs(out_C.astype(np.int32) - waveform[start:start+B].astype(np.int32))
print(f'  Max error: {err_C.max()}')
print()

# What about the ACTUAL goal: reconstruct waveform[0:64]?
# y_k reconstructs waveform[k:k+64], so:
# - y_0 reconstructs waveform[0:64]   -> y_0[n] = waveform[n]
# - y_1 reconstructs waveform[1:65]   -> y_1[n] = waveform[n+1]
# - y_k reconstructs waveform[k:k+64] -> y_k[n] = waveform[n+k]
#
# To get waveform[n] for n in 0..63, we need:
# - For n=0: y_0[0]
# - For n=1: y_1[0]  (because y_1[0] = waveform[1])
# - For n=k: y_k[0]  (because y_k[0] = waveform[k])
# - For n=8: y_0[8]  (because y_0[8] = waveform[8])
# - For n=9: y_1[8]  (because y_1[8] = waveform[9])
#
# Pattern: waveform[n] = y_{n%K}[n - (n%K)] = y_{n%K}[(n//K)*K]
#
# Or equivalently: out[k + m*K] = y_k[m*K] for k=0..K-1, m=0..7

print('=== CORRECT DERIVATION ===')
print('y_k reconstructs waveform[k:k+64]')
print('So y_k[n] = waveform[k + n]')
print('To get waveform[j], we need y_{j%K}[j - j%K]')
print()

out_D = np.zeros(B, dtype=np.int16)
for j in range(B):
    k = j % K
    idx = j - k  # This is (j // K) * K
    y_k = idwpt_jit(coeffs_by_k[k], headroom_bits=1)
    out_D[j] = y_k[idx]

print('Option D: out[j] = y_{j%K}[j - j%K]')
print(f'  out[0:16] = {out_D[0:16].tolist()}')
print(f'  Expected  = {waveform[start:start+16].tolist()}')
print(f'  Match: {np.allclose(out_D, waveform[start:start+B], atol=10)}')
err_D = np.abs(out_D.astype(np.int32) - waveform[start:start+B].astype(np.int32))
print(f'  Max error: {err_D.max()}')
print()

# This is equivalent to: out[k::K] = y_k[0::K]
# Because for m = 0, 1, 2, ...:
#   out[k + m*K] = y_k[(k + m*K) - k] = y_k[m*K]
# Which is exactly out[k::K] = y_k[0::K]

print('=== VERIFICATION ===')
print(f'Option A == Option D: {np.array_equal(out_A, out_D)}')
