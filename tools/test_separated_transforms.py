"""Test separated FWPT/IDWPT functions."""
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit, NUMBA_AVAILABLE
import numpy as np

print(f'NUMBA_AVAILABLE: {NUMBA_AVAILABLE}')

# Test roundtrip
samples = [int(16000 * np.sin(2 * np.pi * i / 64)) for i in range(64)]
print(f'Input samples (first 8): {samples[:8]}')

# Forward transform
coeffs = fwpt_jit(samples)
print(f'Coefficients (first 8): {coeffs[:8]}')

# Inverse transform
reconstructed = idwpt_jit(coeffs)
print(f'Reconstructed (first 8): {reconstructed[:8]}')

# Check error
error = [abs(s - r) for s, r in zip(samples, reconstructed)]
print(f'Max roundtrip error: {max(error)}')
print(f'Mean roundtrip error: {sum(error)/len(error):.2f}')

# Verify separated matches fused
from symcrash_python_ap.wavelet.fast_lifting import effect_mode_jit

# Use identity mask (all 32767)
mask_identity = [32767] * 64
fused_out = effect_mode_jit(samples, mask_identity, headroom_bits=1)
print(f'\nFused output (first 8): {fused_out[:8]}')
print(f'Separated reconstructed (first 8): {reconstructed[:8]}')

# Compare
fused_sep_diff = [abs(f - s) for f, s in zip(fused_out, reconstructed)]
print(f'Max fused vs separated diff: {max(fused_sep_diff)}')

if max(error) <= 4:
    print('\nSEPARATED TRANSFORMS: OK (roundtrip error <= 4 LSB)')
else:
    print('\nSEPARATED TRANSFORMS: ERROR (roundtrip error too large)')
