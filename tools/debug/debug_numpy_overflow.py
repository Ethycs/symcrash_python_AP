"""Debug numpy overflow in interpolation."""

import numpy as np
import sys
sys.path.insert(0, 'src')

print("=== NUMPY OVERFLOW DEBUG ===")
print()

# The problem
diff_int32 = np.int32(-185)
frac = 12582912  # Python int

print(f"diff as np.int32: {diff_int32}")
print(f"frac as Python int: {frac}")
print()

# This overflows because numpy broadcasts to int32
result1 = diff_int32 * frac
print(f"np.int32(-185) * 12582912 = {result1}")
print(f"Type: {type(result1)}, dtype: {result1.dtype if hasattr(result1, 'dtype') else 'N/A'}")
print()

# Correct calculation with int64
result2 = np.int64(diff_int32) * frac
print(f"np.int64(-185) * 12582912 = {result2}")
print()

# Or ensure frac is int64
result3 = diff_int32 * np.int64(frac)
print(f"np.int32(-185) * np.int64(12582912) = {result3}")
print()

# The actual maximum product we need
max_diff = 350  # Approximate max coefficient difference
max_frac = (1 << 24) - 1  # Max fractional value for 24-bit frac
max_product = max_diff * max_frac
print(f"Max product: {max_diff} * {max_frac} = {max_product}")
print(f"int32 max: {2**31 - 1}")
print(f"int64 max: {2**63 - 1}")
print(f"Needs int64? {max_product > 2**31 - 1}")
