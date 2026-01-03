"""Test effect_mode_jit behavior."""
import numpy as np
from symcrash_python_ap.wavelet.fast_lifting import effect_mode_jit

# Test: does effect_mode_jit with None mask produce identical output to input?
test_input = [int(1000 * np.sin(2 * np.pi * i / 64)) for i in range(64)]
output = effect_mode_jit(test_input, None)

# Compare
diff = [abs(a - b) for a, b in zip(test_input, output)]
print(f"Input first 8:  {test_input[:8]}")
print(f"Output first 8: {output[:8]}")
print(f"Max diff: {max(diff)}")
print(f"Identity? {max(diff) == 0}")

# Test with identity mask (all 32767)
identity_mask = [32767] * 64
output_identity = effect_mode_jit(test_input, identity_mask)
diff_identity = [abs(a - b) for a, b in zip(test_input, output_identity)]
print(f"\nWith identity mask [32767]*64:")
print(f"Output first 8: {output_identity[:8]}")
print(f"Max diff: {max(diff_identity)}")

# Check block boundary behavior
print("\n--- Block Boundary Test ---")
# Simulate two consecutive blocks with different phase positions
for phase_offset in [0, 16, 32, 48]:
    block = [int(10000 * np.sin(2 * np.pi * (i + phase_offset) / 64)) for i in range(64)]
    out = effect_mode_jit(block, None)
    print(f"Phase offset {phase_offset}: input[-1]={block[-1]}, output[-1]={out[-1]}, diff={block[-1]-out[-1]}")
