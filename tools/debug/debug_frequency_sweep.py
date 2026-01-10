"""Frequency sweep to find problematic frequencies."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit

BLOCK_SIZE = 64
SAMPLE_RATE = 48000


def test_freq(freq_hz, num_blocks=10):
    """Test roundtrip at a specific frequency."""
    all_input = []
    all_output = []

    for block_idx in range(num_blocks):
        start = block_idx * BLOCK_SIZE
        t = np.arange(start, start + BLOCK_SIZE) / SAMPLE_RATE
        block = (16384 * np.sin(2 * np.pi * freq_hz * t)).astype(np.int16)

        coeffs = fwpt_jit(block.tolist(), headroom_bits=1)
        recon = idwpt_jit(coeffs, headroom_bits=1)

        all_input.extend(block.tolist())
        all_output.extend(recon)

    input_arr = np.array(all_input, dtype=np.float64)
    output_arr = np.array(all_output, dtype=np.float64)
    error = input_arr - output_arr

    return np.sqrt(np.mean(error**2)), np.max(np.abs(error))


print("=" * 70)
print("FREQUENCY SWEEP: Finding problematic frequencies")
print("=" * 70)

# Fine sweep around 2200 Hz
print("\n--- Fine sweep 2000-2500 Hz (step 50 Hz) ---")
print(f"{'Freq':>8} {'RMS':>8} {'Max':>8} {'cycles/blk':>12}")
print("-" * 40)

for freq in range(2000, 2501, 50):
    rms, max_err = test_freq(freq)
    cpb = freq * BLOCK_SIZE / SAMPLE_RATE
    marker = "***" if rms > 100 else ""
    print(f"{freq:8d} {rms:8.1f} {max_err:8.0f} {cpb:12.4f} {marker}")

# Analyze what makes a frequency "bad"
print("\n" + "=" * 70)
print("ANALYSIS: cycles/block relationship")
print("=" * 70)

print("\n--- Testing integer vs non-integer cycles/block ---")
# Test frequencies that give exactly integer cycles per block
# cycles/block = freq * 64 / 48000 = freq / 750
# integer cycles when freq is multiple of 750

for multiplier in [1, 1.5, 2, 2.5, 3, 3.5, 4]:
    freq = 750 * multiplier
    rms, max_err = test_freq(freq)
    cpb = freq * BLOCK_SIZE / SAMPLE_RATE
    print(f"freq={freq:6.0f} Hz (cycles/blk={cpb:5.2f}): RMS={rms:6.1f}, Max={max_err:6.0f}")

# The magic frequency for 2200 Hz is between 2 and 3 cycles
# 2200 / 750 = 2.933...
print(f"\n2200 Hz: {2200/750:.4f} cycles/block")

# What if we test frequencies that are exact multiples of 750/64 = 11.71875 Hz?
# These should give exact cycles per block
print("\n--- Testing exact cycle frequencies ---")
base = SAMPLE_RATE / BLOCK_SIZE  # 750 Hz = 1 cycle per block
for cycles in [1, 2, 3, 4, 5, 6]:
    freq = cycles * base
    rms, max_err = test_freq(freq)
    print(f"{cycles} cycles/blk (freq={freq:6.1f} Hz): RMS={rms:6.1f}, Max={max_err:6.0f}")

# Now test non-integer cycles
print("\n--- Testing non-integer cycle frequencies ---")
for cycles in [1.5, 2.5, 2.9, 2.93, 2.95, 3.5, 4.5]:
    freq = cycles * base
    rms, max_err = test_freq(freq)
    print(f"{cycles:.2f} cycles/blk (freq={freq:6.1f} Hz): RMS={rms:6.1f}, Max={max_err:6.0f}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
The wavelet roundtrip error depends on:
1. Whether cycles/block is close to an integer
2. Higher error for cycles close to N but not exactly N

For 440 Hz harmonics:
- H1 (440 Hz): 0.59 cycles/block - OK (far from integers)
- H5 (2200 Hz): 2.93 cycles/block - BAD (close to 3 but not quite)

The CDF 5/3 lifting scheme uses boundary extension that assumes
the signal continues symmetrically beyond block edges. When the
frequency doesn't align with block boundaries, this assumption
creates reconstruction error.

FIX OPTIONS:
1. Use overlap-add with proper windowing
2. Use a block size that better matches the frequency
3. Use a different wavelet that handles boundaries better
4. Accept the artifact for high harmonics (it's ~-20 dB relative)
""")
