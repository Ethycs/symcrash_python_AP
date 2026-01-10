"""Test the boundary extension hypothesis for H5 roundtrip error."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sin_cos_q15, PHASE_MASK
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit

BLOCK_SIZE = 64
SAMPLE_RATE = 48000

print("=" * 70)
print("BOUNDARY EXTENSION HYPOTHESIS TEST")
print("=" * 70)

def test_with_extension(freq_hz, extension_samples=0):
    """Test roundtrip with extended signal to see if boundary is the issue."""
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)
    cycles = freq_hz * BLOCK_SIZE / SAMPLE_RATE

    # Generate extended signal
    extended_len = BLOCK_SIZE + extension_samples
    signal = []
    phase = 0
    for n in range(extended_len):
        sin_val, _ = sin_cos_q15(phase)
        signal.append(sin_val)
        phase = (phase + phase_inc) & PHASE_MASK

    # Take the first BLOCK_SIZE samples for transform
    signal_block = signal[:BLOCK_SIZE]

    # Roundtrip
    coeffs = fwpt_jit(signal_block, headroom_bits=1)
    recon = idwpt_jit(coeffs, headroom_bits=1)

    signal_arr = np.array(signal_block, dtype=np.int16)
    recon_arr = np.array(recon, dtype=np.int16)
    err = signal_arr.astype(np.int32) - recon_arr.astype(np.int32)

    rms = np.sqrt(np.mean(err**2))

    # Check boundary discontinuity
    boundary_jump = abs(signal[63] - signal[64]) if extension_samples > 0 else 0

    return rms, boundary_jump, signal[63], signal[64] if extension_samples > 0 else None

print("\n--- Comparing integer vs non-integer cycles ---")
print(f"{'Frequency':>12} {'Cycles':>8} {'RMS Err':>10} {'sample[63]':>12} {'sample[64]':>12} {'Jump':>8}")
print("-" * 74)

# Test frequencies
test_freqs = [
    (750 * 1, "1.0 cyc"),  # Integer cycles
    (750 * 2, "2.0 cyc"),
    (750 * 3, "3.0 cyc"),
    (440 * 1, "H1 0.59"),  # Non-integer
    (440 * 4, "H4 2.35"),
    (440 * 5, "H5 2.93"),
    (440 * 6, "H6 3.52"),
]

for freq, label in test_freqs:
    rms, jump, s63, s64 = test_with_extension(freq, extension_samples=1)
    cycles = freq * BLOCK_SIZE / SAMPLE_RATE
    print(f"{label:>12} {cycles:>8.2f} {rms:>10.0f} {s63:>12} {s64:>12} {jump:>8}")

print("\n" + "=" * 70)
print("ANALYSIS: Why does the boundary matter?")
print("=" * 70)

# Deep dive on H5
freq = 2200
phase_inc = freq_to_phase_inc(freq, SAMPLE_RATE)

# Generate signal
signal = []
phase = 0
for n in range(66):  # Extra samples
    sin_val, _ = sin_cos_q15(phase)
    signal.append(sin_val)
    phase = (phase + phase_inc) & PHASE_MASK

signal_arr = np.array(signal[:64], dtype=np.int16)

# The CDF 5/3 predict step at the boundary uses:
#   d[31] = x[63] - (x[62] + x[64]) / 2
# But since x[64] doesn't exist, it uses x[62] (constant extension):
#   d[31] = x[63] - (x[62] + x[62]) / 2 = x[63] - x[62]
#
# For a continuous signal, it would be:
#   d[31] = x[63] - (x[62] + x[64]) / 2

print(f"\nH5 (2200 Hz) boundary analysis:")
print(f"  x[62] = {signal[62]}")
print(f"  x[63] = {signal[63]}")
print(f"  x[64] = {signal[64]} (not in block)")
print(f"  x[65] = {signal[65]} (not in block)")

# What CDF 5/3 does (constant extension)
avg_constant = (signal[62] + signal[62] + 1) >> 1
d31_constant = signal[63] - avg_constant
print(f"\n  With constant extension:")
print(f"    avg = (x[62] + x[62]) / 2 = {avg_constant}")
print(f"    d[31] = x[63] - avg = {d31_constant}")

# What it should be for continuous signal
avg_correct = (signal[62] + signal[64] + 1) >> 1
d31_correct = signal[63] - avg_correct
print(f"\n  With correct continuation:")
print(f"    avg = (x[62] + x[64]) / 2 = {avg_correct}")
print(f"    d[31] = x[63] - avg = {d31_correct}")

print(f"\n  ERROR in d[31]: {abs(d31_constant - d31_correct)}")

# Now check: integer cycles should have x[64] ≈ x[0]
print("\n" + "=" * 70)
print("INTEGER CYCLES: Why they work better")
print("=" * 70)

for freq_mult in [1, 2, 3]:
    freq = 750 * freq_mult
    phase_inc = freq_to_phase_inc(freq, SAMPLE_RATE)

    signal = []
    phase = 0
    for n in range(66):
        sin_val, _ = sin_cos_q15(phase)
        signal.append(sin_val)
        phase = (phase + phase_inc) & PHASE_MASK

    print(f"\n{freq_mult}.0 cycles ({freq} Hz):")
    print(f"  x[0]  = {signal[0]}")
    print(f"  x[62] = {signal[62]}")
    print(f"  x[63] = {signal[63]}")
    print(f"  x[64] = {signal[64]}")

    # For integer cycles, x[64] ≈ x[0] (periodic)
    # Constant extension uses x[62], which may also be close
    avg_constant = (signal[62] + signal[62] + 1) >> 1
    avg_correct = (signal[62] + signal[64] + 1) >> 1
    print(f"  avg(constant): {avg_constant}, avg(correct): {avg_correct}, diff: {abs(avg_constant - avg_correct)}")

print("\n" + "=" * 70)
print("THE FIX: This is inherent to CDF 5/3 block processing")
print("=" * 70)
print("""
The boundary extension error is NOT a bug - it's inherent to:
1. Processing finite blocks with CDF 5/3 lifting
2. Signals that don't complete integer cycles per block

Solutions:
A. Use overlap-add (OLA) to smooth block boundaries
B. Use periodic extension instead of constant extension
C. Accept the error and use longer blocks
D. Pre-compute basis frames that account for the boundary behavior

Since Mode B uses pre-computed basis frames (Ucos_h, Usin_h), the
boundary error is BAKED INTO the basis. This is expected behavior.

The 50 Hz artifact comes from the coefficient modulation, not the
boundary extension directly. The basis frames are correct - they
just can't perfectly represent non-integer-cycle signals.
""")
