"""Compare roundtrip methods to find the source of error."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.wavelet.sine_core import sin_cos_q15, freq_to_phase_inc, PHASE_MASK

BLOCK_SIZE = 64
SAMPLE_RATE = 48000

print("=" * 70)
print("ROUNDTRIP COMPARISON: Float vs Q32 Phase")
print("=" * 70)


def test_roundtrip_float_phase(freq_hz, num_blocks=10):
    """Test with floating point numpy sin."""
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


def test_roundtrip_q32_phase(freq_hz, num_blocks=10):
    """Test with Q32 phase accumulator."""
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)
    phase = 0
    all_input = []
    all_output = []

    for _ in range(num_blocks):
        block = []
        for i in range(BLOCK_SIZE):
            sin_val, _ = sin_cos_q15(phase)
            block.append(sin_val)
            phase = (phase + phase_inc) & PHASE_MASK

        coeffs = fwpt_jit(block, headroom_bits=1)
        recon = idwpt_jit(coeffs, headroom_bits=1)

        all_input.extend(block)
        all_output.extend(recon)

    input_arr = np.array(all_input, dtype=np.float64)
    output_arr = np.array(all_output, dtype=np.float64)
    error = input_arr - output_arr

    return np.sqrt(np.mean(error**2)), np.max(np.abs(error))


# Test at various frequencies
print("\n--- Comparison at 440 Hz (H1) ---")
rms_f, max_f = test_roundtrip_float_phase(440)
rms_q, max_q = test_roundtrip_q32_phase(440)
print(f"Float phase: RMS={rms_f:.2f}, Max={max_f:.0f}")
print(f"Q32 phase:   RMS={rms_q:.2f}, Max={max_q:.0f}")

print("\n--- Comparison at 2200 Hz (H5) ---")
rms_f, max_f = test_roundtrip_float_phase(2200)
rms_q, max_q = test_roundtrip_q32_phase(2200)
print(f"Float phase: RMS={rms_f:.2f}, Max={max_f:.0f}")
print(f"Q32 phase:   RMS={rms_q:.2f}, Max={max_q:.0f}")

# Aha! Both methods should give same result if phase generation is correct
# Let's check if the Q32 sine LUT is accurate
print("\n" + "=" * 70)
print("Q32 SINE ACCURACY CHECK")
print("=" * 70)

# Compare Q32 sine to float sine
errors = []
for i in range(1000):
    phase_q32 = int(i * (2**32) / 1000) & PHASE_MASK
    phase_float = i * 2 * np.pi / 1000

    sin_q32, _ = sin_cos_q15(phase_q32)
    sin_float = int(32767 * np.sin(phase_float))

    error = sin_q32 - sin_float
    errors.append(error)

errors = np.array(errors)
print(f"Q32 vs Float sine error:")
print(f"  RMS: {np.sqrt(np.mean(errors**2)):.2f}")
print(f"  Max: {np.max(np.abs(errors)):.0f}")

# Let's check what's actually happening in the blocks
print("\n" + "=" * 70)
print("BLOCK-BY-BLOCK INPUT COMPARISON")
print("=" * 70)

freq_hz = 2200

# Generate with float
t = np.arange(BLOCK_SIZE) / SAMPLE_RATE
block_float = (16384 * np.sin(2 * np.pi * freq_hz * t)).astype(np.int16)

# Generate with Q32
phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)
block_q32 = []
phase = 0
for i in range(BLOCK_SIZE):
    sin_val, _ = sin_cos_q15(phase)
    block_q32.append(sin_val)
    phase = (phase + phase_inc) & PHASE_MASK
block_q32 = np.array(block_q32, dtype=np.int16)

print(f"First 8 samples:")
print(f"  Float: {block_float[:8].tolist()}")
print(f"  Q32:   {block_q32[:8].tolist()}")

diff = block_float.astype(np.float64) - block_q32.astype(np.float64)
print(f"\nInput difference:")
print(f"  RMS: {np.sqrt(np.mean(diff**2)):.2f}")
print(f"  Max: {np.max(np.abs(diff)):.0f}")

# The amplitude is different! Float uses 16384, Q32 uses 32767
print(f"\nAmplitude:")
print(f"  Float input max: {np.max(np.abs(block_float))}")
print(f"  Q32 input max:   {np.max(np.abs(block_q32))}")

# AH! That's the issue - the Q32 sine has amplitude 32767, float has 16384
# Let's normalize and retest
print("\n" + "=" * 70)
print("NORMALIZED TEST (same amplitude)")
print("=" * 70)


def test_roundtrip_q32_normalized(freq_hz, num_blocks=10):
    """Test with Q32 phase, normalized to 16384 amplitude."""
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)
    phase = 0
    all_input = []
    all_output = []

    for _ in range(num_blocks):
        block = []
        for i in range(BLOCK_SIZE):
            sin_val, _ = sin_cos_q15(phase)
            # Scale down to match float test (16384 vs 32767)
            scaled = sin_val // 2
            block.append(scaled)
            phase = (phase + phase_inc) & PHASE_MASK

        coeffs = fwpt_jit(block, headroom_bits=1)
        recon = idwpt_jit(coeffs, headroom_bits=1)

        all_input.extend(block)
        all_output.extend(recon)

    input_arr = np.array(all_input, dtype=np.float64)
    output_arr = np.array(all_output, dtype=np.float64)
    error = input_arr - output_arr

    return np.sqrt(np.mean(error**2)), np.max(np.abs(error))


print("\n--- 440 Hz Normalized ---")
rms_f, max_f = test_roundtrip_float_phase(440)
rms_q, max_q = test_roundtrip_q32_normalized(440)
print(f"Float:      RMS={rms_f:.2f}, Max={max_f:.0f}")
print(f"Q32 norm:   RMS={rms_q:.2f}, Max={max_q:.0f}")

print("\n--- 2200 Hz Normalized ---")
rms_f, max_f = test_roundtrip_float_phase(2200)
rms_q, max_q = test_roundtrip_q32_normalized(2200)
print(f"Float:      RMS={rms_f:.2f}, Max={max_f:.0f}")
print(f"Q32 norm:   RMS={rms_q:.2f}, Max={max_q:.0f}")

# SO: The amplitude doesn't explain the huge error
# Let's check if it's the integer arithmetic overflow
print("\n" + "=" * 70)
print("CHECKING FOR OVERFLOW IN FWPT")
print("=" * 70)

# Generate full-amplitude Q32 sine
phase_inc = freq_to_phase_inc(2200, SAMPLE_RATE)
phase = 0
block = []
for i in range(BLOCK_SIZE):
    sin_val, _ = sin_cos_q15(phase)
    block.append(sin_val)
    phase = (phase + phase_inc) & PHASE_MASK

print(f"Input range: {min(block)} to {max(block)}")

coeffs = fwpt_jit(block, headroom_bits=1)
print(f"Coeff range: {min(coeffs)} to {max(coeffs)}")

# Check if any coeffs are at saturation
saturated = sum(1 for c in coeffs if c == 32767 or c == -32768)
print(f"Saturated coefficients: {saturated} / {len(coeffs)}")

recon = idwpt_jit(coeffs, headroom_bits=1)
print(f"Recon range: {min(recon)} to {max(recon)}")

# If coefficients are saturating, that's the problem!
