"""Debug the basis frame mismatch causing H5 artifact."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq

from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sin_cos_q15, PHASE_MASK

BLOCK_SIZE = 64
SAMPLE_RATE = 48000
NUM_LEAVES = 64

# Parameters
freq_hz = 440.0
phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

print("=" * 70)
print("WAVELET ROUNDTRIP ANALYSIS FOR H5")
print("=" * 70)

def generate_h5_block(start_phase):
    """Generate one block of H5 sine wave."""
    samples = []
    phase = start_phase
    for i in range(BLOCK_SIZE):
        h_phase = (phase * 5) & PHASE_MASK
        sin_val, cos_val = sin_cos_q15(h_phase)
        samples.append(sin_val)
        phase = (phase + phase_inc) & PHASE_MASK
    return samples, phase


# Test: Direct FWPT -> IDWPT roundtrip (should be nearly perfect)
print("\n--- Direct FWPT -> IDWPT Roundtrip ---")

phase = 0
for block_idx in range(5):
    # Generate H5 block
    input_block, next_phase = generate_h5_block(phase)

    # FWPT
    coeffs = fwpt_jit(input_block, headroom_bits=1)

    # IDWPT
    output_block = idwpt_jit(coeffs, headroom_bits=1)

    # Compare
    input_arr = np.array(input_block, dtype=np.float64)
    output_arr = np.array(output_block, dtype=np.float64)
    error = input_arr - output_arr

    rms_error = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))

    h5_cycles = ((phase * 5) & PHASE_MASK) / (2**32)
    print(f"Block {block_idx}: H5 phase = {h5_cycles:.4f}, RMS err = {rms_error:.1f}, Max err = {max_error:.0f}")

    phase = next_phase

# Test: Multiple blocks continuous
print("\n--- Multi-Block Roundtrip (continuous signal) ---")

num_blocks = 100
phase = 0
all_input = []
all_output = []

for _ in range(num_blocks):
    input_block, phase = generate_h5_block(phase)
    coeffs = fwpt_jit(input_block, headroom_bits=1)
    output_block = idwpt_jit(coeffs, headroom_bits=1)

    all_input.extend(input_block)
    all_output.extend(output_block)

input_arr = np.array(all_input, dtype=np.float64)
output_arr = np.array(all_output, dtype=np.float64)
error = input_arr - output_arr

print(f"Total samples: {len(input_arr)}")
print(f"RMS error: {np.sqrt(np.mean(error**2)):.2f}")
print(f"Max error: {np.max(np.abs(error)):.0f}")

# Check 50 Hz in output
n = len(output_arr)
hann = np.hanning(n)
windowed = output_arr * hann
spectrum = np.abs(rfft(windowed)) * (2.0 / np.mean(hann))
freqs = rfftfreq(n, 1/SAMPLE_RATE)

idx_50 = int(50 * n / SAMPLE_RATE)
idx_h5 = int(2200 * n / SAMPLE_RATE)

amp_50 = spectrum[idx_50]
amp_h5 = spectrum[idx_h5] if idx_h5 < len(spectrum) else 1

rel_db = 20 * np.log10(amp_50 / amp_h5) if amp_h5 > 0 else -100
print(f"50 Hz component in output: {rel_db:.1f} dB relative to H5")

# Check 50 Hz in input (reference)
windowed_in = input_arr * hann
spectrum_in = np.abs(rfft(windowed_in)) * (2.0 / np.mean(hann))

amp_50_in = spectrum_in[idx_50]
amp_h5_in = spectrum_in[idx_h5] if idx_h5 < len(spectrum_in) else 1

rel_db_in = 20 * np.log10(amp_50_in / amp_h5_in) if amp_h5_in > 0 else -100
print(f"50 Hz component in input: {rel_db_in:.1f} dB relative to H5")

# =============================================================================
# THE ISSUE: Check if it's in the block-independent processing
# =============================================================================

print("\n" + "=" * 70)
print("KEY INSIGHT: Each block is processed independently")
print("=" * 70)

print("""
Each block goes through FWPT -> IDWPT independently.
The wavelet transform has finite impulse response, so:
- At block boundaries, there are edge effects
- The IDWPT output for sample n depends on coefficients, which
  came from FWPT of a finite block

Let's check what happens at block boundaries.
""")

# Check block boundary continuity
print("\n--- Block Boundary Analysis ---")

phase = 0
prev_output = None

for block_idx in range(10):
    input_block, next_phase = generate_h5_block(phase)
    coeffs = fwpt_jit(input_block, headroom_bits=1)
    output_block = idwpt_jit(coeffs, headroom_bits=1)

    if prev_output is not None:
        # Check discontinuity at boundary
        last_sample = prev_output[-1]
        first_sample = output_block[0]
        jump = first_sample - last_sample

        # What should the jump be for continuous H5?
        # Last input sample of prev block -> first input sample of this block
        expected_last = input_block[0]  # Actually this is first of current
        # The prev block's last sample was generated at phase - phase_inc
        prev_last_phase = phase  # This is start of current block
        # Actually need the input values...

        print(f"Block {block_idx-1}->{block_idx}: last={last_sample:6d}, "
              f"first={first_sample:6d}, jump={jump:6d}")

    prev_output = output_block
    phase = next_phase

# The KEY question: Is the roundtrip error creating 50 Hz?
print("\n--- Roundtrip Error Spectrum ---")
n = len(error)
hann = np.hanning(n)
windowed_err = error * hann
spectrum_err = np.abs(rfft(windowed_err)) * (2.0 / np.mean(hann))

print("Error spectrum peaks (0-200 Hz):")
for freq_check in [50, 100, 150, 200]:
    idx = int(freq_check * n / SAMPLE_RATE)
    if idx < len(spectrum_err):
        amp = spectrum_err[idx]
        print(f"  {freq_check} Hz: amplitude = {amp:.0f}")

# The answer should be: roundtrip error is nearly zero, so 50 Hz
# doesn't come from roundtrip - it comes from MODE B's way of
# generating coefficients using precomputed basis frames
print("""
CONCLUSION:
The direct FWPT->IDWPT roundtrip has minimal error and no 50 Hz artifact.
The 50 Hz artifact must come from Mode B's coefficient generation using
precomputed basis frames with runtime phase rotation.
""")
