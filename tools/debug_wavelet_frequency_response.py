"""Debug wavelet frequency response - find frequencies with poor reconstruction."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq

from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sin_cos_q15, PHASE_MASK

BLOCK_SIZE = 64
SAMPLE_RATE = 48000

print("=" * 70)
print("WAVELET FREQUENCY RESPONSE ANALYSIS")
print("=" * 70)


def test_frequency(freq_hz, num_blocks=50, verbose=False):
    """Test FWPT->IDWPT roundtrip at a specific frequency."""
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

    all_input = []
    all_output = []
    phase = 0

    for _ in range(num_blocks):
        # Generate sine block
        input_block = []
        for i in range(BLOCK_SIZE):
            sin_val, _ = sin_cos_q15(phase)
            input_block.append(sin_val)
            phase = (phase + phase_inc) & PHASE_MASK

        # FWPT -> IDWPT
        coeffs = fwpt_jit(input_block, headroom_bits=1)
        output_block = idwpt_jit(coeffs, headroom_bits=1)

        all_input.extend(input_block)
        all_output.extend(output_block)

    input_arr = np.array(all_input, dtype=np.float64)
    output_arr = np.array(all_output, dtype=np.float64)
    error = input_arr - output_arr

    rms_error = np.sqrt(np.mean(error**2))

    # Check for 50 Hz artifact
    n = len(output_arr)
    if n > 1000:
        hann = np.hanning(n)
        windowed = output_arr * hann
        spectrum = np.abs(rfft(windowed)) * (2.0 / np.mean(hann))
        freqs = rfftfreq(n, 1/SAMPLE_RATE)

        idx_50 = int(50 * n / SAMPLE_RATE)
        idx_fund = int(freq_hz * n / SAMPLE_RATE)

        amp_50 = spectrum[idx_50]
        amp_fund = spectrum[idx_fund] if idx_fund < len(spectrum) else 1

        rel_50 = 20 * np.log10(amp_50 / amp_fund) if amp_fund > 0 else -100
    else:
        rel_50 = 0

    return rms_error, rel_50


# Test across the frequency range
print("\n--- Roundtrip Error vs Frequency ---")
print(f"{'Freq (Hz)':>10} {'RMS Error':>12} {'50Hz (dB)':>12} {'cycles/blk':>12}")
print("-" * 50)

test_freqs = [100, 200, 300, 400, 440, 500, 600, 700, 800, 880, 1000,
              1100, 1200, 1320, 1500, 1760, 2000, 2200, 2400, 2640, 3000]

bad_freqs = []
for freq in test_freqs:
    rms_err, rel_50 = test_frequency(freq)
    cycles_per_block = freq * BLOCK_SIZE / SAMPLE_RATE

    marker = "***" if rms_err > 5000 else ""
    print(f"{freq:10.0f} {rms_err:12.1f} {rel_50:12.1f} {cycles_per_block:12.4f} {marker}")

    if rms_err > 5000:
        bad_freqs.append((freq, rms_err, cycles_per_block))

# Analyze bad frequencies
print("\n" + "=" * 70)
print("ANALYSIS OF HIGH-ERROR FREQUENCIES")
print("=" * 70)

if bad_freqs:
    print("\nFrequencies with RMS error > 5000:")
    for freq, err, cpb in bad_freqs:
        frac = cpb % 1
        beat = (SAMPLE_RATE / BLOCK_SIZE) * frac if frac < 0.5 else (SAMPLE_RATE / BLOCK_SIZE) * (1 - frac)
        print(f"  {freq} Hz: error={err:.0f}, cycles/block={cpb:.4f}, frac={frac:.4f}, beat={beat:.1f} Hz")

# Test harmonics of 440 Hz specifically
print("\n--- 440 Hz Harmonics Analysis ---")
print(f"{'Harmonic':>10} {'Freq (Hz)':>10} {'RMS Error':>12} {'50Hz (dB)':>12} {'cycles/blk':>12}")
print("-" * 60)

for h in range(1, 9):
    freq = 440 * h
    if freq < SAMPLE_RATE / 2:  # Below Nyquist
        rms_err, rel_50 = test_frequency(freq)
        cycles_per_block = freq * BLOCK_SIZE / SAMPLE_RATE
        frac = cycles_per_block % 1
        beat = (SAMPLE_RATE / BLOCK_SIZE) * frac if frac < 0.5 else (SAMPLE_RATE / BLOCK_SIZE) * (1 - frac)

        marker = f"(beat={beat:.0f}Hz)" if rms_err > 3000 else ""
        print(f"{h:10d} {freq:10.0f} {rms_err:12.1f} {rel_50:12.1f} {cycles_per_block:12.4f} {marker}")

# Key insight
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
The wavelet roundtrip (FWPT->IDWPT) has significant reconstruction error
for frequencies where:
  cycles_per_block is NOT close to an integer

This is because the CDF 5/3 wavelet assumes the signal continues
beyond the block boundaries (uses boundary extension). When the signal
doesn't align with block boundaries, the boundary extension creates errors.

For 440 Hz:
- H1 (440 Hz): 0.59 cycles/block - error ~1800
- H2 (880 Hz): 1.17 cycles/block - error ~3000
- H3 (1320 Hz): 1.76 cycles/block - error ~2700
- H4 (1760 Hz): 2.35 cycles/block - error ~4500
- H5 (2200 Hz): 2.93 cycles/block - error ~9200 ***
- H6 (2640 Hz): 3.52 cycles/block - error ~8500
- H7 (3080 Hz): 4.11 cycles/block - error ~3000
- H8 (3520 Hz): 4.69 cycles/block - error ~8500

H5 has the worst error because 0.93 is very close to 1.0 but not quite,
creating maximum boundary mismatch.

SOLUTION: The wavelet synthesis needs overlap-add or proper boundary handling.
""")
