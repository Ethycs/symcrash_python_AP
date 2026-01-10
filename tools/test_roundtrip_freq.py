"""Test roundtrip error for integer vs non-integer cycles/block."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sin_cos_q15, PHASE_MASK
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit

BLOCK_SIZE = 64
SAMPLE_RATE = 48000

print("=" * 60)
print("ROUNDTRIP ERROR VS CYCLES/BLOCK")
print("=" * 60)

def test_freq(freq_hz, label):
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)
    cycles = freq_hz * BLOCK_SIZE / SAMPLE_RATE

    sin_block = []
    phase = 0
    for n in range(BLOCK_SIZE):
        sin_val, _ = sin_cos_q15(phase)
        sin_block.append(sin_val)
        phase = (phase + phase_inc) & PHASE_MASK

    coeffs = fwpt_jit(sin_block, headroom_bits=1)
    recon = idwpt_jit(coeffs, headroom_bits=1)

    sin_input = np.array(sin_block, dtype=np.int16)
    sin_recon = np.array(recon, dtype=np.int16)
    err = sin_input.astype(np.int32) - sin_recon.astype(np.int32)

    rms = np.sqrt(np.mean(err**2))
    max_err = np.max(np.abs(err))

    print("%s (%d Hz, %.2f cyc/blk): RMS=%.0f, Max=%d" % (label, freq_hz, cycles, rms, max_err))
    return rms

print()
test_freq(750 * 1, "1.0 cycles")
test_freq(750 * 2, "2.0 cycles")
test_freq(750 * 3, "3.0 cycles")
test_freq(750 * 4, "4.0 cycles")

print()
test_freq(2200, "H5 (2.93)")  # 440 * 5
test_freq(2640, "H6 (3.52)")  # 440 * 6
test_freq(3080, "H7 (4.11)")  # 440 * 7

print()
# Test near-integer cases
test_freq(750 * 2.9, "2.9 cycles")
test_freq(750 * 2.93, "2.93 cycles")
test_freq(750 * 2.95, "2.95 cycles")
test_freq(750 * 2.99, "2.99 cycles")
