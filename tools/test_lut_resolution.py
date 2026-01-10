"""Test if higher LUT resolution helps H5."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq

from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sine_q15
from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_block
from symcrash_python_ap.vm.coeff_lut_streaming import StreamingCoeffLUT, VM_K, BLOCK_SIZE, NUM_LEAVES

SAMPLE_RATE = 48000

print("=" * 70)
print("LUT RESOLUTION TEST FOR H5")
print("=" * 70)

freq_hz = 440.0
phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)
h = 5

def test_resolution(P):
    """Test H5 with P phase bins."""
    def gen_h5(phase):
        h_phase = (phase * h) & 0xFFFFFFFF
        return sine_q15(h_phase)

    lut = StreamingCoeffLUT.generate_for_frequency(
        wave_id=0,
        name="h5",
        target_freq_hz=freq_hz,
        generate_sample_fn=gen_h5,
        K=VM_K,
        P=P,
        normalize=False,
    )

    samples = []
    phase = 0
    for _ in range(100):
        output = np.zeros((NUM_LEAVES, VM_K), dtype=np.int16)
        lut.get_subband_samples(phase, output)
        audio = idwpt_polyphase_block(output, headroom_bits=1)
        samples.extend(audio.tolist())
        phase = (phase + BLOCK_SIZE * phase_inc) & 0xFFFFFFFF

    samples = np.array(samples, dtype=np.float64)
    n = len(samples)
    hann = np.hanning(n)
    spectrum = np.abs(rfft(samples * hann)) * (2.0 / np.mean(hann))

    idx_h5 = int(2200 * n / SAMPLE_RATE)
    amp_h5 = spectrum[idx_h5] if idx_h5 < len(spectrum) else 1
    idx_50 = int(50 * n / SAMPLE_RATE)
    amp_50 = spectrum[idx_50]

    rel = 20 * np.log10(amp_50 / amp_h5) if amp_h5 > 0 else -100

    # Also check max coefficient discontinuity
    data = lut.data
    max_jump = 0
    for leaf in range(NUM_LEAVES):
        for k in range(VM_K):
            coeffs = data[leaf, k, :]
            diffs = np.abs(np.diff(coeffs.astype(np.int32)))
            max_jump = max(max_jump, np.max(diffs))

    return rel, max_jump

print(f"\nH5 (2200 Hz) at 440 Hz base:")
print(f"Phase bins per block: {(BLOCK_SIZE * phase_inc * h) % (2**32) / (2**32) * 64:.1f} (at P=64)")
print()
print(f"{'P':>6} {'50Hz (dB)':>12} {'Max Jump':>12}")
print("-" * 35)

for P in [32, 64, 128, 256, 512, 1024]:
    rel, max_jump = test_resolution(P)
    bins_per_block = ((BLOCK_SIZE * phase_inc * h) % (2**32)) / (2**32) * P
    print(f"{P:6d} {rel:12.1f} {max_jump:12d}  ({bins_per_block:.1f} bins/blk)")
