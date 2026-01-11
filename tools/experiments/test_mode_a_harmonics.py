"""Test Mode A with each harmonic individually."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq

from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sine_q15
from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_block
from symcrash_python_ap.vm.coeff_lut_streaming import StreamingCoeffLUT, VM_K, BLOCK_SIZE, NUM_LEAVES

SAMPLE_RATE = 48000
freq_hz = 440.0
phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

print("=" * 70)
print("MODE A - INDIVIDUAL HARMONIC TEST")
print(f"VM_K={VM_K}")
print("=" * 70)

def test_harmonic(h):
    def gen_h(phase):
        h_phase = (phase * h) & 0xFFFFFFFF
        return sine_q15(h_phase)

    lut = StreamingCoeffLUT.generate_for_frequency(
        wave_id=0,
        name="h" + str(h),
        target_freq_hz=freq_hz,
        generate_sample_fn=gen_h,
        K=VM_K,
        P=64,
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

    idx_h = int(440 * h * n / SAMPLE_RATE)
    amp_h = spectrum[idx_h] if idx_h < len(spectrum) else 1
    idx_50 = int(50 * n / SAMPLE_RATE)
    amp_50 = spectrum[idx_50]

    rel = 20 * np.log10(amp_50 / amp_h) if amp_h > 0 else -100
    cycles_per_block = (440 * h) * BLOCK_SIZE / SAMPLE_RATE
    print("H%d (%4d Hz): 50 Hz = %6.1f dB  (%.2f cycles/block)" % (h, 440*h, rel, cycles_per_block))

print()
for h in range(1, 9):
    test_harmonic(h)
