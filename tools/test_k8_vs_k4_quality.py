"""
Test K=4/K=8 polyphase quality.

Both K=4 and K=8 use the same interleaved reconstruction strategy:
  - K_STRIDE = 1 (1-sample phase offset per slot)
  - out[k::K] = idwpt(slot_k)[0::K]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq

from symcrash_python_ap.vm.harmonic_bank import (
    generate_basis_frames, mode_b_generate_coeffs, ModeBState,
    clear_basis_atlas, BLOCK_SIZE, SAMPLE_RATE, VM_K, K_STRIDE, NUM_LEAVES
)
from symcrash_python_ap.vm.tables import COEFF_LEN
from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_block
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc


def render_current_config(num_blocks=100):
    """Render using current K configuration (K=4 or K=8)."""
    note = 69
    freq_hz = 440.0
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

    print(f"Configuration: VM_K={VM_K}, K_STRIDE={K_STRIDE}, COEFF_LEN={COEFF_LEN}")

    clear_basis_atlas()
    basis = generate_basis_frames(anchor_note=note, H_max=8, k_stride=K_STRIDE)

    # Verify basis shapes
    print(f"Basis shapes: DC_frame={basis.DC_frame.shape}, Ucos_h={basis.Ucos_h.shape}")

    mb_state = ModeBState()
    mb_state.reset(note=note, phase_inc=phase_inc)
    mb_state.weights.set_sine()
    mb_state.H_active = 1

    all_samples = []

    for _ in range(num_blocks):
        carrier = np.zeros(COEFF_LEN, dtype=np.int16)
        mode_b_generate_coeffs(mb_state, basis, carrier)

        coeff_2d = carrier.reshape(NUM_LEAVES, VM_K)
        block_samples = idwpt_polyphase_block(coeff_2d, headroom_bits=1)
        all_samples.extend(block_samples.tolist())
        mb_state.advance_block()

    return np.array(all_samples, dtype=np.float64)


def analyze(samples, name):
    """Analyze spectrum for spurs."""
    print(f"\n=== {name} ===")

    spectrum = np.abs(rfft(samples))
    freqs = rfftfreq(len(samples), 1/SAMPLE_RATE)

    # Find fundamental
    idx_440 = int(440 * len(samples) / SAMPLE_RATE)
    amp_440 = spectrum[idx_440]

    # Find strongest spurs (non-harmonics)
    spurs = []
    for i in range(1, len(spectrum)-1):
        if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
            freq = freqs[i]
            ratio = freq / 440.0
            nearest_h = round(ratio)
            if abs(ratio - nearest_h) > 0.05 or nearest_h == 0:
                spurs.append((freq, spectrum[i]))

    spurs.sort(key=lambda x: -x[1])

    print(f"Fundamental (440 Hz): amplitude = {amp_440:.0f}")
    print(f"Top 5 spurs:")
    for freq, amp in spurs[:5]:
        rel_db = 20 * np.log10(amp / amp_440) if amp_440 > 0 else -100
        print(f"  {freq:.1f} Hz: {rel_db:.1f} dB")

    # Cycle correlation
    period = int(SAMPLE_RATE / 440)
    correlations = []
    for i in range(10):
        c1 = samples[i*period:(i+1)*period]
        c2 = samples[(i+1)*period:(i+2)*period]
        if len(c1) == len(c2) == period:
            corr = np.corrcoef(c1, c2)[0, 1]
            correlations.append(corr)

    if correlations:
        print(f"Cycle correlation: avg={np.mean(correlations):.4f}, min={np.min(correlations):.4f}")


def main():
    print("=" * 60)
    print(f"K={VM_K} Polyphase Quality Test")
    print("=" * 60)

    samples = render_current_config()

    analyze(samples, f"K={VM_K} interleaved (k_stride={K_STRIDE})")


if __name__ == "__main__":
    main()
