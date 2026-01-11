"""Test coefficient smoothing to reduce block-boundary artifacts."""
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


def render_with_coeff_smoothing(harmonics, num_blocks=2000, smooth_factor=0.0):
    """Render with optional coefficient smoothing between blocks.

    smooth_factor: 0.0 = no smoothing, 1.0 = full low-pass
    """
    note = 69  # A4 = 440 Hz
    freq_hz = 440.0
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

    clear_basis_atlas()
    basis = generate_basis_frames(anchor_note=note, H_max=8, k_stride=K_STRIDE)

    mb_state = ModeBState()
    mb_state.reset(note=note, phase_inc=phase_inc)
    mb_state.weights.set_custom(harmonics)
    mb_state.H_active = len(harmonics)

    all_samples = []
    prev_coeffs = None

    for _ in range(num_blocks):
        carrier = np.zeros(COEFF_LEN, dtype=np.int16)
        mode_b_generate_coeffs(mb_state, basis, carrier)

        # Apply coefficient smoothing (low-pass filter between blocks)
        if smooth_factor > 0 and prev_coeffs is not None:
            smoothed = np.zeros(COEFF_LEN, dtype=np.float64)
            for i in range(COEFF_LEN):
                smoothed[i] = smooth_factor * prev_coeffs[i] + (1 - smooth_factor) * carrier[i]
            carrier = np.clip(smoothed, -32768, 32767).astype(np.int16)

        prev_coeffs = carrier.copy().astype(np.float64)

        coeff_2d = carrier.reshape(NUM_LEAVES, VM_K)
        block_samples = idwpt_polyphase_block(coeff_2d, headroom_bits=1)
        all_samples.extend(block_samples.tolist())
        mb_state.advance_block()

    return np.array(all_samples, dtype=np.float64)


def analyze_artifact(samples, name):
    """Analyze artifact level."""
    n = len(samples)
    hann = np.hanning(n)
    windowed = samples * hann
    window_scale = 2.0 / np.mean(hann)

    spectrum = np.abs(rfft(windowed)) * window_scale
    freqs = rfftfreq(n, 1/SAMPLE_RATE)

    idx_50 = int(50 * n / SAMPLE_RATE)
    idx_440 = int(440 * n / SAMPLE_RATE)

    amp_50 = spectrum[idx_50]
    amp_440 = spectrum[idx_440]

    rel_db = 20 * np.log10(amp_50 / amp_440) if amp_440 > 0 else -100

    print(f"{name}: 50 Hz = {rel_db:.1f} dB")
    return rel_db


print("=" * 60)
print("COEFFICIENT SMOOTHING TEST")
print("=" * 60)

harmonics = [32767, 32767, 32767, 32767, 32767]  # H1-H5 equal

print("\n--- Testing different smoothing factors ---")
for smooth in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
    samples = render_with_coeff_smoothing(harmonics, smooth_factor=smooth)
    analyze_artifact(samples, f"smooth={smooth:.1f}")

# Test with violin profile
print("\n--- Violin Profile with Smoothing ---")
violin_harmonics = [30890, 32767, 6926, 23969, 3673, 6138, 1164, 1228]

for smooth in [0.0, 0.2, 0.4]:
    samples = render_with_coeff_smoothing(violin_harmonics, smooth_factor=smooth)
    analyze_artifact(samples, f"Violin smooth={smooth:.1f}")
