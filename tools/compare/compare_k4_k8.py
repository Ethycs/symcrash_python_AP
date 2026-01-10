"""Compare K=4 vs K=8 to find the source of the artifact."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq

from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sin_cos_q15, PHASE_MASK
from symcrash_python_ap.fixed import RSHIFT_RNE, SAT16

BLOCK_SIZE = 64
SAMPLE_RATE = 48000
NUM_LEAVES = 64

print("=" * 70)
print("K=4 vs K=8 COMPARISON")
print("=" * 70)


def generate_basis_frames_local(anchor_note, H_max, K, k_stride):
    """Generate basis frames with configurable K and k_stride."""
    freq_hz = 440.0 * (2.0 ** ((anchor_note - 69) / 12.0))
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

    DC_frame = np.zeros((NUM_LEAVES, K), dtype=np.int16)
    Ucos_h = np.zeros((H_max, NUM_LEAVES, K), dtype=np.int16)
    Usin_h = np.zeros((H_max, NUM_LEAVES, K), dtype=np.int16)

    for k in range(K):
        # DC basis
        dc_block = [32767] * BLOCK_SIZE
        dc_coeffs = fwpt_jit(dc_block, headroom_bits=1)
        DC_frame[:, k] = dc_coeffs

        # Phase offset for this k-slot
        k_phase_offset = (k * k_stride * phase_inc) & PHASE_MASK

        # Harmonic bases
        for h in range(1, H_max + 1):
            cos_block = []
            sin_block = []

            phase = k_phase_offset
            for n in range(BLOCK_SIZE):
                h_phase = (phase * h) & PHASE_MASK
                sin_val, cos_val = sin_cos_q15(h_phase)
                cos_block.append(cos_val)
                sin_block.append(sin_val)
                phase = (phase + phase_inc) & PHASE_MASK

            cos_coeffs = fwpt_jit(cos_block, headroom_bits=1)
            sin_coeffs = fwpt_jit(sin_block, headroom_bits=1)

            Ucos_h[h - 1, :, k] = cos_coeffs
            Usin_h[h - 1, :, k] = sin_coeffs

    return DC_frame, Ucos_h, Usin_h, phase_inc


def generate_coeffs_mode_b(base_phase, DC_frame, Usin_h, Ucos_h, K, h, a_h=32767):
    """Generate coefficients for harmonic h using Mode B."""
    coeffs = np.zeros((NUM_LEAVES, K), dtype=np.int16)

    for k in range(K):
        phi = base_phase
        h_phase = (phi * h) & PHASE_MASK
        sin_phi, cos_phi = sin_cos_q15(h_phase)

        w_sin = RSHIFT_RNE(a_h * cos_phi, 15)
        w_cos = RSHIFT_RNE(a_h * sin_phi, 15)

        for leaf in range(NUM_LEAVES):
            sin_basis = int(Usin_h[h-1, leaf, k])
            cos_basis = int(Ucos_h[h-1, leaf, k])
            sin_contrib = RSHIFT_RNE(w_sin * sin_basis, 15)
            cos_contrib = RSHIFT_RNE(w_cos * cos_basis, 15)
            coeffs[leaf, k] = SAT16(sin_contrib + cos_contrib)

    return coeffs


def idwpt_polyphase(coeff_2d, headroom_bits=1):
    """Polyphase interleave reconstruction."""
    K = coeff_2d.shape[1]
    out = np.zeros(64, dtype=np.int16)

    for k in range(K):
        coeffs_k = coeff_2d[:, k].astype(np.int16).tolist()
        audio_k = np.array(idwpt_jit(coeffs_k, headroom_bits=headroom_bits), dtype=np.int16)
        out[k::K] = audio_k[0::K]

    return out


def render_and_analyze(K, k_stride, h, num_blocks=100):
    """Render harmonic h with given K and k_stride, analyze 50 Hz artifact."""
    note = 69
    DC_frame, Ucos_h, Usin_h, phase_inc = generate_basis_frames_local(note, 8, K, k_stride)

    phase = 0
    all_samples = []

    for _ in range(num_blocks):
        coeffs = generate_coeffs_mode_b(phase, DC_frame, Usin_h, Ucos_h, K, h)
        out = idwpt_polyphase(coeffs)
        all_samples.extend(out.tolist())
        phase = (phase + BLOCK_SIZE * phase_inc) & PHASE_MASK

    samples = np.array(all_samples, dtype=np.float64)

    # Analyze 50 Hz
    n = len(samples)
    hann = np.hanning(n)
    windowed = samples * hann
    spectrum = np.abs(rfft(windowed)) * (2.0 / np.mean(hann))
    freqs = rfftfreq(n, 1/SAMPLE_RATE)

    idx_50 = int(50 * n / SAMPLE_RATE)
    idx_h = int(440 * h * n / SAMPLE_RATE)

    amp_50 = spectrum[idx_50]
    amp_h = spectrum[idx_h] if idx_h < len(spectrum) else 1

    rel_db = 20 * np.log10(amp_50 / amp_h) if amp_h > 0 else -100
    return rel_db


# Test K=4 with k_stride=1
print("\n--- K=4, k_stride=1 (current config) ---")
for h in range(1, 9):
    db = render_and_analyze(K=4, k_stride=1, h=h)
    print(f"H{h}: 50 Hz = {db:.1f} dB")

# Test K=8 with k_stride=1
print("\n--- K=8, k_stride=1 ---")
for h in range(1, 9):
    db = render_and_analyze(K=8, k_stride=1, h=h)
    print(f"H{h}: 50 Hz = {db:.1f} dB")

# Test K=4 with k_stride=16 (consecutive blocks)
print("\n--- K=4, k_stride=16 (consecutive blocks) ---")
for h in range(1, 9):
    db = render_and_analyze(K=4, k_stride=16, h=h)
    print(f"H{h}: 50 Hz = {db:.1f} dB")

# Test K=8 with k_stride=8 (original K=8 config?)
print("\n--- K=8, k_stride=8 ---")
for h in range(1, 9):
    db = render_and_analyze(K=8, k_stride=8, h=h)
    print(f"H{h}: 50 Hz = {db:.1f} dB")

print("\n" + "=" * 70)
print("The question: What K/k_stride combination eliminates the artifact?")
print("=" * 70)
