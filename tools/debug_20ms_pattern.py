"""Debug the 20ms (50 Hz) periodic artifact pattern."""
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
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc


def render_comparison(num_blocks=100):
    """Compare polyphase interleave vs single IDWPT reconstruction."""
    note = 69  # A4 = 440 Hz
    freq_hz = 440.0
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

    clear_basis_atlas()
    basis = generate_basis_frames(anchor_note=note, H_max=8, k_stride=K_STRIDE)

    mb_state = ModeBState()
    mb_state.reset(note=note, phase_inc=phase_inc)
    mb_state.weights.set_sine()
    mb_state.H_active = 1

    polyphase_samples = []
    single_slot_samples = []

    for _ in range(num_blocks):
        carrier = np.zeros(COEFF_LEN, dtype=np.int16)
        mode_b_generate_coeffs(mb_state, basis, carrier)

        coeff_2d = carrier.reshape(NUM_LEAVES, VM_K)

        # Method 1: Polyphase interleave (current method)
        block_poly = idwpt_polyphase_block(coeff_2d, headroom_bits=1)
        polyphase_samples.extend(block_poly.tolist())

        # Method 2: Single slot reconstruction (slot 0 only)
        coeffs_0 = coeff_2d[:, 0].astype(np.int16).tolist()
        block_single = np.array(idwpt_jit(coeffs_0, headroom_bits=1), dtype=np.int16)
        single_slot_samples.extend(block_single.tolist())

        mb_state.advance_block()

    return np.array(polyphase_samples), np.array(single_slot_samples)


def analyze_spectrum(samples, name):
    """Analyze spectrum for artifacts."""
    spectrum = np.abs(rfft(samples))
    freqs = rfftfreq(len(samples), 1/SAMPLE_RATE)

    # Find 50 Hz and 440 Hz components
    idx_50 = int(50 * len(samples) / SAMPLE_RATE)
    idx_440 = int(440 * len(samples) / SAMPLE_RATE)

    amp_50 = spectrum[idx_50]
    amp_440 = spectrum[idx_440]

    rel_db = 20 * np.log10(amp_50 / amp_440) if amp_440 > 0 else -100

    print(f"{name}:")
    print(f"  50 Hz component: {rel_db:.1f} dB relative to 440 Hz")

    # Find other significant spurs below 100 Hz
    low_freq_mask = (freqs > 10) & (freqs < 100)
    low_freqs = freqs[low_freq_mask]
    low_spectrum = spectrum[low_freq_mask]

    if len(low_spectrum) > 0:
        max_idx = np.argmax(low_spectrum)
        max_freq = low_freqs[max_idx]
        max_amp = low_spectrum[max_idx]
        max_db = 20 * np.log10(max_amp / amp_440) if amp_440 > 0 else -100
        print(f"  Strongest low-freq component: {max_freq:.1f} Hz at {max_db:.1f} dB")


def check_interleave_math():
    """Verify the polyphase interleave math is correct."""
    print("\n--- Interleave Math Check ---")
    print(f"VM_K = {VM_K}")
    print(f"BLOCK_SIZE = {BLOCK_SIZE}")
    print(f"K_STRIDE = {K_STRIDE}")
    print()

    # The interleave pattern is: out[k::K] = audio_k[0::K]
    # For K=4:
    #   out[0, 4, 8, ..., 60] = audio_0[0, 4, 8, ..., 60]
    #   out[1, 5, 9, ..., 61] = audio_1[0, 4, 8, ..., 60]
    #   out[2, 6, 10, ..., 62] = audio_2[0, 4, 8, ..., 60]
    #   out[3, 7, 11, ..., 63] = audio_3[0, 4, 8, ..., 60]

    print("Interleave pattern:")
    for k in range(VM_K):
        out_indices = list(range(k, BLOCK_SIZE, VM_K))
        print(f"  Slot {k} -> output indices: {out_indices[:8]}... (every {VM_K}th sample)")

    # The basis for slot k has phase offset of k * K_STRIDE * phase_inc
    # For K_STRIDE=1, this is 1 sample of phase offset per slot
    print(f"\nBasis phase offset per slot (K_STRIDE={K_STRIDE}):")
    freq_hz = 440.0
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)
    for k in range(VM_K):
        offset_samples = k * K_STRIDE
        offset_rad = offset_samples * 2 * np.pi * freq_hz / SAMPLE_RATE
        print(f"  Slot {k}: {offset_samples} sample(s) = {offset_rad:.4f} rad")

    print("\nFor correct reconstruction:")
    print("  - Slot k's basis is for time offset k samples")
    print("  - Output sample n uses slot (n % K)")
    print("  - So sample 0 uses slot 0's basis (offset 0)")
    print("  - Sample 1 uses slot 1's basis (offset 1 sample)")
    print("  - etc.")

    # This seems correct - but let's verify the idwpt output mapping
    print("\nIDWPT output sample mapping:")
    print("  idwpt_jit returns 64 samples from 64 leaf coefficients")
    print("  For polyphase: out[k::K] = audio_k[0::K]")
    print("  This means:")
    for k in range(VM_K):
        src_indices = list(range(0, BLOCK_SIZE, VM_K))
        dst_indices = list(range(k, BLOCK_SIZE, VM_K))
        print(f"    audio_{k}[{src_indices[:4]}...] -> out[{dst_indices[:4]}...]")


def test_direct_vs_polyphase():
    """Compare direct sine generation vs Mode B polyphase."""
    print("\n--- Direct vs Polyphase Comparison ---")

    freq_hz = 440.0
    num_samples = 6400  # 100 blocks

    # Generate direct reference sine
    t = np.arange(num_samples) / SAMPLE_RATE
    direct_sine = (32767 * np.sin(2 * np.pi * freq_hz * t)).astype(np.int16)

    # Get polyphase output
    poly_samples, _ = render_comparison(num_blocks=100)

    # Compute difference
    min_len = min(len(direct_sine), len(poly_samples))
    diff = direct_sine[:min_len].astype(np.float64) - poly_samples[:min_len].astype(np.float64)

    print(f"RMS difference: {np.sqrt(np.mean(diff**2)):.1f}")
    print(f"Max difference: {np.max(np.abs(diff)):.1f}")
    print(f"Mean difference: {np.mean(diff):.1f}")

    # Check if difference has periodicity
    diff_spectrum = np.abs(rfft(diff))
    diff_freqs = rfftfreq(len(diff), 1/SAMPLE_RATE)

    # Find peaks in difference spectrum
    print("\nDifference spectrum peaks:")
    for freq_target in [50, 100, 150, 200, 250]:
        idx = int(freq_target * len(diff) / SAMPLE_RATE)
        if idx < len(diff_spectrum):
            amp = diff_spectrum[idx]
            print(f"  {freq_target} Hz: amplitude = {amp:.0f}")


if __name__ == "__main__":
    print("=" * 60)
    print("20ms (50 Hz) ARTIFACT ANALYSIS")
    print("=" * 60)

    check_interleave_math()

    print("\n" + "=" * 60)
    print("SPECTRUM COMPARISON")
    print("=" * 60)

    poly_samples, single_samples = render_comparison(num_blocks=100)

    analyze_spectrum(poly_samples, "Polyphase interleave")
    analyze_spectrum(single_samples, "Single slot (slot 0)")

    test_direct_vs_polyphase()
