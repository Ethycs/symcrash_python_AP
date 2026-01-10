"""Debug block-boundary artifacts (20ms / 50Hz "mini wavelet" artifact)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from symcrash_python_ap.vm.harmonic_bank import (
    generate_basis_frames, mode_b_generate_coeffs, ModeBState,
    clear_basis_atlas, BLOCK_SIZE, SAMPLE_RATE, VM_K, K_STRIDE, NUM_LEAVES
)
from symcrash_python_ap.vm.tables import COEFF_LEN
from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_block
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc

def render_blocks(num_blocks=20, freq_hz=440.0, harmonics=None):
    """Render multiple blocks and return audio + coefficients for analysis."""
    note = 69  # A4 = 440 Hz
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

    clear_basis_atlas()
    basis = generate_basis_frames(anchor_note=note, H_max=8, k_stride=K_STRIDE)

    mb_state = ModeBState()
    mb_state.reset(note=note, phase_inc=phase_inc)

    if harmonics:
        mb_state.weights.set_custom(harmonics)
        mb_state.H_active = len(harmonics)

    all_samples = []
    all_coeffs = []
    block_boundaries = []

    for block_idx in range(num_blocks):
        carrier = np.zeros(COEFF_LEN, dtype=np.int16)
        mode_b_generate_coeffs(mb_state, basis, carrier)

        coeff_2d = carrier.reshape(NUM_LEAVES, VM_K)
        block_samples = idwpt_polyphase_block(coeff_2d, headroom_bits=1)

        block_boundaries.append(len(all_samples))
        all_samples.extend(block_samples.tolist())
        all_coeffs.append(carrier.copy())
        mb_state.advance_block()

    return np.array(all_samples, dtype=np.float64), np.array(all_coeffs), block_boundaries


def analyze_block_boundaries():
    """Analyze what happens at block boundaries."""
    print("=" * 60)
    print("BLOCK BOUNDARY ANALYSIS")
    print("=" * 60)

    # Render pure sine
    samples, coeffs, boundaries = render_blocks(num_blocks=20, harmonics=[32767])

    # Compute block-to-block differences at boundaries
    print(f"\nBlock size: {BLOCK_SIZE}, K={VM_K}, K_STRIDE={K_STRIDE}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Block rate: {SAMPLE_RATE / BLOCK_SIZE:.1f} Hz ({BLOCK_SIZE / SAMPLE_RATE * 1000:.2f} ms)")

    # Check samples at boundaries
    print("\n--- Boundary Sample Analysis ---")
    for i in range(min(5, len(boundaries) - 1)):
        b_start = boundaries[i]
        b_end = boundaries[i + 1]

        # Last sample of block i
        last_sample = samples[b_end - 1]
        # First sample of block i+1
        first_sample = samples[b_end]

        # Difference (discontinuity)
        disc = first_sample - last_sample

        # Expected difference for continuous sine
        # sin(phi + delta) - sin(phi) ≈ delta * cos(phi) for small delta
        # With 440 Hz, phase advances by 2*pi*440/48000 ≈ 0.0576 rad/sample
        phase_per_sample = 2 * np.pi * 440 / SAMPLE_RATE

        print(f"Block {i}->{i+1}: last={last_sample:.0f}, first={first_sample:.0f}, "
              f"jump={disc:.0f}")

    # Look for periodicity in discontinuities
    print("\n--- Periodicity Analysis ---")
    discontinuities = []
    for i in range(len(boundaries) - 1):
        b_end = boundaries[i + 1]
        last = samples[b_end - 1]
        first = samples[b_end]
        discontinuities.append(first - last)

    disc_arr = np.array(discontinuities)
    print(f"Mean discontinuity: {np.mean(disc_arr):.2f}")
    print(f"Std discontinuity: {np.std(disc_arr):.2f}")
    print(f"Max discontinuity: {np.max(np.abs(disc_arr)):.2f}")

    # Autocorrelation of discontinuities
    if len(disc_arr) > 10:
        from numpy.fft import rfft, irfft
        fft_disc = rfft(disc_arr - np.mean(disc_arr))
        autocorr = irfft(fft_disc * np.conj(fft_disc))
        autocorr = autocorr / autocorr[0]  # Normalize

        print(f"\nAutocorrelation at lag 1: {autocorr[1]:.4f}")
        print(f"Autocorrelation at lag 2: {autocorr[2]:.4f}")
        print(f"Autocorrelation at lag 3: {autocorr[3]:.4f}")

    return samples, coeffs, boundaries, disc_arr


def plot_waveform_detail(samples, boundaries):
    """Plot waveform showing block boundaries (skipped - no matplotlib)."""
    print("\n[Skipping plot - matplotlib not available]")


def analyze_coefficient_continuity(coeffs):
    """Analyze coefficient continuity across blocks."""
    print("\n--- Coefficient Continuity Analysis ---")

    # Reshape to [blocks, leaves, K]
    coeffs_3d = coeffs.reshape(-1, NUM_LEAVES, VM_K)

    # Check how coefficients change between blocks
    for leaf in [0, 1, 2, 31, 32, 63]:  # Sample some leaves
        coeff_series = coeffs_3d[:, leaf, :]  # [blocks, K]

        # Average coefficient for this leaf across K slots
        avg_coeff = np.mean(coeff_series, axis=1)

        # Discontinuity between blocks
        disc = np.diff(avg_coeff)

        print(f"Leaf {leaf:2d}: mean={np.mean(avg_coeff):8.1f}, "
              f"std={np.std(avg_coeff):8.1f}, "
              f"max_jump={np.max(np.abs(disc)):8.1f}")


def check_phase_continuity():
    """Verify that phase advances correctly across blocks."""
    print("\n--- Phase Continuity Check ---")

    freq_hz = 440.0
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

    # Expected phase advance per block
    block_phase_advance = (BLOCK_SIZE * phase_inc) % (2**32)

    print(f"Phase increment per sample: {phase_inc}")
    print(f"Phase advance per block: {block_phase_advance}")
    print(f"Block phase advance (radians): {block_phase_advance / (2**32) * 2 * np.pi:.6f}")
    print(f"Expected cycles per block: {BLOCK_SIZE * freq_hz / SAMPLE_RATE:.4f}")

    # Check if K_STRIDE phase offsets align correctly
    print(f"\nK_STRIDE = {K_STRIDE}")
    print("Phase offset per K slot (radians):")
    for k in range(VM_K):
        offset_phase = (k * K_STRIDE * phase_inc) % (2**32)
        offset_rad = offset_phase / (2**32) * 2 * np.pi
        print(f"  k={k}: {offset_rad:.6f} rad = {offset_rad/np.pi:.6f}*pi")


def analyze_50hz_origin():
    """Investigate what could cause 50 Hz (20ms period)."""
    print("\n--- 50 Hz Artifact Investigation ---")

    freq_hz = 440.0
    block_rate = SAMPLE_RATE / BLOCK_SIZE  # 750 Hz

    print(f"Block rate: {block_rate} Hz")
    print(f"50 Hz = 1 event per {block_rate/50:.1f} blocks")
    print(f"Period of 50 Hz at 440 Hz signal: {440/50:.1f} fundamental cycles")

    # Check beat frequency between 440 Hz and block rate
    # 440 mod 750 = 440 (doesn't fit evenly)
    # Number of blocks per 440 Hz cycle: 750/440 = 1.704...
    cycles_per_block = freq_hz / block_rate
    print(f"\nFundamental cycles per block: {cycles_per_block:.4f}")
    print(f"Fractional part: {cycles_per_block % 1:.4f}")

    # The fractional part means phase doesn't align at block boundaries
    # This creates a beat pattern at:
    # beat_freq = block_rate * (cycles_per_block % 1)
    beat_freq = block_rate * (cycles_per_block % 1)
    print(f"Phase misalignment beat frequency: {beat_freq:.1f} Hz")

    # Also check relationship to wavelet tree structure
    print(f"\nWavelet tree depth: 6 (64 leaves)")
    print(f"Samples per lowest-frequency leaf: {BLOCK_SIZE}")
    print(f"Lowest leaf center freq: {SAMPLE_RATE / BLOCK_SIZE / 2:.1f} Hz")


if __name__ == "__main__":
    samples, coeffs, boundaries, disc_arr = analyze_block_boundaries()
    plot_waveform_detail(samples, boundaries)
    analyze_coefficient_continuity(coeffs)
    check_phase_continuity()
    analyze_50hz_origin()
