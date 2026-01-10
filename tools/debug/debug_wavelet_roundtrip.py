"""Debug wavelet FWPT->IDWPT roundtrip artifacts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq

from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit

BLOCK_SIZE = 64
SAMPLE_RATE = 48000


def test_roundtrip_perfect():
    """Test FWPT->IDWPT roundtrip with same headroom."""
    print("=== Perfect Roundtrip Test ===")

    # Generate a simple sine block
    freq_hz = 440.0
    t = np.arange(BLOCK_SIZE) / SAMPLE_RATE
    sine_block = (16384 * np.sin(2 * np.pi * freq_hz * t)).astype(np.int16)

    # Forward transform
    coeffs = fwpt_jit(sine_block.tolist(), headroom_bits=1)

    # Inverse transform
    reconstructed = idwpt_jit(coeffs, headroom_bits=1)

    # Compute error
    error = np.array(sine_block) - np.array(reconstructed)

    print(f"Input range: {sine_block.min()} to {sine_block.max()}")
    print(f"Reconstructed range: {min(reconstructed)} to {max(reconstructed)}")
    print(f"Max error: {np.max(np.abs(error))}")
    print(f"RMS error: {np.sqrt(np.mean(error.astype(np.float64)**2)):.2f}")


def test_multi_block_sine():
    """Test multiple blocks of continuous sine."""
    print("\n=== Multi-Block Continuous Sine Test ===")

    freq_hz = 440.0
    num_blocks = 100
    num_samples = num_blocks * BLOCK_SIZE

    # Generate continuous sine
    t = np.arange(num_samples) / SAMPLE_RATE
    direct_sine = (16384 * np.sin(2 * np.pi * freq_hz * t)).astype(np.int16)

    # Process block by block
    reconstructed = []
    for block_idx in range(num_blocks):
        start = block_idx * BLOCK_SIZE
        end = start + BLOCK_SIZE
        block = direct_sine[start:end]

        # FWPT -> IDWPT
        coeffs = fwpt_jit(block.tolist(), headroom_bits=1)
        recon_block = idwpt_jit(coeffs, headroom_bits=1)
        reconstructed.extend(recon_block)

    reconstructed = np.array(reconstructed, dtype=np.int16)

    # Compute difference
    diff = direct_sine.astype(np.float64) - reconstructed.astype(np.float64)

    print(f"RMS difference: {np.sqrt(np.mean(diff**2)):.2f}")
    print(f"Max difference: {np.max(np.abs(diff)):.0f}")

    # Spectrum of difference
    diff_spectrum = np.abs(rfft(diff))
    diff_freqs = rfftfreq(len(diff), 1/SAMPLE_RATE)

    # Check for periodicity in errors
    print("\nDifference spectrum peaks (50-250 Hz range):")
    for freq_target in [50, 100, 150, 200, 250]:
        idx = int(freq_target * len(diff) / SAMPLE_RATE)
        if idx < len(diff_spectrum):
            amp = diff_spectrum[idx]
            print(f"  {freq_target} Hz: amplitude = {amp:.0f}")

    return reconstructed, diff


def test_no_transform():
    """Test what happens with NO wavelet transform (direct output)."""
    print("\n=== No Transform Reference Test ===")

    freq_hz = 440.0
    num_blocks = 100
    num_samples = num_blocks * BLOCK_SIZE

    # Generate continuous sine
    t = np.arange(num_samples) / SAMPLE_RATE
    direct_sine = (16384 * np.sin(2 * np.pi * freq_hz * t)).astype(np.int16)

    # Analyze the input sine spectrum
    spectrum = np.abs(rfft(direct_sine))
    freqs = rfftfreq(len(direct_sine), 1/SAMPLE_RATE)

    idx_440 = int(440 * len(direct_sine) / SAMPLE_RATE)
    idx_50 = int(50 * len(direct_sine) / SAMPLE_RATE)

    amp_440 = spectrum[idx_440]
    amp_50 = spectrum[idx_50]

    print(f"Pure sine 440 Hz amplitude: {amp_440:.0f}")
    print(f"Pure sine 50 Hz amplitude: {amp_50:.0f}")
    print(f"50 Hz relative to 440 Hz: {20*np.log10(amp_50/amp_440):.1f} dB")


def test_block_boundary_continuity():
    """Check if block boundaries cause discontinuities after roundtrip."""
    print("\n=== Block Boundary Continuity Test ===")

    freq_hz = 440.0
    num_blocks = 20

    # Generate continuous sine
    num_samples = num_blocks * BLOCK_SIZE
    t = np.arange(num_samples) / SAMPLE_RATE
    direct_sine = (16384 * np.sin(2 * np.pi * freq_hz * t)).astype(np.int16)

    reconstructed = []
    for block_idx in range(num_blocks):
        start = block_idx * BLOCK_SIZE
        end = start + BLOCK_SIZE
        block = direct_sine[start:end]

        coeffs = fwpt_jit(block.tolist(), headroom_bits=1)
        recon_block = idwpt_jit(coeffs, headroom_bits=1)
        reconstructed.extend(recon_block)

    reconstructed = np.array(reconstructed, dtype=np.int16)

    # Check block boundary jumps
    print("Block boundary discontinuities (recon - input):")
    for i in range(min(5, num_blocks - 1)):
        boundary = (i + 1) * BLOCK_SIZE

        # Last sample of block i (reconstructed vs input)
        last_recon = reconstructed[boundary - 1]
        last_input = direct_sine[boundary - 1]
        last_error = last_recon - last_input

        # First sample of block i+1 (reconstructed vs input)
        first_recon = reconstructed[boundary]
        first_input = direct_sine[boundary]
        first_error = first_recon - first_input

        print(f"  Block {i}->{i+1}: last_err={last_error:+4d}, first_err={first_error:+4d}, "
              f"jump={first_error-last_error:+4d}")


def test_accumulating_phase_error():
    """Check if phase errors accumulate across blocks in Mode B."""
    print("\n=== Accumulating Phase Error Test ===")
    print("(This tests if the basis frame approach has inherent drift)")

    from symcrash_python_ap.vm.harmonic_bank import (
        generate_basis_frames, mode_b_generate_coeffs, ModeBState,
        clear_basis_atlas, VM_K, K_STRIDE, NUM_LEAVES
    )
    from symcrash_python_ap.vm.tables import COEFF_LEN
    from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_block
    from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc

    freq_hz = 440.0
    note = 69
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

    clear_basis_atlas()
    basis = generate_basis_frames(anchor_note=note, H_max=8, k_stride=K_STRIDE)

    mb_state = ModeBState()
    mb_state.reset(note=note, phase_inc=phase_inc)
    mb_state.weights.set_sine()
    mb_state.H_active = 1

    num_blocks = 200
    all_samples = []

    for _ in range(num_blocks):
        carrier = np.zeros(COEFF_LEN, dtype=np.int16)
        mode_b_generate_coeffs(mb_state, basis, carrier)

        coeff_2d = carrier.reshape(NUM_LEAVES, VM_K)
        block_samples = idwpt_polyphase_block(coeff_2d, headroom_bits=1)
        all_samples.extend(block_samples.tolist())
        mb_state.advance_block()

    all_samples = np.array(all_samples, dtype=np.float64)

    # Generate reference sine
    num_samples = len(all_samples)
    t = np.arange(num_samples) / SAMPLE_RATE
    ref_sine = 32767 * np.sin(2 * np.pi * freq_hz * t)  # Mode B produces full-scale

    # Normalize Mode B output to match reference
    mode_b_rms = np.sqrt(np.mean(all_samples**2))
    ref_rms = np.sqrt(np.mean(ref_sine**2))
    scale = ref_rms / mode_b_rms if mode_b_rms > 0 else 1.0

    # Check phase alignment at different points
    print("Phase alignment check (cross-correlation peaks):")
    for check_block in [10, 50, 100, 150, 199]:
        start = check_block * BLOCK_SIZE
        end = min(start + BLOCK_SIZE * 2, len(all_samples))
        if end > start:
            chunk = all_samples[start:end] * scale
            ref_chunk = ref_sine[start:end]

            # Cross-correlation to find phase offset
            from numpy.fft import fft, ifft
            corr = ifft(fft(chunk) * np.conj(fft(ref_chunk)))
            peak_idx = np.argmax(np.abs(corr))
            if peak_idx > len(corr) // 2:
                peak_idx -= len(corr)

            phase_offset_samples = peak_idx
            phase_offset_deg = 360 * peak_idx * freq_hz / SAMPLE_RATE

            print(f"  Block {check_block}: peak_idx={peak_idx}, "
                  f"phase_offset={phase_offset_deg:.1f} deg")


if __name__ == "__main__":
    test_roundtrip_perfect()
    test_multi_block_sine()
    test_no_transform()
    test_block_boundary_continuity()
    test_accumulating_phase_error()
