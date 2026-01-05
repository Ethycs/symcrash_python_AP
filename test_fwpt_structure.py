"""
Investigate FWPT/IDWPT structure to understand the correct streaming architecture.
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.wavelet.sine_core import sine_q15

def midi_to_phase_inc(note: int, sample_rate: int = 48000) -> int:
    freq_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))
    return int((freq_hz / sample_rate) * (1 << 32))


def generate_sine_block(phase_start: int, phase_inc: int, num_samples: int = 64) -> np.ndarray:
    """Generate a sine block."""
    audio = np.zeros(num_samples, dtype=np.int16)
    phase = phase_start
    for i in range(num_samples):
        audio[i] = sine_q15(phase)
        phase = (phase + phase_inc) & 0xFFFFFFFF
    return audio, phase


def test_single_block_roundtrip():
    """Test FWPT -> IDWPT roundtrip on a single block."""
    print("Test: Single Block Roundtrip")
    print("=" * 50)

    phase_inc = midi_to_phase_inc(60)  # Middle C
    audio, _ = generate_sine_block(0, phase_inc)

    print(f"Input audio: min={audio.min()}, max={audio.max()}")
    print(f"Input RMS: {np.sqrt(np.mean(audio.astype(np.float32)**2)):.1f}")

    # Forward transform
    coeffs = fwpt_jit(audio.tolist(), headroom_bits=1)
    coeffs_arr = np.array(coeffs, dtype=np.int16)

    print(f"\nFWPT coefficients: min={coeffs_arr.min()}, max={coeffs_arr.max()}")
    print(f"Coeffs RMS: {np.sqrt(np.mean(coeffs_arr.astype(np.float32)**2)):.1f}")

    # Which coefficients have significant energy?
    energy = coeffs_arr.astype(np.float32)**2
    total_energy = energy.sum()
    print(f"\nCoefficient energy distribution:")
    for i in range(64):
        pct = 100 * energy[i] / total_energy if total_energy > 0 else 0
        if pct > 1:
            print(f"  Leaf {i}: {pct:.1f}% ({coeffs_arr[i]:6d})")

    # Inverse transform
    recon = idwpt_jit(coeffs, headroom_bits=1)
    recon_arr = np.array(recon, dtype=np.int16)

    print(f"\nReconstructed: min={recon_arr.min()}, max={recon_arr.max()}")
    print(f"Recon RMS: {np.sqrt(np.mean(recon_arr.astype(np.float32)**2)):.1f}")

    # Error
    error = audio.astype(np.int32) - recon_arr.astype(np.int32)
    print(f"Reconstruction error: max={np.abs(error).max()}, RMS={np.sqrt(np.mean(error.astype(np.float32)**2)):.1f}")


def test_multi_block_continuity():
    """Test coefficient evolution across blocks with continuous audio."""
    print("\n\nTest: Multi-Block Coefficient Evolution")
    print("=" * 50)

    phase_inc = midi_to_phase_inc(60)
    phase = 0

    print("Block | Audio RMS | Coeffs RMS | Coeff[0] | Recon RMS | Error RMS")
    print("-" * 70)

    for block in range(10):
        audio, phase = generate_sine_block(phase, phase_inc)
        coeffs = fwpt_jit(audio.tolist(), headroom_bits=1)
        recon = idwpt_jit(coeffs, headroom_bits=1)

        coeffs_arr = np.array(coeffs, dtype=np.int16)
        recon_arr = np.array(recon, dtype=np.int16)
        error = audio.astype(np.int32) - recon_arr.astype(np.int32)

        audio_rms = np.sqrt(np.mean(audio.astype(np.float32)**2))
        coeffs_rms = np.sqrt(np.mean(coeffs_arr.astype(np.float32)**2))
        recon_rms = np.sqrt(np.mean(recon_arr.astype(np.float32)**2))
        error_rms = np.sqrt(np.mean(error.astype(np.float32)**2))

        print(f"  {block:2d}  | {audio_rms:9.1f} | {coeffs_rms:10.1f} | {coeffs_arr[0]:8d} | {recon_rms:9.1f} | {error_rms:9.1f}")


def test_coefficient_injection():
    """Test directly injecting coefficients and checking IDWPT output."""
    print("\n\nTest: Direct Coefficient Injection")
    print("=" * 50)

    # What happens if we inject constant coefficients?
    coeffs_const = [1000] * 64

    audio = idwpt_jit(coeffs_const, headroom_bits=1)
    audio_arr = np.array(audio, dtype=np.int16)

    print(f"Constant coeffs (all 1000):")
    print(f"  Audio: min={audio_arr.min()}, max={audio_arr.max()}")
    print(f"  Audio[0:8]: {audio_arr[:8]}")

    # What about a single non-zero coefficient?
    for leaf in [0, 1, 31, 32, 63]:
        coeffs = [0] * 64
        coeffs[leaf] = 10000

        audio = idwpt_jit(coeffs, headroom_bits=1)
        audio_arr = np.array(audio, dtype=np.int16)

        print(f"\nSingle coeff at leaf {leaf} (value 10000):")
        print(f"  Audio: min={audio_arr.min()}, max={audio_arr.max()}")
        # Check if it has a pattern
        fft = np.abs(np.fft.fft(audio_arr.astype(np.float32)))
        peak_bin = np.argmax(fft[:32])
        peak_freq = peak_bin * 48000 / 64
        print(f"  FFT peak bin: {peak_bin} ({peak_freq:.1f} Hz)")


def test_sine_coefficient_pattern():
    """Understand what coefficient pattern produces a sine wave."""
    print("\n\nTest: Sine Coefficient Pattern Analysis")
    print("=" * 50)

    phase_inc = midi_to_phase_inc(60)

    # Generate multiple blocks and analyze coefficient pattern
    for phase_offset in [0, 0x20000000, 0x40000000, 0x60000000]:  # 0, 45, 90, 135 degrees
        audio, _ = generate_sine_block(phase_offset, phase_inc)
        coeffs = fwpt_jit(audio.tolist(), headroom_bits=1)
        coeffs_arr = np.array(coeffs, dtype=np.int16)

        # Find top 5 coefficients by magnitude
        abs_coeffs = np.abs(coeffs_arr)
        top_indices = np.argsort(abs_coeffs)[::-1][:5]

        deg = phase_offset / (1 << 32) * 360
        print(f"\nPhase offset: {deg:.0f} degrees")
        print("  Top coefficients (leaf: value):")
        for idx in top_indices:
            print(f"    Leaf {idx:2d}: {coeffs_arr[idx]:7d}")


if __name__ == '__main__':
    test_single_block_roundtrip()
    test_multi_block_continuity()
    test_coefficient_injection()
    test_sine_coefficient_pattern()
