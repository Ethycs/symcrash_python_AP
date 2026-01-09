"""Debug THD sources in CCS path."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from scipy.fft import rfft, rfftfreq

from symcrash_python_ap.vm.coeff_lut_streaming import (
    CCSState, ccs_generate_coeffs_polyphase, get_lut_for_note,
    NUM_LEAVES, VM_K, WAVE_SINE
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit

SAMPLE_RATE = 48000
NUM_BLOCKS = 100
BLOCK_SIZE = 64


def calc_thd(audio, fundamental_freq):
    """Calculate THD from audio samples."""
    audio = audio - np.mean(audio)  # Remove DC
    spectrum = rfft(audio)
    freqs = rfftfreq(len(audio), 1.0/SAMPLE_RATE)

    # Find fundamental
    fund_idx = np.argmin(np.abs(freqs - fundamental_freq))
    fund_power = np.abs(spectrum[fund_idx])**2

    # Sum harmonic power
    harmonic_power = 0
    for h in range(2, 11):
        h_freq = fundamental_freq * h
        if h_freq > SAMPLE_RATE / 2:
            break
        h_idx = np.argmin(np.abs(freqs - h_freq))
        harmonic_power += np.abs(spectrum[h_idx])**2

    thd = 100 * np.sqrt(harmonic_power / (fund_power + 1e-10))
    return thd


def test_pure_lut():
    """Test 1: Pure LUT lookup without IDWPT - what does the LUT contain?"""
    print("\n" + "=" * 60)
    print("Test 1: Pure LUT Content Analysis")
    print("=" * 60)

    lut = get_lut_for_note(WAVE_SINE, 60)
    print(f"LUT: K={lut.K}, P={lut.P}, num_leaves={lut.num_leaves}")

    # Sample LUT at multiple phases and look at coefficient structure
    phases = [0, 1 << 30, 1 << 31, 3 << 30]  # 0°, 90°, 180°, 270°
    phase_names = ["0°", "90°", "180°", "270°"]

    for phase, name in zip(phases, phase_names):
        subband = np.zeros((NUM_LEAVES, VM_K), dtype=np.int16)
        lut.get_subband_samples(phase, subband)

        # Analyze coefficient distribution
        k0_coeffs = subband[:, 0]
        print(f"\n  Phase {name}:")
        print(f"    k=0 coeffs: sum={k0_coeffs.sum()}, max={k0_coeffs.max()}, min={k0_coeffs.min()}")
        print(f"    Non-zero coeffs: {np.count_nonzero(k0_coeffs)}/64")

        # Check if coefficients are symmetric (pure sine should be)
        # For sine, coefficients should be anti-symmetric around center
        first_half = k0_coeffs[:32]
        second_half = k0_coeffs[32:][::-1]
        symmetry_error = np.abs(first_half + second_half).max()
        print(f"    Anti-symmetry error: {symmetry_error}")


def test_idwpt_distortion():
    """Test 2: IDWPT reconstruction distortion."""
    print("\n" + "=" * 60)
    print("Test 2: IDWPT Reconstruction Accuracy")
    print("=" * 60)

    # Create a known pure sine in time domain
    freq = 261.63
    t = np.arange(BLOCK_SIZE) / SAMPLE_RATE
    pure_sine = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)

    # Forward transform to get coefficients
    from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit

    coeffs = fwpt_jit(pure_sine.tolist())
    print(f"FWPT coefficients: max={max(coeffs)}, min={min(coeffs)}")

    # Inverse transform
    reconstructed = idwpt_jit(coeffs, headroom_bits=1)
    reconstructed = np.array(reconstructed, dtype=np.int16)

    # Compare
    diff = np.abs(pure_sine.astype(int) - reconstructed.astype(int))
    print(f"Reconstruction error: max={diff.max()}, mean={diff.mean():.2f}")

    # Calculate THD of pure sine vs reconstructed
    # Extend to multiple blocks for better FFT resolution
    pure_extended = np.tile(pure_sine, NUM_BLOCKS)
    recon_extended = np.tile(reconstructed, NUM_BLOCKS)

    thd_pure = calc_thd(pure_extended.astype(float), freq)
    thd_recon = calc_thd(recon_extended.astype(float), freq)

    print(f"\nTHD of pure sine (reference): {thd_pure:.4f}%")
    print(f"THD after FWPT->IDWPT round-trip: {thd_recon:.4f}%")


def test_lut_quality():
    """Test 3: LUT-based generation quality."""
    print("\n" + "=" * 60)
    print("Test 3: LUT-Based Generation Quality")
    print("=" * 60)

    freq = 261.63
    phase_inc = int((freq / SAMPLE_RATE) * (1 << 32))

    ccs = CCSState()
    ccs.note = 60
    ccs.wave_id = WAVE_SINE
    ccs.phase_inc = phase_inc
    ccs.phase = 0

    all_audio = []
    for _ in range(NUM_BLOCKS):
        carrier = np.zeros(NUM_LEAVES * VM_K, dtype=np.int16)
        ccs_generate_coeffs_polyphase(ccs, carrier)

        # Extract k=0 and IDWPT
        coeff_2d = carrier.reshape(NUM_LEAVES, VM_K)
        coeffs = coeff_2d[:, 0].tolist()
        audio = idwpt_jit(coeffs, headroom_bits=1)
        all_audio.append(np.array(audio, dtype=np.int16))

        ccs.advance_block()

    audio = np.concatenate(all_audio).astype(float)
    thd = calc_thd(audio, freq)

    print(f"THD from LUT + IDWPT: {thd:.4f}%")

    # Check block boundaries
    jumps = []
    for i in range(1, NUM_BLOCKS):
        idx = i * BLOCK_SIZE
        jump = abs(audio[idx] - audio[idx - 1])
        jumps.append(jump)

    print(f"Block boundary jumps: max={max(jumps):.0f}, mean={np.mean(jumps):.0f}")


def test_headroom_impact():
    """Test 4: Impact of headroom_bits on THD."""
    print("\n" + "=" * 60)
    print("Test 4: Headroom Bits Impact on THD")
    print("=" * 60)

    freq = 261.63
    phase_inc = int((freq / SAMPLE_RATE) * (1 << 32))

    for headroom in [0, 1, 2, 3]:
        ccs = CCSState()
        ccs.note = 60
        ccs.wave_id = WAVE_SINE
        ccs.phase_inc = phase_inc
        ccs.phase = 0

        all_audio = []
        for _ in range(NUM_BLOCKS):
            carrier = np.zeros(NUM_LEAVES * VM_K, dtype=np.int16)
            ccs_generate_coeffs_polyphase(ccs, carrier)

            coeff_2d = carrier.reshape(NUM_LEAVES, VM_K)
            coeffs = coeff_2d[:, 0].tolist()
            audio = idwpt_jit(coeffs, headroom_bits=headroom)
            all_audio.append(np.array(audio, dtype=np.int16))

            ccs.advance_block()

        audio = np.concatenate(all_audio).astype(float)
        thd = calc_thd(audio, freq)
        rms = np.sqrt(np.mean((audio - np.mean(audio))**2))

        print(f"  headroom_bits={headroom}: THD={thd:.4f}%, RMS={rms:.0f}")


def test_direct_sine_generation():
    """Test 5: Direct sine generation (bypass LUT) as reference."""
    print("\n" + "=" * 60)
    print("Test 5: Direct Sine Generation (Reference)")
    print("=" * 60)

    freq = 261.63
    total_samples = NUM_BLOCKS * BLOCK_SIZE
    t = np.arange(total_samples) / SAMPLE_RATE

    # Pure floating-point sine
    pure_sine_float = np.sin(2 * np.pi * freq * t)

    # Quantized to int16
    pure_sine_int16 = (pure_sine_float * 16000).astype(np.int16)

    thd_float = calc_thd(pure_sine_float * 16000, freq)
    thd_int16 = calc_thd(pure_sine_int16.astype(float), freq)

    print(f"THD of pure float sine: {thd_float:.6f}%")
    print(f"THD of int16 quantized sine: {thd_int16:.6f}%")

    # Check block boundaries on pure sine
    jumps = []
    for i in range(1, NUM_BLOCKS):
        idx = i * BLOCK_SIZE
        jump = abs(float(pure_sine_int16[idx]) - float(pure_sine_int16[idx - 1]))
        jumps.append(jump)

    print(f"Pure sine block boundaries: max={max(jumps):.0f}, mean={np.mean(jumps):.0f}")


def test_coefficient_quantization():
    """Test 6: Coefficient quantization error."""
    print("\n" + "=" * 60)
    print("Test 6: Coefficient Quantization Analysis")
    print("=" * 60)

    # Generate a pure sine and compute its wavelet coefficients
    freq = 261.63
    t = np.arange(BLOCK_SIZE) / SAMPLE_RATE
    pure_sine = np.sin(2 * np.pi * freq * t) * 16000

    from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit

    # Float coefficients
    coeffs_float = fwpt_jit(pure_sine.tolist())

    # Int16 coefficients (what LUT stores)
    coeffs_int16 = [int(round(c)) for c in coeffs_float]
    coeffs_int16 = [max(-32768, min(32767, c)) for c in coeffs_int16]

    # Quantization error
    quant_error = [abs(f - i) for f, i in zip(coeffs_float, coeffs_int16)]
    print(f"Coefficient quantization error: max={max(quant_error):.2f}, mean={np.mean(quant_error):.2f}")

    # Reconstruct from both
    recon_float = idwpt_jit(coeffs_float, headroom_bits=1)
    recon_int16 = idwpt_jit(coeffs_int16, headroom_bits=1)

    # THD comparison
    recon_float_ext = np.tile(recon_float, NUM_BLOCKS)
    recon_int16_ext = np.tile(recon_int16, NUM_BLOCKS)

    thd_float = calc_thd(np.array(recon_float_ext, dtype=float), freq)
    thd_int16 = calc_thd(np.array(recon_int16_ext, dtype=float), freq)

    print(f"THD from float coeffs: {thd_float:.4f}%")
    print(f"THD from int16 coeffs: {thd_int16:.4f}%")


def main():
    print("=" * 60)
    print("THD SOURCE ANALYSIS")
    print("=" * 60)

    test_pure_lut()
    test_idwpt_distortion()
    test_lut_quality()
    test_headroom_impact()
    test_direct_sine_generation()
    test_coefficient_quantization()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
