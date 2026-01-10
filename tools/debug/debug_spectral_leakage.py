"""Check if 50 Hz artifact is real or FFT spectral leakage."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq

SAMPLE_RATE = 48000


def analyze_with_window(samples, name, window=True):
    """Analyze spectrum with optional Hann window to reduce spectral leakage."""
    n = len(samples)
    freqs = rfftfreq(n, 1/SAMPLE_RATE)

    if window:
        # Apply Hann window to reduce spectral leakage
        hann = np.hanning(n)
        windowed = samples * hann
        # Correct for window energy loss
        window_scale = 2.0 / np.mean(hann)
        spectrum = np.abs(rfft(windowed)) * window_scale
        window_name = "Hann windowed"
    else:
        spectrum = np.abs(rfft(samples))
        window_name = "Rectangular (no window)"

    # Find 50 Hz and 440 Hz components
    idx_50 = int(50 * n / SAMPLE_RATE)
    idx_440 = int(440 * n / SAMPLE_RATE)

    amp_50 = spectrum[idx_50]
    amp_440 = spectrum[idx_440]

    rel_db = 20 * np.log10(amp_50 / amp_440) if amp_440 > 0 else -100

    print(f"{name} ({window_name}):")
    print(f"  50 Hz: {rel_db:.1f} dB relative to 440 Hz")

    return spectrum, freqs


def test_pure_sine():
    """Test pure sine with no processing."""
    print("=== Pure Sine Reference ===")
    freq_hz = 440.0
    duration_sec = 3.0
    num_samples = int(duration_sec * SAMPLE_RATE)

    t = np.arange(num_samples) / SAMPLE_RATE
    sine = 32767 * np.sin(2 * np.pi * freq_hz * t)

    analyze_with_window(sine, "Pure sine", window=False)
    analyze_with_window(sine, "Pure sine", window=True)


def test_mode_b_output():
    """Test Mode B output."""
    print("\n=== Mode B Output ===")

    from symcrash_python_ap.vm.harmonic_bank import (
        generate_basis_frames, mode_b_generate_coeffs, ModeBState,
        clear_basis_atlas, BLOCK_SIZE, VM_K, K_STRIDE, NUM_LEAVES
    )
    from symcrash_python_ap.vm.tables import COEFF_LEN
    from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_block
    from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc

    freq_hz = 440.0
    note = 69
    duration_sec = 3.0
    num_blocks = int(duration_sec * SAMPLE_RATE / BLOCK_SIZE)

    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

    clear_basis_atlas()
    basis = generate_basis_frames(anchor_note=note, H_max=8, k_stride=K_STRIDE)

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

    samples = np.array(all_samples, dtype=np.float64)

    analyze_with_window(samples, "Mode B sine", window=False)
    analyze_with_window(samples, "Mode B sine", window=True)


def test_difference_spectrum():
    """Analyze the difference between Mode B and pure sine."""
    print("\n=== Difference Analysis ===")

    from symcrash_python_ap.vm.harmonic_bank import (
        generate_basis_frames, mode_b_generate_coeffs, ModeBState,
        clear_basis_atlas, BLOCK_SIZE, VM_K, K_STRIDE, NUM_LEAVES
    )
    from symcrash_python_ap.vm.tables import COEFF_LEN
    from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_block
    from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc

    freq_hz = 440.0
    note = 69
    duration_sec = 3.0
    num_blocks = int(duration_sec * SAMPLE_RATE / BLOCK_SIZE)
    num_samples = num_blocks * BLOCK_SIZE

    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

    clear_basis_atlas()
    basis = generate_basis_frames(anchor_note=note, H_max=8, k_stride=K_STRIDE)

    mb_state = ModeBState()
    mb_state.reset(note=note, phase_inc=phase_inc)
    mb_state.weights.set_sine()
    mb_state.H_active = 1

    mode_b_samples = []

    for _ in range(num_blocks):
        carrier = np.zeros(COEFF_LEN, dtype=np.int16)
        mode_b_generate_coeffs(mb_state, basis, carrier)

        coeff_2d = carrier.reshape(NUM_LEAVES, VM_K)
        block_samples = idwpt_polyphase_block(coeff_2d, headroom_bits=1)
        mode_b_samples.extend(block_samples.tolist())
        mb_state.advance_block()

    mode_b = np.array(mode_b_samples, dtype=np.float64)

    # Generate reference sine
    t = np.arange(len(mode_b)) / SAMPLE_RATE
    ref_sine = 32767 * np.sin(2 * np.pi * freq_hz * t)

    # Compute difference
    diff = mode_b - ref_sine

    print(f"Difference RMS: {np.sqrt(np.mean(diff**2)):.1f}")
    print(f"Difference max: {np.max(np.abs(diff)):.1f}")

    # Analyze difference spectrum with window
    n = len(diff)
    freqs = rfftfreq(n, 1/SAMPLE_RATE)

    hann = np.hanning(n)
    windowed_diff = diff * hann
    window_scale = 2.0 / np.mean(hann)
    diff_spectrum = np.abs(rfft(windowed_diff)) * window_scale

    print("\nDifference spectrum peaks (windowed):")
    for freq_target in [50, 100, 200, 375, 440, 750]:
        idx = int(freq_target * n / SAMPLE_RATE)
        if idx < len(diff_spectrum):
            amp = diff_spectrum[idx]
            print(f"  {freq_target} Hz: amplitude = {amp:.0f}")


def test_integer_period_length():
    """Test with sample count that's an integer number of 440 Hz cycles."""
    print("\n=== Integer Period Length Test ===")

    freq_hz = 440.0
    # 440 Hz period = 48000/440 = 109.09... samples
    # For integer cycles, use LCM approach or just use many cycles
    # Actually, 440 Hz at 48 kHz: period = 48000/440 = 109.09 samples
    # This never produces integer period, so leakage is inevitable

    # Best we can do: use very long sample to minimize leakage
    duration_sec = 10.0  # 10 seconds = 4400 cycles (still fractional samples)
    num_samples = int(duration_sec * SAMPLE_RATE)

    t = np.arange(num_samples) / SAMPLE_RATE
    sine = 32767 * np.sin(2 * np.pi * freq_hz * t)

    print("With 10 second sample (more frequency resolution):")
    analyze_with_window(sine, "Pure sine", window=False)
    analyze_with_window(sine, "Pure sine", window=True)


def check_spectral_resolution():
    """Explain spectral leakage issue."""
    print("\n=== Spectral Resolution Analysis ===")

    freq_hz = 440.0
    duration_sec = 3.0
    num_samples = int(duration_sec * SAMPLE_RATE)

    # Frequency resolution of FFT
    freq_resolution = SAMPLE_RATE / num_samples

    print(f"FFT length: {num_samples} samples")
    print(f"Frequency resolution: {freq_resolution:.4f} Hz")
    print(f"Number of 440 Hz cycles in sample: {duration_sec * freq_hz:.1f}")
    print(f"Is 440 Hz on a bin? {freq_hz / freq_resolution:.4f} (integer = on bin)")

    # 440 / freq_resolution tells us if 440 lands exactly on a bin
    bin_440 = freq_hz / freq_resolution
    if abs(bin_440 - round(bin_440)) < 0.001:
        print("  -> 440 Hz is exactly on a bin, no leakage to 50 Hz")
    else:
        print("  -> 440 Hz is NOT exactly on a bin, leakage will occur")
        print(f"  -> Leakage spreads across bins, including 50 Hz")


if __name__ == "__main__":
    test_pure_sine()
    check_spectral_resolution()
    test_integer_period_length()
    test_mode_b_output()
    test_difference_spectrum()
