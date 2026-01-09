"""Test THD with different FFT resolutions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import windows

SAMPLE_RATE = 48000


def calc_thd_windowed(audio, fundamental_freq, use_window=True):
    """Calculate THD with optional windowing to reduce spectral leakage."""
    audio = np.array(audio, dtype=float)
    audio = audio - np.mean(audio)  # Remove DC

    if use_window:
        # Use Hann window to reduce spectral leakage
        window = windows.hann(len(audio))
        audio = audio * window
        # Compensate for window energy loss
        audio = audio * 2

    spectrum = rfft(audio)
    freqs = rfftfreq(len(audio), 1.0/SAMPLE_RATE)

    # Find fundamental (search in small range)
    fund_range = np.abs(freqs - fundamental_freq) < 10
    fund_idx = np.argmax(np.abs(spectrum) * fund_range)
    fund_power = np.abs(spectrum[fund_idx])**2

    # Sum harmonic power (search in small range around each harmonic)
    harmonic_power = 0
    for h in range(2, 11):
        h_freq = fundamental_freq * h
        if h_freq > SAMPLE_RATE / 2:
            break
        h_range = np.abs(freqs - h_freq) < 10
        if np.any(h_range):
            h_idx = np.argmax(np.abs(spectrum) * h_range)
            harmonic_power += np.abs(spectrum[h_idx])**2

    thd = 100 * np.sqrt(harmonic_power / (fund_power + 1e-10))
    return thd


def test_fft_resolution():
    """Test how FFT resolution affects apparent THD."""
    print("=" * 60)
    print("FFT Resolution Impact on THD Measurement")
    print("=" * 60)

    freq = 261.63  # C4

    # Test with different numbers of blocks
    for num_blocks in [100, 500, 1000, 2000]:
        total_samples = num_blocks * 64
        t = np.arange(total_samples) / SAMPLE_RATE

        # Pure sine
        pure_sine = np.sin(2 * np.pi * freq * t) * 16000

        fft_resolution = SAMPLE_RATE / total_samples

        thd_no_window = calc_thd_windowed(pure_sine, freq, use_window=False)
        thd_windowed = calc_thd_windowed(pure_sine, freq, use_window=True)

        print(f"\n{num_blocks} blocks ({total_samples} samples):")
        print(f"  FFT resolution: {fft_resolution:.3f} Hz/bin")
        print(f"  THD without window: {thd_no_window:.4f}%")
        print(f"  THD with Hann window: {thd_windowed:.4f}%")


def test_coherent_frequency():
    """Test with a frequency that lands exactly on FFT bins."""
    print("\n" + "=" * 60)
    print("Coherent Frequency Test")
    print("=" * 60)

    num_blocks = 100
    total_samples = num_blocks * 64

    # Find a frequency that lands exactly on a bin
    # freq = k * (SAMPLE_RATE / total_samples) for integer k
    fft_resolution = SAMPLE_RATE / total_samples
    k = round(261.63 / fft_resolution)
    coherent_freq = k * fft_resolution

    print(f"\nOriginal frequency: 261.63 Hz")
    print(f"Coherent frequency: {coherent_freq:.4f} Hz (bin {k})")

    t = np.arange(total_samples) / SAMPLE_RATE

    # Test both frequencies
    for freq, name in [(261.63, "Non-coherent (C4)"), (coherent_freq, "Coherent")]:
        pure_sine = np.sin(2 * np.pi * freq * t) * 16000

        thd_no_window = calc_thd_windowed(pure_sine, freq, use_window=False)
        thd_windowed = calc_thd_windowed(pure_sine, freq, use_window=True)

        print(f"\n{name} ({freq:.4f} Hz):")
        print(f"  THD without window: {thd_no_window:.6f}%")
        print(f"  THD with Hann window: {thd_windowed:.6f}%")


def test_lut_with_windowing():
    """Test LUT-based synthesis with windowed THD measurement."""
    print("\n" + "=" * 60)
    print("LUT-Based Synthesis with Windowed THD")
    print("=" * 60)

    from symcrash_python_ap.vm.coeff_lut_streaming import (
        CCSState, ccs_generate_coeffs_polyphase, NUM_LEAVES, VM_K, WAVE_SINE
    )
    from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit

    freq = 261.63
    phase_inc = int((freq / SAMPLE_RATE) * (1 << 32))

    for num_blocks in [100, 500, 1000]:
        ccs = CCSState()
        ccs.note = 60
        ccs.wave_id = WAVE_SINE
        ccs.phase_inc = phase_inc
        ccs.phase = 0

        all_audio = []
        for _ in range(num_blocks):
            carrier = np.zeros(NUM_LEAVES * VM_K, dtype=np.int16)
            ccs_generate_coeffs_polyphase(ccs, carrier)

            coeff_2d = carrier.reshape(NUM_LEAVES, VM_K)
            coeffs = coeff_2d[:, 0].tolist()
            audio = idwpt_jit(coeffs, headroom_bits=1)
            all_audio.append(np.array(audio, dtype=np.int16))

            ccs.advance_block()

        audio = np.concatenate(all_audio).astype(float)

        thd_no_window = calc_thd_windowed(audio, freq, use_window=False)
        thd_windowed = calc_thd_windowed(audio, freq, use_window=True)

        print(f"\n{num_blocks} blocks:")
        print(f"  THD without window: {thd_no_window:.4f}%")
        print(f"  THD with Hann window: {thd_windowed:.4f}%")


def main():
    test_fft_resolution()
    test_coherent_frequency()
    test_lut_with_windowing()


if __name__ == "__main__":
    main()
