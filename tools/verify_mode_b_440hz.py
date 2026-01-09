"""
Verify Mode B produces correct 440 Hz.

Use multiple blocks to get enough samples for proper frequency resolution.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq

from symcrash_python_ap.vm.harmonic_bank import (
    generate_basis_frames, mode_b_generate_coeffs, ModeBState,
    BLOCK_SIZE, SAMPLE_RATE, VM_K
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_block
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc

def test_mode_b_multiblock():
    """Test Mode B over multiple blocks for proper frequency resolution."""
    print("=" * 60)
    print("Mode B 440 Hz Verification (Multi-Block)")
    print("=" * 60)

    note = 69
    freq_hz = 440.0
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

    print(f"Target: {freq_hz} Hz (MIDI {note})")
    print(f"Phase inc: {phase_inc}")

    # Generate basis frames
    basis = generate_basis_frames(anchor_note=note, H_max=8)

    # Create Mode B state - pure sine
    mb_state = ModeBState()
    mb_state.reset(note=note, phase_inc=phase_inc)
    mb_state.weights.set_sine()
    mb_state.H_active = 1

    print(f"Harmonic weights: a_h[0]={mb_state.weights.a_h_q15[0]}")

    # Generate many blocks
    num_blocks = 100  # 100 * 64 = 6400 samples = ~0.13s at 48kHz
    all_samples = []

    for block_idx in range(num_blocks):
        # Generate 512 coefficients
        carrier_512 = np.zeros(512, dtype=np.int16)
        mode_b_generate_coeffs(mb_state, basis, carrier_512)

        # Use K=4 polyphase IDWPT (like lane_ccs_render does)
        # Reshape to [64, 8] and take every 2nd k-slot
        coeff_2d_k8 = carrier_512.reshape(64, VM_K)
        coeff_2d_k4 = coeff_2d_k8[:, ::2].astype(np.int16)

        block_samples = idwpt_polyphase_block(coeff_2d_k4, headroom_bits=1)
        all_samples.extend(block_samples.tolist())

        # Advance phase
        mb_state.advance_block()

    samples = np.array(all_samples, dtype=np.float64)
    print(f"\nGenerated {len(samples)} samples ({len(samples)/SAMPLE_RATE:.3f}s)")

    # FFT analysis with proper resolution
    spectrum = np.abs(rfft(samples))
    freqs = rfftfreq(len(samples), 1/SAMPLE_RATE)

    bin_size = SAMPLE_RATE / len(samples)
    print(f"FFT bin size: {bin_size:.2f} Hz")

    # Find peaks
    threshold = np.max(spectrum) * 0.1
    peaks = []
    for i in range(1, len(spectrum)-1):
        if spectrum[i] > threshold and spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
            peaks.append((freqs[i], spectrum[i]))

    peaks.sort(key=lambda x: -x[1])

    print(f"\n=== Top Frequency Peaks ===")
    for freq, amp in peaks[:10]:
        # Check if harmonic of 440
        if freq > 0:
            ratio = freq / 440.0
            harmonic = round(ratio)
            error = abs(freq - harmonic * 440.0)
            harmonic_str = f"H{harmonic} (error: {error:.1f} Hz)" if error < 20 else ""
        else:
            harmonic_str = ""
        print(f"  {freq:8.1f} Hz  (amp: {amp:12.0f})  {harmonic_str}")

    # Verify fundamental
    if peaks:
        fund_freq = peaks[0][0]
        print(f"\n*** Detected fundamental: {fund_freq:.1f} Hz ***")

        if abs(fund_freq - 440) < 10:
            print("SUCCESS: Mode B produces correct 440 Hz!")
        else:
            print(f"ISSUE: Expected ~440 Hz, got {fund_freq:.1f} Hz")
            print(f"       Error: {abs(fund_freq - 440):.1f} Hz")


def test_mode_b_with_custom_harmonics():
    """Test Mode B with violin-like custom harmonics."""
    print("\n" + "=" * 60)
    print("Mode B with Violin Custom Harmonics")
    print("=" * 60)

    note = 69
    freq_hz = 440.0
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

    # Violin-like harmonics (H2 strongest, H4 body resonance)
    # H1: 94.3%, H2: 100%, H3: 21.1%, H4: 73.2%, H5-H8: smaller
    violin_harmonics = [
        30890,  # H1: 94.3%
        32767,  # H2: 100.0%
        6926,   # H3: 21.1%
        23969,  # H4: 73.2%
        3673,   # H5: 11.2%
        6138,   # H6: 18.7%
        1164,   # H7: 3.6%
        1228,   # H8: 3.7%
    ]

    print(f"Target: {freq_hz} Hz with violin harmonics")
    print(f"Harmonics: {violin_harmonics}")

    # Generate basis frames
    basis = generate_basis_frames(anchor_note=note, H_max=len(violin_harmonics))

    # Create Mode B state with custom harmonics
    mb_state = ModeBState()
    mb_state.reset(note=note, phase_inc=phase_inc)
    mb_state.weights.set_custom(violin_harmonics)
    mb_state.H_active = len(violin_harmonics)

    print(f"H_active: {mb_state.H_active}")
    print(f"Harmonic weights: {mb_state.weights.a_h_q15[:8].tolist()}")

    # Generate many blocks
    num_blocks = 100
    all_samples = []

    for block_idx in range(num_blocks):
        carrier_512 = np.zeros(512, dtype=np.int16)
        mode_b_generate_coeffs(mb_state, basis, carrier_512)

        coeff_2d_k8 = carrier_512.reshape(64, VM_K)
        coeff_2d_k4 = coeff_2d_k8[:, ::2].astype(np.int16)

        block_samples = idwpt_polyphase_block(coeff_2d_k4, headroom_bits=1)
        all_samples.extend(block_samples.tolist())

        mb_state.advance_block()

    samples = np.array(all_samples, dtype=np.float64)
    print(f"\nGenerated {len(samples)} samples")

    # FFT analysis
    spectrum = np.abs(rfft(samples))
    freqs = rfftfreq(len(samples), 1/SAMPLE_RATE)

    # Find peaks
    threshold = np.max(spectrum) * 0.05
    peaks = []
    for i in range(1, len(spectrum)-1):
        if spectrum[i] > threshold and spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
            peaks.append((freqs[i], spectrum[i]))

    peaks.sort(key=lambda x: -x[1])

    print(f"\n=== Harmonic Analysis ===")
    print(f"{'Freq':>8} | {'Amplitude':>12} | {'Harmonic':>10} | {'Expected':>10}")
    print("-" * 50)

    max_amp = peaks[0][1] if peaks else 1

    for freq, amp in peaks[:10]:
        if freq > 0:
            ratio = freq / 440.0
            harmonic = round(ratio)
            if harmonic > 0 and harmonic <= 8:
                expected = violin_harmonics[harmonic-1] / 32767.0 * 100
                actual = amp / max_amp * 100
                print(f"{freq:8.1f} | {amp:12.0f} | H{harmonic:9d} | {expected:8.1f}% -> {actual:.1f}%")
            else:
                print(f"{freq:8.1f} | {amp:12.0f} | {'SPUR':>10} |")
        else:
            print(f"{freq:8.1f} | {amp:12.0f} | {'DC':>10} |")

    # Verify harmonics are in right place
    print(f"\n*** Fundamental: {peaks[0][0]:.1f} Hz ***")
    if abs(peaks[0][0] - 440) < 10:
        print("SUCCESS: Fundamental at correct frequency!")
    else:
        # Maybe H2 is strongest
        if abs(peaks[0][0] - 880) < 10:
            print("NOTE: H2 is strongest (as expected for violin)")


if __name__ == "__main__":
    test_mode_b_multiblock()
    test_mode_b_with_custom_harmonics()
