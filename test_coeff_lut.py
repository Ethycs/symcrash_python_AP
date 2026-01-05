"""
Test WSE-STD-121 CoeffLUT generation and synthesis.
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from symcrash_python_ap.vm.coeff_lut import (
    CoeffLUT,
    get_coeff_lut,
    polyphase_idwpt_streaming,
    BLOCK_SIZE,
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit


def midi_to_phase_inc(note: int, sample_rate: int = 48000) -> int:
    freq_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))
    return int((freq_hz / sample_rate) * (1 << 32))


def test_lut_generation():
    """Test CoeffLUT generation."""
    print("Test: CoeffLUT Generation")
    print("=" * 60)

    lut = CoeffLUT.generate_sine()

    print(f"  Shape: {lut.data.shape}")
    print(f"  Leaves: {lut.num_leaves}")
    print(f"  K (taps): {lut.K}")
    print(f"  P (phases): {lut.P}")
    print(f"  Data range: [{lut.data.min()}, {lut.data.max()}]")

    # Check a few phase steps
    for p in [0, 64, 128, 192]:
        coeffs = lut.get_coeffs_for_phase(p)
        rms = np.sqrt(np.mean(coeffs.astype(np.float32)**2))
        print(f"  Phase {p}: RMS={rms:.1f}, shape={coeffs.shape}")

    print("[OK] CoeffLUT generated\n")


def test_simple_synthesis():
    """Test simple synthesis using single-tap IDWPT."""
    print("Test: Simple Synthesis (Single Tap)")
    print("=" * 60)

    lut = get_coeff_lut(0)  # Sine
    phase_inc = midi_to_phase_inc(60)

    all_audio = []
    phase = 0

    for block in range(20):
        # Get phase index (0..P-1)
        phase_idx = (phase >> 24) % lut.P

        # Get single tap (first one for simplicity)
        coeffs = lut.get_single_tap(phase_idx, 0)

        # IDWPT
        audio = idwpt_jit(coeffs.astype(np.int16).tolist(), headroom_bits=1)
        audio_arr = np.array(audio, dtype=np.int16)
        all_audio.append(audio_arr)

        if block < 5:
            print(f"  Block {block}: phase_idx={phase_idx:3d}, range=[{audio_arr.min():6d},{audio_arr.max():6d}]")

        # Advance phase
        phase = (phase + phase_inc * BLOCK_SIZE) & 0xFFFFFFFF

    full_audio = np.concatenate(all_audio)

    # Check boundaries
    max_jump = 0
    for i in range(len(all_audio) - 1):
        jump = abs(int(all_audio[i][-1]) - int(all_audio[i+1][0]))
        max_jump = max(max_jump, jump)

    print(f"\n  Max boundary jump: {max_jump}")

    # FFT
    try:
        from scipy.fft import fft
        N = len(full_audio)
        fft_result = np.abs(fft(full_audio.astype(np.float32)))[:N//2]
        freqs = np.fft.fftfreq(N, 1/48000)[:N//2]
        peak_idx = np.argmax(fft_result)
        peak_freq = freqs[peak_idx]
        print(f"  FFT peak: {peak_freq:.2f} Hz (expected 261.63 Hz)")

        if abs(peak_freq - 261.63) < 10:
            print("[OK] Correct frequency!\n")
        else:
            print(f"[WARNING] Frequency off by {abs(peak_freq - 261.63):.2f} Hz\n")
    except ImportError:
        print("  (scipy not available)\n")


def test_polyphase_synthesis():
    """Test polyphase streaming synthesis."""
    print("Test: Polyphase Streaming Synthesis")
    print("=" * 60)

    lut = get_coeff_lut(0)  # Sine
    phase_inc = midi_to_phase_inc(60)

    all_audio = []
    phase = 0

    for block in range(20):
        audio = polyphase_idwpt_streaming(lut, phase, phase_inc)
        all_audio.append(audio)

        if block < 5:
            print(f"  Block {block}: range=[{audio.min():6d},{audio.max():6d}], RMS={np.sqrt(np.mean(audio.astype(np.float32)**2)):.1f}")

        # Advance phase
        phase = (phase + phase_inc * BLOCK_SIZE) & 0xFFFFFFFF

    full_audio = np.concatenate(all_audio)

    # Check boundaries
    max_jump = 0
    for i in range(len(all_audio) - 1):
        jump = abs(int(all_audio[i][-1]) - int(all_audio[i+1][0]))
        max_jump = max(max_jump, jump)

    print(f"\n  Max boundary jump: {max_jump}")

    # FFT
    try:
        from scipy.fft import fft
        N = len(full_audio)
        fft_result = np.abs(fft(full_audio.astype(np.float32)))[:N//2]
        freqs = np.fft.fftfreq(N, 1/48000)[:N//2]
        peak_idx = np.argmax(fft_result)
        peak_freq = freqs[peak_idx]
        print(f"  FFT peak: {peak_freq:.2f} Hz (expected 261.63 Hz)")

        # DC level
        dc = np.abs(fft_result[0])
        ac = np.abs(fft_result[peak_idx])
        print(f"  DC/AC ratio: {dc/ac:.3f}")

        if abs(peak_freq - 261.63) < 10:
            print("[OK] Correct frequency!\n")
        else:
            print(f"[WARNING] Frequency off by {abs(peak_freq - 261.63):.2f} Hz\n")
    except ImportError:
        print("  (scipy not available)\n")


def test_different_waveforms():
    """Test different waveforms."""
    print("Test: Different Waveforms")
    print("=" * 60)

    phase_inc = midi_to_phase_inc(60)

    for wave_id, name in [(0, "sine"), (1, "saw"), (2, "square")]:
        print(f"\n  {name}:")

        lut = get_coeff_lut(wave_id)
        phase = 0
        all_audio = []

        for block in range(20):
            audio = polyphase_idwpt_streaming(lut, phase, phase_inc)
            all_audio.append(audio)
            phase = (phase + phase_inc * BLOCK_SIZE) & 0xFFFFFFFF

        full_audio = np.concatenate(all_audio)

        try:
            from scipy.fft import fft
            N = len(full_audio)
            fft_result = np.abs(fft(full_audio.astype(np.float32)))[:N//2]
            freqs = np.fft.fftfreq(N, 1/48000)[:N//2]
            peak_idx = np.argmax(fft_result)
            peak_freq = freqs[peak_idx]
            print(f"    FFT peak: {peak_freq:.2f} Hz")
        except ImportError:
            pass

    print("\n[OK] Waveform test complete\n")


if __name__ == '__main__':
    test_lut_generation()
    test_simple_synthesis()
    test_polyphase_synthesis()
    test_different_waveforms()
