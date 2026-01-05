"""
Test runtime coefficient generation for CCS.

Validates:
1. Phase continuity across blocks
2. Correct frequency output
3. Minimal boundary jumps
4. Operator integration
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from symcrash_python_ap.vm.coeff_gen import (
    CoeffGenState,
    generate_coeffs,
    reconstruct_audio,
    ccs_render_block,
    midi_to_freq,
)


def test_basic_ccs():
    """Test basic CCS render."""
    print("Test: Basic CCS Render")
    print("=" * 60)

    state = CoeffGenState()
    state.set_note(60)  # C4

    all_audio = []

    for block in range(20):
        audio = ccs_render_block(state)
        all_audio.append(audio)

        if block < 3:
            print(f"  Block {block}: range=[{audio.min():6d}, {audio.max():6d}], "
                  f"RMS={np.sqrt(np.mean(audio.astype(float)**2)):.1f}")

    full_audio = np.concatenate(all_audio)

    # Check boundary continuity
    max_jump = 0
    worst_block = -1
    for i in range(len(all_audio) - 1):
        jump = abs(int(all_audio[i][-1]) - int(all_audio[i+1][0]))
        if jump > max_jump:
            max_jump = jump
            worst_block = i

    print(f"\n  Max boundary jump: {max_jump} (at block {worst_block})")

    # FFT analysis
    try:
        from scipy.fft import fft
        N = len(full_audio)
        fft_result = np.abs(fft(full_audio.astype(np.float32)))[:N//2]
        freqs = np.fft.fftfreq(N, 1/48000)[:N//2]
        peak_idx = np.argmax(fft_result)
        peak_freq = freqs[peak_idx]
        expected_freq = midi_to_freq(60)
        print(f"  FFT peak: {peak_freq:.2f} Hz (expected {expected_freq:.2f} Hz)")

        if abs(peak_freq - expected_freq) < 5 and max_jump < 2000:
            print("[OK] Basic CCS render successful!\n")
        else:
            print("[WARNING] CCS render has issues\n")
    except ImportError:
        print("[OK] Basic CCS complete (no scipy for FFT)\n")


def test_multiple_notes():
    """Test CCS at different pitches."""
    print("Test: CCS at Multiple Notes")
    print("=" * 60)

    for note in [48, 60, 72, 84]:  # C3, C4, C5, C6
        state = CoeffGenState()
        state.set_note(note)

        all_audio = []
        for _ in range(10):
            audio = ccs_render_block(state)
            all_audio.append(audio)

        full_audio = np.concatenate(all_audio)

        # Check boundary jumps
        max_jump = 0
        for i in range(len(all_audio) - 1):
            jump = abs(int(all_audio[i][-1]) - int(all_audio[i+1][0]))
            max_jump = max(max_jump, jump)

        # FFT
        try:
            from scipy.fft import fft
            N = len(full_audio)
            fft_result = np.abs(fft(full_audio.astype(np.float32)))[:N//2]
            freqs = np.fft.fftfreq(N, 1/48000)[:N//2]
            peak_idx = np.argmax(fft_result)
            peak_freq = freqs[peak_idx]
            expected_freq = midi_to_freq(note)
            error = abs(peak_freq - expected_freq)
            status = "[OK]" if error < 10 and max_jump < 2000 else "[WARN]"
            print(f"  {status} Note {note}: freq={peak_freq:.1f} Hz (expected {expected_freq:.1f}), "
                  f"max_jump={max_jump}")
        except ImportError:
            print(f"  Note {note}: max_jump={max_jump}")

    print()


def test_waveforms():
    """Test different waveforms."""
    print("Test: Different Waveforms")
    print("=" * 60)

    for waveform, name in [(0, "Sine"), (1, "Saw"), (2, "Square")]:
        state = CoeffGenState()
        state.set_note(60)
        state.waveform = waveform

        all_audio = []
        for _ in range(10):
            audio = ccs_render_block(state)
            all_audio.append(audio)

        full_audio = np.concatenate(all_audio)

        # Check boundary jumps
        max_jump = 0
        for i in range(len(all_audio) - 1):
            jump = abs(int(all_audio[i][-1]) - int(all_audio[i+1][0]))
            max_jump = max(max_jump, jump)

        rms = np.sqrt(np.mean(full_audio.astype(float)**2))
        print(f"  {name}: RMS={rms:.1f}, max_jump={max_jump}")

    print()


def test_with_operators():
    """Test CCS with coefficient operators."""
    print("Test: CCS with Operators")
    print("=" * 60)

    # Simple tilt operator: boost low frequencies, cut high
    def tilt_operator(coeffs):
        coeffs = coeffs.astype(np.float32)
        for i in range(64):
            # Boost low leaves (i < 16), cut high leaves (i > 48)
            if i < 16:
                coeffs[i] *= 1.5
            elif i > 48:
                coeffs[i] *= 0.5
        return np.clip(coeffs, -32768, 32767).astype(np.int16)

    state = CoeffGenState()
    state.set_note(60)

    # Without operator
    all_audio_plain = []
    for _ in range(10):
        audio = ccs_render_block(state)
        all_audio_plain.append(audio)

    # Reset and run with operator
    state.phase = 0
    all_audio_tilted = []
    for _ in range(10):
        audio = ccs_render_block(state, apply_operators=tilt_operator)
        all_audio_tilted.append(audio)

    full_plain = np.concatenate(all_audio_plain)
    full_tilted = np.concatenate(all_audio_tilted)

    print(f"  Plain:  RMS={np.sqrt(np.mean(full_plain.astype(float)**2)):.1f}")
    print(f"  Tilted: RMS={np.sqrt(np.mean(full_tilted.astype(float)**2)):.1f}")

    # Check that they're different
    diff = np.abs(full_plain.astype(np.int32) - full_tilted.astype(np.int32))
    if diff.max() > 100:
        print("[OK] Operator modifies output\n")
    else:
        print("[WARNING] Operator has no effect\n")


def test_long_duration():
    """Test long duration for stability."""
    print("Test: Long Duration Stability")
    print("=" * 60)

    state = CoeffGenState()
    state.set_note(60)

    all_audio = []
    max_jumps = []

    for block in range(200):
        audio = ccs_render_block(state)
        all_audio.append(audio)

        if block > 0:
            jump = abs(int(all_audio[-2][-1]) - int(all_audio[-1][0]))
            max_jumps.append(jump)

    overall_max_jump = max(max_jumps)
    avg_jump = sum(max_jumps) / len(max_jumps)

    print(f"  200 blocks rendered")
    print(f"  Max boundary jump: {overall_max_jump}")
    print(f"  Average boundary jump: {avg_jump:.1f}")

    # FFT on full signal
    full_audio = np.concatenate(all_audio)
    try:
        from scipy.fft import fft
        N = len(full_audio)
        fft_result = np.abs(fft(full_audio.astype(np.float32)))[:N//2]
        freqs = np.fft.fftfreq(N, 1/48000)[:N//2]
        peak_idx = np.argmax(fft_result)
        peak_freq = freqs[peak_idx]
        print(f"  FFT peak: {peak_freq:.2f} Hz")

        # Check for spurious peaks
        noise_floor = np.median(fft_result)
        num_significant = np.sum(fft_result > 20 * noise_floor)
        print(f"  Significant peaks: {num_significant} (expect ~1-2)")

    except ImportError:
        pass

    if overall_max_jump < 2000 and avg_jump < 500:
        print("[OK] Long duration stable\n")
    else:
        print("[WARNING] Stability issues\n")


if __name__ == '__main__':
    test_basic_ccs()
    test_multiple_notes()
    test_waveforms()
    test_with_operators()
    test_long_duration()
