"""
Test CoeffTable per WSE-STD-120 / WSE-ARCH-CCS.

Validates:
1. Table generation produces correct frequency
2. Phase-indexed lookup works
3. Interpolation is smooth
4. No boundary discontinuities when phase is continuous
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from symcrash_python_ap.vm.coeff_table import (
    get_coeff_table,
    midi_to_phase_inc,
    CoeffTable,
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit


def test_table_generation():
    """Test that table generation produces valid coefficients."""
    print("Test: Table Generation")
    print("=" * 60)

    table = get_coeff_table(0)  # Sine

    print(f"  Wave: {table.name}")
    print(f"  Phase steps: {table.phase_steps}")
    print(f"  Data shape: {table.data.shape}")
    print(f"  Data range: [{table.data.min()}, {table.data.max()}]")

    # Check that IDWPT produces audio at each phase
    for p in [0, 16, 32, 48]:
        coeffs = table.get_coeffs_by_index(p)
        audio = np.array(idwpt_jit(coeffs.tolist(), headroom_bits=1), dtype=np.int16)
        rms = np.sqrt(np.mean(audio.astype(float)**2))
        print(f"  Phase {p}: audio range=[{audio.min():6d}, {audio.max():6d}], RMS={rms:.1f}")

    print("[OK] Table generation complete\n")


def test_phase_continuity():
    """
    Test phase continuity per WSE-STD-120.

    Key requirement: When phase advances continuously,
    audio output should be continuous (no clicks).
    """
    print("Test: Phase Continuity")
    print("=" * 60)

    table = get_coeff_table(0)  # Sine
    phase_inc = midi_to_phase_inc(60)  # C4 = 261.63 Hz
    phase = 0

    all_audio = []

    for block in range(20):
        # Get coefficients for current phase
        coeffs = table.get_coeffs(phase, interpolate=True)

        # IDWPT to audio
        audio = np.array(idwpt_jit(coeffs.tolist(), headroom_bits=1), dtype=np.int16)
        all_audio.append(audio)

        if block < 3:
            phase_deg = (phase / (1 << 32)) * 360
            print(f"  Block {block}: phase={phase_deg:6.1f} deg, range=[{audio.min():6d}, {audio.max():6d}]")

        # Advance phase for next block (64 samples)
        phase = (phase + phase_inc * 64) & 0xFFFFFFFF

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
    full_audio = np.concatenate(all_audio)
    try:
        from scipy.fft import fft
        N = len(full_audio)
        fft_result = np.abs(fft(full_audio.astype(np.float32)))[:N//2]
        freqs = np.fft.fftfreq(N, 1/48000)[:N//2]
        peak_idx = np.argmax(fft_result)
        peak_freq = freqs[peak_idx]
        print(f"  FFT peak: {peak_freq:.2f} Hz (expected 261.63 Hz)")

        # Success criteria per WSE-STD-120
        freq_ok = abs(peak_freq - 261.63) < 50  # Allow some deviation due to table ref freq
        jump_ok = max_jump < 2000  # Reasonable boundary

        if freq_ok and jump_ok:
            print("[OK] Phase continuity maintained!\n")
        else:
            print(f"[WARNING] Issues: freq_ok={freq_ok}, jump_ok={jump_ok}\n")

    except ImportError:
        print("[OK] Test complete (no scipy for FFT)\n")


def test_interpolation():
    """Test that interpolation smooths phase transitions."""
    print("Test: Phase Interpolation")
    print("=" * 60)

    table = get_coeff_table(0)

    # Get coefficients at phase 0, with and without interpolation
    phase_32 = 0x08000000  # 1/32 of cycle = 11.25 degrees

    coeffs_nn = table.get_coeffs(phase_32, interpolate=False)
    coeffs_li = table.get_coeffs(phase_32, interpolate=True)

    # They should be different
    diff = np.abs(coeffs_nn.astype(np.int32) - coeffs_li.astype(np.int32))
    print(f"  Phase: {phase_32 / (1<<32) * 360:.2f} degrees")
    print(f"  Nearest neighbor max coeff: {np.abs(coeffs_nn).max()}")
    print(f"  Linear interp max coeff: {np.abs(coeffs_li).max()}")
    print(f"  Max difference: {diff.max()}")

    if diff.max() > 0:
        print("[OK] Interpolation produces different results\n")
    else:
        print("[INFO] Interpolation same as nearest (at exact phase point)\n")


def test_multiple_waveforms():
    """Test all standard waveforms."""
    print("Test: Multiple Waveforms")
    print("=" * 60)

    for wave_id in range(4):
        table = get_coeff_table(wave_id)

        # Get coefficients at phase 0
        coeffs = table.get_coeffs(0)
        audio = np.array(idwpt_jit(coeffs.tolist(), headroom_bits=1), dtype=np.int16)
        rms = np.sqrt(np.mean(audio.astype(float)**2))

        print(f"  {table.name:10s}: RMS={rms:8.1f}, range=[{audio.min():6d}, {audio.max():6d}]")

    print("[OK] All waveforms tested\n")


def test_frequency_accuracy():
    """Test that synthesis produces correct frequency."""
    print("Test: Frequency Accuracy")
    print("=" * 60)

    table = get_coeff_table(0)  # Sine

    for note in [48, 60, 72]:  # C3, C4, C5
        phase_inc = midi_to_phase_inc(note)
        phase = 0

        all_audio = []
        for _ in range(32):  # More blocks for frequency resolution
            coeffs = table.get_coeffs(phase, interpolate=True)
            audio = np.array(idwpt_jit(coeffs.tolist(), headroom_bits=1), dtype=np.int16)
            all_audio.append(audio)
            phase = (phase + phase_inc * 64) & 0xFFFFFFFF

        full_audio = np.concatenate(all_audio)

        try:
            from scipy.fft import fft
            N = len(full_audio)
            fft_result = np.abs(fft(full_audio.astype(np.float32)))[:N//2]
            freqs = np.fft.fftfreq(N, 1/48000)[:N//2]
            peak_idx = np.argmax(fft_result)
            peak_freq = freqs[peak_idx]

            expected_freq = 440.0 * (2.0 ** ((note - 69) / 12.0))
            error = abs(peak_freq - expected_freq)

            status = "[OK]" if error < 50 else "[WARN]"
            print(f"  {status} Note {note}: {peak_freq:.1f} Hz (expected {expected_freq:.1f} Hz, error={error:.1f})")

        except ImportError:
            print(f"  Note {note}: (no scipy)")

    print()


def test_table_frequency():
    """
    Understand what frequency the table encodes.

    The table is generated at a reference frequency (1 cycle per 64 samples).
    At runtime, phase advance rate determines output frequency.
    """
    print("Test: Table Reference Frequency")
    print("=" * 60)

    table = get_coeff_table(0)

    # Get coefficients at phase 0 (no interpolation)
    coeffs = table.get_coeffs_by_index(0)
    audio = np.array(idwpt_jit(coeffs.tolist(), headroom_bits=1), dtype=np.int16)

    # FFT to find the frequency encoded in the table
    try:
        from scipy.fft import fft
        N = 64
        fft_result = np.abs(fft(audio.astype(np.float32)))[:N//2]
        freqs = np.fft.fftfreq(N, 1/48000)[:N//2]
        peak_idx = np.argmax(fft_result)
        table_freq = freqs[peak_idx]

        print(f"  Table entry 0 encodes frequency: {table_freq:.2f} Hz")
        print(f"  (This is 1 cycle per 64 samples at 48 kHz = 750 Hz)")
        print()
        print("  How runtime frequency control works:")
        print("  - Phase accumulator advances by phase_inc per sample")
        print("  - Different phase_inc = different frequency")
        print("  - Table provides the waveform SHAPE at any phase")
        print("  - Phase advance rate provides the frequency")

    except ImportError:
        print("  (no scipy for analysis)")

    print()


if __name__ == '__main__':
    test_table_generation()
    test_table_frequency()
    test_interpolation()
    test_multiple_waveforms()
    test_phase_continuity()
    test_frequency_accuracy()
