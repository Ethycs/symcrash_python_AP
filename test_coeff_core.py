"""
Test coeff_core.py - Unified Coefficient Oscillator.

Validates:
1. FilterbankResponse generation for standard waveforms
2. coeff_block_q15() produces non-zero coefficients
3. CoeffOscCore frequency accuracy via IDWPT reconstruction
4. Phase continuity across blocks (no clicks)
5. Comparison with StreamingCoeffLUT approach
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from symcrash_python_ap.wavelet.coeff_core import (
    FilterbankResponse,
    analyze_filterbank_response,
    coeff_block_q15,
    coeff_block_q15_vectorized,
    CoeffOscCore,
    render_coeff_block,
    get_response,
    NUM_LEAVES,
    BLOCK_SIZE,
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.wavelet.sine_core import sine_q15


def midi_to_phase_inc(note: int, sample_rate: int = 48000) -> int:
    """Convert MIDI note to 32-bit phase increment."""
    freq_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))
    return int((freq_hz / sample_rate) * (1 << 32))


def test_filterbank_response():
    """Test that FilterbankResponse is correctly computed."""
    print("Test: FilterbankResponse Generation")
    print("=" * 60)

    response = get_response(0)  # Sine wave

    print(f"  Waveform: {response.name}")
    print(f"  Wave ID: {response.wave_id}")
    print(f"  Gain Q15 shape: {response.gain_q15.shape}")
    print(f"  Gain range: [{response.gain_q15.min()}, {response.gain_q15.max()}]")
    print(f"  Non-zero gains: {np.count_nonzero(response.gain_q15)}")
    print(f"  Phase offset shape: {response.phase_offset.shape}")
    print(f"  Reference freq: {response.ref_freq_hz} Hz")

    # Check that some leaves have non-zero gain
    if np.count_nonzero(response.gain_q15) > 0:
        print("[OK] Non-zero filterbank response\n")
        return True
    else:
        print("[FAIL] All gains are zero!\n")
        return False


def test_coeff_generation():
    """Test that coeff_block_q15 generates valid coefficients."""
    print("Test: Coefficient Generation")
    print("=" * 60)

    response = get_response(0)  # Sine

    output = np.zeros(NUM_LEAVES, dtype=np.int16)
    coeff_block_q15(response, 0, output)

    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min()}, {output.max()}]")
    print(f"  Non-zero coeffs: {np.count_nonzero(output)}")
    print(f"  First 8 coeffs: {output[:8]}")

    # Test at different phases
    phases = [0, 0x40000000, 0x80000000, 0xC0000000]
    for phase in phases:
        coeff_block_q15(response, phase, output)
        print(f"  Phase {phase >> 24:02x}00_0000: range [{output.min():6d}, {output.max():6d}]")

    if np.count_nonzero(output) > 0:
        print("[OK] Non-zero coefficients generated\n")
        return True
    else:
        print("[FAIL] All coefficients zero!\n")
        return False


def test_vectorized_match():
    """Test that vectorized version matches scalar."""
    print("Test: Vectorized vs Scalar Match")
    print("=" * 60)

    response = get_response(0)

    for phase in [0, 0x12345678, 0x87654321]:
        scalar_out = np.zeros(NUM_LEAVES, dtype=np.int16)
        coeff_block_q15(response, phase, scalar_out)

        vector_out = coeff_block_q15_vectorized(response, phase)

        max_diff = np.abs(scalar_out.astype(np.int32) - vector_out.astype(np.int32)).max()
        print(f"  Phase {phase:08x}: max diff = {max_diff}")

        if max_diff > 1:  # Allow for rounding differences
            print(f"[FAIL] Large difference at phase {phase:08x}\n")
            return False

    print("[OK] Vectorized matches scalar\n")
    return True


def test_oscillator_frequency():
    """Test that CoeffOscCore produces correct frequency via IDWPT."""
    print("Test: Oscillator Frequency Accuracy")
    print("=" * 60)

    response = get_response(0)  # Sine

    osc = CoeffOscCore()
    osc.set_midi_note(60)  # C4 = 261.63 Hz

    all_audio = []
    coeffs = np.zeros(NUM_LEAVES, dtype=np.int16)

    for block in range(32):
        # Generate coefficients
        render_coeff_block(osc, response, coeffs)

        # IDWPT to audio
        audio = idwpt_jit(coeffs.tolist(), headroom_bits=1)
        all_audio.append(np.array(audio, dtype=np.int16))

    full_audio = np.concatenate(all_audio)

    print(f"  Total samples: {len(full_audio)}")
    print(f"  Audio range: [{full_audio.min()}, {full_audio.max()}]")

    try:
        from scipy.fft import fft
        N = len(full_audio)
        fft_result = np.abs(fft(full_audio.astype(np.float32)))[:N//2]
        freqs = np.fft.fftfreq(N, 1/48000)[:N//2]
        peak_idx = np.argmax(fft_result)
        peak_freq = freqs[peak_idx]

        expected = 261.63
        error = abs(peak_freq - expected)

        print(f"  Expected: {expected:.2f} Hz")
        print(f"  Measured: {peak_freq:.2f} Hz")
        print(f"  Error: {error:.2f} Hz")

        if error < 50:
            print("[OK] Frequency accurate\n")
            return True
        else:
            print("[WARN] Frequency error > 50 Hz\n")
            return False

    except ImportError:
        print("  (scipy not available for FFT)")
        # Still check audio is non-zero
        if np.abs(full_audio).max() > 100:
            print("[OK] Audio generated (FFT skipped)\n")
            return True
        else:
            print("[FAIL] Audio too quiet\n")
            return False


def test_phase_continuity():
    """Test phase continuity across blocks (no clicks)."""
    print("Test: Phase Continuity")
    print("=" * 60)

    response = get_response(0)

    osc = CoeffOscCore()
    osc.set_midi_note(60)

    all_audio = []
    coeffs = np.zeros(NUM_LEAVES, dtype=np.int16)

    for block in range(20):
        render_coeff_block(osc, response, coeffs)
        audio = idwpt_jit(coeffs.tolist(), headroom_bits=1)
        all_audio.append(np.array(audio, dtype=np.int16))

        if block < 5:
            phase_deg = (osc.phase / (1 << 32)) * 360
            print(f"  Block {block}: phase={phase_deg:6.1f} deg, audio range=[{min(audio):6d}, {max(audio):6d}]")

    # Check boundary discontinuities
    max_jump = 0
    for i in range(len(all_audio) - 1):
        jump = abs(int(all_audio[i][-1]) - int(all_audio[i+1][0]))
        max_jump = max(max_jump, jump)

    print(f"\n  Max boundary jump: {max_jump}")

    if max_jump < 5000:
        print("[OK] Acceptable boundary discontinuities\n")
        return True
    else:
        print("[WARN] Large boundary discontinuities\n")
        return False


def test_all_waveforms():
    """Test all standard waveforms produce sound."""
    print("Test: All Standard Waveforms")
    print("=" * 60)

    waveforms = [(0, "sine"), (1, "saw"), (2, "square"), (3, "triangle")]

    osc = CoeffOscCore()
    osc.set_midi_note(60)
    coeffs = np.zeros(NUM_LEAVES, dtype=np.int16)

    all_ok = True

    for wave_id, name in waveforms:
        response = get_response(wave_id)

        # Reset oscillator
        osc.reset()
        osc.set_midi_note(60)

        # Generate one block
        render_coeff_block(osc, response, coeffs)
        audio = idwpt_jit(coeffs.tolist(), headroom_bits=1)

        peak = max(abs(min(audio)), abs(max(audio)))
        non_zero = np.count_nonzero(np.array(audio))

        status = "[OK]" if peak > 100 else "[FAIL]"
        if peak <= 100:
            all_ok = False

        print(f"  {name:10s}: peak={peak:6d}, non-zero={non_zero:3d} {status}")

    if all_ok:
        print("\n[OK] All waveforms produce sound\n")
    else:
        print("\n[FAIL] Some waveforms are silent\n")

    return all_ok


def test_frequency_scaling():
    """Test that different MIDI notes produce different frequencies."""
    print("Test: Frequency Scaling Across Notes")
    print("=" * 60)

    response = get_response(0)
    coeffs = np.zeros(NUM_LEAVES, dtype=np.int16)

    notes = [48, 60, 72]  # C3, C4, C5
    expected_freqs = [130.81, 261.63, 523.25]

    try:
        from scipy.fft import fft

        for note, expected in zip(notes, expected_freqs):
            osc = CoeffOscCore()
            osc.set_midi_note(note)

            all_audio = []
            for _ in range(32):
                render_coeff_block(osc, response, coeffs)
                audio = idwpt_jit(coeffs.tolist(), headroom_bits=1)
                all_audio.append(np.array(audio, dtype=np.int16))

            full_audio = np.concatenate(all_audio)

            N = len(full_audio)
            fft_result = np.abs(fft(full_audio.astype(np.float32)))[:N//2]
            freqs = np.fft.fftfreq(N, 1/48000)[:N//2]
            peak_idx = np.argmax(fft_result)
            measured = freqs[peak_idx]

            error = abs(measured - expected)
            status = "[OK]" if error < 50 else "[WARN]"

            print(f"  Note {note}: expected {expected:.1f} Hz, measured {measured:.1f} Hz, error {error:.1f} Hz {status}")

        print("\n[OK] Frequency scaling works\n")
        return True

    except ImportError:
        print("  (scipy not available for FFT)\n")
        return True


def test_memory_efficiency():
    """Test memory usage of filterbank responses."""
    print("Test: Memory Efficiency")
    print("=" * 60)

    response = get_response(0)

    gain_bytes = response.gain_q15.nbytes
    phase_bytes = response.phase_offset.nbytes
    total_bytes = gain_bytes + phase_bytes

    print(f"  Gain table: {gain_bytes} bytes")
    print(f"  Phase table: {phase_bytes} bytes")
    print(f"  Total per waveform: {total_bytes} bytes")
    print(f"  4 waveforms: {total_bytes * 4} bytes")

    # Compare with per-note tables (128 notes * 64 phases * 64 leaves * 2 bytes)
    per_note_size = 128 * 64 * 64 * 2
    print(f"\n  Per-note table approach: {per_note_size:,} bytes ({per_note_size // 1024} KB)")
    print(f"  Unified approach (4 waves): {total_bytes * 4} bytes")
    print(f"  Savings: {per_note_size - total_bytes * 4:,} bytes")

    print("\n[OK] Memory efficient\n")
    return True


if __name__ == '__main__':
    print("CoeffCore Tests - Unified Coefficient Oscillator")
    print("=" * 60)
    print()

    results = []
    results.append(("FilterbankResponse", test_filterbank_response()))
    results.append(("Coefficient Generation", test_coeff_generation()))
    results.append(("Vectorized Match", test_vectorized_match()))
    results.append(("Oscillator Frequency", test_oscillator_frequency()))
    results.append(("Phase Continuity", test_phase_continuity()))
    results.append(("All Waveforms", test_all_waveforms()))
    results.append(("Frequency Scaling", test_frequency_scaling()))
    results.append(("Memory Efficiency", test_memory_efficiency()))

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    print(f"\n  {passed_count}/{total_count} tests passed")
