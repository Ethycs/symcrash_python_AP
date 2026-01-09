"""
Sample Playback Test Suite
==========================

Validates MODE_SAMPLE via FWPT gateway (Phase 3).

Tests:
1. Basic sample loading and playback
2. Pitch shifting
3. Looping
4. Pre-computed coefficient mode
5. Integration with CCS path
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from symcrash_python_ap.vm.sample_source import (
    SampleOscState, read_sample_block, read_coeff_block,
    sample_generate_coeffs, compute_playback_rate, create_hann_window,
    BLOCK_SIZE, VM_K, NUM_LEAVES, COEFF_LEN
)
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit

SAMPLE_RATE = 48000


def generate_sine_buffer(freq_hz: float, duration_sec: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a sine wave test buffer."""
    num_samples = int(duration_sec * sample_rate)
    t = np.arange(num_samples) / sample_rate
    audio = (np.sin(2 * np.pi * freq_hz * t) * 32000).astype(np.int16)
    return audio


def test_basic_sample_loading():
    """Test basic sample buffer loading."""
    print("\n=== Basic Sample Loading Test ===\n")

    state = SampleOscState()

    # Create a simple test buffer
    buffer = generate_sine_buffer(440.0, 0.1)  # 100ms of A4
    print(f"Buffer size: {len(buffer)} samples")

    state.set_buffer(buffer)

    assert state.buffer is not None
    assert len(state.buffer) == len(buffer)
    assert state.is_playing == True
    assert state.is_finished == False
    assert state.position_q16 == 0

    print("[PASS] Basic sample loading works")
    return True


def test_sample_playback():
    """Test basic sample playback (audio mode)."""
    print("\n=== Sample Playback Test ===\n")

    state = SampleOscState()

    # Create test buffer
    freq = 440.0
    buffer = generate_sine_buffer(freq, 0.1)  # 100ms
    state.set_buffer(buffer)

    # Read a few blocks
    blocks = []
    for _ in range(10):
        block = read_sample_block(state, BLOCK_SIZE)
        blocks.append(block)
        if state.is_finished:
            break

    audio = np.concatenate(blocks)
    print(f"Played: {len(audio)} samples")

    # Verify audio has content
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    print(f"RMS: {rms:.1f}")

    assert rms > 1000, f"Audio too quiet: RMS={rms}"

    print("[PASS] Sample playback works")
    return True


def test_pitch_shifting():
    """Test pitch shifting via rate adjustment."""
    print("\n=== Pitch Shifting Test ===\n")

    state = SampleOscState()

    # Create 440 Hz buffer - make it shorter for faster test
    buffer = generate_sine_buffer(440.0, 0.2)  # 200ms = 9600 samples
    state.set_buffer(buffer)
    print(f"Buffer: {len(buffer)} samples")

    # Pitch up 12 semitones (2x rate = octave up)
    state.set_pitch_semitones(12)
    assert state.rate_q16 == 131072, f"Rate should be 2.0 (131072), got {state.rate_q16}"

    # Read blocks at 2x speed until finished
    samples_played = 0
    max_blocks = 200
    for _ in range(max_blocks):
        block = read_sample_block(state, BLOCK_SIZE)
        samples_played += len(block)
        if state.is_finished:
            break

    print(f"Played until finished: {samples_played} samples")

    # At 2x rate, we consume buffer samples twice as fast
    # So we produce ~half as many output samples before buffer ends
    expected_output = len(buffer) // 2
    print(f"Expected output samples: ~{expected_output}")

    ratio = samples_played / expected_output
    print(f"Output ratio: {ratio:.2f} (expected ~1.0)")

    # Allow tolerance for block alignment
    assert 0.8 < ratio < 1.3, f"Pitch shift timing wrong: ratio={ratio}"

    # Verify we finished
    assert state.is_finished, "Should have finished playback"

    print("[PASS] Pitch shifting works")
    return True


def test_looping():
    """Test sample looping."""
    print("\n=== Looping Test ===\n")

    state = SampleOscState()

    # Create short buffer with loop
    buffer = generate_sine_buffer(440.0, 0.05)  # 50ms = 2400 samples
    loop_start = 0
    loop_end = len(buffer)

    state.set_buffer(buffer, loop_start=loop_start, loop_end=loop_end, loop_enabled=True)

    # Read more samples than buffer contains
    blocks = []
    for _ in range(100):  # 100 blocks = 6400 samples
        block = read_sample_block(state, BLOCK_SIZE)
        blocks.append(block)

    audio = np.concatenate(blocks)
    print(f"Played with looping: {len(audio)} samples from {len(buffer)} sample buffer")

    # Should not finish (looping)
    assert state.is_playing == True, "Should still be playing (looping)"
    assert not state.is_finished, "Should not be finished (looping)"

    # Audio should have content throughout
    rms_start = np.sqrt(np.mean(audio[:1000].astype(np.float64) ** 2))
    rms_end = np.sqrt(np.mean(audio[-1000:].astype(np.float64) ** 2))
    print(f"RMS start: {rms_start:.1f}, RMS end: {rms_end:.1f}")

    assert rms_end > 1000, f"Loop audio too quiet: RMS={rms_end}"

    print("[PASS] Looping works")
    return True


def test_precomputed_coefficients():
    """Test pre-computed coefficient mode."""
    print("\n=== Pre-computed Coefficients Test ===\n")

    state = SampleOscState()

    # Create buffer and precompute coefficients
    buffer = generate_sine_buffer(440.0, 0.2)  # 200ms
    num_expected_blocks = len(buffer) // BLOCK_SIZE

    state.set_buffer(buffer, precompute_coeffs=True)

    assert state.use_coeff_buffer == True
    assert state.coeff_buffer is not None
    print(f"Coeff buffer shape: {state.coeff_buffer.shape}")
    assert state.coeff_buffer.shape[0] == num_expected_blocks
    assert state.coeff_buffer.shape[1] == NUM_LEAVES

    # Read coefficient blocks
    output_512 = np.zeros(COEFF_LEN, dtype=np.int16)
    blocks_read = 0

    while state.is_playing and blocks_read < 20:
        read_coeff_block(state, output_512)
        blocks_read += 1

        # Verify coefficients have content
        if blocks_read == 1:
            rms = np.sqrt(np.mean(output_512.astype(np.float64) ** 2))
            print(f"First block coeff RMS: {rms:.1f}")
            assert rms > 100, f"Coefficients too quiet"

    print(f"Read {blocks_read} coefficient blocks")

    print("[PASS] Pre-computed coefficients work")
    return True


def test_coeff_interpolation():
    """Test coefficient interpolation for pitch shifting."""
    print("\n=== Coefficient Interpolation Test ===\n")

    state = SampleOscState()

    # Create buffer and precompute
    buffer = generate_sine_buffer(440.0, 0.5)
    state.set_buffer(buffer, precompute_coeffs=True)

    # Set 1.5x playback rate (pitch up ~7 semitones)
    state.rate_q16 = int(1.5 * 65536)

    # Read blocks
    output_512 = np.zeros(COEFF_LEN, dtype=np.int16)
    blocks_read = 0
    coeff_values = []

    while state.is_playing and blocks_read < 30:
        read_coeff_block(state, output_512)
        # Store DC coefficient as reference
        coeff_values.append(output_512[0])
        blocks_read += 1

    print(f"Read {blocks_read} blocks at 1.5x rate")
    print(f"Block position at end: {state.block_position_q16 >> 16}")

    # At 1.5x rate, should advance faster through buffer
    expected_position = int(blocks_read * 1.5)
    actual_position = state.block_position_q16 >> 16

    print(f"Expected position: ~{expected_position}, Actual: {actual_position}")

    # Allow some tolerance
    assert abs(actual_position - expected_position) < 2, "Position advancement wrong"

    print("[PASS] Coefficient interpolation works")
    return True


def test_sample_generate_coeffs():
    """Test the main sample_generate_coeffs function."""
    print("\n=== sample_generate_coeffs Test ===\n")

    state = SampleOscState()

    # Test audio mode (no precompute)
    buffer = generate_sine_buffer(440.0, 0.1)
    state.set_buffer(buffer, precompute_coeffs=False)

    output_512 = np.zeros(COEFF_LEN, dtype=np.int16)
    sample_generate_coeffs(state, output_512)

    # Check output has content
    rms = np.sqrt(np.mean(output_512.astype(np.float64) ** 2))
    print(f"Audio mode coeff RMS: {rms:.1f}")
    assert rms > 100, f"Audio mode output too quiet: {rms}"

    # Test coeff mode
    state2 = SampleOscState()
    state2.set_buffer(buffer, precompute_coeffs=True)

    output_512_coeff = np.zeros(COEFF_LEN, dtype=np.int16)
    sample_generate_coeffs(state2, output_512_coeff)

    rms_coeff = np.sqrt(np.mean(output_512_coeff.astype(np.float64) ** 2))
    print(f"Coeff mode coeff RMS: {rms_coeff:.1f}")
    assert rms_coeff > 100, f"Coeff mode output too quiet: {rms_coeff}"

    print("[PASS] sample_generate_coeffs works in both modes")
    return True


def test_round_trip_quality():
    """Test FWPT -> IDWPT round-trip quality."""
    print("\n=== Round-Trip Quality Test ===\n")

    # Generate a test signal
    freq = 440.0
    audio = generate_sine_buffer(freq, 0.01)[:BLOCK_SIZE]  # One block

    # FWPT
    coeffs = fwpt_jit(audio.tolist(), headroom_bits=1)

    # IDWPT
    reconstructed = np.array(idwpt_jit(coeffs, headroom_bits=1), dtype=np.int16)

    # Measure error
    error = audio.astype(np.float64) - reconstructed.astype(np.float64)
    max_error = np.max(np.abs(error))
    rms_error = np.sqrt(np.mean(error ** 2))
    signal_rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))

    snr_db = 20 * np.log10(signal_rms / rms_error) if rms_error > 0 else float('inf')

    print(f"Max error: {max_error:.1f}")
    print(f"RMS error: {rms_error:.1f}")
    print(f"Signal RMS: {signal_rms:.1f}")
    print(f"SNR: {snr_db:.1f} dB")

    # Should have very low error (round-trip should be ~lossless for 16-bit)
    assert snr_db > 60, f"Round-trip SNR too low: {snr_db} dB"

    print("[PASS] Round-trip quality is good")
    return True


def test_compute_playback_rate():
    """Test playback rate computation."""
    print("\n=== Playback Rate Computation Test ===\n")

    # Test cases: (original_note, target_note, expected_ratio)
    test_cases = [
        (60, 60, 1.0),       # Same note = 1.0x
        (60, 72, 2.0),       # Octave up = 2.0x
        (60, 48, 0.5),       # Octave down = 0.5x
        (69, 81, 2.0),       # A4 to A5 = 2.0x
        (60, 61, 2 ** (1/12)),  # Semitone up
    ]

    for orig, target, expected in test_cases:
        rate_q16 = compute_playback_rate(orig, target)
        actual_ratio = rate_q16 / 65536

        error = abs(actual_ratio - expected) / expected * 100
        print(f"Note {orig} -> {target}: ratio = {actual_ratio:.4f} (expected {expected:.4f}, error {error:.2f}%)")

        assert error < 0.01, f"Rate computation error too large: {error}%"

    print("[PASS] Playback rate computation works")
    return True


def test_one_shot_vs_loop():
    """Test one-shot vs loop playback."""
    print("\n=== One-Shot vs Loop Test ===\n")

    buffer = generate_sine_buffer(440.0, 0.05)  # 50ms

    # One-shot mode
    state_oneshot = SampleOscState()
    state_oneshot.set_buffer(buffer, loop_enabled=False)

    blocks = 0
    while state_oneshot.is_playing and blocks < 100:
        read_sample_block(state_oneshot, BLOCK_SIZE)
        blocks += 1

    print(f"One-shot: finished after {blocks} blocks, is_finished={state_oneshot.is_finished}")
    assert state_oneshot.is_finished, "One-shot should finish"

    # Loop mode
    state_loop = SampleOscState()
    state_loop.set_buffer(buffer, loop_enabled=True)

    for _ in range(100):
        read_sample_block(state_loop, BLOCK_SIZE)

    print(f"Loop: still playing after 100 blocks, is_finished={state_loop.is_finished}")
    assert not state_loop.is_finished, "Loop should not finish"
    assert state_loop.is_playing, "Loop should still be playing"

    print("[PASS] One-shot vs loop modes work correctly")
    return True


def main():
    print("=" * 60)
    print("Sample Playback Test Suite (MODE_SAMPLE)")
    print("=" * 60)

    results = []

    results.append(("Basic Sample Loading", test_basic_sample_loading()))
    results.append(("Sample Playback", test_sample_playback()))
    results.append(("Pitch Shifting", test_pitch_shifting()))
    results.append(("Looping", test_looping()))
    results.append(("Pre-computed Coefficients", test_precomputed_coefficients()))
    results.append(("Coefficient Interpolation", test_coeff_interpolation()))
    results.append(("sample_generate_coeffs", test_sample_generate_coeffs()))
    results.append(("Round-Trip Quality", test_round_trip_quality()))
    results.append(("Playback Rate Computation", test_compute_playback_rate()))
    results.append(("One-Shot vs Loop", test_one_shot_vs_loop()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        all_passed = all_passed and passed

    print()
    if all_passed:
        print("[PASS] All sample playback tests passed!")
        return 0
    else:
        print("[FAIL] Some sample playback tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
