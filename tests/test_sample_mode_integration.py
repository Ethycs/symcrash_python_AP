"""
Sample Mode Integration Test Suite
==================================

Validates MODE_SAMPLE integration with the CCS hot path.

Tests:
1. Sample mode activation via VoiceFrame
2. Pitch shifting based on root note
3. Sample playback through executor
4. Integration with coefficient operators
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.state import VMState, OSC_MODE_SAMPLE
from symcrash_python_ap.vm.sample_source import SampleOscState, BLOCK_SIZE
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit

SAMPLE_RATE = 48000


def generate_sine_buffer(freq_hz: float, duration_sec: float) -> np.ndarray:
    """Generate a sine wave test buffer."""
    num_samples = int(duration_sec * SAMPLE_RATE)
    t = np.arange(num_samples) / SAMPLE_RATE
    audio = (np.sin(2 * np.pi * freq_hz * t) * 28000).astype(np.int16)
    return audio


def test_sample_mode_activation():
    """Test that coeff_osc_mode=2 activates sample mode."""
    print("\n=== Sample Mode Activation Test ===\n")

    # Create voice state
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=60)
    state.use_coeff_osc = True  # Enable CCS path

    # Load a sample into sample_state
    buffer = generate_sine_buffer(440.0, 0.1)  # 100ms of A4
    state.sample_state.set_buffer(buffer, root_note=69)  # A4 = MIDI 69

    # Create frame with sample mode
    frame = VoiceFrame(
        note=60,  # C4
        coeff_osc_mode=OSC_MODE_SAMPLE,
    )

    # Run one block
    state, output = vm_step_block(state, frame)

    # Output should have content
    rms = np.sqrt(np.mean(output.astype(np.float64) ** 2))
    print(f"Output RMS: {rms:.1f}")

    assert rms > 100, f"Sample mode output too quiet: {rms}"
    print("[PASS] Sample mode activation works")
    return True


def test_sample_pitch_shifting():
    """Test automatic pitch shifting based on root note."""
    print("\n=== Sample Pitch Shifting Test ===\n")

    # Create sample at A4 (440 Hz)
    buffer = generate_sine_buffer(440.0, 0.5)

    # Test 1: Play at root note (should be 1.0x rate)
    state1 = VMState.create(voice_id=0)
    state1.reset_for_note_on(note=69)  # A4
    state1.use_coeff_osc = True
    state1.sample_state.set_buffer(buffer.copy(), root_note=69)

    frame1 = VoiceFrame(note=69, coeff_osc_mode=OSC_MODE_SAMPLE)
    state1, _ = vm_step_block(state1, frame1)

    rate_at_root = state1.sample_state.rate_q16
    print(f"Rate at root note (A4): {rate_at_root / 65536:.4f}x")
    assert rate_at_root == 65536, f"Rate at root should be 1.0x, got {rate_at_root}"

    # Test 2: Play octave up (should be 2.0x rate)
    state2 = VMState.create(voice_id=1)
    state2.reset_for_note_on(note=81)  # A5
    state2.use_coeff_osc = True
    state2.sample_state.set_buffer(buffer.copy(), root_note=69)

    frame2 = VoiceFrame(note=81, coeff_osc_mode=OSC_MODE_SAMPLE)
    state2, _ = vm_step_block(state2, frame2)

    rate_octave_up = state2.sample_state.rate_q16
    print(f"Rate at octave up (A5): {rate_octave_up / 65536:.4f}x")
    assert abs(rate_octave_up - 131072) < 100, f"Rate octave up should be ~2.0x, got {rate_octave_up}"

    # Test 3: Play octave down (should be 0.5x rate)
    state3 = VMState.create(voice_id=2)
    state3.reset_for_note_on(note=57)  # A3
    state3.use_coeff_osc = True
    state3.sample_state.set_buffer(buffer.copy(), root_note=69)

    frame3 = VoiceFrame(note=57, coeff_osc_mode=OSC_MODE_SAMPLE)
    state3, _ = vm_step_block(state3, frame3)

    rate_octave_down = state3.sample_state.rate_q16
    print(f"Rate at octave down (A3): {rate_octave_down / 65536:.4f}x")
    assert abs(rate_octave_down - 32768) < 100, f"Rate octave down should be ~0.5x, got {rate_octave_down}"

    print("[PASS] Sample pitch shifting works")
    return True


def test_sample_playback_through_executor():
    """Test full sample playback through executor."""
    print("\n=== Sample Playback Through Executor Test ===\n")

    # Create a longer sample
    buffer = generate_sine_buffer(440.0, 0.2)  # 200ms

    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=69)
    state.use_coeff_osc = True
    state.sample_state.set_buffer(buffer, root_note=69, loop_enabled=False)

    frame = VoiceFrame(
        note=69,
        coeff_osc_mode=OSC_MODE_SAMPLE,
    )

    # Generate blocks until sample finishes
    outputs = []
    max_blocks = 200

    for _ in range(max_blocks):
        state, output = vm_step_block(state, frame)
        outputs.append(output.copy())

        if state.sample_state.is_finished:
            break

    audio = np.concatenate(outputs)
    print(f"Generated {len(outputs)} blocks, {len(audio)} samples")

    # Analyze output
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    peak = np.max(np.abs(audio))

    print(f"Output RMS: {rms:.1f}")
    print(f"Output peak: {peak}")

    assert rms > 500, f"Output too quiet: {rms}"
    assert state.sample_state.is_finished, "Sample should have finished"

    print("[PASS] Sample playback through executor works")
    return True


def test_sample_with_loop():
    """Test looped sample playback."""
    print("\n=== Sample Loop Test ===\n")

    # Create a short sample that will loop
    buffer = generate_sine_buffer(440.0, 0.05)  # 50ms

    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=69)
    state.use_coeff_osc = True
    state.sample_state.set_buffer(buffer, root_note=69, loop_enabled=True)

    frame = VoiceFrame(
        note=69,
        coeff_osc_mode=OSC_MODE_SAMPLE,
    )

    # Generate more blocks than the sample contains
    outputs = []
    for _ in range(100):
        state, output = vm_step_block(state, frame)
        outputs.append(output.copy())

    audio = np.concatenate(outputs)
    print(f"Generated {len(outputs)} blocks from {len(buffer)} sample buffer")

    # Should not be finished (looping)
    assert not state.sample_state.is_finished, "Looped sample should not finish"
    assert state.sample_state.is_playing, "Looped sample should still be playing"

    # Should have content throughout
    rms_start = np.sqrt(np.mean(audio[:1000].astype(np.float64) ** 2))
    rms_end = np.sqrt(np.mean(audio[-1000:].astype(np.float64) ** 2))

    print(f"RMS start: {rms_start:.1f}, end: {rms_end:.1f}")
    assert rms_end > 100, f"Loop end too quiet: {rms_end}"

    print("[PASS] Sample loop works")
    return True


def test_sample_with_precomputed_coeffs():
    """Test precomputed coefficient mode."""
    print("\n=== Precomputed Coefficients Test ===\n")

    buffer = generate_sine_buffer(440.0, 0.2)

    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=69)
    state.use_coeff_osc = True
    state.sample_state.set_buffer(
        buffer, root_note=69, precompute_coeffs=True, loop_enabled=True
    )

    assert state.sample_state.use_coeff_buffer, "Should use coefficient buffer"
    assert state.sample_state.coeff_buffer is not None, "Coeff buffer should exist"

    frame = VoiceFrame(
        note=69,
        coeff_osc_mode=OSC_MODE_SAMPLE,
    )

    # Generate blocks
    outputs = []
    for _ in range(50):
        state, output = vm_step_block(state, frame)
        outputs.append(output.copy())

    audio = np.concatenate(outputs)
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))

    print(f"Precomputed coeff mode RMS: {rms:.1f}")
    assert rms > 500, f"Precomputed mode too quiet: {rms}"

    print("[PASS] Precomputed coefficients work")
    return True


def test_sample_rate_override():
    """Test manual sample_rate_q16 override from VoiceFrame."""
    print("\n=== Sample Rate Override Test ===\n")

    buffer = generate_sine_buffer(440.0, 0.2)

    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=69)
    state.use_coeff_osc = True
    # No root_note - so manual rate should be used
    state.sample_state.set_buffer(buffer)

    # Override with 1.5x rate
    frame = VoiceFrame(
        note=69,
        coeff_osc_mode=OSC_MODE_SAMPLE,
        sample_rate_q16=98304,  # 1.5x
    )

    state, _ = vm_step_block(state, frame)

    # Rate should be the override value since no root_note
    rate = state.sample_state.rate_q16
    print(f"Rate with override: {rate / 65536:.4f}x")
    assert rate == 98304, f"Rate should be 1.5x override, got {rate}"

    print("[PASS] Sample rate override works")
    return True


def main():
    print("=" * 60)
    print("Sample Mode Integration Test Suite")
    print("=" * 60)

    results = []

    results.append(("Sample Mode Activation", test_sample_mode_activation()))
    results.append(("Sample Pitch Shifting", test_sample_pitch_shifting()))
    results.append(("Sample Playback Through Executor", test_sample_playback_through_executor()))
    results.append(("Sample Loop", test_sample_with_loop()))
    results.append(("Precomputed Coefficients", test_sample_with_precomputed_coeffs()))
    results.append(("Sample Rate Override", test_sample_rate_override()))

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
        print("[PASS] All sample mode integration tests passed!")
        return 0
    else:
        print("[FAIL] Some sample mode integration tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
