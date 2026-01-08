"""Quick test for Sample Playback - Phase 3."""
from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.state import VMState, OSC_MODE_SAMPLE
from symcrash_python_ap.frames import VoiceFrame
import numpy as np


def test_sample_playback():
    """Test sample playback via FWPT gateway (Phase 3)."""
    # Create a simple sine wave sample buffer
    sample_rate = 48000
    duration_samples = 4800  # 100ms
    freq = 440.0
    t = np.arange(duration_samples) / sample_rate
    sample_buffer = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)

    # Create voice state and load sample buffer
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=60)

    # Load sample into sample_state (this would normally be done by Voice Engine)
    state.sample_state.set_buffer(sample_buffer, loop_enabled=False)

    # Create frame for sample playback mode
    frame = VoiceFrame(
        note=60,
        coeff_osc_mode=2,  # OSC_MODE_SAMPLE
    )

    # Run several blocks
    outputs = []
    for _ in range(10):
        state, output = vm_step_block(state, frame)
        outputs.append(output.copy())

    # Verify output is non-zero
    rms = np.sqrt(np.mean(outputs[0].astype(float)**2))
    assert rms > 0, "Sample playback should produce output"

    # Verify sample state advances
    assert state.sample_state.position_q16 > 0, "Sample position should advance"


def test_sample_pitch_shift():
    """Test pitch-shifted sample playback."""
    # Create sample buffer
    sample_rate = 48000
    duration_samples = 9600  # 200ms
    freq = 440.0
    t = np.arange(duration_samples) / sample_rate
    sample_buffer = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)

    # Test 1: Normal speed
    state1 = VMState.create(voice_id=0)
    state1.sample_state.set_buffer(sample_buffer.copy())

    # Test 2: Double speed (octave up)
    state2 = VMState.create(voice_id=1)
    state2.sample_state.set_buffer(sample_buffer.copy())
    state2.sample_state.set_pitch_ratio(2.0)  # Octave up

    frame = VoiceFrame(note=60, coeff_osc_mode=2)

    # Run 10 blocks
    for _ in range(10):
        state1, _ = vm_step_block(state1, frame)
        state2, _ = vm_step_block(state2, frame)

    # Double-speed should have advanced twice as far
    pos1 = state1.sample_state.position_q16
    pos2 = state2.sample_state.position_q16

    # Allow some tolerance for fixed-point math
    ratio = pos2 / pos1 if pos1 > 0 else 0
    assert 1.9 < ratio < 2.1, f"Double speed should advance ~2x: {ratio}"


def test_sample_loop():
    """Test looped sample playback."""
    # Create short sample buffer
    sample_rate = 48000
    duration_samples = 256  # Very short for testing loops
    freq = 880.0
    t = np.arange(duration_samples) / sample_rate
    sample_buffer = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)

    # Enable looping
    state = VMState.create(voice_id=0)
    state.sample_state.set_buffer(sample_buffer, loop_enabled=True)

    frame = VoiceFrame(note=60, coeff_osc_mode=2)

    # Run enough blocks to loop multiple times
    outputs = []
    for _ in range(20):
        state, output = vm_step_block(state, frame)
        outputs.append(output.copy())

    # Should still be playing (not finished) due to looping
    assert state.sample_state.is_playing, "Looped sample should keep playing"
    assert not state.sample_state.is_finished, "Looped sample should not finish"

    # All outputs should be non-zero
    for i, out in enumerate(outputs):
        rms = np.sqrt(np.mean(out.astype(float)**2))
        assert rms > 0, f"Block {i} should have non-zero output"


def test_coeff_mode_loop():
    """Test coefficient-mode looping (no FWPT in hot path)."""
    # Create short sample buffer (must be multiple of 64 for clean block alignment)
    sample_rate = 48000
    duration_samples = 256  # 4 blocks
    freq = 880.0
    t = np.arange(duration_samples) / sample_rate
    sample_buffer = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)

    # Load with precompute_coeffs=True for coefficient-mode looping
    state = VMState.create(voice_id=0)
    state.sample_state.set_buffer(
        sample_buffer,
        loop_enabled=True,
        precompute_coeffs=True  # Key: pre-transform to coefficients
    )

    # Verify coefficient buffer was created
    assert state.sample_state.use_coeff_buffer, "Should be in coefficient mode"
    assert state.sample_state.coeff_buffer is not None, "Coeff buffer should exist"
    assert state.sample_state.coeff_buffer.shape[0] == 4, "Should have 4 blocks (256/64)"

    frame = VoiceFrame(note=60, coeff_osc_mode=2)

    # Run enough blocks to loop multiple times (20 blocks = 5x through 4-block sample)
    outputs = []
    for _ in range(20):
        state, output = vm_step_block(state, frame)
        outputs.append(output.copy())

    # Should still be playing (not finished) due to looping
    assert state.sample_state.is_playing, "Coeff-mode looped sample should keep playing"
    assert not state.sample_state.is_finished, "Coeff-mode looped sample should not finish"

    # All outputs should be non-zero
    for i, out in enumerate(outputs):
        rms = np.sqrt(np.mean(out.astype(float)**2))
        assert rms > 0, f"Block {i} should have non-zero output"

    # Verify block position wraps correctly
    # block_position_q16 >> 16 gives the integer block index
    # After 20 iterations at 1.0x rate on 4-block loop: 20 % 4 = 0, but position
    # may be at boundary (4) before next read wraps it
    block_int = state.sample_state.block_position_q16 >> 16
    assert block_int <= 4, f"Block position should wrap: {block_int}"


def test_coeff_mode_pitch_shift():
    """Test pitch shifting in coefficient mode (block interpolation)."""
    sample_rate = 48000
    duration_samples = 640  # 10 blocks for more room
    freq = 440.0
    t = np.arange(duration_samples) / sample_rate
    sample_buffer = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)

    # Test 1: Normal speed in coeff mode
    state1 = VMState.create(voice_id=0)
    state1.sample_state.set_buffer(sample_buffer.copy(), precompute_coeffs=True)
    assert state1.sample_state.use_coeff_buffer, "Should be in coefficient mode"

    # Test 2: Double speed in coeff mode (should stay in coeff mode!)
    state2 = VMState.create(voice_id=1)
    state2.sample_state.set_buffer(sample_buffer.copy(), precompute_coeffs=True)
    state2.sample_state.set_pitch_ratio(2.0)  # Octave up
    assert state2.sample_state.use_coeff_buffer, "Should stay in coefficient mode with pitch shift"

    frame = VoiceFrame(note=60, coeff_osc_mode=2)

    # Run 5 blocks
    for _ in range(5):
        state1, _ = vm_step_block(state1, frame)
        state2, _ = vm_step_block(state2, frame)

    # Both should produce output
    assert state1.sample_state.is_playing, "Normal speed should be playing"
    assert state2.sample_state.is_playing, "Double speed should be playing"

    # Double-speed should have advanced twice as far in block position
    pos1 = state1.sample_state.block_position_q16
    pos2 = state2.sample_state.block_position_q16

    # Allow some tolerance for fixed-point math
    ratio = pos2 / pos1 if pos1 > 0 else 0
    assert 1.9 < ratio < 2.1, f"Double speed should advance ~2x in blocks: {ratio}"


def test_coeff_mode_pitch_down():
    """Test pitch down (rate < 1.0) in coefficient mode."""
    sample_rate = 48000
    duration_samples = 640  # 10 blocks
    freq = 440.0
    t = np.arange(duration_samples) / sample_rate
    sample_buffer = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)

    # Half speed (octave down)
    state = VMState.create(voice_id=0)
    state.sample_state.set_buffer(sample_buffer.copy(), precompute_coeffs=True)
    state.sample_state.set_pitch_ratio(0.5)  # Octave down
    assert state.sample_state.use_coeff_buffer, "Should stay in coefficient mode"

    frame = VoiceFrame(note=60, coeff_osc_mode=2)

    # Run 10 blocks - at half speed, should only advance 5 blocks worth
    outputs = []
    for _ in range(10):
        state, output = vm_step_block(state, frame)
        outputs.append(output.copy())

    # Should still be playing (10 blocks at 0.5x = 5 blocks consumed, have 10 total)
    assert state.sample_state.is_playing, "Half speed should still be playing"

    # Block position should be ~5 (10 iterations * 0.5 rate)
    block_int = state.sample_state.block_position_q16 >> 16
    assert 4 <= block_int <= 6, f"Block position should be ~5: {block_int}"

    # All outputs should be non-zero
    for i, out in enumerate(outputs):
        rms = np.sqrt(np.mean(out.astype(float)**2))
        assert rms > 0, f"Block {i} should have non-zero output"


if __name__ == "__main__":
    test_sample_playback()
    print("test_sample_playback passed!")

    test_sample_pitch_shift()
    print("test_sample_pitch_shift passed!")

    test_sample_loop()
    print("test_sample_loop passed!")

    test_coeff_mode_loop()
    print("test_coeff_mode_loop passed!")

    test_coeff_mode_pitch_shift()
    print("test_coeff_mode_pitch_shift passed!")

    test_coeff_mode_pitch_down()
    print("test_coeff_mode_pitch_down passed!")

    print("All sample playback tests passed!")
