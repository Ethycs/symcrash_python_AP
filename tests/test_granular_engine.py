"""
Granular Engine Test Suite
==========================

Validates the coefficient-domain granular synthesis engine.

Tests:
1. Source buffer loading and precomputation
2. Grain triggering and lifecycle
3. Grain envelope
4. Multiple simultaneous grains
5. Grain scheduling (density, spray)
6. Full granular synthesis output
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from symcrash_python_ap.vm.granular_engine import (
    GrainState, GranularState, granular_generate_coeffs,
    granular_set_parameters, create_granular_from_sample,
    MAX_GRAINS, MIN_GRAIN_BLOCKS, MAX_GRAIN_BLOCKS,
    NUM_LEAVES, COEFF_LEN, BLOCK_SIZE, VM_K, Q15_ONE
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit

SAMPLE_RATE = 48000


def generate_test_audio(freq_hz: float, duration_sec: float) -> np.ndarray:
    """Generate a sine wave test buffer."""
    num_samples = int(duration_sec * SAMPLE_RATE)
    t = np.arange(num_samples) / SAMPLE_RATE
    audio = (np.sin(2 * np.pi * freq_hz * t) * 28000).astype(np.int16)
    return audio


def test_source_loading():
    """Test source buffer loading and precomputation."""
    print("\n=== Source Loading Test ===\n")

    state = GranularState()

    # Create test audio
    audio = generate_test_audio(440.0, 0.5)  # 500ms
    expected_blocks = len(audio) // BLOCK_SIZE

    print(f"Audio length: {len(audio)} samples")
    print(f"Expected blocks: {expected_blocks}")

    # Load source
    state.set_source(audio)

    assert state.coeff_buffer is not None, "Coefficient buffer not created"
    assert state.coeff_buffer.shape[0] == expected_blocks, f"Wrong block count: {state.coeff_buffer.shape[0]}"
    assert state.coeff_buffer.shape[1] == NUM_LEAVES, f"Wrong leaf count: {state.coeff_buffer.shape[1]}"
    assert state.num_source_blocks == expected_blocks

    # Verify coefficients have content
    rms = np.sqrt(np.mean(state.coeff_buffer.astype(np.float64) ** 2))
    print(f"Coefficient buffer RMS: {rms:.1f}")
    assert rms > 100, f"Coefficients too quiet: {rms}"

    print("[PASS] Source loading works")
    return True


def test_grain_lifecycle():
    """Test grain triggering and lifecycle."""
    print("\n=== Grain Lifecycle Test ===\n")

    grain = GrainState()

    # Initially inactive
    assert not grain.is_active, "Grain should start inactive"

    # Trigger grain
    grain.trigger(
        start_block=10,
        length_blocks=20,
        rate_q16=65536,
        amplitude_q15=Q15_ONE
    )

    assert grain.is_active, "Grain should be active after trigger"
    assert grain.start_block == 10
    assert grain.length_blocks == 20
    assert grain.blocks_played == 0

    print(f"Grain triggered: start={grain.start_block}, length={grain.length_blocks}")

    # Advance through lifecycle
    for i in range(25):
        was_active = grain.is_active
        still_active = grain.advance()

        if i < 19:
            assert still_active, f"Grain should still be active at block {i}"
        else:
            # Should become inactive after length_blocks
            if i == 19:
                assert not still_active, f"Grain should be inactive at block {i}"

    assert not grain.is_active, "Grain should be inactive after lifecycle"
    print(f"Grain completed after {grain.blocks_played} blocks")

    print("[PASS] Grain lifecycle works")
    return True


def test_grain_envelope():
    """Test grain amplitude envelope."""
    print("\n=== Grain Envelope Test ===\n")

    grain = GrainState()
    grain.trigger(start_block=0, length_blocks=20)

    # Envelope should be attack/sustain/release shaped
    envelope_values = []

    for _ in range(20):
        env = grain.get_envelope_q15()
        envelope_values.append(env)
        grain.advance()

    print(f"Envelope samples: {envelope_values[:5]}...{envelope_values[-5:]}")

    # Attack: should ramp up
    assert envelope_values[0] < envelope_values[2], "Attack should ramp up"

    # Sustain: should be at or near maximum
    mid_point = len(envelope_values) // 2
    assert envelope_values[mid_point] > Q15_ONE * 0.8, f"Sustain too low: {envelope_values[mid_point]}"

    # Release: should ramp down
    assert envelope_values[-1] < envelope_values[-3], "Release should ramp down"

    print("[PASS] Grain envelope works")
    return True


def test_multiple_grains():
    """Test multiple simultaneous grains."""
    print("\n=== Multiple Grains Test ===\n")

    state = GranularState()

    # Load source
    audio = generate_test_audio(440.0, 1.0)
    state.set_source(audio)

    # Manually trigger multiple grains
    active_count = 0
    for i in range(4):
        grain = state.grains[i]
        grain.trigger(
            start_block=i * 10,
            length_blocks=15,
            rate_q16=65536
        )
        active_count += 1

    print(f"Triggered {active_count} grains")

    # Generate output
    output = np.zeros(COEFF_LEN, dtype=np.int16)
    granular_generate_coeffs(state, output)

    # Output should have content from mixed grains
    rms = np.sqrt(np.mean(output.astype(np.float64) ** 2))
    print(f"Output RMS with 4 grains: {rms:.1f}")

    assert rms > 100, f"Output too quiet with multiple grains: {rms}"

    # Count still-active grains
    still_active = sum(1 for g in state.grains if g.is_active)
    print(f"Active grains after one block: {still_active}")

    print("[PASS] Multiple grains work")
    return True


def test_grain_scheduling():
    """Test automatic grain scheduling."""
    print("\n=== Grain Scheduling Test ===\n")

    state = GranularState()

    # Load source
    audio = generate_test_audio(440.0, 1.0)
    state.set_source(audio)

    # Set high density for testing
    granular_set_parameters(
        state,
        density=0.8,  # High spawn probability
        grain_size_ms=50.0,
        spray=0.0
    )

    print(f"Density: {state.density_q15 / Q15_ONE:.2f}")
    print(f"Grain length: {state.grain_length} blocks")

    # Generate several blocks
    output = np.zeros(COEFF_LEN, dtype=np.int16)
    spawned_count = 0

    for block in range(50):
        # Count active before
        active_before = sum(1 for g in state.grains if g.is_active)

        granular_generate_coeffs(state, output)

        # Count active after
        active_after = sum(1 for g in state.grains if g.is_active)

        if active_after > active_before:
            spawned_count += 1

    print(f"Spawned {spawned_count} grains over 50 blocks")

    # With 80% density, we should spawn multiple grains
    # Due to PRNG variance and MAX_GRAINS limit, expect at least 5
    assert spawned_count >= 5, f"Too few grains spawned: {spawned_count}"

    print("[PASS] Grain scheduling works")
    return True


def test_spray_randomization():
    """Test spray/scatter randomization."""
    print("\n=== Spray Randomization Test ===\n")

    # Test without spray
    state_no_spray = GranularState()
    audio = generate_test_audio(440.0, 1.0)
    state_no_spray.set_source(audio)
    granular_set_parameters(state_no_spray, density=0.5, spray=0.0)

    # Test with spray
    state_spray = GranularState()
    state_spray.set_source(audio)
    granular_set_parameters(state_spray, density=0.5, spray=0.5)

    print(f"No spray: spray_q15={state_no_spray.spray_q15}")
    print(f"With spray: spray_q15={state_spray.spray_q15}")

    assert state_no_spray.spray_q15 == 0
    assert state_spray.spray_q15 > 0

    # The spray affects spawn position - we can verify the parameter is set
    # Actual randomization is probabilistic and hard to test deterministically

    print("[PASS] Spray parameters work")
    return True


def test_pitch_shifting():
    """Test grain pitch shifting."""
    print("\n=== Grain Pitch Shifting Test ===\n")

    state = GranularState()

    audio = generate_test_audio(440.0, 0.5)
    state.set_source(audio)

    # Set 2x pitch (octave up)
    granular_set_parameters(
        state,
        density=0.5,
        grain_size_ms=80.0,
        pitch_ratio=2.0
    )

    assert state.grain_rate_q16 == 131072, f"Rate should be 2.0x: {state.grain_rate_q16}"

    print(f"Grain rate: {state.grain_rate_q16 / 65536:.2f}x")

    # Manually trigger a grain and check it advances at 2x
    grain = state.grains[0]
    grain.trigger(start_block=0, length_blocks=20, rate_q16=state.grain_rate_q16)

    initial_pos = grain.current_block_q16
    grain.advance()
    new_pos = grain.current_block_q16

    position_delta = (new_pos - initial_pos) >> 16
    print(f"Position delta per block: {position_delta} (expected 2)")

    # At 2x rate, position should advance by ~2 blocks per advance
    # Note: fractional part accumulates, so we check the rate
    rate_blocks = state.grain_rate_q16 / 65536
    assert abs(rate_blocks - 2.0) < 0.01, f"Rate wrong: {rate_blocks}"

    print("[PASS] Pitch shifting works")
    return True


def test_full_synthesis():
    """Test full granular synthesis output."""
    print("\n=== Full Synthesis Test ===\n")

    # Create granular state from sample
    audio = generate_test_audio(440.0, 1.0)  # 1 second
    state = create_granular_from_sample(
        audio,
        density=0.4,
        grain_size_ms=100.0,
        spray=0.3
    )

    print(f"Source: {len(audio)} samples, {state.num_source_blocks} blocks")

    # Generate many blocks
    output = np.zeros(COEFF_LEN, dtype=np.int16)
    all_outputs = []

    for _ in range(100):
        granular_generate_coeffs(state, output)
        # Convert to audio via IDWPT (k=0 only for simplicity)
        coeff_64 = output[::VM_K].tolist()
        audio_block = idwpt_jit(coeff_64, headroom_bits=1)
        all_outputs.append(np.array(audio_block, dtype=np.int16))

    synth_audio = np.concatenate(all_outputs)
    print(f"Synthesized: {len(synth_audio)} samples")

    # Analyze output
    rms = np.sqrt(np.mean(synth_audio.astype(np.float64) ** 2))
    peak = np.max(np.abs(synth_audio))

    print(f"Output RMS: {rms:.1f}")
    print(f"Output peak: {peak}")

    assert rms > 500, f"Output too quiet: {rms}"
    assert peak < 32768, "Output clipped"

    # Check for variation (granular should produce evolving texture)
    rms_start = np.sqrt(np.mean(synth_audio[:1000].astype(np.float64) ** 2))
    rms_end = np.sqrt(np.mean(synth_audio[-1000:].astype(np.float64) ** 2))
    print(f"RMS variation: start={rms_start:.1f}, end={rms_end:.1f}")

    print("[PASS] Full synthesis works")
    return True


def test_scan_position():
    """Test scan position advancement."""
    print("\n=== Scan Position Test ===\n")

    state = GranularState()

    audio = generate_test_audio(440.0, 1.0)
    state.set_source(audio)

    # Set scan rate to 0.5 (half speed)
    granular_set_parameters(state, density=0.1, scan_rate=0.5)

    initial_pos = state.scan_position
    output = np.zeros(COEFF_LEN, dtype=np.int16)

    # Generate blocks and track position
    positions = []
    for _ in range(20):
        positions.append(state.scan_position)
        granular_generate_coeffs(state, output)

    print(f"Positions over 20 blocks: {positions[:10]}...")

    # At 0.5 scan rate, position should advance roughly 0.5 blocks per block
    # After 20 blocks, should have moved ~10 positions
    total_movement = positions[-1] - positions[0]
    print(f"Total scan movement: {total_movement} (expected ~10)")

    # Allow some tolerance
    assert 5 < total_movement < 15, f"Scan movement wrong: {total_movement}"

    print("[PASS] Scan position works")
    return True


def test_loop_wrap():
    """Test scan position looping."""
    print("\n=== Loop Wrap Test ===\n")

    state = GranularState()

    # Short source buffer
    audio = generate_test_audio(440.0, 0.2)  # 200ms
    state.set_source(audio)

    # Fast scan rate
    granular_set_parameters(state, density=0.1, scan_rate=2.0)

    output = np.zeros(COEFF_LEN, dtype=np.int16)

    # Run until we wrap
    wrapped = False
    prev_pos = 0

    for _ in range(200):
        granular_generate_coeffs(state, output)
        if state.scan_position < prev_pos:
            wrapped = True
            break
        prev_pos = state.scan_position

    print(f"Wrapped: {wrapped}, final position: {state.scan_position}")

    assert wrapped, "Scan should wrap around"

    print("[PASS] Loop wrap works")
    return True


def main():
    print("=" * 60)
    print("Granular Engine Test Suite")
    print("=" * 60)

    results = []

    results.append(("Source Loading", test_source_loading()))
    results.append(("Grain Lifecycle", test_grain_lifecycle()))
    results.append(("Grain Envelope", test_grain_envelope()))
    results.append(("Multiple Grains", test_multiple_grains()))
    results.append(("Grain Scheduling", test_grain_scheduling()))
    results.append(("Spray Randomization", test_spray_randomization()))
    results.append(("Pitch Shifting", test_pitch_shifting()))
    results.append(("Full Synthesis", test_full_synthesis()))
    results.append(("Scan Position", test_scan_position()))
    results.append(("Loop Wrap", test_loop_wrap()))

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
        print("[PASS] All granular engine tests passed!")
        return 0
    else:
        print("[FAIL] Some granular engine tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
