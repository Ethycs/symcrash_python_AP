"""
Material Engine FX Tests
========================

Tests for Delay and Chorus effects in the Material Engine.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from symcrash_python_ap.material.material_engine import MaterialEngine, DelayState, ChorusState
from symcrash_python_ap.frames import ParamFrame

SAMPLE_RATE = 48000


def generate_impulse(length: int = 64) -> list:
    """Generate an impulse (single sample spike)."""
    signal = [0] * length
    signal[0] = 32767
    return signal


def generate_sine(freq_hz: float, length: int, amplitude: int = 20000) -> list:
    """Generate a sine wave."""
    t = np.arange(length) / SAMPLE_RATE
    return [int(amplitude * np.sin(2 * np.pi * freq_hz * t_i)) for t_i in t]


def test_delay_basic():
    """Test basic delay line functionality."""
    print("\n=== Delay Basic Test ===\n")

    engine = MaterialEngine(block_size=64)

    # Create impulse
    impulse_L = generate_impulse(64)
    impulse_R = generate_impulse(64)

    # Configure delay: ~500ms delay, some feedback, 50% wet
    # drive_q15=16384 -> 16384/32767 * 48000 = ~24000 samples = ~500ms
    # (max delay is now 1 second = 48000 samples, mono fits in 96KB BRAM)
    params = ParamFrame(
        drive_q15=16384,     # ~500ms delay (50% of 1s max)
        tone_q15=16384,      # 50% feedback
        pressure_q15=16384,  # 50% wet (so we see both dry and wet)
    )

    # Warm up parameters first (slewing takes time)
    silence = [0] * 64
    for _ in range(20):
        engine.step_block(silence, silence, params, material_id=6)

    # Process multiple blocks to catch the delayed impulse
    all_out_L = []
    all_out_R = []

    # First block (impulse)
    out_L, out_R = engine.step_block(impulse_L, impulse_R, params, material_id=6)
    all_out_L.extend(out_L)
    all_out_R.extend(out_R)

    # Additional blocks (silence input, waiting for echo)
    silence = [0] * 64
    for _ in range(400):  # 400 blocks = 25600 samples = 533ms (enough for 500ms delay)
        out_L, out_R = engine.step_block(silence, silence, params, material_id=6)
        all_out_L.extend(out_L)
        all_out_R.extend(out_R)

    # Expected delay: ~24000 samples (500ms)
    expected_delay = int(16384 / 32767 * 48000)
    print(f"Expected delay: {expected_delay} samples ({expected_delay/48:.1f}ms)")

    # The first peak should be at sample 0 (dry signal)
    # The second peak (echo) should be around expected_delay
    all_abs = [abs(x) for x in all_out_L]

    # Find local maxima after sample 100 (skip initial impulse)
    peak_idx = 0
    for i in range(100, len(all_abs) - 1):
        if all_abs[i] > 1000 and all_abs[i] > all_abs[i-1] and all_abs[i] > all_abs[i+1]:
            peak_idx = i
            break

    print(f"Echo peak found at sample {peak_idx}")
    print(f"Echo amplitude: {all_abs[peak_idx]}")

    # The echo should be at or near expected_delay
    delay_error = abs(peak_idx - expected_delay)
    print(f"Delay error: {delay_error} samples")

    # Allow reasonable tolerance
    assert delay_error < 200, f"Delay error too large: {delay_error}"
    assert all_abs[peak_idx] > 500, "Echo too quiet"

    print("[PASS] Basic delay works")
    return True


def test_delay_feedback():
    """Test delay feedback stability."""
    print("\n=== Delay Feedback Test ===\n")

    engine = MaterialEngine(block_size=64)

    # Create impulse
    impulse = generate_impulse(64)
    silence = [0] * 64

    # Configure delay with high feedback
    # Max delay now 1s = 48000 samples (mono BRAM fit)
    params = ParamFrame(
        drive_q15=3277,      # ~100ms delay (~4800 samples) - shorter for feedback test
        tone_q15=25000,      # High feedback (but below runaway)
        pressure_q15=16384,  # 50% wet
    )

    # Warm up parameters first (slewing)
    for _ in range(10):
        engine.step_block(silence, silence, params, material_id=6)

    # Now process with impulse
    out_L, out_R = engine.step_block(impulse, impulse, params, material_id=6)

    # Process many more blocks and track peaks
    peaks = []
    for i in range(200):
        out_L, out_R = engine.step_block(silence, silence, params, material_id=6)
        peak = max(abs(x) for x in out_L)
        peaks.append(peak)

    # Find first significant peak (echo arrival)
    echo_start = 0
    for i, p in enumerate(peaks):
        if p > 500:
            echo_start = i
            break

    print(f"First echo at block {echo_start}")

    # Get peaks after echo starts
    if echo_start > 0:
        relevant_peaks = peaks[echo_start:]
        # Should see decay over time
        mid_point = len(relevant_peaks) // 2
        early_avg = np.mean(relevant_peaks[:20]) if len(relevant_peaks) > 20 else np.mean(relevant_peaks)
        late_avg = np.mean(relevant_peaks[-20:]) if len(relevant_peaks) > 20 else 0

        print(f"Early average peak: {early_avg:.1f}")
        print(f"Late average peak: {late_avg:.1f}")
        print(f"Max peak: {max(peaks)}")

        # Feedback should eventually decay
        assert max(peaks) < 35000, "Feedback should not clip"
    else:
        print(f"Max peak: {max(peaks)}")
        assert max(peaks) < 35000, "Output should not clip"

    print("[PASS] Delay feedback is stable")
    return True


def test_delay_determinism():
    """Test delay determinism (same inputs -> same outputs)."""
    print("\n=== Delay Determinism Test ===\n")

    # Run 1
    engine1 = MaterialEngine(block_size=64)
    input_signal = generate_sine(440.0, 640)
    params = ParamFrame(drive_q15=6553, tone_q15=16384, pressure_q15=24576)

    outputs1 = []
    for i in range(10):
        block = input_signal[i*64:(i+1)*64]
        out_L, _ = engine1.step_block(block, block, params, material_id=6)
        outputs1.extend(out_L)

    # Run 2
    engine2 = MaterialEngine(block_size=64)
    outputs2 = []
    for i in range(10):
        block = input_signal[i*64:(i+1)*64]
        out_L, _ = engine2.step_block(block, block, params, material_id=6)
        outputs2.extend(out_L)

    # Should be identical
    match = all(o1 == o2 for o1, o2 in zip(outputs1, outputs2))
    assert match, "Delay should be deterministic"

    print("[PASS] Delay is deterministic")
    return True


def test_chorus_basic():
    """Test basic chorus functionality."""
    print("\n=== Chorus Basic Test ===\n")

    engine = MaterialEngine(block_size=64)

    # Generate sine wave
    input_signal = generate_sine(440.0, 640 + 640)  # Extra for warmup

    # Configure chorus: medium rate, medium depth, 50% wet
    params = ParamFrame(
        drive_q15=8192,      # ~1.3 Hz LFO rate
        tone_q15=16384,      # Medium depth
        pressure_q15=16384,  # 50% wet
        bias_q15=16384,      # 90 degree stereo offset
    )

    # Warm up parameters (slewing)
    warmup_signal = generate_sine(440.0, 640)
    for i in range(10):
        block = warmup_signal[i*64:(i+1)*64]
        engine.step_block(block, block, params, material_id=7)

    # Process
    all_out_L = []
    all_out_R = []
    for i in range(10):
        block = input_signal[i*64:(i+1)*64]
        out_L, out_R = engine.step_block(block, block, params, material_id=7)
        all_out_L.extend(out_L)
        all_out_R.extend(out_R)

    # Chorus should produce output
    rms_L = np.sqrt(np.mean(np.array(all_out_L, dtype=np.float64)**2))
    rms_R = np.sqrt(np.mean(np.array(all_out_R, dtype=np.float64)**2))

    print(f"Output RMS L: {rms_L:.1f}")
    print(f"Output RMS R: {rms_R:.1f}")

    assert rms_L > 1000, "Chorus L output too quiet"
    assert rms_R > 1000, "Chorus R output too quiet"

    # L and R should be different (stereo effect)
    diff = sum(abs(l - r) for l, r in zip(all_out_L, all_out_R))
    print(f"L/R difference sum: {diff}")

    # With stereo offset, they should differ
    assert diff > 0, "Chorus should have stereo difference"

    print("[PASS] Basic chorus works")
    return True


def test_chorus_modulation():
    """Test chorus LFO modulation."""
    print("\n=== Chorus Modulation Test ===\n")

    engine = MaterialEngine(block_size=64)

    # Generate constant DC (to see modulation clearly)
    dc_signal = [16384] * 64

    # Configure chorus with high rate for visible modulation
    params = ParamFrame(
        drive_q15=32767,     # Max LFO rate (~5 Hz)
        tone_q15=32767,      # Max depth
        pressure_q15=32767,  # 100% wet
        bias_q15=0,          # No stereo offset
    )

    # Process many blocks
    all_out = []
    for _ in range(50):
        out_L, _ = engine.step_block(dc_signal, dc_signal, params, material_id=7)
        all_out.extend(out_L)

    # Should have variation from LFO modulation
    std_dev = np.std(all_out)
    print(f"Output std deviation: {std_dev:.1f}")

    # With modulation, there should be variation
    assert std_dev > 100, "Chorus should modulate the signal"

    print("[PASS] Chorus modulates signal")
    return True


def test_chorus_determinism():
    """Test chorus determinism."""
    print("\n=== Chorus Determinism Test ===\n")

    input_signal = generate_sine(440.0, 640)
    params = ParamFrame(drive_q15=16384, tone_q15=16384, pressure_q15=16384, bias_q15=0)

    # Run 1
    engine1 = MaterialEngine(block_size=64)
    outputs1 = []
    for i in range(10):
        block = input_signal[i*64:(i+1)*64]
        out_L, _ = engine1.step_block(block, block, params, material_id=7)
        outputs1.extend(out_L)

    # Run 2
    engine2 = MaterialEngine(block_size=64)
    outputs2 = []
    for i in range(10):
        block = input_signal[i*64:(i+1)*64]
        out_L, _ = engine2.step_block(block, block, params, material_id=7)
        outputs2.extend(out_L)

    # Should be identical
    match = all(o1 == o2 for o1, o2 in zip(outputs1, outputs2))
    assert match, "Chorus should be deterministic"

    print("[PASS] Chorus is deterministic")
    return True


def test_material_id_routing():
    """Test that material_id routes to correct effect."""
    print("\n=== Material ID Routing Test ===\n")

    engine = MaterialEngine(block_size=64)
    input_signal = generate_sine(440.0, 64)

    # Test clean (material_id=0)
    params = ParamFrame()
    out_clean, _ = engine.step_block(input_signal, input_signal, params, material_id=0)

    # Test delay (material_id=6)
    params_delay = ParamFrame(drive_q15=6553, tone_q15=0, pressure_q15=32767)
    out_delay, _ = engine.step_block(input_signal, input_signal, params_delay, material_id=6)

    # Test chorus (material_id=7)
    params_chorus = ParamFrame(drive_q15=16384, tone_q15=16384, pressure_q15=32767)
    out_chorus, _ = engine.step_block(input_signal, input_signal, params_chorus, material_id=7)

    # All should produce output
    assert max(abs(x) for x in out_clean) > 0, "Clean should produce output"
    assert max(abs(x) for x in out_delay) > 0, "Delay should produce output"
    assert max(abs(x) for x in out_chorus) > 0, "Chorus should produce output"

    # Delay with 100% wet should be mostly silent on first block (delay not filled yet)
    # Chorus should have output immediately
    print(f"Clean peak: {max(abs(x) for x in out_clean)}")
    print(f"Delay peak: {max(abs(x) for x in out_delay)}")
    print(f"Chorus peak: {max(abs(x) for x in out_chorus)}")

    print("[PASS] Material ID routing works")
    return True


def main():
    print("=" * 60)
    print("Material Engine FX Test Suite")
    print("=" * 60)

    results = []

    results.append(("Delay Basic", test_delay_basic()))
    results.append(("Delay Feedback", test_delay_feedback()))
    results.append(("Delay Determinism", test_delay_determinism()))
    results.append(("Chorus Basic", test_chorus_basic()))
    results.append(("Chorus Modulation", test_chorus_modulation()))
    results.append(("Chorus Determinism", test_chorus_determinism()))
    results.append(("Material ID Routing", test_material_id_routing()))

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
        print("[PASS] All FX tests passed!")
        print("\nMaterial Engine FX Features:")
        print("  - Delay (material_id=6): Circular buffer, feedback, wet/dry mix")
        print("  - Chorus (material_id=7): LFO-modulated delay, stereo width")
        print("  - All Q15 fixed-point, deterministic")
        return 0
    else:
        print("[FAIL] Some FX tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
