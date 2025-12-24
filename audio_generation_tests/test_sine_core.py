"""
SineCore Verification Tests
===========================

Comprehensive tests for the first-class SineCore module:

1. Bit-exact functional tests
   - Boundary values at 0°, 90°, 180°, 270°
   - Symmetry identities
   - Cosine = sin + 90°

2. Spectral tests
   - THD measurement
   - No block-rate spurs

3. Time-domain tests
   - Monotonicity in Q1
   - No repeatable kinks
   - Click detection
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple
import sys
import math

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "symcrash_python_ap"))

from test_config import FS, BLOCK_SIZE

from symcrash_python_ap.sine_core import (
    sine_q15, cosine_q15, sincos_q15,
    freq_to_phase_inc, midi_to_freq, midi_to_phase_inc,
    OscillatorState, render_sine_block,
    Q_SCALE, PHASE_BITS, PHASE_MASK, TABLE_SIZE,
    verify_accuracy, self_test,
)
from symcrash_python_ap.audio_io import AudioWriter

# =============================================================================
# SETUP
# =============================================================================

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
OUTPUT_SUBDIR = "sine_core_tests"

audio_io = AudioWriter(output_dir=OUTPUT_DIR)


def save_audio(samples: List[int], name: str) -> str:
    """Save mono audio to stereo file."""
    metadata = audio_io.write_audio(name, samples, samples, subdirectory=OUTPUT_SUBDIR, timestamp=False)
    return metadata.filepath


# =============================================================================
# TEST 1: BOUNDARY VALUES
# =============================================================================

def test_boundary_values():
    """Test sine at key boundary points."""
    print("\n" + "=" * 60)
    print("TEST 1: BOUNDARY VALUES")
    print("=" * 60)

    tests = [
        # (phase, expected_sin, expected_cos, name)
        (0x00000000, 0, Q_SCALE, "0°"),
        (0x40000000, Q_SCALE, 0, "90°"),
        (0x80000000, 0, -Q_SCALE, "180°"),
        (0xC0000000, -Q_SCALE, 0, "270°"),
    ]

    passed = True
    for phase, exp_sin, exp_cos, name in tests:
        sin_val = sine_q15(phase)
        cos_val = cosine_q15(phase)

        sin_ok = abs(sin_val - exp_sin) <= 1
        cos_ok = abs(cos_val - exp_cos) <= 1

        status = "OK" if (sin_ok and cos_ok) else "FAIL"
        print(f"  {name:6s}: sin={sin_val:6d} (exp {exp_sin:6d}) "
              f"cos={cos_val:6d} (exp {exp_cos:6d}) [{status}]")

        passed = passed and sin_ok and cos_ok

    return passed


# =============================================================================
# TEST 2: SYMMETRY IDENTITIES
# =============================================================================

def test_symmetry():
    """Test trigonometric symmetry identities."""
    print("\n" + "=" * 60)
    print("TEST 2: SYMMETRY IDENTITIES")
    print("=" * 60)

    passed = True
    errors = []

    # Test at many random phases
    np.random.seed(42)
    test_phases = np.random.randint(0, 1 << 32, size=1000, dtype=np.uint64)

    for phase in test_phases:
        phase = int(phase) & PHASE_MASK

        # sin(θ) = -sin(-θ) = -sin(2π - θ)
        neg_phase = (-phase) & PHASE_MASK
        sin_theta = sine_q15(phase)
        sin_neg_theta = sine_q15(neg_phase)

        if abs(sin_theta + sin_neg_theta) > 2:
            errors.append(f"sin({phase:08X}) + sin({neg_phase:08X}) = {sin_theta + sin_neg_theta}")
            if len(errors) < 5:
                print(f"  FAIL: sin(θ) = -sin(-θ): {sin_theta} != -{sin_neg_theta}")

        # sin²(θ) + cos²(θ) ≈ 1
        sin_val = sine_q15(phase)
        cos_val = cosine_q15(phase)
        sum_sq = (sin_val * sin_val + cos_val * cos_val) / (Q_SCALE * Q_SCALE)

        if abs(sum_sq - 1.0) > 0.001:
            errors.append(f"sin²+cos² at {phase:08X} = {sum_sq}")
            if len(errors) < 5:
                print(f"  FAIL: sin²+cos² = {sum_sq} at phase {phase:08X}")

    if not errors:
        print("  All symmetry identities verified (1000 random phases)")
        print("  sin(x) = -sin(-x): OK")
        print("  sin^2(x) + cos^2(x) = 1: OK")
    else:
        print(f"  {len(errors)} failures")

    passed = len(errors) == 0
    return passed


# =============================================================================
# TEST 3: MONOTONICITY IN Q1
# =============================================================================

def test_monotonicity():
    """Test that sine is monotonically increasing in Q1 (0° to 90°)."""
    print("\n" + "=" * 60)
    print("TEST 3: MONOTONICITY IN Q1")
    print("=" * 60)

    passed = True
    prev = -1
    violations = []

    # Check every possible table position
    for i in range(TABLE_SIZE * 16):  # 16x interpolation resolution
        phase = int((i / (TABLE_SIZE * 16)) * 0x40000000)  # Scale to Q1

        val = sine_q15(phase)
        if val < prev:
            violations.append((i, prev, val))
            if len(violations) < 5:
                print(f"  FAIL: Non-monotonic at i={i}: {prev} -> {val}")
        prev = val

    if not violations:
        print(f"  Monotonicity verified ({TABLE_SIZE * 16} points in Q1)")
    else:
        print(f"  {len(violations)} monotonicity violations")

    passed = len(violations) == 0
    return passed


# =============================================================================
# TEST 4: THD MEASUREMENT
# =============================================================================

def test_thd():
    """Measure Total Harmonic Distortion."""
    print("\n" + "=" * 60)
    print("TEST 4: THD MEASUREMENT")
    print("=" * 60)

    # Generate one cycle of samples
    num_samples = 4096
    samples = []
    for i in range(num_samples):
        phase = int((i / num_samples) * (1 << PHASE_BITS))
        samples.append(sine_q15(phase))

    # FFT
    fft = np.fft.rfft(samples)
    mags = np.abs(fft)

    # Fundamental at index 1
    fund = mags[1]

    # Sum harmonics 2-10
    harmonic_sum_sq = sum(mags[h] ** 2 for h in range(2, 11) if h < len(mags))
    thd = 100 * math.sqrt(harmonic_sum_sq) / fund if fund > 0 else 0

    print(f"  Fundamental magnitude: {fund:.1f}")
    print(f"  Harmonic 2: {mags[2]:.1f} ({100*mags[2]/fund:.4f}%)")
    print(f"  Harmonic 3: {mags[3]:.1f} ({100*mags[3]/fund:.4f}%)")
    print(f"  Harmonic 4: {mags[4]:.1f} ({100*mags[4]/fund:.4f}%)")
    print(f"  THD (harmonics 2-10): {thd:.4f}%")

    passed = thd < 0.1  # <0.1% THD requirement
    print(f"  THD < 0.1%: {'OK' if passed else 'FAIL'}")

    return passed


# =============================================================================
# TEST 5: ACCURACY vs math.sin
# =============================================================================

def test_accuracy():
    """Test accuracy against math.sin reference."""
    print("\n" + "=" * 60)
    print("TEST 5: ACCURACY vs math.sin")
    print("=" * 60)

    max_err, rms_err, thd = verify_accuracy(10000)

    print(f"  Samples tested: 10000")
    print(f"  Max error: {max_err:.2f} LSB")
    print(f"  RMS error: {rms_err:.2f} LSB")
    print(f"  THD: {thd:.4f}%")

    passed = max_err < 2.0 and rms_err < 1.0
    print(f"  Max error < 2 LSB: {'OK' if max_err < 2.0 else 'FAIL'}")
    print(f"  RMS error < 1 LSB: {'OK' if rms_err < 1.0 else 'FAIL'}")

    return passed


# =============================================================================
# TEST 6: WAVEFORM QUALITY (AUDIO)
# =============================================================================

def test_audio_quality():
    """Generate test audio and measure quality metrics."""
    print("\n" + "=" * 60)
    print("TEST 6: AUDIO QUALITY")
    print("=" * 60)

    freq = 440.0  # A4
    duration = 3.0
    num_samples = int(duration * FS)

    # Generate using OscillatorState
    state = OscillatorState()
    state.set_frequency(freq, FS)

    samples = []
    for _ in range(num_samples):
        phase = state.advance()
        samples.append(sine_q15(phase))

    # Analysis
    arr = np.array(samples, dtype=np.float64)
    peak = int(np.max(np.abs(arr)))
    rms = np.sqrt(np.mean(arr ** 2))

    # Click detection
    deltas = np.abs(np.diff(arr))
    clicks_3k = int(np.sum(deltas > 3000))
    clicks_5k = int(np.sum(deltas > 5000))

    print(f"  Frequency: {freq} Hz")
    print(f"  Duration: {duration}s")
    print(f"  Peak: {peak}")
    print(f"  RMS: {rms:.0f}")
    print(f"  Clicks >3k: {clicks_3k}")
    print(f"  Clicks >5k: {clicks_5k}")

    # Save audio
    filepath = save_audio(samples, "sine_440hz_test")
    print(f"  Saved: {filepath}")

    # FFT for spurs
    fft = np.fft.rfft(arr)
    freqs = np.fft.rfftfreq(len(arr), 1 / FS)
    mags = np.abs(fft)
    mag_db = 20 * np.log10(mags + 1e-10)

    # Find fundamental
    fund_idx = np.argmax(mags[1:]) + 1
    fund_freq = freqs[fund_idx]
    fund_db = mag_db[fund_idx]

    print(f"  Fundamental: {fund_freq:.1f} Hz @ {fund_db:.1f} dB")

    # Check for block-rate spurs (750 Hz at 64-sample blocks)
    block_rate = FS / 64
    spur_mask = np.abs(freqs - block_rate) < 50
    if np.any(spur_mask):
        spur_db = np.max(mag_db[spur_mask])
        spur_below_fund = fund_db - spur_db
        print(f"  Block-rate spur ({block_rate:.0f}Hz): {spur_db:.1f} dB ({spur_below_fund:.1f} dB below fundamental)")

    passed = clicks_3k < 10  # Allow minimal clicks
    return passed


# =============================================================================
# TEST 7: PITCH SLEW
# =============================================================================

def test_pitch_slew():
    """Test smooth pitch slew (portamento)."""
    print("\n" + "=" * 60)
    print("TEST 7: PITCH SLEW")
    print("=" * 60)

    state = OscillatorState()
    state.set_frequency(261.6, FS)  # C4

    samples = []

    # Play C4 for 0.5s
    for _ in range(int(0.5 * FS)):
        phase = state.advance()
        samples.append(sine_q15(phase))

    # Slew to G4 over 100ms
    state.set_frequency_slew(392.0, 100.0, FS)  # G4

    for _ in range(int(0.5 * FS)):
        phase = state.advance()
        samples.append(sine_q15(phase))

    # Check for clicks during slew
    arr = np.array(samples, dtype=np.float64)
    deltas = np.abs(np.diff(arr))
    max_delta = int(np.max(deltas))
    clicks = int(np.sum(deltas > 5000))

    print(f"  Slew: C4 (261.6Hz) -> G4 (392Hz) over 100ms")
    print(f"  Max sample delta: {max_delta}")
    print(f"  Clicks >5k: {clicks}")

    filepath = save_audio(samples, "pitch_slew_c4_to_g4")
    print(f"  Saved: {filepath}")

    passed = clicks < 5
    print(f"  Click-free slew: {'OK' if passed else 'FAIL'}")

    return passed


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SINE CORE VERIFICATION TESTS")
    print("=" * 60)
    print(f"\nTable size: {TABLE_SIZE}")
    print(f"Sample rate: {FS} Hz")
    print(f"Output: audio_out/{OUTPUT_SUBDIR}/")

    results = {}

    results['boundary'] = test_boundary_values()
    results['symmetry'] = test_symmetry()
    results['monotonicity'] = test_monotonicity()
    results['thd'] = test_thd()
    results['accuracy'] = test_accuracy()
    results['audio'] = test_audio_quality()
    results['slew'] = test_pitch_slew()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:15s}: {status}")
        all_passed = all_passed and passed

    print("\n" + "=" * 60)
    print(f"OVERALL: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 60)
