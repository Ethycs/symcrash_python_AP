"""
Sine Approximation Comparison
=============================

Compares:
1. fast_sine_q15 (parabolic approximation)
2. True numpy sin

Generates two audio files for A/B comparison.
"""

import numpy as np
from pathlib import Path
from typing import List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "symcrash_python_ap"))

from test_config import FS, BLOCK_SIZE

from symcrash_python_ap.audio_io import AudioWriter
from symcrash_python_ap.wavelet.carrier import fast_sine_q15, PHASE_BITS, PHASE_MASK

# =============================================================================
# SETUP
# =============================================================================

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
OUTPUT_SUBDIR = "sine_comparison"

audio_io = AudioWriter(output_dir=OUTPUT_DIR)

Q15_ONE = 32767


def save_audio(L: List[int], R: List[int], name: str) -> str:
    metadata = audio_io.write_audio(name, L, R, subdirectory=OUTPUT_SUBDIR, timestamp=False)
    return metadata.filepath


# =============================================================================
# SINE GENERATORS
# =============================================================================

def generate_fast_sine(freq_hz: float, duration_sec: float) -> List[int]:
    """Generate audio using fast_sine_q15 (parabolic approximation)."""
    num_samples = int(duration_sec * FS)
    phase_inc = int((freq_hz * (1 << PHASE_BITS)) / FS)

    phase = 0
    samples = []

    for _ in range(num_samples):
        sample = fast_sine_q15(phase)
        samples.append(sample)
        phase = (phase + phase_inc) & PHASE_MASK

    return samples


def generate_true_sine(freq_hz: float, duration_sec: float) -> List[int]:
    """Generate audio using numpy sin (true sine)."""
    num_samples = int(duration_sec * FS)
    t = np.arange(num_samples) / FS

    # True sine wave
    wave = np.sin(2 * np.pi * freq_hz * t)

    # Convert to Q15
    samples = [int(s * Q15_ONE) for s in wave]

    return samples


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_waveform(samples: List[int], name: str) -> dict:
    """Analyze waveform for defects."""
    arr = np.array(samples, dtype=np.float64)

    # Basic stats
    peak = int(np.max(np.abs(arr)))
    rms = np.sqrt(np.mean(arr ** 2))

    # FFT for THD analysis
    fft = np.fft.rfft(arr)
    mags = np.abs(fft)

    # Find fundamental (should be at freq_hz)
    fund_idx = np.argmax(mags[1:]) + 1
    fund_mag = mags[fund_idx]

    # Sum of harmonics (2nd, 3rd, 4th, 5th)
    harmonic_sum_sq = 0
    for h in range(2, 6):
        h_idx = fund_idx * h
        if h_idx < len(mags):
            harmonic_sum_sq += mags[h_idx] ** 2

    thd = 100 * np.sqrt(harmonic_sum_sq) / fund_mag if fund_mag > 0 else 0

    # Max delta (clicks)
    deltas = np.abs(np.diff(arr))
    max_delta = int(np.max(deltas))

    print(f"\n{name}:")
    print(f"  Peak: {peak}")
    print(f"  RMS: {rms:.0f}")
    print(f"  THD: {thd:.3f}%")
    print(f"  Max sample delta: {max_delta}")

    return {
        'peak': peak,
        'rms': rms,
        'thd': thd,
        'max_delta': max_delta
    }


def visualize_one_cycle(fast_samples: List[int], true_samples: List[int],
                        samples_per_cycle: int, name: str):
    """Print ASCII visualization of one cycle."""
    print(f"\n{name} - One cycle comparison (fast vs true):")
    print("Sample  Fast    True    Diff")
    print("-" * 40)

    for i in range(min(samples_per_cycle, 32)):
        idx = i * samples_per_cycle // 32
        fast = fast_samples[idx]
        true = true_samples[idx]
        diff = fast - true

        # Scale for display
        bar_fast = int((fast + Q15_ONE) * 20 // (2 * Q15_ONE))
        bar_true = int((true + Q15_ONE) * 20 // (2 * Q15_ONE))

        print(f"{idx:4d}  {fast:7d} {true:7d} {diff:+6d}  |{'F' * bar_fast}{'T' * (bar_true - bar_fast) if bar_true > bar_fast else ''}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SINE APPROXIMATION COMPARISON")
    print("=" * 60)

    freq = 261.6  # C4
    duration = 3.0
    samples_per_cycle = int(FS / freq)

    print(f"\nFrequency: {freq} Hz")
    print(f"Duration: {duration}s")
    print(f"Samples per cycle: {samples_per_cycle}")

    # Generate both versions
    print("\nGenerating fast_sine_q15 (parabolic)...")
    fast_samples = generate_fast_sine(freq, duration)

    print("Generating true sine (numpy)...")
    true_samples = generate_true_sine(freq, duration)

    # Analyze
    fast_analysis = analyze_waveform(fast_samples, "fast_sine_q15 (parabolic)")
    true_analysis = analyze_waveform(true_samples, "True sine (numpy)")

    # Compute error
    fast_arr = np.array(fast_samples[:samples_per_cycle], dtype=np.float64)
    true_arr = np.array(true_samples[:samples_per_cycle], dtype=np.float64)

    error = fast_arr - true_arr
    max_error = int(np.max(np.abs(error)))
    rms_error = np.sqrt(np.mean(error ** 2))

    print(f"\nError analysis (one cycle):")
    print(f"  Max error: {max_error} ({100*max_error/Q15_ONE:.2f}%)")
    print(f"  RMS error: {rms_error:.0f} ({100*rms_error/Q15_ONE:.2f}%)")

    # Save audio
    filepath1 = save_audio(fast_samples, fast_samples, "1_fast_sine_parabolic")
    filepath2 = save_audio(true_samples, true_samples, "2_true_sine_numpy")

    print(f"\nSaved: {filepath1}")
    print(f"Saved: {filepath2}")

    # Generate difference signal (amplified 10x)
    diff_samples = [(f - t) * 10 for f, t in zip(fast_samples, true_samples)]
    diff_samples = [max(-Q15_ONE, min(Q15_ONE, d)) for d in diff_samples]
    filepath3 = save_audio(diff_samples, diff_samples, "3_difference_10x")
    print(f"Saved: {filepath3}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nfast_sine_q15 THD: {fast_analysis['thd']:.3f}%")
    print(f"True sine THD:     {true_analysis['thd']:.3f}%")
    print(f"\nThe parabolic approximation introduces {fast_analysis['thd']:.3f}% THD")
    print("Listen to the difference file (3_difference_10x.wav) to hear the artifact")
    print("=" * 60)
