"""
Debug Sine Jaggies
==================

Generate a pure sine directly from SineCore to isolate if jaggies
are from SineCore or from the carrier pipeline.
"""

import numpy as np
from pathlib import Path
from typing import List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "symcrash_python_ap"))

from test_config import FS, BLOCK_SIZE

from symcrash_python_ap.sine_core import (
    sine_q15, OscillatorState, PHASE_BITS, PHASE_MASK, Q_SCALE
)
from symcrash_python_ap.audio_io import AudioWriter

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
audio_io = AudioWriter(output_dir=OUTPUT_DIR)


def save_audio(samples: List[int], name: str) -> str:
    metadata = audio_io.write_audio(name, samples, samples, subdirectory="sine_debug", timestamp=False)
    return metadata.filepath


def generate_raw_sine(freq_hz: float, duration_sec: float) -> List[int]:
    """Generate sine using SineCore directly - no carrier, no blocks."""
    num_samples = int(duration_sec * FS)
    phase_inc = int((freq_hz * (1 << PHASE_BITS)) / FS)

    phase = 0
    samples = []

    for _ in range(num_samples):
        samples.append(sine_q15(phase))
        phase = (phase + phase_inc) & PHASE_MASK

    return samples


def generate_via_oscillator_state(freq_hz: float, duration_sec: float) -> List[int]:
    """Generate sine using OscillatorState wrapper."""
    num_samples = int(duration_sec * FS)

    state = OscillatorState()
    state.set_frequency(freq_hz, FS)

    samples = []
    for _ in range(num_samples):
        phase = state.advance()
        samples.append(sine_q15(phase))

    return samples


def analyze_waveform(samples: List[int], name: str, samples_per_cycle: int):
    """Analyze for jaggies and discontinuities."""
    arr = np.array(samples, dtype=np.float64)

    # Sample-to-sample deltas
    deltas = np.diff(arr)

    # For a smooth sine, deltas should change gradually
    # "Jaggies" = sudden changes in delta (second derivative spikes)
    delta_of_deltas = np.diff(deltas)

    # Find any large discontinuities in the second derivative
    threshold = 500  # Significant discontinuity
    spikes = np.where(np.abs(delta_of_deltas) > threshold)[0]

    print(f"\n{name}:")
    print(f"  Samples per cycle: {samples_per_cycle}")
    print(f"  Total samples: {len(samples)}")
    print(f"  Peak: {int(np.max(np.abs(arr)))}")
    print(f"  Max delta (sample-to-sample): {int(np.max(np.abs(deltas)))}")
    print(f"  Max delta-of-delta: {int(np.max(np.abs(delta_of_deltas)))}")
    print(f"  Spikes (|d2| > {threshold}): {len(spikes)}")

    if len(spikes) > 0 and len(spikes) < 20:
        print(f"  Spike locations (first 10): {spikes[:10].tolist()}")
        # Check if spikes are at regular intervals
        if len(spikes) > 1:
            intervals = np.diff(spikes)
            unique_intervals = np.unique(intervals)
            print(f"  Spike intervals: {unique_intervals[:5].tolist()}")

    # Check for block-boundary spikes (every 64 samples)
    block_boundary_spikes = [s for s in spikes if s % BLOCK_SIZE < 2 or s % BLOCK_SIZE > BLOCK_SIZE - 3]
    print(f"  Block-boundary spikes: {len(block_boundary_spikes)}")

    return {
        'max_delta': int(np.max(np.abs(deltas))),
        'max_d2': int(np.max(np.abs(delta_of_deltas))),
        'num_spikes': len(spikes),
        'block_boundary_spikes': len(block_boundary_spikes),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("DEBUG: SINE JAGGIES ISOLATION")
    print("=" * 60)

    freq = 261.6  # C4
    duration = 3.0
    samples_per_cycle = int(FS / freq)

    print(f"\nFrequency: {freq} Hz")
    print(f"Samples per cycle: {samples_per_cycle}")
    print(f"Block size: {BLOCK_SIZE}")

    # Test 1: Raw sine from SineCore
    print("\n--- Test 1: Raw SineCore ---")
    raw_samples = generate_raw_sine(freq, duration)
    analyze_waveform(raw_samples, "Raw SineCore", samples_per_cycle)
    filepath1 = save_audio(raw_samples, "1_raw_sine_core")
    print(f"  Saved: {filepath1}")

    # Test 2: Via OscillatorState
    print("\n--- Test 2: Via OscillatorState ---")
    osc_samples = generate_via_oscillator_state(freq, duration)
    analyze_waveform(osc_samples, "OscillatorState", samples_per_cycle)
    filepath2 = save_audio(osc_samples, "2_oscillator_state")
    print(f"  Saved: {filepath2}")

    # Test 3: Check if samples match
    print("\n--- Comparison ---")
    if raw_samples == osc_samples:
        print("  Raw and OscillatorState outputs are IDENTICAL")
    else:
        diff_count = sum(1 for a, b in zip(raw_samples, osc_samples) if a != b)
        print(f"  Difference count: {diff_count}")

    # Dump first cycle for visual inspection
    print("\n--- First 32 samples ---")
    print("Sample | Raw     | Phase (hex)")
    print("-" * 40)
    phase_inc = int((freq * (1 << PHASE_BITS)) / FS)
    phase = 0
    for i in range(32):
        val = sine_q15(phase)
        print(f"{i:6d} | {val:7d} | {phase:08X}")
        phase = (phase + phase_inc) & PHASE_MASK

    print("\n" + "=" * 60)
    print("If jaggies are present in these files, the issue is in SineCore.")
    print("If clean, the issue is in CarrierOscillator or pipeline.")
    print("=" * 60)
