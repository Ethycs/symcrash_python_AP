"""
Debug Carrier Pipeline Jaggies
==============================

Test carrier pipeline directly to find where jaggies are introduced.
"""

import numpy as np
from pathlib import Path
from typing import List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "symcrash_python_ap"))

from test_config import FS, BLOCK_SIZE, BLOCKS_PER_SEC

from symcrash_python_ap.wavelet.carrier import CarrierOscillator, fast_sine_q15, PHASE_MASK, PHASE_BITS
from symcrash_python_ap.wavelet.carrier_lift import CarrierState, Q15_ONE, Q15_HALF
from symcrash_python_ap.audio_io import AudioWriter

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
audio_io = AudioWriter(output_dir=OUTPUT_DIR)


def save_audio(samples: List[int], name: str) -> str:
    metadata = audio_io.write_audio(name, samples, samples, subdirectory="carrier_debug", timestamp=False)
    return metadata.filepath


def analyze_waveform(samples: List[int], name: str):
    """Analyze for jaggies."""
    arr = np.array(samples, dtype=np.float64)
    deltas = np.diff(arr)
    delta_of_deltas = np.diff(deltas)

    threshold = 500
    spikes = np.where(np.abs(delta_of_deltas) > threshold)[0]

    print(f"\n{name}:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Peak: {int(np.max(np.abs(arr)))}")
    print(f"  RMS: {np.sqrt(np.mean(arr ** 2)):.0f}")
    print(f"  Max delta: {int(np.max(np.abs(deltas)))}")
    print(f"  Max delta-of-delta: {int(np.max(np.abs(delta_of_deltas)))}")
    print(f"  Spikes (|d2| > {threshold}): {len(spikes)}")

    # Check for block-boundary spikes
    block_boundary_spikes = [s for s in spikes if s % BLOCK_SIZE < 3 or s % BLOCK_SIZE > BLOCK_SIZE - 4]
    print(f"  Block-boundary spikes: {len(block_boundary_spikes)}")

    if len(spikes) > 0 and len(spikes) < 30:
        print(f"  Spike positions (mod 64): {[s % BLOCK_SIZE for s in spikes[:20]]}")

    return len(spikes)


def midi_to_freq(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


# =============================================================================
# TEST 1: CarrierOscillator with pure sine (fundamental only)
# =============================================================================

def test_carrier_pure_sine():
    """Test CarrierOscillator with only fundamental harmonic."""
    print("\n" + "=" * 60)
    print("TEST 1: CarrierOscillator - Pure Sine (fundamental only)")
    print("=" * 60)

    carrier = CarrierOscillator()

    # Configure for pure sine (only fundamental)
    carrier.configure_voice(
        voice_id=0,
        note=60,  # C4
        harmonic_amps=[Q15_ONE, 0, 0, 0, 0, 0, 0, 0],  # Only fundamental
        reset_phase=True,
    )
    carrier.gate_on(0)

    samples = []
    for _ in range(int(3.0 * BLOCKS_PER_SEC)):
        block = carrier.render_block(0, BLOCK_SIZE)
        samples.extend(block)

    spikes = analyze_waveform(samples, "CarrierOscillator Pure Sine")
    filepath = save_audio(samples, "1_carrier_pure_sine")
    print(f"  Saved: {filepath}")

    return spikes


# =============================================================================
# TEST 2: CarrierOscillator with saw harmonics
# =============================================================================

def test_carrier_saw():
    """Test CarrierOscillator with saw harmonics."""
    print("\n" + "=" * 60)
    print("TEST 2: CarrierOscillator - Saw (8 harmonics)")
    print("=" * 60)

    carrier = CarrierOscillator()

    # Default saw harmonics
    carrier.configure_voice(
        voice_id=0,
        note=60,  # C4
        harmonic_amps=[Q15_ONE // n for n in range(1, 9)],
        reset_phase=True,
    )
    carrier.gate_on(0)

    samples = []
    for _ in range(int(3.0 * BLOCKS_PER_SEC)):
        block = carrier.render_block(0, BLOCK_SIZE)
        samples.extend(block)

    spikes = analyze_waveform(samples, "CarrierOscillator Saw")
    filepath = save_audio(samples, "2_carrier_saw")
    print(f"  Saved: {filepath}")

    return spikes


# =============================================================================
# TEST 3: Manual sine rendering (bypass CarrierOscillator)
# =============================================================================

def test_manual_sine():
    """Render sine manually without CarrierOscillator."""
    print("\n" + "=" * 60)
    print("TEST 3: Manual Sine (bypass CarrierOscillator)")
    print("=" * 60)

    freq = midi_to_freq(60)
    phase_inc = int((freq * (1 << PHASE_BITS)) / FS)
    phase = 0

    samples = []
    for _ in range(int(3.0 * FS)):
        val = fast_sine_q15(phase)
        # Apply same gain as carrier would (Q15_HALF = 50%)
        val = (val * Q15_HALF) >> 15
        samples.append(val)
        phase = (phase + phase_inc) & PHASE_MASK

    spikes = analyze_waveform(samples, "Manual Sine")
    filepath = save_audio(samples, "3_manual_sine")
    print(f"  Saved: {filepath}")

    return spikes


# =============================================================================
# TEST 4: Check envelope contribution
# =============================================================================

def test_carrier_no_envelope():
    """Test if envelope is causing jaggies."""
    print("\n" + "=" * 60)
    print("TEST 4: CarrierOscillator with disabled envelope")
    print("=" * 60)

    carrier = CarrierOscillator()

    carrier.configure_voice(
        voice_id=0,
        note=60,
        harmonic_amps=[Q15_ONE, 0, 0, 0, 0, 0, 0, 0],
        reset_phase=True,
    )
    carrier.gate_on(0)

    # Force envelope to full sustain
    state = carrier.states[0]
    state.envelope.stage = 2  # SUSTAIN
    state.envelope.level_q15 = Q15_ONE

    samples = []
    for _ in range(int(3.0 * BLOCKS_PER_SEC)):
        block = carrier.render_block(0, BLOCK_SIZE)
        samples.extend(block)

    spikes = analyze_waveform(samples, "CarrierOscillator No Envelope")
    filepath = save_audio(samples, "4_carrier_no_envelope")
    print(f"  Saved: {filepath}")

    return spikes


# =============================================================================
# TEST 5: Check DC blocker contribution
# =============================================================================

def test_dc_blocker_impact():
    """Check if DC blocker causes jaggies."""
    print("\n" + "=" * 60)
    print("TEST 5: DC Blocker Impact Analysis")
    print("=" * 60)

    freq = midi_to_freq(60)
    phase_inc = int((freq * (1 << PHASE_BITS)) / FS)
    phase = 0
    dc_block_state = 0

    samples_with_dc = []
    samples_without_dc = []

    for _ in range(int(3.0 * FS)):
        val = fast_sine_q15(phase)

        # Without DC block
        samples_without_dc.append(val)

        # With DC block (same as carrier)
        dc_block_state = (dc_block_state * 32700 + val * 67) >> 15
        dc_blocked = val - dc_block_state
        samples_with_dc.append(dc_blocked)

        phase = (phase + phase_inc) & PHASE_MASK

    spikes_without = analyze_waveform(samples_without_dc, "Without DC Blocker")
    spikes_with = analyze_waveform(samples_with_dc, "With DC Blocker")

    filepath1 = save_audio(samples_without_dc, "5a_without_dc_blocker")
    filepath2 = save_audio(samples_with_dc, "5b_with_dc_blocker")
    print(f"  Saved: {filepath1}")
    print(f"  Saved: {filepath2}")

    return spikes_with - spikes_without


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CARRIER PIPELINE JAGGIE DEBUG")
    print("=" * 60)
    print(f"\nSample rate: {FS}")
    print(f"Block size: {BLOCK_SIZE}")

    results = {}

    results['carrier_sine'] = test_carrier_pure_sine()
    results['carrier_saw'] = test_carrier_saw()
    results['manual_sine'] = test_manual_sine()
    results['no_envelope'] = test_carrier_no_envelope()
    results['dc_blocker_diff'] = test_dc_blocker_impact()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, spikes in results.items():
        status = "CLEAN" if spikes == 0 else f"{spikes} spikes"
        print(f"  {name:20s}: {status}")

    print("\n" + "=" * 60)
