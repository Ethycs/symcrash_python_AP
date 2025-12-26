"""
Classic Synth Waveform Comparison
=================================

Tests classic waveforms (sine, saw, square, triangle) on:
- DRY path (carrier only, beta=0)
- WET path (IDWPT only, beta=1)

8 test samples total:
1-4: DRY sine, saw, square, triangle
5-8: WET sine, saw, square, triangle
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "symcrash_python_ap"))

from test_config import FS, BLOCK_SIZE, BLOCKS_PER_SEC

from symcrash_python_ap.audio_io import AudioWriter
from symcrash_python_ap.safety.unified_renderer import CarrierLiftPipeline
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.fixed import sat16

# =============================================================================
# SETUP
# =============================================================================

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
OUTPUT_SUBDIR = "classic_waveforms"

audio_io = AudioWriter(output_dir=OUTPUT_DIR)

Q15_ONE = 32767
Q15_HALF = 16384
Q15_3_4 = 24576

# Number of harmonics for carrier synthesis
NUM_HARMONICS = 8


def midi_to_freq(note: int, cents: int = 0) -> float:
    """MIDI note to frequency."""
    return 440.0 * (2.0 ** ((note - 69 + cents / 100.0) / 12.0))


# =============================================================================
# WAVEFORM DEFINITIONS (harmonic amplitudes)
# =============================================================================

def get_waveform_harmonics(waveform: str) -> List[int]:
    """
    Get Q15 harmonic amplitudes for classic waveforms.

    Args:
        waveform: "sine", "saw", "square", or "triangle"

    Returns:
        List of 8 Q15 harmonic amplitudes
    """
    if waveform == "sine":
        # Pure sine - only fundamental
        return [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    elif waveform == "saw":
        # Sawtooth - all harmonics with 1/n amplitude
        return [Q15_ONE // n for n in range(1, NUM_HARMONICS + 1)]

    elif waveform == "square":
        # Square - only odd harmonics with 1/n amplitude
        harmonics = []
        for n in range(1, NUM_HARMONICS + 1):
            if n % 2 == 1:  # Odd harmonics only
                harmonics.append(Q15_ONE // n)
            else:
                harmonics.append(0)
        return harmonics

    elif waveform == "triangle":
        # Triangle - only odd harmonics with 1/n^2 amplitude, alternating sign
        # For additive synthesis, we use absolute values
        harmonics = []
        for n in range(1, NUM_HARMONICS + 1):
            if n % 2 == 1:  # Odd harmonics only
                harmonics.append(Q15_ONE // (n * n))
            else:
                harmonics.append(0)
        return harmonics

    else:
        raise ValueError(f"Unknown waveform: {waveform}")


def save_audio(L: List[int], R: List[int], name: str) -> str:
    """Save audio to file."""
    metadata = audio_io.write_audio(name, L, R, subdirectory=OUTPUT_SUBDIR, timestamp=False)
    return metadata.filepath


def analyze_audio(samples: List[int], name: str) -> dict:
    """Quick analysis of audio."""
    arr = np.array(samples, dtype=np.float64)
    peak = int(np.max(np.abs(arr)))
    rms = np.sqrt(np.mean(arr ** 2))

    # Click detection
    deltas = np.abs(np.diff(arr))
    clicks_3k = int(np.sum(deltas > 3000))
    clicks_5k = int(np.sum(deltas > 5000))
    clicks_10k = int(np.sum(deltas > 10000))

    print(f"\n  {name}:")
    print(f"    Peak: {peak}, RMS: {rms:.0f}")
    print(f"    Clicks: >3k={clicks_3k}, >5k={clicks_5k}, >10k={clicks_10k}")

    return {
        'peak': peak,
        'rms': rms,
        'clicks_3k': clicks_3k,
        'clicks_5k': clicks_5k,
        'clicks_10k': clicks_10k,
    }


# =============================================================================
# CLASSIC SYNTH PATTERNS
# =============================================================================

def classic_synth_1_sequence() -> List[Tuple[int, float]]:
    """
    Classic synth pattern 1: Arpeggiated chord progression
    Returns list of (note, duration_sec)
    """
    # C minor arpeggio pattern
    pattern = [
        (48, 0.2),  # C3
        (51, 0.2),  # Eb3
        (55, 0.2),  # G3
        (60, 0.2),  # C4
        (55, 0.2),  # G3
        (51, 0.2),  # Eb3
        # G minor
        (43, 0.2),  # G2
        (50, 0.2),  # D3
        (55, 0.2),  # G3
        (58, 0.2),  # Bb3
        (55, 0.2),  # G3
        (50, 0.2),  # D3
        # Ab major
        (44, 0.2),  # Ab2
        (48, 0.2),  # C3
        (51, 0.2),  # Eb3
        (56, 0.2),  # Ab3
        (51, 0.2),  # Eb3
        (48, 0.2),  # C3
        # Bb major -> resolve
        (46, 0.3),  # Bb2
        (50, 0.3),  # D3
        (53, 0.3),  # F3
        (58, 0.5),  # Bb3 (longer)
    ]
    return pattern


def classic_synth_2_sequence() -> List[Tuple[int, float]]:
    """
    Classic synth pattern 2: Bass + lead melody
    Returns list of (note, duration_sec)
    """
    # Punchy bass with melodic lead
    pattern = [
        # Bass intro
        (36, 0.15),  # C2
        (36, 0.15),  # C2
        (48, 0.1),   # C3 (octave punch)
        (36, 0.2),   # C2
        # Melodic rise
        (60, 0.25),  # C4
        (62, 0.25),  # D4
        (63, 0.25),  # Eb4
        (65, 0.35),  # F4
        # Bass response
        (41, 0.15),  # F2
        (41, 0.15),  # F2
        (53, 0.1),   # F3
        (41, 0.2),   # F2
        # Melodic descent
        (67, 0.25),  # G4
        (65, 0.25),  # F4
        (63, 0.25),  # Eb4
        (60, 0.35),  # C4
        # Final bass
        (36, 0.2),   # C2
        (43, 0.2),   # G2
        (36, 0.4),   # C2 (sustain)
    ]
    return pattern


# =============================================================================
# TEST GENERATORS
# =============================================================================

def generate_waveform_test(
    name: str,
    waveform: str,  # "sine", "saw", "square", "triangle"
    wet: bool,  # True = 100% wet (IDWPT), False = 0% wet (carrier only)
    note: int = 60,  # MIDI note (default C4)
    duration_sec: float = 3.0,
) -> Tuple[List[int], List[int], dict]:
    """
    Generate a single waveform test.

    Args:
        name: Test name for logging
        waveform: Waveform type
        wet: True = 100% wet (IDWPT only), False = 0% wet (carrier only)
        note: MIDI note to play
        duration_sec: Duration in seconds

    Returns:
        (L_samples, R_samples, analysis_dict)
    """
    print(f"\n{'='*60}")
    print(f"Generating: {name}")
    print(f"Waveform: {waveform.upper()}")
    print(f"Path: {'WET (IDWPT)' if wet else 'DRY (Carrier)'}")
    print(f"Note: {note} ({midi_to_freq(note):.1f} Hz)")
    print(f"{'='*60}")

    pipeline = CarrierLiftPipeline()

    # Get harmonic amplitudes for this waveform
    harmonic_amps = get_waveform_harmonics(waveform)
    print(f"Harmonics: {harmonic_amps}")

    all_L = []
    all_R = []

    # Fixed beta: 0 = dry (carrier only), Q15_ONE = wet (IDWPT only)
    beta_target = Q15_ONE if wet else 0

    num_blocks = int(duration_sec * BLOCKS_PER_SEC)

    # Allocate voice
    pipeline.voice_on(0, note=note)

    # Set carrier harmonic amplitudes
    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = harmonic_amps

    # If wet, force wet ready immediately
    if wet:
        pipeline.blend_states[0].wet_ready_counter = 0
        pipeline.blend_states[0].beta_q15 = Q15_ONE

    for i in range(num_blocks):
        # Force beta to stay at target each block
        if wet:
            pipeline.blend_states[0].beta_q15 = Q15_ONE

        frame = VoiceFrame(
            voice_id=0,
            note=note,
            f0_hz=midi_to_freq(note),
            gain_q15=Q15_HALF,
            beta_target_q15=beta_target,
        )

        L, R = pipeline.render_block(frame)
        all_L.extend(L)
        all_R.extend(R)

    pipeline.voice_off(0)

    # Analysis
    analysis = analyze_audio(all_L, name)

    return all_L, all_R, analysis


# =============================================================================
# WAVEFORM TESTS
# =============================================================================

WAVEFORMS = ["sine", "saw", "square", "triangle"]


def run_all_waveform_tests():
    """Run all waveform tests for both dry and wet paths."""
    results = {}

    # DRY tests (carrier only)
    for i, waveform in enumerate(WAVEFORMS, start=1):
        name = f"dry_{waveform}"
        L, R, analysis = generate_waveform_test(
            name=f"DRY {waveform.upper()}",
            waveform=waveform,
            wet=False,
        )
        filepath = save_audio(L, R, f"{i}_dry_{waveform}")
        print(f"Saved: {filepath}")
        results[name] = analysis

    # WET tests (IDWPT only)
    for i, waveform in enumerate(WAVEFORMS, start=5):
        name = f"wet_{waveform}"
        L, R, analysis = generate_waveform_test(
            name=f"WET {waveform.upper()}",
            waveform=waveform,
            wet=True,
        )
        filepath = save_audio(L, R, f"{i}_wet_{waveform}")
        print(f"Saved: {filepath}")
        results[name] = analysis

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("CLASSIC WAVEFORM COMPARISON: DRY vs WET")
    print("="*60)
    print(f"\nSample rate: {FS} Hz")
    print(f"Block size: {BLOCK_SIZE}")
    print(f"Output: audio_out/{OUTPUT_SUBDIR}/")
    print(f"\nWaveforms: {', '.join(WAVEFORMS)}")

    results = run_all_waveform_tests()

    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)

    print("\n{:12} {:>10} {:>10} {:>10} {:>10}".format(
        "Waveform", "DRY Peak", "DRY RMS", "WET Peak", "WET RMS"))
    print("-" * 54)

    for waveform in WAVEFORMS:
        dry = results.get(f"dry_{waveform}", {})
        wet = results.get(f"wet_{waveform}", {})
        print("{:12} {:>10} {:>10.0f} {:>10} {:>10.0f}".format(
            waveform.upper(),
            dry.get('peak', 'N/A'),
            dry.get('rms', 0),
            wet.get('peak', 'N/A'),
            wet.get('rms', 0),
        ))

    print("\n" + "="*60)
    print(f"Files saved to: audio_out/{OUTPUT_SUBDIR}/")
    print("="*60)
