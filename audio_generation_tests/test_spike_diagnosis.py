"""
Spike Diagnosis - 3 Sweeps
==========================

Chromatic C3->C5 sweep in 3 modes to isolate spike source:
1. Carrier-only (beta=0) - isolate carrier/sine-approx artifacts
2. Wet-only (beta=1) - isolate IDWPT/regularity artifacts
3. Carrier+Wet (beta=0.75) - check for intermodulation

If spikes at 7k/16k appear in:
- Carrier-only -> carrier sine approx or fixed-point issue
- Wet-only -> IDWPT/regularity path is the source
- Only when combined -> intermodulation from nonlinearity
"""

import numpy as np
from pathlib import Path
from typing import List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "symcrash_python_ap"))

from test_config import FS, BLOCK_SIZE, BLOCKS_PER_SEC

from symcrash_python_ap.audio_io import AudioWriter
from symcrash_python_ap.unified_renderer import CarrierLiftPipeline
from symcrash_python_ap.frames import VoiceFrame

# =============================================================================
# SETUP
# =============================================================================

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
OUTPUT_SUBDIR = "spike_diagnosis"

audio_io = AudioWriter(output_dir=OUTPUT_DIR)

Q15_ONE = 32767
Q15_HALF = 16384
Q15_3_4 = 24576


def midi_to_freq(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def save_audio(L: List[int], R: List[int], name: str) -> str:
    metadata = audio_io.write_audio(name, L, R, subdirectory=OUTPUT_SUBDIR, timestamp=False)
    return metadata.filepath


def analyze_spectrum(samples: List[int], name: str) -> dict:
    """Analyze for spike artifacts."""
    arr = np.array(samples, dtype=np.float64)

    # FFT
    fft = np.fft.rfft(arr)
    freqs = np.fft.rfftfreq(len(arr), 1/FS)
    magnitudes = np.abs(fft)

    # Convert to dB
    mag_db = 20 * np.log10(magnitudes + 1e-10)

    # Find peaks in specific bands
    def find_peak_in_band(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        if not np.any(mask):
            return 0, 0
        band_mags = mag_db[mask]
        band_freqs = freqs[mask]
        peak_idx = np.argmax(band_mags)
        return band_freqs[peak_idx], band_mags[peak_idx]

    # Check specific artifact bands
    peak_7k_freq, peak_7k_db = find_peak_in_band(6500, 7500)
    peak_16k_freq, peak_16k_db = find_peak_in_band(15000, 17000)

    # Low frequency energy (DC/drone)
    low_mask = freqs < 200
    low_energy_db = 10 * np.log10(np.sum(magnitudes[low_mask]**2) + 1e-10)

    # RMS
    rms = np.sqrt(np.mean(arr ** 2))
    peak = int(np.max(np.abs(arr)))

    print(f"\n{name}:")
    print(f"  Peak: {peak}, RMS: {rms:.0f}")
    print(f"  Low freq (<200Hz) energy: {low_energy_db:.1f} dB")
    print(f"  7kHz band peak: {peak_7k_freq:.0f} Hz @ {peak_7k_db:.1f} dB")
    print(f"  16kHz band peak: {peak_16k_freq:.0f} Hz @ {peak_16k_db:.1f} dB")

    return {
        'peak': peak,
        'rms': rms,
        'low_energy_db': low_energy_db,
        'peak_7k_freq': peak_7k_freq,
        'peak_7k_db': peak_7k_db,
        'peak_16k_freq': peak_16k_freq,
        'peak_16k_db': peak_16k_db,
    }


# =============================================================================
# CHROMATIC SWEEP GENERATOR
# =============================================================================

def generate_chromatic_sweep(
    name: str,
    beta_mode: str,  # "carrier", "wet", "blend"
) -> tuple:
    """
    Generate chromatic C3->C5 sweep.

    beta_mode:
        "carrier" = beta=0 (carrier only)
        "wet" = beta=1 (wet/IDWPT only)
        "blend" = beta=0.75 (default blend)
    """
    print(f"\n{'='*60}")
    print(f"Generating: {name}")
    print(f"Beta mode: {beta_mode}")
    print(f"{'='*60}")

    pipeline = CarrierLiftPipeline()

    all_L = []
    all_R = []

    # C3 (48) to C5 (72) = 25 notes
    notes = list(range(48, 73))
    note_duration = 0.25  # seconds per note

    for note in notes:
        pipeline.voice_on(0, note=note)

        # Configure beta based on mode
        if beta_mode == "wet":
            pipeline.blend_states[0].wet_ready_counter = 0
            pipeline.blend_states[0].beta_q15 = Q15_ONE
            beta_target = Q15_ONE
        elif beta_mode == "carrier":
            beta_target = 0
        else:  # blend
            beta_target = Q15_3_4

        num_blocks = int(note_duration * BLOCKS_PER_SEC)

        for _ in range(num_blocks):
            # Force beta for wet mode
            if beta_mode == "wet":
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

        # Small gap between notes
        all_L.extend([0] * BLOCK_SIZE)
        all_R.extend([0] * BLOCK_SIZE)

    return all_L, all_R


# =============================================================================
# TESTS
# =============================================================================

def test_1_carrier_only():
    """Chromatic sweep with carrier only (beta=0)."""
    L, R = generate_chromatic_sweep("Carrier Only (beta=0)", "carrier")
    filepath = save_audio(L, R, "1_carrier_only_c3_c5")
    result = analyze_spectrum(L, "Carrier Only")
    print(f"Saved: {filepath}")
    return result


def test_2_wet_only():
    """Chromatic sweep with wet only (beta=1)."""
    L, R = generate_chromatic_sweep("Wet Only (beta=1)", "wet")
    filepath = save_audio(L, R, "2_wet_only_c3_c5")
    result = analyze_spectrum(L, "Wet Only")
    print(f"Saved: {filepath}")
    return result


def test_3_carrier_wet_blend():
    """Chromatic sweep with carrier+wet blend (beta=0.75)."""
    L, R = generate_chromatic_sweep("Carrier+Wet (beta=0.75)", "blend")
    filepath = save_audio(L, R, "3_blend_c3_c5")
    result = analyze_spectrum(L, "Carrier+Wet Blend")
    print(f"Saved: {filepath}")
    return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("SPIKE DIAGNOSIS - 3 SWEEPS")
    print("="*60)
    print(f"\nChromatic C3->C5 sweep")
    print(f"Sample rate: {FS} Hz")
    print(f"Output: audio_out/{OUTPUT_SUBDIR}/")

    results = {}

    results['carrier'] = test_1_carrier_only()
    results['wet'] = test_2_wet_only()
    results['blend'] = test_3_carrier_wet_blend()

    # Summary
    print("\n" + "="*60)
    print("SPIKE DIAGNOSIS SUMMARY")
    print("="*60)

    print("\n7kHz band artifact:")
    print(f"  Carrier: {results['carrier']['peak_7k_db']:.1f} dB")
    print(f"  Wet:     {results['wet']['peak_7k_db']:.1f} dB")
    print(f"  Blend:   {results['blend']['peak_7k_db']:.1f} dB")

    print("\n16kHz band artifact:")
    print(f"  Carrier: {results['carrier']['peak_16k_db']:.1f} dB")
    print(f"  Wet:     {results['wet']['peak_16k_db']:.1f} dB")
    print(f"  Blend:   {results['blend']['peak_16k_db']:.1f} dB")

    print("\nLow freq (<200Hz) energy:")
    print(f"  Carrier: {results['carrier']['low_energy_db']:.1f} dB")
    print(f"  Wet:     {results['wet']['low_energy_db']:.1f} dB")
    print(f"  Blend:   {results['blend']['low_energy_db']:.1f} dB")

    # Diagnosis
    print("\n" + "-"*60)
    print("DIAGNOSIS:")

    carrier_7k = results['carrier']['peak_7k_db']
    wet_7k = results['wet']['peak_7k_db']
    blend_7k = results['blend']['peak_7k_db']

    if wet_7k > carrier_7k + 6:
        print("  -> 7kHz spikes are from WET/IDWPT path")
        print("     Check: RegularityOperator, slew limiter, coefficient quantization")
    elif carrier_7k > wet_7k + 6:
        print("  -> 7kHz spikes are from CARRIER path")
        print("     Check: Sine approximation, fixed-point saturation")
    elif blend_7k > max(carrier_7k, wet_7k) + 6:
        print("  -> 7kHz spikes are INTERMODULATION")
        print("     Check: Nonlinearity where paths combine (regularity, saturation)")
    else:
        print("  -> 7kHz artifacts similar across all modes")

    print("\n" + "="*60)
    print(f"Files saved to: audio_out/{OUTPUT_SUBDIR}/")
    print("="*60)
