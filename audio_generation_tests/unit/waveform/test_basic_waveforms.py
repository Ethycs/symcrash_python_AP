"""
Basic Waveform Validation Test
==============================

Generate pure sine, square, triangle, and sawtooth waveforms through the
CarrierLiftPipeline to validate the synthesis chain is working correctly.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # for test_config

from test_config import FS, BLOCK_SIZE, BLOCKS_PER_SEC

from symcrash_python_ap.safety.unified_renderer import CarrierLiftPipeline
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import AudioWriter
from symcrash_python_ap.wavelet.harmonic_injector import WaveType, get_partial_amplitude

# Constants
Q15_ONE = 32767
Q15_HALF = 16384

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
audio_io = AudioWriter(output_dir=OUTPUT_DIR)


def generate_partial_amps(wave_type: WaveType, num_partials: int = 16) -> list:
    """Generate partial amplitude list for a given waveform type."""
    return [get_partial_amplitude(wave_type, k + 1) for k in range(num_partials)]


def test_waveform(name: str, wave_type: WaveType, note: int = 69, duration_sec: float = 2.0):
    """
    Generate a single sustained note with a specific waveform type.

    Args:
        name: Name for the output file
        wave_type: WaveType enum (SINE, SAW, PULSE, TRIANGLE)
        note: MIDI note number (69 = A4 = 440Hz)
        duration_sec: Duration in seconds
    """
    print(f"\n{'='*60}")
    print(f"Testing: {name} (WaveType.{wave_type.name})")
    print(f"Note: {note} ({440.0 * (2.0 ** ((note - 69) / 12.0)):.1f} Hz)")
    print("="*60)

    # Generate partial amplitudes
    partial_amps = generate_partial_amps(wave_type)
    print(f"Partial amps (first 8): {partial_amps[:8]}")

    # Create pipeline
    pipeline = CarrierLiftPipeline()

    # Calculate blocks needed
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)

    # Start voice
    pipeline.voice_on(0, note=note)

    samples_L = []
    samples_R = []

    # Render with constant envelope
    f0_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))

    for block_idx in range(num_blocks):
        # Simple attack/sustain envelope
        if block_idx < 10:
            env = block_idx / 10.0
        else:
            env = 1.0

        frame = VoiceFrame(
            voice_id=0,
            note=note,
            f0_hz=f0_hz,
            gain_q15=int(Q15_HALF * env),
            beta_target_q15=0,  # Pure carrier (dry)
            partial_amps_q15=partial_amps,
        )

        L, R = pipeline.render_block(frame)
        samples_L.extend(L)
        samples_R.extend(R)

    # End voice
    pipeline.voice_off(0)

    # Analyze
    arr = np.array(samples_L, dtype=np.float64)
    peak = int(np.max(np.abs(arr)))
    rms = np.sqrt(np.mean(arr ** 2))

    print(f"Peak: {peak} ({100*peak/32767:.1f}%)")
    print(f"RMS: {rms:.0f}")

    # Check if we have actual audio
    if peak < 100:
        print("WARNING: Very low output level - possible pipeline issue!")

    # Save
    metadata = audio_io.write_audio(
        f"test_{name}",
        samples_L, samples_R,
        subdirectory="waveform_tests",
        timestamp=False
    )
    print(f"Saved: {metadata.filepath}")

    return {
        "name": name,
        "wave_type": wave_type,
        "peak": peak,
        "rms": rms,
        "partial_amps": partial_amps[:8],
    }


def main():
    print("=" * 70)
    print("BASIC WAVEFORM VALIDATION TEST")
    print("=" * 70)
    print(f"\nSample rate: {FS} Hz")
    print(f"Block size: {BLOCK_SIZE}")
    print(f"Blocks per second: {BLOCKS_PER_SEC}")

    # Test each waveform type
    results = []

    results.append(test_waveform("sine", WaveType.SINE))
    results.append(test_waveform("square", WaveType.PULSE))
    results.append(test_waveform("triangle", WaveType.TRIANGLE))
    results.append(test_waveform("sawtooth", WaveType.SAW))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_ok = True
    for r in results:
        status = "OK" if r["peak"] > 1000 else "LOW"
        if r["peak"] < 1000:
            all_ok = False
        print(f"  {r['name']:12s}: Peak={r['peak']:5d}, RMS={r['rms']:5.0f} [{status}]")
        print(f"               Partials: {r['partial_amps']}")

    print()
    if all_ok:
        print("All waveforms generating correctly!")
    else:
        print("WARNING: Some waveforms have low output - check pipeline!")

    print(f"\nOutput directory: audio_out/waveform_tests/")


if __name__ == "__main__":
    main()
