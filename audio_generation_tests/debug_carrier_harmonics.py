"""
Debug Carrier Harmonics
=======================

Verify that the carrier oscillator is actually using the harmonic amplitudes
we pass to it, and diagnose any issues.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from test_config import FS, BLOCK_SIZE, BLOCKS_PER_SEC

from symcrash_python_ap.wavelet.carrier import CarrierOscillator
from symcrash_python_ap.wavelet.harmonic_injector import WaveType, get_partial_amplitude
from symcrash_python_ap.audio_io import AudioWriter

# Constants
Q15_ONE = 32767
Q15_HALF = 16384

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
audio_io = AudioWriter(output_dir=OUTPUT_DIR)


def generate_partial_amps(wave_type: WaveType, num_partials: int = 8) -> list:
    """Generate partial amplitude list for a given waveform type."""
    return [get_partial_amplitude(wave_type, k + 1) for k in range(num_partials)]


def test_carrier_direct(name: str, harmonic_amps: list, note: int = 69, duration_sec: float = 2.0):
    """
    Test CarrierOscillator directly with specific harmonics.
    """
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Note: {note} (440 Hz)")
    print(f"Harmonic amps: {harmonic_amps[:8]}")
    print("="*60)

    # Create carrier oscillator
    carrier = CarrierOscillator()

    # Configure voice with specific harmonics
    carrier.configure_voice(
        voice_id=0,
        note=note,
        harmonic_amps=harmonic_amps,
        reset_phase=True
    )

    # Trigger envelope
    carrier.gate_on(0)

    # Verify the state
    state = carrier.states[0]
    print(f"State harmonic_amps_q15: {state.harmonic_amps_q15}")

    # Calculate blocks needed
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)

    samples = []
    for _ in range(num_blocks):
        block = carrier.render_block(0, BLOCK_SIZE)
        samples.extend(block)

    # Release
    carrier.gate_off(0)

    # Render release
    for _ in range(int(0.5 * BLOCKS_PER_SEC)):
        block = carrier.render_block(0, BLOCK_SIZE)
        samples.extend(block)

    # Analyze
    arr = np.array(samples, dtype=np.float64)
    peak = int(np.max(np.abs(arr)))
    rms = np.sqrt(np.mean(arr ** 2))

    print(f"Peak: {peak} ({100*peak/32767:.1f}%)")
    print(f"RMS: {rms:.0f}")

    if peak < 100:
        print("WARNING: Very low output!")

    # FFT analysis for harmonic content
    chunk_start = len(samples) // 3
    chunk_end = 2 * len(samples) // 3
    chunk = np.array(samples[chunk_start:chunk_end], dtype=np.float64)
    if np.max(np.abs(chunk)) > 0:
        chunk = chunk / np.max(np.abs(chunk))

    spectrum = np.abs(np.fft.rfft(chunk))
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / FS)

    # Find fundamental
    low_mask = (freqs > 200) & (freqs < 600)
    if np.any(low_mask):
        low_spectrum = spectrum.copy()
        low_spectrum[~low_mask] = 0
        fund_idx = np.argmax(low_spectrum)
        f0_detected = freqs[fund_idx]
        fund_power = spectrum[fund_idx]

        print(f"\nFFT Analysis:")
        print(f"  Detected f0: {f0_detected:.1f} Hz")

        # Measure harmonics
        for h in range(1, 9):
            h_freq = f0_detected * h
            h_mask = np.abs(freqs - h_freq) < 20
            if np.any(h_mask):
                h_power = np.max(spectrum[h_mask])
                ratio = h_power / fund_power if fund_power > 0 else 0
                expected = harmonic_amps[h-1] / Q15_ONE if h <= len(harmonic_amps) else 0
                print(f"  H{h} ({h_freq:.0f}Hz): {ratio:.3f} (expected ~{expected:.3f})")

    # Save
    L = samples
    R = samples
    metadata = audio_io.write_audio(
        f"carrier_test_{name}",
        L, R,
        subdirectory="carrier_debug",
        timestamp=False
    )
    print(f"Saved: {metadata.filepath}")

    return {"name": name, "peak": peak, "rms": rms}


def main():
    print("=" * 70)
    print("CARRIER OSCILLATOR HARMONIC DEBUG TEST")
    print("=" * 70)
    print(f"\nSample rate: {FS} Hz")
    print(f"Block size: {BLOCK_SIZE}")

    results = []

    # Test 1: Pure sine (only fundamental)
    sine_amps = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]
    results.append(test_carrier_direct("sine", sine_amps))

    # Test 2: Square wave (odd harmonics)
    square_amps = generate_partial_amps(WaveType.PULSE, 8)
    print(f"\nSquare wave partials from generator: {square_amps}")
    results.append(test_carrier_direct("square", square_amps))

    # Test 3: Sawtooth (all harmonics)
    saw_amps = generate_partial_amps(WaveType.SAW, 8)
    print(f"\nSaw wave partials from generator: {saw_amps}")
    results.append(test_carrier_direct("sawtooth", saw_amps))

    # Test 4: Triangle (odd harmonics with 1/k² decay)
    tri_amps = generate_partial_amps(WaveType.TRIANGLE, 8)
    print(f"\nTriangle wave partials from generator: {tri_amps}")
    results.append(test_carrier_direct("triangle", tri_amps))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for r in results:
        status = "OK" if r["peak"] > 1000 else "LOW"
        print(f"  {r['name']:12s}: Peak={r['peak']:5d}, RMS={r['rms']:5.0f} [{status}]")


if __name__ == "__main__":
    main()
