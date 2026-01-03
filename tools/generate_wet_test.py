#!/usr/bin/env python3
"""
Generate test with wet path enabled (ETA > 0) to check for wavelet artifacts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import windows

from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import write_audio

Q15_ONE = 32767
BLOCK_SIZE = 64
SAMPLE_RATE = 48000


def generate_and_analyze(beta_target, label):
    """Generate audio with specified wet amount and analyze."""
    print("\n" + "=" * 70)
    print("Generating: {} (beta_target={})".format(label, beta_target))
    print("=" * 70)

    frame = VoiceFrame(
        voice_id=0,
        note=60,  # Middle C
        cents=0,
        f0_hz=261.63,
        gain_q15=24576,
        pan_q15=0,
        tree_a_id=0,
        tree_b_id=0,
        tree_blend_q15=0,
        is_active=True,
        beta_target_q15=beta_target,
        partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    )

    stage = UnifiedOutputStage(block_size=BLOCK_SIZE)
    stage.voice_on(0, frame.note, frame.cents)

    all_samples = []
    for i in range(1500):  # 2 seconds
        left, right = stage.render_block(frame)
        all_samples.extend(left)

    samples = np.array(all_samples, dtype=np.int16)

    # Basic stats
    rms = np.sqrt(np.mean(samples.astype(np.float64)**2))
    print("Samples: {}  RMS: {:.1f}  Max: {}".format(len(samples), rms, np.max(np.abs(samples))))

    # Check block boundaries
    discontinuities = 0
    for i in range(64, min(len(samples), 6400), 64):
        prev_deriv = samples[i-1] - samples[i-2]
        actual_jump = samples[i] - samples[i-1]
        if abs(actual_jump - prev_deriv) > 100:
            discontinuities += 1

    print("Block boundary discontinuities: {}".format(discontinuities))

    # FFT analysis with window
    window = windows.blackman(len(samples))
    windowed = samples.astype(np.float64) * window
    spectrum = np.abs(rfft(windowed))
    freqs = rfftfreq(len(samples), 1/SAMPLE_RATE)

    # Find fundamental
    start_idx = int(50 * len(freqs) / (SAMPLE_RATE/2))
    fund_idx = start_idx + np.argmax(spectrum[start_idx:start_idx+2000])
    fund_freq = freqs[fund_idx]
    fund_mag = spectrum[fund_idx]

    print("Fundamental: {:.2f} Hz".format(fund_freq))

    # Check for non-harmonic content
    non_harm_peaks = []
    for i in range(len(spectrum)):
        freq = freqs[i]
        if freq < 20:
            continue
        mag = spectrum[i]
        rel_db = 20 * np.log10(mag / fund_mag + 1e-10)
        if rel_db < -60:
            continue

        # Check if near a harmonic
        is_harmonic = False
        for h in range(1, 30):
            if abs(freq - fund_freq * h) < 5:
                is_harmonic = True
                break

        if not is_harmonic:
            non_harm_peaks.append((freq, rel_db))

    if non_harm_peaks:
        print("NON-HARMONIC PEAKS:")
        for freq, db in sorted(non_harm_peaks, key=lambda x: -x[1])[:10]:
            print("  {:.1f} Hz: {:+.1f} dB".format(freq, db))
    else:
        print("PURE: No non-harmonic content above -60 dB")

    # Write WAV using centralized audio_io
    meta = write_audio(
        f"test_wet_{beta_target}",
        samples,  # mono
        timestamp=False,
        subdirectory="wet_tests"
    )
    print("Wrote: {}".format(meta.filepath))

    return samples


def main():
    # Test different wet amounts
    generate_and_analyze(0, "Pure carrier (0% wet)")
    generate_and_analyze(8192, "25% wet")
    generate_and_analyze(16384, "50% wet")
    generate_and_analyze(24576, "75% wet")
    generate_and_analyze(32767, "100% wet")


if __name__ == "__main__":
    main()
