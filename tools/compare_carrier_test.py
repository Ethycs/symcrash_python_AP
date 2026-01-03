#!/usr/bin/env python3
"""
Compare carrier generation: original test vs new test.

Find why original test_pipeline_sine was clean but new carrier tests have combing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import wave
import numpy as np
from scipy.fft import rfft, rfftfreq

from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.frames import VoiceFrame

Q15_ONE = 32767
BLOCK_SIZE = 64
SAMPLE_RATE = 48000


def generate_original_style():
    """Generate using EXACT parameters from original generate_test_audio.py."""
    # This is copy-pasted from the original
    frame = VoiceFrame(
        voice_id=0,
        note=60,  # Middle C
        cents=0,
        f0_hz=261.63,  # EXPLICIT f0_hz
        gain_q15=24576,
        pan_q15=0,  # Center
        tree_a_id=0,
        tree_b_id=0,
        tree_blend_q15=0,
        is_active=True,
        beta_target_q15=0,  # No wet/wavelet
        partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],  # Pure sine
    )

    stage = UnifiedOutputStage(block_size=BLOCK_SIZE)
    stage.voice_on(0, frame.note, frame.cents)

    all_samples = []
    for i in range(1000):  # ~1.3 seconds
        left, right = stage.render_block(frame)
        all_samples.extend(left)

    return np.array(all_samples, dtype=np.int16), "Original style (f0_hz=261.63)"


def generate_new_style():
    """Generate using my new test parameters (no explicit f0_hz)."""
    frame = VoiceFrame(
        voice_id=0,
        note=60,  # Middle C
        cents=0,
        # NO f0_hz - uses default 440.0
        gain_q15=24576,
        pan_q15=0,
        beta_target_q15=0,  # 0% wet = pure carrier
        is_active=True,
        partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    )

    stage = UnifiedOutputStage(block_size=BLOCK_SIZE)
    stage.voice_on(0, frame.note, frame.cents)

    all_samples = []
    for i in range(1000):
        left, right = stage.render_block(frame)
        all_samples.extend(left)

    return np.array(all_samples, dtype=np.int16), "New style (no f0_hz)"


def analyze_spectrum(samples, label, rate=SAMPLE_RATE):
    """Analyze spectrum of samples."""
    print("\n" + "=" * 60)
    print("Analyzing: {}".format(label))
    print("=" * 60)
    print("Samples: {}, RMS: {:.1f}".format(
        len(samples),
        np.sqrt(np.mean(samples.astype(np.float64)**2))
    ))

    # FFT
    spectrum = np.abs(rfft(samples.astype(np.float64)))
    freqs = rfftfreq(len(samples), 1/rate)

    # Find fundamental
    start_idx = int(50 * len(freqs) / (rate/2))
    fundamental_idx = start_idx + np.argmax(spectrum[start_idx:start_idx+1000])
    fundamental_freq = freqs[fundamental_idx]
    fundamental_mag = spectrum[fundamental_idx]

    print("Fundamental: {:.1f} Hz (magnitude: {:.0f})".format(
        fundamental_freq, fundamental_mag))

    # Look for non-harmonic peaks (combing indicators)
    peak_indices = np.argsort(spectrum)[-15:][::-1]

    non_harmonic = []
    for idx in peak_indices[:15]:
        freq = freqs[idx]
        if freq < 50 or freq == 0:
            continue
        harmonic_ratio = freq / fundamental_freq
        nearest_harmonic = round(harmonic_ratio)
        if nearest_harmonic > 0:
            deviation = abs(harmonic_ratio - nearest_harmonic) / nearest_harmonic
            rel_db = 20 * np.log10(spectrum[idx] / fundamental_mag + 1e-10)
            if deviation > 0.02 and rel_db > -60:
                non_harmonic.append((freq, rel_db, harmonic_ratio))

    if non_harmonic:
        print("NON-HARMONIC PEAKS (combing):")
        for freq, rel_db, ratio in non_harmonic:
            print("  {:.1f} Hz ({:+.1f} dB) ratio={:.3f}".format(freq, rel_db, ratio))
    else:
        print("OK: All major peaks are harmonic")

    return spectrum, freqs, fundamental_freq


def main():
    print("Comparing carrier generation methods...")
    print("Both use note=60 (Middle C = 261.63 Hz)")
    print("=" * 60)

    # Generate both styles
    samples_orig, label_orig = generate_original_style()
    samples_new, label_new = generate_new_style()

    # Analyze spectra
    analyze_spectrum(samples_orig, label_orig)
    analyze_spectrum(samples_new, label_new)

    # Direct comparison
    print("\n" + "=" * 60)
    print("DIRECT COMPARISON")
    print("=" * 60)

    min_len = min(len(samples_orig), len(samples_new))
    diff = samples_orig[:min_len].astype(np.int32) - samples_new[:min_len].astype(np.int32)
    diff_rms = np.sqrt(np.mean(diff**2))

    print("Difference RMS: {:.1f}".format(diff_rms))
    print("Max difference: {}".format(np.max(np.abs(diff))))

    if diff_rms < 1.0:
        print("IDENTICAL: Both methods produce same output")
    elif diff_rms < 100:
        print("SIMILAR: Minor differences")
    else:
        print("DIFFERENT: Significant differences detected")

    # Check if samples are bit-identical
    if np.array_equal(samples_orig[:min_len], samples_new[:min_len]):
        print("BIT IDENTICAL: Yes")
    else:
        # Find first difference
        for i in range(min_len):
            if samples_orig[i] != samples_new[i]:
                print("BIT IDENTICAL: No (first diff at sample {}: {} vs {})".format(
                    i, samples_orig[i], samples_new[i]))
                break

    # Write both to WAV for auditory comparison
    base = Path(__file__).parent.parent
    for samples, label in [(samples_orig, "compare_orig"), (samples_new, "compare_new")]:
        output_path = base / "{}.wav".format(label)
        with wave.open(str(output_path), 'w') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(SAMPLE_RATE)
            f.writeframes(samples.tobytes())
        print("Wrote: {}".format(output_path))


if __name__ == "__main__":
    main()
