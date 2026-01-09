"""Analyze violin output for spurs and waveform quality."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import wave
from numpy.fft import rfft, rfftfreq


def analyze_wav(filepath, name):
    """Analyze WAV for spurs and waveform structure."""
    with wave.open(filepath, 'rb') as w:
        params = w.getparams()
        raw = w.readframes(params.nframes)
        samples = np.frombuffer(raw, dtype=np.int16)

        if params.nchannels == 2:
            samples = samples[::2]

    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Duration: {len(samples)/params.framerate:.2f}s")
    print(f"Peak: {np.max(np.abs(samples))}")
    print(f"RMS: {np.sqrt(np.mean(samples.astype(float)**2)):.0f}")

    # Take sustained portion for analysis
    start = len(samples) // 3
    chunk = samples[start:start + 8192].astype(np.float64)

    # FFT analysis
    window = np.hanning(len(chunk))
    spectrum = np.abs(rfft(chunk * window))
    freqs = rfftfreq(len(chunk), 1/params.framerate)

    # Find fundamental - look for strongest peak in musical range (100-2000 Hz)
    # For A4 (440 Hz) we expect the fundamental in this range
    mask_low = freqs > 100
    mask_high = freqs < 2000
    mask = mask_low & mask_high
    masked = spectrum.copy()
    masked[~mask] = 0
    f0_idx = np.argmax(masked)
    f0 = freqs[f0_idx]
    f0_amp = spectrum[f0_idx]

    # Also find the overall strongest peak (may differ from fundamental)
    mask_all = freqs > 100
    masked_all = spectrum.copy()
    masked_all[~mask_all] = 0
    strongest_idx = np.argmax(masked_all)
    strongest_freq = freqs[strongest_idx]

    print(f"\nFundamental (100-2000 Hz): {f0:.1f} Hz")
    if strongest_freq != f0:
        print(f"Strongest peak overall: {strongest_freq:.1f} Hz")

    # Check for spurs (non-harmonic content)
    print("\n=== Spur Analysis ===")
    print("Freq (Hz) | Amplitude | Harmonic? | dB rel F0")
    print("-" * 50)

    # Find all peaks above noise floor
    noise_floor = np.median(spectrum) * 3
    peaks = []
    for i in range(1, len(spectrum)-1):
        if spectrum[i] > noise_floor and spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
            peaks.append((freqs[i], spectrum[i]))

    # Sort by amplitude
    peaks.sort(key=lambda x: -x[1])

    spur_count = 0
    for freq, amp in peaks[:20]:
        # Check if it's a harmonic of f0
        ratio = freq / f0
        nearest_harmonic = round(ratio)
        is_harmonic = abs(ratio - nearest_harmonic) < 0.05 and nearest_harmonic > 0

        db_rel = 20 * np.log10(amp / f0_amp) if f0_amp > 0 else -100

        harmonic_str = f"H{nearest_harmonic}" if is_harmonic else "SPUR"
        if not is_harmonic and amp > noise_floor * 2:
            spur_count += 1

        print(f"{freq:8.1f} | {amp:9.0f} | {harmonic_str:9s} | {db_rel:+.1f} dB")

    print(f"\nSpurs detected: {spur_count}")

    # Waveform periodicity analysis
    print("\n=== Waveform Periodicity ===")

    # Expected period in samples for f0
    expected_period = params.framerate / f0
    print(f"Expected period: {expected_period:.1f} samples ({1000/f0:.2f} ms)")

    # Check cycle-to-cycle consistency
    period_int = int(round(expected_period))
    num_cycles = min(20, len(chunk) // period_int - 1)

    cycle_correlations = []
    for i in range(num_cycles - 1):
        cycle1 = chunk[i * period_int:(i+1) * period_int]
        cycle2 = chunk[(i+1) * period_int:(i+2) * period_int]
        if len(cycle1) == len(cycle2):
            corr = np.corrcoef(cycle1, cycle2)[0, 1]
            cycle_correlations.append(corr)

    if cycle_correlations:
        avg_corr = np.mean(cycle_correlations)
        min_corr = np.min(cycle_correlations)
        print(f"Cycle-to-cycle correlation: avg={avg_corr:.4f}, min={min_corr:.4f}")

        if avg_corr < 0.95:
            print("WARNING: Low correlation suggests non-repeating waveform (jaggies)")
        elif avg_corr > 0.99:
            print("GOOD: High correlation - stable repeating waveform")

    # Check for DC offset
    dc = np.mean(chunk)
    print(f"DC offset: {dc:.1f}")

    return f0, spectrum, freqs


if __name__ == "__main__":
    # Analyze our output
    analyze_wav(
        "audio_generation_tests/audio_out/presets/violin_ir_single_note.wav",
        "Our Violin IR Output"
    )

    # Analyze reference
    analyze_wav("violin.wav", "Reference violin.wav")
