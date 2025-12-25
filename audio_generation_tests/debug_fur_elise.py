"""
Debug Für Elise Output
======================

Check what's happening with the Für Elise rendering - analyze levels,
frequency content, and whether the waveforms are correct.
"""

import sys
import wave
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

AUDIO_DIR = Path(__file__).parent.parent / "audio_out" / "fur_elise"


def analyze_wav(filepath: Path):
    """Analyze a WAV file in detail."""
    print(f"\n{'='*60}")
    print(f"File: {filepath.name}")
    print("="*60)

    with wave.open(str(filepath), 'rb') as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    if sample_width == 2:
        data = np.frombuffer(raw_data, dtype=np.int16)
    else:
        print(f"Unsupported sample width: {sample_width}")
        return

    if n_channels == 2:
        data = data.reshape(-1, 2)
        samples = data[:, 0]  # Left channel
    else:
        samples = data

    duration = len(samples) / sample_rate

    # Basic stats
    peak = np.max(np.abs(samples))
    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))

    print(f"Duration: {duration:.2f}s")
    print(f"Sample rate: {sample_rate}")
    print(f"Peak: {peak} ({100*peak/32767:.1f}%)")
    print(f"RMS: {rms:.0f}")

    # Check for silence
    silent_samples = np.sum(np.abs(samples) < 100)
    silence_percent = 100 * silent_samples / len(samples)
    print(f"Near-silent samples (<100): {silence_percent:.1f}%")

    # Check for clipping
    clipped_samples = np.sum(np.abs(samples) > 32700)
    clip_percent = 100 * clipped_samples / len(samples)
    print(f"Clipped samples (>32700): {clip_percent:.1f}%")

    # Find sections with audio
    chunk_size = sample_rate // 10  # 100ms chunks
    num_chunks = len(samples) // chunk_size

    loud_chunks = 0
    for i in range(num_chunks):
        chunk = samples[i*chunk_size:(i+1)*chunk_size]
        chunk_rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))
        if chunk_rms > 500:
            loud_chunks += 1

    print(f"Active chunks (RMS>500): {loud_chunks}/{num_chunks}")

    # FFT on a loud portion
    # Find the loudest 100ms section
    max_rms = 0
    max_idx = 0
    for i in range(num_chunks):
        chunk = samples[i*chunk_size:(i+1)*chunk_size]
        chunk_rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))
        if chunk_rms > max_rms:
            max_rms = chunk_rms
            max_idx = i

    if max_rms > 100:
        chunk = samples[max_idx*chunk_size:(max_idx+1)*chunk_size].astype(np.float64)
        chunk = chunk / np.max(np.abs(chunk)) if np.max(np.abs(chunk)) > 0 else chunk

        spectrum = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / sample_rate)

        # Find peak frequency
        peak_idx = np.argmax(spectrum)
        peak_freq = freqs[peak_idx]

        print(f"\nLoudest section analysis (t={max_idx/10:.1f}s, RMS={max_rms:.0f}):")
        print(f"  Peak frequency: {peak_freq:.1f} Hz")

        # Find fundamental (strongest low freq)
        low_mask = (freqs > 100) & (freqs < 1000)
        if np.any(low_mask):
            low_spectrum = spectrum.copy()
            low_spectrum[~low_mask] = 0
            fund_idx = np.argmax(low_spectrum)
            f0 = freqs[fund_idx]
            fund_power = spectrum[fund_idx]

            print(f"  Fundamental: {f0:.1f} Hz")

            # Calculate THD
            harmonic_powers = []
            for h in range(2, 9):
                h_freq = f0 * h
                h_mask = np.abs(freqs - h_freq) < 20
                if np.any(h_mask):
                    harmonic_powers.append(np.max(spectrum[h_mask]))
                else:
                    harmonic_powers.append(0)

            if fund_power > 0:
                thd = np.sqrt(sum(p**2 for p in harmonic_powers)) / fund_power * 100
                print(f"  THD: {thd:.1f}%")

                # Harmonic ratios
                ratios = [p/fund_power for p in harmonic_powers]
                print(f"  Harmonic ratios: {[f'{r:.2f}' for r in ratios[:4]]}")

    return {
        "name": filepath.stem,
        "peak": peak,
        "rms": rms,
        "duration": duration,
    }


def main():
    print("=" * 70)
    print("FÜR ELISE OUTPUT ANALYSIS")
    print("=" * 70)

    if not AUDIO_DIR.exists():
        print(f"ERROR: Directory not found: {AUDIO_DIR}")
        return

    wav_files = sorted(AUDIO_DIR.glob("*.wav"))

    if not wav_files:
        print("No WAV files found!")
        return

    print(f"\nFound {len(wav_files)} files")

    results = []
    for wav_path in wav_files:
        result = analyze_wav(wav_path)
        if result:
            results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for r in results:
        status = "OK" if r["peak"] > 5000 else "LOW"
        print(f"  {r['name']:30s}: Peak={r['peak']:5d} ({100*r['peak']/32767:4.1f}%), RMS={r['rms']:5.0f} [{status}]")


if __name__ == "__main__":
    main()
