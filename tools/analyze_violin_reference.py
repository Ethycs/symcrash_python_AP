"""Analyze reference violin.wav to understand target spectrum."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import wave

def analyze_wav(filepath, name):
    """Analyze a WAV file's spectrum."""
    with wave.open(filepath, 'rb') as w:
        params = w.getparams()
        print(f'\n=== {name} ===')
        print(f'Channels: {params.nchannels}')
        print(f'Sample rate: {params.framerate} Hz')
        print(f'Sample width: {params.sampwidth} bytes')
        print(f'Duration: {params.nframes / params.framerate:.2f}s')

        # Read samples
        raw = w.readframes(params.nframes)
        if params.sampwidth == 2:
            samples = np.frombuffer(raw, dtype=np.int16)
        elif params.sampwidth == 4:
            # 32-bit, could be int32 or float32
            samples = np.frombuffer(raw, dtype=np.int32)
            samples = (samples / 2147483648 * 32767).astype(np.int16)
        else:
            samples = np.frombuffer(raw, dtype=np.int8)

        if params.nchannels == 2:
            samples = samples[::2]  # Take left channel

        peak = np.max(np.abs(samples))
        rms = np.sqrt(np.mean(samples.astype(np.float64)**2))
        print(f'Peak: {peak} ({peak/32768*100:.1f}%)')
        print(f'RMS: {rms:.0f}')

        # Analyze spectrum
        from numpy.fft import rfft, rfftfreq

        # Take a chunk from the middle (sustained portion)
        chunk_start = len(samples) // 3
        chunk_size = min(4096, len(samples) - chunk_start)
        chunk = samples[chunk_start:chunk_start + chunk_size].astype(np.float64)

        # Window and FFT
        window = np.hanning(len(chunk))
        spectrum = np.abs(rfft(chunk * window))
        freqs = rfftfreq(len(chunk), 1/params.framerate)

        # Find peaks
        peak_indices = np.argsort(spectrum)[-10:][::-1]
        print(f'\nDominant frequencies:')
        shown = 0
        for i in peak_indices:
            if freqs[i] > 50 and shown < 5:  # Skip low freq noise
                print(f'  {freqs[i]:.1f} Hz (amplitude: {spectrum[i]:.0f})')
                shown += 1

        # Harmonic rolloff analysis
        # Find fundamental (strongest peak above 100 Hz)
        mask = freqs > 100
        masked_spectrum = spectrum.copy()
        masked_spectrum[~mask] = 0
        fundamental_idx = np.argmax(masked_spectrum)
        f0 = freqs[fundamental_idx]
        print(f'\nFundamental: ~{f0:.1f} Hz')

        # Check harmonic amplitudes
        print('Harmonic structure (relative to fundamental):')
        fund_amp = spectrum[fundamental_idx]
        harmonics = []
        for h in range(1, 9):
            h_freq = f0 * h
            h_idx = int(h_freq / (params.framerate/2) * len(spectrum))
            if h_idx < len(spectrum):
                # Find peak near expected harmonic
                search_range = max(1, int(20 / (params.framerate/2) * len(spectrum)))
                local_peak = np.max(spectrum[max(0, h_idx-search_range):h_idx+search_range+1])
                rel_amp = local_peak / fund_amp
                harmonics.append(rel_amp)
                print(f'  H{h} ({h_freq:.0f} Hz): {rel_amp*100:.1f}%')

        return harmonics, f0


if __name__ == "__main__":
    # Analyze reference
    ref_harmonics, ref_f0 = analyze_wav("violin.wav", "Reference violin.wav")

    # Analyze our rendered version
    our_harmonics, our_f0 = analyze_wav(
        "audio_generation_tests/audio_out/presets/violin_ir_single_note.wav",
        "Our violin_ir_single_note.wav"
    )

    print("\n" + "=" * 50)
    print("COMPARISON")
    print("=" * 50)
    print("\nHarmonic rolloff comparison:")
    print("Harmonic | Reference | Ours")
    print("-" * 35)
    for i, (ref, ours) in enumerate(zip(ref_harmonics, our_harmonics), 1):
        print(f"H{i:d}       | {ref*100:6.1f}%   | {ours*100:6.1f}%")

    print(f"\nFundamental: Reference={ref_f0:.1f} Hz, Ours={our_f0:.1f} Hz")
