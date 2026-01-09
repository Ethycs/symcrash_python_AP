"""Analyze noise differences between single note and phrase renders."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq
import wave
import struct

def analyze_wav(filepath, name):
    with wave.open(filepath, 'r') as wf:
        n_frames = wf.getnframes()
        data = wf.readframes(n_frames)
        samples = np.array(struct.unpack(f'<{n_frames}h', data), dtype=np.float64)

    # Take middle 1 second for analysis (avoid attack/release)
    start = len(samples) // 3
    end = start + 48000
    if end > len(samples):
        end = len(samples)
        start = max(0, end - 48000)
    chunk = samples[start:end]

    spectrum = np.abs(rfft(chunk))
    freqs = rfftfreq(len(chunk), 1/48000)

    # Find peak
    peak_idx = np.argmax(spectrum)
    peak_freq = freqs[peak_idx]
    peak_amp = spectrum[peak_idx]

    print(f'\n=== {name} ===')
    print(f'Peak frequency: {peak_freq:.1f} Hz (amplitude: {peak_amp:.0f})')

    # Find top 5 spurs (non-harmonics)
    spurs = []
    for i in range(1, len(spectrum)-1):
        if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
            freq = freqs[i]
            ratio = freq / peak_freq if peak_freq > 0 else 0
            nearest_h = round(ratio)
            if abs(ratio - nearest_h) > 0.05 or nearest_h == 0:
                spurs.append((freq, spectrum[i]))

    spurs.sort(key=lambda x: -x[1])
    print('Top 5 non-harmonic spurs:')
    for freq, amp in spurs[:5]:
        rel_db = 20 * np.log10(amp / peak_amp) if peak_amp > 0 else -100
        print(f'  {freq:.1f} Hz: {rel_db:.1f} dB')

    # RMS noise floor estimate (high frequencies > 10kHz)
    hf_start = int(10000 * len(chunk) / 48000)
    hf_rms = np.sqrt(np.mean(spectrum[hf_start:]**2))
    hf_db = 20 * np.log10(hf_rms / peak_amp) if peak_amp > 0 else -100
    print(f'HF noise floor (>10kHz): {hf_db:.1f} dB')


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent / "audio_generation_tests/audio_out/presets"

    analyze_wav(str(base_path / "violin_ir_single_note.wav"), "Single Note (A4 = 440 Hz)")
    analyze_wav(str(base_path / "violin_ir_phrase.wav"), "Phrase (C4-E4-G4-C5)")
