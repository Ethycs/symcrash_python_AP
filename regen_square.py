"""Compare square vs triangle FFT."""
import sys
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

N = 8192

def analyze_wave(filepath, name):
    fs, data = wav.read(filepath)
    if len(data.shape) > 1:
        samples = data[:, 0]
    else:
        samples = data

    fft = np.abs(np.fft.rfft(samples[1000:1000+N])) / (N/2)
    freqs = np.fft.rfftfreq(N, 1/fs)

    idx_h1 = np.argmin(np.abs(freqs - 261.63))
    fundamental = fft[idx_h1]

    print(f"\n{name}:")
    print(f"  Fundamental: {fundamental:.1f}")

    # Check even harmonics
    print(f"  Even harmonics (should be ~0 for square/triangle):")
    for h in [2, 4, 6, 8]:
        idx = np.argmin(np.abs(freqs - 261.63 * h))
        ratio = fft[idx] / fundamental * 100
        print(f"    h={h}: {fft[idx]:.1f} ({ratio:.2f}%)")

    # Check odd harmonics
    print(f"  Odd harmonics:")
    for h in [3, 5, 7, 9]:
        idx = np.argmin(np.abs(freqs - 261.63 * h))
        expected_square = 1/h * 100  # 1/h rolloff for square
        expected_tri = 1/(h*h) * 100  # 1/h^2 rolloff for triangle
        actual = fft[idx] / fundamental * 100
        print(f"    h={h}: {actual:.2f}% (square expect {expected_square:.2f}%, tri expect {expected_tri:.2f}%)")

    # Sub-harmonics (spurious)
    print(f"  Sub-harmonics (spurious):")
    for f in [99.6, 134.8, 175.8]:
        idx = np.argmin(np.abs(freqs - f))
        ratio = fft[idx] / fundamental * 100
        print(f"    {f:.1f} Hz: {fft[idx]:.1f} ({ratio:.2f}%)")

analyze_wave('audio_generation_tests/audio_out/waveforms/02_square.wav', 'SQUARE')
analyze_wave('audio_generation_tests/audio_out/waveforms/05_triangle.wav', 'TRIANGLE')
