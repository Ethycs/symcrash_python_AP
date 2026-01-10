"""Debug the visible waveform artifact (mini wavelet every 20ms)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import wave
import struct

SAMPLE_RATE = 48000
BLOCK_SIZE = 64


def load_wav(filepath):
    """Load wav file as float array."""
    with wave.open(str(filepath), 'r') as wf:
        n_frames = wf.getnframes()
        data = wf.readframes(n_frames)
        samples = np.array(struct.unpack(f'<{n_frames}h', data), dtype=np.float64)
    return samples


def analyze_waveform_envelope():
    """Analyze the envelope of the waveform for periodic modulation."""
    print("=== Waveform Envelope Analysis ===")

    wav_path = Path(__file__).parent.parent / "audio_generation_tests/audio_out/presets/violin_ir_single_note.wav"
    if not wav_path.exists():
        print(f"Wav file not found: {wav_path}")
        return

    samples = load_wav(wav_path)
    print(f"Loaded {len(samples)} samples ({len(samples)/SAMPLE_RATE:.2f} seconds)")

    # Compute envelope (absolute value, smoothed)
    # Use a window of ~2.3ms (110 samples at 48kHz) for envelope detection
    envelope_window = 110
    envelope = np.zeros(len(samples))

    for i in range(len(samples)):
        start = max(0, i - envelope_window // 2)
        end = min(len(samples), i + envelope_window // 2)
        envelope[i] = np.max(np.abs(samples[start:end]))

    # Look for 20ms periodicity in envelope
    # 20ms = 960 samples at 48kHz
    period_samples = int(0.02 * SAMPLE_RATE)

    # Take a chunk from the sustain portion (1.5-2.5 seconds)
    sustain_start = int(1.5 * SAMPLE_RATE)
    sustain_end = int(2.5 * SAMPLE_RATE)
    chunk = envelope[sustain_start:sustain_end]

    # Compute autocorrelation
    from numpy.fft import rfft, irfft
    fft_chunk = rfft(chunk - np.mean(chunk))
    autocorr = irfft(fft_chunk * np.conj(fft_chunk))
    autocorr = autocorr / autocorr[0]  # Normalize

    print(f"\nEnvelope autocorrelation at specific lags:")
    for lag_ms in [10, 15, 20, 25, 30, 40]:
        lag = int(lag_ms / 1000 * SAMPLE_RATE)
        if lag < len(autocorr):
            print(f"  {lag_ms}ms: {autocorr[lag]:.4f}")

    # Find peaks in autocorrelation (potential periodicities)
    print("\nAutocorrelation peaks (5-50ms range):")
    search_start = int(0.005 * SAMPLE_RATE)
    search_end = int(0.05 * SAMPLE_RATE)

    for i in range(search_start + 1, min(search_end, len(autocorr) - 1)):
        if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
            if autocorr[i] > 0.1:  # Significant peak
                lag_ms = i / SAMPLE_RATE * 1000
                print(f"  {lag_ms:.1f}ms: correlation = {autocorr[i]:.4f}")


def analyze_sample_differences():
    """Look for periodic patterns in sample-to-sample differences."""
    print("\n=== Sample Difference Analysis ===")

    wav_path = Path(__file__).parent.parent / "audio_generation_tests/audio_out/presets/violin_ir_single_note.wav"
    if not wav_path.exists():
        print(f"Wav file not found: {wav_path}")
        return

    samples = load_wav(wav_path)

    # Compute differences
    diff = np.diff(samples)

    # Look for spike patterns
    # A "mini wavelet" artifact would show up as periodic spikes in the derivative
    spike_threshold = np.std(diff) * 3

    print(f"Difference std: {np.std(diff):.1f}")
    print(f"Spike threshold (3*std): {spike_threshold:.1f}")

    # Find spikes
    spike_indices = np.where(np.abs(diff) > spike_threshold)[0]
    print(f"Number of spikes above threshold: {len(spike_indices)}")

    if len(spike_indices) > 10:
        # Check spacing between spikes
        spike_spacing = np.diff(spike_indices)
        mean_spacing = np.mean(spike_spacing)
        std_spacing = np.std(spike_spacing)

        print(f"Mean spike spacing: {mean_spacing:.1f} samples ({mean_spacing/SAMPLE_RATE*1000:.2f}ms)")
        print(f"Std spike spacing: {std_spacing:.1f} samples")

        # Check if spacing is consistent with 20ms
        period_20ms = int(0.02 * SAMPLE_RATE)
        print(f"Expected 20ms period: {period_20ms} samples")


def analyze_block_rate_modulation():
    """Check for modulation at block rate and its harmonics."""
    print("\n=== Block Rate Modulation Analysis ===")

    wav_path = Path(__file__).parent.parent / "audio_generation_tests/audio_out/presets/violin_ir_single_note.wav"
    if not wav_path.exists():
        print(f"Wav file not found: {wav_path}")
        return

    samples = load_wav(wav_path)

    # Take sustain portion
    sustain_start = int(1.5 * SAMPLE_RATE)
    sustain_end = int(2.5 * SAMPLE_RATE)
    chunk = samples[sustain_start:sustain_end]

    # Compute spectrum
    from numpy.fft import rfft, rfftfreq
    spectrum = np.abs(rfft(chunk))
    freqs = rfftfreq(len(chunk), 1/SAMPLE_RATE)

    # Block rate = 48000/64 = 750 Hz
    block_rate = SAMPLE_RATE / BLOCK_SIZE

    print(f"Block rate: {block_rate} Hz")
    print(f"Possible artifact frequencies:")

    # Check for artifacts at block rate sidebands around 440 Hz
    for sideband in [-2, -1, 0, 1, 2]:
        freq = 440 + sideband * block_rate
        if freq > 0:
            idx = int(freq * len(chunk) / SAMPLE_RATE)
            if idx < len(spectrum):
                amp = spectrum[idx]
                # Compare to 440 Hz
                idx_440 = int(440 * len(chunk) / SAMPLE_RATE)
                amp_440 = spectrum[idx_440]
                rel_db = 20 * np.log10(amp / amp_440) if amp_440 > 0 else -100
                print(f"  {freq:.0f} Hz (440 + {sideband}*750): {rel_db:.1f} dB")

    # Also check for low-frequency beat patterns
    print("\nLow-frequency components:")
    for freq_check in [50, 100, 150, 310, 375, 500]:
        idx = int(freq_check * len(chunk) / SAMPLE_RATE)
        if idx < len(spectrum):
            amp = spectrum[idx]
            idx_440 = int(440 * len(chunk) / SAMPLE_RATE)
            amp_440 = spectrum[idx_440]
            rel_db = 20 * np.log10(amp / amp_440) if amp_440 > 0 else -100
            print(f"  {freq_check} Hz: {rel_db:.1f} dB")


def analyze_visual_pattern():
    """Analyze what would be visible as a 'mini wavelet' artifact."""
    print("\n=== Visual Pattern Analysis ===")

    # 20ms at 48kHz = 960 samples
    # At 440 Hz, one cycle = 109 samples
    # So 20ms = 8.8 cycles of 440 Hz

    period_20ms = int(0.02 * SAMPLE_RATE)
    cycle_440 = SAMPLE_RATE / 440

    print(f"20ms = {period_20ms} samples")
    print(f"440 Hz cycle = {cycle_440:.1f} samples")
    print(f"Cycles in 20ms: {period_20ms / cycle_440:.2f}")

    # Check vibrato modulation from preset
    print("\nVibrato modulation check:")
    print("  Preset vibrato rate: 5.5 Hz")
    print(f"  Vibrato period: {1/5.5*1000:.1f} ms")
    print("  -> Not 20ms, so vibrato isn't the cause")

    # Check body warble
    print("\nBody warble check:")
    print("  Preset body warble rate: 1.2 Hz")
    print(f"  Body warble period: {1/1.2*1000:.1f} ms")
    print("  -> Not 20ms, so body warble isn't the cause")

    # 50 Hz = 20ms
    # What could create 50 Hz modulation?
    print("\n50 Hz possibilities:")
    print("  - Beat between fundamental and some internal signal")
    print("  - Block processing artifact (750/15 = 50)")
    print("  - Intermodulation between harmonics")

    # 750/50 = 15 blocks per 20ms cycle
    print(f"\nBlocks in 20ms: {period_20ms / BLOCK_SIZE:.1f}")
    print("15 blocks suggests artifact repeats every 15 blocks")


if __name__ == "__main__":
    analyze_visual_pattern()
    analyze_waveform_envelope()
    analyze_sample_differences()
    analyze_block_rate_modulation()
