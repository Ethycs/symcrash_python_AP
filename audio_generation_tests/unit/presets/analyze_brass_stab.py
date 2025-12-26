"""Analyze the brass stab audio file."""

import numpy as np
import wave
import os

def analyze_wav(wav_path):
    """Comprehensive analysis of a WAV file for brass stab characteristics."""

    if not os.path.exists(wav_path):
        print(f"ERROR: File not found: {wav_path}")
        return

    with wave.open(wav_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        duration = n_frames / sample_rate
        raw_data = wf.readframes(n_frames)

    print("=" * 60)
    print("BRASS STAB AUDIO ANALYSIS")
    print("=" * 60)
    print(f"\nFile: {wav_path}")
    print(f"Channels: {n_channels}")
    print(f"Sample width: {sample_width} bytes ({sample_width*8} bit)")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Total frames: {n_frames}")

    # Convert to numpy array
    if sample_width == 2:
        samples = np.frombuffer(raw_data, dtype=np.int16)
    elif sample_width == 4:
        samples = np.frombuffer(raw_data, dtype=np.int32)
    else:
        samples = np.frombuffer(raw_data, dtype=np.uint8)

    # If stereo, convert to mono for analysis
    if n_channels == 2:
        left = samples[0::2]
        right = samples[1::2]
        mono = (left.astype(np.float64) + right.astype(np.float64)) / 2
    else:
        mono = samples.astype(np.float64)

    # Normalize for analysis
    mono_norm = mono / 32768.0

    # Amplitude analysis
    peak = np.max(np.abs(mono_norm))
    rms = np.sqrt(np.mean(mono_norm**2))

    print("\n" + "=" * 60)
    print("AMPLITUDE ANALYSIS")
    print("=" * 60)
    print(f"Peak amplitude: {peak:.4f} ({20*np.log10(peak+1e-10):.1f} dBFS)")
    print(f"RMS level: {rms:.4f} ({20*np.log10(rms+1e-10):.1f} dBFS)")
    print(f"Crest factor: {peak / (rms+1e-10):.1f}x ({20*np.log10(peak/(rms+1e-10)):.1f} dB)")

    # Attack analysis
    print("\n" + "=" * 60)
    print("ATTACK ANALYSIS (Target: 2ms attack)")
    print("=" * 60)

    # Find first note attack
    threshold_10 = 0.1 * peak  # 10% of peak
    threshold_90 = 0.9 * peak  # 90% of peak

    above_10 = np.where(np.abs(mono_norm) > threshold_10)[0]
    above_90 = np.where(np.abs(mono_norm) > threshold_90)[0]

    if len(above_10) > 0:
        first_10 = above_10[0]
        time_10 = (first_10 / sample_rate) * 1000
        print(f"Signal reaches 10% at: {time_10:.1f} ms")

        if len(above_90) > 0:
            first_90 = above_90[0]
            time_90 = (first_90 / sample_rate) * 1000
            attack_time = time_90 - time_10
            print(f"Signal reaches 90% at: {time_90:.1f} ms")
            print(f"Attack time (10%-90%): {attack_time:.1f} ms")

            if attack_time < 5:
                print("  -> GOOD: Attack is punchy (<5ms)")
            elif attack_time < 20:
                print("  -> OK: Attack is moderate (5-20ms)")
            else:
                print("  -> ISSUE: Attack is too slow (>20ms) - should be ~2ms for stab")

    # Envelope shape analysis
    print("\n" + "=" * 60)
    print("ENVELOPE SHAPE (50ms blocks)")
    print("=" * 60)

    block_ms = 50
    block_size = int(sample_rate * block_ms / 1000)
    n_blocks = min(40, len(mono_norm) // block_size)

    envelope = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size
        block_rms = np.sqrt(np.mean(mono_norm[start:end]**2))
        envelope.append(block_rms)

    max_env = max(envelope) if envelope else 1
    for i, env in enumerate(envelope[:20]):  # First 1 second
        bar_len = int((env / max_env) * 40)
        time_ms = i * block_ms
        print(f"{time_ms:4d}ms: {'#' * bar_len:<40} {env:.3f}")

    # Analyze note attacks
    print("\n" + "=" * 60)
    print("NOTE DETECTION")
    print("=" * 60)

    # Simple envelope follower
    env_window = int(sample_rate * 0.01)  # 10ms window
    envelope_fine = np.array([
        np.sqrt(np.mean(mono_norm[max(0,i-env_window):i+1]**2))
        for i in range(0, len(mono_norm), env_window)
    ])

    # Find note onsets (significant increases in envelope)
    note_threshold = 0.05
    in_note = False
    notes = []

    for i in range(1, len(envelope_fine)):
        if not in_note and envelope_fine[i] > note_threshold and envelope_fine[i] > envelope_fine[i-1] * 2:
            note_start = i * env_window / sample_rate
            in_note = True
            peak_in_note = envelope_fine[i]
        elif in_note:
            peak_in_note = max(peak_in_note, envelope_fine[i])
            if envelope_fine[i] < note_threshold * 0.5:
                note_end = i * env_window / sample_rate
                notes.append((note_start, note_end, peak_in_note))
                in_note = False

    print(f"Detected {len(notes)} notes")
    for i, (start, end, peak) in enumerate(notes[:10]):
        dur = (end - start) * 1000
        print(f"  Note {i+1}: {start:.2f}s - {end:.2f}s ({dur:.0f}ms) peak={peak:.3f}")

    # Spectral analysis of a note
    print("\n" + "=" * 60)
    print("SPECTRAL ANALYSIS (first loud section)")
    print("=" * 60)

    # Find first loud section
    loud_threshold = 0.3 * peak
    loud_samples = np.where(np.abs(mono_norm) > loud_threshold)[0]

    if len(loud_samples) > 0:
        analyze_start = loud_samples[0]
        analyze_end = min(analyze_start + sample_rate // 10, len(mono_norm))  # 100ms window

        section = mono_norm[analyze_start:analyze_end]

        # Apply window and FFT
        window = np.hanning(len(section))
        windowed = section * window
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(windowed), 1/sample_rate)

        # Find fundamental and harmonics
        # Look for peaks
        peak_indices = []
        for i in range(1, len(spectrum)-1):
            if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
                if spectrum[i] > 0.05 * np.max(spectrum):
                    peak_indices.append(i)

        print("Strongest frequencies:")
        sorted_peaks = sorted(peak_indices, key=lambda x: spectrum[x], reverse=True)[:10]
        for idx in sorted_peaks:
            freq = freqs[idx]
            mag = spectrum[idx] / np.max(spectrum)
            bar = '#' * int(mag * 30)
            print(f"  {freq:7.1f} Hz: {bar:<30} {mag:.2f}")

        # Estimate fundamental
        if len(sorted_peaks) > 0:
            fundamental = freqs[sorted_peaks[0]]
            print(f"\nEstimated fundamental: {fundamental:.1f} Hz")

            # Check harmonic content (brightness)
            low_energy = np.sum(spectrum[freqs < 1000]**2)
            mid_energy = np.sum(spectrum[(freqs >= 1000) & (freqs < 4000)]**2)
            high_energy = np.sum(spectrum[freqs >= 4000]**2)
            total_energy = low_energy + mid_energy + high_energy

            print(f"\nSpectral balance:")
            print(f"  Low (<1kHz):    {100*low_energy/total_energy:.1f}%")
            print(f"  Mid (1-4kHz):   {100*mid_energy/total_energy:.1f}%")
            print(f"  High (>4kHz):   {100*high_energy/total_energy:.1f}%")

            if high_energy / total_energy > 0.15:
                print("  -> Sound has good brightness (brassy)")
            elif high_energy / total_energy > 0.05:
                print("  -> Sound is moderately bright")
            else:
                print("  -> ISSUE: Sound may be too dark, lacking brassy sheen")

    # Check for clicks/pops
    print("\n" + "=" * 60)
    print("CLICK/POP DETECTION")
    print("=" * 60)

    # Look for sudden jumps in sample values
    diff = np.abs(np.diff(mono_norm))
    click_threshold = 0.3  # 30% of full scale jump
    clicks = np.where(diff > click_threshold)[0]

    if len(clicks) > 0:
        print(f"WARNING: Found {len(clicks)} potential clicks/pops")
        for click_idx in clicks[:5]:
            click_time = click_idx / sample_rate
            click_mag = diff[click_idx]
            print(f"  Click at {click_time:.3f}s, magnitude: {click_mag:.3f}")
    else:
        print("No significant clicks detected - audio is clean")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    issues = []
    goods = []

    # Check attack
    if len(above_10) > 0 and len(above_90) > 0:
        attack_time = (above_90[0] - above_10[0]) / sample_rate * 1000
        if attack_time < 5:
            goods.append("Punchy attack")
        else:
            issues.append(f"Attack too slow ({attack_time:.1f}ms, target ~2ms)")

    # Check level
    if peak > 0.8:
        goods.append("Good level (not too quiet)")
    elif peak > 0.5:
        goods.append("Adequate level")
    else:
        issues.append("Output may be too quiet")

    # Check clicks
    if len(clicks) > 0:
        issues.append(f"{len(clicks)} clicks detected")
    else:
        goods.append("Click-free audio")

    print("\nGOOD:")
    for g in goods:
        print(f"  + {g}")

    if issues:
        print("\nISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo major issues detected!")


if __name__ == "__main__":
    wav_path = "audio_out/fur_elise/fur_elise_brass_stab.wav"
    analyze_wav(wav_path)
