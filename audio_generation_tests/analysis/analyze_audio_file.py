"""
Analyze WAV file for discontinuities
"""
import sys
import numpy as np
import wave

def analyze_wav(filepath):
    """Load and analyze a WAV file for discontinuities."""
    print(f"Analyzing: {filepath}")
    print("-" * 60)

    with wave.open(filepath, 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()

        print(f"Channels: {n_channels}")
        print(f"Sample width: {sample_width} bytes")
        print(f"Sample rate: {framerate} Hz")
        print(f"Duration: {n_frames/framerate:.2f}s")

        # Read all frames
        raw = wf.readframes(n_frames)

    # Convert to numpy array
    if sample_width == 2:
        dtype = np.int16
    else:
        print(f"Unsupported sample width: {sample_width}")
        return

    data = np.frombuffer(raw, dtype=dtype)

    if n_channels == 2:
        # Deinterleave
        left = data[0::2]
        right = data[1::2]
        print(f"Samples per channel: {len(left)}")
    else:
        left = data
        right = None
        print(f"Samples: {len(left)}")

    # Analyze left channel
    arr = left.astype(np.float64)
    diffs = np.abs(np.diff(arr))

    max_diff = np.max(diffs)
    mean_diff = np.mean(diffs)
    threshold = mean_diff * 10

    spikes = np.where(diffs > threshold)[0]

    print(f"\nLeft channel analysis:")
    print(f"  Peak: {np.max(np.abs(arr)):.0f}")
    print(f"  Max diff: {max_diff:.0f}")
    print(f"  Mean diff: {mean_diff:.2f}")
    print(f"  Threshold (10x mean): {threshold:.0f}")
    print(f"  Spikes above threshold: {len(spikes)}")

    if len(spikes) > 0:
        print(f"\n  First 10 spikes:")
        for loc in spikes[:10]:
            print(f"    Sample {loc}: {arr[loc]:.0f} -> {arr[loc+1]:.0f} (diff={diffs[loc]:.0f})")

    # Block boundary analysis
    block_size = 64
    boundary_spikes = 0
    print(f"\n  Block boundary check (first 50 blocks):")
    for block in range(1, 50):
        idx = block * block_size - 1
        if idx < len(diffs):
            if diffs[idx] > threshold:
                boundary_spikes += 1
                if boundary_spikes <= 5:
                    print(f"    Block {block} (sample {idx}): diff={diffs[idx]:.0f}")

    if boundary_spikes > 5:
        print(f"    ... and {boundary_spikes - 5} more")
    elif boundary_spikes == 0:
        print(f"    No boundary spikes detected")

    print()


if __name__ == "__main__":
    import glob
    from pathlib import Path

    audio_dir = Path(__file__).parent.parent / "audio_out" / "classic_synths"

    print("=" * 60)
    print("AUDIO FILE ANALYSIS")
    print("=" * 60)

    # List all wav files
    wav_files = sorted(audio_dir.glob("*.wav"))

    for wav in wav_files:
        analyze_wav(str(wav))
