"""Analyze click periodicity."""
import numpy as np
from pathlib import Path
import wave

audio_dir = Path(__file__).parent.parent / 'audio_out' / 'comparison'
chromatic_files = sorted(audio_dir.glob('chromatic_C3_C5_WET*.wav'),
                         key=lambda p: p.stat().st_mtime, reverse=True)

if chromatic_files:
    latest = chromatic_files[0]
    print(f'Analyzing: {latest.name}')

    with wave.open(str(latest), 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16)

    d1 = np.diff(samples)
    d2 = np.diff(d1)

    spike_threshold = 1000
    spike_locs = np.where(np.abs(d2) > spike_threshold)[0]

    print(f'Total spikes: {len(spike_locs)}')

    # Analyze spacing between spikes
    if len(spike_locs) > 2:
        diffs = np.diff(spike_locs)
        # Filter out paired spikes (back-to-back at same location)
        diffs = diffs[diffs > 10]

        if len(diffs) > 0:
            print(f'\nSpacing between spikes (samples):')
            for i, d in enumerate(diffs[:10]):
                freq = 48000 / d
                print(f'  {d} samples ({freq:.1f} Hz)')

            print(f'\nMean spacing: {np.mean(diffs):.1f} samples ({48000/np.mean(diffs):.1f} Hz)')

    # Check what note is playing at each spike
    sample_rate = 48000
    note_duration_samples = int(0.3 * sample_rate)

    print(f'\nSpikes by note:')
    for loc in spike_locs[::2]:  # Skip paired spikes
        note_num = loc // note_duration_samples
        midi_note = 48 + note_num
        time_in_note = (loc % note_duration_samples) / sample_rate
        freq = 440 * (2 ** ((midi_note - 69) / 12))
        print(f'  Spike at {loc}: note {midi_note} ({freq:.1f} Hz), {time_in_note:.3f}s into note')
