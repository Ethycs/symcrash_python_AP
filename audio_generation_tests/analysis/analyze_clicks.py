"""Analyze click locations in chromatic audio files."""
import numpy as np
from pathlib import Path
import wave

# Find the latest chromatic file
audio_dir = Path(__file__).parent.parent / 'audio_out' / 'comparison'
chromatic_files = sorted(audio_dir.glob('chromatic_C3_C5_WET*.wav'),
                         key=lambda p: p.stat().st_mtime, reverse=True)

if chromatic_files:
    latest = chromatic_files[0]
    print(f'Analyzing: {latest.name}')

    with wave.open(str(latest), 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16)

    # Look for clicks (large d2 values)
    d1 = np.diff(samples)
    d2 = np.diff(d1)

    # Find spike locations
    spike_threshold = 1000
    spike_locs = np.where(np.abs(d2) > spike_threshold)[0]

    print(f'Total samples: {len(samples)}')
    print(f'Spikes > {spike_threshold}: {len(spike_locs)}')

    if len(spike_locs) > 0:
        # Analyze timing of spikes
        sample_rate = 48000
        block_size = 64
        note_duration_samples = int(0.3 * sample_rate)  # 0.3s per note

        print(f'\nFirst 30 spike locations (sample, time, block, note):')
        for loc in spike_locs[:30]:
            time_sec = loc / sample_rate
            block_num = loc // block_size
            note_num = loc // note_duration_samples
            pos_in_block = loc % block_size
            print(f'  {loc:7d} @ {time_sec:.4f}s (block {block_num}, pos {pos_in_block}, note {note_num}), d2={d2[loc]}')

        # Check if spikes correlate with block boundaries
        print(f'\nSpikes at block boundaries (pos 0 or 63):')
        boundary_spikes = [loc for loc in spike_locs if (loc % block_size) in [0, 1, 62, 63]]
        print(f'  {len(boundary_spikes)} of {len(spike_locs)} spikes near block boundaries')

        # Check if spikes correlate with note transitions
        print(f'\nSpikes near note transitions:')
        transition_spikes = 0
        for loc in spike_locs:
            pos_in_note = loc % note_duration_samples
            if pos_in_note < 100 or pos_in_note > note_duration_samples - 100:
                transition_spikes += 1
        print(f'  {transition_spikes} of {len(spike_locs)} spikes near note transitions')
else:
    print("No chromatic files found")
