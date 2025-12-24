"""Analyze clicks in clean_carrier chromatic files."""
import numpy as np
from pathlib import Path
import wave

audio_dir = Path(__file__).parent.parent / 'audio_out' / 'clean_carrier'

files_to_check = [
    'chromatic_C4_C6-20251224-114303.wav',  # NEW - with legato fix
    'chromatic_C3_C5-20251224-114255.wav',  # NEW - with legato fix
    'chromatic_C4_C6-20251224-095250.wav',  # OLD - with clicks
    'chromatic_C3_C5-20251224-095234.wav',  # OLD - with clicks
]

for filename in files_to_check:
    filepath = audio_dir / filename
    if not filepath.exists():
        print(f"File not found: {filename}")
        continue

    print(f"\n{'='*60}")
    print(f"Analyzing: {filename}")
    print('='*60)

    with wave.open(str(filepath), 'rb') as wf:
        nchannels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16)

        # If stereo, just take left channel
        if nchannels == 2:
            samples = samples[::2]

    print(f'Total samples: {len(samples)}')
    print(f'Duration: {len(samples)/48000:.2f}s')

    d1 = np.diff(samples)
    d2 = np.diff(d1)

    print(f'Peak amplitude: {np.max(np.abs(samples))}')
    print(f'Max d1 (first derivative): {np.max(np.abs(d1))}')
    print(f'Max d2 (second derivative): {np.max(np.abs(d2))}')

    # Find spike locations
    for threshold in [500, 1000, 2000]:
        spike_locs = np.where(np.abs(d2) > threshold)[0]
        print(f'Spikes > {threshold}: {len(spike_locs)}')

    spike_threshold = 1000
    spike_locs = np.where(np.abs(d2) > spike_threshold)[0]

    if len(spike_locs) > 0:
        print(f'\nFirst 20 spike locations (d2 > {spike_threshold}):')
        sample_rate = 48000
        block_size = 64

        for loc in spike_locs[:20]:
            time_sec = loc / sample_rate
            block_num = loc // block_size
            pos_in_block = loc % block_size
            print(f'  {loc:7d} @ {time_sec:.4f}s (block {block_num}, pos {pos_in_block}), d2={d2[loc]}')

        # Check block boundary correlation
        boundary_spikes = [loc for loc in spike_locs if (loc % block_size) in [0, 1, 62, 63]]
        print(f'\nSpikes at block boundaries: {len(boundary_spikes)} of {len(spike_locs)}')
