#!/usr/bin/env python3
"""Render example notes through the complete live output path, then play them."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

import numpy as np
from symcrash_python_ap.audio.live_renderer import LiveSynthRenderer
from symcrash_python_ap.primitives import SAMPLE_RATE_HZ, BLOCK_SIZE


def render_note(note: int, duration_sec: float) -> np.ndarray:
    """Render a held note plus its release, using the canonical 48 kHz rate."""
    if not np.isfinite(duration_sec) or duration_sec <= 0:
        raise ValueError('Duration must be positive and finite')
    renderer = LiveSynthRenderer()
    renderer.note_on(note)
    blocks = [renderer.render_mono_float()
              for _ in range(max(1, int(duration_sec * SAMPLE_RATE_HZ / BLOCK_SIZE)))]
    renderer.note_off(note)
    for _ in range(2 * SAMPLE_RATE_HZ // BLOCK_SIZE):
        block = renderer.render_mono_float()
        blocks.append(block)
        if not renderer.engine.active_voices() and not np.any(block):
            break
    return np.concatenate(blocks)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=2.0)
    args = parser.parse_args()
    import sounddevice as sd
    for note in (60,72,48):
        print(f'Rendering note {note}...')
        audio = render_note(note, args.duration)
        sd.play(audio, SAMPLE_RATE_HZ)
        sd.wait()


if __name__ == '__main__':
    main()
