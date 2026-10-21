#!/usr/bin/env python3
"""Compose and render a short piece with the target-fit DX7 marimba preset.

Every pitched event is rendered by the Symcrash engine from InstrumentIR.
The mixer adds only panning, delay-derived ambience, and mastering; it never
pitch-shifts or reuses the audition WAV as an oscillator.
"""

from __future__ import annotations

import argparse
import math
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from render.render_preset import render_preset_note  # noqa: E402
from symcrash_python_ap.instrument_ir import load_instrument  # noqa: E402
from symcrash_python_ap.vm.harmonic_bank import clear_basis_atlas  # noqa: E402


FS = 48_000
TEMPO = 96.0
BEAT_SECONDS = 60.0 / TEMPO
BAR_BEATS = 4.0
TITLE = "Algorithm 7: Courtyard"


@dataclass(frozen=True)
class Event:
    beat: float
    note: int
    velocity: float
    gain: float
    pan: float
    send: float


def add_chord(events: list[Event], beat: float, notes: list[int], *,
              velocity: float, gain: float, width: float, send: float) -> None:
    """Add a chord with a small harp-like roll and symmetric stereo spread."""
    count = max(1, len(notes) - 1)
    for index, note in enumerate(notes):
        pan = width * ((2.0 * index / count) - 1.0)
        events.append(Event(beat + 0.035 * index, note, velocity,
                            gain, pan, send))


def build_score() -> list[Event]:
    """Return the deterministic 24-bar score in beat coordinates."""
    events: list[Event] = []

    # Two-bar harmonic fields. Voicings stay open so the half-ratio carriers
    # retain the bell-like low octave without turning the middle into mud.
    harmony = [
        ([50, 57, 60, 64, 65], [62, 65, 69, 72, 76]),  # Dm9
        ([46, 53, 57, 60, 62], [62, 65, 69, 74, 77]),  # Bbmaj9
        ([43, 50, 53, 57, 58], [62, 65, 67, 70, 74]),  # Gm9
        ([45, 52, 55, 59, 62], [61, 64, 67, 71, 74]),  # A7sus / A7
        ([50, 57, 60, 64, 65], [62, 65, 69, 72, 77]),
        ([48, 55, 60, 62, 67], [60, 64, 67, 74, 76]),  # C6/9
        ([46, 53, 57, 60, 65], [62, 65, 69, 72, 77]),
        ([45, 52, 55, 61, 64], [61, 64, 67, 73, 76]),
        ([43, 50, 53, 57, 62], [62, 67, 70, 74, 77]),
        ([45, 52, 57, 60, 62], [60, 64, 69, 72, 76]),
        ([46, 53, 57, 60, 65], [65, 69, 72, 77, 81]),
        ([50, 57, 62, 64, 69], [62, 65, 69, 74, 76]),
    ]

    # Foundation: one dark strike per bar, with a quieter answer on beat 3.
    for bar in range(22):
        roots, _ = harmony[bar // 2]
        start = bar * BAR_BEATS
        events.append(Event(start, roots[0], 0.90 if bar % 4 == 0 else 0.68,
                            0.66, -0.18, 0.26))
        if bar >= 4 and bar not in (15, 19):
            events.append(Event(start + 2.5, roots[1], 0.62,
                                0.34, 0.16, 0.30))

    # Opening: isolated harmonic landmarks with a lot of air.
    for bar in range(4):
        roots, upper = harmony[bar // 2]
        start = bar * BAR_BEATS
        add_chord(events, start + 1.0, [roots[2], roots[4]],
                  velocity=0.62, gain=0.28, width=0.42, send=0.62)
        events.append(Event(start + 3.25, upper[(bar * 2) % len(upper)],
                            0.70, 0.28, 0.55 if bar % 2 == 0 else -0.55, 0.72))

    # Main lattice: a 3+3+2 eighth-note pattern. The phase rotates each bar.
    lattice_steps = [0.0, 0.75, 1.5, 2.25, 3.0, 3.5]
    for bar in range(4, 20):
        _, upper = harmony[bar // 2]
        start = bar * BAR_BEATS
        rotation = (bar * 2) % len(upper)
        order = [0, 2, 1, 4, 2, 3]
        for index, offset in enumerate(lattice_steps):
            note = upper[(order[index] + rotation) % len(upper)]
            accent = index in (0, 3)
            events.append(Event(
                start + offset,
                note,
                0.86 if accent else 0.62,
                0.34 if accent else 0.25,
                -0.62 + 1.24 * (index / (len(lattice_steps) - 1)),
                0.36 if accent else 0.48,
            ))

    # Melody: two related eight-bar statements, then a high suspended answer.
    melody = [
        (0.0, 74), (1.5, 77), (2.5, 81), (3.5, 79),
        (4.0, 77), (5.0, 74), (6.5, 72),
        (8.0, 74), (9.0, 76), (10.0, 77), (11.5, 81),
        (12.0, 84), (13.5, 81), (15.0, 79),
    ]
    for phrase_start, transpose in ((24.0, 0), (48.0, 2)):
        for index, (offset, note) in enumerate(melody):
            events.append(Event(
                phrase_start + offset,
                note + transpose,
                0.90 if index in (0, 7, 11) else 0.72,
                0.39,
                0.32 if index % 2 == 0 else -0.32,
                0.55,
            ))

    # Bell-like pinpoints at structural peaks.
    for beat, note, pan in [
        (31.75, 86, 0.72), (39.25, 84, -0.68),
        (55.75, 88, 0.70), (63.25, 86, -0.72),
        (71.75, 89, 0.62), (79.5, 86, -0.58),
    ]:
        events.append(Event(beat, note, 0.92, 0.27, pan, 0.82))

    # Bars 20-21 thin out; bars 22-23 are a rolled D-minor/add9 resolution.
    for bar in (20, 21):
        roots, upper = harmony[bar // 2]
        start = bar * BAR_BEATS
        events.append(Event(start + 1.5, upper[bar % len(upper)],
                            0.60, 0.24, (-1) ** bar * 0.5, 0.72))
        events.append(Event(start + 3.0, roots[2], 0.54,
                            0.20, (-1) ** (bar + 1) * 0.35, 0.68))

    add_chord(events, 88.0, [50, 57, 62, 65, 69, 76],
              velocity=0.82, gain=0.39, width=0.72, send=0.58)
    events.append(Event(92.0, 50, 0.64, 0.42, 0.0, 0.72))
    events.append(Event(93.5, 69, 0.58, 0.22, 0.38, 0.84))

    return sorted(events, key=lambda event: event.beat)


def equal_power_pan(pan: float) -> tuple[float, float]:
    angle = (max(-1.0, min(1.0, pan)) + 1.0) * math.pi / 4.0
    return math.cos(angle), math.sin(angle)


def delay_ambience(send: np.ndarray) -> np.ndarray:
    """Create a stereo ambience using only delayed copies of the synth bus."""
    wet = np.zeros((2, len(send)), dtype=np.float64)
    taps = [
        (0.071, 0.31, -0.72),
        (0.109, 0.27, 0.65),
        (0.163, 0.22, -0.28),
        (0.227, 0.18, 0.34),
        (0.347, 0.13, -0.58),
        (0.463, 0.10, 0.56),
    ]
    for delay_seconds, gain, pan in taps:
        delay = round(delay_seconds * FS)
        left, right = equal_power_pan(pan)
        for repeat in range(1, 7):
            offset = delay * repeat
            if offset >= len(send):
                break
            repeat_gain = gain * (0.58 ** (repeat - 1))
            wet[0, offset:] += send[:-offset] * repeat_gain * left
            wet[1, offset:] += send[:-offset] * repeat_gain * right

    # Gentle damping: successive averages remove the brittle top from echoes.
    for channel in range(2):
        for _ in range(3):
            wet[channel, 1:] = 0.72 * wet[channel, 1:] + 0.28 * wet[channel, :-1]
    return wet


def write_stereo_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.rint(audio.T * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(FS)
        wav_file.writeframes(pcm.tobytes())


def compose(preset_path: Path, output_path: Path) -> None:
    ir = load_instrument(preset_path)
    score = build_score()
    piece_seconds = 24 * BAR_BEATS * BEAT_SECONDS
    total_seconds = piece_seconds + 5.0
    total_samples = math.ceil(total_seconds * FS)
    dry = np.zeros((2, total_samples), dtype=np.float64)
    send = np.zeros(total_samples, dtype=np.float64)
    cache: dict[tuple[int, int], np.ndarray] = {}

    clear_basis_atlas()
    print(f"{TITLE}")
    print(f"Preset: {preset_path}")
    print(f"Tempo: {TEMPO:g} BPM | Events: {len(score)}")

    for event_index, event in enumerate(score, start=1):
        velocity_layer = 0.62 if event.velocity < 0.76 else 0.90
        velocity_q15 = round(velocity_layer * 32767)
        cache_key = (event.note, velocity_q15)
        if cache_key not in cache:
            print(f"  Synthesizing MIDI {event.note:3d} at velocity {velocity_layer:.2f}")
            rendered = render_preset_note(
                ir,
                note=event.note,
                duration_sec=0.22,
                velocity_q15=velocity_q15,
            )
            if rendered.ndim == 2:
                rendered = np.mean(rendered.astype(np.float64), axis=0)
            else:
                rendered = rendered.astype(np.float64)
            cache[cache_key] = rendered / 32768.0

        voice = cache[cache_key]
        start = round(event.beat * BEAT_SECONDS * FS)
        stop = min(total_samples, start + len(voice))
        voice = voice[:stop - start]
        left, right = equal_power_pan(event.pan)
        dry[0, start:stop] += voice * event.gain * left
        dry[1, start:stop] += voice * event.gain * right
        send[start:stop] += voice * event.gain * event.send

        if event_index % 32 == 0:
            print(f"  Placed {event_index}/{len(score)} events")

    mix = dry + 0.72 * delay_ambience(send)

    # Slow fade-in, musical fade-out, DC removal, and transparent peak control.
    fade_in = round(0.35 * FS)
    fade_out = round(4.5 * FS)
    mix[:, :fade_in] *= np.linspace(0.0, 1.0, fade_in)
    mix[:, -fade_out:] *= np.linspace(1.0, 0.0, fade_out)
    mix -= np.mean(mix, axis=1, keepdims=True)
    mix = np.tanh(mix * 1.08)
    peak = float(np.max(np.abs(mix)))
    if peak > 0.0:
        mix *= (10.0 ** (-1.0 / 20.0)) / peak

    write_stereo_wav(output_path, mix)
    rms = float(np.sqrt(np.mean(mix * mix)))
    print(f"Rendered: {output_path}")
    print(f"Duration: {total_seconds:.2f}s | Peak: -1.00 dBFS | RMS: {20 * math.log10(max(rms, 1e-12)):.2f} dBFS")
    print(f"Unique engine renders: {len(cache)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        type=Path,
        default=REPO_ROOT / "audio_out/refine/dx7_marimba/dx7_marimba_targeted_v0.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "audio_out/compositions/algorithm_7_courtyard.wav",
    )
    args = parser.parse_args()
    compose(args.preset.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
