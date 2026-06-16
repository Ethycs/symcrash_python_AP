"""Compose and render a short marimba piece with dx_marimba_v0.

Three voices over Am - F - C - G (two bars each, 112 BPM):
  - left hand: root/fifth eighth-note ostinato
  - right hand: chord-tone 16th arpeggios with accents
  - melody: sparse pentatonic line on top
Humanized velocity/timing (seeded, deterministic). Mixed from per-note
engine renders; finished with the audition chorus+reverb chain.
"""
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "tools/render")

from render_preset import render_preset_note
from symcrash_python_ap.instrument_ir import load_instrument
from symcrash_python_ap.vm.harmonic_bank import clear_basis_atlas

FS = 48000
BPM = float(sys.argv[3]) if len(sys.argv) > 3 else 112.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
SIX = BEAT / 4

rng = np.random.RandomState(42)

# ---------------------------------------------------------------- composition
# (chord, root_fifth, arp_pattern, register notes are MIDI)
CHORDS = [
    ("Am", (45, 52), [57, 60, 64, 69, 72, 69, 64, 60]),
    ("F",  (41, 48), [53, 57, 60, 65, 69, 65, 60, 57]),
    ("C",  (48, 55), [60, 64, 67, 72, 76, 72, 67, 64]),
    ("G",  (43, 50), [55, 59, 62, 67, 71, 67, 62, 59]),
]

# Melody: (bar, beat, midi, velocity)
MELODY = [
    (0, 0.0, 69, 0.95), (0, 1.5, 72, 0.8), (0, 3.0, 76, 0.9),
    (1, 0.0, 74, 0.85), (1, 1.0, 72, 0.75), (1, 2.0, 69, 0.8),
    (2, 0.0, 65, 0.9), (2, 1.0, 69, 0.8), (2, 2.0, 72, 0.9),
    (3, 0.0, 74, 0.85), (3, 2.0, 72, 0.8),
    (4, 0.0, 76, 0.95), (4, 1.5, 79, 0.85), (4, 3.0, 76, 0.8),
    (5, 0.0, 74, 0.85), (5, 1.0, 72, 0.75), (5, 2.0, 67, 0.8),
    (6, 0.0, 71, 0.9), (6, 1.5, 74, 0.8), (6, 3.0, 79, 0.9),
    (7, 0.0, 76, 0.85), (7, 1.0, 74, 0.8), (7, 2.0, 71, 0.75),
]

events = []  # (time_sec, midi, velocity)

for bar in range(8):
    chord_name, (root, fifth), arp = CHORDS[bar // 2]
    bar_t = bar * BAR
    # Left hand: eighth-note root/fifth ostinato
    for e in range(8):
        note = root if e % 2 == 0 else fifth
        vel = 0.5 + rng.uniform(-0.06, 0.06)
        events.append((bar_t + e * BEAT / 2, note, vel))
    # Right hand: 16th arpeggio, accent the start of each beat
    for s in range(16):
        note = arp[s % 8]
        vel = (0.62 if s % 4 == 0 else 0.42) + rng.uniform(-0.05, 0.05)
        events.append((bar_t + s * SIX, note, vel))

for bar, beat, note, vel in MELODY:
    events.append((bar * BAR + beat * BEAT, note, vel))

# Final rolled Am chord, let it ring
end_t = 8 * BAR
for i, note in enumerate([45, 52, 57, 60, 64, 69]):
    events.append((end_t + i * 0.035, note, 0.85 - i * 0.04))

# ---------------------------------------------------------------- rendering
PRESET = sys.argv[1] if len(sys.argv) > 1 else "dx_marimba_v0"
OUTNAME = sys.argv[2] if len(sys.argv) > 2 else "marimba_music"
print(f"{len(events)} notes, {end_t + 2.5:.1f} s, preset={PRESET}")
ir = load_instrument(f"presets/instruments/{PRESET}.yaml")

# Velocity buckets: real velocity into the engine so velocity_sens
# envelopes (velocity -> FM index) fire per dynamic level. Key scaling
# is now engine-native (Amp key_scale_rate in the preset).
VEL_BUCKETS = [0.45, 0.65, 0.85, 1.0]


def bucket(vel):
    return min(VEL_BUCKETS, key=lambda b: abs(b - vel))


note_cache = {}
for midi, vb in sorted({(e[1], bucket(e[2])) for e in events}):
    clear_basis_atlas()
    s = np.asarray(render_preset_note(ir, note=midi, duration_sec=1.1,
                                      velocity_q15=int(vb * 32767)),
                   dtype=np.float64) / 32768.0
    note_cache[(midi, vb)] = s
print(f"rendered {len(note_cache)} (pitch, velocity) variants")

total = int((end_t + 2.5) * FS)
mix = np.zeros(total)
for t, midi, vel in events:
    t += rng.uniform(-0.004, 0.004)  # humanize timing
    start = max(0, int(t * FS))
    vb = bucket(vel)
    # engine handled velocity gain at the bucket level; fine-trim residual
    s = note_cache[(midi, vb)] * (vel / vb)
    end = min(total, start + len(s))
    mix[start:end] += s[: end - start]

mix /= max(np.abs(mix).max(), 1e-9)
mix *= 0.85

# ---------------------------------------------------------------- space (audition FX)
def _comb(sig, d, g):
    from scipy.signal import lfilter
    b = np.zeros(d + 1); b[d] = 1.0
    a = np.zeros(d + 1); a[0] = 1.0; a[d] = -g
    return lfilter(b, a, sig)

def _allpass(sig, d, g):
    from scipy.signal import lfilter
    b = np.zeros(d + 1); b[0] = -g; b[d] = 1.0
    a = np.zeros(d + 1); a[0] = 1.0; a[d] = -g
    return lfilter(b, a, sig)

BIG_HALL = "--hall" in sys.argv
try:
    rev = np.zeros_like(mix)
    if BIG_HALL:  # floaty: long wet hall
        combs = ((47.3, 0.84), (59.1, 0.82), (65.7, 0.80), (70.2, 0.78))
        wet, ap1, ap2 = 0.55, 8.1, 2.8
    else:  # recipe: a nice room for space
        combs = ((29.7, 0.78), (37.1, 0.76), (41.1, 0.74), (43.7, 0.72))
        wet, ap1, ap2 = 0.32, 5.0, 1.7
    for d_ms, g in combs:
        rev += _comb(mix, int(d_ms * FS / 1000), g)
    rev /= 4.0
    rev = _allpass(rev, int(ap1 * FS / 1000), 0.7)
    rev = _allpass(rev, int(ap2 * FS / 1000), 0.7)
    mix = mix + wet * rev
    mix /= max(np.abs(mix).max(), 1e-9)
    mix *= 0.9
except ImportError:
    print("scipy missing - writing dry")

out = Path(f"audio_out/{OUTNAME}.wav")
out.parent.mkdir(exist_ok=True)
with wave.open(str(out), "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(FS)
    w.writeframes((mix * 32767).astype("<i2").tobytes())
print(f"wrote {out} ({len(mix) / FS:.1f} s)")
