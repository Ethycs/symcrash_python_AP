"""Render the DX7/stock/wild batches to audio_out/new_presets/ for audition,
with motion verification (centroid trajectory). Temporary script."""
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
OUT = Path("audio_out/new_presets")
OUT.mkdir(parents=True, exist_ok=True)

BATCH = [
    ("dx_epiano_v0", 60, 2.5), ("dx_lately_bass_v0", 45, 1.5),
    ("dx_marimba_v0", 69, 1.5), ("dx_clav_v0", 57, 1.5),
    ("acid_bass_v0", 45, 1.5), ("detroit_bass_v0", 45, 1.5),
    ("analog_strings_v0", 57, 3.0), ("rnb_slide_v0", 64, 2.5),
    ("prime_storm_v0", 57, 2.5), ("fold_drone_v0", 45, 4.0),
    ("quasi_glass_v0", 69, 2.5), ("shear_vox_v0", 57, 3.0),
]


def _comb(sig, d, g):
    try:
        from scipy.signal import lfilter
        b = np.zeros(d + 1)
        b[d] = 1.0
        a = np.zeros(d + 1)
        a[0] = 1.0
        a[d] = -g
        return lfilter(b, a, sig)
    except ImportError:
        y = np.zeros_like(sig)
        for i in range(d, len(sig)):
            y[i] = sig[i - d] + g * y[i - d]
        return y


def _allpass(sig, d, g):
    try:
        from scipy.signal import lfilter
        b = np.zeros(d + 1)
        b[0] = -g
        b[d] = 1.0
        a = np.zeros(d + 1)
        a[0] = 1.0
        a[d] = -g
        return lfilter(b, a, sig)
    except ImportError:
        y = np.zeros_like(sig)
        for i in range(len(sig)):
            xd = sig[i - d] if i >= d else 0.0
            yd = y[i - d] if i >= d else 0.0
            y[i] = -g * sig[i] + xd + g * yd
        return y


def audition_fx(x_int16):
    """
    Audition monitoring chain: light dual-tap chorus + small Schroeder
    reverb. This is NOT engine output — it stands in for the live system's
    output FX stage so dry timbres can be judged in context (Welsh notes
    'thick chorus and delay' for several patches).
    """
    s = x_int16.astype(np.float64) / 32768.0
    n = len(s)
    t = np.arange(n) / FS

    # Chorus: two slowly-modulated delay taps
    out = s.copy()
    for rate, phase, base_ms in ((0.8, 0.0, 12.0), (0.67, 2.1, 17.0)):
        delay = (base_ms + 4.0 * np.sin(2 * np.pi * rate * t + phase)) * FS / 1000.0
        idx = np.arange(n) - delay
        i0 = np.clip(idx.astype(np.int64), 0, n - 1)
        i1 = np.clip(i0 + 1, 0, n - 1)
        frac = idx - i0
        tap = s[i0] * (1 - frac) + s[i1] * frac
        tap[idx < 0] = 0.0
        out += 0.30 * tap

    # Small room: 4 combs + 2 allpasses, ~22% wet
    rev_in = out
    rev = np.zeros_like(rev_in)
    for d_ms, g in ((29.7, 0.77), (37.1, 0.75), (41.1, 0.73), (43.7, 0.71)):
        rev += _comb(rev_in, int(d_ms * FS / 1000), g)
    rev /= 4.0
    rev = _allpass(rev, int(5.0 * FS / 1000), 0.7)
    rev = _allpass(rev, int(1.7 * FS / 1000), 0.7)
    out = out + 0.22 * rev

    peak = np.abs(out).max() or 1.0
    return (out / peak * 0.9 * 32767).astype(np.int16)


def centroid_at(s, lo_frac, hi_frac):
    seg = s[int(len(s) * lo_frac): int(len(s) * hi_frac)].astype(np.float64)
    if len(seg) < 256 or np.abs(seg).max() < 1:
        return 0.0
    seg = (seg - seg.mean()) * np.hanning(len(seg))
    sp = np.abs(np.fft.rfft(seg)) ** 2
    fr = np.fft.rfftfreq(len(seg), 1 / FS)
    return float((fr * sp).sum() / max(sp.sum(), 1e-12))


print(f"{'preset':20s} {'rms':>8s} {'peak':>6s} {'cent@10%':>9s} {'cent@40%':>9s} {'cent@70%':>9s}")
for stem, note, dur in BATCH:
    try:
        clear_basis_atlas()
        ir = load_instrument(f"presets/instruments/{stem}.yaml")
        s = np.asarray(render_preset_note(ir, note=note, duration_sec=dur), dtype=np.int16)
    except Exception as e:
        print(f"{stem:20s} FAILED: {e}")
        continue
    with wave.open(str(OUT / f"{stem}.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes(s.astype("<i2").tobytes())
    fx = audition_fx(s)
    with wave.open(str(OUT / f"{stem}_fx.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes(fx.astype("<i2").tobytes())
    rms = float(np.sqrt((s.astype(np.float64) ** 2).mean()))
    peak = int(np.abs(s.astype(np.int32)).max())
    c1 = centroid_at(s, 0.05, 0.2)
    c2 = centroid_at(s, 0.35, 0.5)
    c3 = centroid_at(s, 0.65, 0.8)
    print(f"{stem:20s} {rms:8.1f} {peak:6d} {c1:8.0f}H {c2:8.0f}H {c3:8.0f}H")
print(f"\nWAVs in {OUT}")
