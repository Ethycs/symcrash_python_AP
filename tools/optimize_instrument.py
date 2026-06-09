"""
Instrument Optimization Workflow
================================

Analysis-driven preset authoring.  Parse a target instrument waveform, then
iteratively optimize a preset so its RENDERED output matches the target's
measured harmonic spectrum and amplitude envelope.

The objective is fully numerical — no listening required.  Because the engine
colours the input partials (IDWPT realisation + behaviour operators), the loop
optimises the *input* partials so the *output* spectrum matches the target — a
closed-loop deconvolution of the engine itself.  The residual loss is also a
direct, honest measure of the engine's expressive ceiling for that instrument.

Pipeline
--------
  1. analyze_target(wav)      -> {f0, note, harmonics[H], adsr}
  2. render_steady(partials)  -> rendered steady-state samples (through the VM)
  3. extract_harmonics(...)   -> rendered harmonic profile
  4. loss(target, rendered)   -> log-magnitude harmonic distance
  5. optimize(...)            -> partials minimising loss (gradient-free)
  6. write preset + report + best-candidate WAV

Usage
-----
  pixi run python tools/optimize_instrument.py violin.wav \\
      --note 69 --harmonics 12 --iters 400 \\
      --out presets/instruments/violin_opt_v0.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav
from scipy.optimize import minimize

import sys
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from symcrash_python_ap.vm.executor import vm_step_block       # noqa: E402
from symcrash_python_ap.vm.state import VMState                 # noqa: E402
from symcrash_python_ap.vm.harmonic_bank import clear_basis_atlas  # noqa: E402
from symcrash_python_ap.frames import VoiceFrame                # noqa: E402
from symcrash_python_ap.instrument_ir import derive_operators_from_emphasis  # noqa: E402

FS = 48000
BLOCK = 64
Q15 = 32767


# ---------------------------------------------------------------------------
# 1. Target analysis
# ---------------------------------------------------------------------------
def midi_to_hz(note: int) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def analyze_target(wav_path: str, note: int | None, n_harm: int) -> dict:
    fs, x = wav.read(wav_path)
    if x.ndim > 1:
        x = x[:, 0]
    x = x.astype(float)
    x /= (np.abs(x).max() or 1.0)

    # amplitude envelope (rectify + smooth) for ADSR estimate
    env = np.abs(x)
    win = max(1, fs // 200)
    env = np.convolve(env, np.ones(win) / win, mode="same")

    # steady segment for harmonic analysis (middle third)
    seg = x[len(x) // 3: len(x) // 3 + fs // 2]
    seg = (seg - seg.mean()) * np.hanning(len(seg))
    X = np.abs(np.fft.rfft(seg))
    f = np.fft.rfftfreq(len(seg), 1 / fs)

    if note is not None:
        f0 = midi_to_hz(note)
    else:
        band = (f > 60) & (f < 1000)
        f0 = f[band][np.argmax(X[band])]
        note = int(round(69 + 12 * np.log2(f0 / 440.0)))

    harm = []
    for h in range(1, n_harm + 1):
        fc = f0 * h
        m = (f > fc - f0 * 0.4) & (f < fc + f0 * 0.4)
        harm.append(float(X[m].max()) if m.any() else 0.0)
    harm = np.array(harm)
    harm /= (harm.max() or 1.0)

    return {
        "fs": fs, "f0": f0, "note": note,
        "harmonics": harm,
        "adsr": estimate_adsr(env, fs),
    }


def estimate_adsr(env: np.ndarray, fs: int) -> dict:
    peak = env.max() or 1.0
    attack_end = int(np.argmax(env))
    attack_ms = max(1.0, attack_end / fs * 1000.0)
    sustain_level = float(np.median(env[attack_end: attack_end + fs // 2]) / peak) if attack_end < len(env) else 0.7
    # release: time from end-of-note back to where env falls below 10%
    tail = env[-fs // 2:] if len(env) > fs // 2 else env
    rel_ms = max(40.0, len(tail) / fs * 1000.0 * 0.5)
    return {"attack": round(attack_ms, 1), "decay": 150.0,
            "sustain": round(min(0.95, max(0.2, sustain_level)), 2),
            "release": round(rel_ms, 1)}


# ---------------------------------------------------------------------------
# 2-3. Candidate rendering + feature extraction (through the real VM)
# ---------------------------------------------------------------------------
def render_steady(partials: np.ndarray, note: int, axes: dict,
                  n_blocks: int = 180) -> np.ndarray:
    """Render a constant-gain steady tone through the VM (operators applied)."""
    clear_basis_atlas()
    amps = [int(max(0.0, min(1.0, p)) * Q15) for p in partials]
    ops = derive_operators_from_emphasis(axes)
    st = VMState.create(voice_id=0)
    st.reset_for_note_on(note=note)
    out = []
    for _ in range(n_blocks):
        f = VoiceFrame(note=note, coeff_osc_mode=1, H_request=len(amps),
                       gain_q15=24000, beta_target_q15=0,
                       partial_amps_q15=amps, ops_enable=ops,
                       # waveform_type=0 lets custom_harmonics (partials) drive
                       # the spectrum; non-zero forces a built-in waveform that
                       # OVERRIDES the partials (see vm/executor priority order).
                       families=dict(axes), waveform_type=0)
        st, o = vm_step_block(st, f)
        out.extend(o.tolist())
    return np.array(out, dtype=float)


def extract_harmonics(samples: np.ndarray, f0: float, n_harm: int) -> np.ndarray:
    # analyse the back half (skip IDWPT warm-up)
    s = samples[len(samples) // 2:]
    if len(s) < 256:
        return np.zeros(n_harm)
    s = (s - s.mean()) * np.hanning(len(s))
    X = np.abs(np.fft.rfft(s))
    f = np.fft.rfftfreq(len(s), 1 / FS)
    harm = []
    for h in range(1, n_harm + 1):
        fc = f0 * h
        m = (f > fc - f0 * 0.4) & (f < fc + f0 * 0.4)
        harm.append(float(X[m].max()) if m.any() else 0.0)
    harm = np.array(harm)
    harm /= (harm.max() or 1.0)
    return harm


# ---------------------------------------------------------------------------
# 4. Loss
# ---------------------------------------------------------------------------
def harmonic_loss(target: np.ndarray, rendered: np.ndarray) -> float:
    # log-magnitude distance (perceptual-ish), floor to avoid log(0)
    eps = 1e-3
    t = np.log10(target + eps)
    r = np.log10(rendered + eps)
    return float(np.mean((t - r) ** 2))


# ---------------------------------------------------------------------------
# 5. Optimize
# ---------------------------------------------------------------------------
def optimize(target: dict, axes: dict, n_harm: int, iters: int,
             verbose: bool = True) -> tuple[np.ndarray, float, list]:
    note, f0 = target["note"], target["f0"]
    tgt = target["harmonics"]

    history = []

    def objective(p):
        p = np.clip(p, 0.0, 1.0)
        rendered = render_steady(p, note, axes)
        h = extract_harmonics(rendered, f0, n_harm)
        L = harmonic_loss(tgt, h)
        history.append(L)
        return L

    # init at the target profile (best first guess), let the loop deconvolve
    x0 = np.clip(tgt.copy(), 0.0, 1.0)
    res = minimize(objective, x0, method="Powell",
                   options={"maxiter": iters, "xtol": 1e-3, "ftol": 1e-4})
    best = np.clip(res.x, 0.0, 1.0)
    best /= (best.max() or 1.0)
    return best, float(res.fun), history


# ---------------------------------------------------------------------------
# 6. Output
# ---------------------------------------------------------------------------
PRESET_TEMPLATE = """# {name} - optimized against {target}
# ==============================================================
# Auto-generated by tools/optimize_instrument.py
# Harmonic profile fitted to the real waveform (closed measurement loop).
# Harmonic loss (log-mag MSE): {loss:.4f}   (lower = closer to target)
# Conformance: Expressive (full articulation)

instrument_ir_v0:
  meta:
    id: "{ident}"
    name: "{name}"
    version: 0
    author: "symcrash-opt"
    seed: 0x{seed:08X}

  audio:
    sample_rate: 48000
    block_size: 64

  graph:
    osc1:
      type: Osc
      output: amp1
      waveform: partials
      partials: [{partials}]
    amp1:
      type: Amp
      attack: {attack}
      decay: {decay}
      sustain: {sustain}
      release: {release}

  behavior:
    idle:
      emphasis: {{}}
      trees: {{A: harmonic, B: harmonic, blend: 0.0}}
      resolution: 8
      safety: MIN
    stable:
      emphasis: {{{emphasis}}}
      trees: {{A: harmonic, B: harmonic, blend: 0.0}}
      resolution: 32
      safety: FULL
      ramp: 15
      traction: 1.0

  compile:
    material: {{blend: 0.0, target: 0.0, ramp: 1}}
"""


def write_preset(path: str, target: dict, partials: np.ndarray, axes: dict,
                 loss: float, name: str):
    ident = Path(path).stem.replace("_", "-")
    adsr = target["adsr"]
    emphasis = ", ".join(f"{k}: {v}" for k, v in axes.items())
    text = PRESET_TEMPLATE.format(
        name=name, target=Path(target.get("path", "target")).name,
        ident=ident, seed=abs(hash(ident)) & 0xFFFFFFFF,
        partials=", ".join(f"{p:.3f}" for p in partials),
        attack=adsr["attack"], decay=adsr["decay"],
        sustain=adsr["sustain"], release=adsr["release"],
        emphasis=emphasis, loss=loss,
    )
    Path(path).write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Optimize a preset to match a target waveform")
    ap.add_argument("target", help="Target instrument WAV")
    ap.add_argument("--note", type=int, default=None, help="MIDI note (else detected)")
    ap.add_argument("--harmonics", type=int, default=12)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--out", default="presets/instruments/optimized_v0.yaml")
    ap.add_argument("--name", default="Optimized")
    ap.add_argument("--axes", default="color:0.3", help="comma axis:weight list for the stable state")
    args = ap.parse_args()

    axes = {}
    for tok in args.axes.split(","):
        if ":" in tok:
            k, v = tok.split(":")
            axes[k.strip()] = float(v)

    print(f"Analyzing target: {args.target}")
    target = analyze_target(args.target, args.note, args.harmonics)
    target["path"] = args.target
    print(f"  note={target['note']}  f0={target['f0']:.1f}Hz  adsr={target['adsr']}")
    print(f"  target harmonics: {' '.join(f'{h:.3f}' for h in target['harmonics'])}")

    # baseline loss at the raw target profile (no deconvolution)
    base_h = extract_harmonics(render_steady(target['harmonics'], target['note'], axes),
                               target['f0'], args.harmonics)
    base_loss = harmonic_loss(target['harmonics'], base_h)
    print(f"  baseline loss (partials = raw target): {base_loss:.4f}")

    print(f"Optimizing partials ({args.iters} iters, Powell)...")
    best, loss, hist = optimize(target, axes, args.harmonics, args.iters)
    print(f"  optimized loss: {loss:.4f}  (improvement {base_loss - loss:+.4f})")
    print(f"  optimized partials: {' '.join(f'{p:.3f}' for p in best)}")

    write_preset(args.out, target, best, axes, loss, args.name)
    print(f"Wrote optimized preset: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
