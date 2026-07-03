#!/usr/bin/env python
"""
patch_gym.py - WS5 reversal-gym harness (AUTHORING-IMPROVEMENT-PLAN WS5).

Automates the blind patch-reversal exam loop so rounds are reproducible
instead of hand-orchestrated:

  prep    render anonymized target WAVs from a source preset, extract the
          answer key, and emit the exam prompt (contract + deliverables)
          for a fresh agent session. The answer key lives in a sibling
          `answer/` dir the exam contract denies.
  grade   score a candidate YAML against the answer key: topology up to
          operator relabeling (gauge), parameter deltas under the best-fit
          permutation, advisory audio comparison at a held-out note.
  ledger  append a graded round to tools/gym/ledger.jsonl (the machine
          ledger; prose lessons go into playbook ch. 25).

Run via `pixi run python tools/gym/patch_gym.py ...`.

Grading gauge rules (02 - Optimization Geometry: we want the perceptual
equivalence class, not the "true" patch):
  - structure is compared by graph isomorphism over all 720 operator
    relabelings (alg 5 vs 6 style engine-equivalences fall out for free);
  - feedback position is scored separately from the iso (its self-fb
    approximation makes it the weakest structural observable);
  - carrier levels are compared as fractions of the carrier sum (the E2
    gateway auto-normalizes, so absolute carrier level is a flat direction).
"""

import argparse
import itertools
import json
import math
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RENDER_TOOL = REPO_ROOT / "tools" / "render" / "render_preset.py"
AUDIO_OUT = REPO_ROOT / "audio_out" / "presets"
LEDGER_PATH = REPO_ROOT / "tools" / "gym" / "ledger.jsonl"
PLAYBOOK_CH25 = (
    "symcrash_docs/05 - Standards/05 - Library/"
    "WSE-STD-PRESET-AUTHORING/25-agentic-patch-analysis.md"
)

try:
    from symcrash_python_ap.vm.fm_algorithms import get_algorithm
except ImportError:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from symcrash_python_ap.vm.fm_algorithms import get_algorithm


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_ir(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return doc["instrument_ir_v0"]


def find_fm6_node(ir: dict):
    for name, node in ir.get("graph", {}).items():
        if isinstance(node, dict) and node.get("type") == "FM6":
            return name, node
    return None, None


def find_amp_node(ir: dict):
    for name, node in ir.get("graph", {}).items():
        if isinstance(node, dict) and node.get("type") == "Amp":
            return name, node
    return None, None


def render_note(preset_path: Path, note: int, duration: float) -> Path:
    """Render via the canonical harness; return the produced WAV path."""
    subprocess.run(
        [sys.executable, str(RENDER_TOOL), str(preset_path),
         "--note", str(note), "--duration", str(duration)],
        check=True, capture_output=True, cwd=REPO_ROOT,
    )
    wav = AUDIO_OUT / f"{preset_path.stem}_note{note}.wav"
    if not wav.exists():
        raise FileNotFoundError(f"render harness did not produce {wav}")
    return wav


def read_wav_mono(path: Path):
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = np.frombuffer(w.readframes(n), dtype=np.int16)
        if w.getnchannels() == 2:
            raw = raw.reshape(-1, 2).mean(axis=1)
    return raw.astype(np.float64) / 32768.0, sr


# ---------------------------------------------------------------------------
# prep
# ---------------------------------------------------------------------------

EXAM_PROMPT_TEMPLATE = """You are taking a blind patch-reversal exam in the symcrash repo at {repo} (ALWAYS run python via `pixi run python`).

## Task

{n_wavs} WAV renders of ONE unknown DX7 factory voice, played through this repo's FM6 engine ({duration} s each, 48 kHz):

{wav_list}

Reverse-engineer the patch FROM THE AUDIO ALONE. The voice uses one of the 32 canonical DX7 algorithms in src/symcrash_python_ap/vm/fm_algorithms.py (you MAY read that file — it is the hypothesis space, not the answer).

## Exam contract (honor system — violations void the exam)

DENIED (do not read, grep, or otherwise access):
{deny_list}
Also denied: web search or recall of DX7 factory patch data as a substitute for measurement. If you recognize the voice, say so in your report, but every claim must be derived from a measurement you actually ran.

ALLOWED: src/symcrash_python_ap/ (the engine), tools/lib/fft_utils.py, tools/analyze_fm6_waveform.py, tools/render/render_preset.py, non-DX7 presets as YAML schema examples, and the playbook: {playbook} — follow its forward procedure and its Lessons ledger (known engine artifacts and validated predicates live there).

## Deliverable (your final message is data for the grader, not prose)

1. MEASUREMENTS: f0 per note; partial inventory at the middle note (freq, ratio-to-f0, rel dB, decay); centroid/RMS trajectories; cross-note observations.
2. PREDICATES FIRED: each inference as "observation -> constraint".
3. TOPOLOGY VERDICT: carrier chains, mod structure, feedback (state your evidence explicitly — check for excess high-harmonic energy above the fitted Bessel profile before ruling feedback out); ranked algorithm candidates with discriminating evidence; final pick.
4. PARAMETER ESTIMATES: per-op ratio/level, envelopes, confidence each.
5. SELF-SCORE: render your candidate ({candidate_name}) at note {mid_note} and compare against the target measurements. At most 2 fit iterations.
6. INSTRUMENTATION VERDICT: most/least informative measurements; what measurement you wished you had.
7. Paths of candidate YAML(s), renders, and analysis scripts (put all of them in {exam_dir}).
8. HONESTY NOTE: recognition, contamination, any claim from recall.
"""


def cmd_prep(args):
    preset_path = Path(args.preset).resolve()
    ir = load_ir(preset_path)
    exam_dir = Path(args.exam_dir).resolve() / args.round_name
    answer_dir = exam_dir / "answer"
    answer_dir.mkdir(parents=True, exist_ok=True)

    notes = [int(n) for n in args.notes.split(",")]
    wav_paths = []
    for n in notes:
        src = render_note(preset_path, n, args.duration)
        dst = exam_dir / f"target_note{n}.wav"
        shutil.copyfile(src, dst)
        wav_paths.append(dst)

    # Answer key: everything the grader needs, nothing the examinee sees.
    _, fm6 = find_fm6_node(ir)
    _, amp = find_amp_node(ir)
    key = {
        "source_preset": str(preset_path.relative_to(REPO_ROOT)),
        "voice_name": ir.get("meta", {}).get("name"),
        "algorithm": fm6.get("algorithm"),
        "feedback": fm6.get("feedback", 0),
        "operators": fm6.get("operators", []),
        "amp": {k: amp.get(k) for k in
                ("attack", "decay", "sustain", "release",
                 "velocity_sens", "key_scale_rate") if amp and k in amp},
        "envelopes": ir.get("envelopes", {}),
        "notes": notes,
        "duration": args.duration,
    }
    with open(answer_dir / "answer_key.json", "w", encoding="utf-8") as f:
        json.dump(key, f, indent=2)

    # Deny-list: the answer dir, all DX7 artifacts, and any preset sharing a
    # name token with the source (round-1 lesson: dx_epiano leaked the prior).
    tokens = {t for t in preset_path.stem.replace("dx7_", "").split("_")
              if t not in ("v0", "v1", "1", "2")}
    kin = sorted({p.name for p in (REPO_ROOT / "presets" / "instruments").glob("*.yaml")
                  if any(t in p.stem for t in tokens)})
    deny = [
        str(answer_dir),
        "presets/instruments/dx7_*  (all DX7 imports)",
        "ingest/  (banks + import reports)",
        "tools/import_dx7_sysex.py",
        *(f"presets/instruments/{name}  (name-kin of the target)" for name in kin),
        *(args.extra_deny or []),
    ]

    prompt = EXAM_PROMPT_TEMPLATE.format(
        repo=REPO_ROOT,
        n_wavs=len(wav_paths),
        duration=args.duration,
        wav_list="\n".join(f"- {p}" for p in wav_paths),
        deny_list="\n".join(f"- {d}" for d in deny),
        playbook=PLAYBOOK_CH25,
        candidate_name="candidate_v1.yaml / candidate_v2.yaml",
        mid_note=notes[len(notes) // 2],
        exam_dir=exam_dir,
    )
    (exam_dir / "exam_prompt.md").write_text(prompt, encoding="utf-8")

    print(f"Exam ready: {exam_dir}")
    print(f"  targets : {', '.join(p.name for p in wav_paths)}")
    print(f"  prompt  : exam_prompt.md  (feed to a FRESH agent session)")
    print(f"  key     : answer/answer_key.json  (grader only)")


# ---------------------------------------------------------------------------
# grade
# ---------------------------------------------------------------------------

def op_params(operators):
    return [(float(o["ratio"]), float(o["level"])) for o in operators]


def structure_isos(cand_alg: int, truth_alg: int):
    """All operator permutations mapping candidate topology onto truth's.

    perm[i] = truth op index that candidate op i plays the role of.
    Returns (isos, fb_preserving_isos).
    """
    A, B = get_algorithm(cand_alg), get_algorithm(truth_alg)
    isos, fb_isos = [], []
    for perm in itertools.permutations(range(6)):
        ok = all(
            A.modulation_matrix[i, j] == B.modulation_matrix[perm[i], perm[j]]
            for i in range(6) for j in range(6)
        ) and all(A.is_carrier(i) == B.is_carrier(perm[i]) for i in range(6))
        if ok:
            isos.append(perm)
            if (A.feedback_op == -1) == (B.feedback_op == -1) and (
                    A.feedback_op == -1 or perm[A.feedback_op] == B.feedback_op):
                fb_isos.append(perm)
    return isos, fb_isos


def cents(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return float("inf")
    return abs(1200.0 * math.log2(a / b))


def best_perm(isos, cand_ops, truth_ops):
    """Among valid isos, the one minimizing summed ratio distance (cents)."""
    def cost(perm):
        return sum(min(cents(cand_ops[i][0], truth_ops[perm[i]][0]), 2400.0)
                   for i in range(6))
    return min(isos, key=cost) if isos else None


def audio_compare(cand_wav: Path, truth_wav: Path) -> dict:
    x, sr = read_wav_mono(cand_wav)
    y, _ = read_wav_mono(truth_wav)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    def peaks(sig, k=20):
        win = sig[: min(len(sig), sr // 2)] * np.hanning(min(len(sig), sr // 2))
        mag = np.abs(np.fft.rfft(win, 4 * len(win)))
        freqs = np.fft.rfftfreq(4 * len(win), 1.0 / sr)
        idx = [i for i in np.argsort(mag)[::-1]
               if 20 < freqs[i] < 16000][: k * 8]
        # greedy pick separated by >20 Hz
        chosen = []
        for i in idx:
            if all(abs(freqs[i] - freqs[j]) > 20 for j in chosen):
                chosen.append(i)
            if len(chosen) == k:
                break
        ref = max(mag[chosen]) if chosen else 1.0
        return sorted(
            (float(freqs[i]), float(20 * np.log10(mag[i] / ref + 1e-12)))
            for i in chosen
        )

    def decay_60(sig):
        rms = np.sqrt(np.convolve(sig ** 2, np.ones(2048) / 2048, "same"))
        peak = rms.max()
        below = np.nonzero(rms > peak * 1e-3)[0]
        return float(below[-1] / sr) if len(below) else 0.0

    px, py = peaks(x), peaks(y)
    matched, level_err = 0, []
    for fy, dby in py:
        near = [(fx, dbx) for fx, dbx in px if abs(fx - fy) < max(5.0, fy * 0.005)]
        if near:
            matched += 1
            level_err.append(abs(near[0][1] - dby))
    return {
        "peaks_truth": len(py),
        "peaks_matched": matched,
        "mean_level_err_db": round(float(np.mean(level_err)), 2) if level_err else None,
        "decay60_cand_s": round(decay_60(x), 3),
        "decay60_truth_s": round(decay_60(y), 3),
    }


def cmd_grade(args):
    exam_dir = Path(args.exam).resolve()
    with open(exam_dir / "answer" / "answer_key.json", encoding="utf-8") as f:
        key = json.load(f)
    cand_path = Path(args.candidate).resolve()
    cand_ir = load_ir(cand_path)
    _, cand_fm6 = find_fm6_node(cand_ir)
    _, cand_amp = find_amp_node(cand_ir)

    cand_alg = cand_fm6.get("algorithm")
    truth_alg = key["algorithm"]
    isos, fb_isos = structure_isos(cand_alg, truth_alg)
    structure = ("EXACT (incl. feedback position)" if fb_isos
                 else "ISO (feedback position differs)" if isos
                 else "MISMATCH")

    cand_ops = op_params(cand_fm6["operators"])
    truth_ops = op_params(key["operators"])
    # Match over ALL isos: pinning the fb op would force chain mis-pairing
    # when the candidate got feedback wrong (round-1 harness lesson).
    perm = best_perm(isos, cand_ops, truth_ops)

    per_op = []
    if perm:
        alg = get_algorithm(truth_alg)
        c_carrier_sum = sum(l for i, (_, l) in enumerate(cand_ops)
                            if alg.is_carrier(perm[i])) or 1.0
        t_carrier_sum = sum(l for i, (_, l) in enumerate(truth_ops)
                            if alg.is_carrier(i)) or 1.0
        for i in range(6):
            (cr, cl), (tr, tl) = cand_ops[i], truth_ops[perm[i]]
            is_c = alg.is_carrier(perm[i])
            cl_n = cl / c_carrier_sum if is_c else cl
            tl_n = tl / t_carrier_sum if is_c else tl
            per_op.append({
                "cand_op": i + 1, "truth_op": perm[i] + 1,
                "role": "carrier" if is_c else "modulator",
                "ratio_err_cents": round(cents(cr, tr), 1),
                "level_err_db": round(abs(20 * math.log10((cl_n + 1e-6) / (tl_n + 1e-6))), 1),
            })

    amp_err = {}
    for k in ("attack", "decay", "sustain", "release", "key_scale_rate"):
        tv, cv = key["amp"].get(k), (cand_amp or {}).get(k)
        if tv is not None and cv is not None:
            amp_err[k] = {"truth": tv, "cand": cv,
                          "rel_err": round(abs(cv - tv) / (abs(tv) + 1e-9), 3)}

    fb_err = {"truth": key["feedback"], "cand": cand_fm6.get("feedback", 0)}

    audio = None
    if not args.no_audio:
        note = args.note
        truth_wav = exam_dir / f"target_note{note}.wav"
        if not truth_wav.exists():
            truth_wav = render_note(
                REPO_ROOT / key["source_preset"], note, key["duration"])
        cand_wav = render_note(cand_path, note, key["duration"])
        audio = audio_compare(cand_wav, truth_wav)

    report = {
        "exam": str(exam_dir), "candidate": str(cand_path),
        "voice": key["voice_name"],
        "algorithm": {"truth": truth_alg, "cand": cand_alg,
                      "structure": structure,
                      "n_isos": len(isos), "n_fb_isos": len(fb_isos)},
        "op_mapping": per_op, "amp_env": amp_err, "feedback": fb_err,
        "audio_advisory": audio,
    }
    out = exam_dir / "answer" / "grade_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"VOICE: {key['voice_name']}   (truth alg {truth_alg}, cand alg {cand_alg})")
    print(f"STRUCTURE: {structure}")
    if per_op:
        print("OP MAP (cand->truth, best-fit gauge):")
        for r in per_op:
            print(f"  op{r['cand_op']}->op{r['truth_op']} [{r['role']:9s}] "
                  f"ratio err {r['ratio_err_cents']:7.1f} cents, "
                  f"level err {r['level_err_db']:5.1f} dB")
    print(f"FEEDBACK: truth {fb_err['truth']}, cand {fb_err['cand']}")
    for k, v in amp_err.items():
        print(f"AMP {k}: truth {v['truth']}, cand {v['cand']} (rel err {v['rel_err']})")
    if audio:
        print(f"AUDIO (advisory): {audio['peaks_matched']}/{audio['peaks_truth']} peaks matched, "
              f"mean level err {audio['mean_level_err_db']} dB, "
              f"decay60 {audio['decay60_cand_s']}s vs {audio['decay60_truth_s']}s")
    print(f"Report: {out}")


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------

def cmd_ledger(args):
    exam_dir = Path(args.exam).resolve()
    with open(exam_dir / "answer" / "grade_report.json", encoding="utf-8") as f:
        report = json.load(f)
    row = {
        "round": exam_dir.name,
        "voice": report["voice"],
        "structure": report["algorithm"]["structure"],
        "truth_alg": report["algorithm"]["truth"],
        "cand_alg": report["algorithm"]["cand"],
        "feedback_hit": report["feedback"]["truth"] == report["feedback"]["cand"],
        "lessons": args.lessons,
    }
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(f"Appended round '{row['round']}' to {LEDGER_PATH}")
    print("Now write the prose lesson into playbook ch. 25 Lessons ledger.")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prep", help="build an exam from a source preset")
    sp.add_argument("preset")
    sp.add_argument("--round-name", required=True)
    sp.add_argument("--exam-dir", default=str(REPO_ROOT / "audio_out" / "gym"))
    sp.add_argument("--notes", default="48,60,72")
    sp.add_argument("--duration", type=float, default=2.5)
    sp.add_argument("--extra-deny", nargs="*")
    sp.set_defaults(fn=cmd_prep)

    sg = sub.add_parser("grade", help="grade a candidate YAML against the key")
    sg.add_argument("candidate")
    sg.add_argument("--exam", required=True)
    sg.add_argument("--note", type=int, default=60)
    sg.add_argument("--no-audio", action="store_true")
    sg.set_defaults(fn=cmd_grade)

    sl = sub.add_parser("ledger", help="append a graded round to the ledger")
    sl.add_argument("--exam", required=True)
    sl.add_argument("--lessons", required=True)
    sl.set_defaults(fn=cmd_ledger)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
