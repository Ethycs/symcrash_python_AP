#!/usr/bin/env python
"""Deterministic block-coordinate preset refinement against a target WAV.

This is the automated inner loop from the authoring plan.  It deliberately
keeps topology fixed: start from an imported/translated near patch, declare
small parameter grids, render every point through the canonical VM, and keep
the best point under the perceptual committee.  Zimtohrli is process-isolated
and used only for the final candidate by default; the fast diagnostic members
order the inner sweep.

Examples::

    pixi run auto-refine target.wav presets/instruments/dx7_marimba_v0.yaml \
        --output audio_out/refine/marimba_v0.yaml \
        --param /fm/mod_index=1.0:3.0:9 \
        --param /fm/op2/level=0.2:0.7:6

The source preset is never edited.  The output is a comment-preserving copy
with one version bump, plus a JSON report and audition WAV.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "src", REPO_ROOT / "tools" / "gym"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from symcrash_workbench.session import WorkbenchSession  # noqa: E402
from timbre_distance import (  # noqa: E402
    BootstrapWeights,
    SAMPLE_RATE,
    read_audio,
    score_arrays,
)


@dataclass(frozen=True)
class GridSpec:
    address: str
    low: float
    high: float
    steps: int
    logarithmic: bool = False

    def values(self) -> list[float]:
        if self.steps < 2:
            raise ValueError(f"{self.address}: grid needs at least two steps")
        if self.low > self.high:
            raise ValueError(f"{self.address}: low bound exceeds high bound")
        if self.logarithmic:
            if self.low <= 0:
                raise ValueError(f"{self.address}: log grid requires low > 0")
            values = np.geomspace(self.low, self.high, self.steps)
        else:
            values = np.linspace(self.low, self.high, self.steps)
        return [float(value) for value in values]


def parse_grid(text: str) -> GridSpec:
    """Parse ADDRESS=LOW:HIGH:STEPS[:log]."""
    try:
        address, encoded = text.split("=", 1)
        parts = encoded.split(":")
        if len(parts) not in (3, 4):
            raise ValueError
        low, high = float(parts[0]), float(parts[1])
        steps = int(parts[2])
        logarithmic = len(parts) == 4 and parts[3].lower() == "log"
        if len(parts) == 4 and not logarithmic:
            raise ValueError
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid grid {text!r}; expected ADDRESS=LOW:HIGH:STEPS[:log]"
        ) from exc
    if not address.startswith("/"):
        raise argparse.ArgumentTypeError(f"parameter address must start with '/': {address}")
    spec = GridSpec(address, low, high, steps, logarithmic)
    try:
        spec.values()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return spec


def _mono_candidate(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.ndim == 2 and samples.shape[0] in (1, 2):
        return samples.mean(axis=0)
    return samples


def _fast_score(reference: np.ndarray, reference_rate: int, candidate: np.ndarray) -> dict:
    # The committee is loudness-normalized by contract.  Original level is
    # retained as a diagnostic but cannot steer cross-engine comparisons.
    weights = BootstrapWeights(level=0.0)
    return score_arrays(
        reference,
        _mono_candidate(candidate),
        reference_rate=reference_rate,
        candidate_rate=SAMPLE_RATE,
        zimtohrli_backend="off",
        weights=weights,
    )


def proposal_improves(baseline: dict[str, Any], proposal: dict[str, Any]) -> bool:
    """A proposal may ship only when the full committee does not regress."""
    return float(proposal["composite"]) <= float(baseline["composite"]) + 1e-12


def _controlled_values(session: WorkbenchSession, specs: list[GridSpec]) -> dict[str, float]:
    values = {}
    for spec in specs:
        param = session.registry.get(spec.address)
        if param is None:
            raise ValueError(f"unknown workbench parameter: {spec.address}")
        value = param.get(session.ir)
        if value is None:
            raise ValueError(f"preset does not expose {spec.address}")
        values[spec.address] = float(value)
    return values


def refine(
    target: Path,
    seed: Path,
    output: Path,
    specs: list[GridSpec],
    *,
    note: int = 60,
    velocity: float = 0.85,
    duration: float | None = None,
    passes: int = 1,
    zimtohrli_backend: str = "auto",
) -> dict[str, Any]:
    if not specs:
        raise ValueError("at least one --param grid is required")
    target_audio, target_rate = read_audio(target)
    inferred_duration = max(0.1, len(target_audio) / target_rate)
    render_duration = float(duration if duration is not None else inferred_duration)

    run_dir = output.parent / f"{output.stem}_artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    working = run_dir / "working.yaml"
    shutil.copyfile(seed, working)

    session = WorkbenchSession(
        working,
        autoplay=False,
        log_dir=run_dir / "trajectory",
    )
    session.note = note
    session.velocity = max(0.0, min(1.0, velocity))
    session.duration = render_duration
    evaluations: list[dict[str, Any]] = []

    def evaluate(label: str) -> tuple[float, np.ndarray, dict[str, Any]]:
        rendered = session.render()
        score = _fast_score(target_audio, target_rate, rendered)
        values = _controlled_values(session, specs)
        row = {
            "label": label,
            "loss": float(score["composite"]),
            "parameters": values,
            "metrics": score["metrics"],
        }
        evaluations.append(row)
        print(f"  {len(evaluations):03d}  {row['loss']:.6f}  {label}  {values}")
        return row["loss"], rendered, score

    try:
        initial_values = _controlled_values(session, specs)
        baseline_loss, baseline_audio, _ = evaluate("baseline")
        current_loss = baseline_loss

        for pass_index in range(max(1, passes)):
            print(f"PASS {pass_index + 1}/{max(1, passes)}")
            for spec in specs:
                incumbent = _controlled_values(session, specs)[spec.address]
                best_value = incumbent
                best_loss = math.inf
                for value in spec.values():
                    if not session.handle(spec.address, value):
                        raise RuntimeError(f"workbench rejected {spec.address}={value}")
                    loss, _, _ = evaluate(f"pass{pass_index + 1}:{spec.address}={value:.8g}")
                    if loss < best_loss:
                        best_loss, best_value = loss, value
                session.handle(spec.address, best_value)
                current_loss = best_loss
                print(f"BEST {spec.address}={best_value:.8g}  loss={best_loss:.6f}")

        proposed_fast_loss, proposed_audio, proposed_fast = evaluate("proposal")
        proposed_parameters = _controlled_values(session, specs)
        baseline_score = score_arrays(
            target_audio,
            _mono_candidate(baseline_audio),
            reference_rate=target_rate,
            candidate_rate=SAMPLE_RATE,
            zimtohrli_backend=zimtohrli_backend,
            weights=BootstrapWeights(level=0.0),
        )
        proposed_score = score_arrays(
            target_audio,
            _mono_candidate(proposed_audio),
            reference_rate=target_rate,
            candidate_rate=SAMPLE_RATE,
            zimtohrli_backend=zimtohrli_backend,
            weights=BootstrapWeights(level=0.0),
        )

        accepted = proposal_improves(baseline_score, proposed_score)
        if accepted:
            final_audio = proposed_audio
            final_fast_loss = proposed_fast_loss
            final_score = proposed_score
        else:
            print("PROPOSAL REJECTED: full committee is worse than baseline; restoring seed values")
            for address, value in initial_values.items():
                if not session.handle(address, value):
                    raise RuntimeError(f"workbench could not restore {address}={value}")
            final_audio = baseline_audio
            final_fast_loss = baseline_loss
            final_score = baseline_score

        # Workbench save is the canonical comment-preserving write-back path;
        # it applies final values and bumps meta.version on the working copy.
        if not session.save():
            raise RuntimeError("optimizer produced no write-back changes")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(working, output)
        audition = output.with_suffix(".wav")
        audio_to_write = np.asarray(final_audio)
        if audio_to_write.ndim == 2 and audio_to_write.shape[0] in (1, 2):
            audio_to_write = audio_to_write.T
        wavfile.write(audition, SAMPLE_RATE, audio_to_write.astype(np.int16))

        ranked = sorted(evaluations, key=lambda row: row["loss"])
        report = {
            "schema": "symcrash.auto-refine.v0",
            "target": str(target.resolve()),
            "seed": str(seed.resolve()),
            "output": str(output.resolve()),
            "audition": str(audition.resolve()),
            "note": note,
            "velocity": session.velocity,
            "duration": render_duration,
            "passes": max(1, passes),
            "grids": [asdict(spec) for spec in specs],
            "initial_parameters": initial_values,
            "final_parameters": _controlled_values(session, specs),
            "baseline_fast_loss": baseline_loss,
            "final_fast_loss": final_fast_loss,
            "improvement": baseline_loss - final_fast_loss,
            "baseline_committee": baseline_score,
            "final_committee": final_score,
            "committee_improvement": baseline_score["composite"] - final_score["composite"],
            "proposal": {
                "accepted": accepted,
                "parameters": proposed_parameters,
                "fast_loss": proposed_fast_loss,
                "committee": proposed_score,
                "rejection_reason": (
                    None if accepted else "full committee score exceeded baseline"
                ),
            },
            "shortlist": ranked[: min(5, len(ranked))],
            "evaluations": evaluations,
            "policy": {
                "topology_fixed": True,
                "inner_loop": "diagnostic committee, Zimtohrli off",
                "final_check": f"Zimtohrli backend {zimtohrli_backend}",
                "human_audition_required": True,
            },
        }
        report_path = output.with_suffix(".report.json")
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"OUTPUT {output}")
        print(f"AUDITION {audition}")
        print(f"REPORT {report_path}")
        print(f"IMPROVEMENT {report['improvement']:+.6f}")
        print(f"COMMITTEE IMPROVEMENT {report['committee_improvement']:+.6f}")
        return report
    finally:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    parser.add_argument("seed", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--param", type=parse_grid, action="append", required=True)
    parser.add_argument("--note", type=int, default=60)
    parser.add_argument("--velocity", type=float, default=0.85)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument(
        "--zimtohrli",
        choices=("auto", "wsl", "local", "off"),
        default="auto",
    )
    args = parser.parse_args()
    refine(
        args.target.resolve(),
        args.seed.resolve(),
        args.output.resolve(),
        args.param,
        note=args.note,
        velocity=args.velocity,
        duration=args.duration,
        passes=args.passes,
        zimtohrli_backend=args.zimtohrli,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
