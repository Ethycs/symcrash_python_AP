#!/usr/bin/env python
"""Batch-refine the eight core DX7 imports against Pixi-hosted Dexed.

Each preset declares its ROM bank and 1-based voice in the generated header.
This driver renders that exact voice through the independent Dexed engine in
WSL, then invokes the generic block-coordinate refiner.  References and
candidates stay under ``audio_out``; source presets are not edited.

Run::

    pixi run dx7-core-refine --zimtohrli wsl
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from auto_refine import GridSpec, refine  # noqa: E402
from timbre_distance import _windows_to_wsl, _wsl_home  # noqa: E402


CORE_PRESETS = (
    "dx7_brass_1_v0.yaml",
    "dx7_strings_1_v0.yaml",
    "dx7_e_piano_1_v0.yaml",
    "dx7_bass_1_v0.yaml",
    "dx7_harpsich_1_v0.yaml",
    "dx7_vibe_1_v0.yaml",
    "dx7_marimba_v0.yaml",
    "dx7_tub_bells_v0.yaml",
)

SOURCE_RE = re.compile(r"^# Source: ([^,]+), voice (\d+)/32\s*$", re.MULTILINE)


def preset_source(path: Path) -> tuple[Path, int]:
    match = SOURCE_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"missing DX7 source header in {path}")
    bank = REPO_ROOT / "ingest" / "dx7_banks" / match.group(1)
    if not bank.exists():
        raise FileNotFoundError(bank)
    return bank, int(match.group(2))


def render_dexed_reference(
    preset: Path,
    output: Path,
    *,
    note: int,
    gate_seconds: float,
    tail_seconds: float,
    velocity: int,
    distro: str,
    force: bool = False,
) -> dict[str, Any]:
    bank, voice = preset_source(preset)
    if output.exists() and not force:
        return {"bank": str(bank), "voice": voice, "path": str(output), "cached": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    home = _wsl_home(distro, 30.0)
    command = [
        "wsl.exe", "-d", distro, "--",
        f"{home}/.local/bin/pixi", "run",
        "--manifest-path", _windows_to_wsl(REPO_ROOT / "pyproject.toml"),
        "--environment", "dexed", "--frozen",
        "python", _windows_to_wsl(Path(__file__).with_name("dexed_reference.py")),
        _windows_to_wsl(bank),
        "--voice", str(voice),
        "--note", str(note),
        "--dur", str(gate_seconds),
        "--tail", str(tail_seconds),
        "--velocity", str(velocity),
        "--out", _windows_to_wsl(output),
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=90.0,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "Dexed render failed").strip())
    return {
        "bank": str(bank),
        "voice": voice,
        "path": str(output),
        "cached": False,
        "renderer_output": result.stdout.strip(),
    }


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    preset_dir = REPO_ROOT / "presets" / "instruments"
    output_dir = args.output_dir.resolve()
    reference_dir = output_dir / "references"
    candidates_dir = output_dir / "candidates"
    rows = []
    selected = args.presets or list(CORE_PRESETS)
    spec = GridSpec("/fm/mod_index", args.index_low, args.index_high, args.index_steps)

    for number, name in enumerate(selected, start=1):
        preset = (preset_dir / name).resolve()
        if not preset.exists():
            raise FileNotFoundError(preset)
        reference = reference_dir / f"{preset.stem}_note{args.note}.wav"
        candidate = candidates_dir / preset.name
        print(f"\n[{number}/{len(selected)}] {preset.name}")
        reference_info = render_dexed_reference(
            preset,
            reference,
            note=args.note,
            gate_seconds=args.gate,
            tail_seconds=args.tail,
            velocity=args.velocity,
            distro=args.wsl_distro,
            force=args.force_references,
        )
        report = refine(
            reference,
            preset,
            candidate,
            [spec],
            note=args.note,
            velocity=args.velocity / 127.0,
            duration=args.gate + args.tail,
            passes=1,
            zimtohrli_backend=args.zimtohrli,
        )
        rows.append({
            "preset": str(preset),
            "reference": reference_info,
            "candidate": report["output"],
            "initial_mod_index": report["initial_parameters"]["/fm/mod_index"],
            "best_mod_index": report["final_parameters"]["/fm/mod_index"],
            "baseline_fast_loss": report["baseline_fast_loss"],
            "final_fast_loss": report["final_fast_loss"],
            "improvement": report["improvement"],
            "proposal_accepted": report["proposal"]["accepted"],
            "proposed_mod_index": report["proposal"]["parameters"]["/fm/mod_index"],
            "baseline_committee": report["baseline_committee"]["composite"],
            "final_committee": report["final_committee"]["composite"],
            "committee_improvement": report["committee_improvement"],
            "zimtohrli": report["final_committee"]["zimtohrli"],
        })

    summary = {
        "schema": "symcrash.dx7-core-refine.v0",
        "policy": {
            "source_presets_edited": False,
            "independent_control": "Dexed 1.0.1 via WSL Pixi",
            "human_audition_required": True,
        },
        "grid": {
            "address": spec.address,
            "values": spec.values(),
        },
        "presets": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nSUMMARY {summary_path}")
    for row in rows:
        print(
            f"  {Path(row['preset']).name}: {row['initial_mod_index']:.3g} -> "
            f"{row['best_mod_index']:.3g}, fast {row['improvement']:+.6f}, "
            f"committee {row['committee_improvement']:+.6f}"
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", nargs="*", help="core preset filenames (default: all eight)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "audio_out" / "dx7_core_refine",
    )
    parser.add_argument("--note", type=int, default=60)
    parser.add_argument("--gate", type=float, default=1.5)
    parser.add_argument("--tail", type=float, default=1.0)
    parser.add_argument("--velocity", type=int, default=100)
    parser.add_argument("--index-low", type=float, default=1.0)
    parser.add_argument("--index-high", type=float, default=3.0)
    parser.add_argument("--index-steps", type=int, default=5)
    parser.add_argument("--zimtohrli", choices=("auto", "wsl", "local", "off"), default="wsl")
    parser.add_argument("--wsl-distro", default="Ubuntu-24.04")
    parser.add_argument("--force-references", action="store_true")
    args = parser.parse_args()
    run_batch(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
