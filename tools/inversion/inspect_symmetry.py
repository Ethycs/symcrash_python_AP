#!/usr/bin/env python3
"""Report operator symmetry groups per FM algorithm (spec §5.6).

Usage:
    python tools/inversion/inspect_symmetry.py                 # graph automorphisms, algorithms 1-38
    python tools/inversion/inspect_symmetry.py --algs 5,22,32  # subset
    python tools/inversion/inspect_symmetry.py --verify --gate 0.4 --tail 0.1   # render-verified
    python tools/inversion/inspect_symmetry.py --preset presets/instruments/dx7_e_piano_1_v0.yaml --canonical
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from symcrash_inversion import canonical  # noqa: E402
from symcrash_inversion.probes import ProbeSet, ProbeSpec, PROBE_SET_SCHEMA, standard_suite  # noqa: E402
from symcrash_python_ap.vm.fm_algorithms import ALGORITHMS  # noqa: E402


def quick_probes(gate: float, tail: float) -> ProbeSet:
    suite = standard_suite("inspect")
    return ProbeSet(PROBE_SET_SCHEMA, "inspect", 48000,
                    [ProbeSpec(p.id, p.audio, p.note, p.velocity, gate, tail, p.seed) for p in suite.probes])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--algs", default=None, help="comma list, default: all known algorithms")
    ap.add_argument("--verify", action="store_true", help="render-verify each automorphism")
    ap.add_argument("--gate", type=float, default=1.5)
    ap.add_argument("--tail", type=float, default=1.0)
    ap.add_argument("--preset", default=None, help="YAML preset: report its orbit and canonical form")
    ap.add_argument("--canonical", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.preset:
        import yaml
        d = yaml.safe_load(open(a.preset, encoding="utf-8"))
        alg = canonical.algorithm_id(d)
        group = canonical.symmetry_group(alg, verify=a.verify, probes=quick_probes(a.gate, a.tail), preset=d)
        canon, perm = canonical.canonicalize(d, group=group)
        rep = {"preset": a.preset, "algorithm": alg, "group_size": len(group),
               "signature": canonical.field_signature(d), "canonical_signature": canonical.field_signature(canon),
               "canonical_permutation": list(perm)}
        print(json.dumps(rep, indent=1, default=str) if a.json else
              f"{a.preset}: algorithm {alg}, group size {len(group)}, canonical permutation {perm}")
        return 0

    algs = [int(x) for x in a.algs.split(",")] if a.algs else sorted(ALGORITHMS)
    probes = quick_probes(a.gate, a.tail) if a.verify else None
    rows = []
    for alg in algs:
        graph = canonical.graph_automorphisms(alg)
        legal = canonical.symmetry_group(alg, verify=a.verify, probes=probes) if a.verify else graph
        rows.append({"algorithm": alg, "name": ALGORITHMS[alg].name, "graph_automorphisms": len(graph) - 1,
                     "legal": len(legal) - 1, "rejected": len(graph) - len(legal),
                     "permutations": [list(p) for p in legal if p != canonical.IDENTITY]})
    if a.json:
        print(json.dumps(rows, indent=1))
    else:
        print(f"{'alg':>4} {'name':<28} {'graph':>5} {'legal':>5} {'rejected':>8}")
        for r in rows:
            print(f"{r['algorithm']:>4} {r['name']:<28} {r['graph_automorphisms']:>5} {r['legal']:>5} {r['rejected']:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
