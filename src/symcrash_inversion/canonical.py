"""Operator symmetry groups, canonical representatives and orbit tools (spec §5.6).

Conventions
-----------
* A permutation is a tuple p of length 6 with p[i] = destination slot of operator i
  (0-based). The identity is (0, 1, 2, 3, 4, 5).
* graph_automorphisms(alg_id) are the permutations that preserve the carrier set, every
  directed modulation edge and the feedback operator of ALGORITHMS[alg_id]. Identity included.
* symmetry_group(alg_id, verify=True, ...) additionally requires every permutation to
  render bit-exactly across a probe set; a non-zero render difference removes it.
* field_signature(preset) = (algorithm, feedback, per-slot tuple) where each slot is
  (round(ratio * 65536), round(level * 32767), sorted names of envelopes targeting FM_OP<slot>).
* canonical_form(preset) is the lexicographically smallest field_signature over the orbit.
"""
from __future__ import annotations

import copy
import itertools
import re
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from symcrash_python_ap.vm.fm_algorithms import ALGORITHMS

Permutation = Tuple[int, int, int, int, int, int]
IDENTITY: Permutation = (0, 1, 2, 3, 4, 5)
Q16_ONE = 65536
Q15_ONE = 32767
_FM_OP_RE = re.compile(r"^FM_OP([1-6])$")


# ---------------------------------------------------------------------------
# preset access
# ---------------------------------------------------------------------------

def _ir(d: dict) -> dict:
    return d.get("instrument_ir_v0", d)


def fm6_node(preset: dict) -> Tuple[str, dict]:
    """(node name, node dict) of the FM6 engine node; ValueError if absent or malformed."""
    graph = _ir(preset).get("graph") or {}
    for name, node in graph.items():
        if (node or {}).get("type") == "FM6":
            ops = node.get("operators")
            if not isinstance(ops, list) or len(ops) != 6:
                raise ValueError(f"FM6 node {name!r} must declare exactly six operators")
            return name, node
    raise ValueError("preset has no FM6 engine node")


def algorithm_id(preset: dict) -> int:
    return int(fm6_node(preset)[1].get("algorithm", 1))


# ---------------------------------------------------------------------------
# groups
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def graph_automorphisms(alg_id: int) -> Tuple[Permutation, ...]:
    """Permutations preserving carrier set, modulation edges and feedback op (identity first)."""
    alg = ALGORITHMS[alg_id]
    M = np.asarray(alg.modulation_matrix, dtype=bool)
    carriers = set(alg.get_carriers())
    out: List[Permutation] = []
    for p in itertools.permutations(range(6)):
        if {p[c] for c in carriers} != carriers:
            continue
        if alg.feedback_op >= 0 and p[alg.feedback_op] != alg.feedback_op:
            continue
        if all(M[p[i], p[j]] == M[i, j] for i in range(6) for j in range(6)):
            out.append(tuple(p))
    out.sort()
    return tuple(out)


def apply_permutation(preset: dict, perm: Sequence[int]) -> dict:
    """Return a deep copy of the preset with operator i moved to slot perm[i] and every
    envelope target FM_OP<i+1> renamed to FM_OP<perm[i]+1>. Nothing else changes."""
    perm = tuple(int(x) for x in perm)
    if sorted(perm) != list(range(6)):
        raise ValueError(f"not a permutation of six operators: {perm}")
    out = copy.deepcopy(preset)
    _, node = fm6_node(out)
    ops = node["operators"]
    new_ops: List[Optional[dict]] = [None] * 6
    for i in range(6):
        new_ops[perm[i]] = ops[i]
    node["operators"] = new_ops
    for env in (_ir(out).get("envelopes") or {}).values():
        targets = (env or {}).get("targets")
        if isinstance(targets, list):
            env["targets"] = [_remap_target(t, perm) for t in targets]
    return out


def _remap_target(target, perm: Permutation):
    m = _FM_OP_RE.match(str(target))
    if not m:
        return target
    return f"FM_OP{perm[int(m.group(1)) - 1] + 1}"


def compose(p: Sequence[int], q: Sequence[int]) -> Permutation:
    """(q ∘ p): apply p then q."""
    return tuple(q[p[i]] for i in range(6))  # type: ignore[return-value]


def inverse(p: Sequence[int]) -> Permutation:
    inv = [0] * 6
    for i, d in enumerate(p):
        inv[d] = i
    return tuple(inv)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# signatures and canonical forms
# ---------------------------------------------------------------------------

def field_signature(preset: dict) -> tuple:
    ir = _ir(preset)
    _, node = fm6_node(preset)
    env_names: Dict[int, List[str]] = {s: [] for s in range(1, 7)}
    for name, env in sorted((ir.get("envelopes") or {}).items()):
        for t in (env or {}).get("targets") or []:
            m = _FM_OP_RE.match(str(t))
            if m:
                env_names[int(m.group(1))].append(str(name))
    slots = []
    for s, op in enumerate(node["operators"], start=1):
        op = op or {}
        slots.append((int(round(float(op.get("ratio", 1.0)) * Q16_ONE)),
                      int(round(float(op.get("level", 1.0)) * Q15_ONE)),
                      tuple(sorted(env_names[s]))))
    return (int(node.get("algorithm", 1)), int(node.get("feedback", 0)), tuple(slots))


def orbit(preset: dict, group: Optional[Sequence[Permutation]] = None) -> List[Tuple[Permutation, dict]]:
    group = tuple(group) if group is not None else graph_automorphisms(algorithm_id(preset))
    return [(p, apply_permutation(preset, p)) for p in group]


def canonicalize(preset: dict, group: Optional[Sequence[Permutation]] = None,
                 verify: bool = False, probes=None) -> Tuple[dict, Permutation]:
    """(canonical preset, permutation that produced it): the orbit member whose
    field_signature is lexicographically smallest. Ties resolve to the smallest permutation."""
    if group is None:
        group = symmetry_group(algorithm_id(preset), verify=verify, probes=probes, preset=preset)
    best: Optional[Tuple[tuple, Permutation, dict]] = None
    for p, variant in orbit(preset, group):
        key = (field_signature(variant), p)
        if best is None or key < (best[0], best[1]):
            best = (key[0], p, variant)
    assert best is not None
    return best[2], best[1]


def canonical_form(preset: dict, group: Optional[Sequence[Permutation]] = None,
                   verify: bool = False, probes=None) -> tuple:
    canon, _ = canonicalize(preset, group=group, verify=verify, probes=probes)
    return field_signature(canon)


# ---------------------------------------------------------------------------
# render verification
# ---------------------------------------------------------------------------

def _default_probe_set(instrument_id: str = "symmetry-check"):
    from .probes import standard_suite
    return standard_suite(instrument_id)


def verify_permutation(preset: dict, perm: Sequence[int], probes=None) -> bool:
    """True iff the permuted preset renders bit-exactly equal to the original on every probe."""
    from .render_service import RenderRequest, render
    probes = probes if probes is not None else _default_probe_set()
    permuted = apply_permutation(preset, perm)
    for spec in probes.probes:
        a = render(RenderRequest(preset, spec.note, spec.velocity, spec.gate_s, spec.tail_s, spec.seed)).audio
        b = render(RenderRequest(permuted, spec.note, spec.velocity, spec.gate_s, spec.tail_s, spec.seed)).audio
        if a.shape != b.shape or not np.array_equal(a, b):
            return False
    return True


def _reference_preset(alg_id: int) -> dict:
    """A synthetic FM6 preset with distinguishable operators, used when no preset is given."""
    alg = ALGORITHMS[alg_id]
    ratios = [1.0, 2.0, 3.01, 0.5, 1.41, 7.0]
    levels = [0.62, 0.41, 0.27, 0.19, 0.13, 0.09]
    return {"instrument_ir_v0": {
        "meta": {"id": f"symmetry-{alg_id}", "name": f"symmetry {alg_id}", "version": 1,
                 "author": "symcrash_inversion", "seed": 7},
        "audio": {"sample_rate": 48000, "block_size": 64},
        "graph": {"fm6": {"type": "FM6", "output": "amp1", "algorithm": alg_id,
                          "feedback": 3 if alg.feedback_op >= 0 else 0,
                          "operators": [{"ratio": ratios[i], "level": levels[i]} for i in range(6)]},
                  "amp1": {"type": "Amp", "attack": 2.0, "decay": 300.0, "sustain": 0.4, "release": 60.0}},
        "controls": {},
        "envelopes": {"env_a": {"attack": 0.5, "decay": 90.0, "sustain": 0.0, "release": 60.0, "targets": ["FM_OP2"]},
                      "env_b": {"attack": 1.0, "decay": 160.0, "sustain": 0.1, "release": 60.0, "targets": ["FM_OP4", "FM_OP6"]}},
        "behavior": {k: {"emphasis": {}, "trees": {"A": "harmonic", "B": "harmonic", "blend": 0.0}}
                     for k in ("idle", "attack", "stable", "recovery")},
        "compile": {"material": {"blend": 0.0, "target": 0.0, "ramp": 1}},
    }}


_VERIFIED: Dict[Tuple[int, str], Tuple[Permutation, ...]] = {}


def symmetry_group(alg_id: int, verify: bool = False, probes=None, preset: Optional[dict] = None
                   ) -> Tuple[Permutation, ...]:
    """The legal symmetry group of an algorithm (identity included, sorted).

    verify=False returns the graph automorphisms. verify=True keeps only those that render
    bit-exactly (verify_permutation) on `probes` (default: the standard nine-probe suite)
    for `preset` (default: a synthetic preset with distinguishable operators); results are
    cached per (algorithm, probe-set identity)."""
    group = graph_automorphisms(alg_id)
    if not verify:
        return group
    probes = probes if probes is not None else _default_probe_set()
    key = (alg_id, repr(sorted((p.note, p.velocity, p.gate_s, p.tail_s, p.seed) for p in probes.probes))
           + ("" if preset is None else repr(field_signature(preset))))
    if key not in _VERIFIED:
        base = preset if preset is not None else _reference_preset(alg_id)
        legal = tuple(p for p in group if p == IDENTITY or verify_permutation(base, p, probes))
        _VERIFIED[key] = legal
    return _VERIFIED[key]
