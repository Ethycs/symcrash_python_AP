"""G0: legal symmetry transforms and canonical forms (spec §5.6, §15)."""
import itertools
from pathlib import Path

import numpy as np
import pytest
import yaml

from symcrash_inversion import canonical
from symcrash_inversion.probes import PROBE_SET_SCHEMA, ProbeSet, ProbeSpec, standard_suite
from symcrash_python_ap.vm.fm_algorithms import ALGORITHMS

PRESETS = Path(__file__).resolve().parents[2] / "presets" / "instruments"


def load(name: str) -> dict:
    return yaml.safe_load((PRESETS / f"{name}.yaml").read_text(encoding="utf-8"))


def quick_probes(gate=0.2, tail=0.05) -> ProbeSet:
    s = standard_suite("t")
    return ProbeSet(PROBE_SET_SCHEMA, "t", 48000,
                    [ProbeSpec(p.id, p.audio, p.note, p.velocity, gate, tail, p.seed) for p in s.probes[:3]])


def brute_force(alg_id):
    alg = ALGORITHMS[alg_id]
    M = np.asarray(alg.modulation_matrix, dtype=bool)
    carriers = set(alg.get_carriers())
    out = set()
    for p in itertools.permutations(range(6)):
        if {p[c] for c in carriers} != carriers:
            continue
        if alg.feedback_op >= 0 and p[alg.feedback_op] != alg.feedback_op:
            continue
        if all(M[p[i], p[j]] == M[i, j] for i in range(6) for j in range(6)):
            out.add(p)
    return out


@pytest.mark.parametrize("alg_id", sorted(ALGORITHMS))
def test_graph_automorphisms_match_brute_force(alg_id):
    g = canonical.graph_automorphisms(alg_id)
    assert canonical.IDENTITY in g
    assert set(g) == brute_force(alg_id)
    assert list(g) == sorted(g)


def test_group_closure_and_inverses():
    for alg_id in (5, 12, 22, 24, 32, 36):
        g = set(canonical.graph_automorphisms(alg_id))
        for p in g:
            assert canonical.inverse(p) in g
            for q in list(g)[:10]:
                assert canonical.compose(p, q) in g


def test_apply_permutation_moves_operators_and_envelope_targets():
    d = load("dx7_e_piano_1_v0")
    group = canonical.graph_automorphisms(canonical.algorithm_id(d))
    p = next(x for x in group if x != canonical.IDENTITY)
    e = canonical.apply_permutation(d, p)
    ops_d = d["instrument_ir_v0"]["graph"]["fm6"]["operators"]
    ops_e = e["instrument_ir_v0"]["graph"]["fm6"]["operators"]
    for i in range(6):
        assert ops_e[p[i]] == ops_d[i]
    for name, env in d["instrument_ir_v0"]["envelopes"].items():
        for t_old, t_new in zip(env["targets"], e["instrument_ir_v0"]["envelopes"][name]["targets"]):
            if t_old.startswith("FM_OP"):
                assert t_new == f"FM_OP{p[int(t_old[5:]) - 1] + 1}"
            else:
                assert t_new == t_old
    # round trip through the inverse restores the preset exactly
    assert canonical.apply_permutation(e, canonical.inverse(p)) == d


@pytest.mark.parametrize("name", ["dx7_e_piano_1_v0", "dx7_harpsich_1_v0", "dx7_brass_1_v0", "dx7_vibe_1_v0", "pad_cheap_v0"])
def test_canonical_form_is_orbit_invariant_and_minimal(name):
    d = load(name)
    group = canonical.graph_automorphisms(canonical.algorithm_id(d))
    sigs = [canonical.field_signature(canonical.apply_permutation(d, p)) for p in group]
    cf = canonical.canonical_form(d)
    assert cf == min(sigs)
    for p in group:
        assert canonical.canonical_form(canonical.apply_permutation(d, p)) == cf
    canon, perm = canonical.canonicalize(d)
    assert canonical.field_signature(canon) == cf
    assert canonical.apply_permutation(d, perm) == canon


def test_field_signature_uses_engine_quantization():
    d = load("dx7_e_piano_1_v0")
    sig = canonical.field_signature(d)
    node = d["instrument_ir_v0"]["graph"]["fm6"]
    assert sig[0] == node["algorithm"] and sig[1] == node.get("feedback", 0)
    for slot, (ratio_q16, level_q15, envs) in enumerate(sig[2], start=1):
        op = node["operators"][slot - 1]
        assert ratio_q16 == round(op.get("ratio", 1.0) * 65536)
        assert level_q15 == round(op.get("level", 1.0) * 32767)
        assert list(envs) == sorted(n for n, e in d["instrument_ir_v0"]["envelopes"].items()
                                    if f"FM_OP{slot}" in e.get("targets", []))


def test_render_verification_keeps_legal_and_rejects_illegal():
    d = load("dx7_e_piano_1_v0")          # algorithm 5: three carrier/modulator pairs
    probes = quick_probes()
    legal = canonical.symmetry_group(5, verify=True, probes=probes, preset=d)
    assert set(legal) == set(canonical.graph_automorphisms(5))
    illegal = (1, 0, 2, 3, 4, 5)           # swaps a carrier with its modulator: not an automorphism
    assert illegal not in legal
    assert canonical.verify_permutation(d, illegal, probes) is False
    for p in legal:
        assert canonical.verify_permutation(d, p, probes) is True
