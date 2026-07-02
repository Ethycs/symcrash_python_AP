"""
DX7 algorithm-table conformance (pins the 2026-06-25 fix).

Verifies ALGORITHMS 1-32 against the canonical DX7 chart, decoded from an
independent embedding of the Dexed/msfa reference table
(music-synthesizer-for-android fm_core.cc, FmCore::algorithms[32]) — the
de-facto bit-accurate DX7 ground truth. Before the fix the engine table
diverged structurally on 28 of 32 algorithms; this test makes any future
drift a failure, not a discovery.

Known, documented approximation: real DX7 algorithms 4 and 6 use LOOP
feedback (op4->op6, op5->op6); the engine models self-feedback only and
approximates both as self-feedback on op6. The expectations below encode
that approximation explicitly.

Legacy WSE topologies (33-38) are pinned to the pre-fix structures that
ear-tuned presets were re-numbered onto (bit-exact sound preservation).
"""

import pytest

from symcrash_python_ap.vm.fm_algorithms import ALGORITHMS, get_algorithm

# msfa FmCore::algorithms[32] — rows are ops[0..5] = op6..op1.
# Flags: OUT_BUS=bits0-1 (0=audio out), OUT_ADD=0x04, IN_BUS=bits4-5,
# FB_IN=0x40, FB_OUT=0x80.
MSFA = {
    1:  [0xc1, 0x11, 0x11, 0x14, 0x01, 0x14],
    2:  [0x01, 0x11, 0x11, 0x14, 0xc1, 0x14],
    3:  [0xc1, 0x11, 0x14, 0x01, 0x11, 0x14],
    4:  [0xc1, 0x11, 0x94, 0x01, 0x11, 0x14],
    5:  [0xc1, 0x14, 0x01, 0x14, 0x01, 0x14],
    6:  [0xc1, 0x94, 0x01, 0x14, 0x01, 0x14],
    7:  [0xc1, 0x11, 0x05, 0x14, 0x01, 0x14],
    8:  [0x01, 0x11, 0xc5, 0x14, 0x01, 0x14],
    9:  [0x01, 0x11, 0x05, 0x14, 0xc1, 0x14],
    10: [0x01, 0x05, 0x14, 0xc1, 0x11, 0x14],
    11: [0xc1, 0x05, 0x14, 0x01, 0x11, 0x14],
    12: [0x01, 0x05, 0x05, 0x14, 0xc1, 0x14],
    13: [0xc1, 0x05, 0x05, 0x14, 0x01, 0x14],
    14: [0xc1, 0x05, 0x11, 0x14, 0x01, 0x14],
    15: [0x01, 0x05, 0x11, 0x14, 0xc1, 0x14],
    16: [0xc1, 0x11, 0x02, 0x25, 0x05, 0x14],
    17: [0x01, 0x11, 0x02, 0x25, 0xc5, 0x14],
    18: [0x01, 0x11, 0x11, 0xc5, 0x05, 0x14],
    19: [0xc1, 0x14, 0x14, 0x01, 0x11, 0x14],
    20: [0x01, 0x05, 0x14, 0xc1, 0x14, 0x14],
    21: [0x01, 0x14, 0x14, 0xc1, 0x14, 0x14],
    22: [0xc1, 0x14, 0x14, 0x14, 0x01, 0x14],
    23: [0xc1, 0x14, 0x14, 0x01, 0x14, 0x04],
    24: [0xc1, 0x14, 0x14, 0x14, 0x04, 0x04],
    25: [0xc1, 0x14, 0x14, 0x04, 0x04, 0x04],
    26: [0xc1, 0x05, 0x14, 0x01, 0x14, 0x04],
    27: [0x01, 0x05, 0x14, 0xc1, 0x14, 0x04],
    28: [0x04, 0xc1, 0x11, 0x14, 0x01, 0x14],
    29: [0xc1, 0x14, 0x01, 0x14, 0x04, 0x04],
    30: [0x04, 0xc1, 0x11, 0x14, 0x04, 0x04],
    31: [0xc1, 0x14, 0x04, 0x04, 0x04, 0x04],
    32: [0xc4, 0x04, 0x04, 0x04, 0x04, 0x04],
}

# Real DX7 loop-feedback algorithms: (fb reader, fb writer). Engine
# approximates both as self-feedback on the reader (op 6).
LOOP_FB = {4: (6, 4), 6: (6, 5)}

# Legacy WSE structures (1-based), pinned verbatim from the pre-fix table.
LEGACY = {
    33: ({1},          {(2, 1), (3, 2), (4, 3), (5, 4), (6, 5)}, 1),
    34: ({1, 4},       {(2, 1), (3, 2), (5, 4), (6, 5)}, 1),
    35: ({1},          {(2, 1), (3, 2), (4, 1), (5, 4), (6, 5)}, 6),
    36: ({1},          {(2, 1), (3, 1), (4, 1), (5, 1), (6, 1)}, 6),
    37: ({1},          {(2, 1), (3, 2), (4, 1), (5, 1), (6, 5)}, 6),
    38: ({1, 2, 3, 4}, {(5, 2), (5, 3), (5, 4), (6, 5)}, 6),
}


def decode_msfa(row):
    """Decode one msfa row -> (carriers, edges, fb_reader, fb_writer), 1-based."""
    ops = list(range(6, 0, -1))
    bus = {1: [], 2: []}
    carriers, edges = set(), set()
    fb_reader, fb_writers = None, []
    for i, flags in enumerate(row):
        op = ops[i]
        inbus = (flags >> 4) & 3
        outbus = flags & 3
        add = bool(flags & 0x04)
        if flags & 0x40:
            fb_reader = op
        if flags & 0x80:
            fb_writers.append(op)
        if inbus in (1, 2):
            for src in bus[inbus]:
                edges.add((src, op))
        if outbus == 0:
            carriers.add(op)
        else:
            if not add:
                bus[outbus] = []
            bus[outbus].append(op)
    return carriers, edges, fb_reader, (fb_writers[-1] if fb_writers else None)


def engine_shape(alg_id):
    a = ALGORITHMS[alg_id]
    carriers = {c + 1 for c in a.get_carriers()}
    edges = {(m + 1, op + 1) for op in range(6) for m in a.get_modulators(op)}
    fb = a.feedback_op + 1 if a.feedback_op >= 0 else None
    return carriers, edges, fb


class TestCanonicalDX7:
    """Engine algorithms 1-32 match the decoded msfa reference."""

    @pytest.mark.parametrize("alg_id", sorted(MSFA))
    def test_structure(self, alg_id):
        ref_c, ref_e, _, _ = decode_msfa(MSFA[alg_id])
        eng_c, eng_e, _ = engine_shape(alg_id)
        assert eng_c == ref_c, f"alg {alg_id} carriers {sorted(eng_c)} != DX7 {sorted(ref_c)}"
        assert eng_e == ref_e, f"alg {alg_id} edges {sorted(eng_e)} != DX7 {sorted(ref_e)}"

    @pytest.mark.parametrize("alg_id", sorted(MSFA))
    def test_feedback(self, alg_id):
        _, _, ref_reader, ref_writer = decode_msfa(MSFA[alg_id])
        _, _, eng_fb = engine_shape(alg_id)
        if alg_id in LOOP_FB:
            # documented approximation: self-fb on the loop reader
            assert (ref_reader, ref_writer) == LOOP_FB[alg_id]
            assert eng_fb == ref_reader, (
                f"alg {alg_id}: loop fb {ref_writer}->{ref_reader} must be "
                f"approximated as self-fb on op{ref_reader}, engine has {eng_fb}")
        else:
            assert ref_reader == ref_writer, f"alg {alg_id}: unexpected loop fb in reference"
            assert eng_fb == ref_reader, f"alg {alg_id} fb op{eng_fb} != DX7 op{ref_reader}"

    def test_eval_order_topological(self):
        """Every modulator is evaluated before each of its targets."""
        for alg_id in sorted(ALGORITHMS):
            a = ALGORITHMS[alg_id]
            pos = {op: i for i, op in enumerate(a.eval_order)}
            for op in range(6):
                for m in a.get_modulators(op):
                    assert pos[m] < pos[op], (
                        f"alg {alg_id}: op{m+1} modulates op{op+1} but is evaluated after it")


class TestLegacyWSE:
    """Legacy ids 33-38 pin the pre-fix structures ear-tuned presets rely on."""

    @pytest.mark.parametrize("alg_id", sorted(LEGACY))
    def test_structure_and_feedback(self, alg_id):
        ref_c, ref_e, ref_fb = LEGACY[alg_id]
        eng_c, eng_e, eng_fb = engine_shape(alg_id)
        assert eng_c == ref_c
        assert eng_e == ref_e
        assert eng_fb == ref_fb

    def test_get_algorithm_accepts_legacy(self):
        for alg_id in LEGACY:
            assert get_algorithm(alg_id).alg_id == alg_id

    def test_unknown_id_rejected(self):
        with pytest.raises(KeyError):
            get_algorithm(99)
