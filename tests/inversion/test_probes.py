"""Probe manifests (spec §6.1, §6.2)."""
import pytest
import yaml

from symcrash_inversion import probes as P

EXAMPLE = """
schema: symcrash.probe-set.v0
instrument_id: marimba_ref
sample_rate_hz: 48000
notes: recorded in room B
probes:
  - id: n48_v035
    audio: n48_v035.wav
    note: 48
    velocity: 0.35
    gate_s: 1.5
    tail_s: 1.0
    seed: 0
    mic: sm57
  - id: n60_v070
    audio: n60_v070.wav
    note: 60
    velocity: 0.70
    gate_s: 1.5
    tail_s: 1.0
    seed: 0
"""


def test_load_preserves_unknown_metadata(tmp_path):
    f = tmp_path / "m.yaml"
    f.write_text(EXAMPLE, encoding="utf-8")
    ps = P.load_probe_set(f)
    assert ps.schema == P.PROBE_SET_SCHEMA and ps.instrument_id == "marimba_ref"
    assert ps.identification_confidence == "high"
    assert ps.extra == {"notes": "recorded in room B"}
    assert ps.probes[0].extra == {"mic": "sm57"} and ps.probes[1].extra == {}
    assert ps.probes[1].note == 60 and ps.probes[1].velocity == 0.70
    assert ps.to_dict()["probes"][0]["mic"] == "sm57"


def test_missing_required_field_is_structured(tmp_path):
    data = yaml.safe_load(EXAMPLE)
    del data["probes"][1]["seed"]
    del data["probes"][1]["tail_s"]
    f = tmp_path / "m.yaml"
    f.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(P.ProbeManifestError) as ei:
        P.load_probe_set(f)
    assert set(ei.value.missing) == {"seed", "tail_s"} and ei.value.probe_index == 1


def test_wrong_schema_rejected(tmp_path):
    data = yaml.safe_load(EXAMPLE)
    data["schema"] = "something-else"
    f = tmp_path / "m.yaml"
    f.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(P.ProbeManifestError):
        P.load_probe_set(f)


def test_standard_suite_is_the_nine_probe_release_gate():
    s = P.standard_suite("x")
    assert len(s.probes) == 9
    assert sorted({p.note for p in s.probes}) == [48, 60, 72]
    assert sorted({p.velocity for p in s.probes}) == [0.35, 0.70, 1.00]
    assert all(p.gate_s == 1.5 and p.tail_s == 1.0 and p.seed == 0 for p in s.probes)
    assert len({p.id for p in s.probes}) == 9


def test_single_wav_wrap_is_low_confidence(tmp_path):
    ps = P.wrap_single_wav(tmp_path / "take1.wav", note=60, velocity=0.7)
    assert ps.identification_confidence == "low"
    assert len(ps.probes) == 1 and ps.probes[0].note == 60 and ps.probes[0].gate_s == 1.5
