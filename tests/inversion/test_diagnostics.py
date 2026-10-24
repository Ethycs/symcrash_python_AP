"""G0: unsupported features are reported and preserved, never silently dropped."""
import copy
from pathlib import Path

import yaml

from symcrash_inversion.diagnostics import Diagnostic, check_supported

PRESETS = Path(__file__).resolve().parents[2] / "presets" / "instruments"


def _load_all():
    return {p.stem: yaml.safe_load(p.read_text(encoding="utf-8")) for p in sorted(PRESETS.glob("*.yaml"))}


def test_fm6_presets_are_supported_and_osc_presets_are_reported():
    seen_fm6 = seen_osc = 0
    for name, d in _load_all().items():
        before = copy.deepcopy(d)
        diags = check_supported(d)
        assert d == before, f"check_supported mutated {name}"
        types = {n.get("type") for n in d["instrument_ir_v0"]["graph"].values()}
        if "FM6" in types:
            seen_fm6 += 1
            assert diags == [], f"{name} unexpectedly reported: {diags}"
        if "Osc" in types:
            seen_osc += 1
            assert any(x.code == "unsupported_feature" and x.path.startswith("/graph/") and x.path.endswith("/type")
                       for x in diags), name
            assert all(isinstance(x, Diagnostic) for x in diags)
    assert seen_fm6 >= 20 and seen_osc >= 20


def test_sections_outside_schema_are_reported_with_paths():
    d = yaml.safe_load((PRESETS / "dx7_e_piano_1_v0.yaml").read_text(encoding="utf-8"))
    d["instrument_ir_v0"]["temperament"] = {"cents": [0] * 7}
    paths = {x.path for x in check_supported(d)}
    assert "/temperament" in paths
