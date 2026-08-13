"""Unit tests for the block-coordinate refinement harness."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "gym" / "auto_refine.py"
SPEC = importlib.util.spec_from_file_location("auto_refine", MODULE_PATH)
auto_refine = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = auto_refine
SPEC.loader.exec_module(auto_refine)


def test_parse_linear_grid():
    spec = auto_refine.parse_grid("/fm/mod_index=1:2:3")
    assert spec.address == "/fm/mod_index"
    assert spec.values() == [1.0, 1.5, 2.0]


def test_parse_log_grid():
    spec = auto_refine.parse_grid("/amp/decay=10:1000:3:log")
    assert np.allclose(spec.values(), [10.0, 100.0, 1000.0])


@pytest.mark.parametrize(
    "encoded",
    ["fm/mod_index=1:2:3", "/fm/mod_index=2:1:3", "/fm/mod_index=1:2:1"],
)
def test_invalid_grid_rejected(encoded):
    with pytest.raises(argparse.ArgumentTypeError):
        auto_refine.parse_grid(encoded)


def test_internal_stereo_is_mixed_on_sample_axis():
    stereo = np.vstack([np.ones(8), np.zeros(8)])
    mono = auto_refine._mono_candidate(stereo)
    assert mono.shape == (8,)
    assert np.allclose(mono, 0.5)


def test_workbench_exposes_fm_mod_index():
    from symcrash_python_ap.instrument_ir import load_instrument
    from symcrash_workbench.params import build_registry

    preset = REPO_ROOT / "presets" / "instruments" / "dx7_e_piano_1_v0.yaml"
    ir = load_instrument(preset)
    param = build_registry()["/fm/mod_index"]
    assert param.get(ir) == 2.0
    assert param.set(ir, 1.75)
    assert param.get(ir) == 1.75


def test_full_committee_is_a_hard_acceptance_gate():
    assert auto_refine.proposal_improves({"composite": 0.2}, {"composite": 0.1})
    assert auto_refine.proposal_improves({"composite": 0.2}, {"composite": 0.2})
    assert not auto_refine.proposal_improves({"composite": 0.2}, {"composite": 0.21})
