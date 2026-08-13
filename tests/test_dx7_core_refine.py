"""Tests for the DX7 core refinement batch driver."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "gym" / "dx7_core_refine.py"
SPEC = importlib.util.spec_from_file_location("dx7_core_refine", MODULE_PATH)
batch = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = batch
SPEC.loader.exec_module(batch)


def test_all_core_presets_resolve_to_local_rom_voices():
    preset_dir = REPO_ROOT / "presets" / "instruments"
    sources = [batch.preset_source(preset_dir / name) for name in batch.CORE_PRESETS]
    assert len(sources) == 8
    assert {bank.name for bank, _ in sources} == {"rom1a.syx"}
    assert {voice for _, voice in sources} == {1, 4, 11, 15, 19, 21, 22, 26}
