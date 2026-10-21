"""Preset source/build ownership contract."""

from pathlib import Path

from symcrash_python_ap.paths import (
    REPO_ROOT,
    PRESETS_DIR,
    PRESET_BUILD_DIR,
    PRESET_CAPSULES_DIR,
    PRESET_INDEX_PATH,
    PRESET_INSTRUMENTS_DIR,
    PRESET_MACROS_DIR,
)
from symcrash_python_ap.synth_master import SynthConfig
from symcrash_workbench import library as workbench_library


def test_preset_sources_and_generated_library_share_one_public_root():
    assert PRESET_INSTRUMENTS_DIR == PRESETS_DIR / "instruments"
    assert PRESET_MACROS_DIR == PRESETS_DIR / "macros"
    assert PRESET_BUILD_DIR == PRESETS_DIR / ".build"
    assert PRESET_CAPSULES_DIR == PRESET_BUILD_DIR / "capsules"
    assert PRESET_INDEX_PATH == PRESET_BUILD_DIR / "index" / "library_index.json"


def test_runtime_and_workbench_use_the_canonical_generated_library():
    assert Path(SynthConfig().library_dir) == PRESET_BUILD_DIR
    assert workbench_library.PRESET_DIR == PRESET_INSTRUMENTS_DIR
    assert workbench_library.LIBRARY_INDEX == PRESET_INDEX_PATH


def test_tracked_preset_sources_are_present():
    assert len(list(PRESET_INSTRUMENTS_DIR.glob("*.yaml"))) >= 50
    assert len(list(PRESET_MACROS_DIR.glob("*.yaml"))) >= 4


def test_legacy_top_level_preset_roots_are_retired():
    assert not (REPO_ROOT / "preset_library").exists()
    assert not (REPO_ROOT / "capsules").exists()
