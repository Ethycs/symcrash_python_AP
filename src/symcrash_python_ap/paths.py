"""Canonical project paths for preset authoring and generated artifacts.

``presets/`` is the only public preset root.  Human-authored YAML lives in
its category folders; disposable compiler output lives in ``presets/.build``.
Keeping these paths here prevents tools from independently recreating the old
top-level ``capsules/`` and ``preset_library/`` trees.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# Tracked, human-authored inputs.
PRESETS_DIR = REPO_ROOT / "presets"
PRESET_INSTRUMENTS_DIR = PRESETS_DIR / "instruments"
PRESET_SYNTHS_DIR = PRESETS_DIR / "synths"
PRESET_MACROS_DIR = PRESETS_DIR / "macros"

# Ignored, reproducible compiler/runtime output.
PRESET_BUILD_DIR = PRESETS_DIR / ".build"
PRESET_CAPSULES_DIR = PRESET_BUILD_DIR / "capsules"
PRESET_INDEX_PATH = PRESET_BUILD_DIR / "index" / "library_index.json"
PRESET_PREVIEWS_DIR = PRESET_BUILD_DIR / "previews"


__all__ = [
    "REPO_ROOT",
    "PRESETS_DIR",
    "PRESET_INSTRUMENTS_DIR",
    "PRESET_SYNTHS_DIR",
    "PRESET_MACROS_DIR",
    "PRESET_BUILD_DIR",
    "PRESET_CAPSULES_DIR",
    "PRESET_INDEX_PATH",
    "PRESET_PREVIEWS_DIR",
]
