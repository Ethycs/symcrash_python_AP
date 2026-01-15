"""
Unified Preset Modification Cache
==================================

Tracks modification times for preset files to enable incremental
compilation, rendering, and testing.

Single source of truth for "has this preset changed?"
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


CACHE_VERSION = 1
CACHE_FILENAME = ".test_cache.json"


@dataclass
class PresetCacheEntry:
    """Cache entry for a single preset."""
    mtime: float = 0.0
    last_compiled: float = 0.0
    last_rendered: float = 0.0
    last_tested: float = 0.0
    capsule_id: str = ""


@dataclass
class PresetCache:
    """Unified modification cache for presets."""
    version: int = CACHE_VERSION
    presets: Dict[str, PresetCacheEntry] = field(default_factory=dict)


def get_cache_path(library_dir: Path) -> Path:
    """Get path to cache file."""
    return library_dir / CACHE_FILENAME


def load_cache(library_dir: Path) -> PresetCache:
    """
    Load cache from disk.

    Args:
        library_dir: Path to preset_library directory

    Returns:
        PresetCache instance (empty if file doesn't exist or is invalid)
    """
    cache_path = get_cache_path(library_dir)

    if not cache_path.exists():
        return PresetCache()

    try:
        with open(cache_path, "r") as f:
            data = json.load(f)

        # Check version
        if data.get("version", 0) != CACHE_VERSION:
            # Version mismatch, return empty cache
            return PresetCache()

        # Parse entries
        cache = PresetCache(version=data.get("version", CACHE_VERSION))
        for rel_path, entry_data in data.get("presets", {}).items():
            cache.presets[rel_path] = PresetCacheEntry(
                mtime=entry_data.get("mtime", 0.0),
                last_compiled=entry_data.get("last_compiled", 0.0),
                last_rendered=entry_data.get("last_rendered", 0.0),
                last_tested=entry_data.get("last_tested", 0.0),
                capsule_id=entry_data.get("capsule_id", ""),
            )

        return cache

    except (json.JSONDecodeError, KeyError, TypeError):
        return PresetCache()


def save_cache(cache: PresetCache, library_dir: Path) -> None:
    """
    Save cache to disk.

    Args:
        cache: PresetCache instance to save
        library_dir: Path to preset_library directory
    """
    cache_path = get_cache_path(library_dir)

    # Ensure directory exists
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable dict
    data = {
        "version": cache.version,
        "presets": {
            rel_path: asdict(entry)
            for rel_path, entry in cache.presets.items()
        }
    }

    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)


def get_modified_presets(
    presets_dir: Path,
    library_dir: Path,
    pattern: Optional[str] = None,
    check_compiled: bool = True,
    check_rendered: bool = True,
) -> List[Path]:
    """
    Get list of preset files that have been modified since last processing.

    Args:
        presets_dir: Path to presets/ directory
        library_dir: Path to preset_library/ directory
        pattern: Optional filter pattern (e.g., "brass")
        check_compiled: If True, check if preset needs recompilation
        check_rendered: If True, check if preset needs re-rendering

    Returns:
        List of paths to modified preset YAML files
    """
    cache = load_cache(library_dir)
    modified = []

    # Scan all subdirectories for YAML files
    for subdir in ["instruments", "synths", "macros"]:
        yaml_dir = presets_dir / subdir
        if not yaml_dir.exists():
            continue

        for yaml_path in yaml_dir.glob("*.yaml"):
            # Apply pattern filter
            if pattern and pattern.lower() not in yaml_path.stem.lower():
                continue

            # Get relative path for cache key
            rel_path = f"{subdir}/{yaml_path.name}"

            # Get current mtime
            current_mtime = yaml_path.stat().st_mtime

            # Check cache
            entry = cache.presets.get(rel_path)

            if entry is None:
                # Not in cache, needs processing
                modified.append(yaml_path)
            elif current_mtime > entry.mtime:
                # File modified since last cache update
                modified.append(yaml_path)
            elif check_compiled and entry.last_compiled < entry.mtime:
                # Needs recompilation
                modified.append(yaml_path)
            elif check_rendered and entry.last_rendered < entry.mtime:
                # Needs re-rendering
                modified.append(yaml_path)

    return modified


def get_all_presets(
    presets_dir: Path,
    pattern: Optional[str] = None,
) -> List[Path]:
    """
    Get list of all preset YAML files.

    Args:
        presets_dir: Path to presets/ directory
        pattern: Optional filter pattern

    Returns:
        List of paths to preset YAML files
    """
    all_presets = []

    for subdir in ["instruments", "synths", "macros"]:
        yaml_dir = presets_dir / subdir
        if not yaml_dir.exists():
            continue

        for yaml_path in yaml_dir.glob("*.yaml"):
            if pattern and pattern.lower() not in yaml_path.stem.lower():
                continue
            all_presets.append(yaml_path)

    return all_presets


def mark_compiled(
    library_dir: Path,
    preset_path: Path,
    presets_dir: Path,
    capsule_id: str,
) -> None:
    """
    Mark a preset as compiled.

    Args:
        library_dir: Path to preset_library/ directory
        preset_path: Path to the preset YAML file
        presets_dir: Path to presets/ directory
        capsule_id: ID of the compiled capsule
    """
    cache = load_cache(library_dir)

    # Get relative path
    try:
        rel_path = str(preset_path.relative_to(presets_dir))
    except ValueError:
        # Not relative, use parts
        rel_path = f"{preset_path.parent.name}/{preset_path.name}"

    # Get or create entry
    entry = cache.presets.get(rel_path, PresetCacheEntry())

    # Update entry
    entry.mtime = preset_path.stat().st_mtime
    entry.last_compiled = time.time()
    entry.capsule_id = capsule_id

    cache.presets[rel_path] = entry
    save_cache(cache, library_dir)


def mark_rendered(
    library_dir: Path,
    preset_path: Path,
    presets_dir: Path,
) -> None:
    """
    Mark a preset as rendered.

    Args:
        library_dir: Path to preset_library/ directory
        preset_path: Path to the preset YAML file
        presets_dir: Path to presets/ directory
    """
    cache = load_cache(library_dir)

    # Get relative path
    try:
        rel_path = str(preset_path.relative_to(presets_dir))
    except ValueError:
        rel_path = f"{preset_path.parent.name}/{preset_path.name}"

    # Get or create entry
    entry = cache.presets.get(rel_path, PresetCacheEntry())

    # Update entry
    entry.mtime = preset_path.stat().st_mtime
    entry.last_rendered = time.time()

    cache.presets[rel_path] = entry
    save_cache(cache, library_dir)


def mark_tested(
    library_dir: Path,
    preset_path: Path,
    presets_dir: Path,
) -> None:
    """
    Mark a preset as tested.

    Args:
        library_dir: Path to preset_library/ directory
        preset_path: Path to the preset YAML file
        presets_dir: Path to presets/ directory
    """
    cache = load_cache(library_dir)

    # Get relative path
    try:
        rel_path = str(preset_path.relative_to(presets_dir))
    except ValueError:
        rel_path = f"{preset_path.parent.name}/{preset_path.name}"

    # Get or create entry
    entry = cache.presets.get(rel_path, PresetCacheEntry())

    # Update entry
    entry.mtime = preset_path.stat().st_mtime
    entry.last_tested = time.time()

    cache.presets[rel_path] = entry
    save_cache(cache, library_dir)


def clear_cache(library_dir: Path) -> None:
    """
    Clear the cache file.

    Args:
        library_dir: Path to preset_library/ directory
    """
    cache_path = get_cache_path(library_dir)
    if cache_path.exists():
        cache_path.unlink()


def get_cache_entry(
    library_dir: Path,
    preset_path: Path,
    presets_dir: Path,
) -> Optional[PresetCacheEntry]:
    """
    Get cache entry for a preset.

    Args:
        library_dir: Path to preset_library/ directory
        preset_path: Path to the preset YAML file
        presets_dir: Path to presets/ directory

    Returns:
        PresetCacheEntry if found, None otherwise
    """
    cache = load_cache(library_dir)

    try:
        rel_path = str(preset_path.relative_to(presets_dir))
    except ValueError:
        rel_path = f"{preset_path.parent.name}/{preset_path.name}"

    return cache.presets.get(rel_path)
