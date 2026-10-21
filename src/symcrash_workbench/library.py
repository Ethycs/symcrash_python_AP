"""
Preset catalog: what the workbench can load, addressable over OSC.

Scans presets/instruments/*.yaml (the authoring sources the session can
mutate and SAVE) and cross-references the generated capsule library index
(presets/.build/index/library_index.json) so capsule ids resolve back to
their YAML source where one exists.

Consumed by:
  * WorkbenchSession.list_presets() / load_preset()  (/preset/list, /preset/load)
  * make_layout.py (preset picker dropdown, populated at generation time)
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from symcrash_python_ap.paths import PRESET_INDEX_PATH, PRESET_INSTRUMENTS_DIR

PRESET_DIR = PRESET_INSTRUMENTS_DIR
LIBRARY_INDEX = PRESET_INDEX_PATH

# capsule ids look like "<name-slug>-<12 hex chars>"
_CAPSULE_HASH_RE = re.compile(r"-[0-9a-f]{12}$")


@dataclass
class PresetEntry:
    """One loadable preset (a YAML authoring source)."""

    stem: str                 # filename stem: string_pad_v0
    path: Path                # absolute YAML path
    preset_id: str = ""       # meta.id: string-pad-v0
    name: str = ""            # meta.name: String Pad
    capsule_ids: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.name or self.stem


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _read_meta(path: Path) -> dict:
    """Cheap meta scrape: id/name from the YAML header, no full IR load."""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        return (doc or {}).get("instrument_ir_v0", {}).get("meta", {}) or {}
    except Exception:
        return {}


def _capsule_index() -> Dict[str, dict]:
    """capsule_id -> latest-revision record (empty on any read problem)."""
    try:
        with open(LIBRARY_INDEX, "r", encoding="utf-8") as f:
            index = json.load(f)
        return dict(index.get("capsules", {}))
    except Exception:
        return {}


def scan_preset_catalog(preset_dir: Optional[Path] = None) -> List[PresetEntry]:
    """
    Build the catalog: every YAML under presets/instruments/, annotated
    with any capsule ids from the library index whose name slug resolves
    to that preset.
    """
    preset_dir = Path(preset_dir) if preset_dir else PRESET_DIR
    entries: List[PresetEntry] = []
    by_slug: Dict[str, PresetEntry] = {}
    for path in sorted(preset_dir.glob("*.yaml")):
        meta = _read_meta(path)
        entry = PresetEntry(
            stem=path.stem,
            path=path.resolve(),
            preset_id=str(meta.get("instrument_id") or meta.get("id") or ""),
            name=str(meta.get("name") or ""),
        )
        entries.append(entry)
        for key in filter(None, (entry.preset_id, _slug(entry.name),
                                 _slug(entry.stem))):
            by_slug.setdefault(_slug(key), entry)
            # meta ids conventionally end in -v<N>; capsules drop it
            base = re.sub(r"-v\d+$", "", _slug(key))
            if base:
                by_slug.setdefault(base, entry)

    for capsule_id in _capsule_index():
        base = _CAPSULE_HASH_RE.sub("", capsule_id)
        entry = by_slug.get(base)
        if entry is not None and capsule_id not in entry.capsule_ids:
            entry.capsule_ids.append(capsule_id)
    return entries


def resolve_preset(token: str,
                   catalog: Optional[List[PresetEntry]] = None,
                   preset_dir: Optional[Path] = None) -> Optional[PresetEntry]:
    """
    Resolve an id-or-filename to a catalog entry. Accepts a filename
    stem (string_pad_v0), a meta id (string-pad-v0), a display name, a
    capsule id (string-pad-<hash>), or a path to a YAML file.
    """
    token = str(token).strip()
    if not token:
        return None
    catalog = catalog if catalog is not None else scan_preset_catalog(preset_dir)

    p = Path(token)
    if p.suffix in (".yaml", ".yml") and p.exists():
        rp = p.resolve()
        for entry in catalog:
            if entry.path == rp:
                return entry
        meta = _read_meta(rp)
        return PresetEntry(stem=rp.stem, path=rp,
                           preset_id=str(meta.get("instrument_id") or ""),
                           name=str(meta.get("name") or ""))

    slug = _slug(token)
    for entry in catalog:
        if token in (entry.stem, entry.preset_id) or token in entry.capsule_ids:
            return entry
    for entry in catalog:
        if slug in (_slug(entry.stem), _slug(entry.preset_id),
                    _slug(entry.name)):
            return entry
    # capsule id with the hash suffix stripped
    base = _CAPSULE_HASH_RE.sub("", slug)
    if base != slug:
        return resolve_preset(base, catalog)
    return None


def send_preset_list(send, catalog: Optional[List[PresetEntry]] = None) -> int:
    """
    Reply to /preset/list on a feedback channel: one

        /preset/item  <stem> <meta-id> <name> <capsule-ids,csv>

    per catalog entry, then /preset/count <n>. `send` is any
    (address, args) callable (e.g. SimpleUDPClient.send_message).
    """
    catalog = catalog if catalog is not None else scan_preset_catalog()
    for entry in catalog:
        send("/preset/item", [entry.stem, entry.preset_id, entry.name,
                              ",".join(entry.capsule_ids)])
    send("/preset/count", len(catalog))
    return len(catalog)
