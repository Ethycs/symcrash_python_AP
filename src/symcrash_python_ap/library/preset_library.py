"""
Preset Library (WSE-STD-PRESETLIB v0)
======================================

Capsule storage, indexing, validation, import/export, and resolution.

Features:
- Transactional import with CRC validation (§3)
- Strict vs lenient compatibility modes (§3.2)
- Deterministic resolution + dependency ordering (§4, §7)
- Export produces valid IPLC capsule bundles (§5)
- Metadata mutation (§6)

Storage Layout:
    library_root/
      capsules/
        instruments/<capsule_id>/<revision_id>/
          manifest.json
          voice_params.bin
          control_defaults.bin
        synths/<capsule_id>/<revision_id>/
          ...
        assets/<capsule_id>/<revision_id>/
          ...
        macros/<capsule_id>/<revision_id>/
          ...
      assets/<asset_id>/<revision_id>/
        ...
      index/library_index.json

Category folders are determined by CapsuleKind:
  INSTRUMENT -> instruments/
  SYNTH      -> synths/
  ASSET      -> assets/
  MACRO      -> macros/
"""

import json
import os
import shutil
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Union

from .capsule import CapsuleV0, CapsuleKind, AssetRef, AssetType
from .capsule_codec import (
    capsule_to_bytes,
    capsule_from_bytes,
    save_capsule_bundle,
    load_capsule_bundle,
)
from .capsule_loader import check_compatibility
from .lifecycle import (
    LifecycleState,
    CapsuleRecord,
    create_published_record,
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ImportResult:
    """Result of import transaction (§3)."""
    success: bool
    capsule_id: Optional[str] = None
    revision_id: Optional[int] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class AssetRecord:
    """Asset storage record."""
    asset_id: str
    asset_type: AssetType
    revision_id: int
    storage_path: str
    hash: Optional[str] = None


@dataclass
class Query:
    """Query filter for list_capsules (§4.3)."""
    kind: Optional[CapsuleKind] = None
    tags: Optional[List[str]] = None
    name_substring: Optional[str] = None
    sample_rate_hz: Optional[int] = None
    block_size: Optional[int] = None
    include_deprecated: bool = False
    include_retired: bool = False


class ResolutionSelector(IntEnum):
    """Resolution strategy (§4.1)."""
    LATEST_COMPATIBLE = 0  # Highest revision where compatible=True
    LATEST = 1             # Highest revision (any compatibility)
    BY_REVISION = 2        # Explicit revision_id


# =============================================================================
# LIBRARY INDEX
# =============================================================================

@dataclass
class RevisionEntry:
    """Index entry for a single revision."""
    path: str
    deprecated: bool = False
    retired: bool = False
    compatible: bool = True
    imported_at: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    name: str = ""
    kind: str = ""


@dataclass
class CapsuleIndex:
    """Index entry for a capsule_id."""
    latest_revision: int = 0
    revisions: Dict[int, RevisionEntry] = field(default_factory=dict)


class LibraryIndex:
    """
    In-memory library index with JSON persistence.

    Provides deterministic serialization (sorted keys).
    """

    def __init__(self):
        self.version = 0
        self.capsules: Dict[str, CapsuleIndex] = {}
        self.assets: Dict[str, Dict[int, str]] = {}  # asset_id → {revision → path}
        # program_id → capsule_id (for VoiceFramePacket.program_id resolution)
        self.program_id_map: Dict[int, str] = {}

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict with sorted keys."""
        return {
            "version": self.version,
            "capsules": {
                cid: {
                    "latest_revision": idx.latest_revision,
                    "revisions": {
                        str(rev): {
                            "path": entry.path,
                            "deprecated": entry.deprecated,
                            "retired": entry.retired,
                            "compatible": entry.compatible,
                            "imported_at": entry.imported_at,
                            "tags": sorted(entry.tags),
                            "name": entry.name,
                            "kind": entry.kind,
                        }
                        for rev, entry in sorted(idx.revisions.items())
                    },
                }
                for cid, idx in sorted(self.capsules.items())
            },
            "assets": {
                aid: {str(rev): path for rev, path in sorted(revs.items())}
                for aid, revs in sorted(self.assets.items())
            },
            # program_id → capsule_id mapping (for VoiceFramePacket resolution)
            "program_id_map": {
                str(pid): cid for pid, cid in sorted(self.program_id_map.items())
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'LibraryIndex':
        """Load from dict."""
        index = cls()
        index.version = data.get("version", 0)

        for cid, cdata in data.get("capsules", {}).items():
            capsule_idx = CapsuleIndex(latest_revision=cdata.get("latest_revision", 0))
            for rev_str, rdata in cdata.get("revisions", {}).items():
                rev = int(rev_str)
                capsule_idx.revisions[rev] = RevisionEntry(
                    path=rdata["path"],
                    deprecated=rdata.get("deprecated", False),
                    retired=rdata.get("retired", False),
                    compatible=rdata.get("compatible", True),
                    imported_at=rdata.get("imported_at"),
                    tags=rdata.get("tags", []),
                    name=rdata.get("name", ""),
                    kind=rdata.get("kind", ""),
                )
            index.capsules[cid] = capsule_idx

        for aid, revs in data.get("assets", {}).items():
            index.assets[aid] = {int(rev): path for rev, path in revs.items()}

        # Load program_id → capsule_id map
        for pid_str, cid in data.get("program_id_map", {}).items():
            index.program_id_map[int(pid_str)] = cid

        return index

    def save(self, path: str):
        """Save index to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> 'LibraryIndex':
        """Load index from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# CATEGORY FOLDER MAPPING
# =============================================================================

# Map CapsuleKind to storage folder name
CAPSULE_KIND_FOLDERS = {
    CapsuleKind.INSTRUMENT: "instruments",
    CapsuleKind.SYNTH: "synths",
    CapsuleKind.ASSET: "assets",
    CapsuleKind.MACRO: "macros",
}


def get_category_folder(kind: CapsuleKind) -> str:
    """Get storage folder name for capsule kind."""
    return CAPSULE_KIND_FOLDERS.get(kind, "instruments")


# =============================================================================
# PRESET LIBRARY V0
# =============================================================================

class PresetLibraryV0:
    """
    Preset Library per WSE-STD-PRESETLIB v0 §8.

    Provides:
    - Transactional import with CRC validation
    - Resolution by selector (latest_compatible, latest, by_revision)
    - Query with filters
    - Export to capsule bundle
    - Metadata mutation (flags, tags)
    """

    def __init__(
        self,
        root_path: str,
        mode: str = "lenient",
        sample_rate: int = 48000,
        block_size: int = 64,
    ):
        """
        Initialize library.

        Args:
            root_path: Library storage root directory
            mode: Default compatibility mode ("strict" or "lenient")
            sample_rate: Engine sample rate for compatibility checking
            block_size: Engine block size for compatibility checking
        """
        self.root_path = Path(root_path)
        self.mode = mode
        self.sample_rate = sample_rate
        self.block_size = block_size

        # ABI versions for compatibility
        self.control_registry_abi = 0
        self.voice_registry_abi = 0
        self.voiceframe_abi = 0

        # Initialize storage
        self._init_storage()

        # Load or create index
        self._index = self._load_or_create_index()

    def _init_storage(self):
        """Create storage directories if needed."""
        # Create category subfolders for capsules
        for folder in CAPSULE_KIND_FOLDERS.values():
            (self.root_path / "capsules" / folder).mkdir(parents=True, exist_ok=True)
        (self.root_path / "assets").mkdir(parents=True, exist_ok=True)
        (self.root_path / "index").mkdir(parents=True, exist_ok=True)

    def _load_or_create_index(self) -> LibraryIndex:
        """Load existing index or create new one."""
        index_path = self.root_path / "index" / "library_index.json"
        if index_path.exists():
            return LibraryIndex.load(str(index_path))
        return LibraryIndex()

    def _save_index(self):
        """Save index to disk."""
        index_path = self.root_path / "index" / "library_index.json"
        self._index.save(str(index_path))

    # =========================================================================
    # IMPORT (§3)
    # =========================================================================

    def import_capsule(
        self,
        bundle: Union[bytes, str, CapsuleV0],
        mode: str = None,
    ) -> ImportResult:
        """
        Import capsule with transactional semantics (§3.1).

        Steps:
        1. Validate bundle structure (required files present)
        2. Validate CRCs match manifest
        3. Validate compat block exists and is parseable
        4. Validate dependencies exist or can be imported
        5. Allocate revision_id
        6. Commit atomically (all files + index update)

        On failure: No partial capsule remains.

        Args:
            bundle: CapsuleV0 object (in-process import, e.g. straight from
                the compiler), path to bundle directory, or bytes (.wsecap)
            mode: Override default compatibility mode

        Returns:
            ImportResult
        """
        result = ImportResult(success=True)
        mode = mode or self.mode

        # Step 1: Load capsule
        try:
            if isinstance(bundle, CapsuleV0):
                capsule = bundle
            elif isinstance(bundle, bytes):
                capsule = capsule_from_bytes(bundle)
            else:
                capsule = load_capsule_bundle(bundle)
        except Exception as e:
            result.success = False
            result.errors.append(f"Failed to load bundle: {e}")
            return result

        capsule_id = capsule.identity.capsule_id
        result.capsule_id = capsule_id

        # Step 2: Validate CRCs (already done by load_capsule_bundle)

        # Step 3: Check compatibility
        compat = check_compatibility(
            capsule,
            current_control_abi=self.control_registry_abi,
            current_voice_abi=self.voice_registry_abi,
            current_voiceframe_abi=self.voiceframe_abi,
            current_sample_rate=self.sample_rate,
            current_block_size=self.block_size,
            strict=(mode == "strict"),
        )

        is_compatible = compat.compatible
        result.warnings.extend(compat.warnings)

        # In lenient mode, compat errors become warnings (§3.2)
        if mode == "lenient" and not is_compatible:
            for error in compat.errors:
                result.warnings.append(f"[lenient] {error}")

        if mode == "strict" and not is_compatible:
            result.success = False
            result.errors.extend(compat.errors)
            return result

        # Step 4: Validate dependencies (simplified - just check assets exist)
        for asset in capsule.assets:
            if asset.asset_id not in self._index.assets:
                result.warnings.append(f"Asset {asset.asset_id} not in library")

        # Step 5: Allocate revision_id
        if capsule_id in self._index.capsules:
            revision_id = self._index.capsules[capsule_id].latest_revision + 1
        else:
            revision_id = 1
            self._index.capsules[capsule_id] = CapsuleIndex()

        result.revision_id = revision_id

        # Step 6: Commit atomically
        # Get category folder from capsule kind
        category_folder = get_category_folder(capsule.identity.kind)
        storage_path = self.root_path / "capsules" / category_folder / capsule_id / str(revision_id)

        try:
            # Create directory
            storage_path.mkdir(parents=True, exist_ok=True)

            # Save capsule bundle
            save_capsule_bundle(capsule, str(storage_path))

            # Update index
            self._index.capsules[capsule_id].latest_revision = revision_id
            self._index.capsules[capsule_id].revisions[revision_id] = RevisionEntry(
                path=f"capsules/{category_folder}/{capsule_id}/{revision_id}",
                deprecated=False,
                retired=False,
                compatible=is_compatible,
                imported_at=datetime.now(timezone.utc).isoformat(),
                tags=capsule.identity.tags,
                name=capsule.identity.name,
                kind=capsule.identity.kind.name,
            )

            # Update program_id → capsule_id map (for VoiceFramePacket resolution)
            program_id = capsule.program.program_id
            if program_id is not None:
                self._index.program_id_map[program_id] = capsule_id

            # Save index
            self._save_index()

        except Exception as e:
            # Rollback: remove partial storage
            if storage_path.exists():
                shutil.rmtree(storage_path)
            result.success = False
            result.errors.append(f"Commit failed: {e}")
            return result

        return result

    def import_asset(
        self,
        bundle: Union[bytes, str],
        asset_id: str,
        asset_type: AssetType,
        mode: str = None,
    ) -> ImportResult:
        """
        Import asset bundle (§3).

        Args:
            bundle: Asset data (bytes or path)
            asset_id: Unique asset identifier
            asset_type: Type of asset
            mode: Compatibility mode (unused for assets)

        Returns:
            ImportResult
        """
        result = ImportResult(success=True)
        result.capsule_id = asset_id

        # Allocate revision
        if asset_id in self._index.assets:
            revision_id = max(self._index.assets[asset_id].keys()) + 1
        else:
            revision_id = 1
            self._index.assets[asset_id] = {}

        result.revision_id = revision_id

        # Store asset
        storage_path = self.root_path / "assets" / asset_id / str(revision_id)

        try:
            storage_path.mkdir(parents=True, exist_ok=True)

            if isinstance(bundle, bytes):
                with open(storage_path / "data.bin", 'wb') as f:
                    f.write(bundle)
            else:
                shutil.copytree(bundle, storage_path, dirs_exist_ok=True)

            # Update index
            self._index.assets[asset_id][revision_id] = f"assets/{asset_id}/{revision_id}"
            self._save_index()

        except Exception as e:
            if storage_path.exists():
                shutil.rmtree(storage_path)
            result.success = False
            result.errors.append(f"Asset import failed: {e}")

        return result

    # =========================================================================
    # RESOLVE (§4)
    # =========================================================================

    def resolve_capsule(
        self,
        capsule_id: str,
        selector: ResolutionSelector = ResolutionSelector.LATEST_COMPATIBLE,
        revision_id: int = None,
    ) -> Optional[CapsuleRecord]:
        """
        Resolve capsule by selector (§4.1).

        Args:
            capsule_id: Capsule identifier
            selector: Resolution strategy
            revision_id: Required if selector is BY_REVISION

        Returns:
            CapsuleRecord or None if not found
        """
        if capsule_id not in self._index.capsules:
            return None

        capsule_idx = self._index.capsules[capsule_id]

        if selector == ResolutionSelector.BY_REVISION:
            if revision_id is None or revision_id not in capsule_idx.revisions:
                return None
            rev = revision_id
        elif selector == ResolutionSelector.LATEST:
            rev = capsule_idx.latest_revision
        else:  # LATEST_COMPATIBLE
            # Find highest compatible revision
            compatible_revs = [
                r for r, entry in capsule_idx.revisions.items()
                if entry.compatible and not entry.retired
            ]
            if not compatible_revs:
                return None
            rev = max(compatible_revs)

        entry = capsule_idx.revisions[rev]

        # Load capsule from storage
        storage_path = self.root_path / entry.path
        capsule = load_capsule_bundle(str(storage_path))

        return CapsuleRecord(
            capsule=capsule,
            state=LifecycleState.PUBLISHED,
            revision_id=rev,
            storage_path=entry.path,
            deprecated=entry.deprecated,
            retired=entry.retired,
            compatible=entry.compatible,
            imported_at=entry.imported_at,
        )

    def resolve_by_program_id(
        self,
        program_id: int,
        selector: ResolutionSelector = ResolutionSelector.LATEST_COMPATIBLE,
    ) -> Optional[CapsuleRecord]:
        """
        Resolve capsule by program_id (from VoiceFramePacket).

        This is the primary lookup used by the runtime to resolve which
        capsule a VoiceFramePacket refers to based on its program_id field.

        Args:
            program_id: Program ID (from VoiceFramePacket.program_id)
            selector: Resolution strategy (default: LATEST_COMPATIBLE)

        Returns:
            CapsuleRecord or None if no capsule is registered for this program_id
        """
        # Look up capsule_id from program_id map
        capsule_id = self._index.program_id_map.get(program_id)
        if capsule_id is None:
            return None

        # Delegate to standard resolution
        return self.resolve_capsule(capsule_id, selector)

    def get_program_id_capsule_id(self, program_id: int) -> Optional[str]:
        """
        Get capsule_id for a program_id (without loading capsule).

        Args:
            program_id: Program ID to look up

        Returns:
            capsule_id if registered, None otherwise
        """
        return self._index.program_id_map.get(program_id)

    def get_capsule_bundle(self, record: CapsuleRecord) -> bytes:
        """
        Get capsule bundle bytes (§4.2).

        Args:
            record: CapsuleRecord from resolve

        Returns:
            Binary capsule data
        """
        return capsule_to_bytes(record.capsule)

    # =========================================================================
    # QUERY (§4.3)
    # =========================================================================

    def list_capsules(self, query: Query = None) -> List[CapsuleRecord]:
        """
        List capsules matching query (§4.3).

        Args:
            query: Filter criteria (None = all non-deprecated)

        Returns:
            List of CapsuleRecords (deterministic order)
        """
        query = query or Query()
        results = []

        # Iterate in sorted order for determinism
        for capsule_id in sorted(self._index.capsules.keys()):
            capsule_idx = self._index.capsules[capsule_id]

            for rev in sorted(capsule_idx.revisions.keys(), reverse=True):
                entry = capsule_idx.revisions[rev]

                # Apply filters
                if entry.deprecated and not query.include_deprecated:
                    continue
                if entry.retired and not query.include_retired:
                    continue

                if query.kind is not None:
                    if entry.kind != query.kind.name:
                        continue

                if query.tags:
                    if not any(tag in entry.tags for tag in query.tags):
                        continue

                if query.name_substring:
                    if query.name_substring.lower() not in entry.name.lower():
                        continue

                # Load capsule for full filtering
                storage_path = self.root_path / entry.path
                capsule = load_capsule_bundle(str(storage_path))

                if query.sample_rate_hz is not None:
                    if capsule.identity.engine_compat.sample_rate_hz != query.sample_rate_hz:
                        continue

                if query.block_size is not None:
                    if capsule.identity.engine_compat.block_size != query.block_size:
                        continue

                record = CapsuleRecord(
                    capsule=capsule,
                    state=LifecycleState.PUBLISHED,
                    revision_id=rev,
                    storage_path=entry.path,
                    deprecated=entry.deprecated,
                    retired=entry.retired,
                    compatible=entry.compatible,
                    imported_at=entry.imported_at,
                )
                results.append(record)

                # Only include latest revision per capsule in list
                break

        return results

    def resolve_dependencies(self, record: CapsuleRecord) -> List[AssetRecord]:
        """
        Get ordered list of required assets (§4.4).

        Order: (type asc, id asc, revision asc) for determinism.

        Args:
            record: CapsuleRecord

        Returns:
            Ordered list of AssetRecords
        """
        deps = []

        for asset_ref in record.capsule.assets:
            if asset_ref.asset_id in self._index.assets:
                revisions = self._index.assets[asset_ref.asset_id]
                if revisions:
                    latest_rev = max(revisions.keys())
                    deps.append(AssetRecord(
                        asset_id=asset_ref.asset_id,
                        asset_type=asset_ref.asset_type,
                        revision_id=latest_rev,
                        storage_path=revisions[latest_rev],
                        hash=asset_ref.asset_hash,
                    ))

        # Sort by (type, id, revision) for determinism
        deps.sort(key=lambda a: (a.asset_type, a.asset_id, a.revision_id))
        return deps

    # =========================================================================
    # EXPORT (§5)
    # =========================================================================

    def export_capsule(self, record: CapsuleRecord, output_path: str):
        """
        Export capsule to bundle directory (§5).

        Args:
            record: CapsuleRecord to export
            output_path: Destination directory
        """
        save_capsule_bundle(record.capsule, output_path)

    # =========================================================================
    # MUTATION (§6)
    # =========================================================================

    def set_flags(
        self,
        capsule_id: str,
        revision_id: int,
        deprecated: bool = None,
        retired: bool = None,
    ):
        """
        Set deprecated/retired flags (§6).

        Args:
            capsule_id: Capsule identifier
            revision_id: Revision to update
            deprecated: New deprecated flag (None = no change)
            retired: New retired flag (None = no change)
        """
        if capsule_id not in self._index.capsules:
            raise KeyError(f"Capsule not found: {capsule_id}")

        capsule_idx = self._index.capsules[capsule_id]
        if revision_id not in capsule_idx.revisions:
            raise KeyError(f"Revision not found: {revision_id}")

        entry = capsule_idx.revisions[revision_id]

        if deprecated is not None:
            entry.deprecated = deprecated
        if retired is not None:
            entry.retired = retired

        self._save_index()

    def set_tags(
        self,
        capsule_id: str,
        revision_id: int,
        tags: List[str],
    ):
        """
        Set tags (§6).

        Args:
            capsule_id: Capsule identifier
            revision_id: Revision to update
            tags: New tag list
        """
        if capsule_id not in self._index.capsules:
            raise KeyError(f"Capsule not found: {capsule_id}")

        capsule_idx = self._index.capsules[capsule_id]
        if revision_id not in capsule_idx.revisions:
            raise KeyError(f"Revision not found: {revision_id}")

        capsule_idx.revisions[revision_id].tags = sorted(tags)
        self._save_index()


__all__ = [
    'ImportResult',
    'AssetRecord',
    'Query',
    'ResolutionSelector',
    'PresetLibraryV0',
    'CAPSULE_KIND_FOLDERS',
    'get_category_folder',
]
