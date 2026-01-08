
---

# WSE-STD-PRESETLIB v0 — Preset Library Storage + Index + Transactions

**Standard ID:** WSE-STD-PRESETLIB v0
**Title:** Preset Library (Capsule Storage, Indexing, Validation, Import/Export)
**Status:** Normative (v0 frozen)
**Applies to:** PresetLibrary service/module (Python + embedded host)
**Depends on:**

* WSE-STD-IPLC v0
* WSE-STD-CAPSULE-COMPAT v0

**Non-goal (v0):** The library does **not** morph or interpolate presets. It stores and serves capsules.

---

## 0. Purpose

Provide a deterministic, portable, editor-friendly mechanism to:

* store capsule bundles (Instrument/Synth/Asset)
* validate compatibility
* index/search by metadata and tags
* support transactional import/export
* provide “apply by id” and “resolve dependencies” operations

---

## 1. First-Class Objects (Frozen v0)

### 1.1 Library

`PresetLibraryV0` contains:

* **Capsules**: stored artifacts keyed by `(capsule_id, revision_id)`
* **Index**: searchable metadata store
* **Policy**: rules about deprecation/retirement and strict compatibility
* **Dependency table**: assets required by capsules

### 1.2 Capsule Record

A stored record MUST include:

* `capsule_id` (string)
* `revision_id` (u32 or string; monotonically increasing per capsule_id)
* manifest.json (as stored)
* content hashes (crc32/sha256 allowed; at least crc32)
* storage location pointer (path/blob id)
* flags: deprecated/retired

### 1.3 Asset Record

Assets are stored similarly, keyed by `asset_id` and `revision_id`.

---

## 2. Storage Layout (Normative)

Library MAY be:

* filesystem directory
* sqlite db + blob store
* flash segments (embedded)

But MUST present a logical layout:

```
/capsules/<capsule_id>/<revision_id>/...
/assets/<asset_id>/<revision_id>/...
/index/library_index.json OR sqlite index
```

**Rule:** Storage back-end is not standardized; the API is.

---

## 3. Import Transaction (Normative)

### 3.1 import_capsule(bundle) -> ImportResult (MUST)

Import is transactional:

1. Validate capsule bundle structure (required files present)
2. Validate CRCs match manifest.hashes
3. Validate compat block exists and is parseable
4. Validate (optional) dependencies exist or can be imported
5. Allocate `revision_id`
6. Commit all files + index update atomically

If any step fails, no partial capsule may remain.

### 3.2 Compatibility validation modes

Library MUST support two modes:

* `strict`: reject if ABIs/sr/block mismatch
* `lenient`: accept into library but mark “incompatible”; runtime apply must still refuse

---

## 4. Resolve + Apply Interface (Normative)

The library MUST provide:

### 4.1 resolve_capsule(capsule_id, selector) -> CapsuleRecord

`selector` can be:

* `latest_compatible`
* `latest`
* explicit `revision_id`

### 4.2 get_capsule_bundle(record) -> BundleBytes/Path

Return either path or bytes.

### 4.3 list_capsules(query) -> records

Query fields (v0 minimum):

* `kind`
* `tags`
* `name substring`
* `compat.sample_rate_hz`, `compat.block_size`
* `deprecated` filter

### 4.4 dependency closure

`resolve_dependencies(record)` returns a deterministic ordered list of required assets (order stable: `(type asc, id asc, revision asc)`).

---

## 5. Export Format (Normative)

Library export MUST produce a capsule bundle identical in structure to IPLC COMPILED/PUBLISHED bundles.

Export MUST preserve:

* manifest.json
* binary override files
* asset refs (either embedded or referenced, per policy)

---

## 6. Mutation Operations (v0)

Allowed metadata mutations (MUST support):

* set tags
* set deprecated/retired flags
* set notes/description

Not required in v0:

* editing overrides inside a stored capsule (that’s an editor/compiler job)

---

## 7. Determinism Rules (v0)

Library must be deterministic in:

* `latest_compatible` selection (tie-breaking: highest revision_id, then lexicographic stable)
* dependency ordering (type/id/revision)
* index serialization order (sort keys and lists)

---

## 8. Minimal API (Frozen v0)

```python
class PresetLibraryV0:
    def import_capsule(self, bundle: bytes | path, mode: str) -> ImportResult
    def import_asset(self, bundle: bytes | path, mode: str) -> ImportResult

    def resolve_capsule(self, capsule_id: str, selector: str) -> CapsuleRecord
    def get_capsule_bundle(self, record: CapsuleRecord) -> bytes | path

    def list_capsules(self, query: Query) -> list[CapsuleRecord]
    def resolve_dependencies(self, record: CapsuleRecord) -> list[AssetRecord]

    def set_flags(self, capsule_id: str, revision_id: int, deprecated: bool, retired: bool) -> None
    def set_tags(self, capsule_id: str, revision_id: int, tags: list[str]) -> None
```

---

## 9. Conformance Checklist (PresetLib v0)

* [ ] Transactional import with CRC validation (§3)
* [ ] Strict vs lenient compatibility modes (§3.2)
* [ ] Deterministic resolution + dependency ordering (§4, §7)
* [ ] Export produces valid IPLC capsule bundles (§5)
* [ ] Metadata mutation supported (§6)

---

## Two “what’s left” notes (so we don’t accidentally re-scope)

1. **Morphing standard** is explicitly deferred. In v0, morphing is runtime/pipeline behavior and not a library feature.
2. If you later add “multi-regime apples-to-apples morphing”, that will likely be a **Cockpit macro-space** / **VoiceEngine L5 space** standard, not PresetLib.

If you want, next I can write the **minimal “Capsule Apply RPC/packet”** (UART control-plane) so Pico/host can request: `apply(capsule_id)` and `apply(revision_id)` deterministically, with an optional “strict compat” flag.
