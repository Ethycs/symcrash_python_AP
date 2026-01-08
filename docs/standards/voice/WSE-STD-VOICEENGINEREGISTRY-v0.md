Below are **two normative specs**:

1. **WSE-STD-VOICEENGINEREGISTRY v0** (full, implementable, frozen interfaces)
2. **WSE-STD-CAPSULE-COMPAT v0** — a **ControlRegistry + VoiceFrame contract** that makes a **Capsule object** portable and engine-compatible (without making the Library responsible for morphing)

---

# WSE-STD-VOICEENGINEREGISTRY v0 — Voice Engine Registry (Truth Table)

**Status:** Normative (v0 frozen)
**Applies to:** VoiceEngine (CMMM/VM), per-voice init, runtime param updates, preset/capsule load, snapshot load
**Depends on:**

* WSE-STD-CONTROLREGISTRY v0 (cockpit controls truth table)
* WSE-STD-020 (VoiceFrame ABI)
* WSE-STD-DEGRADE v0 (planner outputs)

---

## 0. Purpose

The **VoiceEngineRegistry** is the **single source of truth** for **engine-owned parameters** (not cockpit macros), including:

* **Voice DSP configuration** (oscillator mode selection, tree ids, IDWPT config, wet path toggles)
* **CMMM/L5 selectors** (regime graph id, program routing ids, L5 object selectors)
* **Planner hints** (requested K policy, tail policy hints, feature gates)
* **Engine-owned stateful sub-engines** that must be configured deterministically (if enabled)

**Non-goal:** Replacing cockpit macro controls. Cockpit remains the macro surface and emits `VoiceFrame`s via ControlRegistry bindings.

---

## 0.1 Presets vs Snapshots (MUST)

This standard distinguishes two persistence artifacts:

### A) ProgramPreset (aka “capsule preset”)

Portable “instrument identity” + defaults:

* program identity / InstrumentIR reference
* registry default overrides (GLOBAL / PROGRAM / PER_VOICE init)
* deterministic seed policy

**MUST NOT** contain runtime histories (coeff vectors, osc phase, ADSR phase, PRNG state, dwell timers, lane timers).

### B) Snapshot (aka “checkpoint/replay”)

Exact runtime state checkpoint that can resume bit-identically:

* full per-voice runtime state (VMState + CMMM time state)
* global runtime state required for determinism (scheduler cursors, global PRNG if used)

---

## 1. Registry Entry Format (Frozen v0)

```python
@dataclass
class VoiceParamEntry:
    """Single voice-engine parameter definition."""
    param_id: int                # VoiceParamID enum value (frozen allocation §11)
    name: str                    # Human-readable name
    value_type: ValueType        # Q15U, I16, U16, BOOL, ENUM8
    min_value: int               # Inclusive minimum
    max_value: int               # Inclusive maximum
    default_value: int           # Default value used if no preset override
    update_mode: UpdateMode      # ABS, DELTA, BOTH
    persistence: Persistence     # PER_VOICE, PROGRAM, GLOBAL
    domain: Domain               # CORE, CMMM, CCS, IDWPT, PLANNER, DEBUG
    binding_target: str          # What it affects (§3)
    flags: int                   # bitflags (cumulative, init_only, etc.) §1.5
```

### 1.1 ValueType (Frozen v0)

Reuse the exact ValueType enum used by ControlRegistry:

* `Q15U`, `I16`, `U16`, `BOOL`, `ENUM8`

### 1.2 UpdateMode (Frozen v0)

```python
class UpdateMode(IntEnum):
    ABS = 0      # set value directly
    DELTA = 1    # add delta (signed)
    BOTH = 2     # accept both modes
```

### 1.3 Persistence (Frozen v0)

```python
class Persistence(IntEnum):
    PER_VOICE = 0   # applied per-voice on allocation; may be mutated at runtime per-voice
    PROGRAM  = 1    # shared defaults for voices under a program; may be mutated via program change
    GLOBAL   = 2    # global engine-wide parameter
```

### 1.4 Determinism rule (MUST)

For any param:

* clamp to `[min_value, max_value]` **before use**
* updates are **integer-only**
* snapshot/load must reproduce **bit-identical state** when re-run with identical subsequent input frames

### 1.5 Flags (Frozen v0)

`flags` bitfield (u16 recommended):

* `CUMULATIVE` (1<<0): multiple params with same binding add deterministically (e.g., cents offsets)
* `INIT_ONLY`  (1<<1): only applied at voice allocation; runtime updates are ignored
* `DEBUG_ONLY` (1<<2): ignored unless debug build
* `BUDGET_HINT` (1<<3): may be used by planner as hint but MUST NOT mutate stored defaults

---

## 2. Parameter Application Model (v0)

### 2.1 Ownership boundary

* **Cockpit controls** → CockpitEngine (ControlRegistry)
* **VoiceEngine parameters** → VoiceEngine (VoiceEngineRegistry)

### 2.2 Precedence (MUST)

Per block, per voice:

1. Apply `VoiceFrame` fields emitted by cockpit (already slewed)
2. Apply VoiceEngineRegistry parameters **only** for engine-owned targets (mode selects, ids, feature flags)
3. Planner (DEGRADE) may override **effective** K/tail/op plan to fit budget, but **MUST NOT** mutate stored defaults

### 2.3 Voice allocation application (MUST)

On allocation (NOTE_ON / allocator event), apply:

1. GLOBAL defaults (registry defaults overridden by preset)
2. PROGRAM defaults
3. PER_VOICE init defaults
4. runtime mutable state starts from deterministic reset state (§7 Snapshot)

---

## 3. Binding Target Strings (Frozen v0)

Binding targets are **strings** interpreted by VoiceEngine. v0 freezes the *meaning* of each target used by the registry.

**Core targets (examples):**

* `core.regime_graph_id` (CMMM regime graph selector)
* `core.program_id` (instrument/program identity selector)
* `core.tree_id` (tree selection)
* `core.tree_family_id` (harmonic vs exotic family selection)
* `core.seed_policy` (determinism seed rule selector)

**CCS/osc targets:**

* `ccs.osc_mode` (Mode A/B selection)
* `ccs.wave_id` (wave table / harmonic weights selection)
* `ccs.harmonic_cap` (requested harmonic count; actual may be bandlimited)

**IDWPT targets:**

* `idwpt.k_request` (requested K rung; planner can override effective)
* `idwpt.tail_policy` (hint: FULL/MID/MIN preference; planner can override effective)
* `idwpt.tree_xfade_blocks` (tree switching crossfade length)

**Planner targets:**

* `planner.feature_mask` (enable flags, subject to budget)

**Rule (MUST):** If multiple params bind to the same target and include `CUMULATIVE`, they MUST sum deterministically in canonical order: `(domain asc, param_id asc)`.

---

## 4. Default Registry Table (Frozen v0)

**Normative:** v0 must ship with a registry table. You may extend later, but **may not change** types/ranges for existing IDs.

At minimum (recommended), include:

### CORE (0–31)

* `core.program_id` (ENUM8 or U16)
* `core.regime_graph_id` (ENUM8 or U16)
* `core.tree_id` (ENUM8)
* `core.voice_class` (ENUM8) (planner routing hint)
* `core.seed_policy` (ENUM8)

### CMMM (32–63)

* `cmmm.route_id` (ENUM8)
* `cmmm.regime_bias` (Q15U)
* `cmmm.dwell_min_blocks` (U16)
* `cmmm.transition_policy` (ENUM8)

### CCS (64–127)

* `ccs.osc_mode` (ENUM8)
* `ccs.wave_id` (ENUM8)
* `ccs.h_request` (U16)
* `ccs.pwm_enable` (BOOL)
* `ccs.fm_enable` (BOOL)
* `ccs.unison_voices` (U16)

### IDWPT (128–159)

* `idwpt.k_request` (ENUM8 mapped to K ladder)
* `idwpt.tail_policy_hint` (ENUM8)
* `idwpt.tree_xfade_blocks` (U16)
* `idwpt.wet_enable` (BOOL)

### PLANNER (160–191)

* `planner.max_ticks_hint` (U16)
* `planner.allow_nonlinear_hint` (BOOL)

### DEBUG (240–255)

* `debug.bypass_idwpt` (BOOL)
* `debug.freeze_coeffs` (BOOL)

---

## 5. Param Update Packets (v0)

### 5.1 VoiceParamUpdateFrame (UART-friendly)

**Endianness:** little-endian
**All integer-only.**

```
struct VoiceParamUpdateFrameV0 {
  u8  abi_version = 0;
  u8  update_mode;        // ABS/DELTA
  u8  persistence_scope;  // PER_VOICE/PROGRAM/GLOBAL (targeting scope)
  u8  domain;             // Domain enum
  u16 voice_id;           // 0..V-1 or 0xFFFF for “all voices” if allowed for this param
  u16 param_id;           // VoiceParamID
  i16 value;              // interpreted per ValueType; clamped by registry
  u16 crc16;              // optional but recommended
}
```

**Rules (MUST):**

* Loader clamps based on registry entry.
* If `INIT_ONLY`, runtime updates are ignored (but may be accepted for program changes that trigger re-allocation).
* If `persistence_scope` conflicts with the registry’s persistence, the update MUST be rejected deterministically.

---

## 6. ProgramPreset (Capsule Preset) Format (v0)

### 6.1 Required contents (MUST)

A ProgramPreset MUST contain:

1. **Identity**

* program_id or InstrumentIR reference
* optional regime_graph_id
* engine compatibility metadata (abi versions)

2. **Default overrides**

* overrides for `GLOBAL`, `PROGRAM`, `PER_VOICE init` subsets of VoiceEngineRegistry

3. **Seed policy**

* master seed
* derivation rule

### 6.2 Override rules (MUST)

Preset may override `default_value` but MUST NOT change:

* `value_type`
* `min_value/max_value`
* `update_mode`
* `persistence`
* binding_target meaning

On load, preset values MUST be clamped by registry ranges.

### 6.3 Bundle layout (Normative)

A ProgramPreset bundle MUST include:

* `manifest.json`
* `voice_params.bin` (packed overrides)
* `control_defaults.bin` (optional; ControlRegistry defaults override)
* `instrument_ir.json` **OR** `(program_id, asset_refs)` (implementation-defined)

**manifest.json (MUST fields):**

* `format`: `"WSE_PRESET_V0"`
* `preset_version`: `0`
* `control_registry_abi`: `0`
* `voice_registry_abi`: `0`
* `voiceframe_abi`: `0`
* `sample_rate_hz`, `block_size`
* `program_id` or `instrument_ir_id`
* CRCs for each blob (recommended)

### 6.4 voice_params.bin packing (MUST)

Entries sorted lexicographically by:
`(persistence asc, domain asc, param_id asc)`

Each entry:

```
u8  persistence
u8  value_type
u16 param_id
i16 value
```

Optional: CRC16 per block.

---

## 7. Snapshot Format (v0)

### 7.1 Snapshot requirements (MUST)

Snapshot MUST allow:

* PROGRAM-only restore (no runtime history)
* per-voice subset restore
* FULL checkpoint restore (all voices + globals)

### 7.2 No dicts, no floats (MUST)

Snapshot is:

* fixed endianness (little)
* integer-only
* fixed-length arrays for core state vectors

### 7.3 Container (Normative)

```
Header:
  char[4] magic = "WSES"
  u16 snapshot_version = 0
  u16 endian = 0xFFFE
  u16 chunk_count
  u16 reserved
  u32 global_crc32

Chunk:
  u16 chunk_type   // CORE=1, CMMM=2, CCS=3, IDWPT=4, GLOBAL=5
  u16 chunk_version
  u16 voice_id     // 0..V-1 or 0xFFFF for global chunk
  u16 reserved
  u32 payload_len
  u8  payload[payload_len]
  u32 chunk_crc32
```

**Loader rule (MUST):** unknown chunk types are skipped deterministically.

---

## 8. Relationship to Other Standards

* ControlRegistry defines cockpit controls (IDs, ranges, slew, binding targets).
* VoiceEngineRegistry defines engine-owned parameters (IDs, ranges, persistence, binding targets).
* VoiceFrame ABI carries cockpit-emitted, slewed block controls into the engine.
* DEGRADE chooses effective plan (K/tail/op list) to satisfy budgets without mutating stored defaults.

---

## 9. Conformance Checklist (v0)

An implementation conforms iff it:

* [ ] Implements registry entry format and frozen ID allocation ranges (§1, §11)
* [ ] Clamps all param values to registry ranges (§1.4)
* [ ] Applies precedence rules: VoiceFrame → VoiceRegistry → Planner (§2.2)
* [ ] Implements deterministic preset override rules (§6.2)
* [ ] Implements packed override binary sorting and layout (§6.4)
* [ ] Implements snapshot container + chunking + CRCs (§7)
* [ ] Supports partial restore modes (§7.1)
* [ ] Produces bit-identical output after snapshot restore given identical subsequent inputs (§0.1B)

---

## 10. Implementation Guide (reference)

Recommended modules:

* `voice_engine_registry.py`

  * `VOICE_ENGINE_REGISTRY_V0: List[VoiceParamEntry]`
  * clamp/apply/update methods

* `preset_bundle.py`

  * load/save manifest.json
  * pack/unpack `voice_params.bin`
  * apply override policy and precedence

* `snapshot_bundle.py`

  * chunk container codec
  * per-domain `to_bytes()/from_bytes()`

* `voice_state_codec.py`

  * `VMState` serialization
  * `CMMMState` serialization

---

## 11. VoiceParamID allocation plan (Frozen v0)

* CORE: `0–31`
* CMMM: `32–63`
* CCS: `64–127`
* IDWPT: `128–159`
* PLANNER: `160–191`
* DEBUG: `240–255`

---

---
