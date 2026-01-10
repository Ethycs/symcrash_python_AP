# WSE-STD-IPLC v0 — Instrument Preset Lifecycle

**Standard ID:** WSE-STD-IPLC v0
**Title:** Instrument Preset Lifecycle (Capsule Authoring → Compile → Library → Runtime Apply)
**Status:** Normative (v0 frozen)
**Applies to:** Toolchain + PresetLibrary + runtime loader (Cockpit + VoiceEngine)
**Depends on:**

* WSE-STD-CONTROLREGISTRY v0
* WSE-STD-VOICEENGINEREGISTRY v0
* WSE-STD-020 VoiceFrame ABI
* WSE-STD-CAPSULE-COMPAT v0 (Capsule compatibility contract)

**Non-goal (v0):** PresetLibrary does **not** own morphing. Morphing is pipeline behavior (later standard).
**Scope:** Lifecycle, artifact classes, versioning, compatibility gates, apply semantics, and required metadata.

---

## 0. Terms

* **Capsule**: a portable bundle that defines a program identity + defaults + asset refs (no runtime histories).
* **ProgramPreset**: the musical/shareable identity (capsule content).
* **Snapshot**: runtime checkpoint (out of scope for IPLC details; referenced only).
* **Compiler**: converts authoring inputs (InstrumentIR etc.) into a capsule bundle.
* **Apply**: load capsule into runtime registries and program selector, then run via VoiceFrames.

---

## 1. Artifact Kinds (Frozen v0)

IPLC recognizes four capsule kinds:

1. **CapsuleInstrument** — "instrument identity" (regime graph, tree family, voice defaults)
2. **CapsuleSynth** — "synth architecture identity" (osc mode, harmonic basis, FX toggles, etc.)
3. **CapsuleAsset** — "data only" (tables/materials/atlases); no control defaults required
4. **CapsuleMacro** — "performance macro" (vibrato, legato, tremolo, portamento); additive modulation

Each capsule kind MUST declare `kind` in its manifest.

---

## 2. IPLC States and Transitions (Frozen v0)

### 2.1 State machine

Capsule lifecycle states:

* **DRAFT**: authoring form (InstrumentIR graphs, loose JSON/YAML, editor graphs)
* **COMPILED**: capsule bundle produced by compiler; ABI-checked; checksummed
* **PUBLISHED**: stored in PresetLibrary with stable `capsule_id` and version
* **APPLIED**: loaded into runtime registries/program selector (no runtime histories)
* **RUNNING**: being exercised via VoiceFrames + VoiceEngine
* **DEPRECATED**: still loadable but not default-recommended
* **RETIRED**: not loadable unless explicit override (library policy)

### 2.2 Allowed transitions (MUST)

| From       | To         | Trigger                             |
| ---------- | ---------- | ----------------------------------- |
| DRAFT      | COMPILED   | compiler emits capsule bundle       |
| COMPILED   | PUBLISHED  | library import transaction succeeds |
| PUBLISHED  | APPLIED    | runtime `apply_capsule()`           |
| APPLIED    | RUNNING    | first VoiceFrame accepted           |
| PUBLISHED  | DEPRECATED | library metadata update             |
| DEPRECATED | RETIRED    | library policy update               |

**Rule:** A capsule MUST NOT transition backwards (e.g., PUBLISHED → COMPILED) except by producing a **new version**.

---

## 3. Canonical Capsule Bundle Layout (Frozen v0)

A capsule bundle is a directory or container with these required files:

* `manifest.json` (MUST)
* `voice_params.bin` (MUST for Instrument/Synth; optional for Asset)
* `control_defaults.bin` (OPTIONAL; recommended for Instrument/Synth)
* `instrument_ir.json` OR `program_ref.json` (MUST for Instrument/Synth)
* `assets/` (OPTIONAL; may store or reference tables/materials)

### 3.1 manifest.json (MUST fields)

```json
{
  "format": "WSE_CAPSULE_V0",
  "capsule_version": 0,
  "capsule_id": "string-stable-id",
  "kind": "INSTRUMENT|SYNTH|ASSET",
  "name": "human name",
  "author": "string",
  "tags": ["..."],

  "compat": {
    "control_registry_abi": 0,
    "voice_registry_abi": 0,
    "voiceframe_abi": 0,
    "sample_rate_hz": 48000,
    "block_size": 64
  },

  "program": {
    "program_id": 123,
    "instrument_ir_id": "optional",
    "regime_graph_id": 7,
    "tree_family_id": 2
  },

  "hashes": {
    "voice_params_crc32": 0,
    "control_defaults_crc32": 0,
    "instrument_ir_crc32": 0
  }
}
```

**Rule (MUST):** The compat block is the gatekeeper. Runtime MUST check it before apply.

---

## 4. Compiler Contract (Normative)

The compiler’s job is to produce a COMPILED capsule such that:

1. **All overrides are clampable** against registries
2. **All IDs are resolved** (program_id, graph_id)
3. **All assets are referenced or embedded** deterministically
4. **All ABI requirements are declared** in `manifest.json`

### 4.1 Compiler inputs (v0 minimum)

* InstrumentIR (or program authoring data)
* optional macro/control mapping hints
* optional asset table refs

### 4.2 Compiler outputs (MUST)

* Capsule bundle with required files and CRCs
* `capsule_id` stable across rebuilds unless identity changes (policy in §7)

---

## 5. Runtime Apply Semantics (Normative)

### 5.1 apply_capsule() (MUST)

When applying a capsule:

1. Validate `manifest.compat` matches runtime ABIs + sample rate + block size
2. Load program identity into VoiceEngine (program selector / graph selector)
3. Apply `voice_params.bin` into VoiceEngineRegistry defaults (clamp)
4. Apply `control_defaults.bin` into ControlRegistry defaults (clamp)
5. Reset voice allocation state deterministically (no runtime histories)
6. Start RUNNING when VoiceFrames begin

### 5.2 Unknown fields / IDs (MUST)

* Unknown **ControlIDs** / **VoiceParamIDs** MUST be ignored deterministically unless `strict_compat=true`
* Unknown **asset types** MUST be ignored deterministically unless required by program selector

### 5.3 Preset vs Snapshot boundary (MUST)

Applying a capsule MUST NOT:

* restore coeff vectors, oscillator phase, PRNG state, dwell timers
* restore scheduler cursors

Those belong to Snapshot standards (separate).

---

## 6. CapsuleMacro Lifecycle (Frozen v0)

CapsuleMacro is a first-class capsule kind for performance modulation presets.

### 6.1 MacroType Enum

| ID | Type | Description |
|----|------|-------------|
| 0 | VIBRATO | Pitch/amplitude/timbre LFO modulation |
| 1 | LEGATO | Smooth pitch transition (preserves phase) |
| 2 | TREMOLO | Amplitude-only LFO modulation |
| 3 | PORTAMENTO | Pitch glide between notes |

### 6.2 MacroProgram Object

```text
MacroProgram:
  macro_type: MacroType          # VIBRATO | LEGATO | TREMOLO | PORTAMENTO

  # LFO parameters (VIBRATO, TREMOLO)
  rate_hz: float                 # LFO rate in Hz (default 5.5)
  depth_pitch_cents: int         # Pitch modulation depth in cents
  depth_amp_q15: int             # Amplitude modulation depth (Q15)
  depth_timbre_q15: int          # Timbre modulation depth (Q15)

  # Envelope parameters
  delay_ms: float                # Delay before modulation onset
  ramp_ms: float                 # Ramp time to full depth

  # Glide parameters (LEGATO, PORTAMENTO)
  glide_time_ms: float           # Pitch glide duration
  glide_curve: int               # 0=linear, 1=exp, 2=log

  # Phase behavior (LEGATO)
  preserve_phase: bool           # Keep oscillator phase on note change
```

### 6.3 SystemMacro Mapping

| MacroType | SystemMacro ID | Slew Policy |
|-----------|----------------|-------------|
| VIBRATO | 11 (VIBRATO) | ONE_POLE(4096) |
| TREMOLO | 11 (VIBRATO) | ONE_POLE(4096) |
| LEGATO | 12 (LEGATO) | NONE |
| PORTAMENTO | 12 (LEGATO) | NBLOCK(8) |

### 6.4 Apply Semantics (MUST)

When applying a CapsuleMacro:

1. **Register SystemMacro** in MacroRuntime (for controller depth modulation)
2. **Configure ModulationState** in VoiceEngine (LFO rate, delay, ramp)
3. **DO NOT** reset voice allocation or program selector
4. **DO NOT** modify SystemMacros 0-10 (CMMM control axes)

CapsuleMacro is **additive** - it configures modulation on top of the current instrument.

### 6.5 Stacking Rules

* Same MacroType **replaces** previous (new vibrato replaces old vibrato)
* Different MacroTypes are **cumulative** (can have vibrato + legato)

### 6.6 Controller Modulation

After apply, controllers can modulate depth via SystemMacro:

```python
macro_runtime.set_system_macro(SystemMacro.VIBRATO, depth_q15)
```

The modulation engine reads this value each block:

```python
effective_depth = (base_depth * macro_runtime.get_system_macro(SystemMacro.VIBRATO)) >> 15
```

---

## 7. Determinism Requirements (v0)

* `voice_params.bin` and `control_defaults.bin` MUST be applied in canonical sorted order (as per their standards)
* Seeds MUST be derivable from a declared seed policy (master seed + derivation rule id)
* After apply, two runs with identical incoming VoiceFrames MUST be bit-identical (assuming same engine build)

---

## 8. Versioning and Identity Rules (Frozen v0)

* `capsule_id` identifies "the instrument identity" across versions
* A new compilation that changes identity MUST mint a new `capsule_id`
* A new compilation that only changes tuning defaults SHOULD bump the library version but keep `capsule_id`

**Library version tuple (recommended):** `(major, minor, patch)` in library metadata; not required inside the manifest for v0.

---

## 9. Conformance Checklist (IPLC v0)

A system conforms iff:

* [ ] Implements lifecycle states + allowed transitions (§2)
* [ ] Implements capsule bundle layout + required manifest fields (§3)
* [ ] Implements compiler outputs with ABI compat declared (§4)
* [ ] Implements runtime apply semantics exactly (§5)
* [ ] Enforces preset vs snapshot boundary (§5.3)
* [ ] Implements CapsuleMacro additive apply semantics (§6)
* [ ] Deterministic application order + seed policy enforcement (§7)

---
