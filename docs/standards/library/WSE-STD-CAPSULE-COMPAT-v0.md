
# WSE-STD-CAPSULE-COMPAT v0 — Capsule Compatibility Contract (ControlRegistry + VoiceFrame + VoiceEngineRegistry)

**Status:** Normative (v0 frozen)
**Applies to:** PresetLibrary storage, compiler output, capsule load/apply pipeline
**Depends on:**

* WSE-STD-CONTROLREGISTRY v0
* WSE-STD-VOICEENGINEREGISTRY v0
* WSE-STD-020 VoiceFrame ABI

---

## 0. Purpose

Define a **portable Capsule object** that can be:

* stored in a PresetLibrary
* compiled-to by tools (instead of ad-hoc JSON)
* loaded by the runtime pipeline
* applied without ambiguity across engines

**Key design choice (per your direction):**

* The **Library stores capsules**
* The **pipeline owns morphing**
* Capsules include **control/macro defaults** and **engine defaults**, not runtime histories

---

## 1. First-class Capsule Types (v0)

v0 defines three artifact kinds:

1. `CapsuleInstrument` (ProgramPreset + optional InstrumentIR references)
2. `CapsuleSynth` (ProgramPreset + synth-focused defaults)
3. `CapsuleAsset` (references only; no control semantics)

This spec covers (1) and (2) primarily; (3) is a manifest-only bundle.

---

## 2. Capsule Object Model (Normative)

A capsule is a bundle with this logical structure:

```text
CapsuleV0:
  identity:
    capsule_id (string)
    kind (INSTRUMENT | SYNTH | ASSET)
    name, author, tags
    engine_compat:
      control_registry_abi = 0
      voice_registry_abi   = 0
      voiceframe_abi       = 0
      sample_rate_hz
      block_size

  program:
    program_id OR instrument_ir_id
    optional regime_graph_id
    optional tree_family_id

  defaults:
    control_defaults:    overrides for ControlRegistry defaults (cockpit-facing)
    voice_param_defaults: overrides for VoiceEngineRegistry defaults (engine-facing)

  assets:
    asset_refs[] (optional):
      type (IR | WAVETABLE | MATERIAL | TABLE_ATLAS | OTHER)
      id / hash / path

  seeds:
    master_seed
    derivation_rule_id
```

**Capsule MUST NOT contain runtime histories** (coeff_state, phase, ADSR state, PRNG state).

---

## 3. Control Defaults Section (ControlRegistry Compatibility)

### 3.1 Legal content (MUST)

`control_defaults` may override only **default values** of controls defined in ControlRegistry.

It MUST NOT modify:

* control ranges
* value types
* slew policy / slew param
* binding targets

**If a control_id is unknown** to the current ControlRegistry ABI:

* loader MUST ignore it deterministically (or fail only if `strict_compat=true`)

### 3.2 Packing (Recommended)

Use `control_defaults.bin`:

```
u16 count
repeat count times:
  u8  value_type
  u8  control_id
  i16 value
```

Sorted by `control_id asc`.

---

## 4. Voice Param Defaults Section (VoiceEngineRegistry Compatibility)

Same rules as ProgramPreset overrides:

* may override only default values
* cannot change types/ranges/persistence/update_mode/binding targets
* unknown `param_id` is ignored deterministically unless strict

Binary `voice_params.bin` uses the packing defined in VoiceEngineRegistry §6.4.

---

## 5. VoiceFrame Contract (Cockpit ↔ VoiceEngine)

This is the compatibility bridge that makes capsules meaningful in real-time.

### 5.1 VoiceFrame ABI versioning (MUST)

Every VoiceFrame MUST include:

* `voiceframe_abi_version = 0` (or implicit by transport)

### 5.2 Responsibilities (MUST)

**CockpitEngine responsibilities:**

* Maintain current values for ControlIDs (registry-defined)
* Apply slew policies (integer-only)
* Emit VoiceFrame fields according to binding targets

**VoiceEngine responsibilities:**

* Treat VoiceFrame fields as authoritative cockpit intent for this block
* Apply VoiceEngineRegistry params for engine-owned targets (mode selects, ids, feature flags)
* Run planner (DEGRADE) to choose effective execution plan

### 5.3 Binding contract (MUST)

Binding targets in ControlRegistry are interpreted as:

* mapping from control values → specific VoiceFrame fields (or emitter behaviors)

Binding targets in VoiceEngineRegistry are interpreted as:

* mapping from param values → engine-owned configuration/state selection

**Rule (MUST):** A binding target string MUST have exactly one owner:

* ControlRegistry binding targets affect **VoiceFrame emission**
* VoiceEngineRegistry binding targets affect **engine internals**

No target may be simultaneously written by both registries.

---

## 6. “Capsule Apply” Algorithm (Normative)

When loading a capsule:

1. Validate `engine_compat` (abi versions, sample_rate, block_size)
2. Load program identity (program_id / IR ref) into engine program selector
3. Apply `voice_param_defaults` into VoiceEngineRegistry defaults (clamped)
4. Apply `control_defaults` into ControlRegistry defaults (clamped)
5. Reset/allocate voices deterministically (no runtime histories)
6. Begin runtime:

   * cockpit emits VoiceFrames
   * voice engine applies registry + planner
   * material/analog later in pipeline (outside capsule responsibility)

---

## 7. Compiler Targeting the Library (Answer to your question)

### 7.1 Does the compiler target the library now instead of JSON?

**Yes in v0:** the compiler’s output artifact SHOULD be a **Capsule bundle** (manifest + defaults + asset refs), because it is:

* versioned
* ABI compatible
* clampable and deterministic
* directly loadable by runtime without ad-hoc schema drift

### 7.2 Where is the compiler needed?

The compiler is needed when converting:

* InstrumentIR / authoring graphs / “instrument flow” products
  into:
* `CapsuleInstrument` / `CapsuleSynth` artifacts with frozen defaults + asset refs

The compiler is **not** required for runtime loading: runtime only needs the capsule codec + registries.

---

## 8. Conformance Checklist (Capsule Compat v0)

* [ ] Capsule bundle includes compat ABI versions for ControlRegistry, VoiceEngineRegistry, VoiceFrame
* [ ] Capsule defaults override only default values (never ranges/types/slew/bindings)
* [ ] Unknown IDs are ignored deterministically unless strict mode
* [ ] Capsule apply algorithm implemented exactly (§6)
* [ ] Clear ownership: cockpit writes VoiceFrame; voice engine reads VoiceFrame and applies engine params (§5.3)

---

If you want, I can also emit:

* a concrete `CapsuleV0` **binary container** (single `.wsecap` file: chunked like snapshots),
* a minimal `CapsuleLoader.apply_capsule()` reference implementation outline (Python + FPGA-friendly structs),
* and the **VoiceFrame v0 field list** that matches your current ControlRegistry binding targets (so it’s fully pinned, not implied).
