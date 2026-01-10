
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

v0 defines four artifact kinds:

1. `CapsuleInstrument` (ProgramPreset + optional InstrumentIR references)
2. `CapsuleSynth` (ProgramPreset + synth-focused defaults)
3. `CapsuleAsset` (references only; no control semantics)
4. `CapsuleMacro` (performance macro: vibrato, legato, tremolo, portamento)

This spec covers (1) and (2) primarily; (3) is a manifest-only bundle; (4) defines modulation behavior.

---

## 2. Capsule Object Model (Normative)

A capsule is a bundle with this logical structure:

```text
CapsuleV0:
  identity:
    capsule_id (string)
    kind (INSTRUMENT | SYNTH | ASSET | MACRO)
    name, author, tags
    engine_compat:
      control_registry_abi = 0
      voice_registry_abi   = 0
      voiceframe_abi       = 0
      sample_rate_hz
      block_size

  program:
    program_id OR instrument_ir_id  (required for INSTRUMENT/SYNTH)
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

  macro: (only for kind=MACRO)
    macro_type (VIBRATO | LEGATO | TREMOLO | PORTAMENTO)
    rate_hz, depth_pitch_cents, depth_amp_q15, depth_timbre_q15
    delay_ms, ramp_ms
    glide_time_ms, glide_curve
    preserve_phase
```

**Capsule MUST NOT contain runtime histories** (coeff_state, phase, ADSR state, PRNG state).

---

## 2.4 CapsuleMacro Section (Performance Macros)

`CapsuleMacro` defines portable performance modulation behavior that can be applied to any instrument.

### 2.4.1 MacroType (MUST)

| Type | ID | Description |
|------|----|-------------|
| VIBRATO | 0 | Pitch/amplitude/timbre LFO modulation |
| LEGATO | 1 | Smooth pitch transition (preserves phase) |
| TREMOLO | 2 | Amplitude-only LFO modulation |
| PORTAMENTO | 3 | Pitch glide between notes |

### 2.4.2 MacroProgram Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `macro_type` | MacroType | VIBRATO | Which modulation behavior |
| `rate_hz` | float | 5.5 | LFO rate in Hz (VIBRATO, TREMOLO) |
| `depth_pitch_cents` | int | 0 | Pitch modulation depth in cents |
| `depth_amp_q15` | int | 0 | Amplitude modulation depth (Q15) |
| `depth_timbre_q15` | int | 0 | Timbre modulation depth (Q15) |
| `delay_ms` | float | 0.0 | Delay before modulation onset |
| `ramp_ms` | float | 0.0 | Ramp time to full depth |
| `glide_time_ms` | float | 0.0 | Pitch glide duration (LEGATO, PORTAMENTO) |
| `glide_curve` | int | 0 | 0=linear, 1=exp, 2=log |
| `preserve_phase` | bool | false | Keep oscillator phase on note change (LEGATO) |

### 2.4.3 Validation Rules

* `CapsuleMacro` MUST have `kind = MACRO`
* `CapsuleMacro` MUST have `macro` field set
* `CapsuleMacro` does NOT require `program_id` or `instrument_ir_id`
* `control_defaults` and `voice_param_defaults` MAY be empty

### 2.4.4 Apply Semantics

When applying a CapsuleMacro:

1. Configure the runtime modulation state (VibratoState, LegatoState, etc.)
2. Set SystemMacro values (VIBRATO=11, LEGATO=12) in MacroRuntime
3. DO NOT reset voice allocation or program selector

CapsuleMacro is **additive** — it configures modulation on top of the current instrument.

---

## 2.5 SystemMacro Control Axes (Global Controller State)

SystemMacros are **global semantic control axes** managed by the controller layer, not preset-specific.
They drive CMMM decisions and are distinct from CapsuleMacro performance modulation.

### 2.5.1 SystemMacroID (per WSE-STD-MACRO v0 §2)

| ID | Name | Scope | Owner | Semantic Meaning |
|----|------|-------|-------|------------------|
| 0 | ENERGY_STATE | per-voice | Envelope only | Envelope phase (0=silent, 32767=peak) |
| 1 | ENERGY_INTENT | per-voice | Controller → CMMM | Playing intensity / attack character |
| 2 | MOTION | per-voice | Controller → CMMM | Movement / modulation amount |
| 3 | FOCUS | per-voice | Controller → CMMM | Attention / emphasis (spectral concentration) |
| 4 | BRIGHTNESS | per-voice | Controller → CMMM | Timbral brightness (harmonic content) |
| 5 | DENSITY | per-voice | Controller → CMMM | Body / thickness (low-mid presence) |
| 6 | RISK | per-voice | Controller → CMMM | Chaos / instability (stochastic behavior) |
| 7 | AGE | per-voice | CMMM integrator | Voice age (accumulates over note duration) |
| 8 | SPACE | per-voice | Controller → CMMM | Spatial / reverb character |
| 9 | MATERIAL | per-voice | Controller → CMMM | Surface texture (wood=low, metal=mid, glass=high) |
| 10 | UNITY | per-voice | Controller → CMMM | Coherence / blend (how voices blend together) |
| 11 | VIBRATO | per-voice | Controller → Mode B | Vibrato depth modulation |
| 12 | LEGATO | per-voice | Controller → Mode B | Legato mode (0=retrigger, >0=legato) |

### 2.5.2 Ownership Rules (MUST)

* `ENERGY_STATE` (0) MUST be written ONLY by the envelope generator
* `AGE` (7) MUST evolve via CMMM integrator, never set directly by controllers
* All other macros (1-6, 8-12) are Controller → CMMM/Mode B

### 2.5.3 ControlProgram (Session-Level Defaults)

`ControlProgram` holds default values for semantic control axes at the **session/controller level**.
It is NOT part of instrument presets — presets define timbre, not global expression state.

```text
ControlProgram:
  energy_intent_q15: int  # 0.5 = neutral intensity
  motion_q15: int         # 0.0 = no modulation
  focus_q15: int          # 0.5 = balanced emphasis
  brightness_q15: int     # 0.5 = neutral brightness
  density_q15: int        # 0.5 = neutral body
  risk_q15: int           # 0.0 = stable, no chaos
  space_q15: int          # 0.25 = subtle spatial
  material_q15: int       # 0.5 = neutral texture
  unity_q15: int          # 0.5 = balanced blend
```

### 2.5.4 Apply Semantics

ControlProgram is applied at **session initialization** or **controller reset**, NOT at preset load:

```python
# At session start
runtime = MacroRuntime()
runtime.apply_control_defaults(ControlProgram())

# At preset load — DO NOT apply ControlProgram
capsule = load_preset("violin_ir_v1.yaml")
# Only apply MacroProgram if present
if capsule.macro:
    runtime.apply_macro_capsule(capsule.macro)
```

**Rule (MUST):** Presets MUST NOT override SystemMacro values (0-10). They configure modulation behavior via MacroProgram only.

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
