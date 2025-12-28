# WSE-STD-PRESET v0 — Preset Format + Compile Semantics

## 0. Scope

This standard defines:

1. A **portable preset format** (text, versioned) that can be stored as an L5 “instrument”
2. A **deterministic compiler** from preset → **VoiceFrame latch defaults + per-block deltas**
3. A **conformance test contract** (golden traces + acceptance tests)

This standard does **not** define:

* VoiceFrame ABI encoding (see **WSE-STD-020 v1**)
* VM execution semantics / lane order (see **WSE-STD-VM-HYBRID v0**)
* Operator math internals (operator library is assumed stable within a version)

---

## 1. Normative goals

A conforming preset system MUST be:

* **Deterministic:** same preset + same note/macro stream → same VoiceFrame stream
* **Bit-exact friendly:** no float dependencies required to compile or run (fixed-point preferred)
* **Version-safe:** presets declare the VoiceFrame + VM standard versions they target
* **Portable:** same preset loads in Python and FPGA with identical meaning

---

## 2. Core concept

A preset is **not** “a patch graph.” It is a **compiler target** that produces:

1. **Latch defaults** for each voice (persistent VoiceFrame fields)
2. A set of **macro mappings** (controls → VoiceFrame deltas)
3. Optional **time-curves** that generate block-rate deltas (e.g., beta ramps)

The runtime executes by:

* Allocating a voice
* Loading preset latch defaults into that voice’s **VoiceFrame latch**
* On each block, applying **deltas** (from macros + note lifecycle) to the latch
* Emitting a VoiceFrame (absolute or delta) into the VM executor

---

## 3. File format

### 3.1 Preset container

A preset is stored as UTF-8 text in either:

* YAML (recommended for humans), or
* JSON (recommended for tooling)

The logical schema below is canonical (field names + meanings).

### 3.2 Required header

```yaml
preset:
  preset_id: "BRASS_STAB_V0"
  preset_version: "v0"
  author: "..."
  created_utc: "2025-12-27T00:00:00Z"
  description: "Classic punchy synth brass stab (no filter)."

compat:
  voiceframe_std: "WSE-STD-020 v1"
  vm_std: "WSE-STD-VM-HYBRID v0"
  operator_lib_id: "WSE-OPLIB-24 v0"   # must match your op-name→id table version
  sine_core_lut_id: "1024-Q15-12bit-interp"
```

**Normative:** A loader MUST reject a preset if `voiceframe_std` or `vm_std` is unsupported.

---

## 4. Preset schema (v0)

### 4.1 Latch defaults (VoiceFrame base state)

These values initialize the per-voice latch on allocation.

```yaml
latch_defaults:
  regime_id: 2
  gear_q15: 32767

  tree_id_a: 0
  tree_id_b: 0
  tree_blend_q15: 0

  gain_q15: 32767
  traction_q15: 13107

  pattern_id: 0          # canonical mask rung (K ladder)
  tail_mode: "FULL"      # FULL|MID|MIN|EMERGENCY

  carrier:
    model: "SAW"         # SINE|SAW|PULSE|TRI|PARTIALS|NOISE
    role: "AUDIBLE"      # OFF|AUDIBLE|EXCITER_ONLY
    transition: "RETRIGGER"  # RETRIGGER|GLIDE|BRIDGE
    detune_cents: 0
    pw_q15: 16384
    pitchbend_cents: 0
```

**Normative mapping:** These fields compile to the corresponding VoiceFrame v1 carrier header + base fields (WSE-STD-020 v1).

### 4.2 Backbone (ordered operator chain)

```yaml
backbone:
  - op: "IDENTITY"
    strength_q15: 32767
    param0_q15: 0
    param1_q15: 0

  - op: "TILT"
    strength_q15: 14746
    param0_q15: 0
    param1_q15: 0

  - op: "INTRA_MIX"
    strength_q15: 8192
    param0_q15: 0
    param1_q15: 0

  - op: "DIFFUSE_DOWN"
    strength_q15: 6554
    param0_q15: 0
    param1_q15: 0
```

**Normative:**

* Operator order is semantic and MUST be preserved.
* `op` resolves via the operator name table of the declared `operator_lib_id`.
* `strength_q15`, `param0_q15`, `param1_q15` are Q15 unless operator defines otherwise.

### 4.3 Tail policy

Tail mode is a *compiler decision knob* and a *runtime degradable* knob.

```yaml
tail:
  default_mode: "FULL"   # FULL|MID|MIN|EMERGENCY
```

**Normative:** Tail expansion MUST match VM definition:

* FULL = [FineClamp(2), NormalizeScale(11), SheafProject(23)]
* MID  = [2, 11]
* MIN  = [2]
* EMERGENCY = [2, 19]  (FineClamp + EmergencySimplify)

### 4.4 Events (optional per preset)

Presets may define default lane mappings and/or event patterns.

```yaml
events:
  lane_defaults:
    0: {op: "IMPULSE"}
    2: {op: "NOISE_BAND"}   # advances PRNG
    4: {op: "GATE_MASK"}
    7: {op: "CUSP_INJECTION"}
```

**Normative:** Lane mappings MUST match VM’s EVENT_LANE_TO_OP for the targeted vm_std, unless the VM explicitly allows overrides.

---

## 5. Macros (controls) and ranges

### 5.1 Macro declarations

Macros are named controls in normalized Q15 space unless declared otherwise.

```yaml
macros:
  - id: "IMPACT"
    default_q15: 27853     # 0.85
    min_q15: 0
    max_q15: 32767

  - id: "BRIGHT"
    default_q15: 24575     # 0.75
    min_q15: 0
    max_q15: 32767

  - id: "BODY"
    default_q15: 21299     # 0.65
    min_q15: 0
    max_q15: 32767

  - id: "BITE"
    default_q15: 18022     # 0.55
    min_q15: 0
    max_q15: 32767

  - id: "MOTION"
    default_q15: 3277      # 0.10
    min_q15: 0
    max_q15: 32767

  - id: "DIRT"
    default_q15: 3277      # 0.10
    min_q15: 0
    max_q15: 32767
```

---

## 6. Compile semantics

### 6.1 Deterministic compile function

The compiler consumes:

* preset (this document)
* note event stream (note_on, note_off, pitch, velocity)
* macro updates (macro_id → value_q15)
* block index timeline

And produces:

* latch init state
* per-block VoiceFrame deltas

### 6.2 “Presence, not truthiness” rule (critical)

**Normative:** Field completion and override MUST be based on **presence**, not boolean truthiness.

Example (required behavior):

* `gain_q15 = 0` is a valid value and MUST NOT be replaced by default.
* Same rule applies to any Q15 field (strengths, beta, event strengths, params).

### 6.3 Envelope ownership

**Normative:** ADSR is owned by Voice Engine / CMMM.

* Preset defines envelope specs (if present)
* Voice Engine emits block-rate `gain_q15` into VoiceFrame
* Renderer/VM applies **intra-block ramping** deterministically

Preset may include envelope specs:

```yaml
env:
  amp:
    attack_ms: 2
    decay_ms: 120
    sustain_q15: 16384
    release_ms: 100
```

### 6.4 Material wet/dry (beta / eta) as a ramp

If the VM uses `beta_target_q15` (dry/wet) as a per-block control, presets may define a ramp policy.

```yaml
material:
  beta_start_q15: 4915       # 0.15
  beta_target_q15: 11468     # 0.35
  beta_ramp_blocks: 8
  beta_curve: "LINEAR"       # LINEAR only in v0
```

**Normative:** The ramp MUST be computed in integers:

For block `t` (0-index from note_on):

* if `t >= beta_ramp_blocks`: beta = beta_target_q15
* else beta = beta_start_q15 + (t * (beta_target_q15 - beta_start_q15)) / beta_ramp_blocks  (integer division rule MUST be specified by implementation and frozen; recommend floor)

### 6.5 Classic subtractive oscillator spec (carrier stack)

If the preset wants “two detuned saws,” v0 expresses it in one of two ways:

**Option A (recommended for v0):** represent it as a *carrier recipe* that compiles into your carrier/harmonic dial representation.

```yaml
carrier_recipe:
  osc_count: 2
  osc:
    - wave: "SAW"
      detune_cents: -3
      mix_q15: 16384
    - wave: "SAW"
      detune_cents: +3
      mix_q15: 16384
```

**Normative:** If the runtime cannot represent multi-osc directly, it MUST compile deterministically into:

* a single carrier configuration plus
* a deterministic harmonic/scaffold representation (implementation-defined, but stable)

### 6.6 Operator strength mapping from macros (v0)

Presets MAY define macro bindings to operator strengths and latch fields:

```yaml
bindings:
  - macro: "BRIGHT"
    target: "op_strength"
    op: "TILT"
    map: {type: "AFFINE_Q15", a_q15: 19660, b_q15: 0}   # strength = a*macro + b

  - macro: "BODY"
    target: "carrier_detune_cents"
    map: {type: "AFFINE_I16", a: 6, b: 0}  # detune = round(a*(macro/32767)) cents

  - macro: "BITE"
    target: "traction_q15"
    map: {type: "AFFINE_Q15", a_q15: 16384, b_q15: 6553}
```

**Normative (v0):**

* Only `AFFINE_*` maps are required.
* All mapping must be integer and deterministic.
* Any rounding rule MUST be frozen per implementation (recommend round-to-nearest for i16, floor for Q15 division).

---

## 7. VoiceFrame emission policy

Presets define **what changes**, not how transport works. But to ensure portability:

* On **voice allocation**, compiler MUST emit an initialization that fully defines the latch (VF_ABSOLUTE or equivalent).
* On subsequent blocks, compiler SHOULD emit VF_DELTA with only changed fields.

**Normative:** If the transport drops frames, the engine MUST remain deterministic due to latch completion rules in WSE-STD-020.

---

## 8. Test contract (Golden + Acceptance)

### 8.1 Required artifacts per preset

A conforming preset repository MUST include:

1. **Golden Trace** (ABI regression)

   * A fixed-length block sequence (e.g., 48 blocks)
   * Frozen per-block hashes and a final trace hash
   * Version header including `voiceframe_std`, `vm_std`, `operator_lib_id`, LUT version

2. **Acceptance Tests** (character tests)

   * Attack immediacy (gain ramp expectations)
   * Envelope shape invariants
   * Click-free constraints (max delta per sample or bounded ramp)
   * Harmonic/content sanity tests (implementation-defined but deterministic)

### 8.2 Stress Trace (recommended)

A shared stress trace SHOULD exercise:

* PRNG gating (NoiseBand)
* permutes
* tail ladder transitions
* K ladder transitions
* event lanes + ramps

---

## 9. Example: Brass Stab v0 (no-filter)

```yaml
preset:
  preset_id: "BRASS_STAB_V0"
  preset_version: "v0"
  author: "Jasper"
  created_utc: "2025-12-27T00:00:00Z"
  description: "Punchy brass stab. Carrier + material (no filter)."

compat:
  voiceframe_std: "WSE-STD-020 v1"
  vm_std: "WSE-STD-VM-HYBRID v0"
  operator_lib_id: "WSE-OPLIB-24 v0"
  sine_core_lut_id: "1024-Q15-12bit-interp"

latch_defaults:
  regime_id: 2
  gear_q15: 32767
  tree_id_a: 0
  tree_id_b: 0
  tree_blend_q15: 0
  traction_q15: 13107
  pattern_id: 0
  tail_mode: "FULL"
  carrier:
    model: "SAW"
    role: "AUDIBLE"
    transition: "RETRIGGER"
    detune_cents: 0
    pw_q15: 16384
    pitchbend_cents: 0

carrier_recipe:
  osc_count: 2
  osc:
    - wave: "SAW"
      detune_cents: -3
      mix_q15: 16384
    - wave: "SAW"
      detune_cents: +3
      mix_q15: 16384

env:
  amp: {attack_ms: 2, decay_ms: 120, sustain_q15: 16384, release_ms: 100}

material:
  beta_start_q15: 4915
  beta_target_q15: 11468
  beta_ramp_blocks: 8
  beta_curve: "LINEAR"

backbone:
  - {op: "IDENTITY", strength_q15: 32767, param0_q15: 0, param1_q15: 0}
  - {op: "TILT",     strength_q15: 14746, param0_q15: 0, param1_q15: 0}
  - {op: "INTRA_MIX",strength_q15: 8192,  param0_q15: 0, param1_q15: 0}
  - {op: "DIFFUSE_DOWN", strength_q15: 6554, param0_q15: 0, param1_q15: 0}

macros:
  - {id: "IMPACT", default_q15: 27853, min_q15: 0, max_q15: 32767}
  - {id: "BRIGHT", default_q15: 24575, min_q15: 0, max_q15: 32767}
  - {id: "BODY",   default_q15: 21299, min_q15: 0, max_q15: 32767}

bindings:
  - macro: "BRIGHT"
    target: "op_strength"
    op: "TILT"
    map: {type: "AFFINE_Q15", a_q15: 19660, b_q15: 0}

  - macro: "BODY"
    target: "carrier_recipe_detune"
    map: {type: "AFFINE_I16", a: 6, b: 0}
```

---

## 10. Conformance checklist (v0)

An implementation conforms to WSE-STD-PRESET v0 iff it:

* Parses the schema and validates compatibility fields
* Initializes per-voice latch deterministically from `latch_defaults`
* Resolves operator names to IDs using the declared operator library ID
* Emits VoiceFrames consistent with WSE-STD-020 v1
* Applies the **presence-not-truthiness** completion rule
* Implements deterministic integer macro mappings and ramps
* Provides golden trace artifacts for ABI regression