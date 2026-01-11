# WSE-STD-PRESET v0 — Preset Format + Compile Semantics

**Standard ID:** WSE-STD-PRESET v0
**Title:** Preset Format, Compile Semantics, and Canonical Examples
**Status:** Normative (v0 frozen) — **DEPRECATION NOTICE BELOW**
**Applies to:** PresetCompiler + VoiceFrameEmitter + PresetLibrary
**Depends on:**

* WSE-STD-020 v1 (VoiceFrame ABI)
* WSE-STD-VM-HYBRID v0 (VM execution semantics)
* WSE-STD-IPLC v0 (Capsule lifecycle)
* WSE-STD-MACRO v0 (Macro ABI)

---

## IMPORTANT: Format Migration Notice

The `preset:` format described in this document is **deprecated** in favor of the newer canonical formats:

| Preset Type | Legacy Format | Canonical Format |
|-------------|---------------|------------------|
| Instruments | `preset:` | `instrument_ir_v0:` |
| Macros | `preset:` + `kind: MACRO` | `macro_ir_v0:` |

**For new presets, use:**
- **`instrument_ir_v0:`** for instrument presets (see [PRESET-AUTHORING-GUIDE.md](../../library/PRESET-AUTHORING-GUIDE.md))
- **`macro_ir_v0:`** for performance macros (vibrato, legato, tremolo, portamento)

The canonical examples in Appendix A use the legacy format and are preserved for historical reference.

---

## 0. Scope

This standard defines:

1. A **portable preset format** (text, versioned) that can be stored as an L5 "instrument"
2. A **deterministic compiler** from preset → **VoiceFrame latch defaults + per-block deltas**
3. A **conformance test contract** (golden traces + acceptance tests)
4. **Canonical preset examples** (12 classic voice presets — legacy format)

This standard does **not** define:

* VoiceFrame ABI encoding (see **WSE-STD-020 v1**)
* VM execution semantics / lane order (see **WSE-STD-VM-HYBRID v0**)
* Operator math internals (operator library is assumed stable within a version)

---

## 1. Normative Goals

A conforming preset system MUST be:

* **Deterministic:** same preset + same note/macro stream → same VoiceFrame stream
* **Bit-exact friendly:** no float dependencies required to compile or run (fixed-point preferred)
* **Version-safe:** presets declare the VoiceFrame + VM standard versions they target
* **Portable:** same preset loads in Python and FPGA with identical meaning

---

## 2. Core Concept

A preset is **not** "a patch graph." It is a **compiler target** that produces:

1. **Latch defaults** for each voice (persistent VoiceFrame fields)
2. A set of **macro mappings** (controls → VoiceFrame deltas)
3. Optional **time-curves** that generate block-rate deltas (e.g., beta ramps)

The runtime executes by:

* Allocating a voice
* Loading preset latch defaults into that voice's **VoiceFrame latch**
* On each block, applying **deltas** (from macros + note lifecycle) to the latch
* Emitting a VoiceFrame (absolute or delta) into the VM executor

---

## 3. File Format

### 3.1 Preset Container

A preset is stored as UTF-8 text in either:

* YAML (recommended for humans), or
* JSON (recommended for tooling)

The logical schema below is canonical (field names + meanings).

### 3.2 Required Header

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

### 3.3 Operator Aliases (Optional)

Presets may define operator aliases for forward compatibility:

```yaml
op_aliases:
  compress_k:
    preferred_op_id: null          # set when you add a real COMPRESS_K
    fallback_op_id: 20             # foldpressure (current library)
    note: "Legacy reso/bite proxy used in Classic Presets docs"
```

**Normative:** If `preferred_op_id` is null, compiler MUST use `fallback_op_id`.

---

## 4. Preset Schema (v0)

### 4.1 Latch Defaults (VoiceFrame Base State)

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

### 4.2 Backbone (Ordered Operator Chain)

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

### 4.3 Tail Policy

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

### 4.4 Events (Optional Per Preset)

Presets may define default lane mappings and/or event patterns.

```yaml
events:
  lane_defaults:
    0: {op: "IMPULSE"}
    2: {op: "NOISE_BAND"}   # advances PRNG
    4: {op: "GATE_MASK"}
    7: {op: "CUSP_INJECTION"}
```

**Normative:** Lane mappings MUST match VM's EVENT_LANE_TO_OP for the targeted vm_std, unless the VM explicitly allows overrides.

### 4.5 Event Triggers

Events may specify trigger conditions:

```yaml
events:
  - trigger: "note_on"      # Fire for first N blocks after note-on
    lane: 0
    op: "impulse"
    op_id: 13
    active_blocks: 1
    strength_q15: 8192

  - trigger: "periodic"     # Fire when (block_index % period == phase)
    lane: 4
    op: "rotatenodes"
    op_id: 10
    period_blocks: 8
    phase_blocks: 0
    strength_q15: 6553

  - trigger: "early_decay"  # Fire only during ATTACK/DECAY stages
    lane: 1
    op: "globalpulse"
    op_id: 8
    period_blocks: 12
    max_blocks: 60
    strength_q15: 6553
```

**Event Trigger Semantics:**

| Trigger | Fires When |
|---------|------------|
| `note_on` | First N blocks after note-on (`active_blocks`) |
| `periodic` | `(block_index % period_blocks) == phase_blocks` |
| `early_decay` | Envelope stage in {ATTACK, DECAY} and within `max_blocks` |

**Implementation Status:** Event parsing is implemented; trigger dispatch is pending.

---

## 5. Macros (Controls) and Ranges

### 5.1 Macro Declarations

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

## 6. Compile Semantics

### 6.1 Deterministic Compile Function

The compiler consumes:

* preset (this document)
* note event stream (note_on, note_off, pitch, velocity)
* macro updates (macro_id → value_q15)
* block index timeline

And produces:

* latch init state
* per-block VoiceFrame deltas

### 6.2 "Presence, Not Truthiness" Rule (Critical)

**Normative:** Field completion and override MUST be based on **presence**, not boolean truthiness.

Example (required behavior):

* `gain_q15 = 0` is a valid value and MUST NOT be replaced by default.
* Same rule applies to any Q15 field (strengths, beta, event strengths, params).

### 6.3 Envelope Ownership

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

### 6.4 Material Wet/Dry (Beta / Eta) as a Ramp

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

### 6.5 Classic Subtractive Oscillator Spec (Carrier Stack)

If the preset wants "two detuned saws," v0 expresses it in one of two ways:

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

### 6.6 Operator Strength Mapping from Macros (v0)

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

**Implementation Status:** Bindings are parsed but AFFINE maps are not yet applied at runtime.

### 6.7 LFO Modulation (v0)

Presets may define LFOs that modulate operator strengths:

```yaml
lfos:
  - lfo_id: "lfo0"
    shape: "sine_q15"
    period_blocks: 96      # ~0.128s at 750 blocks/s
    phase_q15: 0

backbone:
  - op: "tilt"
    op_id: 1
    strength_q15: 6553
    mod: { source: "lfo0", add_q15: 11469 }  # strength oscillates
```

**Implementation Status:** LFO modulation is not yet implemented.

---

## 7. VoiceFrame Emission Policy

Presets define **what changes**, not how transport works. But to ensure portability:

* On **voice allocation**, compiler MUST emit an initialization that fully defines the latch (VF_ABSOLUTE or equivalent).
* On subsequent blocks, compiler SHOULD emit VF_DELTA with only changed fields.

**Normative:** If the transport drops frames, the engine MUST remain deterministic due to latch completion rules in WSE-STD-020.

---

## 8. Test Contract (Golden + Acceptance)

### 8.1 Required Artifacts Per Preset

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

### 8.2 Stress Trace (Recommended)

A shared stress trace SHOULD exercise:

* PRNG gating (NoiseBand)
* permutes
* tail ladder transitions
* K ladder transitions
* event lanes + ramps

**Implementation Status:** Golden trace infrastructure is pending.

---

## 9. Conformance Checklist (v0)

An implementation conforms to WSE-STD-PRESET v0 iff it:

* [ ] Parses the schema and validates compatibility fields
* [ ] Initializes per-voice latch deterministically from `latch_defaults`
* [ ] Resolves operator names to IDs using the declared operator library ID
* [ ] Emits VoiceFrames consistent with WSE-STD-020 v1
* [ ] Applies the **presence-not-truthiness** completion rule
* [ ] Implements deterministic integer macro mappings and ramps
* [ ] Provides golden trace artifacts for ABI regression

---

## Appendix A: Canonical Preset Examples (12 Classic Voices)

These 12 presets form the "Classic Voices" basis set for testing and demonstration.

**Design choice:** VO presets are "compiler-friendly, human-readable patch sources" that compile into **VoiceFrameEmitter rules** (block-rate frames + event_mask logic).

### A.1 Saw Lead (Supersaw / Classic Lead)

```yaml
preset_id: "classic.saw_lead.v0"
name: "Saw Lead (Supersaw / Classic Lead)"
carrier:
  wave: "saw"
  partials_N: 12
defaults:
  pattern_id: 0        # K=32
  tail_mode: "FULL"
  gain_q15: 30000
  pan_q15: 16384
  env:
    attack_blocks: 2
    decay_blocks: 40
    sustain_level_q15: 25000
    release_blocks: 60
backbone:
  - { op: "tilt",         op_id: 1,  strength_q15: 18022 }  # 0.55
  - { op: "compress_k",   fallback_op_id: 20, strength_q15: 6553 }   # 0.20
  - { op: "lowrankmorph", op_id: 7,  strength_q15: 14745 }  # 0.45
  - { op: "quasirotate",  op_id: 17, strength_q15: 8192 }   # 0.25
  - { op: "diffuseup",    op_id: 4,  strength_q15: 3277 }   # 0.10
  - { op: "normalizescale", op_id: 11, strength_q15: 26214 }# 0.80
events:
  - trigger: "periodic"
    lane: 4
    op: "rotatenodes"
    op_id: 10
    period_blocks: 8
    phase_blocks: 0
    strength_q15: 6553   # 0.20
  - trigger: "periodic"
    lane: 1
    op: "globalpulse"
    op_id: 8
    period_blocks: 32
    phase_blocks: 0
    strength_q15: 4915   # 0.15
```

### A.2 Square Bass (Subtractive Bass)

```yaml
preset_id: "classic.square_bass.v0"
name: "Square Bass (Subtractive Bass)"
carrier:
  wave: "square"
  partials_N: 8
defaults:
  pattern_id: 2          # K=16 (often enough for bass)
  tail_mode: "FULL"
  gain_q15: 29000
  env: { attack_blocks: 2, decay_blocks: 25, sustain_level_q15: 28000, release_blocks: 30 }
backbone:
  - { op: "tilt",           op_id: 1,  strength_q15: 8192 }   # 0.25
  - { op: "diffusedown",    op_id: 5,  strength_q15: 8192 }   # 0.25
  - { op: "compress_k",     fallback_op_id: 20, strength_q15: 11469 } # 0.35
  - { op: "sheafproject",   op_id: 23, strength_q15: 8192 }   # 0.25
  - { op: "normalizescale", op_id: 11, strength_q15: 26214 }  # 0.80
events:
  - trigger: "periodic"
    lane: 0
    op: "impulse"
    op_id: 13
    period_blocks: 16
    phase_blocks: 0
    strength_q15: 8192   # 0.25
    enabled: false       # optional accent
```

### A.3 Sync Lead (Hard Sync Archetype)

```yaml
preset_id: "classic.sync_lead.v0"
name: "Sync Lead (Hard Sync Archetype)"
carrier:
  wave: "saw"
  partials_N: 12
defaults:
  pattern_id: 0
  tail_mode: "FULL"
  gain_q15: 30000
  env: { attack_blocks: 1, decay_blocks: 60, sustain_level_q15: 22000, release_blocks: 50 }
backbone:
  - { op: "bandshear",      op_id: 6,  strength_q15: 14745 }  # 0.45
  - { op: "rotatenodes",    op_id: 10, strength_q15: 9830 }   # 0.30
  - { op: "compress_k",     fallback_op_id: 20, strength_q15: 9830 }  # 0.30
  - { op: "tilt",           op_id: 1,  strength_q15: 18022 }  # 0.55
  - { op: "normalizescale", op_id: 11, strength_q15: 24575 }  # 0.75
events:
  - trigger: "periodic"
    lane: 3
    op: "swapbands"
    op_id: 9
    period_blocks: 8
    phase_blocks: 0
    strength_q15: 11469   # 0.35
  - trigger: "periodic"
    lane: 1
    op: "globalpulse"
    op_id: 8
    period_blocks: 16
    phase_blocks: 0
    strength_q15: 6553    # 0.20
```

### A.4 Brass Stab (Poly Subtractive Brass)

```yaml
preset_id: "classic.brass_stab.v0"
name: "Brass Stab (Poly Subtractive Brass)"
carrier:
  wave: "saw"
  partials_N: 10
defaults:
  pattern_id: 0
  tail_mode: "FULL"
  gain_q15: 31000
  env: { attack_blocks: 2, decay_blocks: 90, sustain_level_q15: 16000, release_blocks: 75 }
backbone:
  - { op: "sheafproject",   op_id: 23, strength_q15: 13107 }  # 0.40
  - { op: "tilt",           op_id: 1,  strength_q15: 14745 }  # 0.45
  - { op: "compress_k",     fallback_op_id: 20, strength_q15: 9830 }  # 0.30
  - { op: "intramix",       op_id: 3,  strength_q15: 8192 }   # 0.25
  - { op: "normalizescale", op_id: 11, strength_q15: 26214 }  # 0.80
events:
  - trigger: "note_on"
    lane: 1
    op: "globalpulse"
    op_id: 8
    active_blocks: 2
    strength_q15: 11469   # 0.35
  - trigger: "note_on"
    lane: 0
    op: "impulse"
    op_id: 13
    active_blocks: 1
    strength_q15: 8192    # 0.25
notes:
  - "Classic recipe: fast attack + bark via operators; amp/brightness env lives in VoiceEngine."
```

### A.5 String Pad (Analog Ensemble Pad)

```yaml
preset_id: "classic.string_pad.v0"
name: "String Pad (Analog Ensemble Pad)"
carrier:
  wave: "saw"
  partials_N: 12
defaults:
  pattern_id: 0
  tail_mode: "FULL"
  gain_q15: 27000
  env: { attack_blocks: 120, decay_blocks: 220, sustain_level_q15: 28000, release_blocks: 240 }
backbone:
  - { op: "lowrankmorph",   op_id: 7,  strength_q15: 18022 }  # 0.55
  - { op: "quasirotate",    op_id: 17, strength_q15: 11469 }  # 0.35
  - { op: "diffuseup",      op_id: 4,  strength_q15: 4915 }   # 0.15
  - { op: "sheafproject",   op_id: 23, strength_q15: 6553 }   # 0.20
  - { op: "tilt",           op_id: 1,  strength_q15: 13107 }  # 0.40
  - { op: "normalizescale", op_id: 11, strength_q15: 26214 }  # 0.80
events:
  - trigger: "periodic"
    lane: 4
    op: "rotatenodes"
    op_id: 10
    period_blocks: 16
    phase_blocks: 0
    strength_q15: 6553   # 0.20
  - trigger: "periodic"
    lane: 7
    op: "gluepair"
    op_id: 12
    period_blocks: 64
    phase_blocks: 0
    strength_q15: 4915   # 0.15
```

### A.6 FM Bell (2-3 Operator FM)

```yaml
preset_id: "classic.fm_bell.v0"
name: "FM Bell (2-3 Operator FM)"
carrier:
  wave: "sine"
  partials_N: 3
defaults:
  pattern_id: 2
  tail_mode: "FULL"
  gain_q15: 28000
  env: { attack_blocks: 1, decay_blocks: 140, sustain_level_q15: 0, release_blocks: 120 }
backbone:
  - { op: "sheafproject",   op_id: 23, strength_q15: 18022 }  # 0.55
  - { op: "bandshear",      op_id: 6,  strength_q15: 11469 }  # 0.35
  - { op: "quasirotate",    op_id: 17, strength_q15: 8192 }   # 0.25
  - { op: "tilt",           op_id: 1,  strength_q15: 16384 }  # 0.50
  - { op: "compress_k",     fallback_op_id: 20, strength_q15: 8192 }  # 0.25
  - { op: "normalizescale", op_id: 11, strength_q15: 24575 }  # 0.75
events:
  - trigger: "note_on"
    lane: 0
    op: "impulse"
    op_id: 13
    active_blocks: 1
    strength_q15: 14745  # 0.45
  - trigger: "early_decay"
    lane: 1
    op: "globalpulse"
    op_id: 8
    period_blocks: 12
    max_blocks: 60
    strength_q15: 6553   # 0.20
```

### A.7 Jump Brass (Bright Unison Brass)

```yaml
preset_id: "classic.jump_brass.v0"
name: "Jump Brass (Bright Unison Brass)"
carrier:
  wave: "saw"
  partials_N: 12
defaults:
  pattern_id: 0
  tail_mode: "FULL"
  gain_q15: 31500
  env: { attack_blocks: 2, decay_blocks: 120, sustain_level_q15: 14000, release_blocks: 90 }
backbone:
  - { op: "lowrankmorph",   op_id: 7,  strength_q15: 14745 }  # 0.45
  - { op: "sheafproject",   op_id: 23, strength_q15: 11469 }  # 0.35
  - { op: "tilt",           op_id: 1,  strength_q15: 19660 }  # 0.60
  - { op: "compress_k",     fallback_op_id: 20, strength_q15: 11469 } # 0.35
  - { op: "intramix",       op_id: 3,  strength_q15: 6553 }   # 0.20
  - { op: "normalizescale", op_id: 11, strength_q15: 26214 }  # 0.80
events:
  - trigger: "note_on"
    lane: 1
    op: "globalpulse"
    op_id: 8
    active_blocks: 2
    strength_q15: 9830   # 0.30
  - trigger: "periodic"
    lane: 4
    op: "rotatenodes"
    op_id: 10
    period_blocks: 16
    phase_blocks: 0
    strength_q15: 4915   # 0.15
```

### A.8 Sub Bass (Pure Fundamental)

```yaml
preset_id: "classic.sub_bass.v0"
name: "Sub Bass (Pure Fundamental)"
carrier:
  wave: "sine"
  partials_N: 1
defaults:
  pattern_id: 6          # K=1 is fine here
  tail_mode: "FULL"
  gain_q15: 30000
  env: { attack_blocks: 3, decay_blocks: 30, sustain_level_q15: 31000, release_blocks: 60 }
backbone:
  - { op: "sheafproject",   op_id: 23, strength_q15: 24575 }  # 0.75
  - { op: "diffusedown",    op_id: 5,  strength_q15: 11469 }  # 0.35
  - { op: "tilt",           op_id: 1,  strength_q15: 3277 }   # 0.10
  - { op: "normalizescale", op_id: 11, strength_q15: 27852 }  # 0.85
events:
  - trigger: "note_on"
    lane: 0
    op: "impulse"
    op_id: 13
    active_blocks: 1
    strength_q15: 3277   # 0.10
    enabled: false
```

### A.9 PWM Pad (Pulse Modulation Pad)

```yaml
preset_id: "classic.pwm_pad.v0"
name: "PWM Pad (Pulse Modulation Pad)"
carrier:
  wave: "pulse"
  partials_N: 10
defaults:
  pattern_id: 0
  tail_mode: "FULL"
  gain_q15: 27000
  env: { attack_blocks: 80, decay_blocks: 200, sustain_level_q15: 28000, release_blocks: 240 }
backbone:
  - { op: "quasirotate",    op_id: 17, strength_q15: 14745 }  # 0.45
  - { op: "lowrankmorph",   op_id: 7,  strength_q15: 11469 }  # 0.35
  - { op: "sheafproject",   op_id: 23, strength_q15: 8192 }   # 0.25
  - { op: "tilt",           op_id: 1,  strength_q15: 13107 }  # 0.40
  - { op: "normalizescale", op_id: 11, strength_q15: 26214 }  # 0.80
events:
  - trigger: "periodic"
    lane: 4
    op: "rotatenodes"
    op_id: 10
    period_blocks: 8
    phase_blocks: 0
    strength_q15: 5898   # 0.18
  - trigger: "periodic"
    lane: 5
    op: "bandshear"
    op_id: 6
    period_blocks: 16
    phase_blocks: 0
    strength_q15: 3277   # 0.10
    enabled: false
```

### A.10 Pluck (Resonant Decay Pluck)

```yaml
preset_id: "classic.pluck.v0"
name: "Pluck (Resonant Decay Pluck)"
carrier:
  wave: "saw"
  partials_N: 8
defaults:
  pattern_id: 2
  tail_mode: "FULL"
  gain_q15: 30000
  env: { attack_blocks: 1, decay_blocks: 120, sustain_level_q15: 0, release_blocks: 90 }
backbone:
  - { op: "compress_k",     fallback_op_id: 20, strength_q15: 18022 } # 0.55
  - { op: "tilt",           op_id: 1,  strength_q15: 18022 }  # 0.55
  - { op: "sheafproject",   op_id: 23, strength_q15: 11469 }  # 0.35
  - { op: "normalizescale", op_id: 11, strength_q15: 24575 }  # 0.75
events:
  - trigger: "note_on"
    lane: 0
    op: "impulse"
    op_id: 13
    active_blocks: 1
    strength_q15: 18022  # 0.55
  - trigger: "note_on"
    lane: 2
    op: "gatemask"
    op_id: 16
    active_blocks: 16
    strength_q15: 8192   # 0.25
```

### A.11 Wobble Bass (LFO-Driven Filter Bass)

```yaml
preset_id: "classic.wobble_bass.v0"
name: "Wobble Bass (LFO-Driven Filter Bass)"
carrier:
  wave: "square"
  partials_N: 6
defaults:
  pattern_id: 2
  tail_mode: "FULL"
  gain_q15: 31000
  env: { attack_blocks: 2, decay_blocks: 60, sustain_level_q15: 30000, release_blocks: 70 }
backbone:
  - { op: "diffusedown",    op_id: 5,  strength_q15: 8192 }   # 0.25
  - { op: "compress_k",     fallback_op_id: 20, strength_q15: 11469 } # 0.35
  - { op: "sheafproject",   op_id: 23, strength_q15: 8192 }   # 0.25
  - { op: "normalizescale", op_id: 11, strength_q15: 26214 }  # 0.80
  # wobble: tilt is modulated (compiler generates per-block strength update)
  - { op: "tilt", op_id: 1, strength_q15: 6553, mod: { source: "lfo0", add_q15: 11469 } }
lfos:
  - lfo_id: "lfo0"
    shape: "sine_q15"
    period_blocks: 96      # ~0.128s at 750 blocks/s
    phase_q15: 0
events:
  - trigger: "periodic"
    lane: 1
    op: "globalpulse"
    op_id: 8
    period_blocks: 16
    phase_blocks: 0
    strength_q15: 4915   # 0.15
    enabled: false
```

### A.12 Theremin Lead (Continuous Pitch Lead)

```yaml
preset_id: "classic.theremin_lead.v0"
name: "Theremin Lead (Continuous Pitch Lead)"
carrier:
  wave: "sine"
  partials_N: 2
defaults:
  pattern_id: 6
  tail_mode: "FULL"
  gain_q15: 26000
  env: { attack_blocks: 40, decay_blocks: 0, sustain_level_q15: 26000, release_blocks: 80 }
  glide:
    mode: "portamento"
    glide_blocks: 60
backbone:
  - { op: "sheafproject",   op_id: 23, strength_q15: 19660 }  # 0.60
  - { op: "tilt",           op_id: 1,  strength_q15: 11469 }  # 0.35
  - { op: "quasirotate",    op_id: 17, strength_q15: 3277 }   # 0.10
  - { op: "normalizescale", op_id: 11, strength_q15: 27852 }  # 0.85
events: []
```

---

## Appendix B: Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Preset parsing (YAML/JSON) | Implemented | Full schema support |
| Latch defaults | Implemented | VoiceFrameEmitter |
| Backbone operators | Implemented | 24-op library |
| Envelope specs | Implemented | VoiceEngine ADSR |
| Beta ramp | Implemented | Linear ramp |
| Event lane parsing | Implemented | Schema validated |
| Event trigger dispatch | Pending | Parsed but not dispatched |
| Macro bindings (AFFINE) | Pending | Parsed but not applied |
| LFO modulation | Pending | Schema only |
| Operator aliases | Pending | No resolution logic |
| Golden trace infrastructure | Pending | Test framework needed |
| Carrier recipe (multi-osc) | Partial | Single carrier only |

---

## Appendix C: Related Documentation

- [PRESET-AUTHORING-GUIDE.md](../../library/PRESET-AUTHORING-GUIDE.md) - InstrumentIR API guide
- [OPERATORS.md](../../library/OPERATORS.md) - Complete operator reference
- [MACROS-AND-CONTROLS.md](../../library/MACROS-AND-CONTROLS.md) - Control system details
- [WSE-STD-IPLC-v0.md](../../library/WSE-STD-IPLC-v0.md) - Capsule lifecycle
