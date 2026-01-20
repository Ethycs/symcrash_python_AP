# WSE-STD-CMMM-ROADMAP v0

**Roadmap + conformance checklists for "CMMMs that can copy extant synthesizer designs"**
Status: **v0 (planning standard)** — normative where marked **MUST/SHOULD/MAY**

**Last Updated**: 2026-01-15

---

## Implementation Status

### Voice Engine (CMMM) ✅ COMPLETE

The CMMM (Controlled Semi-Markov Mealy Machine) voice engine is fully implemented in `voice_engine.py`:

- 7 regimes (R0-R6) with hysteresis and dwell rules
- Deterministic xorshift32 RNG
- Tower levels (L0-L4) for hierarchical coefficient projection
- Degradation levels (NONE → LIGHT → HEAVY → EMERGENCY)
- GestureBasis → IntentVector → FamilyWeights pipeline
- VoiceFrame ABI output

### Phase Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0 — Substrate | ✅ Complete | ABI/VM/budgets/voice engine locked |
| Phase 1 — USS/Preset | ✅ Complete | InstrumentIR v0 + PresetLibraryV0 operational |
| Phase 2 — SUBTRACTIVE | 🟡 Partial | Carrier + Amp ADSR done; filter audio path + LFO missing |
| Phase 3 — FM/PM | 🟢 Largely Complete | FM6 engine + matrix envelopes working; formal profile spec pending |
| Phase 4+ | 🔲 Not started | |

### Integration Bindings (per CMMM-addenda)

| Binding | Status | Notes |
|---------|--------|-------|
| payload_type 0x30 (NODEMASK_SET) | ✅ Complete | ExecutorLatch + crossfade dispatch in lanes.py |
| payload_type 0x31 (TAILMODE_SET) | ✅ Complete | ExecutorLatch traction + tail mode handling |
| Ramp enforcement | ✅ Complete | Crossfade weights via `dispatch_coeff_op_crossfade()` |
| VoiceFrameEmitter CMMM fields | ✅ Complete | preset.cmmm → VoiceFrame emission |

### Related Systems

- **CCS Carrier**: Mode A + Mode B oscillators working. See `CCS-STATUS.md`.
- **Budgets**: `regime_budgets.py`, `tail_modes.py`, `degradation_ladder.py` ✅
- **Degradation Ladder**: Full deterministic degradation (events→permutes→lowrank→K→tail→emergency) ✅
- **VoiceFrame Emitter**: `voiceframe_emitter.py` - Preset→VoiceFrame with envelope/beta + CMMM fields ✅
- **VM Executor**: Polyphase IDWPT synthesis operational ✅
- **ExecutorLatch**: `vm/state.py` - CMMM event state with crossfade weights ✅
- **Crossfade Dispatch**: `vm/lanes.py` - `dispatch_coeff_op_crossfade()` for mask transitions ✅
- **InstrumentIR**: `instrument_ir.py` - Complete L0-L3 layered tower IR ✅
- **FM6 Engine**: `fm6_engine.py` - 32 DX7 algorithms + feedback ✅
- **Matrix Envelopes**: `envelope_targets.py` - Per-operator envelope routing ✅
- **Preset Library**: `library/preset_library.py` - YAML presets + compilation ✅
- **SynthMaster**: `synth_master.py` - High-level synthesis pipeline ✅
- **Regime Registry**: `library/regime_registry.py` - Runtime regime graph storage ✅
- **Tree Registry**: `library/tree_registry.py` - Runtime tree family storage ✅

---

## 0. Scope

This document defines a **phased implementation roadmap** to reach the project goal:

> **Any extant synthesizer design** (subtractive, FM/PM, wavetable, sampler/granular, modal/physical, hybrids) can be represented as an **Instrument L5** and executed deterministically and within bounded budgets by compiling into **VoiceFrame** microcode and running on the **WSE VM Hybrid**.

This standard does **not** redefine:

* VoiceFrame ABI: **WSE-STD-020 v1**
* VM lane schedule/executor semantics: **WSE-STD-VM-HYBRID v0**
* Operator library semantics: **WSE-STD-030** (and operator implementation tables)
* Budget + degradation ladder: **WSE-STD-BUDGET v0** (your implemented module set)

---

## 1. Definitions

### 1.1 Core objects

* **CMMM**: Control + Mode + Memory Manager. Per-voice, per-block state machine that:

  1. ingests control deltas and note lifecycle,
  2. updates regime state,
  3. chooses a bounded plan (ops + K + tail + events),
  4. emits **VoiceFrame**.
* **Carrier**: the stable oscillator/exciter that guarantees pitch quality and phase continuity.
* **Material Stack**: post-carrier transform stack (IDWPT, operator algebra, saturation/tape, etc.).
* **USS** (Universal Synth Surface): canonical patch schema spanning major synth families.

### 1.2 Architectural invariants (MUST)

1. **Determinism**: same inputs → same outputs (bit-exact on reference).
2. **Bounded Work**: cost per voice per block is predictable and capped.
3. **Block-time only**: downstream sees no sub-block time except intra-block ramps applied in fixed policy.
4. **Degradation is semantic**: if over budget, degrade via deterministic ladder (drop events → drop heavy ops → reduce K → tail downgrade → emergency).

---

## 2. Global deliverables (applies to every phase)

Every synthesis family you add MUST come with:

### 2.1 Spec artifacts

* A **CMMM Profile Spec** (section 6 format)
* **InstrumentIR fields** (L0/L1) required for that profile
* Compilation rules: **USS → InstrumentIR → VoiceFrames**

### 2.2 Regression artifacts

* **Golden Trace**: at least one musical trace per profile
* **Stress Trace**: must exercise PRNG gating, permutes, K ladder, tail ladder, event lanes
* Hash-frozen traces + regeneration audit trail

### 2.3 Budget artifacts

* Regime budgets for **MAIN** and **RESIDUAL**
* A degradation mapping table: which features degrade first and what “musical meaning” is preserved

---

## 3. Phase plan overview (what to build, in what order)

### Phase 0 — Substrate freeze ✅ COMPLETE

**Goal:** lock ABI/VM/budgets/determinism.

Exit criteria:

* WSE-STD-020 v1 stable ✅
* WSE-STD-VM-HYBRID v0 stable ✅
* Golden + stress traces pass and are audited ✅

---

### Phase 1 — Universal Synth Surface (USS) + Preset standard ✅ COMPLETE

**Goal:** represent "patch intent" uniformly.

Deliverables:

* **WSE-STD-PRESET v0** with:

  * `MacroPatch` (producer knobs) → Implemented as `ControlConfig` with 9 semantic axes
  * `CanonicalPatch` (osc/filter/env/LFO/routing) → Implemented as `InstrumentIR` L0 graph
  * `ProfileSelector` (which carrier family / stack is active) → Implemented as `osc_mode` enum
* Minimal translation layer to current VM: `CanonicalPatch → VoiceFrame defaults` → `to_capsule()` + `CapsuleCompiler`

**Implemented artifacts:**

* `instrument_ir.py` (1857 lines) — Complete layered IR schema (L0-L3)
* `library/preset_library.py` — PresetLibraryV0 with versioning + lifecycle
* `library/compiler.py` — CapsuleCompiler: IR → CapsuleV0
* Factory methods: `quick_synth()`, `quick_pad()`, `quick_pluck()`, `quick_bass()`, `quick_lead()`
* YAML preset format: `instrument_ir_v0:` schema with auto-completion

Exit criteria:

* You can express: basic subtractive patches ✅, basic modulation routings ✅, glide/unison (partial), amp ADSR ✅

---

### Phase 2 — SUBTRACTIVE baseline (Virtual Analog) 🟡 PARTIAL

**Goal:** copy the largest share of extant synth patches first.

Deliverables:

* CMMM Profile: **SUBTRACTIVE** — 🔲 Formal spec not written
* Carrier: multi-osc (saw/pulse/tri/sine), detune/unison, phase policy — ✅ Mode A + Mode B working
* Filter: one canonical filter core (SVF or ladder-ish) with cutoff/res + env amount — 🔲 Schema exists, audio path missing
* Amp ADSR + Filter ADSR — ✅ Amp ADSR complete; filter ADSR schema ready
* Mod matrix v0: LFO→(pitch/cutoff/pw/amp), velocity tracking — 🟡 Macro→param routing works; LFO generator missing

**Implemented artifacts:**

* `composite_osc.py` — Mode A (LUT) + Mode B (harmonic) blending
* `coeff_lut_streaming.py` — 8-bit half-phase LUT optimization
* `harmonic_bank.py` — Custom harmonic profiles (e.g., violin)
* `envelope_scheduler.py` — Block-rate ADSR
* Preset examples: `brass_stab_v0.yaml`, `bass_cheap_v0.yaml`, `pad_cheap_v0.yaml`

**Remaining work:**

* [ ] Filter audio implementation (SVF operator)
* [ ] LFO generator + mod routing
* [ ] Formal SUBTRACTIVE CMMM profile spec

Exit criteria:

* Clone pack: brass stabs ✅, plucks ✅, pads ✅, basses ✅, PWM strings 🔲
* Budget: degradation preserves pitch and envelope feel under load ✅

---

### Phase 3 — FM/PM (DX family) 🟢 LARGELY COMPLETE

**Goal:** cover FM operator graphs + feedback.

Deliverables:

* CMMM Profile: **FM_PM** — 🟡 Implementation complete; formal spec pending
* Operator graph fixed per preset; updates via macro params (no runtime graph mutation mid-block) — ✅
* Per-operator envelopes (or grouped envelopes v0) — ✅ **Matrix Envelope System** exceeds spec
* Optional post-filter — 🔲 Not implemented

**Implemented artifacts:**

* `fm6_engine.py` — 32 DX7 algorithms with sample-rate FM generation
* `fm6_state.py` — Operator state management
* `fm6_morph_engine.py` — Algorithm morphing
* `envelope_targets.py` — Matrix envelope system with FM_OP1-6 targets
* Feedback system working (tested in `test_fm_feedback.py`)
* FWPT gateway for coefficient-domain conversion

**Matrix Envelope System (exceeds original spec):**

* Named envelopes with independent ADSR
* Multi-target routing: `FM_OP1`-`FM_OP6`, `GAIN`, `FILTER_CUTOFF`, `PITCH`, `VIBRATO_DEPTH`
* Per-operator envelope independence (e.g., carrier vs modulator envelopes)
* Example: `fm_brass_v0.yaml` demonstrates carrier_env → FM_OP1-4, mod_env → FM_OP5-6

**FM6 preset examples:**

* `fm_brass_v0.yaml` — Algorithm 19 (Jump Brass)
* `fm_bell_v0.yaml` — Bell tones
* `fm_trumpet_v0.yaml` — Brass
* `fm_horn_v0.yaml` — Horn

**Remaining work:**

* [ ] Post-filter implementation
* [ ] Formal FM_PM CMMM profile spec document
* [ ] Fix: 5 FM6 presets fail to compile (operator ratios exceed int16 range)

Exit criteria:

* Clone pack: EP 🟡, bells ✅, metallic plucks 🟡, FM bass 🟡, glass pads 🟡

---

### Phase 4 — Wavetable + Waveshaping (modern)

**Goal:** table scanning, warp, sync-like behaviors.

Deliverables:

* CMMM Profile: **WAVETABLE**
* Osc: table interpolation, position scan, unison, hard-sync emulation (phase reset policy)
* Warp: waveshaper / phase distortion stage
* IDWPT stack as “material rack” (optional)

Exit criteria:

* Smooth scans, controlled aliasing, aggressive modern tones

---

### Phase 5 — Sampler + Granular

**Goal:** sample playback ecosystem.

Deliverables:

* CMMM Profile: **SAMPLER**

  * sample playback, loop modes, crossfades, pitch shift, filter, envelopes
* CMMM Profile: **GRANULAR**

  * deterministic grain scheduler (seeded PRNG), bounded grains/block
  * IDWPT as grain conditioner

Exit criteria:

* Basic ROMpler behavior + modern granular textures that are deterministic

---

### Phase 6 — Modal/Physical/Resonator

**Goal:** Karplus–Strong, resonator banks, modal percussion.

Deliverables:

* CMMM Profile: **MODAL_RESON**

  * exciter (noise/impulse/burst)
  * resonator bank (modal filters / waveguides)
  * coupling/motion via operator algebra patterns

Exit criteria:

* Plucked strings, mallets, metallic resonances with musically stable decay

---

### Phase 7 — Gearbox unification

**Goal:** unify all profiles under one “gearbox” selector without rewriting runtime.

Deliverables:

* **WSE-STD-GEARBOX v0**:

  * Carrier family selection
  * Material stack selection
  * Standard degrade semantics across families
* Unified “profile compiler” that emits VoiceFrame plans

Exit criteria:

* Switching profiles is “preset-only”; VM/ABI unchanged

---

### Phase 8 — Clone packs + similarity loop

**Goal:** systematic copying with measurable targets.

Deliverables:

* Patch packs per family (20–50 each)
* MIDI patterns per pack
* A similarity metric tool (offline) to guide tuning
* L5 instruments stored and versioned

Exit criteria:

* “Copy claims” backed by trace packs + metrics + reproducible renders

---

### Phase 9 — FPGA parity hardening

**Goal:** prove Python ↔ FPGA bit-exact for each profile’s golden/stress traces.

Deliverables:

* Fixed-point parity modules per profile
* FPGA test harness consuming VoiceFrames and comparing hashes
* Per-profile LUT/DSP/BRAM budgets and achieved voice counts

Exit criteria:

* Bit-exact trace match for all family golden packs

---

## 4. Required lane usage policy (VM mapping)

Your VM schedule is fixed. Each profile MUST declare which lanes it uses and how.

**Canonical VM execution (fixed):**
`CARRIER_RENDER → INJECT → BACKBONE → TAIL → IDWPT_RENDER → MIX_GAIN`

Rules (MUST):

* **Carrier_RENDER** must be the only required pitched signal source for “copy-grade” pitch stability.
* **IDWPT_RENDER** is treated as effect/material unless the profile explicitly overrides (rare).
* **PRNG** must only advance when a PRNG-marked operator executes (noise/granular).

---

## 5. InstrumentIR requirements per profile

This section defines the minimum IR fields (L0/L1) each profile MUST support.

> **Note**: InstrumentIR v0 is now implemented in `instrument_ir.py`. The actual field names may differ slightly from the original spec below. See the implementation for canonical names.

### 5.1 Shared L0 fields (all profiles MUST) ✅ IMPLEMENTED

* `voice_class`: MAIN/RESIDUAL → `voice_class` enum
* `regime_graph_id`: selects regime machine → `regime_id`
* `budgets_id`: regime budget table selector → `budget_profile`
* `degradation_policy_id`: ladder selector → `degradation_profile`
* `carrier_spec` (even if "off") → `osc_mode` + `carrier` block
* `material_spec` (IDWPT + ops stack config) → `material_blend`
* `env_amp_spec` (ADSR) → `envelopes[]` with `GAIN` target
* `mod_router_spec` (at least macro routing) → `control_defaults` (ControlConfig)

**Additional L0 fields implemented:**

* `osc_mode`: MODE_A | MODE_B | MODE_FM6 | MODE_SAMPLE
* `K`: wavelet decomposition depth
* `pattern_id`: coefficient pattern selector
* `envelopes[]`: Matrix envelope bindings (EnvelopeBinding)
* `control_defaults`: 9-axis semantic control config

### 5.2 SUBTRACTIVE profile fields 🟡 PARTIAL

* `carrier_osc[]`: list of osc specs → ✅ `carrier.harmonics[]` for Mode B

  * wave: SINE/SAW/PULSE/TRI → ✅ `wave_type` enum
  * detune_cents, phase_policy → 🟡 `detune_cents` exists; phase_policy partial
  * mix_gain → ✅ `gain`
  * pulsewidth (if PULSE) → 🔲 Not implemented
* `filter_spec`: → 🟡 Schema exists in IR

  * type: LP/BP/HP (start with LP) → ✅ `FilterMode` enum (LP, HP, BP, NOTCH)
  * cutoff_hz + resonance → ✅ `cutoff_hz`, `resonance`
  * env_amount → ✅ `env_amount`
  * keytrack → 🔲 Not implemented
* `env_filter_spec`: ADSR → ✅ Via matrix envelope with `FILTER_CUTOFF` target
* `unison_spec`: voices, detune spread, stereo spread (optional) → 🔲 Not implemented
* `glide_spec`: on/off, time → 🔲 Not implemented

### 5.3 FM_PM profile fields ✅ IMPLEMENTED

* `fm_graph_spec`: → ✅ `fm6` block in IR

  * operator count N → ✅ Fixed at 6 (DX7-style)
  * ratios/fixed freqs → ✅ `operators[].ratio` (float or fixed Hz)
  * modulation matrix (fixed structure) → ✅ `algorithm` (1-32, DX7 algorithms)
  * feedback coefficients → ✅ `feedback` (0.0-1.0)
* `env_op_spec[]`: envelopes per operator or grouped → ✅ **Matrix Envelope System**

  * Named envelopes with `targets: [FM_OP1, FM_OP2, ...]`
  * Independent ADSR per envelope
  * Carrier/modulator grouping via target lists
* `post_filter_spec` (optional) → 🔲 Not implemented

**FM6-specific fields implemented:**

* `algorithm`: DX7 algorithm number (1-32)
* `operators[]`: 6 operator definitions
  * `ratio`: frequency ratio or fixed Hz
  * `level`: output level (0.0-1.0)
  * `velocity_sens`: velocity sensitivity
* `feedback`: feedback amount for self-modulating operators
* `output_level`: master output scaling

### 5.4 WAVETABLE profile fields 🔲 NOT STARTED

* `wavetable_spec`:

  * table_id, interp mode, pos_q15
  * unison params
  * sync policy (phase reset rules)
* `warp_spec`:

  * warp_type, amount, pre/post filter ordering

### 5.5 SAMPLER profile fields 🔲 NOT STARTED

* `sample_spec`:

  * sample_id, root_key, loop mode, loop points
  * pitch mode, crossfade length
* `filter_spec`, `env_amp_spec`, (optional `env_filter_spec`)

> **Note**: `osc_mode = MODE_SAMPLE` exists in executor but sample playback not implemented.

### 5.6 GRANULAR profile fields 🔲 NOT STARTED

* `grain_spec`:

  * grain rate, duration, window type
  * position jitter, pitch jitter
  * max grains/block (bounded)
  * deterministic seed policy
* `conditioner_spec` (IDWPT/ops)

### 5.7 MODAL_RESON profile fields 🔲 NOT STARTED

* `exciter_spec`:

  * impulse/noise/burst, amplitude, color
* `resonator_spec`:

  * mode frequencies/decays
  * coupling matrix (sparse/banded)
* `body_material_spec` (optional operator stack)

---

## 6. CMMM Profile Spec template (MUST use for each profile)

Each profile MUST produce a document with:

### 6.1 Profile header

* Profile ID (e.g., SUBTRACTIVE)
* VM lanes used
* Required IR fields
* Budget regimes used
* Determinism notes (PRNG rules, phase rules)

### 6.2 Control manifold

* List of macro controls (producer-facing)
* Mapping to canonical parameters (USS)
* Mapping to VoiceFrame fields

### 6.3 State machine

* Regimes (IDLE/TRANSIENT/STABLE/GLITCH/RECOVERY + any profile-specific)
* Hysteresis and dwell rules
* Note lifecycle behavior: note_on/note_off, retrigger vs legato, glide

### 6.4 Compiler rules (normative)

* How L0/L1 compiles into:

  * carrier config
  * operator backbone + strengths
  * event lanes + payload
  * K/pattern_id selection
  * tail_mode selection
* Degradation semantics: what is preserved first

### 6.5 Tests

* Golden trace definition
* Stress trace definition
* Coverage assertions (noise lane, permutes, K ladder, tail ladder, ramps)

---

## 7. Checklists (what "done" means)

### 7.1 "Profile implemented" checklist

#### FM_PM Profile (Phase 3)

* [x] IR fields exist and are versioned — `fm6` block in InstrumentIR v0
* [x] USS → IR → VoiceFrame compile works — `to_capsule()` + CapsuleCompiler
* [x] VM executes profile under budgets deterministically — FM6 engine operational
* [ ] Golden trace exists and is hash-frozen — Presets exist; formal trace pack pending
* [x] Stress trace exists and covers:

  * [x] PRNG gating — deterministic RNG in voice engine
  * [x] permutes — degradation ladder
  * [x] K ladder — K selection in capsule
  * [x] tail ladder — tail_modes.py
  * [x] event lanes — ✅ 0x30/0x31 wired via ExecutorLatch + crossfade dispatch
  * [x] gain/beta ramps — envelope scheduler
* [x] Degradation ladder meaning documented — degradation_ladder.py

#### SUBTRACTIVE Profile (Phase 2)

* [x] IR fields exist and are versioned — carrier/filter in InstrumentIR v0
* [x] USS → IR → VoiceFrame compile works — Mode A/B compilation
* [x] VM executes profile under budgets deterministically — composite_osc operational
* [ ] Golden trace exists and is hash-frozen — pending
* [x] Stress trace exists and covers:

  * [x] PRNG gating
  * [x] permutes
  * [x] K ladder
  * [x] tail ladder
  * [x] event lanes — ✅ 0x30/0x31 wired via ExecutorLatch + crossfade dispatch
  * [x] gain/beta ramps
* [x] Degradation ladder meaning documented

### 7.2 "Copy-grade" checklist

#### FM_PM Profile

* [x] Pitch stable (carrier continuity) — FM6 phase accumulator
* [ ] Transients match (attack/brightness contour) — needs tuning
* [x] Macro controls feel correct (musician expectation) — basic controls work
* [x] Under load, degradation preserves intent (no random collapse)

#### SUBTRACTIVE Profile

* [x] Pitch stable (carrier continuity) — Mode A/B phase continuity
* [ ] Transients match (attack/brightness contour) — filter missing
* [ ] Macro controls feel correct (musician expectation) — LFO missing
* [x] Under load, degradation preserves intent (no random collapse)

---

## 8. Immediate next steps (your current best move)

~~1. **Implement WSE-STD-PRESET v0 (USS)**~~ ✅ Done (InstrumentIR v0)

2. **Complete SUBTRACTIVE CMMM profile** end-to-end:
   * [ ] Implement SVF/ladder filter audio operator
   * [ ] Implement LFO generator
   * [ ] Complete mod matrix routing (LFO → cutoff/pitch/pw)
   * [ ] Write formal WSE-STD-SUBTRACTIVE-PROFILE spec document

3. **Formalize FM_PM CMMM profile**:
   * [ ] Write WSE-STD-FM-PROFILE spec document
   * [ ] Fix FM6 preset serialization (int16 overflow on operator ratios)
   * [ ] Add post-filter to FM6 pipeline

4. Produce:
   * [ ] Subtractive golden trace pack (5–10 canonical patches)
   * [ ] FM golden trace pack (existing presets can serve as basis)
   * Stress trace machinery already exists—reuse it

After that, WAVETABLE becomes straightforward because it slots into the same VM/VoiceFrame contract.

---

## 9. New systems delivered (since v0)

The following systems were added after the initial roadmap:

### 9.1 InstrumentIR v0 (WSE-STD-INSTRUMENT-IR)

Complete layered tower IR with:

* **L0 Graph IR**: DSP node graph, edges, macros
* **L1 Lift IR**: Gesture basis, macro semantics
* **L2 Regimes IR**: Regime nodes, transitions, hysteresis
* **L3 CMMM IR**: State, transitions, overlays, emit templates
* Auto-completion: missing layers filled with sensible defaults
* Factory methods for quick preset creation

### 9.2 Matrix Envelope System (WSE-STD-ENVELOPE-MATRIX)

Enhanced envelope routing beyond simple ADSR:

* Named envelopes with independent timing
* Multi-target routing per envelope
* FM operator-level granularity (FM_OP1-6)
* Supports: `GAIN`, `FILTER_CUTOFF`, `PITCH`, `VIBRATO_DEPTH`

### 9.3 ControlConfig (WSE-STD-MACRO)

Semantic control axis defaults for cockpit integration:

* 9 dimensions: `energy_intent`, `brightness`, `density`, `risk`, `material`, `unity`, `space`, `motion`, `focus`
* Preset-level defaults loaded at capsule compile time

### 9.4 PresetLibraryV0

Production-quality preset management:

* YAML/JSON preset format (`instrument_ir_v0:`)
* Versioning + lifecycle state machine (draft → compiled → verified)
* Asset references for samples/wavetables
* Indexed library with search

### 9.5 SynthMaster

High-level synthesis pipeline:

* Unified entry point for preset → audio
* Toggleable stages for debugging
* Preset library integration
* Audio I/O handling

### 9.6 CMMM Event Bindings (WSE-STD-070 §10)

Full runtime event pipeline for K degradation and tail mode control:

* **ExecutorLatch** (`vm/state.py`): Holds CMMM event state with crossfade support
  * `node_mask_u32`, `mask_target_u32`, `mask_ramp_remaining` for smooth K transitions
  * `tail_mode`, `traction_q15` for tail operator control
  * `get_crossfade_weights()` returns Q15 blend weights during ramp
* **CMMM Event Processing** (`vm/executor.py`): Decodes VoiceFrame CMMM fields
  * 0x30 (NODEMASK_SET): Updates mask target + starts crossfade ramp
  * 0x31 (TAILMODE_SET): Updates tail mode + traction
* **Crossfade Dispatch** (`vm/lanes.py`): `dispatch_coeff_op_crossfade()` blends operator outputs
  * During mask ramp, applies operators with both masks and blends results
  * Prevents clicks during K degradation (32→24→16→12→8→4→1)
* **VoiceFrameEmitter CMMM Fields** (`voiceframe_emitter.py`): Emits CMMM fields from preset
  * `preset.cmmm` → `VoiceFrame.cmmm_node_mask`, `cmmm_tail_mode`, etc.
* **CmmmDefaults** (`library/preset_compiler.py`): Preset-level CMMM configuration
  * Pattern table resolution (`pattern:N` → mask)
  * Tail mode name resolution (`FULL`, `MID`, `MIN`, `EMERGENCY`)

**Test coverage:**

* `test_crossfade_dispatch.py` — ExecutorLatch weights, ramp behavior
* `test_cmmm_events.py` — Event roundtrip, K computation
* `test_golden_voiceframe_stream.py` — Deterministic replay verification
* `test_cmmm_template_emission.py` — Preset→VoiceFrame CMMM field flow

**Cross-reference:**
- [Tower Conformance Master Roadmap](./TOWER-CONFORMANCE-ROADMAP.md) - Authoritative conformance status
- [InstrumentIR Tower Conformance Spec](../../design/core-system/04%20-%20InstrumentIR%20Tower%20Conformance%20Spec.md) - Tower architecture details

---

## 10. Conformance Summary

The CMMM roadmap has reached the following conformance level:

| Requirement | Status | Notes |
|-------------|--------|-------|
| **Phase 0**: ABI/VM frozen | ✅ Complete | WSE-STD-020 v1, WSE-STD-VM-HYBRID v0 |
| **Phase 1**: USS/Preset | ✅ Complete | InstrumentIR v0 + PresetLibraryV0 |
| **Phase 2**: SUBTRACTIVE | 🟡 Partial | Carrier done; filter/LFO missing |
| **Phase 3**: FM/PM | 🟢 Largely Complete | FM6 engine + matrix envelopes |
| **Integration Bindings** | ✅ Complete | 0x30/0x31, crossfade, VoiceFrameEmitter |
| **Determinism** | ✅ Verified | test_golden_voiceframe_stream.py |
| **Degradation Ladder** | ✅ Complete | K ladder, tail modes, operator drop |

**Remaining for full conformance:**

1. Filter audio operator (SVF)
2. LFO generator + mod routing
3. Golden trace packs (hash-frozen)
4. Control plane wiring (ControlFrame → VoiceEngine) - see [TOWER-CONFORMANCE-ROADMAP](./TOWER-CONFORMANCE-ROADMAP.md)
5. Material Engine audio implementation
