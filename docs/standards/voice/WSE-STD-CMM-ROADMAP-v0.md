# WSE-STD-CMMM-ROADMAP v0

**Roadmap + conformance checklists for "CMMMs that can copy extant synthesizer designs"**
Status: **v0 (planning standard)** — normative where marked **MUST/SHOULD/MAY**

**Last Updated**: 2026-01-06

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
| Phase 1 — USS/Preset | 🔲 Not started | |
| Phase 2 — SUBTRACTIVE | 🟡 Partial | Carrier done (CCS Mode A/B), filter/ADSR/mod not started |
| Phase 3 — FM/PM | 🟡 Partial | Mode B has FM/PM hooks, profiles not started |
| Phase 4+ | 🔲 Not started | |

### Integration Bindings (per CMMM-addenda)

| Binding | Status | Notes |
|---------|--------|-------|
| payload_type 0x30 (NODEMASK_SET) | 🔲 Not wired | Pattern table exists, event lane decode needed |
| payload_type 0x31 (TAILMODE_SET) | 🔲 Not wired | Tail modes defined, event lane decode needed |
| Ramp enforcement | 🟡 Partial | Envelope ramps work; tree/mask crossfade needs latch |

### Related Systems

- **CCS Carrier**: Mode A + Mode B oscillators working. See `CCS-STATUS.md`.
- **Budgets**: `regime_budgets.py`, `tail_modes.py`, `degradation_ladder.py` ✅
- **Degradation Ladder**: Full deterministic degradation (events→permutes→lowrank→K→tail→emergency) ✅
- **VoiceFrame Emitter**: `voiceframe_emitter.py` - Preset→VoiceFrame with envelope/beta ✅
- **VM Executor**: Polyphase IDWPT synthesis operational ✅

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

### Phase 1 — Universal Synth Surface (USS) + Preset standard

**Goal:** represent “patch intent” uniformly.

Deliverables:

* **WSE-STD-PRESET v0** with:

  * `MacroPatch` (producer knobs)
  * `CanonicalPatch` (osc/filter/env/LFO/routing)
  * `ProfileSelector` (which carrier family / stack is active)
* Minimal translation layer to current VM: `CanonicalPatch → VoiceFrame defaults`

Exit criteria:

* You can express: basic subtractive patches, basic modulation routings, glide/unison, amp ADSR

---

### Phase 2 — SUBTRACTIVE baseline (Virtual Analog)

**Goal:** copy the largest share of extant synth patches first.

Deliverables:

* CMMM Profile: **SUBTRACTIVE**
* Carrier: multi-osc (saw/pulse/tri/sine), detune/unison, phase policy
* Filter: one canonical filter core (SVF or ladder-ish) with cutoff/res + env amount
* Amp ADSR + Filter ADSR
* Mod matrix v0: LFO→(pitch/cutoff/pw/amp), velocity tracking

Exit criteria:

* Clone pack: brass stabs, plucks, pads, basses, PWM strings
* Budget: degradation preserves pitch and envelope feel under load

---

### Phase 3 — FM/PM (DX family)

**Goal:** cover FM operator graphs + feedback.

Deliverables:

* CMMM Profile: **FM_PM**
* Operator graph fixed per preset; updates via macro params (no runtime graph mutation mid-block)
* Per-operator envelopes (or grouped envelopes v0)
* Optional post-filter

Exit criteria:

* Clone pack: EP, bells, metallic plucks, FM bass, glass pads

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

### 5.1 Shared L0 fields (all profiles MUST)

* `voice_class`: MAIN/RESIDUAL
* `regime_graph_id`: selects regime machine
* `budgets_id`: regime budget table selector
* `degradation_policy_id`: ladder selector
* `carrier_spec` (even if “off”)
* `material_spec` (IDWPT + ops stack config)
* `env_amp_spec` (ADSR)
* `mod_router_spec` (at least macro routing)

### 5.2 SUBTRACTIVE profile fields

* `carrier_osc[]`: list of osc specs

  * wave: SINE/SAW/PULSE/TRI
  * detune_cents, phase_policy
  * mix_gain
  * pulsewidth (if PULSE)
* `filter_spec`:

  * type: LP/BP/HP (start with LP)
  * cutoff_hz + resonance
  * env_amount
  * keytrack
* `env_filter_spec`: ADSR
* `unison_spec`: voices, detune spread, stereo spread (optional)
* `glide_spec`: on/off, time

### 5.3 FM_PM profile fields

* `fm_graph_spec`:

  * operator count N
  * ratios/fixed freqs
  * modulation matrix (fixed structure)
  * feedback coefficients
* `env_op_spec[]`: envelopes per operator or grouped
* `post_filter_spec` (optional)

### 5.4 WAVETABLE profile fields

* `wavetable_spec`:

  * table_id, interp mode, pos_q15
  * unison params
  * sync policy (phase reset rules)
* `warp_spec`:

  * warp_type, amount, pre/post filter ordering

### 5.5 SAMPLER profile fields

* `sample_spec`:

  * sample_id, root_key, loop mode, loop points
  * pitch mode, crossfade length
* `filter_spec`, `env_amp_spec`, (optional `env_filter_spec`)

### 5.6 GRANULAR profile fields

* `grain_spec`:

  * grain rate, duration, window type
  * position jitter, pitch jitter
  * max grains/block (bounded)
  * deterministic seed policy
* `conditioner_spec` (IDWPT/ops)

### 5.7 MODAL_RESON profile fields

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

## 7. Checklists (what “done” means)

### 7.1 “Profile implemented” checklist

* [ ] IR fields exist and are versioned
* [ ] USS → IR → VoiceFrame compile works
* [ ] VM executes profile under budgets deterministically
* [ ] Golden trace exists and is hash-frozen
* [ ] Stress trace exists and covers:

  * [ ] PRNG gating
  * [ ] permutes
  * [ ] K ladder
  * [ ] tail ladder
  * [ ] event lanes
  * [ ] gain/beta ramps
* [ ] Degradation ladder meaning documented

### 7.2 “Copy-grade” checklist

* [ ] Pitch stable (carrier continuity)
* [ ] Transients match (attack/brightness contour)
* [ ] Macro controls feel correct (musician expectation)
* [ ] Under load, degradation preserves intent (no random collapse)

---

## 8. Immediate next steps (your current best move)

1. **Implement WSE-STD-PRESET v0 (USS)**
2. **Implement SUBTRACTIVE CMMM profile** end-to-end
3. Produce:

   * a subtractive golden trace pack (5–10 canonical patches)
   * a subtractive stress trace (already have stress machinery—reuse it)

After that, FM_PM and WAVETABLE become straightforward because they slot into the same VM/VoiceFrame contract.
