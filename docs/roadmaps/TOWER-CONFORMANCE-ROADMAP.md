# Tower Conformance Master Roadmap

**Status:** Active
**Version:** 1.0
**Date:** 2026-01-19
**Cross-ref:** WSE-STD-070, Full Tower v0, WSE-STD-VM-HYBRID, WSE-STD-CMM-ROADMAP-v0

---

## Executive Summary

This document is the **authoritative master roadmap** for achieving full Tower conformance. It consolidates requirements from:

- `symcrash_docs/03 - Audio Pipeline/01 - Full Tower v0.md`
- `symcrash_docs/WSE-STD-070.md` (Editor Primitives, Tower Completion)
- `symcrash_docs/03 - Audio Pipeline/00 - Voice Engine/Evolution/02 - The Voice Engine Tower.md`
- `symcrash_docs/00 - Theory/04 - Voice Engine Background/04 - Lifting the Tower.md`
- `symcrash_docs/03 - Audio Pipeline/03 - Material Engine/00 - Operation.md`
- `symcrash_docs/03 - Audio Pipeline/04 - Safety Engine/04 - The Safety Engine All Mechanisms Fully Enumerated v0.md`
- `docs/roadmaps/WSE-STD-CMM-ROADMAP-v0.md`
- `docs/status-reports/implementation/PIPELINE-ARCHITECTURE-GAP.md`

**Current Conformance Level: P2 (control plane wired)**

---

## 1. Tower Level Status

The Tower is a functorial lifting chain in the category of Controlled Semi-Markov Mealy Machines (CMMM). Each level adds structure via left adjoints.

| Level | Name | Spec Definition | Implementation | Status |
|-------|------|-----------------|----------------|--------|
| **L0** | Patch | VoiceModel + GestureMap | `L0Graph`, `L0Node`, carrier, envelopes | ✅ 100% |
| **L1** | Family | + ArcLibrary (LFO/env modulators) | `L1Lift`, `GestureBasis`, `arc_phase_q16` | 🟡 80% |
| **L2** | Instrument | + RegimeGraph | `L2Regimes`, `RegimeNode`, 7 regimes | ✅ 95% |
| **L3** | Cockpit | + TreePolicy + envelope bounds | `L3Cmmm`, `EmitTemplate`, `CmmmControl` | 🟡 85% |
| **L4** | Forge | + Graph edits | `TowerLevel.L4_FORGE` enum only | 🔲 10% |
| **L5** | Lab | + New ops/trees | `TowerLevel.L5_LAB` enum only | 🔲 5% |

### L1 Family Gaps
- No family templates (String, Air, Metal, Choir, Noise-field)
- Family → regime graph template mapping undefined

### L3 Cockpit Gaps
- Envelope constraint set ($\mathcal{E}$) not formalized
- Atlas ($\mathcal{A}$) chart definitions missing

### L4/L5 Gaps
- GraphRewrite operations not defined
- NewOpDef/NewTreeDef format not specified
- Compilation validators ($C_{4\to3}$, $C_{5\to3}$) not implemented

---

## 2. CMMM Conformance

Per symcrash_docs CMMM definition: $\mathcal{M} = (S, U, P_\theta, E_\phi, M, Q)$

| Component | Spec | Implementation | Status |
|-----------|------|----------------|--------|
| **S** (State space) | Regimes, envelope phases, arc phases | `VoiceState` with 7 regimes, `arc_phase_q16[8]` | ✅ |
| **U** (Control input) | GestureBasis, IntentVector | `GestureBasis`, 8 intent lanes | ✅ |
| **P_θ** (Transition kernel) | Hysteresis, dwell, guards | `RegimeGraph`, `Hysteresis`, `dwell_min_blocks` | ✅ |
| **E_φ** (Emission function) | VoiceFrame per block | `VoiceFrame` ABI, family weights, events | ✅ |
| **M** (Memory) | Accumulated tension, fatigue | Not explicitly implemented | ❌ Gap |
| **Q** (Closure/safety) | Bounded, deterministic | Tail ops, degradation ladder | ✅ |

---

## 3. Conformance Profiles

| Profile | Requires | Status | Notes |
|---------|----------|--------|-------|
| **P0** | L0 + envelopes | ✅ Complete | Static synthesis works |
| **P1** | P0 + L2 regimes | ✅ Complete | State machine drives emission |
| **P2** | P1 + L3 CMMM + 0x30/0x31 | ✅ Complete | Control plane wired 2026-01-19 |
| **P3** | P2 + budgeting | 🟡 85% | Budget wiring incomplete |

---

## 4. Pipeline Stage Status

```
IntentFrame (GestureTime - async)
         │
         ▼
┌─────────────────────────────────┐
│   Stage 1: COCKPIT ENGINE       │  ✅ Implemented
│   Produces: ControlFrame        │  ✅ Flows to VoiceEngine
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Stage 2: VOICE ENGINE         │  ✅ Complete
│   Consumes: ControlFrame        │  ✅ step_block() canonical API
│   Produces: RenderBatch +       │
│             ParamFrame          │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Stage 3-6: SYMBOL ENGINE      │  ✅ Unified
│   24 Operators, 4 Tail Modes    │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Stage 7: WAVELET ENGINE       │  ✅ Working
│   Polyphase IDWPT               │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Stage 8: VOICE MIX            │  ✅ Working
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Stage 9: MATERIAL ENGINE      │  🟡 TPT SVF Filter
│   7 programs specified          │  ✅ Filter audio operational
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Stage 10: SAFETY ENGINE       │  ✅ Working
│   OLA, limiter, slew            │  ✅ ParamFrame from VoiceEngine
└─────────────────────────────────┘
```

---

## 5. Critical Gaps

### Tier 0: Architecture Blockers (Must Fix First)

| Gap | Severity | Location | Description |
|-----|----------|----------|-------------|
| ~~**ControlFrame discarded**~~ | ✅ FIXED | `voice_engine.py` | VoiceEngine.step_block() consumes ControlFrame |
| ~~**ParamFrame empty**~~ | ✅ FIXED | `voice_engine.py` | ParamFrame extracted from ControlFrame.globals |
| **IR → MacroRuntime** | 🟡 MEDIUM | `instrument_ir.py` | `ir.control` not consumed by MacroRuntime |

### Tier 1: Missing Audio Implementation

| Gap | Severity | Spec Location | Description |
|-----|----------|---------------|-------------|
| **Material Engine** | 🟡 MEDIUM | symcrash_docs Material Engine v0 | TPT SVF filter implemented; 6 programs remaining |
| ~~**SVF Filter**~~ | ✅ FIXED | WSE-STD-CMM-ROADMAP Phase 2 | TPT SVF in Material Engine (material_id=9) |
| ~~**LFO Generator**~~ | ✅ FIXED | WSE-STD-CMM-ROADMAP Phase 2 | `LFOState` + `arc_phase_q16` wired in voice_engine |

### Tier 2: Missing Specifications

| Gap | Severity | Description |
|-----|----------|-------------|
| **L1 Family Templates** | 🟡 MEDIUM | Need 5 templates with regime graph mappings |
| **Atlas $\mathcal{A}$** | 🟡 MEDIUM | Chart definitions, UI exposure rules |
| **Envelope Constraints $\mathcal{E}$** | 🟡 MEDIUM | Bounds guaranteeing tail predicate satisfiable |
| **Memory Operator M** | 🟢 LOW | Accumulated tension/fatigue (CMMM component) |

### Tier 3: L4/L5 (Deferred)

| Gap | Priority | Description |
|-----|----------|-------------|
| **GraphRewrite Ops** | LOW | ADD_REGIME, REMOVE_REGIME, ADD_TRANSITION typed operations |
| **NewOpDef Format** | DEFERRED | Operator kernel definition format |
| **Compilation Validators** | DEFERRED | $C_{4\to3}$, $C_{5\to3}$ proof obligations |

---

## 6. Material Engine Programs

Per `symcrash_docs/03 - Audio Pipeline/03 - Material Engine/00 - Operation.md`:

| ID | Program | Building Blocks | Status |
|----|---------|-----------------|--------|
| 0 | Clean Glue | P1 tilt, P2 saturator, P4 LP | 🔲 Not implemented |
| 1 | Tape Bus | P1 tilt, P3 tape hysteresis, P4 LP | 🔲 Not implemented |
| 2 | Rank Grit | P2 saturator, P5 rank window | 🔲 Not implemented |
| 3 | Tropical Cut Mix | P6 tropical, P2 saturator, P4 LP | 🔲 Not implemented |
| 4 | Catastrophe Snap (A2/A3) | P7 catastrophe, P4 LP | 🔲 Not implemented |
| 5 | Catastrophe Swallowtail (A4) | P7 catastrophe, P3 tape, P4 LP | 🔲 Not implemented |
| 6 | Terrain Crackle | Waveshaper, P2 saturator, P4 LP | 🔲 Optional v0 |

### Building Blocks Required

| ID | Primitive | Description | State |
|----|-----------|-------------|-------|
| P1 | Pre-emphasis/tilt | 1st-order shelving EQ | 1 pole/channel |
| P2 | Soft saturator | tanh/rational/cubic | Memoryless |
| P3 | Tape hysteresis | Lightweight magnetization model | m per channel |
| P4 | One-pole LP/HP | IIR filter | z per channel |
| P5 | Rank nonlinearity | Median/trimmed mean window | Ring buffer |
| P6 | Tropical min/max | min-plus mixer | Optional short delay |
| P7 | Catastrophe shaper | A2/A3/A4 polynomial maps | Latch variable |

---

## 7. Safety Engine Mechanisms (16 Total)

Per `symcrash_docs/03 - Audio Pipeline/04 - Safety Engine/04 - The Safety Engine All Mechanisms Fully Enumerated v0.md`:

### Structural Safety (Coefficient-Space)
| # | Mechanism | Operator | Status |
|---|-----------|----------|--------|
| 1 | FineClamp | Op 2 | ✅ |
| 2 | NormalizeScale | Op 11 | ✅ |
| 3 | SheafProject | Op 23 | ✅ |
| 4 | GluePair | Op 12 | ✅ |

### Behavioral Safety (State & Time)
| # | Mechanism | Location | Status |
|---|-----------|----------|--------|
| 5 | Regime hysteresis | `RegimeGraph` | ✅ |
| 6 | Catastrophe gating | Implicit | 🟡 |
| 7 | Recovery regimes | R6_RECOVERY | ✅ |

### Execution Safety (Determinism)
| # | Mechanism | Location | Status |
|---|-----------|----------|--------|
| 8 | Mandatory tail | `vm/lanes.py` | ✅ |
| 9 | Degradation policy | `degradation_ladder.py` | ✅ |
| 10 | State freeze | `VoiceState` | ✅ |

### Numerical Safety (Fixed-Point)
| # | Mechanism | Location | Status |
|---|-----------|----------|--------|
| 11 | Saturation discipline | Q15 throughout | ✅ |
| 12 | PRNG draw order | xorshift32 | ✅ |

### Audio-Plane Safety
| # | Mechanism | Location | Status |
|---|-----------|----------|--------|
| 13 | OLA crossfade | `unified_renderer.py` | ✅ |
| 14 | Final limiter | `unified_renderer.py` | ✅ |

### Meta-Safety (Human Factors)
| # | Mechanism | Location | Status |
|---|-----------|----------|--------|
| 15 | Bounded macros | `ControlConfig` | ✅ |
| 16 | Editor separation | Partial | 🟡 |

---

## 8. Synth Profile Status

Per WSE-STD-CMM-ROADMAP:

| Phase | Profile | IR Fields | Audio | Formal Spec | Golden Traces |
|-------|---------|-----------|-------|-------------|---------------|
| 2 | SUBTRACTIVE | ✅ | ✅ 90% | 🔲 | 🔲 |
| 3 | FM/PM | ✅ | ✅ 85% | 🔲 | 🟡 |
| 4 | WAVETABLE | 🔲 | 🔲 | 🔲 | 🔲 |
| 5 | SAMPLER | 🔲 | 🔲 | 🔲 | 🔲 |
| 6 | GRANULAR | 🔲 | 🔲 | 🔲 | 🔲 |
| 7 | MODAL/RESON | 🔲 | 🔲 | 🔲 | 🔲 |

### SUBTRACTIVE Remaining
- [x] SVF filter audio operator — ✅ TPT SVF in Material Engine (2026-01-20)
- [x] LFO generator — ✅ LFOState with arc_phase_q16
- [x] LFO → mod matrix routing — ✅ FILTER_CUTOFF, FILTER_RESO targets
- [x] BRIGHTNESS → filter cutoff — ✅ Macro wiring in voice_engine.py
- [ ] Formal WSE-STD-SUBTRACTIVE-PROFILE spec

### FM/PM Remaining
- [ ] Formal WSE-STD-FM-PROFILE spec
- [ ] Fix 5 FM6 presets (int16 overflow on operator ratios)
- [ ] Post-filter implementation
- [ ] Golden trace pack

---

## 9. Implementation Roadmap

### Phase A: Wire Control Plane (Unlocks P2 Conformance) ✅ COMPLETE

**Goal:** ControlFrame flows from Cockpit → VoiceEngine → ParamFrame

| Task | Location | Status |
|------|----------|--------|
| Wire ControlFrame → VoiceEngine | `voice_engine.py` | ✅ Done |
| Populate ParamFrame from ControlFrame.globals | `voice_engine.py` | ✅ Done |
| VoiceEngine.step_block() canonical API | `voice_engine.py` | ✅ Done |
| SynthMaster uses VoiceEngine.step_block() | `synth_master.py` | ✅ Done |
| Connect InstrumentIR.control → MacroRuntime | `preset_compiler.py` | 🟡 Remaining |

**Exit Criteria:**
- [x] ControlFrame.events routed to VoiceEngine (NOTE_ON/OFF)
- [x] ParamFrame extracted from ControlFrame.globals
- [x] VoiceEngine returns RenderBatch + ParamFrame + Telemetry
- [ ] `ir.control` defaults applied at preset load

**Completed:** 2026-01-19
**Key Changes:**
- Added `VoiceEngine.step_block(control_frame) -> VoiceEngineOutput`
- Added `RenderBatchItem` and `VoiceEngineOutput` dataclasses
- SynthMaster now routes ControlFrame through VoiceEngine
- ParamFrame flows from VoiceEngine to Material/Safety engines

### Phase B: Complete SUBTRACTIVE Profile ✅ COMPLETE

**Goal:** Full virtual analog synthesis path

| Task | Location | Status |
|------|----------|--------|
| Implement SVF filter audio operator | `material/material_engine.py` | ✅ Done (TPT SVF) |
| Implement LFO generator | `vm/harmonic_bank.py` | ✅ Done (LFOState) |
| Wire LFO → mod matrix | `voice/voice_engine.py` | ✅ Done |
| Wire VIBRATO macro → LFO depth | `voice/voice_engine.py` | ✅ Done |
| Wire BRIGHTNESS macro → filter cutoff | `voice/voice_engine.py` | ✅ Done |
| Write WSE-STD-SUBTRACTIVE-PROFILE | `docs/standards/` | 🟢 LOW |

**Exit Criteria:**
- [x] Filter cutoff modulates audio — TPT SVF in Material Engine
- [x] LFO → pitch/cutoff/pw working — LFOTarget.FILTER_CUTOFF/FILTER_RESO
- [ ] PWM strings preset playable — needs testing
- [x] BRIGHTNESS macro controls filter — voice_engine.py wiring

**Completed:** 2026-01-20
**Key Changes:**
- Added TPT SVF filter in `material_engine.py` (material_id=9)
- Added `LFOTarget.FILTER_CUTOFF` and `LFOTarget.FILTER_RESO` enums
- Wired `SystemMacro.VIBRATO` to scale LFO output
- Wired `SystemMacro.BRIGHTNESS` to filter cutoff
- Filter params flow through VoiceFrame → SynthMaster → MaterialEngine

### Phase C: Material Engine v0

**Goal:** Analog coloration path operational

| Task | Priority |
|------|----------|
| Implement P1-P4 building blocks (tilt, saturator, tape, one-pole) | 🟡 MEDIUM |
| Implement material_id 0 (Clean Glue) | 🟡 MEDIUM |
| Implement material_id 1 (Tape Bus) | 🟡 MEDIUM |
| Wire ParamFrame → material controls | 🟡 MEDIUM |

**Exit Criteria:**
- [ ] Clean Glue applies gentle saturation
- [ ] Tape Bus adds weight/memory
- [ ] `material_id` selectable from preset

### Phase D: L1 Family Templates

**Goal:** Family classification with regime templates

| Task | Priority |
|------|----------|
| Define 5 family templates (String, Air, Metal, Choir, Noise-field) | 🟡 MEDIUM |
| Map family → default regime graph | 🟡 MEDIUM |
| Add family inference from `tree_id` | 🟢 LOW |

**Exit Criteria:**
- [ ] `L1Lift.family` populated from template
- [ ] Family affects default regimes

### Phase E: Formalize FM Profile

**Goal:** FM/PM fully documented and tested

| Task | Priority |
|------|----------|
| Write WSE-STD-FM-PROFILE spec | 🟢 LOW |
| Fix FM6 preset serialization (int16 overflow) | 🟡 MEDIUM |
| Create golden trace pack | 🟢 LOW |

**Exit Criteria:**
- [ ] All FM6 presets compile
- [ ] Hash-frozen golden traces exist

### Phase F: L4/L5 (When Needed)

**Goal:** Instrument builder features

| Task | Priority |
|------|----------|
| Define GraphRewrite typed operations | 🟢 LOW |
| Implement compilation validators | 🟢 LOW |
| Define NewOpDef kernel format | DEFERRED |

**Exit Criteria:**
- [ ] ADD_REGIME/REMOVE_REGIME work
- [ ] $V_4$ validator rejects invalid graphs

---

## 10. Test Coverage Requirements

### Per-Profile Requirements (from WSE-STD-CMM-ROADMAP)

Each profile MUST have:
- [ ] Golden trace (hash-frozen)
- [ ] Stress trace covering:
  - [ ] PRNG gating
  - [ ] Permutes
  - [ ] K ladder
  - [ ] Tail ladder
  - [ ] Event lanes (0x30/0x31)
  - [ ] Gain/beta ramps

### Existing Test Coverage

| Test | Coverage |
|------|----------|
| `test_crossfade_dispatch.py` | ExecutorLatch weights, ramp behavior |
| `test_cmmm_events.py` | Event roundtrip, K computation |
| `test_golden_voiceframe_stream.py` | Deterministic replay |
| `test_fm6_pipeline.py` | FM6 → coefficients → audio |

---

## 11. References

### Normative Specs (symcrash_docs)
- `03 - Audio Pipeline/01 - Full Tower v0.md`
- `WSE-STD-070.md`
- `03 - Audio Pipeline/00 - Voice Engine/Evolution/02 - The Voice Engine Tower.md`
- `00 - Theory/04 - Voice Engine Background/04 - Lifting the Tower.md`
- `03 - Audio Pipeline/00 - Voice Engine/Presets Developement/CMMM.md`
- `03 - Audio Pipeline/03 - Material Engine/00 - Operation.md`
- `03 - Audio Pipeline/04 - Safety Engine/04 - The Safety Engine All Mechanisms Fully Enumerated v0.md`

### Implementation Docs (docs)
- `docs/standards/architecture/WSE-STD-070.md`
- `docs/standards/voice/WSE-STD-VM-HYBRID.md`
- `docs/roadmaps/WSE-STD-CMM-ROADMAP-v0.md`
- `docs/roadmaps/VOICE-ENGINE-INTEGRATION-PLAN.md`
- `docs/status-reports/implementation/PIPELINE-ARCHITECTURE-GAP.md`
- `docs/design/core-system/04 - InstrumentIR Tower Conformance Spec.md`

### Key Source Files
- `src/symcrash_python_ap/instrument_ir.py` - Tower IR definitions
- `src/symcrash_python_ap/voice/voice_engine.py` - CMMM implementation
- `src/symcrash_python_ap/vm/executor.py` - VM execution
- `src/symcrash_python_ap/vm/state.py` - ExecutorLatch
- `src/symcrash_python_ap/synth_master.py` - Pipeline orchestrator
- `src/symcrash_python_ap/primitives.py` - Canonical enums and tables

---

## 12. Document Authority Matrix

This matrix clarifies which document is authoritative for each Tower concept.

| Topic | Authoritative Document | Category | Notes |
|-------|----------------------|----------|-------|
| **Tower level definitions** | [WSE-STD-070](../standards/architecture/WSE-STD-070.md) §3 | Standard | Normative L0-L5 specs |
| **Tower quick reference** | [WSE-STD-TOWER-OVERVIEW](../standards/architecture/WSE-STD-TOWER-OVERVIEW.md) | Standard | Entry point for developers |
| **Preset format** | [WSE-STD-PRESET-v0](../standards/control/WSE-STD-PRESET-v0.md) | Standard | YAML/JSON schema |
| **Preset authoring** | [PRESET-AUTHORING-GUIDE](../standards/library/PRESET-AUTHORING-GUIDE.md) | Guide | Python API usage |
| **Editor primitives** | [WSE-STD-070](../standards/architecture/WSE-STD-070.md) §2 | Standard | Normative |
| **Tower completion rules** | [WSE-STD-070](../standards/architecture/WSE-STD-070.md) §4 | Standard | Normative |
| **CMMM definition** | [Voice Engine Tower](../../symcrash_docs/03%20-%20Audio%20Pipeline/00%20-%20Voice%20Engine/Evolution/02%20-%20The%20Voice%20Engine%20Tower.md) | Theory | Mathematical model |
| **Functorial structure** | [Lifting the Tower](../../symcrash_docs/00%20-%20Theory/04%20-%20Voice%20Engine%20Background/04%20-%20Lifting%20the%20Tower.md) | Theory | Category theory |
| **IR layer schemas** | [InstrumentIR Conformance](../design/core-system/04%20-%20InstrumentIR%20Tower%20Conformance%20Spec.md) §1-3 | Design | Implementation details |
| **Emit templates** | [InstrumentIR Conformance](../design/core-system/04%20-%20InstrumentIR%20Tower%20Conformance%20Spec.md) §3 | Design | Template format |
| **Envelope system** | [WSE-STD-ENVELOPE-SUITE-v0](../standards/control/WSE-STD-ENVELOPE-SUITE-v0.md) | Standard | ADSR, AR, LFO, etc. |
| **Macro system** | [MACROS-AND-CONTROLS](../standards/library/MACROS-AND-CONTROLS.md) | Guide | Control layer |
| **Fixed-point arithmetic** | [WSE-STD-001](../standards/core/WSE-STD-001.md) | Standard | Q15/Q16 formats |
| **FM6 synthesis** | [FM6_PATHWAY_GUIDE](../FM6_PATHWAY_GUIDE.md) | Guide | Complete FM6 docs |
| **Implementation status** | This document | Roadmap | Living status tracker |
| **L5 Lab (future)** | WSE-STD-L5-LAB (TBD) | Standard | When specified |

### Document Categories

| Category | Mutability | Authority |
|----------|------------|-----------|
| **Standard** | Frozen once finalized | Primary - normative |
| **Guide** | Updated for clarity | Secondary - practical |
| **Design** | Evolving | Secondary - exploratory |
| **Theory** | Reference only | Background - educational |
| **Roadmap** | Actively maintained | Status tracking only |

---

## Changelog

### v1.2 (2026-01-20)
- **Phase B Complete:** SUBTRACTIVE profile audio operational
  - TPT SVF filter implemented in Material Engine (material_id=9)
  - LFOState system with arc_phase_q16 unified storage
  - LFO → filter cutoff/resonance routing (LFOTarget enums)
  - VIBRATO macro → LFO depth scaling
  - BRIGHTNESS macro → filter cutoff modulation
  - Filter params flow: VoiceFrame → SynthMaster → MaterialEngine
- Updated SUBTRACTIVE audio status: 60% → 90%
- Tier 1 gaps (SVF Filter, LFO Generator) marked FIXED
- Material Engine status updated: schema only → TPT SVF operational

### v1.1 (2026-01-19)
- **Phase A Complete:** Control plane wired
  - VoiceEngine.step_block() is now canonical API
  - ControlFrame flows from CockpitEngine → VoiceEngine
  - ParamFrame extracted and flows to Material/Safety
  - RenderBatch output enables future scheduler integration
- Updated conformance level: P1+ → P2
- Tier 0 gaps (ControlFrame discarded, ParamFrame empty) marked FIXED
- Updated pipeline diagram to show new data flow

### v1.0 (2026-01-19)
- Initial consolidation from all spec sources
- Identified Tier 0 control plane gaps
- Mapped Material Engine programs to building blocks
- Enumerated all 16 Safety Engine mechanisms
- Created phased implementation roadmap
