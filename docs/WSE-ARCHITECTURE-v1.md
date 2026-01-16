# Wavelet Synth Engine — Systems Architecture v1

**Status:** CURRENT
**Date:** 2026-01-06
**Supersedes:** [WSE-ARCHITECTURE-v0.md](standards/architecture/WSE-ARCHITECTURE-v0.md) (frozen 2025-12-22)
**Scope:** Complete architecture specification for Python reference implementation and FPGA target

---

## Executive Summary

The Wavelet Synth Engine (WSE) is a deterministic, real-time audio synthesis system built on wavelet packet decomposition and catastrophe-theoretic operator families. This document describes the current v1 architecture from a systems perspective, covering data flow, timing ownership, execution semantics, and determinism guarantees.

The architecture is designed for:
- **Bit-exact reproducibility** between Python simulation and FPGA hardware
- **Deterministic replay** regardless of scheduling decisions
- **Graceful degradation** under compute pressure
- **Clean separation** between musical intent and mechanical execution

### What's New in v1

This version updates WSE-ARCHITECTURE-v0 with:
- **Material Engine** - New analog processing layer (6 material programs)
- **Implementation Status Dashboard** - Current state of all subsystems
- **CCS Status** - Current implementation issues and fix roadmap
- **Control Plane Architecture** - Cockpit Engine integration details
- **Standards Cross-Reference** - Complete mapping of WSE-STD documents

---

## 1. Implementation Status Dashboard

Current state of major subsystems (as of 2026-01-06):

| Subsystem | Status | Details |
|-----------|--------|---------|
| **Budget System** | ✅ **FROZEN & CONFORMANT** | 11/11 requirements PASS. [BUDGET-CONFORMANCE-SUMMARY.md](status-reports/reviews/BUDGET-CONFORMANCE-SUMMARY.md) |
| **Control Plane** | ✅ **INTEGRATED** | Cockpit Engine operational. [WSE-STD-COCKPIT-ENGINE.md](standards/control/WSE-STD-COCKPIT-ENGINE.md) |
| **Material Engine** | ✅ **SPECIFIED** | 6 material programs defined. [WSE-STD-MATERIAL-v0.md](standards/budget/WSE-STD-MATERIAL-v0.md) |
| **CCS** | ❌ **NOT FUNCTIONAL** | Wrong frequencies (750 Hz vs target). [CCS-STATUS.md](status-reports/implementation/CCS-STATUS.md) |
| **CMMM** | ✅ **PHASE 1 COMPLETE** | 0x30/0x31 events, ExecutorLatch, crossfade dispatch. [WSE-STD-070.md](standards/architecture/WSE-STD-070.md) |
| **Wavelet Engine** | ✅ **OPERATIONAL** | Polyphase IDWPT working. [WSE-STD-120.md](standards/ccs/WSE-STD-120.md) |
| **Symbol Engine** | ✅ **OPERATIONAL** | Operator execution working. [WSE-STD-030.md](standards/core/WSE-STD-030.md) |
| **Voice Engine** | ✅ **OPERATIONAL** | Time ownership enforced. [WSE-STD-060.md](standards/runtime/WSE-STD-060.md) |

---

## 2. System Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER LAYER                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │    MIDI     │  │   Sensors   │  │     UI      │  │   Presets   │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │                │
│         └────────────────┴────────────────┴────────────────┘                │
│                                   │                                          │
│                                   ▼                                          │
│                          ┌───────────────┐                                  │
│                          │  IntentFrame  │  (GestureTime - async)           │
│                          └───────┬───────┘                                  │
└──────────────────────────────────┼──────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────┼──────────────────────────────────────────┐
│                          CONTROL PLANE                                       │
│                     ┌─────────────────────┐                                 │
│                     │  Cockpit Engine     │  (NEW in v1)                    │
│                     │  WSE-STD-COCKPIT    │                                 │
│                     └──────────┬──────────┘                                 │
│                                │                                             │
└────────────────────────────────┼─────────────────────────────────────────────┘
                                   │
┌──────────────────────────────────┼──────────────────────────────────────────┐
│                          VOICE ENGINE                                        │
│                     (Sole Owner of Musical Time)                            │
│                                   │                                          │
│  ┌─────────────┐  ┌─────────────┐ │ ┌─────────────┐  ┌─────────────┐       │
│  │    ADSR     │  │   Regime    │ │ │    PRNG     │  │  Scheduler  │       │
│  │  Envelopes  │  │   Machine   │ │ │   Manager   │  │   (Degrade) │       │
│  └─────────────┘  └─────────────┘ │ └─────────────┘  └─────────────┘       │
│                                   │                                          │
│                                   ▼                                          │
│                          ┌───────────────┐                                  │
│                          │  VoiceFrame   │  (BlockTime - 64 samples)        │
│                          └───────┬───────┘                                  │
└──────────────────────────────────┼──────────────────────────────────────────┘
                                   │
┌──────────────────────────────────┼──────────────────────────────────────────┐
│                          EXECUTION LAYER                                     │
│                         (Time-Blind Engines)                                │
│                                   │                                          │
│         ┌─────────────────────────┼─────────────────────────┐               │
│         │                         │                         │               │
│         ▼                         ▼                         ▼               │
│  ┌─────────────┐          ┌─────────────┐          ┌─────────────┐         │
│  │   Symbol    │          │   Wavelet   │          │   Voice     │         │
│  │   Engine    │    ───►  │   Engine    │    ───►  │    Mix      │         │
│  │ (Operators) │          │  (Inverse)  │          │   Stage     │         │
│  └─────────────┘          └─────────────┘          └──────┬──────┘         │
│                                                            │                │
│                                                            ▼                │
│                                                   ┌─────────────────┐       │
│                                                   │   Material      │  (v1) │
│                                                   │   Engine        │       │
│                                                   │ (Analog Layer)  │       │
│                                                   └────────┬────────┘       │
│                                                            │                │
│                                                            ▼                │
│                                                   ┌─────────────────┐       │
│                                                   │   Safety        │       │
│                                                   │   Engine        │       │
│                                                   │  (OLA/Tail)     │       │
│                                                   └────────┬────────┘       │
└────────────────────────────────────────────────────────────┼────────────────┘
                                                             │
                                                             ▼
                                                    ┌───────────────┐
                                                    │  Audio Output │
                                                    │  (64 samples) │
                                                    └───────────────┘
```

### 2.2 Core Standards Reference

| Standard | Title | Purpose | Status |
|----------|-------|---------|--------|
| [WSE-STD-001](standards/core/WSE-STD-001.md) | Numeric Determinism | Fixed-point Q15/Q8, rounding rules | FROZEN |
| [WSE-STD-010](standards/wavelet/WSE-STD-010.md) | Tree Library | Wavelet tree contract, leaf ordering | FROZEN |
| [WSE-STD-020](standards/core/WSE-STD-020.md) | Frame ABI | VoiceFrame packet (136 bytes) | FROZEN |
| [WSE-STD-030](standards/core/WSE-STD-030.md) | Symbol Engine Executor | Operator scheduling, tail predicate | FROZEN |
| [WSE-STD-040](standards/runtime/WSE-STD-040.md) | Deterministic PRNG | PCG-32, per-voice isolation | FROZEN |
| [WSE-STD-050](standards/runtime/WSE-STD-050.md) | Degradation Semantics | Freeze/Decay, voice scheduling | FROZEN |
| [WSE-STD-060](standards/runtime/WSE-STD-060.md) | Time Ownership | Single owner of musical time | FROZEN |
| [WSE-STD-070](standards/architecture/WSE-STD-070.md) | Presets & Tower | L0-L5 layers, completion rules | FROZEN |
| [WSE-STD-120](standards/ccs/WSE-STD-120.md) | Polyphase Upgrade | IDWPT synthesis operator | FROZEN |
| [WSE-STD-ARCH-CCS](standards/ccs/WSE-STD-ARCH-CCS.md) | CCS Architecture | No FWPT in voice path | FROZEN |
| [WSE-STD-MATERIAL-v0](standards/budget/WSE-STD-MATERIAL-v0.md) | Material Engine | Analog processing layer | FROZEN |
| [WSE-STD-DEGRADE-v0](standards/budget/WSE-STD-DEGRADE-v0.md) | Budget & Degradation | K ladder, tail modes | FROZEN |
| [WSE-STD-COCKPIT-ENGINE](standards/control/WSE-STD-COCKPIT-ENGINE.md) | Control Plane | Slew bank, binding table | FROZEN |

---

## 3. Time Domains

### 3.1 Three Time Domains (WSE-STD-060)

| Domain | Resolution | Owner | Characteristics |
|--------|------------|-------|-----------------|
| **GestureTime** | Asynchronous, continuous | UI/MIDI/Sensors | Human input, variable rate |
| **VoiceTime** | Discrete, block-quantized | Voice Engine | Musical decisions, state machines |
| **BlockTime** | Discrete, block-exact | Symbol/Wavelet/Material/Safety | Mechanical execution, 64 samples |

### 3.2 Time Flow

```
GestureTime          VoiceTime              BlockTime
    │                    │                      │
    │  IntentFrame       │                      │
    │ ─────────────────► │                      │
    │  (async arrival)   │                      │
    │                    │  VoiceFrame          │
    │                    │ ───────────────────► │
    │                    │  (per block)         │
    │                    │                      │
```

**Critical Rule:** Only VoiceTime → BlockTime conversion is permitted. No engine downstream of Voice Engine may observe sub-block time.

### 3.3 BlockTime Definition

- **One block = exactly 64 audio samples**
- At 48kHz: 1 block ≈ 1.33ms
- All downstream engines operate strictly at BlockTime resolution

---

## 4. Engine Responsibilities

### 4.1 Voice Engine (Time-Aware)

The Voice Engine is the **sole owner of musical time**.

**MUST:**
- Maintain all counters, clocks, and phase state
- Quantize all timing to BlockTime
- Emit exactly one VoiceFrame per active voice per block
- Decide **when** events occur, not how they execute
- Own ADSR envelopes, regime state machines, phase progression

**MAY:**
- Skip updates for Cloud voices (per WSE-STD-050)
- Run at lower rates internally if outputs are block-quantized

**State Owned:**
- ADSR phase and time
- Regime ID and dwell counters
- PRNG state (per voice)
- Intent accumulation

### 4.2 Symbol Engine (Time-Blind)

The Symbol Engine applies wavelet coefficient operators.

**MUST:**
- Treat each VoiceFrame as a complete, self-contained command
- Apply operators exactly as instructed
- Enforce legality, bounds, sparsity, and safety rules
- Remain stateless with respect to musical time

**MUST NOT:**
- Maintain counters or clocks
- Use `block % N` modulo logic
- Detect phases or regime progress
- Trigger events based on elapsed time

**Operator Budget (per WSE-STD-030):**
- DIAG operators: 1N cost
- PERMUTE operators: 0.5N compute, 2N memory
- BANDED operators: 3N cost
- LOWRANK(r) operators: 2rN cost

### 4.3 Wavelet Engine (Purely Mechanical)

The Wavelet Engine performs inverse wavelet transforms.

**MUST:**
- Accept coefficient states and reconstruction parameters
- Perform deterministic inverse transforms
- Perform numerical interpolation when instructed

**MUST NOT:**
- Observe note on/off
- Contain envelopes
- Schedule transitions
- Trigger events

### 4.4 Material Engine (NEW in v1)

The Material Engine applies analog-inspired physical modeling.

**Pipeline Placement:**
```
Symbol Engine (CMMM)
    ↓ (coefficient evolution)
Wavelet Engine (IDWPT)
    ↓ (audio reconstruction)
Voice Mix Stage
    ↓ (sum all voices → stereo)
━━━━━━━━━━━━━━━━━━━━━━━━
MATERIAL ENGINE ← HERE
━━━━━━━━━━━━━━━━━━━━━━━━
    ↓ (analog processing)
Safety/Limiter (Tail Stage)
    ↓ (bounded output)
Audio Output
```

**Material Programs (6 normative):**

| material_id | Name | Category | State Type |
|-------------|------|----------|------------|
| 0 | Clean Glue | Transparent | None |
| 1 | Tape Bus | Hysteresis + Tone | TapeState |
| 2 | Rank Grit | Order Statistics | RankState |
| 3 | Tropical Cut | Hard Threshold | None |
| 4 | Catastrophe Fold | A2 Bifurcation | CatastropheState |
| 5 | Catastrophe Cusp | A3 Bifurcation | CatastropheState |

**MUST:**
- Process after voice mix, before safety/limiter
- Use Q1.15 fixed-point arithmetic (WSE-STD-001)
- Maintain per-channel (L/R) state
- Implement parameter slewing for click-free transitions
- Satisfy safety contract (bounded output, no wrap/NaN)

**Reference:** [WSE-STD-MATERIAL-v0.md](standards/budget/WSE-STD-MATERIAL-v0.md)

### 4.5 Safety Engine (Block-Shaped Only)

The Safety Engine enforces audio safety.

**MUST:**
- Enforce block-local invariants (tail predicate)
- Apply OLA, de-click, slew limiting, DC correction
- Be idempotent with respect to musical meaning

**MUST NOT:**
- Detect musical phases
- Alter regime state
- Introduce new events

---

## 5. Control Plane Architecture (NEW in v1)

### 5.1 Cockpit Engine

The **Cockpit Engine** is the canonical control plane that converts all external inputs (Pots, MIDI, ImGui, serial deltas) into pipeline-native objects.

**Invariant:**
> Cockpit samples async inputs into block-rate state. No downstream engine may observe sub-block timing jitter.

**Architecture:**

```
┌─────────────────────────────────────────────────────┐
│             CONTROL INPUTS                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Pico2    │  │  MIDI    │  │  ImGui   │          │
│  │ Pots     │  │  Input   │  │  Knobs   │          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│       │             │             │                 │
│       └─────────────┼─────────────┘                 │
│                     ▼                               │
│            ┌──────────────────┐                     │
│            │ Input Normalize  │                     │
│            │ → ControlDelta   │                     │
│            └────────┬─────────┘                     │
│                     ▼                               │
│            ┌──────────────────┐                     │
│            │   Slew Bank      │                     │
│            │ (integer-only)   │                     │
│            └────────┬─────────┘                     │
│                     ▼                               │
│            ┌──────────────────┐                     │
│            │  Binding Table   │                     │
│            │ (pure function)  │                     │
│            └────────┬─────────┘                     │
│                     ▼                               │
│         ┌────────────────────────┐                  │
│         │  FrameModifier         │                  │
│         │  (continuous controls) │                  │
│         └────────────────────────┘                  │
└─────────────────────────────────────────────────────┘
```

**Canonical Control IDs:**
- TIMBRE → FamilyID.COLOR scale
- TEXTURE → FamilyID.LOCAL scale
- DYNAMICS → gain_scale_q15
- MORPH → tree_blend_q15
- CHAOS → FamilyID.CATASTROPHE scale
- ATTACK/DECAY/SUSTAIN/RELEASE → ADSR parameters

**Reference:** [WSE-STD-COCKPIT-ENGINE.md](standards/control/WSE-STD-COCKPIT-ENGINE.md)

### 5.2 ControlFrame ABI

**Purpose:** Cockpit → VoiceFrameEmitter communication

**Contents:**
- Family scale factors (8 × Q15)
- Tree blend (Q15)
- Gain scale (Q15)
- ADSR params (blocks, Q15)
- Pitch offset (cents)

**Reference:** [WSE-STD-CONTROLFRAME.md](standards/control/WSE-STD-CONTROLFRAME.md)

### 5.3 Parameter Registry

**Domains:**

| Domain | Scope | Examples |
|--------|-------|----------|
| VOICE | Per-voice | gain_q15, pan_q15, activity_q15 |
| SYMBOL | Per-operator | strength, target |
| WAVELET | Reconstruction | tree_id, blend |
| MATERIAL | Per-tree | base coefficients |
| SAFETY | Global | budget, clamp thresholds |

**Reference:** [WSE-STD-CONTROLREGISTRY.md](standards/control/WSE-STD-CONTROLREGISTRY.md)

---

## 6. CCS Architecture & Current Status

### 6.1 Design Goals (WSE-STD-ARCH-CCS)

**Core Principle:**
> No FWPT in the voice path. Coefficients are SYNTHESIZED, not ANALYZED.

**Architecture:**
- Phase accumulator → CoeffTable lookup → 256 coefficients
- IDWPT reconstructs target waveform directly
- Operators modify coefficients (not audio)
- Phase-continuous operation

### 6.2 Current Status: NOT FUNCTIONAL ❌

**Problem:** CCS implementation produces wrong frequencies with massive spectral contamination.

**Test Results (2026-01-05):**
```
LUT target frequency: 440.00 Hz
Single block peak frequency: 750.0 Hz    ← WRONG!
Multi-block peak frequency: 307.50 Hz    ← WRONG!
VM output fundamental: 311.25 Hz         ← WRONG!
Non-harmonic peaks: 1431                 ← SEVERE CONTAMINATION
```

### 6.3 Root Cause Analysis

**The Problem:**
FWPT decomposes audio into subband **ENERGY**, not frequency content. When IDWPT reconstructs from these coefficients, it produces audio at the filterbank's intrinsic center frequencies (750 Hz = 48000/64), **not** the original signal frequency.

**Current (BROKEN) Implementation:**
```
Audio(440 Hz) → FWPT → coefficients → table[phase] → IDWPT → Audio(750 Hz!)
```

**Key Insight:**
> Wavelet coefficients are NOT spectral snapshots. They are TIME-DOMAIN SUBBAND SIGNALS.

Using FWPT for offline table generation IS still "reanalysis" - it doesn't preserve frequency.

### 6.4 Fix Roadmap

**Option A: Disable CCS (Short Term)**
- Set `use_coeff_osc = False` in decode.py
- Legacy path has combing but produces CORRECT frequencies
- Immediate fix, audible

**Option B: Fix Table Generation (Medium Term)**
- Replace FWPT-based table generation with iterative optimization
- Compute coefficients that reconstruct to target waveforms
- Requires understanding IDWPT synthesis filterbank behavior

**Option C: Full Streaming Rewrite (Long Term)**
- Implement "Maximal Solution": true streaming coefficient-domain oscillator
- Polyphase IDWPT (streaming synthesis filterbank)
- Offline subband LUT generation (NOT FWPT-based)
- True streaming coefficient evolution

**Reference:** [CCS-STATUS.md](status-reports/implementation/CCS-STATUS.md)

---

## 7. Budget & Degradation System

### 7.1 K Ladder (WSE-STD-DEGRADE-v0)

**Canonical Rungs:**
```
K = (32, 24, 16, 12, 8, 4, 1)
```

**Band Structure (32 nodes across 6 frequency bands):**
- Band 0 (DC-150 Hz): 1 node (8 coeffs)
- Band 1 (150-300 Hz): 2 nodes (16 coeffs)
- Band 2 (300-600 Hz): 4 nodes (32 coeffs)
- Band 3 (600-1200 Hz): 6 nodes (48 coeffs)
- Band 4 (1200-2400 Hz): 8 nodes (64 coeffs)
- Band 5 (2400-4800 Hz): 11 nodes (88 coeffs)

**Shrink Strategy:** Keep low-frequency bands first (perceptually important).

### 7.2 Tail Modes (4 modes)

| Mode | Op IDs | Operators | Cost (K=32) | Nonlinear |
|------|--------|-----------|-------------|-----------|
| FULL | [2,11,23] | FineClamp, NormalizeScale, SheafProject | 1136 | Yes |
| MID | [2,11] | FineClamp, NormalizeScale | 560 | Yes |
| MIN | [2] | FineClamp only | 272 | No |
| EMERGENCY | [19] | EmergencySimplify | ~100 | No |

**Downgrade Path:**
```
FULL (1136 ticks, sqrt)
  ↓
MID (560 ticks, sqrt)
  ↓
MIN (272 ticks, linear)
  ↓
EMERGENCY (100 ticks, linear)
```

### 7.3 Degradation Ladder Steps (7 steps in order)

1. **Drop INJECT events** (highest priority)
2. **Drop PERMUTE backbone ops**
3. **Drop LOWRANK backbone ops**
4. **Reduce K** (next lower rung)
5. **Drop remaining backbone ops**
6. **Downgrade tail** (FULL→MID→MIN→EMERGENCY)
7. **Emergency mode** (K=1, tail=EMERGENCY, no ops)

### 7.4 Conformance Status

✅ **FULLY CONFORMANT** - 11/11 requirements PASS

**Implementation highlights:**
- Deterministic execution (no floats, no randomness)
- Bounded work guarantee (≤128 iterations)
- Perceptually-informed degradation (low frequencies first)
- Hardware-realistic budgets (100 MHz clock, 64-sample blocks)

**Reference:** [BUDGET-CONFORMANCE-SUMMARY.md](status-reports/reviews/BUDGET-CONFORMANCE-SUMMARY.md)

---

## 8. Data Structures

### 8.1 IntentFrame (GestureTime)

Represents continuous musical intent. Contains **no timing logic**.

```
IntentFrame {
    note: u8
    velocity: float [0.0, 1.0]
    pitch_bend: float [-1.0, 1.0]
    mod_wheel: float [0.0, 1.0]
    expression: float [0.0, 1.0]
    gate: bool
    sustain_pedal: bool
    family_weights: map<string, float>
    regime_hint: optional<int>
}
```

**Properties:**
- May arrive asynchronously
- Is NOT executable
- Carries no block index

### 8.2 VoiceFrame (BlockTime)

Represents time-resolved, executable intent. **136 bytes exact.**

```
VoiceFrame {
    // Header (16 bytes)
    magic: u16 = 0x5646 ("VF")
    version: u8 = 1
    packet_len: u8 = 136
    program_id: u16
    voice_id: u16
    frame_seq: u32
    reject_code: u16
    hdr_pad: u16 = 0

    // Control (16 bytes)
    flags: u16 (gate, gate_rise, gate_fall)
    regime_id: u8
    degradation_level: u8
    note_u8: u8
    pad0: u8 = 0
    cents_i16: i16
    gain_q15: u16
    activity_q15: u16
    pan_q15: i16
    sendA_q15: u16
    sendB_q15: u16

    // Family weights (16 bytes)
    w_family_q15[8]: u16[8]

    // Tree selection (6 bytes)
    tree_a_id: u8
    tree_b_id: u8
    tree_blend_q15: u16
    event_mask: u8
    pad1: u8 = 0

    // Events (80 bytes)
    event_strength_q15[8]: u16[8]
    event_payload_u64[8]: u64[8]
}
```

**Properties:**
- Emitted at BlockTime resolution
- Contains all timing decisions
- IS executable
- Self-contained command for downstream engines

**Reference:** [design/core-system/04 - Wire Contract v0b, VoiceFrame Packet v0b.md](design/core-system/04%20-%20Wire%20Contract%20v0b,%20VoiceFrame%20Packet%20v0b.md)

---

## 9. Voice Classification & Scheduling

### 9.1 Voice Classification (WSE-STD-050)

Classification is recomputed **every block**.

**Foreground** if ANY of:
1. Gate is active (note on)
2. IntentFrame received this block
3. Energy > ENERGY_EPS (33 in Q15, ≈ -60 dBFS)
4. Voice is marked SOLO or ARMED

**Cloud** otherwise.

### 9.2 Skip Semantics

| Voice Class | Skipped? | ADSR | Regime | PRNG | Energy | Audio |
|-------------|----------|------|--------|------|--------|-------|
| Foreground | Yes | Frozen | Frozen | Frozen | Frozen | Held |
| Cloud | Yes | Frozen | Frozen | Frozen | Decays | Held |
| Any | No | Advances | Advances | Advances | Updated | Recomputed |

**Freeze Semantics (Foreground):**
- Time pauses for exactly one block
- All state preserved
- No decay, no events, no timing advancement

**Decay Semantics (Cloud):**
- Energy multiplied by 0.98 per block (Q15: 32112)
- All other state frozen
- No PRNG advancement

### 9.3 Rolling Cursor Scheduler

```
cursor = 0  // Persists across blocks

schedule_block(budget, cost_per_voice):
    voice_list = sorted(active_voices)
    start_pos = cursor % len(voice_list)

    for each voice starting at start_pos:
        if budget >= cost_per_voice:
            service(voice)
            budget -= cost_per_voice
        else:
            skip(voice)

    cursor = (start_pos + num_serviced) % len(voice_list)
```

---

## 10. PRNG System (WSE-STD-040)

### 10.1 PCG-32 Generator

- 64-bit internal state
- 32-bit output
- Deterministic across platforms (Python, FPGA)

### 10.2 Per-Voice Isolation

- Each voice owns independent PRNG state
- Reordering/skipping voices does NOT change results
- PRNG seeded on voice allocation via:

```
seed = HASH64(global_seed, voice_id, note_id, preset_id)
```

### 10.3 Advancement Rules

- PRNG advances ONLY when voice is serviced for a block
- Skipped/frozen voices do NOT advance PRNG
- Each operator documents fixed PRNG_DRAWS_PER_CALL

---

## 11. Tower Architecture (WSE-STD-070)

### 11.1 Tower Levels

| Level | Name | Editor Access | Invariant |
|-------|------|---------------|-----------|
| L0 | Patch | MIDI maps, gains | Always playable |
| L1 | Family | Regime hints | Recognizable class |
| L2 | Instrument | Params, bounds | Identity preserved |
| L3 | Cockpit | Weights, envelopes | No destructive edits |
| L4 | Forge | Graph edits | Always projectable |
| L5 | Lab | New ops/trees | Must compile down |

### 11.2 Completion Direction

Completion proceeds **upward**: Lk → Lk+1 → ... → L5

Missing layers filled with deterministic defaults:
- Missing L1: Infer family from tree_id, regimes = {STABLE}
- Missing L2: ADSR = neutral, operators = SAFE_SET
- Missing L3: Conservative bounds, nominal budget
- Missing L4: Graph edits forbidden
- Missing L5: No-op

### 11.3 Projection Direction

Projection proceeds **downward**: Lj → Lj-1 → ... → L0

- Removes all data above target level
- Re-applies completion rules
- MUST preserve playability

---

## 12. Determinism Guarantees

### 12.1 Core Invariants

1. **Same preset + same IntentFrame stream ⇒ same VoiceFrames**
2. **Same VoiceFrames ⇒ same Symbol execution**
3. **Same blocks ⇒ same Wavelet output** (modulo numeric spec)
4. **Tower completion MUST NOT consult runtime state**
5. **Scheduling changes MUST NOT change musical timing**

### 12.2 Replay Requirements

Given identical:
- Initial state
- Input IntentFrames
- Budget sequence

Then: Audio output MUST be bit-identical, regardless of which voices were skipped in intermediate blocks.

### 12.3 Prohibited Patterns (Outside Voice Engine)

- `if block % N == 0`
- Event scheduling
- ADSR logic
- Phase detection
- Time-based branching

**Violation is a spec breach, not a style issue.**

---

## 13. Fixed-Point Arithmetic (WSE-STD-001)

### 13.1 Q15 Format

- Signed 16-bit: [-32768, 32767]
- Represents [-1.0, ~1.0)
- Scale factor: 32768

### 13.2 Operations

```
q15_mul(a, b) = (a * b + 16384) >> 15  // Round-to-nearest
q15_add(a, b) = saturate(a + b)
q15_sub(a, b) = saturate(a - b)
```

### 13.3 PRNG to Q15 Conversion

```
uint32 r = pcg32();
int16 q15 = (r >> 17) - 0x4000;  // Range: [-16384, 16383]
```

---

## 14. Implementation Modules

### 14.1 Python Reference Implementation

| Module | Purpose | Standard |
|--------|---------|----------|
| `prng.py` | PCG-32, per-voice PRNG | WSE-STD-040 |
| `degradation.py` | Voice classification, skip semantics | WSE-STD-050 |
| `time_ownership.py` | Time domain validation | WSE-STD-060 |
| `presets.py` | Tower layers, completion, projection | WSE-STD-070 |
| `parameter_registry.py` | Parameter model, validation | Closure Appendix |
| `frames.py` | VoiceFramePacketV0b (136 bytes) | Wire Contract |
| `symbol_engine.py` | Operator execution | WSE-STD-030 |
| `wavelet_engine.py` | Inverse transform | WSE-STD-120 |
| `scheduler.py` | Block scheduling, degrade | WSE-STD-050 |
| `material_engine.py` | Analog processing layer | WSE-STD-MATERIAL-v0 |
| `cockpit_engine.py` | Control plane | WSE-STD-COCKPIT-ENGINE |

### 14.2 Test Coverage

| Test File | Tests | Standard |
|-----------|-------|----------|
| `test_prng.py` | 45 | WSE-STD-040 |
| `test_degradation.py` | 39 | WSE-STD-050 |
| `test_time_ownership.py` | 40 | WSE-STD-060 |
| `test_presets.py` | 52 | WSE-STD-070 |
| `test_parameter_registry.py` | 45+ | Closure Appendix |
| `test_budget_ladder.py` | 80+ | WSE-STD-DEGRADE-v0 |

---

## 15. FPGA Mapping

### 15.1 Engine Boundaries

```
┌─────────────────────────────────────────────────────────────────┐
│                         FPGA Fabric                              │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Voice     │    │   Symbol    │    │   Wavelet   │         │
│  │   Engine    │───►│   Engine    │───►│   Engine    │         │
│  │  (Timing)   │    │ (Operators) │    │  (IDWT)     │         │
│  └─────────────┘    └─────────────┘    └──────┬──────┘         │
│                                                │                │
│                                                ▼                │
│                                        ┌─────────────┐          │
│                                        │  Material   │          │
│                                        │  Engine     │          │
│                                        └──────┬──────┘          │
│                                                │                │
│        ┌───────────────────────────────────────┘                │
│        │                                                        │
│        ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Shared Memory                         │   │
│  │  - Coefficient buffers                                   │   │
│  │  - PRNG state (per voice)                               │   │
│  │  - Voice state                                          │   │
│  │  - Material state (per channel)                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 15.2 Critical Paths

- Voice Engine → VoiceFrame generation: Must complete before block deadline
- Symbol Engine → Operator application: Budget-constrained
- Wavelet Engine → IDWPT: Fixed latency per tree depth
- Material Engine → Analog processing: 64-sample block budget

### 15.3 Memory Layout

| Buffer | Size | Access Pattern |
|--------|------|----------------|
| Coefficients | N × 2 bytes | Sequential write, random read |
| PRNG states | V × 16 bytes | Per-voice access |
| Voice states | V × 64 bytes | Per-voice access |
| Material states | 2 × 150 bytes | Per-channel access |
| OLA buffers | 128 samples × V | Circular |

---

## 16. Implementation Roadmap

### 16.1 Immediate (Q1 2026)

**Priority 1: Fix CCS**
- Root cause: FWPT-based table generation doesn't preserve frequency
- Solution: Iterative optimization or streaming rewrite
- Target: Correct frequency reproduction
- Reference: [CCS-STATUS.md](status-reports/implementation/CCS-STATUS.md)

**Priority 2: CMMM Phase 1**
- USS (Universal Symbolic Semantics) + Preset standard
- Reference: [WSE-STD-CMM-ROADMAP-v0.md](standards/architecture/WSE-STD-CMM-ROADMAP-v0.md)

### 16.2 Near-term (Q2 2026)

**CMMM Phase 2:**
- SUBTRACTIVE baseline
- Integration with Material Engine

**Material Engine Integration:**
- Validate Material Engine with real presets
- Performance profiling
- FPGA resource estimation

**FPGA Prototype:**
- Voice Engine core
- Symbol Engine (operator subset)
- Polyphase IDWPT

### 16.3 Long-term

**CMMM Phases 3-9:**
- Full operator family expansion
- Advanced catastrophe operators
- Phase plane navigation

**Full FPGA Implementation:**
- Complete operator set
- Material Engine hardware
- Real-time performance at target polyphony

---

## 17. Standards Cross-Reference

### 17.1 Core Standards

| Standard | Category | Location | Status |
|----------|----------|----------|--------|
| WSE-STD-001 | Numerics | [standards/core/](standards/core/) | FROZEN |
| WSE-STD-020 | Frame ABI | [standards/core/](standards/core/) | FROZEN |
| WSE-STD-030 | Symbol Engine | [standards/core/](standards/core/) | FROZEN |

### 17.2 CCS Standards

| Standard | Category | Location | Status |
|----------|----------|----------|--------|
| WSE-STD-ARCH-CCS | Architecture | [standards/ccs/](standards/ccs/) | FROZEN |
| WSE-STD-070A-CCS-POLYPHASE-IDWPT | Polyphase | [standards/ccs/](standards/ccs/) | FROZEN |
| WSE-STD-210 | Oscillator | [standards/ccs/](standards/ccs/) | FROZEN |
| WSE-STD-120 | Polyphase Upgrade | [standards/ccs/](standards/ccs/) | FROZEN |

### 17.3 Control Standards

| Standard | Category | Location | Status |
|----------|----------|----------|--------|
| WSE-STD-COCKPIT-ENGINE | Control Plane | [standards/control/](standards/control/) | FROZEN |
| WSE-STD-CONTROLFRAME | Control ABI | [standards/control/](standards/control/) | FROZEN |
| WSE-STD-CONTROLREGISTRY | Parameter Registry | [standards/control/](standards/control/) | FROZEN |
| WSE-STD-PRESET-v0 | Presets | [standards/control/](standards/control/) | FROZEN |

### 17.4 Budget Standards

| Standard | Category | Location | Status |
|----------|----------|----------|--------|
| WSE-STD-DEGRADE-v0 | Degradation | [standards/budget/](standards/budget/) | FROZEN |
| WSE-STD-MATERIAL-v0 | Material Engine | [standards/budget/](standards/budget/) | FROZEN |

### 17.5 Runtime Standards

| Standard | Category | Location | Status |
|----------|----------|----------|--------|
| WSE-STD-040 | PRNG | [standards/runtime/](standards/runtime/) | FROZEN |
| WSE-STD-050 | Degradation Semantics | [standards/runtime/](standards/runtime/) | FROZEN |
| WSE-STD-060 | Time Ownership | [standards/runtime/](standards/runtime/) | FROZEN |
| WSE-STD-VM-HYBRID | VM | [standards/runtime/](standards/runtime/) | FROZEN |

### 17.6 Architecture Standards

| Standard | Category | Location | Status |
|----------|----------|----------|--------|
| WSE-ARCHITECTURE-v0 | Architecture (v0) | [standards/architecture/](standards/architecture/) | FROZEN |
| WSE-STD-070 | Tower Architecture | [standards/architecture/](standards/architecture/) | FROZEN |
| WSE-STD-CMM-ROADMAP-v0 | CMMM Roadmap | [standards/architecture/](standards/architecture/) | FROZEN |

---

## 18. Known Issues & Active Work

### 18.1 CCS Implementation

**Status:** ❌ NOT FUNCTIONAL

**Issue:** Wrong frequencies (750 Hz vs target 440 Hz) with severe spectral contamination (1431 non-harmonic peaks)

**Root Cause:** FWPT-based table generation decomposes into subband energy, not frequency-preserving coefficients. IDWPT reconstructs at filterbank center frequencies.

**Impact:** CCS path disabled, legacy path in use (correct frequencies but with combing artifacts)

**Fix Options:**
1. Iterative optimization for table generation
2. Analytic filterbank response computation
3. Full streaming rewrite (maximal solution)

**Reference:** [CCS-STATUS.md](status-reports/implementation/CCS-STATUS.md)

### 18.2 Active Development

**CMMM Phase 1:**
- USS (Universal Symbolic Semantics)
- Preset standard development
- Integration with Cockpit Engine

**Material Engine:**
- Validation with real presets
- Performance profiling
- FPGA resource planning

---

## 19. Closure Statement

After WSE-STD-070, the system is **architecture complete**.

What remains is:
- CCS implementation fixing (frequency preservation)
- CMMM phase execution
- FPGA prototyping
- Musical taste development

The structure is frozen. You are no longer inventing architecture — you are **inhabiting it**.

---

## Appendix A: Quick Reference

### A.1 Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| BLOCK_SIZE | 64 samples | Fundamental time unit |
| SAMPLE_RATE | 48000 Hz | Audio sample rate |
| BLOCK_RATE | 750 Hz | Block processing rate |
| ENERGY_EPS_Q15 | 33 | Cloud threshold (~-60 dBFS) |
| CLOUD_DECAY_Q15 | 32112 | 0.98 decay factor |
| RETIRE_BLOCKS | 8 | Blocks before retirement |
| Q15_MAX | 32767 | Maximum Q15 value |
| Q15_MIN | -32768 | Minimum Q15 value |

### A.2 Safe Operator Set

```
Identity, Tilt, FineClamp, IntraMix,
DiffuseUp, DiffuseDown, NormalizeScale,
GluePair, SheafProject
```

### A.3 Prohibited Patterns

```python
# FORBIDDEN outside Voice Engine:
if block % N == 0:       # Block modulo
if phase == X:           # Phase detection
adsr(attack, decay):     # ADSR logic
if elapsed > threshold:  # Time branching
```

---

## Appendix B: Navigation Guide

### Standards by Topic

- **Core Foundations:** [standards/core/](standards/core/)
- **CCS Architecture:** [standards/ccs/](standards/ccs/)
- **Control Plane:** [standards/control/](standards/control/)
- **Budget System:** [standards/budget/](standards/budget/)
- **Wavelet System:** [standards/wavelet/](standards/wavelet/)
- **Runtime:** [standards/runtime/](standards/runtime/)
- **Architecture:** [standards/architecture/](standards/architecture/)
- **Testing:** [standards/testing/](standards/testing/)

### Status Reports

- **Implementation Status:** [status-reports/implementation/](status-reports/implementation/)
- **Testing Status:** [status-reports/testing/](status-reports/testing/)
- **Reviews:** [status-reports/reviews/](status-reports/reviews/)

### Design Documents

- **Core System:** [design/core-system/](design/core-system/)
- **CMMM Design:** [design/cmmm-infopacket/](design/cmmm-infopacket/)

---

*Document generated from frozen WSE specifications v0 + current implementation status as of 2026-01-06*
