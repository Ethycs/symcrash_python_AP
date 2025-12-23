# Wavelet Synth Engine — Systems Architecture v0

**Status:** FROZEN
**Date:** 2025-12-22
**Scope:** Complete architecture specification for Python reference implementation and FPGA target

---

## Executive Summary

The Wavelet Synth Engine (WSE) is a deterministic, real-time audio synthesis system built on wavelet packet decomposition and catastrophe-theoretic operator families. This document describes the frozen v0 architecture from a systems perspective, covering data flow, timing ownership, execution semantics, and determinism guarantees.

The architecture is designed for:
- **Bit-exact reproducibility** between Python simulation and FPGA hardware
- **Deterministic replay** regardless of scheduling decisions
- **Graceful degradation** under compute pressure
- **Clean separation** between musical intent and mechanical execution

---

## 1. System Overview

### 1.1 High-Level Architecture

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
│  │   Symbol    │          │   Wavelet   │          │   Safety    │         │
│  │   Engine    │    ───►  │   Engine    │    ───►  │   Engine    │         │
│  │ (Operators) │          │  (Inverse)  │          │  (OLA/Tail) │         │
│  └─────────────┘          └─────────────┘          └─────────────┘         │
│                                                            │                │
└────────────────────────────────────────────────────────────┼────────────────┘
                                                             │
                                                             ▼
                                                    ┌───────────────┐
                                                    │  Audio Output │
                                                    │  (64 samples) │
                                                    └───────────────┘
```

### 1.2 Core Standards

| Standard | Title | Purpose |
|----------|-------|---------|
| WSE-STD-001 | Numeric Determinism | Fixed-point Q15/Q8, rounding rules |
| WSE-STD-030 | Symbol Engine Executor | Operator scheduling, tail predicate |
| WSE-STD-040 | Deterministic PRNG | PCG-32, per-voice isolation |
| WSE-STD-050 | Degradation Semantics | Freeze/Decay, voice scheduling |
| WSE-STD-060 | Time Ownership | Single owner of musical time |
| WSE-STD-070 | Presets & Tower | L0-L5 layers, completion rules |

---

## 2. Time Domains

### 2.1 Three Time Domains (WSE-STD-060)

| Domain | Resolution | Owner | Characteristics |
|--------|------------|-------|-----------------|
| **GestureTime** | Asynchronous, continuous | UI/MIDI/Sensors | Human input, variable rate |
| **VoiceTime** | Discrete, block-quantized | Voice Engine | Musical decisions, state machines |
| **BlockTime** | Discrete, block-exact | Symbol/Wavelet/Safety | Mechanical execution, 64 samples |

### 2.2 Time Flow

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

### 2.3 BlockTime Definition

- **One block = exactly 64 audio samples**
- At 48kHz: 1 block ≈ 1.33ms
- All downstream engines operate strictly at BlockTime resolution

---

## 3. Engine Responsibilities

### 3.1 Voice Engine (Time-Aware)

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

### 3.2 Symbol Engine (Time-Blind)

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

### 3.3 Wavelet Engine (Purely Mechanical)

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

### 3.4 Safety Engine (Block-Shaped Only)

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

## 4. Data Structures

### 4.1 IntentFrame (GestureTime)

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

### 4.2 VoiceFrame (BlockTime)

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

### 4.3 VoiceState (Per-Voice)

```
VoiceState {
    voice_id: int
    voice_class: VoiceClass (FOREGROUND | CLOUD)

    // Gate and activity
    gate: bool
    energy_q15: int
    activity_q15: int

    // Control flags
    solo: bool
    armed: bool
    intent_updated: bool

    // Timing state (frozen when skipped)
    adsr_phase: int
    adsr_time: int
    regime_id: int
    regime_dwell: int

    // PRNG state
    prng_state_u64: u64
    prng_stream_u64: u64

    // Retirement tracking
    below_retire_count: int
    retired: bool
}
```

---

## 5. Voice Classification & Scheduling

### 5.1 Voice Classification (WSE-STD-050)

Classification is recomputed **every block**.

**Foreground** if ANY of:
1. Gate is active (note on)
2. IntentFrame received this block
3. Energy > ENERGY_EPS (33 in Q15, ≈ -60 dBFS)
4. Voice is marked SOLO or ARMED

**Cloud** otherwise.

### 5.2 Skip Semantics

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

### 5.3 Rolling Cursor Scheduler

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

## 6. PRNG System (WSE-STD-040)

### 6.1 PCG-32 Generator

- 64-bit internal state
- 32-bit output
- Deterministic across platforms (Python, FPGA)

### 6.2 Per-Voice Isolation

- Each voice owns independent PRNG state
- Reordering/skipping voices does NOT change results
- PRNG seeded on voice allocation via:

```
seed = HASH64(global_seed, voice_id, note_id, preset_id)
```

### 6.3 Advancement Rules

- PRNG advances ONLY when voice is serviced for a block
- Skipped/frozen voices do NOT advance PRNG
- Each operator documents fixed PRNG_DRAWS_PER_CALL

### 6.4 Operator PRNG Draws

| Operator | Draws |
|----------|-------|
| NoiseBand | 2 |
| QuasiRotate | 1 |
| Scatter | 1 |
| GlobalPulse | 1 |
| CuspInjection | 2 |
| SwapBands | 1 |
| All others | 0 |

---

## 7. Symbol Engine Execution (WSE-STD-030)

### 7.1 Operator Classification

| Class | Cost | Examples |
|-------|------|----------|
| DIAG | 1N | Identity, Tilt, FineClamp |
| PERMUTE | 0.5N + 2N mem | SwapBands, RotateNodes |
| BANDED | 3N | IntraMix, DiffuseUp/Down |
| LOWRANK(r) | 2rN | LowRankMorph |

### 7.2 Mandatory Tail Operators

Executed in order after all other operators:

1. **FineClamp** - Bound coefficients to [-1, 1]
2. **NormalizeScale** - Ensure L2 norm ≤ budget
3. **GluePair** (optional) - Smooth adjacent coefficients
4. **SheafProject** - Enforce consistency across tree
5. **EmergencySimplify** (conditional) - Reduce complexity if unstable

### 7.3 Safety Predicate

A coefficient state is **safe** if and only if:
- All coefficients in [-1, 1]
- L2 norm ≤ budget
- Sparsity within bounds
- No NaN/Inf values

---

## 8. Tower Architecture (WSE-STD-070)

### 8.1 Tower Levels

| Level | Name | Editor Access | Invariant |
|-------|------|---------------|-----------|
| L0 | Patch | MIDI maps, gains | Always playable |
| L1 | Family | Regime hints | Recognizable class |
| L2 | Instrument | Params, bounds | Identity preserved |
| L3 | Cockpit | Weights, envelopes | No destructive edits |
| L4 | Forge | Graph edits | Always projectable |
| L5 | Lab | New ops/trees | Must compile down |

### 8.2 Completion Direction

Completion proceeds **upward**: Lk → Lk+1 → ... → L5

Missing layers filled with deterministic defaults:
- Missing L1: Infer family from tree_id, regimes = {STABLE}
- Missing L2: ADSR = neutral, operators = SAFE_SET
- Missing L3: Conservative bounds, nominal budget
- Missing L4: Graph edits forbidden
- Missing L5: No-op

### 8.3 Projection Direction

Projection proceeds **downward**: Lj → Lj-1 → ... → L0

- Removes all data above target level
- Re-applies completion rules
- MUST preserve playability

### 8.4 Panic Projection

`RESET_TO_SAFE()`:
- Project to L2
- Enable only SAFE operator set
- Reset envelopes to neutral
- Clear all event policies

---

## 9. Determinism Guarantees

### 9.1 Core Invariants

1. **Same preset + same IntentFrame stream ⇒ same VoiceFrames**
2. **Same VoiceFrames ⇒ same Symbol execution**
3. **Same blocks ⇒ same Wavelet output** (modulo numeric spec)
4. **Tower completion MUST NOT consult runtime state**
5. **Scheduling changes MUST NOT change musical timing**

### 9.2 Replay Requirements

Given identical:
- Initial state
- Input IntentFrames
- Budget sequence

Then: Audio output MUST be bit-identical, regardless of which voices were skipped in intermediate blocks.

### 9.3 Prohibited Patterns (Outside Voice Engine)

- `if block % N == 0`
- Event scheduling
- ADSR logic
- Phase detection
- Time-based branching

**Violation is a spec breach, not a style issue.**

---

## 10. Parameter Registry

### 10.1 Parameter Domains

| Domain | Scope | Examples |
|--------|-------|----------|
| VOICE | Per-voice | gain_q15, pan_q15, activity_q15 |
| SYMBOL | Per-operator | strength, target |
| WAVELET | Reconstruction | tree_id, blend |
| MATERIAL | Per-tree | base coefficients |
| SAFETY | Global | budget, clamp thresholds |

### 10.2 Update Modes

- **ABS**: Set value directly
- **DELTA**: Add to current value (ignored if uninitialized)

### 10.3 Canonical Defaults

| Parameter | Default | Type |
|-----------|---------|------|
| regime_id | 2 (STABLE) | u8 |
| gain | Q15_ONE (32767) | Q15 |
| pan | 0 | Q15 |
| tree_blend | 0 | Q15 |
| event_mask | 0 | u8 |

---

## 11. Fixed-Point Arithmetic (WSE-STD-001)

### 11.1 Q15 Format

- Signed 16-bit: [-32768, 32767]
- Represents [-1.0, ~1.0)
- Scale factor: 32768

### 11.2 Operations

```
q15_mul(a, b) = (a * b + 16384) >> 15  // Round-to-nearest
q15_add(a, b) = saturate(a + b)
q15_sub(a, b) = saturate(a - b)
```

### 11.3 PRNG to Q15 Conversion

```
uint32 r = pcg32();
int16 q15 = (r >> 17) - 0x4000;  // Range: [-16384, 16383]
```

---

## 12. Implementation Modules

### 12.1 Python Reference Implementation

| Module | Purpose | Standard |
|--------|---------|----------|
| `prng.py` | PCG-32, per-voice PRNG | WSE-STD-040 |
| `degradation.py` | Voice classification, skip semantics | WSE-STD-050 |
| `time_ownership.py` | Time domain validation | WSE-STD-060 |
| `presets.py` | Tower layers, completion, projection | WSE-STD-070 |
| `parameter_registry.py` | Parameter model, validation | Closure Appendix |
| `frames.py` | VoiceFramePacketV0b (136 bytes) | Wire Contract |
| `symbol_engine.py` | Operator execution | WSE-STD-030 |
| `wavelet_engine.py` | Inverse transform | - |
| `scheduler.py` | Block scheduling, degrade | WSE-STD-050 |

### 12.2 Test Coverage

| Test File | Tests | Standard |
|-----------|-------|----------|
| `test_prng.py` | 45 | WSE-STD-040 |
| `test_degradation.py` | 39 | WSE-STD-050 |
| `test_time_ownership.py` | 40 | WSE-STD-060 |
| `test_presets.py` | 52 | WSE-STD-070 |
| `test_parameter_registry.py` | 45+ | Closure Appendix |

---

## 13. FPGA Mapping

### 13.1 Engine Boundaries

```
┌─────────────────────────────────────────────────────────────────┐
│                         FPGA Fabric                              │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Voice     │    │   Symbol    │    │   Wavelet   │         │
│  │   Engine    │───►│   Engine    │───►│   Engine    │         │
│  │  (Timing)   │    │ (Operators) │    │  (IDWT)     │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│        │                   │                   │                │
│        ▼                   ▼                   ▼                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Shared Memory                         │   │
│  │  - Coefficient buffers                                   │   │
│  │  - PRNG state (per voice)                               │   │
│  │  - Voice state                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 13.2 Critical Paths

- Voice Engine → VoiceFrame generation: Must complete before block deadline
- Symbol Engine → Operator application: Budget-constrained
- Wavelet Engine → IDWT: Fixed latency per tree depth

### 13.3 Memory Layout

| Buffer | Size | Access Pattern |
|--------|------|----------------|
| Coefficients | N × 2 bytes | Sequential write, random read |
| PRNG states | V × 16 bytes | Per-voice access |
| Voice states | V × 64 bytes | Per-voice access |
| OLA buffers | 128 samples × V | Circular |

---

## 14. Closure Statement

After WSE-STD-070, the system is **architecture complete**.

What remains is:
- Implementation refinement
- Performance tuning
- Musical taste development

The structure is frozen. You are no longer inventing architecture — you are **inhabiting it**.

---

## Appendix A: Quick Reference

### A.1 Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| BLOCK_SIZE | 64 samples | Fundamental time unit |
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

*Document generated from frozen WSE specifications v0*
