# **WSE-STD-060 — Time Ownership & Temporal Responsibility**

**Status:** FROZEN (v0)
**Scope:** All engines, IRs, and execution layers
**Applies to:** IntentFrame, VoiceFrame, Symbol Engine, Wavelet Engine, Safety Engine, UI / MIDI ingress

---

## 1. Purpose

This standard defines **which subsystem owns musical time**, which subsystems are **time-blind**, and the **only legal way temporal dynamics may enter the execution pipeline**.

Its goal is to:

* Prevent accidental timing logic from leaking into execution engines
* Guarantee deterministic replay under scheduling changes
* Allow aggressive optimization, skipping, and parallelism without musical drift
* Make FPGA execution purely mechanical

---

## 2. Definitions

### 2.1 BlockTime (Normative)

* **BlockTime** is the sole execution time unit.
* One block corresponds to **exactly 64 audio samples**.
* All downstream engines operate **strictly at BlockTime resolution**.

> No engine downstream of the Voice Engine may observe sub-block time.

---

### 2.2 Musical Time Domains

The system distinguishes **three time domains**:

| Domain      | Resolution                | Owner                     |
| ----------- | ------------------------- | ------------------------- |
| GestureTime | asynchronous, continuous  | UI / MIDI / Sensors       |
| VoiceTime   | discrete, block-quantized | **Voice Engine**          |
| BlockTime   | discrete, block-exact     | Symbol / Wavelet / Safety |

Only **VoiceTime → BlockTime** conversion is permitted.

---

## 3. Time Ownership (Normative Rule)

### **Rule 060.1 — Single Owner of Musical Time**

> **The Voice Engine is the sole owner of musical time.**

This includes (but is not limited to):

* ADSR envelopes (amplitude, tree, tilt, etc.)
* Note lifecycle (attack, sustain, release)
* Regime dwell counters
* Phase progression (fold → cusp → swallowtail, etc.)
* Periodic scheduling (every N blocks, odd/even blocks)
* Intent accumulation and decay
* Temporal hysteresis and memory

No other engine may track, infer, or simulate musical time.

---

## 4. Engine Responsibilities

### 4.1 Voice Engine (Time-Aware)

The Voice Engine **MUST**:

* Maintain all counters, clocks, and phase state
* Quantize all timing to BlockTime
* Emit exactly **one VoiceFrame per active voice per block**
* Decide **when** events occur, not how they execute
* Decide **what operators are intended**, not how they apply
* Be the only component allowed to say “every N blocks”

The Voice Engine **MAY**:

* Run at lower rates internally, as long as outputs are block-quantized
* Skip updates for cloud voices (subject to WSE-STD-050)

---

### 4.2 Symbol Engine (Time-Blind)

The Symbol Engine **MUST NOT**:

* Maintain counters or clocks
* Use `block_index`, `local_block`, or modulo logic
* Detect phases or regime progress
* Trigger events based on elapsed time

The Symbol Engine **MUST**:

* Treat each VoiceFrame as a complete, self-contained command
* Apply operators **exactly as instructed**
* Enforce legality, bounds, sparsity, and safety rules
* Remain stateless with respect to musical time

> The Symbol Engine may retain coefficient state, but **not temporal state**.

---

### 4.3 Wavelet Engine (Purely Mechanical)

The Wavelet Engine **MUST**:

* Accept coefficient states and reconstruction parameters
* Perform deterministic inverse transforms
* Perform numerical interpolation or crossfade when instructed
* Never inspect or infer musical structure

The Wavelet Engine **MUST NOT**:

* Observe note on/off
* Contain envelopes
* Schedule transitions
* Trigger events

---

### 4.4 Safety Engine (Block-Shaped Only)

The Safety Engine **MUST**:

* Enforce block-local invariants (see WSE-STD-030 tail predicate)
* Apply OLA, de-click, slew, limiter, DC correction
* Be idempotent with respect to musical meaning

The Safety Engine **MUST NOT**:

* Detect musical phases
* Alter regime state
* Introduce new events

---

## 5. IntentFrame → VoiceFrame Boundary

### Rule 060.2 — Intent is Time-Free, Voice is Time-Full

* **IntentFrame** represents *continuous musical intent*.
* IntentFrames may arrive asynchronously.
* IntentFrames contain **no timing logic**.

The Voice Engine:

* Samples / slews IntentFrame into VoiceTime
* Integrates intent over time
* Emits **VoiceFrame** at BlockTime resolution

> IntentFrame is not executable.
> VoiceFrame is executable.

---

## 6. Prohibited Patterns (Hard Errors)

The following patterns are **explicitly forbidden**:

* `if block % N == 0` outside Voice Engine
* Event scheduling inside Symbol Engine
* ADSR logic inside Wavelet or Symbol Engine
* Phase detection inside operator code
* Time-based branching inside Safety Engine

Violation of any of these is a **spec breach**, not a style issue.

---

## 7. Determinism Guarantee

### Rule 060.3 — Scheduling Invariance

If:

* VoiceFrames are identical
* Numeric rules (WSE-STD-001) are followed

Then:

> Reordering, skipping, or parallelizing downstream engines **MUST NOT** change musical timing or structure.

This is the primary reason for strict time ownership.

---

## 8. Rationale (Non-Normative)

This split enables:

* FPGA as an “automatic transmission”
* Aggressive optimization (voice skipping, cloud rendering)
* Bit-exact Python ↔ FPGA correspondence
* Clean Tower L0–L5 separation
* Future polyphony scaling without musical drift

---

## 9. Summary (Normative)

* Musical time lives **only** in the Voice Engine
* All other engines are **time-blind**
* All timing is quantized to BlockTime
* VoiceFrame is the only legal carrier of time-resolved intent
* Violations are structural errors, not tuning mistakes

---

If you want, the **next logical spec** to finalize is:

* **WSE-STD-030 Appendix A:** Formal Tail Safety Predicate (with equations), or
* **WSE-STD-070:** L0–L5 Tower Table with editor primitives and projection maps

Just tell me which one to lock next.
