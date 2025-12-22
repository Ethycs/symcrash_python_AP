Below is a **normative, drop-in spec** for **WSE-STD-050** written in the same style as your other standards. It is intentionally *boring*, explicit, and FPGA-friendly. No new machinery is introduced.

---

# **WSE-STD-050 — Degradation + Voice Scheduling Semantics (v0)**

## Status

**Normative. Frozen for v0.**

This standard defines how voices are scheduled, skipped, degraded, or preserved under compute pressure, and how those choices interact with determinism, musical continuity, and safety.

This document applies to **Voice Engine → Symbol Engine → Wavelet Engine** execution only. UI, editor, and analysis tools MUST conform to these semantics when simulating runtime behavior.

---

## 1. Scope and Goals

This standard exists to guarantee:

1. **Deterministic replay** under variable compute budgets
2. **Musical continuity** for active (“foreground”) voices
3. **Graceful degradation** for passive (“cloud”) voices
4. **FPGA-safe scheduling** with bounded state mutation
5. **No hidden time advancement** when work is skipped

This standard **does not** define *how much* budget exists — only *what happens* when budget is insufficient.

---

## 2. Definitions

### 2.1 Voice

A **voice** is a persistent stateful entity indexed by `voice_id`, owning:

* Voice Engine state (ADSR, regime, PRNG, etc.)
* Symbol state (coefficients, op toggles)
* Wavelet reconstruction state (OLA buffers, etc.)

### 2.2 Block

A **block** is the atomic execution unit (64 samples). All scheduling decisions occur at block boundaries.

### 2.3 Serviced vs Skipped

For a given block:

* **Serviced voice**: Symbol + Wavelet execution occurs
* **Skipped voice**: No Symbol or Wavelet execution occurs

---

## 3. Voice Classification (Foreground vs Cloud)

Each active voice MUST be classified **per block** into exactly one of the following classes:

### 3.1 Foreground Voice

A voice is **Foreground** if **any** of the following are true:

1. Voice has non-zero gate (note is active)
2. Voice received an IntentFrame update this block
3. Voice energy > `ENERGY_EPS` (implementation-defined, e.g. −60 dBFS)
4. Voice is explicitly marked `SOLO` or `ARMED` by control logic

### 3.2 Cloud Voice

A voice is **Cloud** if it is active but **none** of the Foreground conditions apply.

> **Normative rule:**
> Classification MUST be recomputed every block.
> A voice MAY transition between Foreground and Cloud at block boundaries only.

---

## 4. Scheduling Semantics

### 4.1 Global Scheduling Model

Each block, the scheduler operates as follows:

1. Reserve **mandatory tail budget** (see WSE-STD-030)
2. Iterate voices in a **rolling cursor order**
3. Attempt to service each voice until budget exhausted
4. Apply degradation rules to skipped voices

### 4.2 Ordering Guarantee

* Scheduler MUST use a rolling cursor (round-robin) over voices
* Cursor position MUST persist across blocks
* Voices MUST NOT be reordered based on priority alone

This guarantees fairness and long-term determinism.

---

## 5. Behavior When a Voice Is Skipped

### 5.1 Foreground Voice (Freeze Semantics)

If a **Foreground voice** is skipped:

**MUST:**

* Freeze all Voice Engine state

  * ADSR does NOT advance
  * Regime dwell counters do NOT advance
  * Phase/state machines do NOT advance
* Freeze Symbol state

  * No coefficient updates
  * No operator application
* Freeze PRNG

  * No random draws
* Preserve last audio output via OLA/state buffers

**MUST NOT:**

* Apply decay
* Emit events
* Advance timing counters

> **Interpretation:**
> Skipped Foreground voices behave as if time paused for exactly one block.

---

### 5.2 Cloud Voice (Decay Semantics)

If a **Cloud voice** is skipped:

**MUST:**

* Apply deterministic energy decay:

  ```
  energy ← energy × CLOUD_DECAY
  ```

  where `CLOUD_DECAY ∈ (0,1)` is implementation-defined but fixed (e.g. 0.98 per block)

* Freeze PRNG state

* Freeze regime transitions

* Freeze Symbol operator state

**MAY:**

* Reduce update rate further (multi-block skipping)

**MUST NOT:**

* Emit new events
* Advance PRNG
* Apply Symbol operators

> **Interpretation:**
> Cloud voices fade out *only* due to explicit decay, not time advancement.

---

## 6. Event Handling Under Skipping

### 6.1 Foreground Voices

* Events scheduled for a skipped Foreground voice:

  * **MUST be dropped**, not queued
  * Voice resumes with next block’s events

### 6.2 Cloud Voices

* Events MUST NOT be emitted
* No backlog or accumulation is permitted

> **Rationale:**
> Queuing events introduces nondeterministic bursts and breaks replay.

---

## 7. PRNG Determinism Rule

This standard REQUIRES:

* **Per-voice PRNG state**
* PRNG state advances **only if the voice is serviced**
* Skipped voices MUST NOT advance PRNG, regardless of class

This ensures:

* Deterministic replay under variable scheduling
* No dependence on global execution order

---

## 8. Interaction with Safety Tail

Skipped voices:

* Still pass through **Safety Tail** using their last valid audio state
* Tail operations MUST be idempotent (see WSE-STD-030)
* Tail cost is accounted globally, not per voice

---

## 9. Voice Retirement

A voice MAY be retired when:

* Classified as Cloud
* Energy < `RETIRE_EPS` for `N` consecutive blocks (implementation-defined)

Retirement MUST:

* Be deterministic
* Free all associated state
* Not emit audio discontinuities (OLA fade required)

---

## 10. Determinism Guarantees

Given:

* Identical initial state
* Identical input IntentFrames
* Identical budget sequence

Then:

> **The audio output MUST be bit-identical**, regardless of which voices were skipped in intermediate blocks.

This is the primary invariant of WSE-STD-050.

---

## 11. Summary Table

| Voice Class | Skipped? | ADSR     | Regime   | PRNG     | Energy  | Audio      |
| ----------- | -------- | -------- | -------- | -------- | ------- | ---------- |
| Foreground  | Yes      | Frozen   | Frozen   | Frozen   | Frozen  | Held       |
| Cloud       | Yes      | Frozen   | Frozen   | Frozen   | Decays  | Held       |
| Any         | No       | Advances | Advances | Advances | Updated | Recomputed |

---

## 12. Non-Goals (Explicitly Out of Scope)

This standard does **not** define:

* Budget sizes
* Cost values
* Priority heuristics
* Musical intent semantics
* Operator selection logic

Those are covered by other WSE standards.

---

## 13. Compliance

An implementation is **WSE-STD-050 compliant** if and only if:

1. Foreground voices never decay due to skipping
2. Cloud voices decay deterministically when skipped
3. PRNG never advances on skipped voices
4. Skipping does not change replay output

---

If you want, next I can:

* Write **WSE-STD-030 tail safety predicate** as formal postconditions
* Produce the **L0–L5 table** you requested (editor IR / engine / math / morphisms)
* Collapse **STD-020 + STD-050** into a single “Voice Runtime Contract” appendix

Just say which one you want sealed next.
