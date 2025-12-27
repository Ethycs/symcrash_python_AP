# **WSE-STD-080 — Continuity, Regularity, and Click Freedom**

**Status:** Normative
**Applies to:** Voice Engine, Symbol Engine, Wavelet Engine, Safety Engine
**Scope:** Defines the mandatory continuity guarantees of the WSE pipeline and assigns responsibility for eliminating audible clicks, pops, and discontinuities.

---

## 1. Purpose

This standard defines **what continuity means**, **where it is enforced**, and **what operations are permitted to create or eliminate discontinuities**.

The goal is:

> **No audible discontinuity may be produced by engine mechanics.
> All audible discontinuities must be intentional, bounded, and musically shaped.**

---

## 2. Definitions

### 2.1 Continuity Classes

Let ( x(t) ) be the output waveform.

* **C⁰ continuity**:
  ( x(t) ) is continuous (no step discontinuities).
* **C¹ continuity**:
  ( x(t) ) has continuous first derivative (no sharp corners).
* **Band-limited discontinuity**:
  A rapid change synthesized as a bounded, band-limited transient atom.

---

## 3. Time Domains (Normative)

| Domain         | Description                                                                   |
| -------------- | ----------------------------------------------------------------------------- |
| **BlockTime**  | 64-sample execution unit (sole time visible to Symbol/Wavelet/Safety engines) |
| **VoiceTime**  | Musical time measured in blocks (note lifecycle, envelopes, regimes)          |
| **SampleTime** | Audio-rate samples (Wavelet + Safety internal only)                           |

**Rule:**

> No engine upstream of the Wavelet Engine may observe SampleTime.

---

## 4. Responsibility Assignment (Freeze)

| Engine             | Owns                                      | Must Guarantee                         |
| ------------------ | ----------------------------------------- | -------------------------------------- |
| **Voice Engine**   | Musical intent, envelopes, note lifecycle | No hard steps in musical parameters    |
| **Symbol Engine**  | Operator algebra, coefficient legality    | No illegal coefficient states          |
| **Wavelet Engine** | Sample generation, state continuity       | C⁰ continuity across all state changes |
| **Safety Engine**  | Absolute bounds, seam masking             | No audible block seams, no clipping    |

---

## 5. Mandatory Continuity Rules

### 5.1 Absolute Prohibition

The following are **forbidden**:

* Resetting any state that influences audio output (phase, previous sample, filter memory) to zero or a constant without a continuity mechanism.
* Applying a step change to any parameter that directly influences the waveform.
* Introducing discontinuities inside a block without micro-regularization.

Violation of these rules is a **spec violation**.

---

## 6. Wavelet Engine Regularity Contract (Normative)

The Wavelet Engine **must** enforce C⁰ continuity at all times.

### 6.1 State Changes

Any change to:

* tree selection
* tree blend
* coefficient layout
* oscillator phase or accumulator
* regularity operator internal state
* gain applied post-synthesis

**MUST** be applied using **one of the following mechanisms**:

#### A. Micro-Interpolation (State-Space)

Interpolate state variables over ( N ) samples.

#### B. Micro-Crossfade (Audio-Space)

Render old and new states and crossfade over ( N ) samples.

#### C. Band-Limited Transition

Replace a step with a bounded transient atom.

> **Default:** ( N = 8 ) to ( 32 ) samples
> (Exact value is implementation-defined but MUST be fixed.)

---

### 6.2 Voice On / Retrigger

**Rule (Freeze):**

* `voice_on` MUST NOT reset internal state to zero.
* If a voice was previously inactive, the new state MUST be initialized relative to the **current output sample**, not absolute zero.
* Retriggering a voice MUST be distinguishable from starting a new voice.

---

## 7. Safety Engine Responsibilities

The Safety Engine operates **after** the Wavelet Engine and does **not fix internal discontinuities**.

### 7.1 Mandatory Safety Operations

The Safety Engine MUST enforce:

* Block seam masking (OLA / windowing)
* Peak bounding (limiting / soft clip)
* DC suppression
* NaN / overflow suppression

### 7.2 Non-Responsibilities

The Safety Engine MUST NOT:

* Attempt to repair mid-block discontinuities
* Introduce musical shaping
* Modify symbolic intent

---

## 8. Intentional Discontinuities (Effects)

Clicks, ticks, and glitches are **allowed only if intentional**.

### 8.1 Effect Click Rule

Any intentional discontinuity MUST be synthesized as:

* A **band-limited transient**
* With a **finite envelope**
* With bounded peak energy

This is an **effect**, not a failure.

---

## 9. Distinguishing Glide vs Crossfade

### 9.1 Glide (Pitch / Phase Continuity)

* Phase-continuous
* Frequency interpolates
* No amplitude interruption

### 9.2 Crossfade (State Change)

* Two states rendered
* Short overlap region
* Energy preserved

**Rule:**

> Pitch changes use **glide**.
> Structural changes use **crossfade or state interpolation**.

---

## 10. Canonical Anti-Click Allocation Table

| Click Source       | Resolution         | Engine           |
| ------------------ | ------------------ | ---------------- |
| Block seam         | OLA window         | Safety           |
| Tree switch        | Coefficient interp | Wavelet          |
| Note on/off        | Envelope ramp      | Voice            |
| State reset        | Forbidden          | —                |
| Parameter jump     | Micro-ramp         | Wavelet          |
| Intentional glitch | Transient atom     | Wavelet (effect) |

---

## 11. Formal Continuity Invariant

For all blocks ( b ):

[
|x_b[0] - x_{b-1}[63]| \le \epsilon
]

Where ( \epsilon ) is below audibility after Safety processing.

This invariant **must hold regardless of operator scheduling or budget degradation**.

---

## 12. Determinism Clause

All regularization mechanisms MUST be:

* Deterministic
* Free of global mutable state
* Independent per voice unless explicitly shared

---

## 13. Compliance Checklist

An implementation is compliant with WSE-STD-070 iff:

* No hard resets occur in Wavelet Engine
* All mid-block state changes are regularized
* Safety never masks Wavelet violations
* Intentional clicks are explicit effects
* Block seams are inaudible under silence tests

---

## 14. Summary (One Sentence)

> **The Wavelet Engine guarantees continuity, the Safety Engine guarantees boundedness, and no engine may rely on another to fix its discontinuities.**


