Below is a **clean, normative spec** you can drop into your standards tree.

I’m writing this as a **WSE document**, not commentary.
It freezes intent, architecture, invariants, and implementation guidance.

---

# **WSE-STD-120 — Polyphase Upgrade (Coefficient-Driven IDWPT Synthesis)**

**Status:** Frozen v1.0
**Applies to:** Wavelet Engine, Symbol Engine, Voice Engine
**Supersedes:** Carrier-centric IDWPT usage patterns
**Related:** WSE-STD-030 (Executor), WSE-STD-060 (Time Ownership), WSE-STD-040 (Wire Contract)

---

## 1. Purpose

This standard upgrades the Wavelet Engine architecture from **carrier-conditioned IDWPT** to **coefficient-driven polyphase synthesis**, eliminating block-boundary artifacts while preserving maximal expressive power.

The goal is to:

* Eliminate phase-dependent discontinuities
* Remove the need for safety-window compensation in synthesis
* Promote coefficients to first-class oscillator state
* Align IDWPT usage with its mathematical invariants
* Improve FPGA efficiency and determinism

---

## 2. Core Architectural Shift (Normative)

### 2.1 Old Model (Deprecated)

```
carrier_audio → FWPT → coeffs
coeffs × mask
→ IDWPT → wet_audio
mix(dry, wet)
```

**Problems (non-normative):**

* Phase-variant energy per block
* Comb filtering when mixed with dry
* Requires OLA or safety windows
* Violates critically sampled filterbank assumptions

---

### 2.2 New Model (Normative)

```
coeff_state (persistent)
→ IDWPT
→ audio
```

Optionally:

```
external_audio → FWPT → coeffs → CMMM → IDWPT
```

**Key rule (normative):**

> **IDWPT SHALL be used as a synthesis operator over persistent coefficient state.
> IDWPT SHALL NOT be driven by per-block reanalysis of a carrier unless explicitly in analysis mode.**

---

## 3. Polyphase Interpretation (Normative)

### 3.1 Definition

The Wavelet Engine SHALL be treated as a **tree-structured polyphase synthesis filterbank**, where:

* Each lifting stage is a 2-channel polyphase operator
* The full tree is a cascade of critically sampled filterbanks
* Perfect reconstruction holds **only if coefficient state is continuous**

### 3.2 Consequence

> **Coefficient continuity replaces window overlap as the primary regularity mechanism.**

No overlap-add or analysis window is required for synthesis.

---

## 4. Coefficient State Model (Normative)

### 4.1 Coefficient Oscillator State

Each voice SHALL maintain:

```
coeff_state[v][leaf][k]  // persistent Q1.15
```

Where:

* `leaf ∈ [0..L-1]` (canonical low→high order)
* `k ∈ [0..K-1]` (coeff index, typically K=8)

This state:

* Persists across blocks
* Evolves only via Symbol / CMMM operators
* Is never implicitly reset on note-on

---

### 4.2 Initialization

On voice allocation:

* coeff_state MAY be zeroed OR seeded from a table
* No reset SHALL occur on retrigger unless explicitly commanded

---

## 5. Role of CMMM (Normative)

### 5.1 CMMM as Oscillator Logic

CMMM SHALL be permitted to operate directly in coefficient space:

* Injection
* Diffusion
* Rotation
* Low-rank morph
* Catastrophe operators

This makes CMMM the **primary oscillator/dynamics engine**.

> **CMMM is no longer “modulation of a carrier” — it is the oscillator itself.**

---

### 5.2 Tables and External Inputs

Tables, envelopes, gestures, and PRNG:

* MAY inject energy into coefficient space
* SHALL NOT directly drive audio-domain oscillators unless in compatibility mode

---

## 6. IDWPT Execution Rules (Normative)

### 6.1 Execution

Per block, per voice:

```
audio_block = IDWPT(coeff_state)
```

IDWPT:

* Operates on current coeff_state snapshot
* Is stateless with respect to musical time
* Must be bit-exact given identical coeff_state

---

### 6.2 No Phase Compensation Required

Because coefficients are continuous:

* No phase reset occurs
* No block-boundary discontinuity occurs
* No safety window is required for synthesis correctness

---

## 7. Dry / Wet Semantics (Updated)

### 7.1 Redefinition

With this upgrade:

* **“Wet”** = IDWPT(coeff_state)
* **“Dry”** = optional legacy oscillator OR external audio

### 7.2 Mixing Rule (Normative)

If dry is present:

```
output = dry * (1 - η) + wet * η
```

Constraints:

* Dry and wet MUST be phase-coherent sources OR
* Dry MUST be decorrelated (external input, noise, etc.)

Recommended:

* Prefer **wet-only synthesis** for native instruments

---

## 8. Analysis Mode (Optional, Bounded)

### 8.1 Forward Transform Use

FWPT MAY be used only in **Analysis Mode**:

```
external_audio → FWPT → coeffs → (conditioning) → coeff_state
```

Rules:

* Analysis Mode SHALL be explicitly flagged
* Coeff_state SHALL remain continuous after conditioning
* Analysis is not required for normal synthesis

---

## 9. Safety Engine Interaction (Normative)

### 9.1 What Safety Still Does

Safety Engine SHALL remain responsible for:

* DC blocking
* Hard limiting
* Output OLA only when voices are allocated/deallocated
* Emergency simplify

### 9.2 What Safety SHALL NOT Do

Safety SHALL NOT:

* Compensate for IDWPT discontinuities
* Apply periodic windows during steady-state synthesis

---

## 10. FPGA Implications (Normative)

### 10.1 Benefits

This upgrade:

* Removes FWPT cost from synthesis path
* Eliminates OLA memory and multipliers
* Makes IDWPT purely feedforward
* Simplifies timing closure
* Improves determinism

### 10.2 Parallelism

Coefficient-driven IDWPT:

* Is trivially parallel per voice
* Allows multiple IDWPT cores with shared CMMM

---

## 11. Backward Compatibility (Optional)

A **Legacy Carrier Mode** MAY be retained:

* For testing
* For conditioning external audio
* For transitional development

But it SHALL NOT be the default path.

---

## 12. Invariants (Hard Rules)

1. **IDWPT is synthesis, not an effect**
2. **Coefficient state must be continuous**
3. **No implicit resets on note events**
4. **No per-block reanalysis of carrier in synthesis mode**
5. **No safety window required for correctness**

Violating any of the above voids this standard.

---

## 13. Summary (Non-Normative)

This upgrade:

* Fixes combing at the root
* Aligns math, architecture, and hardware
* Elevates CMMM to a true dynamical engine
* Turns IDWPT into a first-class oscillator
* Removes entire classes of compensation hacks

**This is not a workaround.
This is the correct use of polyphase filterbanks.**

---

If you want next, I can:

* Write **WSE-STD-071: Coefficient Oscillator Tables**
* Formalize **CMMM-as-Oscillator semantics**
* Provide **FPGA block diagrams** for this upgrade
* Give a **migration checklist** from current code

Just say which.
