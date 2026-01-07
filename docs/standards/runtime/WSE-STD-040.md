# **WSE-STD-040 — Deterministic Pseudorandomness (PRNG) Standard**

**Status:** Frozen (v0)
**Applies to:** Voice Engine, Symbol Engine (indirectly), Operator Library
**Does NOT apply to:** UI, Editor, Offline Analysis Tools

---

## 1. Purpose

This standard defines how pseudorandom values are generated, consumed, and sequenced in the Wavelet Synth Engine (WSE) such that:

* Python reference, FPGA implementation, and future simulators are **bit-exact reproducible**
* Reordering of voices, budget skipping, or cloud degradation **does not change results**
* Randomness can be *musically expressive* without becoming *architecturally unstable*

---

## 2. Scope and Non-Goals

### In scope

* Randomness used by:

  * Symbol operators (e.g. NOISE_BAND, SCATTER, QUASI_ROTATE)
  * Catastrophe operators
  * Stochastic modulation inside a block
* Replay determinism
* Voice isolation

### Explicitly out of scope

* UI randomness (visual effects)
* Offline analysis randomness
* “True” entropy sources
* Cryptographic security

---

## 3. Normative Architecture Rule

> **PRNG state is owned by the Voice Engine and is per-voice.**

No other engine may own or advance random state.

---

## 4. PRNG Model (v0)

### 4.1 Generator Type

Each voice SHALL use an independent **PCG-32** generator.

Normative properties:

* 32-bit output
* 64-bit internal state
* Deterministic across platforms

Reason:

* Cheap in FPGA
* Excellent statistical quality for audio
* Well-documented reference behavior

---

### 4.2 Per-Voice PRNG State

Each active voice SHALL maintain:

```text
prng_state_u64
prng_stream_u64   // optional, but MUST be fixed if present
```

These values are part of **VoiceState**, not VoiceFrame.

---

### 4.3 Seeding Rules

On **voice allocation (note-on)**, the PRNG state MUST be seeded as:

```text
seed = HASH64(
    global_seed,
    voice_id,
    note_id,
    preset_id
)
```

Normative requirements:

* HASH64 MUST be deterministic and platform-independent
* XOR-mixing is acceptable
* Floating-point MUST NOT be used
* `global_seed` defaults to 0 if unspecified

---

## 5. PRNG Advancement Rules (Critical)

### 5.1 Block-Local Advancement Only

> **A voice’s PRNG state MUST advance only when that voice is serviced for a block.**

Consequences:

* Skipped voices do NOT advance PRNG
* Frozen voices do NOT advance PRNG
* Budget starvation does NOT desynchronize randomness

---

### 5.2 Draw Order Within a Voice

Within a single voice and block:

1. Operators consume PRNG **in execution order**
2. Each operator MUST document how many draws it consumes
3. Conditional draws MUST still consume a draw (discarded if unused)

Example (normative):

```text
Operator NOISE_BAND:
- Always consumes exactly 2 PRNG draws per invocation
```

This prevents divergence when parameters change.

---

### 5.3 Forbidden Behavior

The following are **MUST NOT**:

* Global PRNG shared across voices
* Advancing PRNG during scheduling decisions
* Advancing PRNG in Safety Engine
* Advancing PRNG during skipped blocks
* Advancing PRNG based on audio amplitude or runtime conditions

---

## 6. Foreground vs Cloud Interaction

This standard is explicitly compatible with WSE-STD-050.

### Foreground Voices

* PRNG is advanced normally when serviced
* If skipped: PRNG is frozen

### Cloud Voices

* If serviced: PRNG advances normally
* If skipped: PRNG is frozen
* Decay applied does NOT consume PRNG

> **Decay MUST be deterministic and non-random.**

---

## 7. Operator Contract (PRNG Consumption)

Each operator that uses randomness MUST declare:

```text
PRNG_DRAWS_PER_CALL = N   // integer, fixed
```

Rules:

* `N` MUST be constant
* Draws MUST occur even if strength = 0
* Draw results MAY be ignored, but NOT skipped

Example:

```text
Operator QUASI_ROTATE:
PRNG_DRAWS_PER_CALL = 1
```

---

## 8. Event Payload Interaction

PRNG MAY be influenced by event payloads **only via parameters**, not control flow.

Allowed:

* Payload selects target leaf, PRNG perturbs magnitude

Forbidden:

* Payload determines whether PRNG is drawn at all

---

## 9. Fixed-Point Conversion Rules

PRNG outputs SHALL be converted as follows:

```text
uint32 r = pcg32();
int16  q15 = (r >> 17) - 0x4000;
```

This yields:

* Symmetric signed Q1.15 noise
* Deterministic mapping

Other mappings MUST be documented per operator.

---

## 10. Replay & Debug Guarantees

If the following are identical:

* preset
* global_seed
* note sequence
* VoiceFrame sequence
* budget availability

Then the audio output MUST be bit-identical.

This is a **hard guarantee**.

---

## 11. Reserved Extensions (Non-Normative)

The following are reserved but NOT part of v0:

* Cross-voice correlated randomness
* Band-local PRNG streams
* Audio-rate PRNG
* Stochastic modulation sources exposed to UI

These MUST NOT be implemented without a new standard revision.

---

## 12. Summary (Normative Guarantees)

This standard guarantees:

* Deterministic randomness under time slicing
* Independence between voices
* FPGA ↔ Python equivalence
* Budget-safe stochastic operators
* Replayable performances

---

### **WSE-STD-040 is now CLOSED for v0.**

If you want, the next natural documents to write are:

* **WSE-STD-030 appendix** (tail safety predicate, now that PRNG is frozen)
* **WSE-STD-050** (degradation semantics)
* **WSE-STD-070** (Tower L0–L5 completion table)

Just tell me which one to seal next.
