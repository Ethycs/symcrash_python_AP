Here is your note, reformatted for Obsidian with math expressions converted to $...$ or $$...$$, and with no HTML or extraneous formatting. There were no mermaid diagrams or HTML in your note, so no changes were needed there.

---

# Closure Appendix — v0.1

**Normative Completion, Defaults, and Determinism Rules**

This appendix defines the **last three required closures** for the architecture:

1. **Parameter Registry Semantics**
2. **Fixed-Point Arithmetic & Randomness Rules**
3. **Default Completion & Coercion Rules (L0 → L5)**

After this appendix, the system is considered **architecture-complete**.  
All remaining work is tuning, UI, and empirical refinement.

---

## A. Parameter Registry (Normative)

### A.1 Canonical Parameter Model

Every controllable quantity in the system is a **registered parameter**:

```
Parameter:
  param_id        : u16
  domain          : {VOICE, SYMBOL, WAVELET, MATERIAL, SAFETY}
  type            : {q15, q8, u8, bool, enum}
  min_value
  max_value
  default_value
  update_mode     : {ABS, DELTA, BOTH}
  persistence     : {PER_VOICE, GLOBAL, PROGRAM}
```

**Normative rule**

> All parameter updates (UART, MIDI, editor, automation) must resolve to this registry before affecting any engine.

---

### A.2 Update Semantics

Given parameter $p$ with current value $v$:

- **ABS update**

  $$
  v := \mathrm{clamp}(\mathrm{value}, \mathrm{min}, \mathrm{max})
  $$

- **DELTA update**

  $$
  v := \mathrm{clamp}(v + \Delta, \mathrm{min}, \mathrm{max})
  $$

**Error handling**

- DELTA applied to an uninitialized parameter is ignored until an ABS or keyframe arrives.
- ABS always initializes state.

---

### A.3 Tree Selection Rule (Restated)

- $tree\_id$ is the **primary selector**
- $tree\_beta$ (if present) is a modifier only
- Events always refer to **canonical leaf ordering** (low → high)

---

## B. Fixed-Point Arithmetic & Randomness (Normative)

This section ensures **bit-exact agreement** between:

- Python reference
- FPGA implementation
- Offline rendering
- Editor preview

---

### B.1 Fixed-Point Types

| Name | Meaning |
|---|---|
| q15 | signed int16, range $[-1.0, +0.99997]$ |
| q8 | signed int8, normalized |
| u8 | unsigned int8 |
| u16 | unsigned int16 |

All intermediate multiplies use **at least 32-bit signed accumulation**.

---

### B.2 Multiply + Round Rule (MANDATORY)

For two q15 values $a$, $b$:

```
prod = int32(a) * int32(b)     // full precision
temp = prod + 0x4000           // rounding bias
out  = temp >> 15
```

- Right shift is **arithmetic**
- Result is clamped to int16 range

This is **ties-to-nearest**, deterministic, and FPGA-safe.

---

### B.3 Saturation & Overflow

- All final writes clamp to parameter range.
- **No silent wraparound** is permitted.
- Overflow flags (if detected) may trigger safety predicates but must not alter math.

---

### B.4 Randomness (PRNG)

**Per-voice PRNG only** (frozen).

- Algorithm: PCG32 (or equivalent)
- Seeded at:
    - voice allocation
    - program change
- PRNG state advances **only when that voice is executed**

**Normative rule**

> Skipped voices must NOT advance PRNG state.

This guarantees deterministic replay under variable scheduling.

---

## C. Default Completion & Coercion Rules (MANDATORY)

This section guarantees that **partial specifications** at any tower level always compile into a valid executable VoiceFrame.

---

### C.1 Completion Principle

If a field is not specified at level $L_k$, it is completed using the nearest lower level default:

$$
L5 \rightarrow L4 \rightarrow L3 \rightarrow L2 \rightarrow L1 \rightarrow L0
$$

No field is ever left undefined.

---

### C.2 Canonical Defaults (v0.1)

| Field | Default |
|---|---|
| regime_id | REG_STABLE |
| gear_strength | 1.0 |
| gain | 1.0 |
| tree_id | TREE_DEFAULT |
| tree_blend | 0.0 |
| traction | 0.5 |
| backbone | empty |
| event_mask | 0 |
| event_strength | 0 |
| event_payload | 0 |

---

### C.3 Layer Completion Examples

#### L0 Patch

- Only tree_id + gain provided
- All operators empty
- No events
- Stable regime

#### L2 Instrument

- Regime graph defined
- Backbone defined
- No events → event_mask = 0

#### L4 Event Layer

- Event lanes override backbone **only for that block**
- Backbone resumes next block automatically

---

### C.4 Coercion Rules (Safety)

If any invalid or dangerous combination appears:

1. **Clamp numeric ranges**
2. **Drop non-tail operators**
3. **Apply EmergencySimplify**
4. **Force regime → stable**
5. **Never mute audio completely**

**Normative invariant**

> Sound must always continue, even in degraded form.

---

## D. Foreground vs Cloud Completion (Restated)

If scheduling budget is exceeded:

- Foreground voices → **freeze**
- Cloud voices → **decay**

No voice identity is destroyed unless explicitly released.

---

## E. Deterministic Ordering Guarantees

### E.1 Operator Order

- Backbone ops: **author order**
- Event lanes: ascending lane index
- Tail ops: always last
- No reordering permitted

---

### E.2 Block Time Rule (Restated)

- All engines observe **block time only**
- No sub-block musical state exists
- Gesture/UI time is resampled upstream

---

## F. Architecture Completeness Statement

With this appendix:

- All IRs are closed
- All execution paths are deterministic
- All partial specs compile safely
- Python ↔ FPGA equivalence is guaranteed

> **This architecture is now closed under composition, degradation, and replay.**

Everything remaining — controls, presets, tuning, instruments — is **design space**, not architecture risk.

---

If you want next steps, the natural options are:

1. **Freeze Parameter Registry v0** (the actual param list)
2. **Write the Editor Schema** (JSON / graph / UI mapping)
3. **Author 3 canonical instruments as regression tests**

You are no longer building foundations — you are designing instruments.