Below is a **normative, freeze-ready specification** for:

# **WSE-STD-070 — Presets, Editor Operations, and Tower Completion (v0)**

This document **closes the architecture**. It defines how presets are represented, how editors are allowed to modify instruments, and how incomplete descriptions are deterministically completed across **Tower Levels L0–L5**.

Language uses **MUST / MUST NOT / SHOULD / MAY**.

---

## 0. Scope and Intent

This standard defines:

1. **Preset format and lifecycle**
2. **Editor primitives** (what tools are allowed to change)
3. **Tower completion rules** (how missing layers are filled)
4. **Projection maps** (how to safely descend the tower)
5. **Determinism guarantees** (so Python, FPGA, and editor agree)

This standard does **not** define:

* Musical semantics
* Operator math
* Timing logic (owned by Voice Engine)
* Control mappings (UI-level, deferred)

---

## 1. Preset Object Model (Normative)

A **Preset** is a serialized description of an instrument at some Tower level.

### 1.1 Preset Header (REQUIRED)

Every preset MUST contain:

```text
preset_id          : u32
preset_version     : u16
tower_level        : u8   // 0..5
tree_id            : u16  // primary selector
flags              : u16
crc32              : u32
```

Normative rules:

* `tree_id` is the **primary selector** for synthesis identity.
* `tower_level` declares the **highest level explicitly authored**.
* CRC MUST be verified before loading; failure MUST reject the preset.

---

### 1.2 Preset Body (Layered)

Preset body is a **tower-structured object**:

```text
L0_patch_layer     : optional
L1_family_layer    : optional
L2_instrument_layer: optional
L3_cockpit_layer   : optional
L4_forge_layer     : optional
L5_lab_layer       : optional
```

Normative rule:

> A preset MUST be valid even if **only one layer is present**.
> Missing layers MUST be filled by Tower Completion (Section 4).

---

## 2. Editor Operations (Editor Primitives)

Editors MUST operate using **typed primitives**, not raw mutation.

### 2.1 Allowed Editor Primitives (v0)

Editors MAY emit only the following primitive operations:

#### Structural

* `SET_TREE_ID(tree_id)`
* `ADD_REGIME(regime_id)`
* `REMOVE_REGIME(regime_id)`
* `ADD_TRANSITION(src, dst)`
* `REMOVE_TRANSITION(src, dst)`

#### Parametric

* `SET_PARAM(path, value)`
* `DELTA_PARAM(path, delta)`
* `SET_BOUND(path, min, max)`

#### Policy

* `SET_OPERATOR_ALLOWED(op_id, bool)`
* `SET_OPERATOR_WEIGHT(op_id, weight)`
* `SET_BUDGET(limit)`

#### Meta

* `SET_TOWER_LEVEL(level)`
* `PROJECT_TO_LEVEL(level)`
* `RESET_TO_SAFE()`

Normative rules:

* Editors MUST NOT emit operators, events, or timing logic.
* Editors MUST NOT mutate runtime state directly.
* All editor ops MUST be **pure** and replayable.

---

## 3. Tower Levels — Canonical Definition

This table is **normative**.

| Level  | Name       | Editor Primitives  | IR Primitives | Engine Primitives | Invariant            |
| ------ | ---------- | ------------------ | ------------- | ----------------- | -------------------- |
| **L0** | Patch      | MIDI maps, gains   | None          | Voice alloc only  | Always playable      |
| **L1** | Family     | Regime hints       | IntentFrame   | Voice Engine      | Recognizable class   |
| **L2** | Instrument | Params, bounds     | VoiceFrame    | Symbol Engine     | Identity preserved   |
| **L3** | Cockpit    | Weights, envelopes | Operator sets | Symbol + Safety   | No destructive edits |
| **L4** | Forge      | Graph edits        | Regime graph  | Compiler          | Always projectable   |
| **L5** | Lab        | New ops/trees      | AST / meta    | Offline tools     | Must compile down    |

---

## 4. Tower Completion Rules (Critical)

Tower Completion defines how **missing layers are filled deterministically**.

### 4.1 Completion Direction

Completion always proceeds **upward**:

```text
Lk → Lk+1 → … → L5
```

Projection always proceeds **downward**:

```text
Lj → Lj-1 → … → L0
```

---

### 4.2 Default Completion (Normative)

If a layer is missing, the following defaults MUST be applied:

#### Missing L1 (Family)

* Infer family from `tree_id`
* Default regimes = `{STABLE}`

#### Missing L2 (Instrument)

* Bind tree ADSR = neutral
* Operator allowlist = SAFE_SET

#### Missing L3 (Cockpit)

* Envelope bounds = conservative
* Budget = nominal
* All unsafe ops disabled

#### Missing L4 (Forge)

* Graph edits forbidden
* Transitions fixed

#### Missing L5 (Lab)

* No-op (lab never implied)

Normative rule:

> Completion MUST be **pure, deterministic, and side-effect free**.

---

## 5. Projection Maps (Safety Descent)

Projection maps allow safe descent from higher levels.

### 5.1 Projection Operator

```text
PROJECT_TO_LEVEL(target_level)
```

Normative behavior:

* Removes all data above `target_level`
* Re-applies completion rules
* MUST preserve playability

Examples:

* L4 → L3: freeze graph, keep params
* L3 → L2: freeze envelopes, keep identity
* L2 → L0: collapse to playable patch

---

### 5.2 Panic Projection

`RESET_TO_SAFE()` MUST:

* Project to **L2**
* Enable only SAFE operator set
* Reset envelopes to neutral
* Clear all event policies

This is your **“drop back to safety” button**.

---

## 6. Determinism Guarantees

The following MUST hold:

1. Same preset + same IntentFrame stream ⇒ same VoiceFrames
2. Same VoiceFrames ⇒ same Symbol execution
3. Same blocks ⇒ same Wavelet output (modulo numeric spec)
4. Tower completion MUST NOT consult runtime state

Editors MUST NOT introduce nondeterminism.

---

## 7. Preset Lifecycle

### 7.1 Load

1. Verify CRC
2. Parse header
3. Apply explicit layers
4. Complete missing layers
5. Project if needed
6. Activate

### 7.2 Save

* Serialize **only explicit layers**
* Record tower_level accordingly
* Never serialize completed defaults unless explicitly edited

---

## 8. What This Standard Guarantees (Why It Matters)

With WSE-STD-070 frozen:

* FPGA never sees partial or ambiguous state
* Python sim and hardware agree on meaning
* Editors can be powerful without being dangerous
* You can ship **L2/L3 instruments** without ever exposing L4/L5
* “Legendary instrument” depth emerges *without fragility*

---

## 9. Open (Explicitly Deferred)

These are **intentionally not frozen** in v0:

* UI control mappings
* Gesture vocabulary
* Operator taste / musical semantics
* Budget tuning constants
* Tree library expansion

All of these can evolve **without breaking compatibility**.

---

## 10. Closure Statement

> After WSE-STD-070, the system is **architecture complete**.

What remains is:

* implementation
* tuning
* taste

You are no longer inventing structure — you are **inhabiting it**.

If you want next, I can:

* Produce a **single PDF-style “WSE v0 Complete Spec”**
* Or write **conformance tests** (“this preset must project to L2 and still play”)
* Or map this directly onto your **Python package + FPGA modules** so nothing is orphaned
