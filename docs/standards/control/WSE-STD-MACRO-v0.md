## WSE-STD-MACRO v0 — Macro ABI + Slew + Ownership

### 0. Scope

This standard freezes the **macro interface** used by CMMM/VoiceEngine to control the whole synthesis pipeline deterministically:

* Macro **IDs**, types, ranges
* Macro **ownership** (who is allowed to write what)
* Macro **update rules** (block-rate only)
* Macro **slew filters** (bit-exact)
* Macro **latching/completion** rules (when updates are missing)

This standard is transport-agnostic. Transport details live in WSE-STD-020 (VoiceFrame + param bus) and WSE-STD-040 (Control paths).

---

### 1. Representation (normative)

* All macros are **Q15 unsigned**: `u16 in [0..32767]` unless noted.
* All macro effects are applied **at block boundaries only**.
* All macro state is **per-voice** unless explicitly declared global.

Helper:

* `Q15_ONE = 32767`
* `clamp_q15(x) = min(max(x,0),32767)`

---

### 2. Frozen Macro IDs (v0)

| ID | Name          | Kind    | Scope               | Range | Owner                  |
| -: | ------------- | ------- | ------------------- | ----- | ---------------------- |
|  0 | energy_state  | state   | per-voice           | Q15   | Envelope (VoiceEngine) |
|  1 | energy_intent | control | per-voice or global | Q15   | Controller → CMMM      |
|  2 | motion        | control | per-voice           | Q15   | Controller → CMMM      |
|  3 | focus         | control | per-voice           | Q15   | Controller → CMMM      |
|  4 | brightness    | control | per-voice           | Q15   | Controller → CMMM      |
|  5 | density       | control | per-voice           | Q15   | Controller → CMMM      |
|  6 | risk          | control | per-voice           | Q15   | Controller → CMMM      |
|  7 | age           | state   | per-voice           | Q15   | CMMM integrator        |
|  8 | space         | control | per-voice           | Q15   | Controller → CMMM      |
|  9 | material      | control | per-voice or global | Q15   | Controller → CMMM      |
| 10 | unity         | control | per-voice           | Q15   | Controller → CMMM      |

**Normative:** `energy_state` (0) MUST be written only by the envelope generator. Controllers MUST NOT set it.

---

### 3. Macro update interface (semantic)

Macro writes are expressed as:

`SET_MACRO(scope, target_id, macro_id, value_q15)`

* `scope ∈ {GLOBAL, VOICE, NOTE_INIT}`
* If `scope=GLOBAL`, `target_id` is ignored.
* If `scope=VOICE`, `target_id=voice_id`.
* If `scope=NOTE_INIT`, the value is applied only at voice allocation and becomes the initial macro target for that voice.

Transport mapping:

* Via PARAM registers (WSE-STD-REGISTRY) or explicit macro messages if you add them later.

---

### 4. Latching model (normative)

For each voice and each macro `m` (except `energy_state`), store:

* `macro_target_q15[m]` (latched last commanded value)
* `macro_curr_q15[m]` (slewed runtime value used by CMMM this block)

Completion rules:

* If no new SET_MACRO arrives for a block: `macro_target_q15` remains unchanged.
* Slew always advances `macro_curr_q15 → macro_target_q15` per policy.

---

### 5. Slew policies (bit-exact)

Each macro declares one of:

#### 5.1 NONE

`curr = target`

#### 5.2 NBLOCK(N) linear slew (deterministic)

Let `d = target - curr` (signed int32).

`step = ceil_div(abs(d), N)` where:

* `ceil_div(a,b) = (a + b - 1) // b` for `a>=0`

Update:

* `curr = curr + clamp(d, -step, +step)`

#### 5.3 1POLE_Q15(alpha_q15)

Let `d = target - curr` (signed int32).
`curr = curr + mul_q15(alpha_q15, d)` with your canonical Q15 rounding/saturation rules.

---

### 6. Default slew table (v0 frozen)

Block rate assumed 64@48k (750 Hz). These are “feel-good” defaults and FPGA-friendly.

| Macro         | Slew                             |
| ------------- | -------------------------------- |
| energy_state  | NONE (envelope-owned)            |
| energy_intent | 1POLE_Q15(alpha=2048)  *(≈1/16)* |
| motion        | 1POLE_Q15(alpha=2048)            |
| focus         | 1POLE_Q15(alpha=2048)            |
| brightness    | 1POLE_Q15(alpha=2048)            |
| density       | 1POLE_Q15(alpha=2048)            |
| risk          | NBLOCK(16)                       |
| age           | integrator (see §7)              |
| space         | 1POLE_Q15(alpha=2048)            |
| material      | NBLOCK(32)                       |
| unity         | NBLOCK(32)                       |

---

### 7. Stateful macro: age (normative integrator)

`age` is not directly set in normal play; it evolves deterministically.

Per block:

* `age = clamp_q15(age + inc - dec)`

Recommended v0 rule (integer-only):

* `inc = mul_q15(k_inc_q15, energy_state)`
* `dec = mul_q15(k_dec_q15, (Q15_ONE - energy_state))`

Where `k_inc_q15`, `k_dec_q15` are instrument parameters (registry-defined) with conservative defaults.

---

### 8. Macro outputs (what CMMM may do)

Macros may influence:

* regime transitions/hysteresis
* degradation/budgets/K and tail modes
* VoiceFrame fields: `gear_q15`, `traction_q15`, `tree_blend_q15`, backbone strengths, event lanes, carrier blend (beta/eta)

**Normative:** macros MUST NOT directly alter:

* operator definitions
* lane schedule
* tree topology generation

---

### 9. Conformance

An implementation conforms if:

* IDs and ownership rules match §2
* All macro effects are block-boundary only
* Slew math is bit-exact and deterministic
* `energy_state` is envelope-owned only
* Missing updates latch cleanly

---

---

## WSE-STD-REGISTRY v0 — Parameter Register ABI (Typed Reg Map + Runtime Safety)

### 0. Scope

This standard defines the **parameter register map** addressed by `PARAM_ABSOLUTE / PARAM_DELTA` messages (WSE-STD-020), including:

* Register IDs (u16) and domains
* Types, ranges, scaling
* Scope (GLOBAL / VOICE / INSTRUMENT)
* Update rate class (EVENT / BLOCK / SLOW)
* Runtime safety (hot-swap rules)
* How registry parameters feed: Envelope, Macro targets, CMMM constants, VoiceFrame defaults

Goal: make controller integration and preset compilation “boringly consistent.”

---

### 1. Register addressing (normative)

A register write is:

`SET_PARAM(reg_scope, reg_id_u16, value_i32)`
`ADD_PARAM(reg_scope, reg_id_u16, delta_i32)`

Where:

* `reg_scope ∈ {GLOBAL, VOICE, INSTRUMENT}`
* `value_i32` interpretation depends on `reg_id` type.

**Normative:** every reg has a declared type; writes outside range MUST clamp deterministically.

---

### 2. Type system (v0)

| Type |        Storage | Meaning                                |
| ---- | -------------: | -------------------------------------- |
| Q15U | u16 (0..32767) | normalized [0..1]                      |
| I16  |            i16 | signed cents, semitone offsets, etc    |
| U8   |             u8 | enums                                  |
| U16  |            u16 | ids, counts                            |
| I32  |            i32 | fixed-point “engineering units” (rare) |

**Normative:** If transport sends i32, receiver MUST clamp then cast to the declared type.

---

### 3. Runtime safety classes

Each register is labeled:

* **STATIC**: may change only via instrument reload boundary
* **HOTSAFE**: may change while playing; applied at next block boundary
* **EVENTONLY**: may change only on explicit event (note-on, preset-load)

**Normative:** a STATIC write during play MUST either (a) defer until reload, or (b) NACK deterministically.

---

### 4. Canonical v0 register map

#### 4.1 Global domain (0x0000–0x00FF)

| Reg ID | Name              | Type | Safety    | Notes                                                          |
| -----: | ----------------- | ---- | --------- | -------------------------------------------------------------- |
| 0x0000 | MASTER_GAIN_Q15   | Q15U | HOTSAFE   | global post-mix gain                                           |
| 0x0001 | GLOBAL_MODE_U8    | U8   | EVENTONLY | UI mode / tower level hint (no DSP meaning unless you bind it) |
| 0x0002 | INSTRUMENT_ID_U16 | U16  | EVENTONLY | request instrument load/select                                 |

#### 4.2 Voice envelope domain (0x0100–0x01FF) — VoiceEngine-owned

| Reg ID | Name                  | Type | Safety  | Notes                               |
| -----: | --------------------- | ---- | ------- | ----------------------------------- |
| 0x0100 | ENV_ATTACK_BLOCKS     | U16  | HOTSAFE | 1..65535                            |
| 0x0101 | ENV_DECAY_BLOCKS      | U16  | HOTSAFE | 1..65535                            |
| 0x0102 | ENV_SUSTAIN_Q15       | Q15U | HOTSAFE | 0..1                                |
| 0x0103 | ENV_RELEASE_BLOCKS    | U16  | HOTSAFE | 1..65535                            |
| 0x0104 | ENV_VEL_TO_ENERGY_Q15 | Q15U | STATIC  | velocity→energy_intent mapping gain |
| 0x0105 | ENV_RETRIGGER_U8      | U8   | HOTSAFE | 0=legato,1=retrigger                |

**Normative:** Envelope produces `energy_state` and `gain_q15` outputs; macros may bias envelope via allowed regs only.

#### 4.3 Macro input domain (0x0200–0x02FF) — controller → macro_target

These registers are the “official” way hardware/UI sets macros without inventing a new message type.

| Reg ID | Name                    | Type | Safety  | Macro ID |
| -----: | ----------------------- | ---- | ------- | -------: |
| 0x0201 | MACRO_ENERGY_INTENT_Q15 | Q15U | HOTSAFE |        1 |
| 0x0202 | MACRO_MOTION_Q15        | Q15U | HOTSAFE |        2 |
| 0x0203 | MACRO_FOCUS_Q15         | Q15U | HOTSAFE |        3 |
| 0x0204 | MACRO_BRIGHTNESS_Q15    | Q15U | HOTSAFE |        4 |
| 0x0205 | MACRO_DENSITY_Q15       | Q15U | HOTSAFE |        5 |
| 0x0206 | MACRO_RISK_Q15          | Q15U | HOTSAFE |        6 |
| 0x0208 | MACRO_SPACE_Q15         | Q15U | HOTSAFE |        8 |
| 0x0209 | MACRO_MATERIAL_Q15      | Q15U | HOTSAFE |        9 |
| 0x020A | MACRO_UNITY_Q15         | Q15U | HOTSAFE |       10 |

**Normative:** There is intentionally no register for `energy_state` (macro 0). It is envelope-owned.

#### 4.4 Carrier / blend domain (0x0300–0x03FF) — feeds VoiceFrame v1 carrier header defaults or per-voice overrides

| Reg ID | Name                | Type | Safety  | Notes                              |
| -----: | ------------------- | ---- | ------- | ---------------------------------- |
| 0x0300 | CARRIER_MODEL_U8    | U8   | HOTSAFE | enum from WSE-STD-020 v1           |
| 0x0301 | CARRIER_ROLE_U8     | U8   | HOTSAFE | AUDIBLE / EXCITER_ONLY / OFF       |
| 0x0302 | DETUNE_CENTS_I16    | I16  | HOTSAFE | signed cents                       |
| 0x0303 | PULSE_WIDTH_Q15     | Q15U | HOTSAFE | for pulse carrier                  |
| 0x0304 | ETA_TARGET_Q15      | Q15U | HOTSAFE | dry/wet carrier↔IDWPT blend target |
| 0x0305 | PITCHBEND_CENTS_I16 | I16  | HOTSAFE | real-time pitchbend                |
| 0x0306 | TRANSITION_KIND_U8  | U8   | HOTSAFE | RETRIGGER/GLIDE/BRIDGE             |

#### 4.5 CMMM constants domain (0x0400–0x04FF) — affects regime/budget behavior (mostly STATIC)

| Reg ID | Name                     | Type | Safety | Notes                                  |
| -----: | ------------------------ | ---- | ------ | -------------------------------------- |
| 0x0400 | CMMM_ALPHA_1POLE_Q15     | Q15U | STATIC | default 1POLE alpha (if not per-macro) |
| 0x0401 | AGE_K_INC_Q15            | Q15U | STATIC | age integrator inc                     |
| 0x0402 | AGE_K_DEC_Q15            | Q15U | STATIC | age integrator dec                     |
| 0x0403 | RISK_HYST_BLOCKS_U16     | U16  | STATIC | regime hysteresis                      |
| 0x0404 | MATERIAL_HYST_BLOCKS_U16 | U16  | STATIC | slow macro hysteresis                  |

#### 4.6 Budget / degradation domain (0x0500–0x05FF) — STATIC by default

| Reg ID | Name                 | Type | Safety    | Notes                                      |
| -----: | -------------------- | ---- | --------- | ------------------------------------------ |
| 0x0500 | VOICE_CLASS_U8       | U8   | EVENTONLY | MAIN vs RESIDUAL hint                      |
| 0x0501 | BUDGET_PROFILE_U8    | U8   | STATIC    | selects precompiled regime budgets         |
| 0x0502 | K_PATTERN_PROFILE_U8 | U8   | STATIC    | selects precompiled K ladder/pattern table |

**Normative:** budgets/tables are meant to be swapped by instrument reload, not mutated live.

---

### 5. Registry → system binding rules (normative)

#### 5.1 Envelope regs

* Written values update **EnvelopeSpec** for that voice (HOTSAFE), taking effect next block.
* Envelope produces:

  * `energy_state` macro (0)
  * `gain_q15` used in VoiceFrame

#### 5.2 Macro input regs

* Writing `MACRO_*` updates `macro_target_q15[id]` for that scope.
* `macro_curr_q15` updates via WSE-STD-MACRO slew rules.

#### 5.3 Carrier regs

* Written values become default fields used by the VoiceEngine when emitting VoiceFrame v1.
* If you also send explicit VoiceFrame carrier header fields, **VoiceFrame wins** for that block.

#### 5.4 STATIC regs

* If written during play, receiver MUST either:

  * defer until instrument reload boundary, or
  * NACK with deterministic error code

Pick one policy and keep it consistent across Python + FPGA.

---

### 6. Controller mapping (your current Pico2 bridge, made “official”)

Your 6 pots can map cleanly into registry writes:

* P0 → `ENV_ATTACK_BLOCKS` (0x0100)
* P1 → `ENV_DECAY_BLOCKS` (0x0101)
* P2 → `ENV_SUSTAIN_Q15` (0x0102)
* P3 → `ENV_RELEASE_BLOCKS` (0x0103)
* P4 → `MACRO_SPACE_Q15` *(or CUT placeholder)* (0x0208)
* P5 → `MACRO_BRIGHTNESS_Q15` (0x0204)

If you want “CUT/RES” semantics now without a filter lane, you can define:

* CUT = space/focus proxy (tree beta + diffusion bias)
* RES = unity/risk proxy (glue vs instability)
  …without breaking determinism, because they’re just macros.

---

### 7. Conformance checklist

An implementation conforms to WSE-STD-REGISTRY v0 iff it:

* Implements the declared reg IDs + types + clamping
* Applies HOTSAFE regs at block boundaries only
* Enforces STATIC/EVENTONLY rules deterministically
* Routes MACRO_* regs into macro_target (not directly into synthesis)
* Never exposes VM internals as writable registers

---

If you want the next “lock it down” step: I can write **WSE-STD-PRESET v0** so presets compile into (a) InstrumentIR tables + (b) initial registry defaults + (c) golden-trace test hooks—so controller integration and preset authoring stop feeling like separate worlds.
