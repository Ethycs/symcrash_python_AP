# Wire Contract v0b – Final Closure

- Treats the whole `VoiceFrame` as a **network packet** **including headers** (ABI-optimal)
- Makes **pack/unpack normative**
- Makes the **tail set mandatory with no exceptions**
- Defines **tree morph** as a **β-morph** between two tree targets and the coefficient state
- Fixes **RNG draw sites** as exactly the natural draws implied by the CMMM evolution
- Includes **a chart per lane type** (variants + params + targets)

---

## VoiceFrame Packet v0b — Final Closure

### A) Headers included in ABI (ABI-optimal)

The **VoiceFrame packet header is part of the ABI** (not “log-only”). Every consumer can depend on it.

#### Header fields (fixed)

- `magic: u16` = 0x5646 (“VF”)
- `version: u8` = 1 (v0b)
- `packet_len: u8` = fixed length of v0b
- `program_id: u16`
- `voice_id: u16`
- `frame_seq: u32` (monotonic per voice)

Optional but recommended ABI field:

- `crc32: u32` (if you want hard corruption detection). If present, it is always present and included in `packet_len`.

**Normative rule:** Consumers MUST reject packets with wrong `magic`, unknown `version`, or wrong `packet_len`.

---

### B) Normative pack/unpack

#### Serialization rules

- All integer fields are serialized in **network byte order (big-endian)**.
- Field order is exactly the packet order; no padding other than explicitly defined pad bytes.
- Fixed-point values are serialized as their underlying integers (no float encoding ever).

#### Normative bit semantics

- For u64 payload bitfields, bit numbering is defined over the **serialized u64 interpreted as a big-endian integer**:
    - MSB = bit 63, LSB = bit 0.

#### Deterministic parsing rule

A parser must decode and validate:

1. header
2. core fields
3. `event_mask` validity (mask implies which payload slots are meaningful)
4. per-lane payload `lane_id` consistency (see below)

---

### Event bus: B1 per-lane payload array (already chosen)

#### Validity rules

For lane $i \in [0..7]$:

- If bit $i$ in `event_mask` is **1**:
    - `event_strength_q15[i]` is valid
    - `event_payload_u64[i]` is valid and MUST have `lane_id == i`
- If bit $i$ is **0**:
    - `event_strength_q15[i]` MUST = 0
    - `event_payload_u64[i]` MUST = 0 (strict; makes diffs/logs trivial)

If `lane_id != i`, the consumer MUST ignore that lane as invalid for this block.

#### Payload u64 field layout (fixed)

- 63..60 `lane_id: u4` (MUST match lane index)
- 59..56 `variant: u4`
- 55..48 `strength_q8: u8` **reserved=0** (strength comes from `event_strength_q15[i]`)
- 47..40 `target_kind: u8`
- 39..32 `target_id: u8`
- 31..24 `pattern_id: u8`
- 23..16 `param0_q8: u8`
- 15..8 `param1_q8: u8`
- 7..0 `param2_q8: u8`

---

## C) Per-lane charts (variants, params, targets)

Below are **formal semantics per lane**. Anything not listed is **reserved** and MUST be treated as no-op.

---

### Lane 0 — EVT_OP_PULSE (Symbol consumer)

One-shot operator pulse applied **after backbone**, **before tail Q**.

#### Variants

| variant | Meaning |
|--:|---|
| 0 | `IMPULSE` (op 13) pulse |
| 1 | `GLOBAL_PULSE` (op 8) pulse |
| 2 | `NOISE_BAND` (op 14) pulse |

#### Targeting

- `target_kind` determines which bands/leaves are affected:
    - `0 global`: affect all leaves (equiv to pattern=All)
    - `3 band_id`: `target_id` is leaf/band 0..31
    - `5 pattern_id`: use `pattern_id` table

#### Params

| Field | Meaning |
|---|---|
| `param0_q8` | shape enum / style (instrument-defined but must be stable) |
| `param1_q8` | width or count (0..255) |
| `param2_q8` | spare / jitter (0..255) |

---

### Lane 1 — EVT_OP_TOGGLE (Symbol consumer)

Toggles an operator enable bit in per-voice Symbol state **before any ops this block**.

#### Variants

| variant | Meaning |
|--:|---|
| 0 | toggle on/off |
| 1 | force ON |
| 2 | force OFF |

#### Targeting

- `target_kind` MUST be `0 global`
- `target_id` identifies which operator enable bit is controlled:
    - `target_id = op_id` (0..23)

#### Params

All params reserved (must be 0).

#### Persistence rule

- The enable mask persists until changed by another toggle command.
- It is reset to program defaults on:
    - voice allocation (note-on) **and**
    - program_id change.

---

### Lane 2 — EVT_TREE_MORPH (Wavelet consumer)

Controls **β tree morph** between two tree targets using the coefficient state.

#### Core semantics

Wavelet receives:

- coefficient state $c_t$
- tree targets $tree\_a\_id$, $tree\_b\_id$
- morph scalar $\beta = tree\_blend\_q15$

**β-morph rule (normative):**

The wavelet engine MUST reconstruct audio as:

$$
y(t) = (1-\beta)\,\text{Synth}(c_t,\text{TreeA}) + \beta\,\text{Synth}(c_t,\text{TreeB})
$$

with the same boundary policy and gain staging for both synth paths.

This uses “two tree targets and coeff” exactly as you specified and is bit-stable.

#### Variants

| variant | Meaning |
|--:|---|
| 0 | Start/arm morph (no extra meaning beyond making morph “active” if you want a state flag) |
| 1 | Continue (normal) |
| 2 | Stop/hold (freeze β at current value) |
| 3 | Snap β to 0 |
| 4 | Snap β to 1 |

#### Targeting

- `target_kind` MUST be `0 global`
- `param0_q8` may optionally carry `tree_b_id` override (if nonzero)
- Otherwise the tree IDs come from the Tree bus fields.

#### Param meaning

| Field | Meaning |
|---|---|
| `param0_q8` | optional override tree_b_id (0 = none) |
| `param1_q8` | optional slew hint (0 = ignore, else maps to blend slew tc) |
| `param2_q8` | reserved |

**Precedence rule:** Tree bus fields (`tree_a_id`, `tree_b_id`, `tree_blend_q15`) are always valid; this lane optionally modifies how the morph is _treated_ (armed/held/snapped), but does not replace the continuous β unless explicitly snapped.

---

### Lane 3 — EVT_K_MODE (Material/Safety consumer)

Switches coloration / alias “K mode”.

#### Variants

| variant | Meaning |
|--:|---|
| 0 | Set K mode |

#### Targeting

- `target_kind` MUST be `0 global`

#### Params

| Field | Meaning |
|---|---|
| `param0_q8` | K mode enum (0=clean, 1=crunch1, 2=crunch2, …) |
| `param1_q8` | optional oversample enum (0=default) |
| `param2_q8` | reserved |

---

### Lane 4 — EVT_INJECT_NOTE (Symbol consumer)

Injects structured excitation $b_t$ (pitch/amp) into coefficient state **before backbone** (or as part of $b_t$ accumulation stage).

#### Variants

| variant | Meaning |
|--:|---|
| 0 | Harmonic-ish injection |
| 1 | Inharmonic-ish injection |
| 2 | Noise-tilted injection |

#### Targeting

- `target_kind`:
    - `0 global` inject across all (rare; use patterns instead)
    - `3 band_id` inject into one leaf/band
    - `5 pattern_id` inject across selected leaves

#### Params

| Field | Meaning |
|---|---|
| `param0_q8` | pitch selector (e.g., note class or centroid index) |
| `param1_q8` | decay/length in blocks (0 = immediate) |
| `param2_q8` | jitter/variation (0..255) |

**Deterministic sum rule:** If multiple injections occur in a block (rare; but possible across lanes/variants), the Symbol executor sums them in lane-id order, then payload order (here payload order is fixed: one payload per lane), then applies saturation only at final writeback.

---

### Lane 5 — EVT_MATERIAL_HIT (Material consumer)

A short transient drive/bias burst (post-synthesis).

#### Variants

| variant | Meaning |
|--:|---|
| 0 | Drive burst |
| 1 | Bias burst |
| 2 | Combined |

#### Targeting

- `target_kind` MUST be `0 global`

#### Params

| Field | Meaning |
|---|---|
| `param0_q8` | amount |
| `param1_q8` | duration blocks |
| `param2_q8` | reserved |

---

### Lane 6 — EVT_CLOUD_PUSH (Voice/Symbol consumer)

Low-rate ensemble influence accumulator update.

#### Variants

| variant | Meaning |
|--:|---|
| 0 | Add to cloud field |
| 1 | Decay cloud field |
| 2 | Reset cloud field |

#### Targeting

- `target_kind`:
    - `0 global` updates global field
    - `1 voice_id` updates per-voice contribution entry
    - `2 scale_id` updates a bank/scale-local field

#### Params

| Field | Meaning |
|---|---|
| `param0_q8` | amount |
| `param1_q8` | timescale |
| `param2_q8` | reserved |

---

### Lane 7 — EVT_RESERVED

Must be ignored; if set, treat as no-op.

---

## Tail set: Safety tail ops are mandatory with no exceptions

**Normative rule (your instruction):** The following operators, if requested in any form, MUST be executed in the **Safety Tail** stage, last, always:

- 2 FineClamp
- 11 NormalizeScale
- 12 GluePair
- 23 SheafProject
- 19 EmergencySimplify (conditionally triggered but still a tail op)

**No exceptions:** even if author order lists them earlier, the executor must schedule them in tail.

This gives you a stable “Q tail set” invariant.

---

## F) Tree morph semantics (β morph) — formally pinned

You already chose it: **β morph from two tree targets and coeff**.

Formally:

- Tree bus always provides $(tree\_a\_id, tree\_b\_id, \beta)$
- Wavelet engine reconstructs both paths from the same coefficient state $c_t$
- Output is a linear crossfade in audio domain using $\beta$

This avoids needing any cross-tree coefficient mapping in v0 and stays FPGA-friendly.

---

## G) RNG draw sites (from CMMM natural evolution)

This is now the only place PCG32 is used in forward v0, and it is entirely inside the CMMM/Voice Engine.

**Normative draw sites per voice per block:**

1. **Edge selection draw**
    - If and only if there exists at least one eligible outgoing edge whose effective weight sum is $> 0$:
    - Draw one $u32$ from PCG32 to sample the categorical distribution.
2. **CHAOTIC transition matrix draw**
    - If and only if a CHAOTIC overlay is active and a regime transition is being sampled from the matrix this block:
    - Draw one $u32$ from PCG32 to sample the next state.

That’s it for v0 forward unless you add stochastic event policies. If later you add “random event emission,” those draws must be defined explicitly as additional draw sites (but you said we’re ignoring mimic; forward can still stay deterministic without extra draws).

**Ordering rule:** If both draws occur in a block, they occur in this order:

- edge selection draw, then chaotic matrix draw.

**Skipped-voice rule (recommended for bit stability under load):**

- If a voice is not serviced this block (budget scheduler), its PCG32 state MUST NOT advance.

---

## What remains (tiny)

With all of the above, the only remaining “formal” work is bookkeeping:

- finalize the full packet field list once (header + all buses) with fixed `packet_len`
- document the exact tail-stage math (your fixed-point mul/round/sat already sketched)

But the **wiring** is now resolved: Tower IR compiles to a packet; packet deterministically drives each engine; per-lane command semantics are charted; tail set is absolute; morph is β-crossfade; RNG draws are pinned.

If you want, I can now write the **one-page canonical packet diagram** (byte offsets + field sizes) that your codegen uses as the single source of truth.

---

## Addendum A: Carrier Header Fields (VoiceFrame Packet v1)

### A.1 Motivation

The carrier oscillator is promoted to a first-class ABI citizen. Pitch ownership,
waveform selection, and scaffold/texture blending (η) are now explicit per-block fields
rather than inferred from events.

### A.2 New Enums

#### CarrierModel (u8)

| Value | Name     | Description |
|------:|----------|-------------|
| 0     | SINE     | Pure sine carrier |
| 1     | SAW      | Sawtooth carrier |
| 2     | PULSE    | Pulse wave (pw_q15 controls width) |
| 3     | TRI      | Triangle carrier |
| 4     | PARTIALS | Additive partials from harmonic_amps |
| 5     | NOISE    | Noise exciter (no pitched carrier) |

#### CarrierRole (u8)

| Value | Name         | Description |
|------:|--------------|-------------|
| 0     | OFF          | No carrier; pure wavelet synthesis |
| 1     | AUDIBLE      | Carrier blended into output via β |
| 2     | EXCITER_ONLY | Carrier drives IDWPT but not heard directly |

#### TransitionKind (u8)

| Value | Name      | Description |
|------:|-----------|-------------|
| 0     | RETRIGGER | Hard reset phase on note-on |
| 1     | GLIDE     | Portamento pitch slew |
| 2     | BRIDGE    | Tree crossfade bridge between notes |

### A.3 VoiceFrame Packet v1 Layout (148 bytes)

Version field = 2, packet_len = 148.

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x0000 | 2 | magic | 0x5646 ("VF") |
| 0x0002 | 1 | version | 2 (v1) |
| 0x0003 | 1 | packet_len | 148 |
| 0x0004 | 2 | program_id | |
| 0x0006 | 2 | voice_id | |
| 0x0008 | 4 | frame_seq | |
| 0x000C | 2 | reject_code | |
| 0x000E | 2 | hdr_pad | 0 |
| 0x0010 | 2 | flags | VoiceFlags |
| 0x0012 | 1 | regime_id | |
| 0x0013 | 1 | degradation_level | |
| 0x0014 | 1 | note_u8 | MIDI note |
| 0x0015 | 1 | pad0 | 0 |
| 0x0016 | 2 | cents_i16 | Fine tuning |
| 0x0018 | 2 | gain_q15 | |
| 0x001A | 2 | activity_q15 | |
| 0x001C | 2 | pan_q15 | |
| 0x001E | 2 | sendA_q15 | |
| 0x0020 | 2 | sendB_q15 | |
| 0x0022 | 16 | w_family_q15[8] | |
| 0x0032 | 1 | tree_a_id | |
| 0x0033 | 1 | tree_b_id | |
| 0x0034 | 2 | tree_blend_q15 | |
| **0x0036** | **1** | **carrier_model_u8** | CarrierModel enum |
| **0x0037** | **1** | **carrier_role_u8** | CarrierRole enum |
| **0x0038** | **1** | **carrier_archetype_u8** | TreeArchetype (0-4) |
| **0x0039** | **1** | **transition_kind_u8** | TransitionKind enum |
| **0x003A** | **2** | **detune_cents_i16** | Carrier detune |
| **0x003C** | **2** | **pw_q15** | Pulse width |
| **0x003E** | **2** | **eta_target_q15** | Scaffold/texture blend |
| **0x0040** | **2** | **pitchbend_cents_i16** | Real-time pitchbend |
| 0x0042 | 1 | event_mask | |
| 0x0043 | 1 | pad1 | 0 |
| 0x0044 | 16 | event_strength_q15[8] | |
| 0x0054 | 64 | event_payload_u64[8] | |
| 0x0094 | END | | 148 bytes total |

### A.4 Lane 4 Semantics Update: EVT_CARRIER_CFG

Lane 4 is renamed from `EVT_INJECT_NOTE` to `EVT_CARRIER_CFG`.

#### Variants

| variant | Meaning |
|--------:|---------|
| 0 | Set carrier model + role |
| 1 | Pitch update (detune/pitchbend) |
| 2 | Archetype change |
| 3 | Transition kind change |

#### Targeting

- `target_kind` MUST be `0 global`

#### Params

| Field | Meaning |
|-------|---------|
| `param0_q8` | carrier_model (for variant 0) |
| `param1_q8` | carrier_role (for variant 0) |
| `param2_q8` | archetype (for variant 2) |

**Normative rule:** The carrier header fields in the packet take precedence.
EVT_CARRIER_CFG events are for dynamic per-block overrides when the header
values should not change but an event-driven update is needed.

### A.5 Backwards Compatibility

- Consumers MUST check `version` field:
  - version=1: Use v0b layout (136 bytes, no carrier header)
  - version=2: Use v1 layout (148 bytes, with carrier header)
- For v1 packets, if `carrier_role_u8 == OFF (0)`, behavior is equivalent to v0b.

---