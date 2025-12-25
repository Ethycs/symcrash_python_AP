Below is a **normative, FPGA-friendly** `WSE-STD-020` for **VoiceFrame ABI + Event Command Bus over UART (v0)**, written so Python + FPGA + editor can’t diverge.

I’m going to assume:

* **Two UART RX into FPGA** (UART0 = high-rate frames, UART1 = UI/MIDI/control) but the protocol is identical on both; you can route message types however you want.
* **TreeID is the primary selector**.
* Controls may send **absolute** or **delta** updates.
* Tail ops are mandatory and reserved outside budget (covered in STD-030; referenced here only for ordering expectations).

---

# WSE-STD-020 — VoiceFrame ABI + Event Command Bus (UART v0)

## 0. Scope

This standard defines:

1. The **VoiceFrame v0 ABI** (block-executable command object emitted by Voice Engine)
2. A **UART packet protocol** to transport:

   * per-block VoiceFrames
   * parameter updates (absolute or delta)
   * preset load/ack
   * debug/telemetry (optional)

This standard does **not** define operator semantics, tree generation math, fixed-point arithmetic, or tail ordering. Those are in `WSE-STD-001`, `WSE-STD-010`, `WSE-STD-030`.

---

## 1. Timebase & Block Alignment

**Normative time unit:** `BlockTime`.

* `BLOCK_SAMPLES = 64`.
* All VoiceFrame commands apply to **one block**.
* A VoiceFrame MAY arrive ahead of time (queued); it MUST include a `block_index` so the FPGA can align it deterministically.

**Normative rule:** No engine downstream of VoiceFrame may observe sub-block time.

---

## 2. VoiceFrame v0 ABI (logical fields)

### 2.1 VoiceFrame v0: required fields

A VoiceFrame describes what the Symbol+Wavelet engines must execute for a **single voice** on a **single block**.

```
VoiceFrame_v0 {
  uint16 voice_id;        // 0..MAX_VOICES-1
  uint16 block_index;     // monotonic per audio stream, wraps allowed

  uint16 regime_id;       // instrument regime selector (Voice-owned)
  uint16 gear_q15;        // 0..32767 (commitment / intensity)

  uint16 tree_id_a;       // primary tree selector
  uint16 tree_id_b;       // optional morph target tree (0 = same as A)
  uint16 tree_blend_q15;  // 0..32767 blend A->B

  uint16 gain_q15;        // 0..32767 (voice output gain into mix)
  uint16 traction_q15;    // 0..32767 (Q clamp strength / legality hardness)

  // Backbone ops: up to N_BACKBONE entries, ordered
  uint8  backbone_len;    // 0..N_BACKBONE_MAX (v0 default max=4)
  BackboneEntry backbone[backbone_len];

  // Events: 8 lanes
  uint8  event_mask;      // bit i = lane i fires this block
  uint16 event_strength_q15[8];  // strength per lane, 0..32767

  // Payload mode
  uint8 payload_mode;     // 0=COMMON, 1=PER_LANE_SPARSE, 2=PER_LANE_FULL (v0 default COMMON)
  uint64 payload_common;  // used when payload_mode=COMMON

  // Optional per-lane payloads (depending on mode)
  // PER_LANE_FULL: uint64 payload_lane[8]
  // PER_LANE_SPARSE: list of (lane_id, payload64)
}
```

### 2.2 BackboneEntry v0

```
BackboneEntry {
  uint8  op_id;          // 0..255 (v0 uses 0..23)
  uint16 strength_q15;   // 0..32767
  uint16 param0_q15;     // operator-specific (or 0)
  uint16 param1_q15;     // operator-specific (or 0)
}
```

**Normative rule:** backbone operator order is **semantic** and MUST NOT be reordered by the executor.

### 2.3 Event lanes (lane IDs 0..7)

Lanes are **engine commands**. Each lane’s meaning is fixed in v0:

| Lane | Name          | Consumer        | Meaning                                                                  |
| ---: | ------------- | --------------- | ------------------------------------------------------------------------ |
|    0 | EVT_OP_FIRE   | Symbol          | Fire one-shot operator/event (uses payload target/pattern)               |
|    1 | EVT_OP_TOGGLE | Symbol          | Toggle persistent op_enable bits (payload selects op or mask)            |
|    2 | EVT_INJECT    | Symbol          | Injection (impulse/noise/burst) into coefficient state                   |
|    3 | EVT_MATERIAL  | Material        | Material engine override / burst (optional forwarding)                   |
|    4 | EVT_TREE      | Wavelet/Symbol  | Tree morph or tree switch event (rare; mostly backbone uses tree fields) |
|    5 | EVT_CLOUD     | Voice/Scheduler | Cloud/foreground reclassify hint or density hint                         |
|    6 | EVT_VOICE     | Voice           | Note lifecycle / state events (optional; for logging)                    |
|    7 | EVT_DEBUG     | Any             | Debug marker / capture request                                           |

**Normative rule:** If a lane is not implemented, it MUST be ignored deterministically (no side effects, no PRNG draws).

---

## 3. Payload64 format (v0)

Payload is a **compact selector** describing “target/pattern/variant” for event lanes.

### 3.1 Payload64 header (common to all payload variants)

`payload_u64` is interpreted as:

```
bits  0.. 7  : payload_type      (uint8)  // selects variant schema below
bits  8..15  : lane_hint         (uint8)  // OPTIONAL; used only in sparse lists; ignored in COMMON mode
bits 16..23  : pattern_id        (uint8)  // resolves via Pattern Table
bits 24..31  : target_kind       (uint8)  // 0=pattern, 1=leaf, 2=band_group, 3=coeff_mask, 4=reserved

bits 32..39  : target_id         (uint8)  // meaning depends on target_kind (e.g., leaf index)
bits 40..47  : coeff_mask        (uint8)  // bit j selects coeff j within leaf (0 => all coeffs)
bits 48..55  : variant0          (uint8)  // subtype (e.g., impulse type, noise color)
bits 56..63  : variant1          (uint8)  // extra subtype or small parameter
```

### 3.2 Canonical leaf indexing

**Normative:** leaf indices are **canonical low→high**:

* `leaf 0` = lowest frequency band
* `leaf 31` = highest frequency band

If a tree internally uses a different ordering, the Wavelet/Symbol engines MUST map canonical leaf IDs to internal structure.

### 3.3 Pattern Table resolution

`pattern_id` resolves to a fixed `(leaf_mask_u32, coeff_mask_default_u8)` entry.

* `leaf_mask_u32` bit i selects leaf i.
* `coeff_mask` in payload (bits 40..47) overrides pattern’s default if nonzero.

**Normative:** pattern table is part of the preset / configuration and MUST be identical in Python + FPGA.

---

## 4. Payload modes (COMMON vs PER_LANE)

### 4.1 payload_mode values

* `0 = COMMON` (v0 default)
* `1 = PER_LANE_SPARSE` (extension)
* `2 = PER_LANE_FULL` (extension)

### 4.2 COMMON mode semantics (v0 default)

* `payload_common` applies to **all lanes** that fire (`event_mask` bits set).
* `lane_hint` in payload header MUST be ignored.
* Different lanes interpret the same payload differently (e.g., lane 2 uses variant fields as injection subtype).

**Tradeoff:** Lowest bandwidth; cannot target different leaves/patterns per lane in the same block.

### 4.3 PER_LANE_FULL mode semantics

* A payload64 exists for each lane: `payload_lane[8]`.
* For lanes where `event_mask` bit is 0, payload_lane value MUST be ignored.

**Tradeoff:** Highest bandwidth; simplest semantics.

### 4.4 PER_LANE_SPARSE mode semantics

* Send `k` payload items, each `(lane_id, payload64)`, for lanes that have set bits.
* Any lane set in `event_mask` but missing a payload item MUST fall back to `payload_common`.

**Normative:** `payload_common` MUST still be present to ensure deterministic completion.

---

## 5. Deterministic completion rules (Tower omission safety)

VoiceFrames MAY be partially specified via delta updates or omission. FPGA MUST fill missing fields deterministically:

### 5.1 Field defaults if omitted in a VoiceFrame message

* `regime_id`: keep previous for that voice
* `gear_q15`: keep previous
* `tree_id_a`: keep previous
* `tree_id_b`: default to `tree_id_a`
* `tree_blend_q15`: default `0`
* `gain_q15`: keep previous
* `traction_q15`: keep previous
* `backbone_len`: keep previous backbone list
* `event_mask`: default `0`
* `event_strength_q15[]`: default `0` for lanes not fired; for fired lanes, if strength omitted use previous strength for that lane
* `payload_mode`: keep previous (default COMMON at voice allocation)
* `payload_common`: keep previous

**Normative:** “keep previous” means **voice-local state** in the Symbol executor’s per-voice command latch, not global.

### 5.2 Voice allocation reset

On voice allocation (note-on or explicit allocate command), the FPGA MUST initialize:

* `event_mask = 0`
* `gear_q15 = preset default`
* `traction_q15 = preset default`
* `payload_mode = COMMON`
* `backbone = preset default`

---

## 6. UART Packet Protocol (v0)

### 6.1 Electrical & framing assumptions

* 8N1
* Baud rate is implementation choice (Pico supports up to 921600; FPGA may do higher if stable).
* Packets are framed with a sync word and include CRC.

### 6.2 Packet format (byte-oriented)

All multi-byte integers are **little-endian**.

```
Packet {
  uint16  sync;        // 0xA55A
  uint8   version;     // 0
  uint8   msg_type;    // enum below
  uint16  seq;         // sender increments per packet (wrap ok)
  uint16  length;      // bytes of payload that follow
  uint8   payload[length];
  uint32  crc32;       // CRC-32 of [version..payload], NOT including sync
}
```

**Normative:** Receiver MUST search for `sync`, then read fixed header, then payload, then CRC32. Invalid CRC MUST drop packet and resync.

### 6.3 Message types

`msg_type` (v0):

| Type | Name           | Direction | Purpose                                                    |
| ---: | -------------- | --------- | ---------------------------------------------------------- |
| 0x01 | VF_ABSOLUTE    | UI→FPGA   | Full VoiceFrame (absolute fields)                          |
| 0x02 | VF_DELTA       | UI→FPGA   | Partial VoiceFrame update (bitmask indicates which fields) |
| 0x03 | PARAM_ABSOLUTE | UI→FPGA   | Set a parameter register (absolute)                        |
| 0x04 | PARAM_DELTA    | UI→FPGA   | Add delta to a parameter register                          |
| 0x05 | PRESET_LOAD    | UI→FPGA   | Load preset by ID / hash                                   |
| 0x06 | ACK            | FPGA→UI   | Ack a seq with status                                      |
| 0x07 | NACK           | FPGA→UI   | Nack with error code                                       |
| 0x08 | TELEMETRY      | FPGA→UI   | Optional: meters, voice counts, budget use                 |
| 0x09 | CAPTURE        | FPGA→UI   | Optional: debug capture frames                             |

**Normative:** FPGA MUST implement at least `VF_ABSOLUTE`, `VF_DELTA`, `ACK/NACK`.

---

## 7. VoiceFrame message payload encodings

### 7.1 VF_ABSOLUTE payload

Binary encoding:

```
uint16 voice_id
uint16 block_index

uint16 regime_id
uint16 gear_q15

uint16 tree_id_a
uint16 tree_id_b
uint16 tree_blend_q15

uint16 gain_q15
uint16 traction_q15

uint8  backbone_len
BackboneEntry[backbone_len]

uint8  event_mask
uint16 event_strength_q15[8]

uint8  payload_mode
uint64 payload_common

if payload_mode == 2 (PER_LANE_FULL):
  uint64 payload_lane[8]
else if payload_mode == 1 (PER_LANE_SPARSE):
  uint8 k
  repeat k:
    uint8 lane_id
    uint64 payload64
```

### 7.2 VF_DELTA payload

`VF_DELTA` updates only selected fields to reduce bandwidth.

```
uint16 voice_id
uint16 block_index
uint32 field_mask0   // which top-level fields are present
uint32 field_mask1   // reserved for future

[fields present in a fixed canonical order]
```

**Canonical field order for delta packing (MUST):**

1. regime_id
2. gear_q15
3. tree_id_a
4. tree_id_b
5. tree_blend_q15
6. gain_q15
7. traction_q15
8. backbone (len + entries)
9. event_mask
10. event_strength[8]
11. payload_mode
12. payload_common
13. per-lane payload section

Where `field_mask0` bits are:

| Bit | Field             |
| --: | ----------------- |
|   0 | regime_id         |
|   1 | gear_q15          |
|   2 | tree_id_a         |
|   3 | tree_id_b         |
|   4 | tree_blend_q15    |
|   5 | gain_q15          |
|   6 | traction_q15      |
|   7 | backbone          |
|   8 | event_mask        |
|   9 | event_strength[8] |
|  10 | payload_mode      |
|  11 | payload_common    |
|  12 | per-lane payloads |

**Normative completion:** any omitted field uses the completion rules in §5.

---

## 8. Parameter register messages (for controls & editor)

This supports “controls later” while keeping wire stable now.

### 8.1 PARAM_ABSOLUTE

```
uint16 reg_id
int32  value
```

### 8.2 PARAM_DELTA

```
uint16 reg_id
int32  delta
```

**Normative:** Parameter registers are separate from per-voice VoiceFrame latches. They may feed Voice Engine compilation, preset edits, or global configuration (e.g., pattern tables) depending on your architecture.

---

## 9. Error handling (ACK/NACK)

### 9.1 ACK

```
uint16 seq_acked
uint8  status   // 0=ok, 1=ok_queued, 2=ok_applied_late
uint8  detail   // optional
```

### 9.2 NACK

```
uint16 seq_nacked
uint8  error_code
uint8  detail
```

Recommended `error_code` values:

* 1: bad_crc
* 2: bad_length
* 3: unknown_msg_type
* 4: invalid_voice_id
* 5: invalid_tree_id
* 6: invalid_payload_mode
* 7: backlog_overflow (queue full)
* 8: block_too_old (missed deadline)

**Normative:** NACK MUST NOT alter engine state.

---

## 10. Scheduling semantics with block_index

The FPGA SHOULD maintain:

* a small queue keyed by `block_index` for each voice
* a “late policy”:

**Normative v0 rule:**

* If `block_index` < current block: apply as `late` only if within a configured grace window; otherwise drop with `block_too_old`.
* If `block_index` > current block: queue until due.

This preserves deterministic replay even if UART jitters.

---

## 11. Minimal bandwidth notes (so you can decide later without redesign)

At 48 kHz, BlockTime = 64 samples → 750 blocks/sec.

If you tried to send VF_ABSOLUTE for every voice every block, UART would explode. So v0 assumes:

* Most blocks use **VF_DELTA** and/or only for a subset of voices
* Voice engine runs locally (Pico) and emits frames primarily for “foreground” voices, while cloud voices are handled by degradation policy and persistent latches.

**Normative:** Protocol supports high-rate, but correctness MUST NOT depend on “all voices updated every block”.

---

## 12. Conformance checklist (v0)

An implementation conforms to `WSE-STD-020 v0` iff:

* It correctly parses packet framing, endian, CRC32.
* It implements `VF_ABSOLUTE` and `VF_DELTA` exactly with the same canonical field order.
* It enforces deterministic completion rules (§5).
* It uses canonical leaf indexing low→high for payload targeting.
* It treats backbone ordering as semantic (no reordering).
* It ignores unsupported lanes deterministically.

---

## What I still need from you (only where your choice changes the spec)

I can finalize this as a “frozen PDF-style” section once you answer these **two** decisions (they’re real forks):

1. **block_index width**: keep `uint16` (wrap every ~87s at 750 blocks/s) or go `uint32` (wrap effectively never).
2. **CRC32 polynomial**: standard Ethernet CRC-32 (0x04C11DB7) vs Castagnoli (0x1EDC6F41). (Either is fine; just pick one.)

If you don't want to decide: I'll assume `uint32 block_index` + CRC-32C (Castagnoli), because it's common in embedded and fast in software.

---

## Addendum A: Carrier Header Extension (VoiceFrame v1)

### A.1 Motivation

The carrier oscillator is promoted to a first-class ABI citizen. This addendum extends
VoiceFrame v0 with explicit carrier configuration fields, enabling pitch ownership,
waveform selection, and scaffold/texture blending (η) as per-block parameters.

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

### A.3 VoiceFrame v1 ABI Extension

VoiceFrame v1 adds the following fields after the tree control section:

```
VoiceFrame_v1 extends VoiceFrame_v0 {
  // ... existing fields from v0 ...

  // Carrier header (NEW - 12 bytes)
  uint8  carrier_model_u8;      // CarrierModel enum
  uint8  carrier_role_u8;       // CarrierRole enum
  uint8  carrier_archetype_u8;  // TreeArchetype (0-4)
  uint8  transition_kind_u8;    // TransitionKind enum
  int16  detune_cents_i16;      // Carrier detune in cents
  uint16 pw_q15;                // Pulse width (for PULSE model)
  uint16 eta_target_q15;        // Scaffold/texture blend target
  int16  pitchbend_cents_i16;   // Real-time pitchbend

  // ... events section follows ...
}
```

### A.4 Event Lane Update

Lane 4 is renamed from `EVT_INJECT` to `EVT_CARRIER_CFG`:

| Lane | Name (v0)    | Name (v1)       | Consumer        |
|-----:|--------------|-----------------|-----------------|
| 0    | EVT_OP_FIRE  | EVT_OP_FIRE     | Symbol          |
| 1    | EVT_OP_TOGGLE| EVT_OP_TOGGLE   | Symbol          |
| 2    | EVT_INJECT   | EVT_INJECT      | Symbol          |
| 3    | EVT_MATERIAL | EVT_MATERIAL    | Material        |
| 4    | EVT_TREE     | **EVT_CARRIER_CFG** | **Carrier/Wavelet** |
| 5    | EVT_CLOUD    | EVT_CLOUD       | Voice/Scheduler |
| 6    | EVT_VOICE    | EVT_VOICE       | Voice           |
| 7    | EVT_DEBUG    | EVT_DEBUG       | Any             |

#### EVT_CARRIER_CFG Semantics

**Variants:**

| variant | Meaning |
|--------:|---------|
| 0 | Set carrier model + role |
| 1 | Pitch update (detune/pitchbend) |
| 2 | Archetype change |
| 3 | Transition kind change |

**Params:**

| Field | Meaning |
|-------|---------|
| `param0_q8` | carrier_model (for variant 0) |
| `param1_q8` | carrier_role (for variant 0) |
| `param2_q8` | archetype (for variant 2) |

**Normative rule:** The carrier header fields in the VoiceFrame take precedence.
EVT_CARRIER_CFG events are for dynamic per-block overrides.

### A.5 VF_DELTA Field Mask Extension

For VF_DELTA messages, add carrier fields to the canonical field order:

| Bit | Field (v0)        | Field (v1)            |
|----:|-------------------|----------------------|
| 0   | regime_id         | regime_id            |
| 1   | gear_q15          | gear_q15             |
| 2   | tree_id_a         | tree_id_a            |
| 3   | tree_id_b         | tree_id_b            |
| 4   | tree_blend_q15    | tree_blend_q15       |
| 5   | gain_q15          | gain_q15             |
| 6   | traction_q15      | traction_q15         |
| 7   | backbone          | backbone             |
| 8   | event_mask        | event_mask           |
| 9   | event_strength[8] | event_strength[8]    |
| 10  | payload_mode      | payload_mode         |
| 11  | payload_common    | payload_common       |
| 12  | per-lane payloads | per-lane payloads    |
| **13** | - | **carrier_model_u8** |
| **14** | - | **carrier_role_u8** |
| **15** | - | **carrier_archetype_u8** |
| **16** | - | **transition_kind_u8** |
| **17** | - | **detune_cents_i16** |
| **18** | - | **pw_q15** |
| **19** | - | **eta_target_q15** |
| **20** | - | **pitchbend_cents_i16** |

### A.6 Completion Rules for Carrier Fields

On voice allocation (note-on), carrier fields initialize to:

- `carrier_model_u8 = SINE (0)`
- `carrier_role_u8 = preset default (typically AUDIBLE)`
- `carrier_archetype_u8 = HARMONIC (0)`
- `transition_kind_u8 = RETRIGGER (0)`
- `detune_cents_i16 = 0`
- `pw_q15 = 16384` (0.5 pulse width)
- `eta_target_q15 = 16384` (0.5 scaffold/texture blend)
- `pitchbend_cents_i16 = 0`

For VF_DELTA, omitted carrier fields use "keep previous" semantics.

### A.7 Conformance (v1)

An implementation conforms to `WSE-STD-020 v1` iff it:

- Conforms to all v0 requirements
- Supports CarrierModel, CarrierRole, TransitionKind enums
- Implements VoiceFrame v1 carrier header fields
- Handles EVT_CARRIER_CFG lane semantics
- Applies correct completion rules for carrier fields

---
