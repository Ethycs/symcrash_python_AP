# **WSE-STD-100 — Dual-UART Link Spec v1.0**

**Status:** Normative
**Applies to:** Host (Python/Pico), FPGA (Audio Engine)
**Scope:** Defines the dual-UART architecture separating control (reliable, low bandwidth) from data (high bandwidth, streaming), with FPGA-native block-synchronous timing.

---

## 1. Roles and Goals

### 1.1 Control UART (C-UART)

**Purpose:** Reliable command/event plane
**Traffic:** MIDI-ish note events, macro changes, instrument/tree selection, parameter updates, queries, acks
**Guarantee:** Ordered + acknowledged (at least once / exactly-once via seq)

### 1.2 Data UART (D-UART)

**Purpose:** High-throughput content plane
**Traffic:** Per-voice "symbol–wavelet complex" deltas: sparse leaf coefficients, event bursts, envelopes, etc.
**Guarantee:** Best-effort ordered; loss handled by hold/decay policies (no hard failure)

Both channels share a global time base: **block index**.

---

## 2. Physical Layer

| UART   | Recommended Baud                                   | Frame | Flow Control                              |
| ------ | -------------------------------------------------- | ----- | ----------------------------------------- |
| C-UART | 1,000,000 (or 115,200 minimum)                     | 8N1   | None or RTS/CTS optional                  |
| D-UART | 3,000,000 to 12,000,000 (as reliable on your wiring) | 8N1   | RTS/CTS strongly recommended at high baud |

**Notes:**

* Use separate UART pins/cores.
* If no RTS/CTS, implement **credit-based flow control** (Section 6).

---

## 3. Shared Time Model (Hard Requirement)

### 3.1 Block Clock

FPGA runs audio in blocks of **N samples** (e.g., 64). Let:

* `block_id` be a monotone 32-bit counter incremented each audio block.

### 3.2 Scheduling Rule

All time-sensitive messages carry:

* `apply_block_id` (the block when the change becomes active)

### 3.3 Late Arrival Policy

* If `apply_block_id <= current_block_id`, FPGA applies at **next block** and sets a "late" flag in status.

### 3.4 Lead Time

* Host should schedule at least `LEAD_BLOCKS` ahead (recommend 4–16).

---

## 4. Common Message Framing (Both UARTs)

### 4.1 Transport Frame

Use a byte-stuffing scheme for robust resync:

* **COBS** (recommended) or **SLIP**.
* Frame boundary is a **0x00** byte (COBS standard).

### 4.2 Frame Header (Inside COBS Payload)

All multi-byte integers are **little-endian**.

| Field      | Size     | Notes                                    |
| ---------- | -------: | ---------------------------------------- |
| `magic`    |        2 | `0xC0 0xDE`                              |
| `version`  |        1 | `0x01`                                   |
| `chan`     |        1 | `0`=control, `1`=data (for sanity)       |
| `msg_type` |        1 | Enumerated                               |
| `flags`    |        1 | Bitfield                                 |
| `seq`      |        2 | Wraps; per-channel                       |
| `length`   |        2 | Payload length bytes                     |
| `crc32`    |        4 | CRC-32 of header(with crc32=0)+payload   |
| `payload`  | `length` |                                          |

**Header Size:** 14 bytes (fixed)

**Resync Rule:** Scan for `0x00` delimiters; COBS decode; validate magic+crc32.

---

## 5. Control UART Message Types (Reliable)

### 5.1 Reliability Model

* Every control message **must be ACKed**.
* Sender retransmits if no ACK within `ACK_TIMEOUT_MS` (recommend 20–50 ms).
* Exactly-once semantics: receiver drops duplicates based on `(seq, sender_id)`.

### 5.2 Control Message Set

| msg_type          | Direction | Purpose                                                |
| ----------------- | --------- | ------------------------------------------------------ |
| `HELLO`           | Host→FPGA | Start session, negotiate features/baud, request status |
| `HELLO_ACK`       | FPGA→Host | Returns build IDs, buffer sizes, current `block_id`    |
| `GET_STATUS`      | Host→FPGA | Poll status                                            |
| `STATUS`          | FPGA→Host | Reports `block_id`, underruns, credits, voice counts   |
| `SET_INSTRUMENT`  | Host→FPGA | Select instrument preset / bank                        |
| `SET_TREE_TARGET` | Host→FPGA | Choose tree/tuning target (0..2) + morph time τ        |
| `SET_MACRO`       | Host→FPGA | Set macro knob (0..1) with `apply_block_id`            |
| `NOTE_EVENT`      | Host→FPGA | NoteOn/Off, velocity, channel, `apply_block_id`        |
| `VOICE_POLICY`    | Host→FPGA | Main/residual counts, steal policy                     |
| `STREAM_CTRL`     | Host→FPGA | Start/stop data streaming, set stream mode             |
| `ACK`             | Either    | ACK a seq                                              |
| `NACK`            | Either    | Reject with error code                                 |

### 5.3 Control Payload Formats

#### 5.3.1 `NOTE_EVENT`

| Field            | Size | Notes                |
| ---------------- | ---: | -------------------- |
| `apply_block_id` |    4 | When it takes effect |
| `note`           |    1 | 0–127                |
| `vel`            |    1 | 0–127                |
| `on`             |    1 | 1=on, 0=off          |
| `chan`           |    1 | MIDI channel         |
| `flags`          |    1 | e.g. legato, retrig  |

**Total:** 9 bytes

#### 5.3.2 `SET_TREE_TARGET`

| Field            | Size | Notes                              |
| ---------------- | ---: | ---------------------------------- |
| `apply_block_id` |    4 |                                    |
| `tree_index`     |    1 | 0..2                               |
| `tau_ms`         |    2 | Morph time constant (1000..5000 typical) |
| `curve`          |    1 | 0=exp, 1=minjerk                   |

**Total:** 8 bytes

#### 5.3.3 `STATUS`

| Field              | Size | Notes              |
| ------------------ | ---: | ------------------ |
| `block_id`         |    4 | Current block      |
| `data_credit_bytes`|    4 | D-UART credit      |
| `active_voices`    |    2 | Currently active   |
| `main_voices`      |    2 | Main voice count   |
| `underrun_count`   |    4 | Cumulative         |
| `late_apply_count` |    4 | Cumulative         |
| `last_error_code`  |    2 | Most recent error  |

**Total:** 22 bytes

---

## 6. Data UART Message Types (Streaming)

### 6.1 Streaming Model

Data messages are **not ACKed** by default.

* Order is detected by `seq`.
* Loss is handled by policies: hold-last-value, exponential decay, or "safe zero" over a short fade.

### 6.2 Data Message Set

| msg_type            | Direction | Purpose                                             |
| ------------------- | --------- | --------------------------------------------------- |
| `DATA_STREAM_START` | Host→FPGA | Begin sending per-voice data with a base `block_id` |
| `DATA_STREAM_STOP`  | Host→FPGA | Stop                                                |
| `VOICE_DELTA`       | Host→FPGA | Per-voice updates (sparse) scheduled for a block    |
| `VOICE_BURST`       | Host→FPGA | Larger payload (e.g. many leaves)                   |
| `DATA_CREDIT`       | FPGA→Host | Credit-based flow control (bytes available)         |
| `DATA_DROP_NOTICE`  | FPGA→Host | FPGA dropped frames (buffer full)                   |
| `DATA_HEARTBEAT`    | Either    | Keepalive + clock alignment hints                   |

### 6.3 Credit-Based Flow Control

FPGA maintains a D-UART RX buffer and periodically sends:

#### `DATA_CREDIT`

| Field           | Size | Notes                                         |
| --------------- | ---: | --------------------------------------------- |
| `block_id`      |    4 | Current                                       |
| `credit_bytes`  |    4 | How many bytes host may send without overflow |
| `rx_fill_bytes` |    4 | Optional                                      |

**Total:** 12 bytes

**Host Rule:** Never exceed `credit_bytes` outstanding.

---

## 7. Per-Voice Data Representation

The symbol–wavelet complex is **per voice**, but **trees are global**. D-UART carries only *voice-local* content (not global tuning tables).

### 7.1 `VOICE_DELTA` (Sparse Leaf + Events)

This is the workhorse message.

| Field            | Size | Notes                                          |
| ---------------- | ---: | ---------------------------------------------- |
| `apply_block_id` |    4 | Block to apply                                 |
| `voice_id`       |    1 | 0..127                                         |
| `subtype`        |    1 | 0=leaf_gains, 1=leaf_coeffs, 2=events, 3=mixed |
| `count`          |    1 | Number of entries                              |
| `reserved`       |    1 |                                                |
| `entries[]`      |  var | Depends on subtype                             |

**Header:** 8 bytes + variable entries

#### Entry Formats

| Subtype       | Entry Format                                    | Size |
| ------------- | ----------------------------------------------- | ---: |
| `leaf_gains`  | `(leaf_idx u8, gain_q15 i16)`                   |    3 |
| `leaf_coeffs` | `(leaf_idx u8, coeff_q15 i16)`                  |    3 |
| `events`      | `(event_id u8, mag_q15 i16, dur_blocks u8, flags u8)` | 5 |

**Receiver Rule:** Entries apply as deltas or absolute depending on subtype flag.

### 7.2 `VOICE_BURST` (Dense Payload)

For rare cases requiring many leaves at once.

| Field            | Size | Notes                    |
| ---------------- | ---: | ------------------------ |
| `apply_block_id` |    4 |                          |
| `voice_id`       |    1 |                          |
| `mode`           |    1 | 0=dense64, 1=rle, 2=topK |
| `payload_len`    |    2 |                          |
| `payload`        |  var |                          |

**Header:** 8 bytes + variable payload

**Dense64 Mode:** 64×int16 (128 bytes) for gains or coeffs.

---

## 8. Receiver-Side Application Rules (FPGA Closure Rules)

### 8.1 Per-Voice State Updates

For any voice-local parameter arriving on D-UART:

* Never hard-step: apply a per-voice slew unless event is explicitly transient-flagged.

### 8.2 Loss Handling (Data Channel)

If `VOICE_DELTA` is missing for a voice at a block:

1. **Hold** last leaf coeffs/gains for that voice for `HOLD_BLOCKS` (recommend 2–8)
2. **Decay** toward zero (or toward carrier-only) over `DECAY_BLOCKS` (recommend 8–64)

This prevents "random pops" on packet loss.

---

## 9. Channel Interaction Rules

### 9.1 Voice Identity

* C-UART allocates voices and announces mapping implicitly by `voice_id`.
* D-UART updates must reference a valid active `voice_id`.
* If D-UART references an inactive voice, FPGA ignores and increments an error counter.

### 9.2 Timebase Sync

* On `HELLO_ACK`, FPGA reports `block_id`.
* Host uses that to choose `apply_block_id = block_id + LEAD_BLOCKS`.

### 9.3 Session Start Order

1. Host sends `HELLO` on C-UART
2. FPGA replies `HELLO_ACK` with `block_id` and `credit_bytes`
3. Host sends `STREAM_CTRL(start)` on C-UART
4. Host sends `DATA_STREAM_START` on D-UART with `base_block_id`

---

## 10. Bandwidth Guidance

With D=6 and N=64, block rate is ~750 Hz. Dense 64-leaf payloads per voice per block would explode bandwidth.

The spec is designed for **sparse top-K updates**:

* Per voice per block: 4–12 leaf entries typical
* Events are even smaller

This makes 128 voices realistic.

### 10.1 Bandwidth Calculations

| Scenario              | Calculation                              | Bandwidth   |
| --------------------- | ---------------------------------------- | ----------- |
| 8 leaves × 128 voices | 8 × 3 bytes × 128 × 750 Hz               | ~2.3 Mbps   |
| 12 leaves × 64 voices | 12 × 3 bytes × 64 × 750 Hz               | ~1.7 Mbps   |
| Burst (dense64)       | 128 bytes × 750 Hz × voices              | Variable    |

---

## 11. Error Codes

Define `NACK.error_code` and `STATUS.last_error_code`:

| Code     | Meaning                                    |
| -------- | ------------------------------------------ |
| `0x0001` | Bad CRC                                    |
| `0x0002` | Bad length                                 |
| `0x0003` | Unknown msg_type                           |
| `0x0004` | Seq out of window                          |
| `0x0005` | Invalid voice_id                           |
| `0x0006` | apply_block too far in future (beyond buffer) |
| `0x0007` | D-UART overflow (drop)                     |

---

## 12. Default Parameters

| Parameter           | Value   | Notes                                      |
| ------------------- | ------- | ------------------------------------------ |
| `LEAD_BLOCKS`       | 8       | Scheduling lead time                       |
| `ACK_TIMEOUT_MS`    | 30      | Control message retransmit timeout         |
| `HOLD_BLOCKS`       | 4       | Hold last value on D-UART loss             |
| `DECAY_BLOCKS`      | 32      | Decay duration after hold expires          |
| `DATA_CREDIT` period| 1–4     | Blocks between credit updates              |

---

## 13. Message Type Enumeration

### 13.1 Control Messages (C-UART)

| msg_type          | Value  |
| ----------------- | -----: |
| `HELLO`           | `0x01` |
| `HELLO_ACK`       | `0x02` |
| `GET_STATUS`      | `0x03` |
| `STATUS`          | `0x04` |
| `SET_INSTRUMENT`  | `0x10` |
| `SET_TREE_TARGET` | `0x11` |
| `SET_MACRO`       | `0x12` |
| `NOTE_EVENT`      | `0x20` |
| `VOICE_POLICY`    | `0x21` |
| `STREAM_CTRL`     | `0x30` |
| `ACK`             | `0xF0` |
| `NACK`            | `0xF1` |

### 13.2 Data Messages (D-UART)

| msg_type            | Value  |
| ------------------- | -----: |
| `DATA_STREAM_START` | `0x01` |
| `DATA_STREAM_STOP`  | `0x02` |
| `VOICE_DELTA`       | `0x10` |
| `VOICE_BURST`       | `0x11` |
| `DATA_CREDIT`       | `0x20` |
| `DATA_DROP_NOTICE`  | `0x21` |
| `DATA_HEARTBEAT`    | `0xF0` |

---

## 14. References

* **WSE-STD-090** — Dual UART Transport & Control Plane (higher-level overview)
* **COBS** — Consistent Overhead Byte Stuffing (Stuart Cheshire, 1999)
* **CRC-32** — ISO 3309 polynomial

---

## Appendix A: Byte-Level Examples

### A.1 HELLO Message

```
Frame (COBS-encoded):
  [COBS payload] 0x00

Decoded payload (14 bytes header + 0 bytes payload):
  C0 DE           ; magic
  01              ; version
  00              ; chan=control
  01              ; msg_type=HELLO
  00              ; flags
  00 00           ; seq=0
  00 00           ; length=0
  XX XX XX XX     ; crc32
```

### A.2 NOTE_EVENT Message

```
Decoded payload (14 bytes header + 9 bytes payload):
  C0 DE           ; magic
  01              ; version
  00              ; chan=control
  20              ; msg_type=NOTE_EVENT
  00              ; flags
  05 00           ; seq=5
  09 00           ; length=9
  XX XX XX XX     ; crc32

  Payload:
  40 1F 00 00     ; apply_block_id=8000
  3C              ; note=60 (middle C)
  64              ; vel=100
  01              ; on=true
  00              ; chan=0
  00              ; flags=0
```

### A.3 VOICE_DELTA Message (4 leaf gains)

```
Decoded payload (14 bytes header + 8 + 12 bytes):
  C0 DE           ; magic
  01              ; version
  01              ; chan=data
  10              ; msg_type=VOICE_DELTA
  00              ; flags
  0A 00           ; seq=10
  14 00           ; length=20
  XX XX XX XX     ; crc32

  Payload:
  50 1F 00 00     ; apply_block_id=8016
  05              ; voice_id=5
  00              ; subtype=leaf_gains
  04              ; count=4
  00              ; reserved

  Entries (4 × 3 bytes):
  03 00 40        ; leaf_idx=3, gain_q15=0x4000 (0.5)
  07 80 7F        ; leaf_idx=7, gain_q15=0x7F80 (~1.0)
  0F 00 20        ; leaf_idx=15, gain_q15=0x2000 (0.25)
  1F 00 10        ; leaf_idx=31, gain_q15=0x1000 (0.125)
```

---

## Appendix B: State Machine Overview

### B.1 FPGA RX State Machine (Per UART)

```
States: IDLE → COLLECT → DECODE → VALIDATE → ENQUEUE → APPLY

IDLE:
  Wait for 0x00 delimiter
  → COLLECT

COLLECT:
  Accumulate bytes until 0x00
  → DECODE

DECODE:
  COBS decode
  If fail → IDLE (increment resync_count)
  → VALIDATE

VALIDATE:
  Check magic == 0xC0DE
  Check crc32
  Check length
  If fail → IDLE (increment error_count, send NACK if C-UART)
  → ENQUEUE

ENQUEUE:
  Add to block-indexed queue
  For C-UART: send ACK
  → IDLE

APPLY (on block tick):
  Process all messages where apply_block_id == current_block_id
  Update voice state
```

### B.2 Host TX State Machine (C-UART)

```
States: IDLE → SEND → WAIT_ACK → RETRY

IDLE:
  Message ready?
  → SEND

SEND:
  COBS encode, transmit
  Start ACK timer
  → WAIT_ACK

WAIT_ACK:
  ACK received for seq?
  → IDLE

  Timeout?
  retry_count < MAX_RETRIES?
  → RETRY
  Else → ERROR

RETRY:
  Increment retry_count
  → SEND
```
