# **WSE-STD-090 — Dual UART Transport & Control Plane**

**Status:** Normative
**Applies to:** Pico (UI/Control Host), FPGA (Audio Engine Host)
**Scope:** Defines the dual-UART architecture, roles, packet classes, timing guarantees, and failure semantics for all control, MIDI, and editor interaction with the FPGA.

---

## 1. Purpose

This standard defines a **two-channel UART transport** that cleanly separates:

* **Hard real-time musical control** from
* **Soft real-time configuration, editing, and telemetry**

The goal is:

> **No UI, editor, or diagnostic traffic may ever interfere with audio determinism.**

---

## 2. Physical Links

### 2.1 UART Links

| UART       | Name                          | Direction             | Role                    |
| ---------- | ----------------------------- | --------------------- | ----------------------- |
| **UART-A** | Real-Time Control Bus         | Pico → FPGA (primary) | Voice / musical control |
| **UART-B** | Configuration & Telemetry Bus | Bi-directional        | Editor, presets, debug  |

Both UARTs are **electrically independent**.

---

### 2.2 Electrical & Timing Parameters (Freeze)

| Parameter    | Value                   |
| ------------ | ----------------------- |
| Baud rate    | 921 600 bps             |
| Data         | 8N1                     |
| Flow control | None                    |
| Clock source | Pico UART hardware      |
| Framing      | Explicit packet headers |

---

## 3. Ownership Model (Normative)

| Plane  | Owns                            | Cannot Affect             |
| ------ | ------------------------------- | ------------------------- |
| UART-A | Musical time, voices, notes     | Scheduling, memory layout |
| UART-B | Programs, presets, editor state | Audio determinism         |

**Rule:**

> FPGA audio execution MUST NOT stall or branch on UART-B traffic.

---

## 4. UART-A: Real-Time Control Plane

### 4.1 Traffic Class

UART-A carries **only time-critical, block-aligned intent**.

Allowed packets:

* `VoiceFramePacket v0b`
* `NoteOn / NoteOff`
* `Global Clock / Reset (optional)`
* `Emergency Kill`

Forbidden packets:

* Preset load
* Parameter enumeration
* Debug logs
* Telemetry streams

---

### 4.2 Timing Guarantees

* Packets are interpreted at **BlockTime** boundaries.
* Late packets apply to the **next block**.
* Dropped packets do **not** stall execution.

**Rule:**

> Missing a VoiceFrame is preferable to blocking the audio pipeline.

---

### 4.3 Determinism Clause

UART-A packet processing MUST be:

* O(1) per packet
* Allocation-free
* Stateless beyond frame sequencing

---

## 5. UART-B: Configuration & Telemetry Plane

### 5.1 Traffic Class

UART-B is **best-effort, asynchronous**.

Allowed packets:

* Program load / unload
* InstrumentIR transfer
* Parameter registry queries
* Health telemetry
* Debug snapshots
* Editor introspection
* Firmware handshake

---

### 5.2 Backpressure & Safety

* FPGA may **drop UART-B packets** under load.
* Pico must retry idempotent commands.
* No UART-B command may force an immediate audio state change.

**Rule:**

> All UART-B effects must be mediated through future UART-A frames.

---

## 6. Packet Class Separation

### 6.1 Canonical Packet Classes

| Class            | UART | Description                   |
| ---------------- | ---- | ----------------------------- |
| VoiceFramePacket | A    | Block-level musical execution |
| ParamDelta       | A    | Small real-time tweaks        |
| ProgramBlob      | B    | InstrumentIR, presets         |
| RegistryQuery    | B    | Parameter metadata            |
| TelemetryFrame   | B    | Health, stats                 |
| DebugFrame       | B    | Optional                      |

---

### 6.2 Cross-Plane Rule

UART-B packets **may modify state**, but **never directly influence sound**.

Instead:

> UART-B → updates compiler state → reflected later in UART-A execution frames.

---

## 7. MIDI Integration

### 7.1 MIDI Routing

MIDI is treated as **an input source**, not a transport.

Two allowed models:

* MIDI → Pico → IntentFrame → UART-A
* MIDI → Pico → ParamDelta → UART-A

**Rule:**

> MIDI never talks directly to the FPGA audio engine.

---

## 8. Failure Modes & Recovery

### 8.1 UART-A Failure

* FPGA continues last valid VoiceFrame
* Gain ramps to zero if gate lost for N blocks
* No crash, no undefined state

### 8.2 UART-B Failure

* Editor desync allowed
* Audio continues uninterrupted
* Reconnect is stateless

---

## 9. Security / Robustness

* All packets include:

  * Magic
  * Version
  * Length
  * CRC or checksum (implementation-defined)
* Invalid packets are dropped silently.

---

## 10. Rationale (Why Dual UART)

| Problem              | Single UART  | Dual UART  |
| -------------------- | ------------ | ---------- |
| UI spam              | Breaks audio | Isolated   |
| Editor introspection | Risky        | Safe       |
| Determinism          | Fragile      | Guaranteed |
| Debug                | Dangerous    | Harmless   |

**Conclusion:**
Dual UART is not an optimization — it is an **architectural boundary**.

---

## 11. Compliance Checklist

An implementation is compliant with WSE-STD-080 iff:

* Audio does not stall if UART-B is flooded
* VoiceFrames are only accepted on UART-A
* MIDI never bypasses Pico
* UART-B cannot introduce mid-block changes
* Both UARTs operate independently

---

## 12. One-Line Summary

> **UART-A moves music. UART-B moves knowledge. Neither is allowed to confuse the other.**

---

If you want, next I can:

* Write **WSE-STD-081: UART Packet Arbitration & Priority**, or
* Provide a **reference Pico firmware layout** (tasks, buffers, IRQ model), or
* Generate a **timing budget proof** that 921 600 bps is sufficient for worst-case VoiceFrame throughput.

Just tell me.
