# WSE-STD-CONTROLFRAME v0 — ControlFrame ABI (Cockpit → VoiceFrameEmitter)

**Status:** Normative (v0 frozen)
**Applies to:** CockpitEngine output, VoiceFrameEmitter input, logging/replay
**Scope:** Block-rate control snapshot format

---

## 0. Purpose

A **ControlFrame** is the **block-rate, deterministic snapshot** of all control intent and events destined for the instrument pipeline.

It is produced by **CockpitEngine** once per block and consumed by **VoiceFrameEmitter** (and optionally logged/recorded).

**ControlFrame is not a preset.**
It is not audio.
It is the **fly-by-wire command surface** for one block.

---

## 1. Timebase

* **BlockTime** is the unit of time.
* `BLOCK_SAMPLES = 64` (matches VoiceFrame standard).
* A ControlFrame MUST be emitted at most **once per block**.

---

## 2. Data Model

### 2.1 ControlFrame_v0 (logical fields)

```c
ControlFrame_v0 {
  uint32 block_index;     // monotonic; wraps allowed
  uint8  mode_L;          // 0..5 (L0..L5), optional but recommended
  uint8  flags;           // bitfield: e.g. cockpit_enabled, record_mark, etc.

  // Global controls (apply to all voices unless overridden)
  GlobalControls globals;

  // Per-voice control updates (sparse list)
  uint8 voice_count;
  VoiceControlPatch voice_patch[voice_count];

  // Event queue (sparse list)
  uint8 event_count;
  ControlEvent event[event_count];
}
```

Notes:

* This is intentionally **sparse** (only changed controls need to be sent/stored).
* If you want "full state every block" for debugging, that is a different artifact (`ControlFrame_FULL`) and not normative v0.

---

## 3. Control IDs and Values

### 3.1 ControlID namespace (frozen v0)

```c
enum ControlID : uint16 {
  // Timbral controls (0-4)
  CTRL_TIMBRE      = 0,
  CTRL_TEXTURE     = 1,
  CTRL_DYNAMICS    = 2,
  CTRL_MORPH       = 3,
  CTRL_CHAOS       = 4,

  // Pitch controls (5-6)
  CTRL_DETUNE      = 5,
  CTRL_PITCH_BEND  = 6,

  // Reserved (7-15)
  CTRL_RESERVED_7  = 7,
  CTRL_RESERVED_8  = 8,
  CTRL_MASTER_GAIN = 9,

  // ADSR controls (16-19)
  CTRL_ATK_BLOCKS  = 16,
  CTRL_DCY_BLOCKS  = 17,
  CTRL_SUS_Q15     = 18,
  CTRL_REL_BLOCKS  = 19,

  // Reserved expansion (20+)
  CTRL_RESERVED_20 = 20,
};
```

### 3.2 Value encoding

All continuous controls are **Q15 unsigned** unless otherwise specified:

* `value_q15`: `uint16` in `[0..32767]`

Some controls are naturally signed or integer:

* detune cents: `int16 detune_cents` (-2400 to +2400)
* pitch bend: `int16 pitchbend_cents` (-200 to +200)
* block counts: `uint16 blocks` (attack/decay/release)

To keep v0 simple, we use a tagged union per control update.

---

## 4. GlobalControls

```c
GlobalControls {
  uint16 master_gain_q15;      // 0..32767
  int16  master_pan_q15;       // optional, -32768..32767 (or omit in v0)
  uint16 tempo_q8_8;           // optional fixed-point tempo (Q8.8)
}
```

**Normative:** If a global field is omitted, the consumer MUST keep the previous value (latch semantics).

---

## 5. Per-voice Patches

Per-voice updates are carried as sparse "patches":

```c
VoiceControlPatch {
  uint16 voice_id;

  uint8  n_items;
  ControlItem item[n_items];
}
```

Each `ControlItem` is one control update:

```c
ControlItem {
  uint16 control_id;
  uint8  value_type;   // 0=Q15U, 1=I16, 2=U16, 3=BOOL, 4=ENUM8
  uint8  reserved;

  union {
    uint16 q15u;
    int16  i16;
    uint16 u16;
    uint8  b8;
    uint8  e8;
  } value;
}
```

**Normative:** Values MUST be clamped to legal ranges for that `control_id` before emission.

---

## 6. Events

Events are discrete commands that must not "stick."

```c
enum EventType : uint8 {
  EVT_NOTE_ON      = 0,
  EVT_NOTE_OFF     = 1,
  EVT_PRESET_LOAD  = 2,
  EVT_MODE_SET     = 3,
  EVT_CAPTURE_MARK = 4,
};
```

```c
ControlEvent {
  uint8  type;
  uint8  channel;      // MIDI channel or logical group
  uint16 voice_id;     // 0xFFFF = "allocator decides" (recommended)
  uint32 a;            // type-specific payload
  uint32 b;            // type-specific payload
}
```

### Type Payloads (v0)

| Event Type | Field `a` | Field `b` |
|------------|-----------|-----------|
| `EVT_NOTE_ON` | `note (0..127) \| (vel_q15 << 8)` | flags (legato, retrigger, etc.) |
| `EVT_NOTE_OFF` | `note (0..127)` | flags |
| `EVT_PRESET_LOAD` | `preset_id` | `preset_hash32` (optional) |
| `EVT_MODE_SET` | `mode_L (0..5)` | 0 |
| `EVT_CAPTURE_MARK` | `mark_id` | 0 |

**Normative:** Events MUST be consumed once and then cleared; if a ControlFrame is replayed, events replay identically.

---

## 7. Completion Rules

### 7.1 Control completion (latch semantics)

If a control is not present in the current ControlFrame, the consumer MUST use the last latched value for that voice/global.

### 7.2 Voice allocation reset

On voice allocation:

* All controls for that voice initialize to preset defaults (or registry defaults).
* ADSR defaults apply if present in registry.
* Any missing values come from defaults, not "random."

---

## 8. Relationship to Other Standards

```
CockpitEngine → ControlFrame_v0 (this standard)
                    ↓
VoiceFrameEmitter (consumes ControlFrame + PresetProgram)
                    ↓
              VoiceFrame_v1 (WSE-STD-020)
                    ↓
              VM / CMMM Executor (lanes.py)
```

**ControlFrame is upstream of VoiceFrame.**
VoiceFrame is downstream, block-aligned, executor-ready.

---

## 9. Minimal Binary Encoding (optional but recommended)

If you want to serialize ControlFrame (UART/logging), use:

```
Header:
  uint32 block_index
  uint8  mode_L
  uint8  flags

GlobalControls:
  uint8  present_bitmask    // which global fields are present
  (conditional fields based on bitmask)

Voice Patches:
  uint8  voice_count
  repeat voice_count:
    uint16 voice_id
    uint8  n_items
    repeat n_items:
      uint16 control_id
      uint8  value_type
      uint8  reserved
      uint16 value          // 2 bytes, interpret based on value_type

Events:
  uint8  event_count
  repeat event_count:
    uint8  type
    uint8  channel
    uint16 voice_id
    uint32 a
    uint32 b
```

### Size Estimate

Minimal ControlFrame (no changes, no events):
- Header: 6 bytes
- GlobalControls: 1 byte (empty bitmask)
- Voice patches: 1 byte (count=0)
- Events: 1 byte (count=0)
- **Total: 9 bytes**

Typical ControlFrame (2 control changes, 1 note on):
- Header: 6 bytes
- GlobalControls: 3 bytes (1 field)
- Voice patches: 1 + 2 + 1 + (2 × 6) = 16 bytes
- Events: 1 + 12 = 13 bytes
- **Total: ~38 bytes**

---

## 10. Python Reference Implementation

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import IntEnum


class ValueType(IntEnum):
    Q15U = 0      # Q15 unsigned: 0..32767
    I16 = 1       # Signed int16: -32768..32767
    U16 = 2       # Unsigned int16: 0..65535
    BOOL = 3      # Boolean: 0 or 1
    ENUM8 = 4     # Enumeration: 0..255


class EventType(IntEnum):
    NOTE_ON = 0
    NOTE_OFF = 1
    PRESET_LOAD = 2
    MODE_SET = 3
    CAPTURE_MARK = 4


@dataclass
class ControlItem:
    """Single control update."""
    control_id: int
    value_type: ValueType
    value: int


@dataclass
class VoiceControlPatch:
    """Sparse control updates for one voice."""
    voice_id: int
    items: List[ControlItem] = field(default_factory=list)


@dataclass
class ControlEvent:
    """Discrete event (consumed once)."""
    type: EventType
    channel: int = 0
    voice_id: int = 0xFFFF  # 0xFFFF = allocator decides
    a: int = 0
    b: int = 0


@dataclass
class GlobalControls:
    """Global control state."""
    master_gain_q15: Optional[int] = None
    master_pan_q15: Optional[int] = None
    tempo_q8_8: Optional[int] = None


@dataclass
class ControlFrame:
    """
    Block-rate control snapshot.

    Produced by CockpitEngine, consumed by VoiceFrameEmitter.
    """
    block_index: int = 0
    mode_L: int = 3  # Default to L3 (performance mode)
    flags: int = 0

    globals: GlobalControls = field(default_factory=GlobalControls)
    voice_patches: List[VoiceControlPatch] = field(default_factory=list)
    events: List[ControlEvent] = field(default_factory=list)

    def clear_events(self) -> None:
        """Clear events after processing."""
        self.events.clear()

    def add_note_on(self, note: int, velocity_q15: int, channel: int = 0) -> None:
        """Add note-on event."""
        self.events.append(ControlEvent(
            type=EventType.NOTE_ON,
            channel=channel,
            voice_id=0xFFFF,
            a=note | (velocity_q15 << 8),
            b=0
        ))

    def add_note_off(self, note: int, channel: int = 0) -> None:
        """Add note-off event."""
        self.events.append(ControlEvent(
            type=EventType.NOTE_OFF,
            channel=channel,
            voice_id=0xFFFF,
            a=note,
            b=0
        ))
```

---

## 11. Why This Is The Right Shape

| Property | Benefit |
|----------|---------|
| **Deterministic** | Integer-only, block-rate, explicit latch semantics |
| **Composable** | Cockpit and presets don't fight; emitter merges them |
| **Efficient** | Sparse patches, sparse events |
| **Replayable** | Record ControlFrames and reproduce behavior exactly |
| **Debuggable** | Clear separation between control intent and audio output |

---

## 12. Conformance Checklist

An implementation conforms to WSE-STD-CONTROLFRAME v0 iff:

- [ ] ControlFrame is emitted at most once per block
- [ ] All control values are clamped to legal ranges before emission
- [ ] Events are consumed exactly once per block
- [ ] Latch semantics: missing controls use last latched value
- [ ] Voice allocation initializes to preset/registry defaults
- [ ] ControlID uses frozen enum values (0-19 in v0)
- [ ] ValueType is correctly tagged for each ControlItem

---

*WSE-STD-CONTROLFRAME v0 — Frozen*
