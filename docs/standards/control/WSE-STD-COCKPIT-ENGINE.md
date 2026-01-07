# WSE-STD-COCKPIT-ENGINE v0 — Control Plane Standard

**Status:** Normative (v0 frozen)
**Applies to:** Python reference implementation, future FPGA control path
**Scope:** Control sampling, slew, binding, and frame emission

---

## 0. Purpose

The **Cockpit Engine** is the canonical control plane that converts all external inputs (Pots, MIDI, ImGui, serial deltas) into **pipeline-native objects** (VoiceFrame field modifications, note events).

This standard **replaces and consolidates**:
- `macro_runtime.py` (MacroRuntime with 11 system macros)
- `pico2_synth_bridge.py` (dead ADSRState/FilterState/OscillatorState)
- InstrumentIR L1 MacroSemantics (never evaluated)

### Invariant (must hold)

> **Cockpit samples async inputs into block-rate state. No downstream engine may observe sub-block timing jitter.**

This is the fly-by-wire principle: sampling + latching at the block boundary ensures determinism.

### Role in the System

CockpitEngine is the canonical control plane. UI modes (L0 museum mode, L3 performance mode, L5 editor mode) choose which controls are active and how they bind, but all control flow passes through Cockpit.

---

## 1. Scope

### What Cockpit Engine Handles (Inputs)
- **Pots** (Pico2 hardware potentiometers, 10 × Q15)
- **MIDI CC** (Continuous Controller messages, 0-127 → Q15)
- **MIDI Notes** (Note On/Off, velocity, pitch bend)
- **ImGui Knobs** (Software control surface, Q15)
- **Deltas** (Incremental changes from any source)

### What Cockpit Engine Produces (Outputs)

**Two output categories:**

1. **Continuous controls** (FrameModifier) — applied every block
   - Family scale factors (Q15)
   - Tree blend (Q15)
   - Gain scale (Q15)
   - ADSR params (blocks, Q15)
   - Pitch offset (cents)

2. **Events** (EventQueue) — consumed once then cleared
   - Note On (note, velocity_q15)
   - Note Off (note)
   - Preset change
   - Mode switch

3. **Global state** — persistent across blocks
   - Master gain (Q15)
   - Tempo (BPM, optional)

### What Cockpit Engine Does NOT Do
- **Execute operators** (VM/CMMM does this)
- **Synthesize audio** (VM + IDWPT does this)
- **Define presets** (PresetCompiler does this)

---

## 2. Canonical Enumerations (Normative)

### FamilyID (frozen)

All family references use this enum, not strings.

```python
class FamilyID(IntEnum):
    """Canonical operator family IDs. Do not reorder."""
    COLOR = 0           # Spectral tilt, brightness (Tilt operator)
    LOCAL = 1           # Intra-leaf mixing, texture (IntraMix)
    TRANSPORT = 2       # Energy diffusion (DiffuseUp/DiffuseDown)
    GLUE = 3            # Inter-leaf coherence (LeafBlend)
    TOPOLOGY = 4        # Band shearing, rotation (BandShear, RotateNodes)
    CATASTROPHE = 5     # Pressure ramps, cusp injection (FoldPressure)
    INJECT = 6          # Event injection operators
    TAIL = 7            # Tail safety operators (FineClamp, NormalizeScale)
    # Reserved: 8-15 for future families
```

### ControlID (frozen)

```python
class ControlID(IntEnum):
    """Canonical control IDs. Do not reorder."""
    # Timbral controls (0-4)
    TIMBRE = 0          # → FamilyID.COLOR scale
    TEXTURE = 1         # → FamilyID.LOCAL scale
    DYNAMICS = 2        # → Gain scale (post-envelope expression)
    MORPH = 3           # → tree_blend_q15
    CHAOS = 4           # → FamilyID.CATASTROPHE scale

    # Pitch controls (5-6)
    DETUNE = 5          # → pitch_offset_cents (additive with pitch bend)
    PITCH_BEND = 6      # → pitch_offset_cents (from MIDI, ±2 semitones default)

    # ADSR controls (7-10) — Cockpit owns ADSR params
    ATTACK = 7          # → attack_blocks
    DECAY = 8           # → decay_blocks
    SUSTAIN = 9         # → sustain_level_q15
    RELEASE = 10        # → release_blocks

    # Global controls (11-15)
    MASTER_GAIN = 11    # → global master_gain_q15
    RESERVED_12 = 12    # (Future: filter cutoff?)
    RESERVED_13 = 13    # (Future: filter reso?)
    RESERVED_14 = 14    # (Future: LFO rate?)
    RESERVED_15 = 15    # (Future: mod depth?)
```

### SlewPolicy (frozen)

```python
class SlewPolicy(IntEnum):
    """Slew filter policies."""
    INSTANT = 0         # No filtering, immediate update
    NBLOCK = 1          # Linear ramp over N blocks
    ONE_POLE = 2        # Exponential smoothing (alpha_q15)
```

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CONTROL INPUTS                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                  │
│  │  Pico2 Pots     │  │  MIDI Input     │  │  ImGui Knobs    │                  │
│  │  (10 × Q15)     │  │  (CC, Notes)    │  │  (Q15)          │                  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘                  │
│           │                    │                    │                            │
└───────────┼────────────────────┼────────────────────┼────────────────────────────┘
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       COCKPIT ENGINE (WSE-STD-COCKPIT-ENGINE)                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                        INPUT NORMALIZATION                                   ││
│  │  ─────────────────────────────────────────────────────────────────────────  ││
│  │  All inputs → ControlDelta(ControlID, value_q15: uint16, source)            ││
│  │                                                                              ││
│  │  Sources: POT, MIDI_CC, MIDI_NOTE, IMGUI, INTERNAL                          ││
│  │  All values normalized to Q15 (0-32767)                                     ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                 │                                                │
│                                 ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                        SLEW BANK (per control, integer-only)                 ││
│  │  ─────────────────────────────────────────────────────────────────────────  ││
│  │  For each ControlID:                                                         ││
│  │    target_q15: uint16 ← new value                                           ││
│  │    current_q15: uint16 ← slew(policy, current, target)                      ││
│  │                                                                              ││
│  │  Policies: INSTANT, NBLOCK(n: uint8), ONE_POLE(alpha_q15: uint16)           ││
│  │  tick() called once per block (750 Hz)                                      ││
│  │  ALL ARITHMETIC IS INTEGER (no floats)                                      ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                 │                                                │
│                                 ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                        BINDING TABLE (pure function)                         ││
│  │  ─────────────────────────────────────────────────────────────────────────  ││
│  │  Maps ControlID → pipeline target                                            ││
│  │                                                                              ││
│  │  Default bindings (frozen):                                                  ││
│  │    TIMBRE → family_scale_q15[FamilyID.COLOR]                                ││
│  │    TEXTURE → family_scale_q15[FamilyID.LOCAL]                               ││
│  │    DYNAMICS → gain_scale_q15                                                ││
│  │    MORPH → tree_blend_q15                                                   ││
│  │    CHAOS → family_scale_q15[FamilyID.CATASTROPHE]                           ││
│  │    DETUNE → pitch_offset_cents (additive)                                   ││
│  │    PITCH_BEND → pitch_offset_cents (additive)                               ││
│  │    ATTACK → attack_blocks                                                   ││
│  │    DECAY → decay_blocks                                                     ││
│  │    SUSTAIN → sustain_level_q15                                              ││
│  │    RELEASE → release_blocks                                                 ││
│  │    MASTER_GAIN → global.master_gain_q15                                     ││
│  │                                                                              ││
│  │  Binding is a pure function of (control_state, preset_binding_table)        ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                 │                                                │
│                                 ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                OUTPUT: Integer-Only Pipeline Objects                         ││
│  │  ─────────────────────────────────────────────────────────────────────────  ││
│  │                                                                              ││
│  │  FrameModifier (continuous, every block):                                    ││
│  │    family_scale_q15: [uint16; 8]  # Per FamilyID, 32767 = 1.0               ││
│  │    tree_blend_q15: uint16                                                   ││
│  │    gain_scale_q15: uint16                                                   ││
│  │    pitch_offset_cents: int16      # DETUNE + PITCH_BEND combined            ││
│  │    attack_blocks: uint16                                                    ││
│  │    decay_blocks: uint16                                                     ││
│  │    sustain_level_q15: uint16                                                ││
│  │    release_blocks: uint16                                                   ││
│  │                                                                              ││
│  │  EventQueue (consumed once, then cleared):                                   ││
│  │    note_on: Optional[(note: uint8, velocity_q15: uint16)]                   ││
│  │    note_off: Optional[note: uint8]                                          ││
│  │    preset_change: Optional[preset_id: uint16]                               ││
│  │                                                                              ││
│  │  GlobalState (persistent):                                                   ││
│  │    master_gain_q15: uint16                                                  ││
│  │    tempo_bpm_q8: uint16  # Q8.8 fixed-point BPM (optional)                  ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    VoiceFrameEmitter (applies modifiers)                         │
│  ────────────────────────────────────────────────────────────────────────────── │
│  emit_frame() with Cockpit integration:                                          │
│    1. Process EventQueue (note_on → allocate, note_off → release)               │
│    2. Apply FrameModifier to frame fields:                                       │
│       frame.families[fam] = mul_q15_sat(preset.families[fam], family_scale[fam])│
│       frame.tree_blend_q15 = modifier.tree_blend_q15                            │
│       frame.gain_q15 = mul_q15_sat(envelope_gain, modifier.gain_scale_q15)      │
│       frame.pitch_cents = note_cents + modifier.pitch_offset_cents              │
│    3. Apply GlobalState (master gain scaling)                                    │
│    4. Clear EventQueue                                                           │
│    5. Return modified VoiceFrame                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    VM / CMMM EXECUTOR (lanes.py)                                 │
│  ────────────────────────────────────────────────────────────────────────────── │
│  Consumes VoiceFrame microcode, produces audio                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Integer Arithmetic Rule (Normative)

All Cockpit computations use integer-only arithmetic:

```python
def mul_q15_sat(a: int, b: int) -> int:
    """Saturating Q15 multiply: (a * b) >> 15, clamped to [0, 32767]."""
    result = (a * b) >> 15
    return max(0, min(32767, result))
```

**No floats in the control path.** This ensures FPGA compatibility and bit-exact determinism.

---

## 4. Default Control Table (Normative)

| ControlID | Name | Binding Target | Slew Policy | Default Value |
|-----------|------|----------------|-------------|---------------|
| 0 | TIMBRE | family_scale_q15[COLOR] | ONE_POLE(2048) | 32767 (1.0) |
| 1 | TEXTURE | family_scale_q15[LOCAL] | ONE_POLE(2048) | 32767 (1.0) |
| 2 | DYNAMICS | gain_scale_q15 | ONE_POLE(2048) | 32767 (1.0) |
| 3 | MORPH | tree_blend_q15 | NBLOCK(16) | 0 |
| 4 | CHAOS | family_scale_q15[CATASTROPHE] | NBLOCK(32) | 0 |
| 5 | DETUNE | pitch_offset_cents | INSTANT | 0 |
| 6 | PITCH_BEND | pitch_offset_cents | INSTANT | 0 |
| 7 | ATTACK | attack_blocks | NBLOCK(4) | 8 (~10ms) |
| 8 | DECAY | decay_blocks | NBLOCK(4) | 32 (~42ms) |
| 9 | SUSTAIN | sustain_level_q15 | NBLOCK(4) | 24576 (0.75) |
| 10 | RELEASE | release_blocks | NBLOCK(4) | 64 (~85ms) |
| 11 | MASTER_GAIN | global.master_gain_q15 | ONE_POLE(4096) | 32767 (1.0) |
| 12-15 | RESERVED | (unassigned) | INSTANT | 0 |

### Pitch Offset Computation (Normative)

DETUNE and PITCH_BEND both contribute to a single pitch offset:

```
pitch_offset_cents = detune_cents + pitchbend_cents
```

Where:
- `detune_cents` is from ControlID.DETUNE (range: -2400 to +2400, ±2 octaves)
- `pitchbend_cents` is from MIDI pitch bend (range: -200 to +200, ±2 semitones default)

The combined pitch_offset_cents is added to the note's base pitch when computing f0_hz.

### Pico2 Pot → ControlID Mapping (Default)

| Pot | ControlID | Purpose |
|-----|-----------|---------|
| 0 | TIMBRE | Brightness / spectral tilt |
| 1 | TEXTURE | Body / density / intra-leaf |
| 2 | DYNAMICS | Expression / post-envelope |
| 3 | MORPH | Tree blend (topology morph) |
| 4 | CHAOS | Catastrophe / instability |
| 5 | DETUNE | Pitch offset (±2 octaves) |
| 6 | ATTACK | ADSR attack time |
| 7 | DECAY | ADSR decay time |
| 8 | SUSTAIN | ADSR sustain level |
| 9 | RELEASE | ADSR release time |

Note: MASTER_GAIN is typically controlled via ImGui or MIDI CC 7, not a dedicated pot.

---

## 5. MIDI Integration

### Note Messages

```
MIDI Note On → CockpitEngine.midi_note_on(note, velocity, channel)
             → EventQueue.note_ons.append(NoteOnEvent(note, velocity_q15))
             → VoiceFrameEmitter.note_on(note, velocity_q15)

MIDI Note Off → CockpitEngine.midi_note_off(note, channel)
              → EventQueue.note_offs.append(NoteOffEvent(note))
              → VoiceFrameEmitter.note_off()
```

### CC Messages

```
MIDI CC → CockpitEngine.midi_cc(cc_num, value)
        → Lookup cc_num in CC_BINDING_TABLE
        → set_control(control_id, (value * 32767) // 127)
        → Normal slew + binding flow
```

### Default CC Bindings

| CC Number | ControlID | Standard Use |
|-----------|-----------|--------------|
| 1 | DYNAMICS | Modulation wheel |
| 7 | MASTER_GAIN | Volume |
| 71 | TEXTURE | Resonance (MPE) |
| 74 | TIMBRE | Brightness (MPE) |

---

## 6. API (Integer-Only)

All types are integer. No floats.

```python
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional, List

# =============================================================================
# OUTPUT STRUCTURES (integer-only)
# =============================================================================

@dataclass
class FrameModifier:
    """
    Continuous control state applied every block.
    All fields are integers (Q15 or raw counts).
    """
    # Family scale factors: index by FamilyID, 32767 = 1.0
    family_scale_q15: List[int] = field(default_factory=lambda: [32767] * 8)

    # Frame fields
    tree_blend_q15: int = 0
    gain_scale_q15: int = 32767  # 1.0

    # Pitch (signed cents)
    pitch_offset_cents: int = 0  # DETUNE + PITCH_BEND

    # ADSR params (blocks / Q15)
    attack_blocks: int = 8
    decay_blocks: int = 32
    sustain_level_q15: int = 24576
    release_blocks: int = 64


@dataclass
class NoteOnEvent:
    """Note-on event (consumed once)."""
    note: int       # MIDI note 0-127
    velocity_q15: int  # Q15 velocity


@dataclass
class NoteOffEvent:
    """Note-off event (consumed once)."""
    note: int       # MIDI note 0-127


@dataclass
class EventQueue:
    """
    Events consumed once, then cleared.
    Events are processed in order: note_offs before note_ons.
    """
    note_ons: List[NoteOnEvent] = field(default_factory=list)
    note_offs: List[NoteOffEvent] = field(default_factory=list)
    preset_change: Optional[int] = None  # preset_id if changing

    def clear(self) -> None:
        """Clear all pending events after processing."""
        self.note_ons.clear()
        self.note_offs.clear()
        self.preset_change = None


@dataclass
class GlobalState:
    """Persistent global state (not per-voice)."""
    master_gain_q15: int = 32767  # Q15, 1.0
    tempo_bpm_q8: int = 0         # Q8.8 fixed-point, 0 = not set


# =============================================================================
# COCKPIT ENGINE
# =============================================================================

@dataclass
class CockpitEngine:
    """
    Unified control plane for symcrash pipeline.

    Converts async inputs (pots, MIDI, ImGui) into block-rate,
    integer-only pipeline objects.
    """

    def set_pot(self, pot_id: int, value_q15: int) -> None:
        """Set pot value. Maps pot_id to ControlID via pot mapping table."""

    def set_control(self, control_id: int, value_q15: int) -> None:
        """Set control directly by ControlID."""

    def midi_note_on(self, note: int, velocity: int, channel: int = 0) -> None:
        """Handle MIDI Note On. Queues NoteOnEvent."""

    def midi_note_off(self, note: int, channel: int = 0) -> None:
        """Handle MIDI Note Off. Queues NoteOffEvent."""

    def midi_cc(self, cc_num: int, value: int, channel: int = 0) -> None:
        """Handle MIDI CC. Maps to ControlID via CC binding table."""

    def midi_pitch_bend(self, value: int, channel: int = 0) -> None:
        """Handle MIDI Pitch Bend (-8192 to +8191)."""

    def tick(self) -> None:
        """Advance all slew filters by one block. MUST be called exactly once per block."""

    def get_frame_modifier(self) -> FrameModifier:
        """Get current frame modifier (continuous controls)."""

    def get_event_queue(self) -> EventQueue:
        """Get pending events. Caller MUST call clear() after processing."""

    def get_global_state(self) -> GlobalState:
        """Get global state (master gain, tempo)."""
```

---

## 7. Integration with VoiceFrameEmitter

```python
def process_block(cockpit: CockpitEngine, emitter: VoiceFrameEmitter, vm) -> None:
    """Main block processing loop with Cockpit integration. All arithmetic is integer-only."""

    # 1. Tick cockpit (advance slew filters)
    cockpit.tick()

    # 2. Process events (consumed once)
    events = cockpit.get_event_queue()

    # Process note-offs first (voice release before new allocation)
    for note_off in events.note_offs:
        emitter.note_off_by_note(note_off.note)

    # Process note-ons
    for note_on in events.note_ons:
        emitter.note_on(note_on.note, note_on.velocity_q15)

    # Handle preset change if requested
    if events.preset_change is not None:
        emitter.load_preset(events.preset_change)

    # Clear events after processing
    events.clear()

    # 3. Get continuous controls
    modifier = cockpit.get_frame_modifier()
    global_state = cockpit.get_global_state()

    # 4. Update emitter ADSR params from modifier
    emitter.set_adsr(
        attack_blocks=modifier.attack_blocks,
        decay_blocks=modifier.decay_blocks,
        sustain_level_q15=modifier.sustain_level_q15,
        release_blocks=modifier.release_blocks,
    )

    # 5. Emit frame
    frame = emitter.emit_frame()

    # 6. Apply frame modifier (integer-only arithmetic)
    for fam_id in range(8):  # FamilyID.COLOR through FamilyID.TAIL
        if fam_id in frame.families:
            frame.families[fam_id] = mul_q15_sat(
                frame.families[fam_id],
                modifier.family_scale_q15[fam_id]
            )

    frame.tree_blend_q15 = modifier.tree_blend_q15
    frame.gain_q15 = mul_q15_sat(frame.gain_q15, modifier.gain_scale_q15)
    frame.cents += modifier.pitch_offset_cents

    # 7. Apply global state
    frame.gain_q15 = mul_q15_sat(frame.gain_q15, global_state.master_gain_q15)

    # 8. Send to VM (CMMM Executor)
    vm.process_frame(frame)


def mul_q15_sat(a: int, b: int) -> int:
    """Saturating Q15 multiply."""
    result = (a * b) >> 15
    return max(0, min(32767, result))
```

---

## 8. Conformance Checklist (Normative)

An implementation conforms to WSE-STD-COCKPIT-ENGINE v0 iff:

### 8.1 Data Types
- [ ] All control values are integer (Q15 uint16 or signed int16 for pitch)
- [ ] No floating-point arithmetic in the control path
- [ ] All values are clamped to declared ranges before storage
- [ ] FamilyID and ControlID use the frozen enum values (no extensions in v0)

### 8.2 Determinism
- [ ] Slews are block-rate only (tick() called exactly once per block)
- [ ] Slew computation is integer-only and bit-exact
- [ ] Bindings are pure functions of (control_state, binding_table)
- [ ] Same input sequence → identical FrameModifier outputs

### 8.3 Event Handling
- [ ] Events are consumed exactly once per block
- [ ] EventQueue.clear() is called after processing
- [ ] Note-offs are processed before note-ons within a block
- [ ] Events do not repeat unless explicitly re-queued

### 8.4 MIDI Integration
- [ ] MIDI CC values (0-127) are converted to Q15 via: `(value * 32767) // 127`
- [ ] MIDI velocity (0-127) is converted to Q15 via: `(velocity * 32767) // 127`
- [ ] MIDI pitch bend (-8192 to +8191) maps to ±200 cents by default
- [ ] MIDI is an input adapter only; it does not bypass the control plane

### 8.5 Output Integrity
- [ ] FrameModifier.family_scale_q15[i] uses FamilyID as index (0-7)
- [ ] pitch_offset_cents = detune_cents + pitchbend_cents (additive)
- [ ] Global master_gain_q15 is applied after per-voice gain_scale_q15
- [ ] All Q15 multiplies use saturating arithmetic: `max(0, min(32767, (a*b) >> 15))`

---

## 9. One-Line Summary

> **CockpitEngine: inputs (pots/MIDI/GUI) → normalize → slew → bind → FrameModifier/Events → VoiceFrame fields (integer-only, deterministic)**

---

*WSE-STD-COCKPIT-ENGINE v0 — Frozen*
