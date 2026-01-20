# Macros and Controls Authoring Guide

Authoring guide for the macro and control systems in the symcrash synthesis engine.

**Normative spec:** WSE-STD-MACRO v0, WSE-STD-IPLC v0 (Section 6)

## Overview

The control system has three layers:

1. **System Macros (0-10)** - CMMM semantic control axes (BRIGHTNESS, DENSITY, RISK, etc.)
2. **Performance Macros (11-12)** - Modulation control (VIBRATO, LEGATO)
3. **Control Registry** - External control inputs mapped to voice parameters

---

## System Macros

System macros are internal state variables managed by the CMMM (Controlled Markov Mealy Machine). These are the semantic control axes that presets use to control synthesis behavior.

### Macro IDs (0-10)

| ID | Name | Purpose | Set By | Default Slew |
|----|------|---------|--------|--------------|
| 0 | ENERGY_STATE | Envelope-owned state | Envelope only | NONE |
| 1 | ENERGY_INTENT | Controller → CMMM, attack character | Controller | ONE_POLE (2048) |
| 2 | MOTION | Movement/modulation depth | Controller | ONE_POLE (2048) |
| 3 | FOCUS | Spectral concentration | Controller | ONE_POLE (2048) |
| 4 | BRIGHTNESS | Harmonic content (timbre) | Controller | ONE_POLE (2048) |
| 5 | DENSITY | Low-mid presence (body) | Controller | ONE_POLE (2048) |
| 6 | RISK | Stochastic behavior (chaos) | Controller | NBLOCK (16) |
| 7 | AGE | CMMM integrator | CMMM only | NONE |
| 8 | SPACE | Spatial/reverb character | Controller | ONE_POLE (2048) |
| 9 | MATERIAL | Surface texture (wood=low, glass=high) | Controller | NBLOCK (32) |
| 10 | UNITY | Voice coherence/blend | Controller | NBLOCK (32) |

### Preset Control Config

Presets can specify default values for system macros via the `control` section:

```yaml
instrument_ir_v0:
  control:
    energy_intent: 0.5      # 0.5 = neutral intensity
    motion: 0.0             # 0.0 = no modulation
    focus: 0.5              # 0.5 = balanced emphasis
    brightness: 0.75        # High for brass/bright sounds
    density: 0.5            # 0.5 = neutral body
    risk: 0.1               # Some instability
    space: 0.25             # Subtle spatial
    material: 0.5           # Neutral texture
    unity: 0.5              # Balanced blend
```

These values are pushed to MacroRuntime at preset load time via `ControlProgram`. The ControlConfig class in `instrument_ir.py` provides a `to_control_program()` method for capsule compilation.

### Preset Macro Mapping

Preset macros (defined in L0_graph.macros) map to system macros:

| Preset Macro | System Macro | Purpose |
|--------------|--------------|---------|
| `M_BRIGHT` | BRIGHTNESS (4) | Harmonic content |
| `M_BODY` | DENSITY (5) | Low-mid presence |
| `M_IMPACT` | ENERGY_INTENT (1) | Attack character |
| `M_GLITCH` | RISK (6) | Chaos/instability |
| `M_MORPH` | MORPH (3) | Tree blend factor |
| `M_TEXTURE` | MATERIAL (9) | Surface texture |
| `M_ATTACK` | ATK_BLOCKS (16) | Attack time |
| `M_DECAY` | DCY_BLOCKS (17) | Decay time |
| `M_SUSTAIN` | SUS_Q15 (18) | Sustain level |
| `M_RELEASE` | REL_BLOCKS (19) | Release time |
| `M_DETUNE` | DETUNE (5) | Pitch detune |

### Performance Macros (11-12)

| ID | Name | Purpose | Set By | Default Slew |
|----|------|---------|--------|--------------|
| 11 | VIBRATO | Vibrato/tremolo depth | Controller + MacroProgram | ONE_POLE (4096) |
| 12 | LEGATO | Legato/portamento mode | Controller + MacroProgram | NONE |

---

## CapsuleMacro Authoring

CapsuleMacro presets define reusable performance modulation that can be applied to any instrument.

### YAML Format

```yaml
preset:
  preset_id: "VIBRATO_CLASSIC_V0"
  preset_version: "v0"
  author: "symcrash"
  description: "Classic vibrato with delayed onset."

compat:
  voiceframe_std: "WSE-STD-020 v1"
  vm_std: "WSE-STD-VM-HYBRID v0"

kind: MACRO

macro:
  macro_type: VIBRATO           # VIBRATO | LEGATO | TREMOLO | PORTAMENTO

  # LFO parameters (VIBRATO, TREMOLO)
  rate_hz: 5.5                  # LFO rate in Hz
  depth_pitch_cents: 25         # Pitch wobble in cents
  depth_amp_q15: 1966           # Amplitude modulation depth (6%)
  depth_timbre_q15: 2621        # Timbre modulation depth (8%)

  # Envelope parameters
  delay_ms: 40.0                # Delay before modulation starts
  ramp_ms: 30.0                 # Ramp to full depth

  # Glide parameters (LEGATO, PORTAMENTO)
  glide_time_ms: 100.0          # Pitch glide duration
  glide_curve: 0                # 0=linear, 1=exp, 2=log

  # Phase behavior (LEGATO)
  preserve_phase: true          # Keep oscillator phase on note change

tags:
  - vibrato
  - classic
  - strings
```

### MacroType Options

| Type | Purpose | Parameters |
|------|---------|------------|
| **VIBRATO** | Pitch/amp/timbre LFO | rate_hz, depth_pitch_cents, depth_amp_q15, depth_timbre_q15, delay_ms, ramp_ms |
| **TREMOLO** | Amplitude-only LFO | rate_hz, depth_amp_q15, delay_ms, ramp_ms |
| **LEGATO** | Smooth transitions | preserve_phase |
| **PORTAMENTO** | Pitch glide | glide_time_ms, glide_curve |

### Apply Workflow

```python
from symcrash_python_ap.library.capsule_loader import load_capsule_yaml
from symcrash_python_ap.macro_runtime import MacroRuntime

# Load instrument first
violin = load_capsule_yaml("violin_ir_v1.yaml")
apply_instrument(violin)

# Load and apply vibrato macro (additive)
vibrato = load_capsule_yaml("vibrato_classic_v0.yaml")
runtime = MacroRuntime()
runtime.apply_macro_capsule(vibrato.macro)

# Controller can now modulate vibrato depth
runtime.set_system_macro(SystemMacro.VIBRATO, depth_q15)
```

---

## Control Registry

The control registry defines all external control inputs with ranges, defaults, and slew policies.

### Control IDs (Frozen v0)

#### Timbral Controls (0-4)

| ID | Name | Type | Range | Default | Slew | Binding Target |
|----|------|------|-------|---------|------|----------------|
| 0 | TIMBRE | Q15U | 0-32767 | 32767 | ONE_POLE (2048) | family_scale[COLOR] |
| 1 | TEXTURE | Q15U | 0-32767 | 32767 | ONE_POLE (2048) | family_scale[LOCAL] |
| 2 | DYNAMICS | Q15U | 0-32767 | 32767 | ONE_POLE (2048) | gain_scale_q15 |
| 3 | MORPH | Q15U | 0-32767 | 0 | NBLOCK (16) | tree_blend_q15 |
| 4 | CHAOS | Q15U | 0-32767 | 0 | NBLOCK (32) | family_scale[CATASTROPHE] |

#### Pitch Controls (5-6)

| ID | Name | Type | Range | Default | Slew | Binding Target |
|----|------|------|-------|---------|------|----------------|
| 5 | DETUNE | I16 | -2400 to 2400 | 0 | INSTANT | pitch_offset_cents |
| 6 | PITCH_BEND | I16 | -200 to 200 | 0 | INSTANT | pitch_offset_cents |

#### Global Controls (9)

| ID | Name | Type | Range | Default | Slew | Binding Target |
|----|------|------|-------|---------|------|----------------|
| 9 | MASTER_GAIN | Q15U | 0-32767 | 32767 | ONE_POLE (4096) | global.master_gain_q15 |

#### ADSR Controls (16-19)

| ID | Name | Type | Range | Default | Slew | Binding Target |
|----|------|------|-------|---------|------|----------------|
| 16 | ATK_BLOCKS | U16 | 1-500 | 8 | NBLOCK (4) | attack_blocks |
| 17 | DCY_BLOCKS | U16 | 1-1000 | 32 | NBLOCK (4) | decay_blocks |
| 18 | SUS_Q15 | Q15U | 0-32767 | 24576 | NBLOCK (4) | sustain_level_q15 |
| 19 | REL_BLOCKS | U16 | 1-1500 | 64 | NBLOCK (4) | release_blocks |

---

## Value Types

| ID | Type | Description | Range |
|----|------|-------------|-------|
| 0 | **Q15U** | Unsigned Q15 fixed-point | 0 to 32767 |
| 1 | **I16** | Signed 16-bit integer | -32768 to 32767 |
| 2 | **U16** | Unsigned 16-bit integer | 0 to 65535 |
| 3 | **BOOL** | Boolean flag | 0 or 1 |
| 4 | **ENUM8** | 8-bit enumeration | 0 to 255 |
| 5 | **U32** | Unsigned 32-bit integer | 0 to 4294967295 |
| 6 | **Q15S** | Signed Q15 fixed-point | -32768 to 32767 |
| 7 | **Q16.16** | Fixed-point ratio | 0 to 4294967295 |

### Q16.16 Format (FM6 Ratios)

FM6 operator frequency ratios use Q16.16 fixed-point where the integer part is the high 16 bits and the fractional part is the low 16 bits.

| Ratio | Q16.16 Value | Hex | Musical Interval |
|-------|--------------|-----|------------------|
| 0.5x | 32768 | 0x00008000 | Octave down |
| 1.0x | 65536 | 0x00010000 | Unison |
| 1.5x | 98304 | 0x00018000 | Fifth |
| 2.0x | 131072 | 0x00020000 | Octave up |
| 3.0x | 196608 | 0x00030000 | Octave + fifth |
| 4.0x | 262144 | 0x00040000 | Two octaves |
| 5.0x | 327680 | 0x00050000 | Two octaves + third |

**Note:** Ratios exceeding int16 range (values > 32767 or > 0.5x) require U32 storage.

---

## Slew Policies

Slew policies control how control values change over time.

### INSTANT
No filtering. Value changes immediately.
```
current = target
```

### NBLOCK(n)
Linear ramp over n blocks.
```
step = (target - current) / n
current += step  (each block)
```

### ONE_POLE(alpha)
Exponential smoothing with alpha coefficient (Q15).
```
current = current + ((target - current) * alpha) >> 15
```

Higher alpha = faster response. Typical values:
- 2048 (~6%): Smooth, ~17 block settling
- 4096 (~12%): Medium, ~8 block settling
- 8192 (~25%): Fast, ~4 block settling

---

## Q15 Fixed-Point

Q15 is a signed 16-bit fixed-point format with 15 fractional bits.

### Conversion

```python
from symcrash_python_ap.instrument_ir import pct_to_q15, q15_to_pct

# Percentage to Q15
pct_to_q15(0.0)   # → 0
pct_to_q15(0.5)   # → 16383
pct_to_q15(1.0)   # → 32767

# Q15 to percentage
q15_to_pct(0)     # → 0.0
q15_to_pct(16383) # → ~0.5
q15_to_pct(32767) # → 1.0
```

### Common Values

| Percentage | Q15 Value | Hex |
|------------|-----------|-----|
| 0% | 0 | 0x0000 |
| 10% | 3277 | 0x0CCD |
| 25% | 8192 | 0x2000 |
| 50% | 16384 | 0x4000 |
| 75% | 24576 | 0x6000 |
| 100% | 32767 | 0x7FFF |

---

## Control Frame

A ControlFrame is the block-rate snapshot of all control state.

### Structure

```python
@dataclass
class ControlFrame:
    block_index: int = 0
    mode_L: int = 3              # Performance mode (L0-L5)
    flags: int = 0

    globals: GlobalControls       # Global state
    voice_patches: List[VoiceControlPatch]  # Per-voice updates
    events: List[ControlEvent]    # Discrete events
```

### GlobalControls

```python
@dataclass
class GlobalControls:
    master_gain_q15: int = 32767  # Master output gain
    tempo_bpm_q8: int = 120 << 8  # Tempo in Q8 (120.0 BPM)
    beat_phase_q15: int = 0       # Beat position 0-32767
```

### VoiceControlPatch

Sparse updates to specific voices:

```python
@dataclass
class VoiceControlPatch:
    voice_id: int
    items: List[ControlItem]

@dataclass
class ControlItem:
    control_id: int      # ControlID enum
    value_type: int      # ValueType enum
    value: int           # The value
```

---

## Control Events

Discrete events that occur at block boundaries.

### Event Types

| Type | ID | Purpose |
|------|----|---------|
| NOTE_ON | 0 | Trigger note with velocity |
| NOTE_OFF | 1 | Release note |
| PRESET_LOAD | 2 | Load preset by ID |
| MODE_SET | 3 | Set performance mode (L0-L5) |
| CAPTURE_MARK | 4 | Debug marker |

### Example: Note On

```python
frame = ControlFrame(block_index=42)
frame.add_note_on(
    note=60,                # MIDI note (C4)
    velocity_q15=32767      # Full velocity
)
```

---

## VoiceFrame Emission

The VoiceFrameEmitter bridges preset config and control input to produce VoiceFrames.

### Gain Chain

```
gain = envelope_gain
gain = (gain * velocity_q15) >> 15        # Velocity scaling
gain = (gain * _gain_scale_q15) >> 15     # DYNAMICS control
gain = (gain * _master_gain_q15) >> 15    # Master gain
```

### Pitch Chain

```
f0_hz = midi_to_hz(note, cents + pitch_offset)
# where pitch_offset = DETUNE + PITCH_BEND cumulative
```

### Beta (Dry/Wet) Ramp

Preset-driven ramp over `beta_ramp_blocks`:

```
beta(t) = beta_start + (beta_target - beta_start) * (t / ramp_blocks)
```

---

## Standard Macro IDs in Presets

When authoring presets, use these standard macro IDs:

| Macro ID | Control Binding | Purpose |
|----------|-----------------|---------|
| `M_BRIGHT` | TIMBRE (0) | Harmonic brightness |
| `M_ATTACK` | ATK_BLOCKS (16) | Attack time |
| `M_DECAY` | DCY_BLOCKS (17) | Decay time |
| `M_SUSTAIN` | SUS_Q15 (18) | Sustain level |
| `M_RELEASE` | REL_BLOCKS (19) | Release time |
| `M_GLITCH` | CHAOS (4) | Glitch/chaos amount |
| `M_MORPH` | MORPH (3) | Tree blend factor |
| `M_TEXTURE` | TEXTURE (1) | Surface texture |
| `M_DETUNE` | DETUNE (5) | Fine pitch adjustment |

---

## YAML Macro Section

In preset YAML files:

```yaml
macros:
  BRIGHT:
    target: brightness
    default_q15: 16384       # 0.5
  BODY:
    target: density
    default_q15: 16384       # 0.5
  IMPACT:
    target: energy_intent
    default_q15: 0           # Not used initially
```

---

## Code Examples

### Reading Control Values

```python
from symcrash_python_ap.control_registry import (
    ControlRegistry, ControlID, get_default_registry
)

registry = get_default_registry()

# Get entry for a control
entry = registry.get_entry(ControlID.TIMBRE)
print(f"Range: {entry.min_value} - {entry.max_value}")
print(f"Default: {entry.default_value}")

# Clamp value to valid range
clamped = registry.clamp(ControlID.DYNAMICS, 40000)  # → 32767
```

### Applying Control Frame

```python
from symcrash_python_ap.voiceframe_emitter import VoiceFrameEmitter

emitter = VoiceFrameEmitter(preset=my_preset)

# Each block:
control_frame = cockpit.tick()
emitter.consume_control_frame(control_frame)
voice_frame = emitter.emit_frame()
```

### Slewing Values

```python
from symcrash_python_ap.control_registry import slew_nblock, slew_one_pole

# Linear ramp over 16 blocks
current = slew_nblock(current, target, n=16)

# Exponential smoothing
current = slew_one_pole(
    current, target,
    alpha_q15=2048,      # Smoothing factor
    min_val=0,
    max_val=32767
)
```

---

## CMMM Event Integration (WSE-STD-070)

The control system integrates with CMMM events for runtime mask and tail mode control.

### VoiceFrame CMMM Fields

When CMMM emits a VoiceFrame, it can include event payloads:

```python
# Node mask control (0x30 NODEMASK_SET)
frame.cmmm_node_mask = 0x0000FFFF      # Target 16-node mask
frame.cmmm_mask_ramp_blocks = 5         # Crossfade duration

# Tail mode control (0x31 TAILMODE_SET)
frame.cmmm_tail_mode = TailMode.MIN     # Tail operator set
frame.cmmm_traction_q15 = 16384         # Projection hardness
```

### Control Flow

```
ControlFrame → VoiceFrameEmitter → VoiceFrame → Executor
                     ↓                              ↓
                CMMM emission                ExecutorLatch
                     ↓                              ↓
              cmmm_node_mask              mask crossfade
              cmmm_tail_mode              tail op selection
```

### ExecutorLatch State

The executor latch maintains per-voice CMMM state:

| Field | Purpose |
|-------|---------|
| `node_mask_u32` | Current active node mask |
| `mask_target_u32` | Target mask for ramping |
| `mask_ramp_remaining` | Blocks left in crossfade |
| `tail_mode` | Current tail operator set |
| `traction_q15` | Projection hardness |

### Macro → CMMM Mapping

System macros can influence CMMM event emission:

| Macro | Affects | Example |
|-------|---------|---------|
| RISK | tail_mode | High risk → MIN/EMERGENCY tail |
| ENERGY_STATE | node_mask | Low energy → reduced K |
| DENSITY | node_mask | High density → more nodes |

---

## Related Documentation

- [PRESET-AUTHORING-GUIDE.md](PRESET-AUTHORING-GUIDE.md) - Main authoring guide
- [OPERATORS.md](OPERATORS.md) - Operator reference
- [BUDGETS-AND-DEGRADATION.md](BUDGETS-AND-DEGRADATION.md) - Performance budgets
- [../architecture/WSE-STD-070.md](../architecture/WSE-STD-070.md) - Tower completion and CMMM events
