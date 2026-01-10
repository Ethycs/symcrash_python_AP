# Preset Authoring Guide

Authoring guide for creating presets using the InstrumentIR API.

**Normative spec:** WSE-STD-PRESET v0 (format + compile semantics), WSE-STD-IPLC v0 (lifecycle)

This guide covers the **Python API** for creating presets. For the raw YAML/JSON schema and canonical examples, see [WSE-STD-PRESET-v0.md](../control/WSE-STD-PRESET-v0.md).

## Quick Start

The fastest way to create a preset is using the factory methods:

```python
from symcrash_python_ap.instrument_ir import InstrumentIR

# One-liner preset creation
ir = InstrumentIR.quick_synth("Bright Saw", brightness=0.8)
capsule = ir.to_capsule()

# Or chain it directly
capsule = InstrumentIR.quick_pad("Warm Pad", brightness=0.3).to_capsule()
```

### Factory Methods

| Method | Attack | Decay | Sustain | Release | Character |
|--------|--------|-------|---------|---------|-----------|
| `quick_synth()` | 10ms | 100ms | 0.7 | 200ms | General purpose |
| `quick_pad()` | 300ms | 200ms | 0.8 | 800ms | Slow, ambient |
| `quick_pluck()` | 2ms | 150ms | 0.0 | 100ms | Percussive |
| `quick_bass()` | 5ms | 80ms | 0.6 | 150ms | Punchy, dark |
| `quick_lead()` | 15ms | 100ms | 0.85 | 200ms | Sustained, bright |

All factory methods accept these parameters:

```python
InstrumentIR.quick_synth(
    name: str,                      # Preset name (auto-generates ID)
    waveform: str = "saw",          # "sine" | "saw" | "pulse" | "tri"
    attack_ms: float = 10,          # Attack time
    decay_ms: float = 100,          # Decay time
    sustain: float = 0.7,           # Sustain level (0.0-1.0)
    release_ms: float = 200,        # Release time
    brightness: float = 0.5,        # Timbre brightness (0.0-1.0)
    author: str = ""                # Optional author
)
```

---

## The Layer System

InstrumentIR uses a 4-layer tower architecture. Each layer compiles to the next:

```
L0 (DSP Graph)     →  Familiar patch workflow (OSC → FILTER → AMP)
    ↓ compile
L1 (Gesture Lift)  →  Maps gestures to macro meanings
    ↓ compile
L2 (Regime Graph)  →  State machine with transitions
    ↓ compile
L3 (Emit Templates)→  Per-block VoiceFrame emission rules
```

For most presets, you only need to define **L0**. The compiler auto-completes L1/L2/L3 with sensible defaults.

For advanced presets with custom state machines, you can define **L2** (regime graph) directly in YAML.

---

## L0: DSP Graph

The L0 layer defines the signal flow using nodes and edges.

### Node Types

#### Oscillator (Osc)
```python
L0Node(
    id="osc1",
    type="Osc",
    params=NodeParams(
        waveform="saw",           # "sine" | "saw" | "pulse" | "tri"
        detune_cents=0.0,         # Fine tuning in cents
        pw=0.5,                   # Pulse width (pulse only)
        role="audible",           # "off" | "audible" | "exciter_only"
        archetype="harmonic"      # "harmonic" | "bright" | "hollow" | "noisy" | "transient"
    )
)
```

#### Noise Generator (Noise)
```python
L0Node(
    id="noise1",
    type="Noise",
    params=NodeParams(
        color="white"             # "white" | "pink" | "brown"
    )
)
```

#### Filter (Filter)
```python
L0Node(
    id="filter1",
    type="Filter",
    params=NodeParams(
        mode="lp"                 # "lp" | "hp" | "bp" | "notch"
    )
)
```

#### Amplifier with Envelope (Amp)
```python
L0Node(
    id="amp1",
    type="Amp",
    params=NodeParams(
        adsr=ADSR(
            a_ms=10.0,            # Attack (milliseconds)
            d_ms=100.0,           # Decay (milliseconds)
            s=0.7,                # Sustain level (0.0-1.0)
            r_ms=200.0            # Release (milliseconds)
        )
    )
)
```

#### Mixer (Mixer)
```python
L0Node(id="mix1", type="Mixer")
```

#### FX Send (FXSend)
```python
L0Node(
    id="fx1",
    type="FXSend",
    params=NodeParams(
        send_a=0.3,               # Send A level (0.0-1.0)
        send_b=0.0                # Send B level (0.0-1.0)
    )
)
```

### Connections (Edges)

Connect nodes using edges:

```python
ir.layers.L0_graph.edges = [
    L0Edge(src="osc1", dst="filter1"),
    L0Edge(src="filter1", dst="amp1"),
]
```

### Macros

Macros expose parameters for real-time control:

```python
ir.layers.L0_graph.macros = [
    Macro(id="M_BRIGHT", name="Brightness", default=0.5),
    Macro(id="M_CUTOFF", name="Filter Cutoff", default=0.7),
]
```

#### Standard Macro IDs

| Macro ID | Maps To | Purpose |
|----------|---------|---------|
| `M_BRIGHT` | TIMBRE | Harmonic brightness |
| `M_ATTACK` | ATK_BLOCKS | Attack time |
| `M_DECAY` | DCY_BLOCKS | Decay time |
| `M_SUSTAIN` | SUS_Q15 | Sustain level |
| `M_RELEASE` | REL_BLOCKS | Release time |
| `M_GLITCH` | CHAOS | Glitch/chaos amount |
| `M_MORPH` | MORPH | Tree blend |
| `M_TEXTURE` | TEXTURE | Surface texture |
| `M_DETUNE` | DETUNE | Pitch detune |

---

## L2: Regime Graph (State Machine)

For presets that need custom behavioral states, define an L2 regime graph. This controls how the voice transitions between different emission modes (attack, sustain, glitch, etc.).

### Regime Kinds

| Kind | Purpose | Typical Use |
|------|---------|-------------|
| `Idle` | No note active | Waiting for note-on |
| `Transient` | Attack phase | Fast onset, high energy |
| `StableTone` | Sustained phase | Full processing budget |
| `NoiseTexture` | Noise-based | Breath, wind, texture |
| `GlitchChatter` | Unstable/glitchy | Risk-driven chaos |
| `Recovery` | Cooldown phase | After glitch or release |

### YAML Schema

```yaml
# Inside instrument_ir_v0 wrapper
layers:
  L2_regimes:
    regime_nodes:
      - id: "R0_IDLE"
        kind: "Idle"
        emit_template: "T_IDLE"

      - id: "R1_ATTACK"
        kind: "Transient"
        emit_template: "T_ATTACK"
        dwell_ms: 20              # Minimum time in this state

      - id: "R2_STABLE"
        kind: "StableTone"
        emit_template: "T_STABLE"

      - id: "R3_GLITCH"
        kind: "GlitchChatter"
        emit_template: "T_GLITCH"
        dwell_ms: 50

      - id: "R4_RECOVERY"
        kind: "Recovery"
        emit_template: "T_RECOVERY"
        dwell_ms: 80

    transitions:
      - from: "R0_IDLE"
        to: "R1_ATTACK"
        when: "Gate rises"

      - from: "R1_ATTACK"
        to: "R2_STABLE"
        when: "dwell_done"

      - from: "R2_STABLE"
        to: "R3_GLITCH"
        when: "risk > 0.7"
        hysteresis:
          enter: 0.7
          exit: 0.4

      - from: "R3_GLITCH"
        to: "R4_RECOVERY"
        when: "dwell_done"

      - from: "R4_RECOVERY"
        to: "R2_STABLE"
        when: "dwell_done"

      - from: "any"
        to: "R0_IDLE"
        when: "Gate falls"
```

### Transition Conditions

| Condition | Meaning |
|-----------|---------|
| `Gate rises` | Note-on event |
| `Gate falls` | Note-off event |
| `dwell_done` | Minimum dwell time elapsed |
| `risk > X` | RISK macro exceeds threshold |
| `energy < X` | Energy state below threshold |
| `age > X` | AGE integrator above threshold |

### Hysteresis

Hysteresis prevents rapid state oscillation. When a condition has hysteresis:
- **enter**: Threshold to enter the target state
- **exit**: Threshold to exit back (must be lower than enter)

```yaml
hysteresis:
  enter: 0.7    # Must exceed 0.7 to trigger
  exit: 0.4     # Must drop below 0.4 to un-trigger
```

### Emit Templates (L3)

Each regime references an emit template that defines:
- Which operator families are active
- Which specific operators to enable
- Tree blend targets
- Event lane policies

```yaml
layers:
  L3_cmmm:
    emit_templates:
      T_IDLE:
        families:
          CONSTRAINT: 1.0
          CALM: 1.0
        ops_enable: ["NormalizeScale", "FineClamp"]
        trees:
          A: "T1"
          B: "T1"
          beta: 0.0

      T_ATTACK:
        families:
          COLOR: 0.6
          EXCITE: 1.0
          CONSTRAINT: 1.0
        ops_enable: ["Tilt", "IntraMix", "NormalizeScale", "FineClamp"]
        trees:
          A: "T1"
          B: "T1"
          beta: 0.0

      T_STABLE:
        families:
          COLOR: 0.5
          COHERENCE: 0.4
          CONSTRAINT: 1.0
          CALM: 0.7
        ops_enable: ["Tilt", "LowRankMorph", "NormalizeScale", "FineClamp", "SheafProject"]

      T_GLITCH:
        families:
          TOPOLOGY: 0.8
          CATASTROPHE: 0.6
          CONSTRAINT: 0.5
        ops_enable: ["BandShear", "SwapBands", "RotateNodes", "FineClamp"]

      T_RECOVERY:
        families:
          COHERENCE: 0.3
          CONSTRAINT: 1.0
          CALM: 1.0
        ops_enable: ["GluePair", "SheafProject", "NormalizeScale", "FineClamp"]
```

### Default Regime Graph

If you don't define L2, the compiler creates a default 4-state graph:

```
R0_IDLE → R1_ATTACK → R2_STABLE → R0_IDLE
              ↓
         R6_RECOVERY
```

---

## YAML Preset Format

For complex presets, use YAML files:

```yaml
# my_preset.yaml
preset:
  preset_id: "MY_PRESET_V1"
  preset_version: "v1"
  author: "Your Name"
  description: "A rich synth pad"

compat:
  voiceframe_std: "WSE-STD-020 v1"
  vm_std: "WSE-STD-VM-HYBRID v0"
  operator_lib_id: "WSE-OPLIB-24 v0"

latch_defaults:
  pattern_id: 0
  tail_mode: "FULL"
  tree_a_id: 0
  tree_b_id: 0
  tree_blend_q15: 0
  carrier:
    model: "SAW"
    role: "AUDIBLE"
    transition: "RETRIGGER"
    detune_cents: 0
    pulse_width_q15: 16384

env:
  amp:
    attack_ms: 300.0
    decay_ms: 200.0
    sustain: 0.8
    release_ms: 800.0

material:
  beta_start_q15: 4915       # 0.15 (mostly dry at attack)
  beta_target_q15: 16384     # 0.50 (bring in texture)
  beta_ramp_blocks: 16       # Ramp time in blocks

backbone:
  - op: "TILT"
    strength_q15: 14746      # 0.45
  - op: "INTRA_MIX"
    strength_q15: 8192       # 0.25
  - op: "DIFFUSE_DOWN"
    strength_q15: 6554       # 0.20
  - op: "LOWRANK_MORPH"
    strength_q15: 4915       # 0.15

carrier:
  harmonic_amps_q15:
    - 32767    # fundamental
    - 16383    # 2nd harmonic
    - 10922    # 3rd harmonic
    - 8191     # 4th harmonic
    - 6553     # 5th harmonic
    - 5461     # 6th harmonic
    - 4681     # 7th harmonic
    - 4096     # 8th harmonic

macros:
  BRIGHT:
    target: brightness
    default_q15: 16384
  BODY:
    target: density
    default_q15: 16384
```

Load and compile:

```python
from symcrash_python_ap.instrument_ir import load_instrument

ir = load_instrument("my_preset.yaml")
capsule = ir.to_capsule()
```

---

## Unit Conversion Helpers

Convert between human-friendly and internal units:

```python
from symcrash_python_ap.instrument_ir import ms_to_blocks, pct_to_q15, q15_to_pct

# Milliseconds to blocks (64 samples @ 48kHz = ~1.33ms/block)
blocks = ms_to_blocks(100)         # 100ms → ~75 blocks

# Percentage to Q15 fixed-point (0-32767)
q15_val = pct_to_q15(0.5)          # 0.5 → 16383

# Q15 back to percentage
pct = q15_to_pct(16383)            # 16383 → ~0.5
```

---

## Compilation

### Basic Compilation

```python
ir = InstrumentIR.quick_synth("My Synth")
capsule = ir.to_capsule()
```

### With Options

```python
capsule = ir.to_capsule(
    program_id=42,              # Override program ID
    capsule_id="custom-id",     # Override capsule ID
    auto_complete=True          # Fill missing L1/L2/L3 (default)
)
```

### Full Compile Result

Get warnings and the CapsuleRecord:

```python
result = ir.to_compile_result()

if result.success:
    capsule = result.capsule
    record = result.record      # CapsuleRecord in COMPILED state

for warning in result.warnings:
    print(f"Warning: {warning}")
```

---

## Minimal IR Example

The simplest possible preset (auto-completed):

```python
ir = InstrumentIR()
ir.meta.instrument_id = "minimal"
ir.meta.name = "Minimal Synth"

# L0 auto-generated with defaults
# L1/L2/L3 auto-completed by compiler

capsule = ir.to_capsule()
```

---

## Complete Custom Example

Full control over the L0 graph:

```python
from symcrash_python_ap.instrument_ir import (
    InstrumentIR, L0Node, L0Edge, NodeParams, ADSR, Macro
)

ir = InstrumentIR()
ir.meta.instrument_id = "dual-osc-pad"
ir.meta.name = "Dual Oscillator Pad"
ir.meta.author = "symcrash"

# Two oscillators into mixer, through filter, to amp
ir.layers.L0_graph.nodes = [
    L0Node(id="osc1", type="Osc", params=NodeParams(waveform="saw")),
    L0Node(id="osc2", type="Osc", params=NodeParams(waveform="pulse", pw=0.3, detune_cents=7)),
    L0Node(id="mix", type="Mixer"),
    L0Node(id="filt", type="Filter", params=NodeParams(mode="lp")),
    L0Node(id="amp", type="Amp", params=NodeParams(
        adsr=ADSR(a_ms=400, d_ms=200, s=0.8, r_ms=1000)
    )),
]

ir.layers.L0_graph.edges = [
    L0Edge(src="osc1", dst="mix"),
    L0Edge(src="osc2", dst="mix"),
    L0Edge(src="mix", dst="filt"),
    L0Edge(src="filt", dst="amp"),
]

ir.layers.L0_graph.macros = [
    Macro(id="M_BRIGHT", name="Brightness", default=0.4),
    Macro(id="M_CUTOFF", name="Filter Cutoff", default=0.6),
]

capsule = ir.to_capsule()
```

---

## Related Documentation

- [WSE-STD-PRESET-v0.md](../control/WSE-STD-PRESET-v0.md) - Normative preset format spec + canonical examples
- [WSE-STD-IPLC-v0.md](WSE-STD-IPLC-v0.md) - Capsule lifecycle (DRAFT → COMPILED → PUBLISHED → APPLIED)
- [OPERATORS.md](OPERATORS.md) - Complete operator reference
- [MACROS-AND-CONTROLS.md](MACROS-AND-CONTROLS.md) - Control system details
- [BUDGETS-AND-DEGRADATION.md](BUDGETS-AND-DEGRADATION.md) - Performance budgets
