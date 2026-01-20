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

The oscillator supports multiple synthesis modes:

```python
L0Node(
    id="osc1",
    type="Osc",
    params=NodeParams(
        # Basic parameters
        waveform="saw",           # "sine" | "saw" | "pulse" | "tri"
        detune_cents=0.0,         # Fine tuning in cents
        pw=0.5,                   # Pulse width (legacy, use pwm_width_q15)
        role="audible",           # "off" | "audible" | "exciter_only"
        archetype="harmonic",     # "harmonic" | "bright" | "hollow" | "noisy" | "transient"

        # Oscillator mode (osc_mode)
        osc_mode=0,               # 0=A/LUT, 1=B/Rotator, 2=Sample, 3=FM6

        # Waveform type (waveform_type)
        waveform_type=0,          # 0=sine, 1=square, 2=pulse, 3=saw, 4=saw_down, 5=tri

        # PWM parameters (Q0.15)
        pwm_enable=False,
        pwm_width_q15=16384,      # Duty cycle Q0.15 (16384=50%)

        # Hard sync parameters
        sync_enable=False,
        sync_ratio_q16=65536,     # Master/slave ratio Q16.16 (65536=1.0)

        # Mode B FM parameters
        fm_enable=False,
        fm_depth_q15=0,           # FM depth Q0.15
        fm_source=0,              # 0=off, 1=feedback, 2=LFO

        # Unison parameters
        unison_voices=1,          # Voice count (1-8)
        unison_spread=10,         # Detune spread in cents

        # Mode flags
        drone_mode=False,         # Sustained coefficient mode
        legato_mode=False,        # Smooth pitch transitions
    )
)
```

##### Oscillator Modes

| Mode | Value | Description | Use Case |
|------|-------|-------------|----------|
| Mode A (LUT) | 0 | Wavetable lookup | Classic waveforms, low CPU |
| Mode B (Rotator) | 1 | Harmonic rotator synthesis | Rich timbres, harmonic profiles |
| Sample | 2 | Sample playback | Acoustic samples |
| FM6 | 3 | 6-operator FM synthesis | DX7-style FM sounds |

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

The filter supports extended parameters for subtractive synthesis:

```python
L0Node(
    id="filter1",
    type="Filter",
    params=NodeParams(
        # Legacy mode string
        mode="lp",                     # "lp" | "hp" | "bp" | "notch"

        # Extended filter parameters (Q0.15)
        filter_enable=True,
        filter_type=0,                 # 0=LP, 1=HP, 2=BP, 3=Notch
        filter_cutoff_q15=32767,       # Cutoff Q0.15 (log scale 20Hz-20kHz)
        filter_resonance_q15=0,        # Resonance Q0.15 (0=flat, 32767=self-osc)
        filter_env_depth_q15=0,        # Envelope mod Q1.15 (signed)
        filter_key_track_q15=0,        # Key tracking Q0.15 (16384=50%)
        filter_drive_q15=0,            # Pre-filter saturation Q0.15
        filter_slope=1,                # 0=12dB/oct, 1=24dB/oct, 2=6dB/oct
    )
)
```

##### Filter Types

| Type | Value | Description |
|------|-------|-------------|
| Low Pass | 0 | Removes high frequencies |
| High Pass | 1 | Removes low frequencies |
| Band Pass | 2 | Passes a frequency band |
| Notch | 3 | Removes a frequency band |

##### Filter Slopes

| Slope | Value | Character |
|-------|-------|-----------|
| 6dB/oct | 2 | Gentle, subtle |
| 12dB/oct | 0 | Standard |
| 24dB/oct | 1 | Aggressive |

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

#### FM6 Synthesis (Osc with osc_mode=3)

6-operator FM synthesis in the style of the Yamaha DX7. Uses NodeParams FM6 fields:

```python
L0Node(
    id="fm1",
    type="Osc",
    params=NodeParams(
        osc_mode=3,                        # FM6 mode

        # FM6 parameters
        algorithm=1,                       # DX7 algorithm 1-32
        feedback=3,                        # Feedback level 0-7
        op_ratios=[65536, 131072, 65536, 65536, 65536, 65536],  # Q16.16 ratios
        op_levels=[32767, 24576, 16384, 8192, 0, 0],           # Q0.15 levels

        # Topology morphing
        fm6_topo_a=0,                      # Source archetype 0-7
        fm6_topo_b=1,                      # Target archetype 0-7
        fm6_topo_blend_q15=0,              # Topology morph Q0.15
    )
)
```

##### DX7 Algorithms

The system supports all 32 original DX7 algorithms. Common starting points:

| Algorithm | Character | Description |
|-----------|-----------|-------------|
| 1 | Brass | Classic brass, stacked carriers |
| 2 | Bright brass | More harmonic content |
| 5 | Electric piano | Classic EP sound |
| 7 | Pads | Lush, evolving pads |
| 11 | Organ | Complex organ tones |
| 21 | Bass | Punchy bass sounds |
| 32 | All parallel | Each op is an independent carrier |

##### Operator Ratios (Q16.16)

Ratios are in Q16.16 fixed-point where 65536 = 1.0x:

| Ratio | Q16.16 Value | Musical Interval |
|-------|--------------|------------------|
| 0.5x | 32768 | Octave down |
| 1.0x | 65536 | Unison |
| 2.0x | 131072 | Octave up |
| 3.0x | 196608 | Octave + fifth |
| 4.0x | 262144 | Two octaves |

##### Topology Archetypes

FM6 supports smooth morphing between 8 canonical topologies:

| ID | Name | Character |
|----|------|-----------|
| 0 | Stack | Vertical carrier stack |
| 1 | Parallel | All carriers parallel |
| 2 | Feedback | Heavy feedback loop |
| 3 | Modulator | Deep modulation |
| 4 | Complex | Multi-carrier complex |
| 5 | Simple | Minimal 2-op |
| 6 | Ring | Ring modulation-like |
| 7 | Hybrid | Balanced hybrid |

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
| `M_BRIGHT` | BRIGHTNESS | Harmonic brightness |
| `M_ATTACK` | ATK_BLOCKS | Attack time |
| `M_DECAY` | DCY_BLOCKS | Decay time |
| `M_SUSTAIN` | SUS_Q15 | Sustain level |
| `M_RELEASE` | REL_BLOCKS | Release time |
| `M_GLITCH` | RISK | Glitch/chaos amount |
| `M_MORPH` | MORPH | Tree blend |
| `M_TEXTURE` | MATERIAL | Surface texture |
| `M_DETUNE` | DETUNE | Pitch detune |

### Control Config (Semantic Axes)

Presets can specify default values for CMMM semantic control axes:

```yaml
instrument_ir_v0:
  # ... meta, layers ...

  control:
    energy_intent: 0.5      # Playing intensity / attack character
    motion: 0.0             # Movement / modulation amount
    focus: 0.5              # Attention / emphasis
    brightness: 0.75        # Timbral brightness (high for brass)
    density: 0.5            # Body / thickness
    risk: 0.1               # Chaos / instability
    space: 0.25             # Spatial / reverb character
    material: 0.5           # Surface texture (wood=low, glass=high)
    unity: 0.5              # Coherence / voice blend
```

These values are pushed to MacroRuntime at preset load time. All values are normalized 0.0-1.0.

#### Semantic Control Axes

| Axis | Default | Description |
|------|---------|-------------|
| `energy_intent` | 0.5 | Playing intensity, attack character |
| `motion` | 0.0 | Movement, modulation depth |
| `focus` | 0.5 | Spectral concentration |
| `brightness` | 0.5 | Harmonic content (timbre) |
| `density` | 0.5 | Low-mid presence (body) |
| `risk` | 0.0 | Stochastic behavior (chaos) |
| `space` | 0.25 | Spatial/reverb character |
| `material` | 0.5 | Surface texture |
| `unity` | 0.5 | Voice coherence/blend |

### Matrix Envelope System

The Matrix Envelope System allows named envelopes to route to multiple targets. This is particularly useful for FM6 synthesis where multiple operators share envelope shapes.

```yaml
instrument_ir_v0:
  envelopes:
    - id: "carrier_env"
      adsr:
        a_ms: 8
        d_ms: 150
        s: 0.7
        r_ms: 100
      targets: [FM_OP1, FM_OP2, FM_OP3, FM_OP4]

    - id: "modulator_env"
      adsr:
        a_ms: 2
        d_ms: 300
        s: 0.3
        r_ms: 200
      targets: [FM_OP5, FM_OP6]

    - id: "filter_env"
      adsr:
        a_ms: 10
        d_ms: 200
        s: 0.5
        r_ms: 150
      targets: [FILTER_CUTOFF]
```

#### Envelope Targets

| Target | Description |
|--------|-------------|
| `FM_OP1` - `FM_OP6` | FM operator envelope values |
| `GAIN` | Voice gain multiplier |
| `FILTER_CUTOFF` | Filter cutoff modulation |
| `FILTER_RESO` | Filter resonance modulation |
| `PITCH` | Pitch modulation |
| `VIBRATO_DEPTH` | Vibrato depth modulation |

### Modulation Parameters

For expressive presets, NodeParams supports vibrato, body warble, and attack texture:

```python
NodeParams(
    # Vibrato/LFO modulation
    vibrato=VibratoParams(
        rate_hz=5.5,            # LFO rate in Hz
        depth_amp=0.06,         # Amplitude modulation depth (0-1)
        depth_timbre=0.08,      # Timbre modulation depth (0-1)
        delay_ms=40.0,          # Delay before vibrato onset
        ramp_ms=30.0,           # Ramp-in time
    ),

    # Body resonance warble (slow formant drift)
    body_warble=BodyWarbleParams(
        rate_hz=1.2,            # Slow LFO rate (0.5-2 Hz)
        depth=0.1,              # Warble depth (0-1)
        target_harmonic=4,      # Which harmonic to modulate
    ),

    # Attack texture (brightness settling)
    attack_texture=AttackTextureParams(
        bright_ms=60.0,         # Time for brightness to settle
        bright_amount=0.3,      # Extra brightness on attack
        harmonics=[5, 6],       # Which harmonics to boost
    ),
)
```

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

## YAML Preset Format (instrument_ir_v0)

The canonical format for instrument presets is `instrument_ir_v0`. This is a layered tower format that provides explicit control over all synthesis layers.

```yaml
# my_preset.yaml
instrument_ir_v0:
  meta:
    instrument_id: "my-preset-v1"
    name: "My Rich Synth Pad"
    version: 1
    author: "Your Name"
    seed_u32: 0x12345678      # Optional: deterministic PRNG seed

  io:
    sample_rate_hz: 48000
    block_size: 64

  layers:
    L0_graph:
      nodes:
        - id: "osc1"
          type: "Osc"
          params:
            waveform: "saw"
            detune_cents: 0

        - id: "amp1"
          type: "Amp"
          params:
            adsr:
              a_ms: 300.0       # Attack (ms)
              d_ms: 200.0       # Decay (ms)
              s: 0.8            # Sustain level (0.0-1.0)
              r_ms: 800.0       # Release (ms)

      edges:
        - src: "osc1"
          dst: "amp1"

      macros:
        - id: "M_BRIGHT"
          name: "Brightness"
          default: 0.5
        - id: "M_BODY"
          name: "Body"
          default: 0.5

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

        T_STABLE:
          families:
            COLOR: 0.5
            COHERENCE: 0.4
            CONSTRAINT: 1.0
            CALM: 0.7
          ops_enable: ["Tilt", "LowRankMorph", "NormalizeScale", "FineClamp"]
          trees:
            A: "T1"
            B: "T1"
            beta: 0.25

  compile:
    material:
      beta_start_q15: 4915       # 0.15 (mostly dry at attack)
      beta_target_q15: 16384     # 0.50 (bring in texture)
      beta_ramp_blocks: 16       # Ramp time in blocks
```

### Required vs Optional Sections

| Section | Required | Purpose |
|---------|----------|---------|
| `meta.instrument_id` | **Yes** | Unique identifier |
| `meta.name` | **Yes** | Human-readable name |
| `meta.version` | No | Version number (default: 0) |
| `meta.author` | No | Author name |
| `meta.seed_u32` | No | PRNG seed (auto-generated if omitted) |
| `io` | No | Sample rate/block size (defaults: 48kHz/64) |
| `layers.L0_graph` | No | DSP graph (auto-completed if omitted) |
| `layers.L2_regimes` | No | State machine (auto-completed if omitted) |
| `layers.L3_cmmm` | No | Emit templates (auto-completed if omitted) |
| `compile.material` | No | Dry/wet blend settings |

### Load and Compile

```python
from symcrash_python_ap.instrument_ir import load_instrument

ir = load_instrument("my_preset.yaml")
capsule = ir.to_capsule()
```

---

## Macro Preset Format (macro_ir_v0)

Performance macros (vibrato, legato, tremolo, portamento) use a separate `macro_ir_v0` format. These are NOT instruments - they are modifiers that can be applied to any instrument.

```yaml
# vibrato_classic.yaml
macro_ir_v0:
  meta:
    macro_id: "vibrato-classic-v0"
    name: "Classic Vibrato"
    version: 0
    author: "symcrash"
    description: "Classic vibrato with delayed onset."

  macro:
    macro_type: VIBRATO       # VIBRATO | TREMOLO | LEGATO | PORTAMENTO

    # LFO parameters (VIBRATO, TREMOLO)
    rate_hz: 5.5
    depth_pitch_cents: 25
    depth_amp_q15: 1966       # 6%
    depth_timbre_q15: 2621    # 8%

    # Envelope parameters
    delay_ms: 40.0
    ramp_ms: 30.0

  tags:
    - vibrato
    - expressive
```

### Macro Types

| Type | Purpose | Key Parameters |
|------|---------|----------------|
| `VIBRATO` | Pitch/amp/timbre LFO | `rate_hz`, `depth_*`, `delay_ms`, `ramp_ms` |
| `TREMOLO` | Amplitude-only LFO | `rate_hz`, `depth_amp_q15`, `delay_ms`, `ramp_ms` |
| `LEGATO` | Smooth pitch glide | `glide_time_ms`, `glide_curve`, `preserve_phase` |
| `PORTAMENTO` | Always-on pitch glide | `glide_time_ms`, `glide_curve` |

### Load Macro Preset

```python
from symcrash_python_ap.macro_ir import load_macro

macro = load_macro("vibrato_classic.yaml")
capsule = macro.to_capsule()
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

## Runtime Control (WSE-STD-070)

Presets can specify runtime behavior via CMMM event payloads. These control how the executor processes each voice.

### Node Mask Control

Control the active node count (K) at runtime for dynamic degradation:

```yaml
layers:
  L3_cmmm:
    emit_templates:
      T_ATTACK:
        # Start with fewer nodes, ramp up
        cmmm_node_mask: 0x000000FF   # K=8 (bands 0-1 only)
        cmmm_mask_ramp_blocks: 10    # Ramp over 10 blocks

      T_STABLE:
        # Full resolution
        cmmm_node_mask: 0xFFFFFFFF   # K=32 (all nodes)
        cmmm_mask_ramp_blocks: 5
```

### Tail Mode Control

Control safety tail operators per regime:

```yaml
emit_templates:
  T_STABLE:
    cmmm_tail_mode: 0       # FULL: FineClamp + NormalizeScale + SheafProject
    cmmm_traction_q15: 32767  # Full projection strength

  T_GLITCH:
    cmmm_tail_mode: 2       # MIN: FineClamp only (cheaper)
    cmmm_traction_q15: 16384  # Softer projection
```

### Tail Mode Reference

| Mode | Value | Operators | Use Case |
|------|-------|-----------|----------|
| FULL | 0 | FineClamp, NormalizeScale, SheafProject | High quality |
| MID | 1 | FineClamp, NormalizeScale | Moderate load |
| MIN | 2 | FineClamp | Low budget |
| EMERGENCY | 3 | FineClamp, EmergencySimplify | Overload |

---

## Related Documentation

- [WSE-STD-PRESET-v0.md](../control/WSE-STD-PRESET-v0.md) - Normative preset format spec + canonical examples
- [WSE-STD-IPLC-v0.md](WSE-STD-IPLC-v0.md) - Capsule lifecycle (DRAFT → COMPILED → PUBLISHED → APPLIED)
- [OPERATORS.md](OPERATORS.md) - Complete operator reference
- [MACROS-AND-CONTROLS.md](MACROS-AND-CONTROLS.md) - Control system details
- [BUDGETS-AND-DEGRADATION.md](BUDGETS-AND-DEGRADATION.md) - Performance budgets
- [../architecture/WSE-STD-070.md](../architecture/WSE-STD-070.md) - Tower completion and CMMM events
