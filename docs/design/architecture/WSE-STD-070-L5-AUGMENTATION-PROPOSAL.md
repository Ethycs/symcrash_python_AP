# WSE-STD-070 L5 Augmentation Proposal

**Status:** Proposal (v2)
**Date:** 2026-01-19
**Augments:** WSE-STD-070 v0

---

## 1. Problem Statement

The current WSE-STD-070 defines L5 (Lab) as a single layer for "new ops/trees" with "offline tools". This is too coarse. Codebase analysis reveals **19 distinct subsystems**, each with its own complexity dimension that could benefit from layered exposure.

### 1.1 Originally Identified Axes (7)

| Subsystem | Canonical Set | What L5 Would Add |
|-----------|---------------|-------------------|
| Operators | 24 ops | Custom operator kernels |
| Trees | 5 families | Custom wavelet topologies |
| Regimes | 7 states | Custom state machines |
| Materials | 8 programs | Custom effect chains |
| Macros | 11 system | Custom choreography |
| Budgets | 7 steps | Custom degradation |
| Safety | 16 mechanisms | Custom invariants |

### 1.2 Additional Axes Discovered in Codebase (12)

| Subsystem | Location | Canonical Set | What L5 Would Add |
|-----------|----------|---------------|-------------------|
| **Envelopes** | `envelope_scheduler.py` | ADSR 4-phase | Custom curves, per-voice templates |
| **Modulation** | `harmonic_bank.py` | 4 LFO waveforms, 3 sync modes | Custom LFO shapes, routing matrices |
| **Harmonics** | `harmonic_bank.py` | 32 harmonics, OEIS sequences | Custom harmonic profiles, morphing |
| **FM Synthesis** | `vm/fm6_*.py` | 6 ops, 32 algorithms | Custom algorithms, operator morphing |
| **Samples** | `vm/sample_source.py` | Loop params, root note | Sample libraries, multi-sample maps |
| **Granular** | `vm/granular_engine.py` | 8 grains, density/spray | Grain patterns, custom envelopes |
| **Tunings** | `tuning.py` | 11 presets (FLAT→VOWEL_U) | Custom formants, spectral morphing |
| **Carrier** | `wavelet/carrier_lift.py` | Beta blend, 5-block latency | Blend curves, carrier waveforms |
| **Voice Alloc** | `voice/*.py` | 128 voices, LRU stealing | Polyphony modes, priority schemes |
| **Pitch** | `tuning.py`, `harmonic_bank.py` | MIDI + cents | Microtonality, glide curves |
| **Events** | `vm/lanes.py` | 8 lanes, 0x30/0x31 payloads | Event patterns, lane routing |
| **Arcs** | `vm/state.py` | 8 arc slots, phase storage | Arc patterns, sync modes |

A single "L5 = Lab" layer conflates these **19 independent dimensions**.

---

## 2. Proposed Concept: Subsystem Complexity Axes

Instead of a single L5, introduce **Complexity Axes** that apply the L0-L5 tower concept to each subsystem independently.

### 2.1 Complexity Axis Definition

Each subsystem has its own complexity axis:

```
Axis ::= (Subsystem, Level)
Level ::= L0 | L1 | L2 | L3 | L4 | L5
```

### 2.2 Subsystem Axes (Complete)

#### Tier 1: Core Synthesis Axes (Originally Identified)

| Axis | L0 (Fixed) | L2 (Tunable) | L4 (Editable) | L5 (Custom) |
|------|------------|--------------|---------------|-------------|
| **Op** | 24 canonical | Select subset | Reorder/weight | Define new |
| **Tree** | 5 families | Select pair | Define splits | New wavelets |
| **Regime** | 7 states | Tune edges | Add/remove | Custom guards |
| **Material** | 8 programs | Select one | Chain params | Custom DSP |
| **Macro** | 11 system | Bind presets | Slew policies | Choreography |
| **Budget** | Fixed costs | Tune limits | Custom ladder | Predictive |
| **Safety** | 16 mechanisms | Enable/disable | Tune thresholds | Custom proofs |

#### Tier 2: Modulation & Envelope Axes (Newly Discovered)

| Axis | L0 (Fixed) | L2 (Tunable) | L4 (Editable) | L5 (Custom) |
|------|------------|--------------|---------------|-------------|
| **Envelope** | Linear ADSR | Time params | Per-voice sets | Custom curves |
| **Modulation** | 4 LFO shapes | Rate/depth | Routing matrix | Custom waveforms |
| **Arcs** | 8 slots | Phase rates | Sync modes | Pattern library |

#### Tier 3: Oscillator & Source Axes (Newly Discovered)

| Axis | L0 (Fixed) | L2 (Tunable) | L4 (Editable) | L5 (Custom) |
|------|------------|--------------|---------------|-------------|
| **Harmonics** | 32 partials | Rolloff | OEIS select | Custom profiles |
| **FM** | 32 algorithms | Op levels | Ratios/feedback | Custom algorithms |
| **Samples** | Single sample | Loop points | Root note | Multi-sample maps |
| **Granular** | 8 grains | Density/spray | Grain size | Pattern engines |
| **Carrier** | Sine + wet | Beta blend | Ramp time | Carrier waveforms |

#### Tier 4: Control & Allocation Axes (Newly Discovered)

| Axis | L0 (Fixed) | L2 (Tunable) | L4 (Editable) | L5 (Custom) |
|------|------------|--------------|---------------|-------------|
| **Voice** | 128 max | Polyphony cap | Steal mode | Priority schemes |
| **Pitch** | 12-TET | Bend range | Glide time | Microtonality |
| **Tuning** | FLAT preset | Select preset | Formant freq | Custom spectral |
| **Events** | 8 lanes | Lane enable | Payload types | Pattern library |

### 2.3 Axis Grouping (Tier System)

To manage complexity, axes are grouped into **4 tiers**:

| Tier | Axes | Focus |
|------|------|-------|
| **T1** | Op, Tree, Regime, Material, Macro, Budget, Safety | Core synthesis identity |
| **T2** | Envelope, Modulation, Arcs | Time-varying modulation |
| **T3** | Harmonics, FM, Samples, Granular, Carrier | Sound source generation |
| **T4** | Voice, Pitch, Tuning, Events | Control & allocation |

### 2.4 Composite Tower Level

The preset's `tower_level` becomes the **maximum** across all axes:

```python
tower_level = max(
    # Tier 1
    op_level, tree_level, regime_level, material_level,
    macro_level, budget_level, safety_level,
    # Tier 2
    envelope_level, modulation_level, arcs_level,
    # Tier 3
    harmonics_level, fm_level, samples_level, granular_level, carrier_level,
    # Tier 4
    voice_level, pitch_level, tuning_level, events_level
)
```

Each axis is tracked independently in the preset body.

---

## 3. Proposed Augmentations to WSE-STD-070

### 3.1 Augment §1.2 Preset Body

Add complexity axis tracking:

```text
L5_lab_layer:
  # Tier 1: Core Synthesis
  op_axis:       AxisLevel   # 0-5
  tree_axis:     AxisLevel   # 0-5
  regime_axis:   AxisLevel   # 0-5
  material_axis: AxisLevel   # 0-5
  macro_axis:    AxisLevel   # 0-5
  budget_axis:   AxisLevel   # 0-5
  safety_axis:   AxisLevel   # 0-5

  # Tier 2: Modulation
  envelope_axis:   AxisLevel   # 0-5
  modulation_axis: AxisLevel   # 0-5
  arcs_axis:       AxisLevel   # 0-5

  # Tier 3: Sources
  harmonics_axis: AxisLevel   # 0-5
  fm_axis:        AxisLevel   # 0-5
  samples_axis:   AxisLevel   # 0-5
  granular_axis:  AxisLevel   # 0-5
  carrier_axis:   AxisLevel   # 0-5

  # Tier 4: Control
  voice_axis:   AxisLevel   # 0-5
  pitch_axis:   AxisLevel   # 0-5
  tuning_axis:  AxisLevel   # 0-5
  events_axis:  AxisLevel   # 0-5

  # Custom definitions (populated when axis >= L5)
  custom_ops:        List[CustomOpDef]
  custom_trees:      List[CustomTreeDef]
  custom_regimes:    List[CustomRegimeDef]
  custom_materials:  List[CustomMaterialDef]
  custom_macros:     List[CustomMacroDef]
  custom_budgets:    CustomBudgetDef
  custom_safety:     List[CustomSafetyDef]
  custom_envelopes:  List[CustomEnvelopeDef]
  custom_lfos:       List[CustomLFODef]
  custom_arcs:       List[CustomArcDef]
  custom_harmonics:  List[CustomHarmonicDef]
  custom_fm:         List[CustomFMAlgorithmDef]
  custom_samples:    List[CustomSampleMapDef]
  custom_granular:   CustomGranularDef
  custom_carrier:    CustomCarrierDef
  custom_voice:      CustomVoiceAllocDef
  custom_pitch:      CustomPitchDef
  custom_tuning:     List[CustomTuningDef]
  custom_events:     List[CustomEventPatternDef]
```

### 3.2 Augment §2.1 Editor Primitives

Add axis-aware primitives:

```text
#### Axis Control (L5)

* `SET_AXIS_LEVEL(axis, level)`
* `REGISTER_CUSTOM_OP(op_def)`
* `REGISTER_CUSTOM_TREE(tree_def)`
* `REGISTER_CUSTOM_REGIME(regime_def)`
* `REGISTER_CUSTOM_MATERIAL(material_def)`
* `REGISTER_CUSTOM_MACRO(macro_def)`
* `SET_CUSTOM_BUDGET(budget_def)`
* `REGISTER_CUSTOM_SAFETY(safety_def)`
```

### 3.3 Augment §3 Tower Levels

Replace monolithic L5 definition:

| Level | Name | Description |
|-------|------|-------------|
| **L5** | Lab | Per-axis custom definitions. Each axis MAY independently reach L5. |

Add subsection:

#### §3.1 Complexity Axes (Normative)

| Axis | ID | Canonical Set | L5 Custom Type |
|------|----|---------------|----------------|
| Operators | `op` | 24 ops (Op 0-23) | `CustomOpDef` |
| Trees | `tree` | 5 families (T0-T4) | `CustomTreeDef` |
| Regimes | `regime` | 7 states (R0-R6) | `CustomRegimeDef` |
| Materials | `material` | 8 programs (M0-M7) | `CustomMaterialDef` |
| Macros | `macro` | 11 system (SM0-SM10) | `CustomMacroDef` |
| Budgets | `budget` | 7-step ladder | `CustomBudgetDef` |
| Safety | `safety` | 16 mechanisms | `CustomSafetyDef` |

Each axis follows the same completion/projection rules as the main tower.

### 3.4 Augment §4.2 Default Completion

Add axis completion rules:

#### Missing L5 (Lab) — Per-Axis

For each axis at level < L5:
* Use canonical set only
* Custom definitions forbidden
* No proof obligations required

For each axis at level = L5:
* Custom definitions allowed
* Compilation proof obligation MUST be satisfied
* Projection to L4 freezes custom → canonical mapping

### 3.5 Augment §5.1 Projection

Add axis projection semantics:

#### Axis Projection

`PROJECT_AXIS(axis, target_level)`:
* If target_level < current level, freeze custom to closest canonical
* Example: Custom tree → project to closest canonical family
* Example: Custom operator → project to closest canonical op + weight adjustment

---

## 4. Custom Definition Schemas

### 4.1 CustomOpDef (Operators)

```yaml
custom_op:
  op_id: 100                    # Custom IDs start at 100
  name: "MyTilt"
  op_type: DIAG                 # Must be canonical type
  kernel:
    type: matrix_q15            # Or: function_ref, lut_q15
    data: [...]                 # 32x32 Q15 matrix
  cost:
    ticks_base: 50
    ticks_per_K: 2
  lane: 3                       # 0-7
  emission_group: MID           # TAIL/BASE/MID/TOP
  is_tail: false
  drop_priority: 128

  # Proof obligation
  proof:
    bounded: true               # Output ≤ input magnitude
    monotonic: false
    preserves_dc: true
```

### 4.2 CustomTreeDef (Wavelets)

```yaml
custom_tree:
  tree_id: 100                  # Custom IDs start at 100
  name: "HarmonicExtended"
  base_family: 1                # Inherit from T1 (Harmonic)

  split_mask: 0xFFFF00FF        # Which nodes split further
  leaf_depths: [6,6,5,5,...]    # Depth per leaf position
  complex_leaf_mask: 0x0F       # Which leaves use complex coeffs

  # Proof obligation
  proof:
    total_leaves: 32            # Must equal canonical
    covers_bandwidth: true      # No frequency gaps
```

### 4.3 CustomRegimeDef (State Machines)

```yaml
custom_regime:
  regime_id: 100
  name: "MyGlitch"
  kind: GlitchChatter           # Must map to canonical kind

  emit_template:
    families: {GLITCH: 0.9, CONSTRAINT: 0.5}
    ops_enable: [Tilt, BandShear, FineClamp]
    trees: {A: T1, B: 100, beta: "chaos"}  # Can reference custom tree

  # Transition guards
  entry_guard: "chaos > 0.8 && pressure > 0.5"
  exit_guard: "calm > 0.6"
  dwell_min_ms: 50

  # Proof obligation
  proof:
    has_exit: true              # Must be able to leave
    no_deadlock: true           # Reachable from R0
```

### 4.4 CustomMaterialDef (Effects)

```yaml
custom_material:
  material_id: 100
  name: "VinylWarmth"

  chain:
    - {type: P1_TILT, freq_hz: 80, gain_db: 2}
    - {type: P2_SATURATOR, curve: tanh, drive: 0.3}
    - {type: P3_TAPE, bias: 0.1, coercivity: 0.05}
    - {type: P4_ONEPOLE, mode: LP, cutoff_hz: 12000}

  state_size: 4                 # Floats per channel

  # Proof obligation
  proof:
    bounded_gain: 2.0           # Max output/input ratio
    stable: true                # No runaway feedback
```

### 4.5 CustomMacroDef (Choreography)

```yaml
custom_macro:
  macro_id: 100
  name: "AttackSwell"

  # Timeline of macro movements
  sequence:
    - {time_ms: 0, targets: {BRIGHTNESS: 0.2, RISK: 0.1}}
    - {time_ms: 100, targets: {BRIGHTNESS: 0.8, RISK: 0.3}}
    - {time_ms: 500, targets: {BRIGHTNESS: 0.5, RISK: 0.2}}

  trigger: gate_rise            # When to start
  loop: false

  # Proof obligation
  proof:
    duration_ms: 500            # Bounded duration
    all_targets_valid: true     # Only known macros
```

### 4.6 CustomBudgetDef (Degradation)

```yaml
custom_budget:
  name: "AggressiveDegradation"

  # Custom K ladder
  k_ladder: [32, 16, 8, 4, 1]   # Skip intermediate steps

  # Custom operator costs
  op_costs:
    100: {base: 100, per_K: 5}  # Custom op cost

  # Custom degradation order
  degradation_order:
    - {action: REDUCE_K, threshold: 0.9}
    - {action: DROP_MID, threshold: 0.8}
    - {action: DROP_TOP, threshold: 0.7}
    - {action: TAIL_ONLY, threshold: 0.5}

  # Proof obligation
  proof:
    always_playable: true       # Can always reach tail-only
    monotonic: true             # Each step reduces cost
```

### 4.7 CustomSafetyDef (Invariants)

```yaml
custom_safety:
  safety_id: 100
  name: "StrictDC"

  invariant:
    type: coefficient_bound
    expression: "abs(c[0]) < 0.01"  # DC coefficient near zero

  enforcement:
    check_frequency: every_block
    action: project             # Or: warn, degrade, panic

  # Proof obligation
  proof:
    decidable: true             # Can always evaluate
    preserves_audio: true       # Doesn't destroy signal
```

---

## 5. Additional Custom Definition Schemas (Newly Discovered Axes)

### 5.1 CustomEnvelopeDef (Tier 2: Modulation)

```yaml
custom_envelope:
  envelope_id: 100
  name: "SlowAttackPercussive"

  # Custom curve definition
  phases:
    - {phase: ATTACK, duration_ms: 500, curve: exponential, target: 1.0}
    - {phase: DECAY, duration_ms: 50, curve: linear, target: 0.7}
    - {phase: SUSTAIN, level: 0.7}
    - {phase: RELEASE, duration_ms: 200, curve: exponential, target: 0.0}

  # Optional velocity scaling
  velocity_scaling:
    attack_time: -0.3    # Faster attack at high velocity
    decay_time: 0.0
    sustain_level: 0.2   # Higher sustain at high velocity

  proof:
    bounded: true        # Always 0-1 range
    monotonic_attack: true
```

### 5.2 CustomLFODef (Tier 2: Modulation)

```yaml
custom_lfo:
  lfo_id: 100
  name: "SmoothRandom"

  waveform:
    type: custom_table    # Or: sine, triangle, saw_up, saw_down, square, sample_hold
    table_q15: [0, 8192, 16384, ...]  # 256 samples

  sync_mode: GATE_RESET   # FREE, GATE_RESET, LEGATO_RESET, TEMPO_SYNC
  tempo_division: "1/4"   # If TEMPO_SYNC

  # Routing targets
  default_routing:
    - {target: PITCH, depth_cents: 30}
    - {target: GAIN, depth_db: 3}

  proof:
    bounded: true         # Output in Q15 range
    continuous: true      # No discontinuities (for smooth waveforms)
```

### 5.3 CustomArcDef (Tier 2: Modulation)

```yaml
custom_arc:
  arc_id: 100
  name: "AttackDecayPattern"

  # Arc is a time-indexed modulation pattern
  type: spline            # Or: lfo, envelope, pattern, script

  keyframes:
    - {time_ms: 0, value: 0.0, curve: linear}
    - {time_ms: 100, value: 1.0, curve: exponential}
    - {time_ms: 500, value: 0.3, curve: linear}
    - {time_ms: 1000, value: 0.0, curve: hold}

  loop: false
  trigger: gate_rise

  proof:
    bounded: true
    finite_duration: true
```

### 5.4 CustomHarmonicDef (Tier 3: Sources)

```yaml
custom_harmonic:
  harmonic_id: 100
  name: "BellPartials"

  # Harmonic amplitude profile
  partials:
    - {harmonic: 1, amplitude: 1.0, phase: 0}
    - {harmonic: 2.4, amplitude: 0.7, phase: 0}   # Inharmonic!
    - {harmonic: 5.3, amplitude: 0.5, phase: 0}
    - {harmonic: 8.2, amplitude: 0.3, phase: 0}

  # OEIS-based generation (alternative)
  oeis_sequence: null     # Or: "A000040" for primes

  rolloff_db_per_octave: 6.0

  proof:
    finite_partials: true
    bounded_amplitude: true
```

### 5.5 CustomFMAlgorithmDef (Tier 3: Sources)

```yaml
custom_fm_algorithm:
  algorithm_id: 100
  name: "CustomBrass"

  # 6x6 modulation matrix
  # mod_matrix[i][j] = how much op j modulates op i
  mod_matrix:
    - [0, 0, 0, 0, 0, 0]      # Op 1 (carrier)
    - [1, 0, 0, 0, 0, 0]      # Op 2 → Op 1
    - [0, 1, 0, 0, 0, 0]      # Op 3 → Op 2
    - [0, 0, 0, 0, 0, 0]      # Op 4 (carrier)
    - [0, 0, 0, 1, 0, 0]      # Op 5 → Op 4
    - [0, 0, 0, 0, 1, 1]      # Op 6 → Op 5 (with feedback)

  carrier_mask: 0b001001    # Ops 1 and 4 are carriers
  feedback_op: 6            # Op 6 has feedback

  # Morphing support
  morph_targets: [1, 5]     # Can morph to algorithms 1 and 5

  proof:
    acyclic: true           # No infinite modulation loops (except feedback)
    has_carrier: true       # At least one carrier
```

### 5.6 CustomSampleMapDef (Tier 3: Sources)

```yaml
custom_sample_map:
  map_id: 100
  name: "PianoMultisample"

  # Key zones
  zones:
    - {sample_id: "piano_A0", root_note: 21, low_note: 21, high_note: 35}
    - {sample_id: "piano_C2", root_note: 36, low_note: 36, high_note: 47}
    - {sample_id: "piano_C3", root_note: 48, low_note: 48, high_note: 59}
    - {sample_id: "piano_C4", root_note: 60, low_note: 60, high_note: 71}

  # Velocity layers (optional)
  velocity_layers:
    - {threshold: 0, suffix: "_p"}
    - {threshold: 64, suffix: "_mf"}
    - {threshold: 100, suffix: "_f"}

  loop_mode: sustain        # none, sustain, release

  proof:
    covers_range: true      # No gaps in note range
    samples_exist: true     # All referenced samples loaded
```

### 5.7 CustomGranularDef (Tier 3: Sources)

```yaml
custom_granular:
  name: "TextureCloud"

  max_grains: 16            # Default is 8

  grain_params:
    size_range_ms: [20, 200]
    density: 0.7            # 0-1, spawn probability
    spray: 0.3              # Position randomization

  envelope:
    type: hanning           # Or: triangular, trapezoidal, custom
    attack_ratio: 0.15
    release_ratio: 0.15

  # Pitch variation
  pitch_spread_cents: 50
  reverse_probability: 0.1

  proof:
    bounded_grains: true    # Never exceeds max_grains
    stable: true            # No amplitude explosion
```

### 5.8 CustomCarrierDef (Tier 3: Sources)

```yaml
custom_carrier:
  name: "WarmCarrier"

  waveform: triangle        # Or: sine, triangle, saw, square, custom_table

  # Carrier/wet blending
  beta_curve:
    type: exponential       # Or: linear, s_curve
    attack_ms: 50           # Time to reach target beta

  default_beta_target: 0.75  # 75% wet (IDWPT), 25% carrier

  # Click prevention
  ola_window: hanning       # Or: triangular, blackman
  crossfade_blocks: 5

  proof:
    click_free: true        # Guaranteed smooth transitions
```

### 5.9 CustomVoiceAllocDef (Tier 4: Control)

```yaml
custom_voice_alloc:
  name: "MonoLegatoPriority"

  polyphony_mode: mono      # mono, poly, unison
  max_voices: 1

  # Voice stealing
  steal_mode: lowest_priority  # Or: oldest, quietest, none
  priority_source: velocity    # Or: note, fixed

  # Legato behavior
  legato:
    enabled: true
    portamento_time_ms: 50
    retrigger_envelope: false

  # Unison (if polyphony_mode == unison)
  unison:
    voices: 4
    detune_cents: 15
    stereo_spread: 0.7

  proof:
    bounded_voices: true    # Never exceeds max
    no_stuck_notes: true    # All notes eventually release
```

### 5.10 CustomPitchDef (Tier 4: Control)

```yaml
custom_pitch:
  name: "MicrotonalArabic"

  # Tuning system
  tuning:
    type: custom_scale      # Or: 12tet, just, pythagorean
    cents_per_note: [0, 100, 200, 300, 350, 500, 600, 700, 800, 900, 1000, 1100]

  # Pitch bend
  bend_range_semitones: 2
  bend_curve: linear        # Or: exponential

  # Glide/portamento
  glide:
    mode: legato_only       # Or: always, off
    time_ms: 100
    curve: exponential

  proof:
    covers_octave: true     # 12 notes defined
    monotonic: true         # Notes ascend in pitch
```

### 5.11 CustomTuningDef (Tier 4: Control)

```yaml
custom_tuning:
  tuning_id: 100
  name: "NasalFormant"

  # 64 leaf gains (global spectral shaping)
  leaf_gains_q15: [32767, 30000, 28000, ...]  # 64 values

  # Or define via formants
  formants:
    - {freq_hz: 400, bandwidth_hz: 100, gain_db: 0}
    - {freq_hz: 2000, bandwidth_hz: 200, gain_db: -3}
    - {freq_hz: 3500, bandwidth_hz: 300, gain_db: -6}

  # Morphing support
  morph_compatible: [VOWEL_A, VOWEL_E]

  proof:
    bounded_gain: true      # No amplification > 1.0
```

### 5.12 CustomEventPatternDef (Tier 4: Control)

```yaml
custom_event_pattern:
  pattern_id: 100
  name: "GateArpeggio"

  # Event sequence
  events:
    - {time_ms: 0, lane: EVT_OP_PULSE, strength: 1.0}
    - {time_ms: 125, lane: EVT_OP_PULSE, strength: 0.8}
    - {time_ms: 250, lane: EVT_OP_PULSE, strength: 0.6}
    - {time_ms: 375, lane: EVT_OP_PULSE, strength: 0.8}

  loop: true
  loop_length_ms: 500

  # Sync
  sync_mode: tempo          # Or: free, gate
  tempo_division: "1/16"

  proof:
    finite_events: true     # Bounded event count per loop
    valid_lanes: true       # All lanes exist
```

---

## 6. Compilation Requirements ($C_{5\to3}$)

### 6.1 Per-Axis Compilation

Each custom definition MUST compile to L3-compatible representation:

#### Tier 1: Core Synthesis

| Axis | Compiles To |
|------|-------------|
| CustomOpDef | Canonical op + weight matrix |
| CustomTreeDef | Canonical tree + split adjustments |
| CustomRegimeDef | Canonical regime + emit template |
| CustomMaterialDef | Canonical material + param overrides |
| CustomMacroDef | Macro sequence events |
| CustomBudgetDef | Budget table |
| CustomSafetyDef | Safety check function |

#### Tier 2: Modulation

| Axis | Compiles To |
|------|-------------|
| CustomEnvelopeDef | ADSRSpec with interpolated curve points |
| CustomLFODef | LFOState + waveform LUT |
| CustomArcDef | Arc keyframe sequence |

#### Tier 3: Sources

| Axis | Compiles To |
|------|-------------|
| CustomHarmonicDef | HarmonicWeights table |
| CustomFMAlgorithmDef | FM6State + mod_matrix |
| CustomSampleMapDef | Sample zone lookup table |
| CustomGranularDef | GranularState configuration |
| CustomCarrierDef | CarrierLift params |

#### Tier 4: Control

| Axis | Compiles To |
|------|-------------|
| CustomVoiceAllocDef | VoiceManager policy |
| CustomPitchDef | Pitch table + glide params |
| CustomTuningDef | InstrumentTuning (64 leaf gains) |
| CustomEventPatternDef | Event lane schedule |

### 5.2 Proof Obligations

Each custom definition MUST satisfy proof obligations:

| Proof | Meaning |
|-------|---------|
| `bounded` | Output magnitude ≤ input magnitude × constant |
| `stable` | No unbounded state growth |
| `has_exit` | State can transition out |
| `no_deadlock` | All states reachable |
| `covers_bandwidth` | No frequency gaps |
| `monotonic` | Degradation always reduces cost |
| `always_playable` | Can always reach safe state |

### 5.3 Validation Pipeline

```
CustomDef → Parse → Validate Schema → Check Proofs → Compile → Register
                           ↓                ↓
                     Schema Error      Proof Failed
                           ↓                ↓
                       Reject           Reject with diagnostic
```

---

## 7. Current Integration Status (Codebase Analysis)

### 7.1 Integration Summary

Analysis of the codebase reveals **uneven integration** across the 19 axes:

| Status | Count | Axes |
|--------|-------|------|
| **Fully Wired (Lanes)** | 8 | Op, Tree, Regime, Budget, Harmonics, FM, Carrier, Events |
| **Partially Wired** | 4 | Material, Macro, Samples, Tuning |
| **Off-Path (No Lanes)** | 5 | Envelope, Modulation, Arcs, Voice, Safety (partial) |
| **Not Exposed** | 2 | Granular, Pitch (glide) |

### 7.2 Lane System Coverage

The lanes system (`vm/lanes.py`) is the central dispatch mechanism:

| Lane | Name | Function | Status |
|------|------|----------|--------|
| 1 | CARRIER_RENDER | Oscillator synthesis | ✓ Fully wired |
| 2 | INJECT | Event-based coefficient injection | ✓ Fully wired |
| 3-6 | DIAG/BANDED/PERMUTE/LOWRANK | Operator dispatch | ✓ Fully wired |
| 7 | TAIL | Safety constraints | ✓ Fully wired |
| 8 | IDWPT_RENDER | Wavelet synthesis | ✓ Fully wired |
| 9 | MIX_GAIN | Carrier + texture blend | ✓ Fully wired |

**Missing Lanes (Gaps):**

| Gap | Needed Lane | Purpose |
|-----|-------------|---------|
| Tier 2 | ENVELOPE_APPLY | Route envelope targets to operators |
| Tier 2 | LFO_APPLY | Route LFO modulation through pipeline |
| Tier 1 | MATERIAL_DSP | Integrate material processing into lanes |

### 7.3 Per-Axis Integration Detail

#### Tier 1: Core Synthesis

| Axis | Lanes | VoiceFrame | Preset | Custom Compile | L5 Ready |
|------|-------|------------|--------|----------------|----------|
| **Op** | ✓ Lane 3-6 | `ops_enable` | ✓ | ✗ 24 fixed | ✗ |
| **Tree** | ✓ Lane 9 | `tree_a/b/blend` | ✓ | ✗ 5 fixed | ✗ |
| **Regime** | ✓ Via regime_id | `regime_id` | ✓ | ✗ 7 fixed | ✗ |
| **Material** | ~ Post-render | ParamFrame | ✓ | ✗ 8 fixed | ✗ |
| **Macro** | ~ ControlFrame | `w_family_q15[8]` | ✓ | ✗ | ✗ |
| **Budget** | ✓ Gating | `degradation_level` | ✓ | ✗ | ✗ |
| **Safety** | ✓ Lane 7 | `cmmm_tail_mode` | ✓ | ✗ | ✗ |

#### Tier 2: Modulation (**CRITICAL GAP**)

| Axis | Lanes | VoiceFrame | Preset | Custom Compile | L5 Ready |
|------|-------|------------|--------|----------------|----------|
| **Envelope** | ✗ OFF-PATH | `env_targets_q15` | ✓ | ✗ Linear only | ✗ |
| **Modulation** | ✗ OFF-PATH | `fm_mod_buffer` | ~ | ✗ 4 waveforms | ✗ |
| **Arcs** | ✗ OFF-PATH | `arc_phase_q16[8]` | ~ | ✗ | ✗ |

**Problem:** Envelopes, LFOs, and arcs compute in VoiceEngine but bypass lanes entirely. Values written directly to VoiceFrame, not dispatched through operator pipeline.

#### Tier 3: Sources

| Axis | Lanes | VoiceFrame | Preset | Custom Compile | L5 Ready |
|------|-------|------------|--------|----------------|----------|
| **Harmonics** | ✓ Mode B | `partial_amps_q15` | ✓ | ✓ Partial | ~ |
| **FM** | ✓ Executor | `fm_*` fields | ✓ | ✓ Full | ~ |
| **Samples** | ✓ Executor | `sample_rate_q16` | ~ | ✓ Full | ~ |
| **Granular** | ✗ NOT EXPOSED | (none) | ✗ | ✗ | ✗ |
| **Carrier** | ✓ Lane 1+9 | `f0_hz`, `beta_target` | ✓ | ✓ Full | ✓ |

**Problem:** Granular engine exists but has no `osc_mode` selector; cannot be activated from presets.

#### Tier 4: Control

| Axis | Lanes | VoiceFrame | Preset | Custom Compile | L5 Ready |
|------|-------|------------|--------|----------------|----------|
| **Voice** | ✗ Policy layer | `voice_id` | ✗ | ✗ | ✗ |
| **Pitch** | ✓ Phase inc | `note`, `cents` | ✓ | ✓ Full | ✓ |
| **Tuning** | ~ Post-render | (none) | ✓ | ✓ Partial | ~ |
| **Events** | ✓ Lane 2 | `cmmm_node_mask` | ✓ | ✓ Full | ✓ |

**Problem:** Voice polyphony modes exist but not preset-configurable. Tuning applied post-IDWPT, outside lane pipeline.

### 7.4 Critical Architecture Gaps

| Priority | Gap | Impact | Fix Required |
|----------|-----|--------|--------------|
| **HIGH** | Tier 2 off-path | Envelopes/LFOs can't participate in operator gating | Add Lane 2b/2c for modulation |
| **HIGH** | Granular not exposed | Cannot select granular synthesis | Add Mode G to Executor |
| **MEDIUM** | Material post-render | Can't participate in degradation | Integrate into lane 8 or 9 |
| **MEDIUM** | Voice not preset-able | No per-preset polyphony control | Add voice mode fields |
| **LOW** | Custom Op/Tree compile | L5 can't define new primitives | Implement proof validators |

---

## 8. Implementation Roadmap

### Phase 0: Architecture Fixes (Prerequisites)

- [ ] **Add Lane 2b (ENVELOPE_APPLY)** - Route envelope targets through pipeline
- [ ] **Add Lane 2c (LFO_APPLY)** - Route LFO modulation through pipeline
- [ ] **Add Mode G (GRANULAR)** to Executor - Enable granular osc_mode
- [ ] **Add voice mode fields** to preset format

### Phase 1: Schema Definition (Spec Work)

- [ ] Finalize CustomOpDef schema
- [ ] Finalize CustomTreeDef schema
- [ ] Finalize CustomRegimeDef schema
- [ ] Finalize CustomEnvelopeDef schema (curves, velocity scaling)
- [ ] Finalize CustomLFODef schema (waveforms, sync modes)
- [ ] Define proof obligation checkers
- [ ] Add axis tracking to preset format

### Phase 2: Quick Wins (Already ~Ready)

| Axis | Work Required | Effort |
|------|---------------|--------|
| FM | Add custom algorithm compiler | Medium |
| Harmonics | Expose OEIS morphing | Low |
| Events | Add pattern library | Medium |
| Carrier | Expose blend curves | Low |
| Pitch | Expose glide curves | Low |

### Phase 3: Compilation Pipeline (Code)

- [ ] Implement $C_{5\to3}$ for operators (kernel validation)
- [ ] Implement $C_{5\to3}$ for trees (split mask validation)
- [ ] Implement $C_{5\to3}$ for regimes (graph validation)
- [ ] Implement $C_{5\to3}$ for envelopes (curve interpolation)
- [ ] Implement $C_{5\to3}$ for LFOs (waveform tables)
- [ ] Implement proof validators
- [ ] Add axis projection logic

### Phase 4: Lab Mode UI (Editor)

- [ ] Custom operator studio
- [ ] Custom tree designer
- [ ] Custom regime graph builder
- [ ] Custom envelope curve editor
- [ ] Custom LFO waveform editor
- [ ] Proof obligation dashboard
- [ ] Live preview with compilation

---

## 7. Backward Compatibility

This proposal is **fully backward compatible**:

1. Existing presets have all axes at L0-L4
2. Missing axis levels default to L0
3. Custom definitions are optional extensions
4. Projection always works (custom → canonical)

---

## 9. Summary

| Current WSE-STD-070 | Proposed Augmentation |
|---------------------|----------------------|
| L5 = monolithic "Lab" | L5 = per-axis custom definitions |
| Single tower_level | Composite of **19 axis levels** |
| Undefined custom formats | **19 CustomDef schemas** specified |
| No proof obligations | Formal proof requirements per axis |
| No axis projection | Per-axis projection semantics |
| 7 implicit subsystems | **19 explicit axes in 4 tiers** |

### Axis Summary by Tier

| Tier | Focus | Axes | Count | Integration |
|------|-------|------|-------|-------------|
| **T1** | Core Synthesis | Op, Tree, Regime, Material, Macro, Budget, Safety | 7 | Mostly wired |
| **T2** | Modulation | Envelope, Modulation (LFO), Arcs | 3 | **OFF-PATH** (gap) |
| **T3** | Sources | Harmonics, FM, Samples, Granular, Carrier | 5 | Well wired |
| **T4** | Control | Voice, Pitch, Tuning, Events | 4 | Partially wired |
| | | **Total** | **19** | |

### Integration Status Summary

| Status | Axes | Notes |
|--------|------|-------|
| **L5 Ready** | Carrier, Pitch, Events | Already fully wired, custom compile possible |
| **Near Ready** | FM, Harmonics, Tuning, Samples | Need custom compile support |
| **Needs Lane Work** | Envelope, Modulation, Arcs | **Critical gap**: off-path, bypass lanes |
| **Needs Exposure** | Granular, Voice | Engine exists, not selectable |
| **Needs Custom Compile** | Op, Tree, Regime, Material, Macro, Budget, Safety | 24/5/7/8/11/7/16 fixed sets |

This makes L5 a **meaningful complexity layer** that exposes the internal structure of each subsystem for research and advanced authoring, while maintaining the safety guarantees of the Tower architecture.

**Critical Finding:** Tier 2 (Modulation) is the major integration gap. Envelopes, LFOs, and arcs run off-path and bypass the lanes system entirely. Enabling full L5 requires adding modulation lanes (2b, 2c) to the pipeline.

---

## 9. Implementation Priority

### Phase 1: High-Value Axes (Immediate)

| Axis | Rationale |
|------|-----------|
| **FM** | 32 algorithms exist, morphing is novel |
| **Harmonics** | OEIS sequences already implemented |
| **Envelope** | Curves already exist, just need exposure |
| **Modulation** | LFOs fully implemented, need custom waveforms |

### Phase 2: Medium-Value Axes (Near-term)

| Axis | Rationale |
|------|-----------|
| **Tuning** | 11 presets exist, custom formants valuable |
| **Voice** | Polyphony modes already implemented |
| **Pitch** | Microtonality frequently requested |
| **Events** | Pattern library enables rhythmic synthesis |

### Phase 3: Advanced Axes (Deferred)

| Axis | Rationale |
|------|-----------|
| **Granular** | Engine exists but less commonly used |
| **Samples** | Rompler mode is secondary use case |
| **Carrier** | Mostly internal, advanced users only |
| **Arcs** | Complex interaction with other systems |

---

## 10. Cross-References

### Implementation Files

| Tier | Axis | Primary Implementation |
|------|------|----------------------|
| T1 | Operators | `operator_library/operators.py` |
| T1 | Trees | `wavelet_trees/__init__.py` |
| T1 | Regimes | `voice/voice_engine.py` |
| T1 | Materials | `material/material_engine.py` |
| T1 | Macros | `macro_runtime.py` |
| T1 | Budgets | `budgets/degradation_ladder.py` |
| T1 | Safety | `safety/safety_engine.py` |
| T2 | Envelopes | `envelope_scheduler.py` |
| T2 | Modulation | `vm/harmonic_bank.py` (LFOState) |
| T2 | Arcs | `vm/state.py` (arc_phase_q16) |
| T3 | Harmonics | `vm/harmonic_bank.py` |
| T3 | FM | `vm/fm6_state.py`, `vm/fm_algorithms.py` |
| T3 | Samples | `vm/sample_source.py` |
| T3 | Granular | `vm/granular_engine.py` |
| T3 | Carrier | `wavelet/carrier_lift.py` |
| T4 | Voice | `voice/voice_manager.py` |
| T4 | Pitch | `tuning.py` |
| T4 | Tuning | `tuning.py` (InstrumentTuning) |
| T4 | Events | `vm/lanes.py` |

### Related Documentation

- [WSE-STD-070](../../../standards/architecture/WSE-STD-070.md) - Base standard
- [Full Tower v0](../../../../symcrash_docs/03%20-%20Audio%20Pipeline/01%20-%20Full%20Tower%20v0.md) - Theory
- [TOWER-CONFORMANCE-ROADMAP](../../roadmaps/TOWER-CONFORMANCE-ROADMAP.md) - Status
