# **WSE-STD-070 — Presets, Editor Operations, and Tower Completion (v1)**

This document **closes the architecture**. It defines how presets are represented, how editors are allowed to modify instruments, and how incomplete descriptions are deterministically completed across **Tower Levels L0–L5**.

**v1 Changes:** Introduces **19 Complexity Axes** for L5 (Lab) layer, replacing monolithic L5 with per-axis custom definitions across 4 tiers.

Language uses **MUST / MUST NOT / SHOULD / MAY**.

---

## 0. Scope and Intent

This standard defines:

1. **Preset format and lifecycle**
2. **Editor primitives** (what tools are allowed to change)
3. **Tower completion rules** (how missing layers are filled)
4. **Projection maps** (how to safely descend the tower)
5. **Determinism guarantees** (so Python, FPGA, and editor agree)

This standard does **not** define:

* Musical semantics
* Operator math
* Timing logic (owned by Voice Engine)
* Control mappings (UI-level, deferred)

---

## 1. Preset Object Model (Normative)

A **Preset** is a serialized description of an instrument at some Tower level.

### 1.1 Preset Header (REQUIRED)

Every preset MUST contain:

```text
preset_id          : u32
preset_version     : u16
tower_level        : u8   // 0..5
tree_id            : u16  // primary selector
flags              : u16
crc32              : u32
```

Normative rules:

* `tree_id` is the **primary selector** for synthesis identity.
* `tower_level` declares the **highest level explicitly authored**.
* CRC MUST be verified before loading; failure MUST reject the preset.

---

### 1.2 Preset Body (Layered)

Preset body is a **tower-structured object**:

```text
L0_patch_layer     : optional
L1_family_layer    : optional
L2_instrument_layer: optional
L3_cockpit_layer   : optional
L4_forge_layer     : optional
L5_lab_layer       : optional   # See §3.1 for axis structure
```

Normative rule:

> A preset MUST be valid even if **only one layer is present**.
> Missing layers MUST be filled by Tower Completion (Section 4).

#### L5 Lab Layer Structure (v1)

When present, L5_lab_layer contains **per-axis complexity levels** and **custom definitions**:

```text
L5_lab_layer:
  # Per-axis complexity levels (0-5 each)
  axis_levels:
    # Tier 1: Core Synthesis
    op_axis:       u8   # Operators
    tree_axis:     u8   # Wavelets
    regime_axis:   u8   # State machines
    material_axis: u8   # Effects
    macro_axis:    u8   # Choreography
    budget_axis:   u8   # Degradation
    safety_axis:   u8   # Invariants

    # Tier 2: Modulation
    envelope_axis:   u8   # ADSR curves
    modulation_axis: u8   # LFO waveforms
    arcs_axis:       u8   # Time-indexed patterns

    # Tier 3: Sources
    harmonics_axis: u8   # Partial profiles
    fm_axis:        u8   # FM algorithms
    samples_axis:   u8   # Multi-sample maps
    granular_axis:  u8   # Grain patterns
    carrier_axis:   u8   # Carrier waveforms

    # Tier 4: Control
    voice_axis:   u8   # Polyphony modes
    pitch_axis:   u8   # Microtonality
    tuning_axis:  u8   # Spectral shaping
    events_axis:  u8   # Event patterns

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

The preset's `tower_level` becomes the **maximum** across all axes:

```python
tower_level = max(axis_levels.values())
```

---

## 2. Editor Operations (Editor Primitives)

Editors MUST operate using **typed primitives**, not raw mutation.

### 2.1 Allowed Editor Primitives (v0)

Editors MAY emit only the following primitive operations:

#### Structural

* `SET_TREE_ID(tree_id)`
* `ADD_REGIME(regime_id)`
* `REMOVE_REGIME(regime_id)`
* `ADD_TRANSITION(src, dst)`
* `REMOVE_TRANSITION(src, dst)`

#### Parametric

* `SET_PARAM(path, value)`
* `DELTA_PARAM(path, delta)`
* `SET_BOUND(path, min, max)`

#### Policy

* `SET_OPERATOR_ALLOWED(op_id, bool)`
* `SET_OPERATOR_WEIGHT(op_id, weight)`
* `SET_BUDGET(limit)`

#### Meta

* `SET_TOWER_LEVEL(level)`
* `PROJECT_TO_LEVEL(level)`
* `RESET_TO_SAFE()`

#### Axis Control (L5)

* `SET_AXIS_LEVEL(axis, level)` — Set individual axis complexity
* `REGISTER_CUSTOM_OP(op_def)` — Register CustomOpDef
* `REGISTER_CUSTOM_TREE(tree_def)` — Register CustomTreeDef
* `REGISTER_CUSTOM_REGIME(regime_def)` — Register CustomRegimeDef
* `REGISTER_CUSTOM_MATERIAL(material_def)` — Register CustomMaterialDef
* `REGISTER_CUSTOM_MACRO(macro_def)` — Register CustomMacroDef
* `SET_CUSTOM_BUDGET(budget_def)` — Set CustomBudgetDef
* `REGISTER_CUSTOM_SAFETY(safety_def)` — Register CustomSafetyDef
* `REGISTER_CUSTOM_ENVELOPE(envelope_def)` — Register CustomEnvelopeDef
* `REGISTER_CUSTOM_LFO(lfo_def)` — Register CustomLFODef
* `REGISTER_CUSTOM_ARC(arc_def)` — Register CustomArcDef
* `PROJECT_AXIS(axis, target_level)` — Project single axis down

Normative rules:

* Editors MUST NOT emit operators, events, or timing logic.
* Editors MUST NOT mutate runtime state directly.
* All editor ops MUST be **pure** and replayable.
* Custom definitions MUST satisfy proof obligations (§3.2).

---

## 3. Tower Levels — Canonical Definition

This table is **normative**.

| Level  | Name       | Editor Primitives  | IR Primitives | Engine Primitives | Invariant            |
| ------ | ---------- | ------------------ | ------------- | ----------------- | -------------------- |
| **L0** | Patch      | MIDI maps, gains   | None          | Voice alloc only  | Always playable      |
| **L1** | Family     | Regime hints       | IntentFrame   | Voice Engine      | Recognizable class   |
| **L2** | Instrument | Params, bounds     | VoiceFrame    | Symbol Engine     | Identity preserved   |
| **L3** | Cockpit    | Weights, envelopes | Operator sets | Symbol + Safety   | No destructive edits |
| **L4** | Forge      | Graph edits        | Regime graph  | Compiler          | Always projectable   |
| **L5** | Lab        | Per-axis customs   | CustomDefs    | Offline tools     | Must compile down    |

### 3.1 Complexity Axes (Normative)

L5 is decomposed into **19 independent complexity axes** across **4 tiers**. Each axis MAY independently reach L5.

#### Tier 1: Core Synthesis (7 axes)

| Axis | ID | L0 (Fixed) | L2 (Tunable) | L4 (Editable) | L5 (Custom) | Canonical Set |
|------|----|------------|--------------|---------------|-------------|---------------|
| **Operators** | `op` | 24 canonical | Select subset | Reorder/weight | Define new | 24 ops (Op 0-23) |
| **Trees** | `tree` | 5 families | Select pair | Define splits | New wavelets | 5 families (T0-T4) |
| **Regimes** | `regime` | 7 states | Tune edges | Add/remove | Custom guards | 7 states (R0-R6) |
| **Materials** | `material` | 8 programs | Select one | Chain params | Custom DSP | 8 programs (M0-M7) |
| **Macros** | `macro` | 11 system | Bind presets | Slew policies | Choreography | 11 system (SM0-SM10) |
| **Budgets** | `budget` | Fixed costs | Tune limits | Custom ladder | Predictive | 7-step ladder |
| **Safety** | `safety` | 16 mechanisms | Enable/disable | Tune thresholds | Custom proofs | 16 mechanisms |

#### Tier 2: Modulation & Envelope (3 axes)

| Axis | ID | L0 (Fixed) | L2 (Tunable) | L4 (Editable) | L5 (Custom) |
|------|----|------------|--------------|---------------|-------------|
| **Envelopes** | `envelope` | Linear ADSR | Time params | Per-voice sets | Custom curves |
| **Modulation** | `modulation` | 4 LFO shapes | Rate/depth | Routing matrix | Custom waveforms |
| **Arcs** | `arcs` | 8 slots | Phase rates | Sync modes | Pattern library |

#### Tier 3: Oscillator & Source (5 axes)

| Axis | ID | L0 (Fixed) | L2 (Tunable) | L4 (Editable) | L5 (Custom) |
|------|----|------------|--------------|---------------|-------------|
| **Harmonics** | `harmonics` | 32 partials | Rolloff | OEIS select | Custom profiles |
| **FM** | `fm` | 32 algorithms | Op levels | Ratios/feedback | Custom algorithms |
| **Samples** | `samples` | Single sample | Loop points | Root note | Multi-sample maps |
| **Granular** | `granular` | 8 grains | Density/spray | Grain size | Pattern engines |
| **Carrier** | `carrier` | Sine + wet | Beta blend | Ramp time | Carrier waveforms |

#### Tier 4: Control & Allocation (4 axes)

| Axis | ID | L0 (Fixed) | L2 (Tunable) | L4 (Editable) | L5 (Custom) |
|------|----|------------|--------------|---------------|-------------|
| **Voice** | `voice` | 128 max | Polyphony cap | Steal mode | Priority schemes |
| **Pitch** | `pitch` | 12-TET | Bend range | Glide time | Microtonality |
| **Tuning** | `tuning` | FLAT preset | Select preset | Formant freq | Custom spectral |
| **Events** | `events` | 8 lanes | Lane enable | Payload types | Pattern library |

### 3.2 Proof Obligations (Normative)

Each custom definition MUST satisfy applicable proof obligations:

| Proof | Applies To | Meaning |
|-------|------------|---------|
| `bounded` | Op, Envelope, LFO, Arc, Granular | Output magnitude ≤ input magnitude × constant |
| `stable` | Material, Granular | No unbounded state growth |
| `has_exit` | Regime | State can transition out |
| `no_deadlock` | Regime | All states reachable |
| `covers_bandwidth` | Tree, Harmonics | No frequency gaps |
| `monotonic` | Budget, Pitch | Degradation always reduces cost; notes ascend |
| `always_playable` | Budget | Can always reach safe state |
| `acyclic` | FM | No infinite modulation loops (except feedback) |
| `has_carrier` | FM | At least one carrier output |
| `finite_duration` | Arc, Macro | Bounded duration if non-looping |
| `valid_lanes` | Events | All referenced lanes exist |

---

## 4. Tower Completion Rules (Critical)

Tower Completion defines how **missing layers are filled deterministically**.

### 4.1 Completion Direction

Completion always proceeds **upward**:

```text
Lk → Lk+1 → … → L5
```

Projection always proceeds **downward**:

```text
Lj → Lj-1 → … → L0
```

---

### 4.2 Default Completion (Normative)

If a layer is missing, the following defaults MUST be applied:

#### Missing L1 (Family)

* Infer family from `tree_id`
* Default regimes = `{STABLE}`

#### Missing L2 (Instrument)

* Bind tree ADSR = neutral
* Operator allowlist = SAFE_SET

#### Missing L3 (Cockpit)

* Envelope bounds = conservative
* Budget = nominal
* All unsafe ops disabled

#### Missing L4 (Forge)

* Graph edits forbidden
* Transitions fixed

#### Missing L5 (Lab) — Per-Axis

For each axis at level < L5:
* Use canonical set only
* Custom definitions forbidden
* No proof obligations required

For each axis at level = L5:
* Custom definitions allowed
* Compilation proof obligation MUST be satisfied
* Projection to L4 freezes custom → canonical mapping

Normative rule:

> Completion MUST be **pure, deterministic, and side-effect free**.

---

## 5. Projection Maps (Safety Descent)

Projection maps allow safe descent from higher levels.

### 5.1 Projection Operator

```text
PROJECT_TO_LEVEL(target_level)
```

Normative behavior:

* Removes all data above `target_level`
* Re-applies completion rules
* MUST preserve playability

Examples:

* L4 → L3: freeze graph, keep params
* L3 → L2: freeze envelopes, keep identity
* L2 → L0: collapse to playable patch

### 5.2 Axis Projection

`PROJECT_AXIS(axis, target_level)`:

* If target_level < current level, freeze custom to closest canonical
* Custom → canonical mapping is deterministic per axis
* Examples:
  - Custom tree → project to closest canonical family
  - Custom operator → project to closest canonical op + weight adjustment
  - Custom envelope → project to linear ADSR with approximated times
  - Custom FM algorithm → project to closest canonical algorithm (1-32)

---

### 5.3 Panic Projection

`RESET_TO_SAFE()` MUST:

* Project to **L2**
* Enable only SAFE operator set
* Reset envelopes to neutral
* Clear all event policies

This is your **“drop back to safety” button**.

---

## 6. Determinism Guarantees

The following MUST hold:

1. Same preset + same IntentFrame stream ⇒ same VoiceFrames
2. Same VoiceFrames ⇒ same Symbol execution
3. Same blocks ⇒ same Wavelet output (modulo numeric spec)
4. Tower completion MUST NOT consult runtime state

Editors MUST NOT introduce nondeterminism.

---

## 7. Preset Lifecycle

### 7.1 Load

1. Verify CRC
2. Parse header
3. Apply explicit layers
4. Complete missing layers
5. Project if needed
6. Activate

### 7.2 Save

* Serialize **only explicit layers**
* Record tower_level accordingly
* Never serialize completed defaults unless explicitly edited

---

## 8. What This Standard Guarantees (Why It Matters)

With WSE-STD-070 frozen:

* FPGA never sees partial or ambiguous state
* Python sim and hardware agree on meaning
* Editors can be powerful without being dangerous
* You can ship **L2/L3 instruments** without ever exposing L4/L5
* “Legendary instrument” depth emerges *without fragility*

---

## 9. Open (Explicitly Deferred)

These are **intentionally not frozen** in v0:

* UI control mappings
* Gesture vocabulary
* Operator taste / musical semantics
* Budget tuning constants
* Tree library expansion

All of these can evolve **without breaking compatibility**.

---

## 10. CMMM Event Bindings (Runtime)

The L3 Cockpit layer (CMMM) emits events that control runtime behavior.

### 10.1 Event Payload Types

| Payload Type | Name | Description |
|--------------|------|-------------|
| `0x30` | NODEMASK_SET | Change active node mask with ramp |
| `0x31` | TAILMODE_SET | Change tail mode with traction |

### 10.2 Node Mask Events (0x30)

```python
@dataclass
class NodeMaskPayload:
    node_mask_u32: int      # Target mask (32 leaves)
    ramp_blocks: int = 5    # Crossfade duration
```

Normative behavior:
* Executor MUST crossfade between old and new masks
* Crossfade uses Q15 linear interpolation
* K (active node count) = popcount(mask)
* Mask changes without ramp MUST still be click-free (minimum 1 block ramp)

### 10.3 Tail Mode Events (0x31)

```python
@dataclass
class TailModePayload:
    tail_mode: int          # 0=FULL, 1=MID, 2=MIN, 3=EMERGENCY
    traction_q15: int       # Projection hardness (0-32767)
```

Normative behavior:
* Tail mode determines which tail operators are enabled
* Traction scales enforcement strength
* Mode changes MAY be instant (no crossfade required)

### 10.4 Crossfade Implementation

During mask ramp, the executor MUST:

1. Track start mask and target mask
2. Compute blend weights: `alpha = progress / total_blocks`
3. Apply operators with BOTH masks
4. Blend outputs: `result = start_output * (1-alpha) + target_output * alpha`
5. Snap to target when ramp completes

This prevents clicks during K degradation (32→24→16→12→8→4→1).

---

## 11. Custom Definition Schemas (L5)

### 11.1 Tier 1: Core Synthesis

#### CustomOpDef (Operators)

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
  proof:
    bounded: true               # Output ≤ input magnitude
    monotonic: false
    preserves_dc: true
```

#### CustomTreeDef (Wavelets)

```yaml
custom_tree:
  tree_id: 100                  # Custom IDs start at 100
  name: "HarmonicExtended"
  base_family: 1                # Inherit from T1 (Harmonic)
  split_mask: 0xFFFF00FF        # Which nodes split further
  leaf_depths: [6,6,5,5,...]    # Depth per leaf position
  complex_leaf_mask: 0x0F       # Which leaves use complex coeffs
  proof:
    total_leaves: 32            # Must equal canonical
    covers_bandwidth: true      # No frequency gaps
```

#### CustomRegimeDef (State Machines)

```yaml
custom_regime:
  regime_id: 100
  name: "MyGlitch"
  kind: GlitchChatter           # Must map to canonical kind
  emit_template:
    families: {GLITCH: 0.9, CONSTRAINT: 0.5}
    ops_enable: [Tilt, BandShear, FineClamp]
    trees: {A: T1, B: 100, beta: "chaos"}  # Can reference custom tree
  entry_guard: "chaos > 0.8 && pressure > 0.5"
  exit_guard: "calm > 0.6"
  dwell_min_ms: 50
  proof:
    has_exit: true              # Must be able to leave
    no_deadlock: true           # Reachable from R0
```

#### CustomMaterialDef (Effects)

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
  proof:
    bounded_gain: 2.0           # Max output/input ratio
    stable: true                # No runaway feedback
```

#### CustomMacroDef (Choreography)

```yaml
custom_macro:
  macro_id: 100
  name: "AttackSwell"
  sequence:
    - {time_ms: 0, targets: {BRIGHTNESS: 0.2, RISK: 0.1}}
    - {time_ms: 100, targets: {BRIGHTNESS: 0.8, RISK: 0.3}}
    - {time_ms: 500, targets: {BRIGHTNESS: 0.5, RISK: 0.2}}
  trigger: gate_rise            # When to start
  loop: false
  proof:
    duration_ms: 500            # Bounded duration
    all_targets_valid: true     # Only known macros
```

#### CustomBudgetDef (Degradation)

```yaml
custom_budget:
  name: "AggressiveDegradation"
  k_ladder: [32, 16, 8, 4, 1]   # Custom K steps
  op_costs:
    100: {base: 100, per_K: 5}  # Custom op cost
  degradation_order:
    - {action: REDUCE_K, threshold: 0.9}
    - {action: DROP_MID, threshold: 0.8}
    - {action: DROP_TOP, threshold: 0.7}
    - {action: TAIL_ONLY, threshold: 0.5}
  proof:
    always_playable: true       # Can always reach tail-only
    monotonic: true             # Each step reduces cost
```

#### CustomSafetyDef (Invariants)

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
  proof:
    decidable: true             # Can always evaluate
    preserves_audio: true       # Doesn't destroy signal
```

### 11.2 Tier 2: Modulation

#### CustomEnvelopeDef

```yaml
custom_envelope:
  envelope_id: 100
  name: "SlowAttackPercussive"
  phases:
    - {phase: ATTACK, duration_ms: 500, curve: exponential, target: 1.0}
    - {phase: DECAY, duration_ms: 50, curve: linear, target: 0.7}
    - {phase: SUSTAIN, level: 0.7}
    - {phase: RELEASE, duration_ms: 200, curve: exponential, target: 0.0}
  velocity_scaling:
    attack_time: -0.3           # Faster attack at high velocity
    decay_time: 0.0
    sustain_level: 0.2          # Higher sustain at high velocity
  proof:
    bounded: true               # Always 0-1 range
    monotonic_attack: true
```

#### CustomLFODef

```yaml
custom_lfo:
  lfo_id: 100
  name: "SmoothRandom"
  waveform:
    type: custom_table          # Or: sine, triangle, saw_up, saw_down, square, sample_hold
    table_q15: [0, 8192, 16384, ...]  # 256 samples
  sync_mode: GATE_RESET         # FREE, GATE_RESET, LEGATO_RESET, TEMPO_SYNC
  tempo_division: "1/4"         # If TEMPO_SYNC
  default_routing:
    - {target: PITCH, depth_cents: 30}
    - {target: GAIN, depth_db: 3}
  proof:
    bounded: true               # Output in Q15 range
    continuous: true            # No discontinuities (for smooth waveforms)
```

#### CustomArcDef

```yaml
custom_arc:
  arc_id: 100
  name: "AttackDecayPattern"
  type: spline                  # Or: lfo, envelope, pattern, script
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

### 11.3 Tier 3: Sources

#### CustomHarmonicDef

```yaml
custom_harmonic:
  harmonic_id: 100
  name: "BellPartials"
  partials:
    - {harmonic: 1, amplitude: 1.0, phase: 0}
    - {harmonic: 2.4, amplitude: 0.7, phase: 0}   # Inharmonic!
    - {harmonic: 5.3, amplitude: 0.5, phase: 0}
    - {harmonic: 8.2, amplitude: 0.3, phase: 0}
  oeis_sequence: null           # Or: "A000040" for primes
  rolloff_db_per_octave: 6.0
  proof:
    finite_partials: true
    bounded_amplitude: true
```

#### CustomFMAlgorithmDef

```yaml
custom_fm_algorithm:
  algorithm_id: 100
  name: "CustomBrass"
  # 6x6 modulation matrix: mod_matrix[i][j] = how much op j modulates op i
  mod_matrix:
    - [0, 0, 0, 0, 0, 0]        # Op 1 (carrier)
    - [1, 0, 0, 0, 0, 0]        # Op 2 → Op 1
    - [0, 1, 0, 0, 0, 0]        # Op 3 → Op 2
    - [0, 0, 0, 0, 0, 0]        # Op 4 (carrier)
    - [0, 0, 0, 1, 0, 0]        # Op 5 → Op 4
    - [0, 0, 0, 0, 1, 1]        # Op 6 → Op 5 (with feedback)
  carrier_mask: 0b001001        # Ops 1 and 4 are carriers
  feedback_op: 6                # Op 6 has feedback
  morph_targets: [1, 5]         # Can morph to algorithms 1 and 5
  proof:
    acyclic: true               # No infinite modulation loops (except feedback)
    has_carrier: true           # At least one carrier
```

#### CustomSampleMapDef

```yaml
custom_sample_map:
  map_id: 100
  name: "PianoMultisample"
  zones:
    - {sample_id: "piano_A0", root_note: 21, low_note: 21, high_note: 35}
    - {sample_id: "piano_C2", root_note: 36, low_note: 36, high_note: 47}
    - {sample_id: "piano_C3", root_note: 48, low_note: 48, high_note: 59}
    - {sample_id: "piano_C4", root_note: 60, low_note: 60, high_note: 71}
  velocity_layers:
    - {threshold: 0, suffix: "_p"}
    - {threshold: 64, suffix: "_mf"}
    - {threshold: 100, suffix: "_f"}
  loop_mode: sustain            # none, sustain, release
  proof:
    covers_range: true          # No gaps in note range
    samples_exist: true         # All referenced samples loaded
```

#### CustomGranularDef

```yaml
custom_granular:
  name: "TextureCloud"
  max_grains: 16                # Default is 8
  grain_params:
    size_range_ms: [20, 200]
    density: 0.7                # 0-1, spawn probability
    spray: 0.3                  # Position randomization
  envelope:
    type: hanning               # Or: triangular, trapezoidal, custom
    attack_ratio: 0.15
    release_ratio: 0.15
  pitch_spread_cents: 50
  reverse_probability: 0.1
  proof:
    bounded_grains: true        # Never exceeds max_grains
    stable: true                # No amplitude explosion
```

#### CustomCarrierDef

```yaml
custom_carrier:
  name: "WarmCarrier"
  waveform: triangle            # Or: sine, triangle, saw, square, custom_table
  beta_curve:
    type: exponential           # Or: linear, s_curve
    attack_ms: 50               # Time to reach target beta
  default_beta_target: 0.75     # 75% wet (IDWPT), 25% carrier
  ola_window: hanning           # Or: triangular, blackman
  crossfade_blocks: 5
  proof:
    click_free: true            # Guaranteed smooth transitions
```

### 11.4 Tier 4: Control

#### CustomVoiceAllocDef

```yaml
custom_voice_alloc:
  name: "MonoLegatoPriority"
  polyphony_mode: mono          # mono, poly, unison
  max_voices: 1
  steal_mode: lowest_priority   # Or: oldest, quietest, none
  priority_source: velocity     # Or: note, fixed
  legato:
    enabled: true
    portamento_time_ms: 50
    retrigger_envelope: false
  unison:                       # If polyphony_mode == unison
    voices: 4
    detune_cents: 15
    stereo_spread: 0.7
  proof:
    bounded_voices: true        # Never exceeds max
    no_stuck_notes: true        # All notes eventually release
```

#### CustomPitchDef

```yaml
custom_pitch:
  name: "MicrotonalArabic"
  tuning:
    type: custom_scale          # Or: 12tet, just, pythagorean
    cents_per_note: [0, 100, 200, 300, 350, 500, 600, 700, 800, 900, 1000, 1100]
  bend_range_semitones: 2
  bend_curve: linear            # Or: exponential
  glide:
    mode: legato_only           # Or: always, off
    time_ms: 100
    curve: exponential
  proof:
    covers_octave: true         # 12 notes defined
    monotonic: true             # Notes ascend in pitch
```

#### CustomTuningDef (Spectral)

```yaml
custom_tuning:
  tuning_id: 100
  name: "NasalFormant"
  leaf_gains_q15: [32767, 30000, 28000, ...]  # 64 values
  # Or define via formants
  formants:
    - {freq_hz: 400, bandwidth_hz: 100, gain_db: 0}
    - {freq_hz: 2000, bandwidth_hz: 200, gain_db: -3}
    - {freq_hz: 3500, bandwidth_hz: 300, gain_db: -6}
  morph_compatible: [VOWEL_A, VOWEL_E]
  proof:
    bounded_gain: true          # No amplification > 1.0
```

#### CustomEventPatternDef

```yaml
custom_event_pattern:
  pattern_id: 100
  name: "GateArpeggio"
  events:
    - {time_ms: 0, lane: EVT_OP_PULSE, strength: 1.0}
    - {time_ms: 125, lane: EVT_OP_PULSE, strength: 0.8}
    - {time_ms: 250, lane: EVT_OP_PULSE, strength: 0.6}
    - {time_ms: 375, lane: EVT_OP_PULSE, strength: 0.8}
  loop: true
  loop_length_ms: 500
  sync_mode: tempo              # Or: free, gate
  tempo_division: "1/16"
  proof:
    finite_events: true         # Bounded event count per loop
    valid_lanes: true           # All lanes exist
```

### 11.5 Compilation Requirements ($C_{5\to3}$)

Each custom definition MUST compile to L3-compatible representation:

| Tier | Axis | Compiles To |
|------|------|-------------|
| T1 | CustomOpDef | Canonical op + weight matrix |
| T1 | CustomTreeDef | Canonical tree + split adjustments |
| T1 | CustomRegimeDef | Canonical regime + emit template |
| T1 | CustomMaterialDef | Canonical material + param overrides |
| T1 | CustomMacroDef | Macro sequence events |
| T1 | CustomBudgetDef | Budget table |
| T1 | CustomSafetyDef | Safety check function |
| T2 | CustomEnvelopeDef | ADSRSpec with interpolated curve points |
| T2 | CustomLFODef | LFOState + waveform LUT |
| T2 | CustomArcDef | Arc keyframe sequence |
| T3 | CustomHarmonicDef | HarmonicWeights table |
| T3 | CustomFMAlgorithmDef | FM6State + mod_matrix |
| T3 | CustomSampleMapDef | Sample zone lookup table |
| T3 | CustomGranularDef | GranularState configuration |
| T3 | CustomCarrierDef | CarrierLift params |
| T4 | CustomVoiceAllocDef | VoiceManager policy |
| T4 | CustomPitchDef | Pitch table + glide params |
| T4 | CustomTuningDef | InstrumentTuning (64 leaf gains) |
| T4 | CustomEventPatternDef | Event lane schedule |

Validation pipeline:

```
CustomDef → Parse → Validate Schema → Check Proofs → Compile → Register
                           ↓                ↓
                     Schema Error      Proof Failed
                           ↓                ↓
                       Reject           Reject with diagnostic
```

---

## 12. Axis Integration Status

### 12.1 Lane System Coverage

The lanes system (`vm/lanes.py`) is the central dispatch mechanism:

| Lane | Name | Function | Status |
|------|------|----------|--------|
| 1 | CARRIER_RENDER | Oscillator synthesis | ✓ Fully wired |
| 2 | INJECT | Event-based coefficient injection | ✓ Fully wired |
| 3-6 | DIAG/BANDED/PERMUTE/LOWRANK | Operator dispatch | ✓ Fully wired |
| 7 | TAIL | Safety constraints | ✓ Fully wired |
| 8 | IDWPT_RENDER | Wavelet synthesis | ✓ Fully wired |
| 9 | MIX_GAIN | Carrier + texture blend | ✓ Fully wired |

### 12.2 Per-Axis Integration Status

#### Tier 1: Core Synthesis

| Axis | Lanes | VoiceFrame | Preset | L5 Ready |
|------|-------|------------|--------|----------|
| **Op** | ✓ Lane 3-6 | `ops_enable` | ✓ | Needs custom compile |
| **Tree** | ✓ Lane 9 | `tree_a/b/blend` | ✓ | Needs custom compile |
| **Regime** | ✓ Via regime_id | `regime_id` | ✓ | Needs custom compile |
| **Material** | ~ Post-render | ParamFrame | ✓ | Needs custom compile |
| **Macro** | ~ ControlFrame | `w_family_q15[8]` | ✓ | Needs custom compile |
| **Budget** | ✓ Gating | `degradation_level` | ✓ | Needs custom compile |
| **Safety** | ✓ Lane 7 | `cmmm_tail_mode` | ✓ | Needs custom compile |

#### Tier 2: Modulation (**CRITICAL GAP**)

| Axis | Lanes | VoiceFrame | Preset | L5 Ready |
|------|-------|------------|--------|----------|
| **Envelope** | ✗ OFF-PATH | `env_targets_q15` | ✓ | Needs lane integration |
| **Modulation** | ✗ OFF-PATH | `fm_mod_buffer` | ~ | Needs lane integration |
| **Arcs** | ✗ OFF-PATH | `arc_phase_q16[8]` | ~ | Needs lane integration |

> **Problem:** Envelopes, LFOs, and arcs compute in VoiceEngine but bypass lanes entirely. Values written directly to VoiceFrame, not dispatched through operator pipeline.

#### Tier 3: Sources

| Axis | Lanes | VoiceFrame | Preset | L5 Ready |
|------|-------|------------|--------|----------|
| **Harmonics** | ✓ Mode B | `partial_amps_q15` | ✓ | Near ready |
| **FM** | ✓ Executor | `fm_*` fields | ✓ | Near ready |
| **Samples** | ✓ Executor | `sample_rate_q16` | ~ | Near ready |
| **Granular** | ✗ NOT EXPOSED | (none) | ✗ | Needs osc_mode exposure |
| **Carrier** | ✓ Lane 1+9 | `f0_hz`, `beta_target` | ✓ | ✓ Ready |

#### Tier 4: Control

| Axis | Lanes | VoiceFrame | Preset | L5 Ready |
|------|-------|------------|--------|----------|
| **Voice** | ✗ Policy layer | `voice_id` | ✗ | Needs preset exposure |
| **Pitch** | ✓ Phase inc | `note`, `cents` | ✓ | ✓ Ready |
| **Tuning** | ~ Post-render | (none) | ✓ | Near ready |
| **Events** | ✓ Lane 2 | `cmmm_node_mask` | ✓ | ✓ Ready |

### 12.3 Integration Summary

| Status | Axes |
|--------|------|
| **L5 Ready** | Carrier, Pitch, Events |
| **Near Ready** | FM, Harmonics, Tuning, Samples |
| **Needs Lane Work** | Envelope, Modulation, Arcs |
| **Needs Exposure** | Granular, Voice |
| **Needs Custom Compile** | Op, Tree, Regime, Material, Macro, Budget, Safety |

### 12.4 Architecture Gaps

| Priority | Gap | Impact | Fix Required |
|----------|-----|--------|--------------|
| **HIGH** | Tier 2 off-path | Envelopes/LFOs can't participate in operator gating | Add Lane 2b/2c for modulation |
| **HIGH** | Granular not exposed | Cannot select granular synthesis | Add Mode G to Executor |
| **MEDIUM** | Material post-render | Can't participate in degradation | Integrate into lane 8 or 9 |
| **MEDIUM** | Voice not preset-able | No per-preset polyphony control | Add voice mode fields |
| **LOW** | Custom Op/Tree compile | L5 can't define new primitives | Implement proof validators |

---

## 13. CMMM Implementation Status

### 13.1 Completed (Python)

| Component | Location | Status |
|-----------|----------|--------|
| ExecutorLatch | `vm/state.py` | ✓ Crossfade weights, ramp tracking |
| CMMM Event Processing | `vm/executor.py` | ✓ 0x30/0x31 decode |
| Crossfade Dispatch | `vm/lanes.py` | ✓ `dispatch_coeff_op_crossfade()` |
| Regime Registry | `library/regime_registry.py` | ✓ Graph storage |
| Tree Registry | `library/tree_registry.py` | ✓ Family storage |
| Pattern Table | `tables/pattern_table.py` | ✓ Mask patterns |
| Tail Modes | `tables/tail_modes.py` | ✓ Tail op sets |

### 13.2 Test Coverage

| Test File | Coverage |
|-----------|----------|
| `test_crossfade_dispatch.py` | ExecutorLatch weights, ramp behavior |
| `test_cmmm_events.py` | 0x30/0x31 roundtrip, K computation |
| `test_golden_voiceframe_stream.py` | Deterministic replay, snapshot restore |

---

## 14. Navigation Guide

### 14.1 Where to Start

| Goal | Read This |
|------|-----------|
| **Using presets** | §1-2 (Preset format, editor primitives) |
| **Writing instruments** | [PRESET-AUTHORING-GUIDE](../library/PRESET-AUTHORING-GUIDE.md), [InstrumentIR Conformance](../../design/core-system/04%20-%20InstrumentIR%20Tower%20Conformance%20Spec.md) |
| **Understanding theory** | [Full Tower v0](../../../symcrash_docs/03%20-%20Audio%20Pipeline/01%20-%20Full%20Tower%20v0.md), [Lifting the Tower](../../../symcrash_docs/00%20-%20Theory/04%20-%20Voice%20Engine%20Background/04%20-%20Lifting%20the%20Tower.md) |
| **Implementation status** | [TOWER-CONFORMANCE-ROADMAP](../../roadmaps/TOWER-CONFORMANCE-ROADMAP.md) |
| **FM6 synthesis** | [FM6_PATHWAY_GUIDE](../../FM6_PATHWAY_GUIDE.md) |
| **Control/macros** | [MACROS-AND-CONTROLS](../library/MACROS-AND-CONTROLS.md) |

### 14.2 CMMM Definition

Controlled Semi-Markov Mealy Machine: $\mathcal{M} = (S, U, P_\theta, E_\phi, M, Q)$

The Tower is a functorial lifting chain in the category of CMMMs. Each level adds structure via left adjoints.

See: [Voice Engine Tower - CMMM](../../../symcrash_docs/03%20-%20Audio%20Pipeline/00%20-%20Voice%20Engine/Evolution/02%20-%20The%20Voice%20Engine%20Tower.md)

### 14.3 Document Authority Matrix

| Topic | Location | Category |
|-------|----------|----------|
| Tower level definitions | This document §3 | Standard |
| Preset format | This document §1 | Standard |
| Editor primitives | This document §2 | Standard |
| Completion rules | This document §4 | Standard |
| Complexity axes | This document §3.1 | Standard |
| CustomDef schemas | This document §11 | Standard |
| CMMM definition | Voice Engine Tower | Theory |
| Functorial structure | Lifting the Tower | Theory |
| IR layer schemas | InstrumentIR Conformance | Design |
| Preset authoring | PRESET-AUTHORING-GUIDE | Guide |
| Implementation status | TOWER-CONFORMANCE-ROADMAP | Roadmap |

---

### 14.4 Implementation Files

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

### 14.5 Related Standards

| Standard | Purpose |
|----------|---------|
| [WSE-STD-001](../core/WSE-STD-001.md) | Fixed-Point Arithmetic |
| [WSE-STD-020](../core/WSE-STD-020.md) | UART Protocol |
| [WSE-STD-PRESET-v0](../control/WSE-STD-PRESET-v0.md) | Preset Format |
| [WSE-STD-ENVELOPE-SUITE-v0](../control/WSE-STD-ENVELOPE-SUITE-v0.md) | Envelope System |

### 14.6 Related Design Documents

| Document | Purpose |
|----------|---------|
| [InstrumentIR v0](../../design/core-system/03%20-%20InstrumentIR%20v0%20(Layered%20Tower%20IR).md) | Layered IR theory |
| [Tower Conformance Spec](../../design/core-system/04%20-%20InstrumentIR%20Tower%20Conformance%20Spec.md) | Conformance rules |
| [TOWER-CONFORMANCE-ROADMAP](../../roadmaps/TOWER-CONFORMANCE-ROADMAP.md) | Implementation status |

---

## 15. Closure Statement

> After WSE-STD-070 v1, the system is **architecture complete** with **19 explicit complexity axes**.

What remains is:

* implementation of custom compilers
* lane integration for Tier 2 modulation
* exposure of granular/voice modes
* tuning and taste

You are no longer inventing structure — you are **inhabiting it**.

---

## 16. Backward Compatibility

This v1 augmentation is **fully backward compatible**:

1. Existing presets have all axes at L0-L4 (implicit)
2. Missing axis levels default to L0
3. Custom definitions are optional extensions
4. Projection always works (custom → canonical)
5. `tower_level` computation is max across all axes
