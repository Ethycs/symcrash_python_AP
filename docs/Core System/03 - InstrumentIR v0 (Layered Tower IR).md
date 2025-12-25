# InstrumentIR v0 (Layered Tower IR)

Below is **InstrumentIR v0**: a layered, editor-friendly IR that starts at **L0 conventional DSP graph**, lifts to **L2 regimes**, and culminates in a **CMMM (L3–L5)** that emits **VoiceFramePacket v0b**.

It’s written as a **schema + normative semantics + completion rules**. You can store it as YAML or JSON.

---

## 0) Top-level object

```yaml
instrument_ir_v0:
  meta:
    instrument_id: "uuid-or-short-id"
    name: "Glitch Violin"
    version: 0
    author: "you"
    seed_u32: 123456789   # PRNG base seed (PCG32), deterministic
    voice_model:
      max_voices: 128
      tile_size: 16       # must be power of two
  io:
    sample_rate_hz: 48000
    block_size: 64
  layers:
    L0_graph: {...}       # conventional DSP graph
    L1_lift: {...}        # gesture basis + macro meaning
    L2_regimes: {...}     # regime graph (typed)
    L3_cmmm: {...}        # CMMM + overlays + transitions
  compile:
    completion_policy: {...}
    degrade_profile: {...}
    frame_emitter_policy: {...}
```

---

## 1) L0 GraphIR (Conventional DSP graph)

**Purpose:** familiar patch workflow (OSC → MIX → FILTER → AMP → FX).  
**Important:** L0 nodes are **semantic sources**; they compile into your primitives (families, events, harmonic injector, trees), not a separate audio engine.

```yaml
L0_graph:
  macros:
    # what the user sees at Level 0 (MIDI-patch style)
    - id: M_GAIN
      name: "Gain"
      range: [0.0, 1.0]
      default: 0.8
    - id: M_BRIGHT
      name: "Brightness"
      range: [0.0, 1.0]
      default: 0.5
    - id: M_CUTOFF
      name: "Cutoff"
      range: [0.0, 1.0]
      default: 0.6
    - id: M_RESO
      name: "Resonance"
      range: [0.0, 1.0]
      default: 0.1
    - id: M_NOISE
      name: "Noise"
      range: [0.0, 1.0]
      default: 0.0
    - id: M_GLITCH
      name: "Glitch"
      range: [0.0, 1.0]
      default: 0.0
    - id: M_HARM
      name: "Harmonic"
      range: [0.0, 1.0]
      default: 0.6

  nodes:
    - id: N_OSC
      type: Osc
      params:
        waveform: "saw"      # sine|saw|pulse|tri
        detune_cents: 0
        pw: 0.5              # only for pulse; optional
    - id: N_NOISE
      type: Noise
      params:
        color: "pink"        # white|pink|brown
    - id: N_MIX
      type: Mixer
      params: {}
    - id: N_FILTER
      type: Filter
      params:
        mode: "lp"           # lp|hp|bp|notch
    - id: N_AMP
      type: Amp
      params:
        adsr: {a_ms: 8, d_ms: 80, s: 0.7, r_ms: 180}
    - id: N_FX
      type: FXSend
      params:
        send_a: 0.15
        send_b: 0.0

  edges:
    - {src: "N_OSC.out",   dst: "N_MIX.in1"}
    - {src: "N_NOISE.out", dst: "N_MIX.in2"}
    - {src: "N_MIX.out",   dst: "N_FILTER.in"}
    - {src: "N_FILTER.out",dst: "N_AMP.in"}
    - {src: "N_AMP.out",   dst: "N_FX.in"}
```

---

## 2) L1 LiftIR (Gestures + macro meaning)

**Purpose:** defines the _control manifold_ and explains how L0 macros map into your internal primitives.

```yaml
L1_lift:
  gesture_basis:
    # normalized; editor can bind MIDI/CC/MPE here
    Gate:       {range: [0,1], source: "note_on_off"}
    Pressure:   {range: [0,1], source: "aftertouch_or_modwheel"}
    Position:   {range: [0,1], source: "cc74_or_macro"}
    Speed:      {range: [0,1], source: "derivative(Position)"}
    Tilt:       {range: [-1,1], source: "macro(M_BRIGHT) mapped bipolar"}
    Chaos:      {range: [0,1], source: "macro(M_GLITCH)"}

  macro_semantics:
    # maps L0 macro knobs into (a) families, (b) events, (c) safety constraints, (d) trees
    M_GAIN:
      affects:
        - {target: "frame.gain_q15", map: "adsr_vca"}   # L0 amp envelope
    M_BRIGHT:
      affects:
        - {target: "families.COLOR", map: "tilt_from_brightness"}
        - {target: "ops.Tilt.param", map: "tilt_db"}
        - {target: "ops.FineClamp.param", map: "hf_cap_from_brightness"}
    M_CUTOFF:
      affects:
        - {target: "families.TRANSPORT", map: "diffuse_balance_from_cutoff"}
        - {target: "ops.DiffuseDown.param", map: "gamma_from_cutoff"}
        - {target: "ops.DiffuseUp.param", map: "gamma_inverse"}
    M_RESO:
      affects:
        - {target: "events.EVT_OP_PULSE", map: "impulse_rate_from_reso"}
    M_NOISE:
      affects:
        - {target: "events.EVT_OP_PULSE", map: "noiseband_amount"}
    M_GLITCH:
      affects:
        - {target: "overlays.chaos.enable", map: "direct"}
    M_HARM:
      affects:
        - {target: "events.EVT_INJECT_NOTE", map: "harmonic_dial"}
```

---

## 3) L2 RegimeIR (Regime graph)

**Purpose:** the graph becomes an instrument via **modes**. Each regime describes a policy for emissions (families, ops, events, trees), and transitions are gesture-driven with hysteresis.

```yaml
L2_regimes:
  regime_nodes:
    - id: R0_IDLE
      kind: Idle
      emit_template: T_IDLE

    - id: R1_ATTACK
      kind: Transient
      emit_template: T_ATTACK
      dwell_ms: 20

    - id: R2_STABLE
      kind: StableTone
      emit_template: T_STABLE

    - id: R3_AIR
      kind: NoiseTexture
      emit_template: T_AIR

    - id: R4_GLITCH
      kind: GlitchChatter
      emit_template: T_GLITCH

    - id: R6_RECOVERY
      kind: Recovery
      emit_template: T_RECOVERY
      dwell_ms: 80

  transitions:
    # hysteresis is explicit: enter_th, exit_th
    - from: R0_IDLE
      to: R1_ATTACK
      when: "Gate rises"
    - from: R1_ATTACK
      to: R2_STABLE
      when: "dwell_done"
    - from: R2_STABLE
      to: R3_AIR
      when: "Speed > 0.6"
      hysteresis: {enter: 0.6, exit: 0.45}
    - from: R2_STABLE
      to: R4_GLITCH
      when: "Chaos > 0.6 and Pressure > 0.5"
      hysteresis: {enter: 0.6, exit: 0.4}
    - from: R3_AIR
      to: R2_STABLE
      when: "Speed < 0.45"
    - from: R4_GLITCH
      to: R6_RECOVERY
      when: "Chaos < 0.4 or Pressure < 0.3"
    - from: R6_RECOVERY
      to: R2_STABLE
      when: "dwell_done"
    - from: any
      to: R0_IDLE
      when: "Gate falls"
```

---

## 4) L3–L5 CMMMIR (Controlled Markov Mealy Machine)

**Purpose:** exact runtime semantics. CMMM emits **VoiceFrame fields** (families, toggles, events, tree targets) every block.

### 4.1 Core CMMM structure

```yaml
L3_cmmm:
  state:
    # persistent per-voice state
    variables:
      regime_id: {type: u8, init: "R0_IDLE"}
      dwell_counter: {type: u16, init: 0}
      overlay_mask: {type: u8, init: 0}
      pcg32: {type: u64, init: "seeded(meta.seed_u32, voice_id)"}

  inputs:
    # per block from gesture basis + note data
    gesture: [Gate, Pressure, Position, Speed, Tilt, Chaos]
    note: {note_u8: u8, cents_i16: i16, vel_u8: u8}
    block_index: u32

  outputs:
    # emitted each block (compiled to VoiceFramePacket v0b)
    frame_fields:
      - gain_q15
      - activity_q15
      - w_family_q15[8]
      - op_enable_mask_u32
      - tree_a_id, tree_b_id, tree_blend_q15
      - event_mask_u8
      - event_strength_q15[8]
      - event_payload_u64[8]
```

### 4.2 Emission templates (the “Mealy outputs”)

Each regime maps to an `EmitTemplate` describing:

- family weights
- enabled ops and params (expressed in “knob space”)
- event lane policies (conditions + payload templates)
- tree targets

```yaml
  emit_templates:

    T_IDLE:
      families: {COLOR: 0.1, TRANSPORT: 0.0, COHERENCE: 0.0, TOPOLOGY: 0.0, CONSTRAINT: 1.0, EXCITE: 0.0, CAT: 0.0, CALM: 1.0}
      ops_enable: [NormalizeScale, FineClamp]
      trees: {A: "T1", B: "T1", beta: 0.0}
      events: []

    T_ATTACK:
      families: {COLOR: 0.6, TRANSPORT: 0.2, COHERENCE: 0.2, TOPOLOGY: 0.0, CONSTRAINT: 1.0, EXCITE: 1.0, CAT: 0.0, CALM: 0.5}
      ops_enable: [Tilt, IntraMix, NormalizeScale, FineClamp]
      trees: {A: "T1", B: "T1", beta: 0.0}
      events:
        - lane: EVT_OP_PULSE
          condition: "Gate rises"
          payload_template: {variant: 0, op: Impulse, target_kind: pattern_id, pattern_id: 4, p0: 0, p1: 0, p2: 0}
          strength: "0.8"

    T_STABLE:
      families: {COLOR: "map(Tilt)", TRANSPORT: 0.2, COHERENCE: 0.4, TOPOLOGY: 0.1, CONSTRAINT: 1.0, EXCITE: 0.2, CAT: 0.0, CALM: 0.7}
      ops_enable: [Tilt, LowRankMorph, NormalizeScale, FineClamp, SheafProject]
      trees: {A: "T1", B: "T1", beta: 0.0}
      events:
        - lane: EVT_INJECT_NOTE
          condition: "Gate==1"
          payload_template: {variant: 0, wave_type: saw, partials_N: 12, pull_q8: 64}
          strength: "macro(M_HARM)"

    T_AIR:
      families: {COLOR: 0.3, TRANSPORT: 0.5, COHERENCE: 0.1, TOPOLOGY: 0.0, CONSTRAINT: 1.0, EXCITE: 0.6, CAT: 0.0, CALM: 0.8}
      ops_enable: [DiffuseDown, NormalizeScale, FineClamp]
      trees: {A: "T2", B: "T2", beta: 0.0}
      events:
        - lane: EVT_OP_PULSE
          condition: "macro(M_NOISE) > 0.01"
          payload_template: {variant: 0, op: NoiseBand, target_kind: pattern_id, pattern_id: 3, p0: 0, p1: 0, p2: 0}
          strength: "macro(M_NOISE)"

    T_GLITCH:
      families: {COLOR: 0.5, TRANSPORT: 0.3, COHERENCE: 0.2, TOPOLOGY: 0.8, CONSTRAINT: 1.0, EXCITE: 0.6, CAT: 0.2, CALM: 0.2}
      ops_enable: [RotateNodes, BandShear, IntraMix, NormalizeScale, FineClamp]
      trees: {A: "T1", B: "T0", beta: "map(Chaos)"}
      events:
        - lane: EVT_OP_PULSE
          condition: "Chaos > 0.5"
          payload_template: {variant: 0, op: Impulse, target_kind: pattern_id, pattern_id: 7, p0: 0, p1: 0, p2: 0}
          strength: "map(Chaos)"
        - lane: EVT_OP_PULSE
          condition: "Chaos > 0.7"
          payload_template: {variant: 0, op: GlobalPulse, target_kind: global, target_id: 0, p0: 0, p1: 0, p2: 0}
          strength: "0.4"

    T_RECOVERY:
      families: {COLOR: 0.2, TRANSPORT: 0.4, COHERENCE: 0.3, TOPOLOGY: 0.0, CONSTRAINT: 1.0, EXCITE: 0.0, CAT: 0.0, CALM: 1.0}
      ops_enable: [GluePair, SheafProject, NormalizeScale, FineClamp]
      trees: {A: "T1", B: "T1", beta: 0.0}
      events: []
```

> Note: $map(...)$ is a pure function from gestures/macros to $[0..1]$ (or bipolar). All such maps are declared in $compile.map_library$ (below) so they’re deterministic.

### 4.3 Transition kernel (weighted choices)

Where you want Markov behavior: use explicit weighted transitions with PCG32 draw sites.

```yaml
  transition_kernel:
    # evaluated in order; first matching rule applies
    - name: "attack"
      from: R0_IDLE
      to: R1_ATTACK
      type: deterministic
      when: "Gate rises"

    - name: "stable"
      from: R1_ATTACK
      to: R2_STABLE
      type: deterministic
      when: "dwell_done"

    - name: "glitch_enter"
      from: R2_STABLE
      to: R4_GLITCH
      type: weighted
      when: "Chaos > 0.6 and Pressure > 0.5"
      choices:
        - {to: R4_GLITCH, weight: 0.8}
        - {to: R3_AIR,    weight: 0.2}
      prng_draw_site: "CMMM.edge_choice"   # draw order site label

    - name: "recovery"
      from: R4_GLITCH
      to: R6_RECOVERY
      type: deterministic
      when: "Chaos < 0.4 or Pressure < 0.3"

    - name: "idle"
      from: any
      to: R0_IDLE
      type: deterministic
      when: "Gate falls"
```

**Determinism rule:** PRNG draws happen **only** when the transition rule type is $weighted$ AND the rule fires AND the voice is serviced (or you explicitly allow transition updates while frozen—v0 says no).

### 4.4 Overlays (conditional superposition)

Overlays add additional emissions without changing regime identity (unless specified).

```yaml
  overlays:
    - id: O_CHAOS
      enable_when: "Chaos > 0.5"
      kind: conditional
      add:
        families: {TOPOLOGY: "0.5*map(Chaos)"}
        events:
          - lane: EVT_OP_PULSE
            condition: "Chaos > 0.8"
            payload_template: {variant: 0, op: SwapBands, target_kind: global, target_id: 0, p0: 0, p1: 0, p2: 0}
            strength: "0.2"
      suppression_rules:
        - when: "transition_type == soft"
          suppress: ["catastrophe_ops", "catastrophe_events"]   # your rule
```

---

## 5) Compile section (completion + mapping library + emission policy)

### 5.1 Map library (deterministic functions)

```yaml
compile:
  map_library:
    map(Tilt):        "clamp_bipolar(gesture.Tilt)"
    map(Chaos):       "clamp01(gesture.Chaos)"
    map(Pressure):    "clamp01(gesture.Pressure)"
    macro(M_HARM):    "clamp01(macro.M_HARM)"
    tilt_from_brightness: "2*macro.M_BRIGHT - 1"    # bipolar
```

### 5.2 Completion policy (L0 → L5 fill)

If a higher layer is missing, it is generated deterministically.

```yaml
  completion_policy:
    if_missing_L1_lift: "derive gesture basis + macro maps from L0 defaults"
    if_missing_L2_regimes: "generate R0/R1/R2/R6 skeleton and templates from L0 blocks"
    if_missing_L3_cmmm: "compile L2 regimes into deterministic kernel with optional weighted edges off"
    always_add_tail_ops: [FineClamp, NormalizeScale, SheafProject]
    default_tree: {A: "T1", B: "T1", beta: 0.0}
```

### 5.3 Degrade profile (how features sacrifice under load)

This must align with your OrangeCrab scheduler’s Top/Mid throttles.

```yaml
  degrade_profile:
    # groups correspond to scheduler TOP/MID gating
    top_group:
      includes: ["overlays", "catastrophe_ops", "dense_patterns"]
      priority: "drop first"
    mid_group:
      includes: ["diffuse", "lowrank", "rich_events"]
      priority: "drop second"
    never_drop:
      includes: ["tail_ops", "gain_envelope", "basic_harmonic_anchor_if_enabled"]
```

### 5.4 FrameEmitter policy (how to emit VoiceFrame fields)

This is the bridge to the ABI; v0 rules:

```yaml
  frame_emitter_policy:
    voiceframe_version: 1
    endianness: "big"
    lane_limit: 8
    # lane assignment fixed
    lanes:
      EVT_OP_PULSE: 0
      EVT_OP_TOGGLE: 1
      EVT_TREE_MORPH: 2
      EVT_K_MODE: 3
      EVT_INJECT_NOTE: 4
      EVT_MATERIAL_HIT: 5
      EVT_CLOUD_PUSH: 6
    # event selection
    events:
      choose_order: "lane_id_ascending_then_template_order"
      merge_policy: "at most one payload per lane per block; strongest wins"
    # reject behavior
    reject_on:
      - "unknown tree_id"
      - "event lane payload lane_id mismatch"
      - "pattern_id out of range"
```

---

## 6) How the editor uses this IR

### Level 0 UI

- Edit $L0\_graph$ nodes/edges + macros
- Preview compiles to $L2/L3$ automatically

### Level 1 UI

- Edit gesture mapping + macro semantics

### Level 2 UI

- Edit regime graph and emission templates (friendly instrument language)

### Level 3–5 UI

- Edit transition kernel (weighted edges), overlays, catastrophe/recovery policies, pattern targets

### Always available debug panels

- show emitted $VoiceFramePacket$ fields
- show lane payload decode
- show TOP/MID due schedule + health bars

---

## 7) Why this matches your primitives (and stays FPGA-ready)

- All runtime work compiles down to:
    - $w\_family[8]$, op enables, events (lane payloads), tree $A/B/\beta$, gain
- PRNG draw sites are explicit ($prng\_draw\_site$ labels)
- Overlays and transition types match your taxonomy (soft, chaotic, recovery, conditional)
- Completion rules guarantee “partial specs still compile”
- Degrade profile aligns with scheduler TOP/MID throttles

---

If you want, next I'll write **FrameEmitter v0** as a canonical spec:

- exact rules to go from (regime + overlays + maps) → $VoiceFramePacket\ v0b$
- including fixed-point conversion, lane arbitration, and "no advance on skip/suppress" determinism rules.

---

## Addendum A: Carrier-First Semantics

### A.1 L0 Osc Node Extensions

The `Osc` node type gains carrier-specific fields:

```yaml
nodes:
  - id: N_OSC
    type: Osc
    params:
      waveform: "saw"        # sine|saw|pulse|tri|partials|noise
      detune_cents: 0
      pw: 0.5                # pulse width (for pulse waveform)
      role: "audible"        # off|audible|exciter_only (NEW)
      archetype: "harmonic"  # harmonic|bright|hollow|noisy|transient (NEW)
```

**Field definitions:**

- `role`: Carrier role enum
  - `"off"` → CarrierRole.OFF (0): No carrier; pure wavelet synthesis
  - `"audible"` → CarrierRole.AUDIBLE (1): Carrier blended into output via β
  - `"exciter_only"` → CarrierRole.EXCITER_ONLY (2): Carrier drives IDWPT but not heard directly

- `archetype`: TreeArchetype for scaffold generation
  - `"harmonic"` → TreeArchetype.HARMONIC (0)
  - `"bright"` → TreeArchetype.BRIGHT (1)
  - `"hollow"` → TreeArchetype.HOLLOW (2)
  - `"noisy"` → TreeArchetype.NOISY (3)
  - `"transient"` → TreeArchetype.TRANSIENT (4)

### A.2 L1 Macro Semantics: Carrier Targets

The M_HARM macro now affects carrier configuration instead of EVT_INJECT_NOTE:

```yaml
macro_semantics:
  M_HARM:
    affects:
      - {target: "carrier.eta", map: "harmonic_dial"}
      - {target: "carrier.detune", map: "harmonic_detune"}
```

Valid carrier targets:

| Target | Type | Description |
|--------|------|-------------|
| `carrier.model` | CarrierModel | Waveform selection (SINE, SAW, PULSE, TRI, PARTIALS, NOISE) |
| `carrier.role` | CarrierRole | OFF, AUDIBLE, or EXCITER_ONLY |
| `carrier.archetype` | TreeArchetype | Scaffold generation mode |
| `carrier.detune` | i16 | Detune in cents |
| `carrier.pw` | Q15 | Pulse width (for PULSE model) |
| `carrier.eta` | Q15 | Scaffold/texture blend target |
| `carrier.pitchbend` | i16 | Real-time pitchbend in cents |
| `carrier.transition` | TransitionKind | RETRIGGER, GLIDE, or BRIDGE |

### A.3 L3 Output Fields

Add carrier fields to `outputs.frame_fields`:

```yaml
outputs:
  frame_fields:
    - gain_q15
    - activity_q15
    - w_family_q15[8]
    - op_enable_mask_u32
    - tree_a_id, tree_b_id, tree_blend_q15
    # Carrier header (NEW)
    - carrier_model_u8
    - carrier_role_u8
    - carrier_archetype_u8
    - transition_kind_u8
    - detune_cents_i16
    - pw_q15
    - eta_target_q15
    - pitchbend_cents_i16
    # Events
    - event_mask_u8
    - event_strength_q15[8]
    - event_payload_u64[8]
```

### A.4 Lane Mapping Update

Lane 4 is renamed from `EVT_INJECT_NOTE` to `EVT_CARRIER_CFG`:

```yaml
frame_emitter_policy:
  lanes:
    EVT_OP_PULSE: 0
    EVT_OP_TOGGLE: 1
    EVT_TREE_MORPH: 2
    EVT_K_MODE: 3
    EVT_CARRIER_CFG: 4    # Was EVT_INJECT_NOTE
    EVT_MATERIAL_HIT: 5
    EVT_CLOUD_PUSH: 6
```

### A.5 Compiler Rules

#### TransitionKind Selection

The compiler selects TransitionKind based on note context:

| Condition | TransitionKind |
|-----------|----------------|
| New note without legato | RETRIGGER (0) |
| Legato flag set + previous note exists | GLIDE (1) |
| Tree crossfade active during note transition | BRIDGE (2) |

#### CarrierRole Selection

The compiler selects CarrierRole from L0 Osc node params:

| L0 Osc `role` param | CarrierRole |
|---------------------|-------------|
| `"off"` or missing | OFF (0) |
| `"audible"` | AUDIBLE (1) |
| `"exciter_only"` | EXCITER_ONLY (2) |

**Default:** If no `role` is specified, default to `AUDIBLE` for backwards compatibility with carrier-enabled instruments.

---