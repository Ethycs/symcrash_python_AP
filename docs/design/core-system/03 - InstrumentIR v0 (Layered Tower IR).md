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

## 2) L1 LiftIR (Gestures + Intent + Macro Meaning)

**Purpose:** defines the _control manifold_: raw gesture inputs, semantic intent mapping, and how L0 macros map into internal primitives.

**Reference:** WSE-STD-GESTURE v0 (§1-3)

```yaml
L1_lift:
  # ─────────────────────────────────────────────────────────────────────────
  # 2.1 Gesture Basis (8 raw controller inputs)
  # ─────────────────────────────────────────────────────────────────────────
  gesture_basis:
    # All values normalized to Q15 fixed-point
    # Editor binds MIDI/CC/MPE sources here
    Gate:       {type: bool, source: "note_on_off"}
    Pressure:   {type: Q15U, range: [0, 32767], source: "aftertouch_or_modwheel"}
    Position:   {type: Q15S, range: [-32768, 32767], source: "cc74_or_mpe_slide"}
    Speed:      {type: Q15U, range: [0, 32767], source: "derivative(Position)"}
    Tilt:       {type: Q15S, range: [-32768, 32767], source: "mpe_y_or_cc1"}
    Chaos:      {type: Q15U, range: [0, 32767], source: "macro(M_GLITCH)"}
    Pitch:      {type: Q15S, range: [-32768, 32767], source: "pitchbend"}
    Pan:        {type: Q15S, range: [-32768, 32767], source: "cc10_or_mpe_x"}

  # ─────────────────────────────────────────────────────────────────────────
  # 2.2 Intent Vector (8 semantic lanes)
  # ─────────────────────────────────────────────────────────────────────────
  # IntentVector is the semantic interpretation of gesture.
  # Maps raw controller state to "what the voice wants to do".
  #
  # Lane indices (frozen):
  #   I_TONE=0, I_BOW=1, I_AIR=2, I_GLITCH=3,
  #   I_WARP=4, I_SNAP=5, I_GLUE=6, I_CALM=7

  intent_vector:
    lanes:
      - {id: I_TONE,   name: "Tone",   desc: "Pitched tone strength"}
      - {id: I_BOW,    name: "Bow",    desc: "Sustained pressure / bowing"}
      - {id: I_AIR,    name: "Air",    desc: "Breathiness / noise desire"}
      - {id: I_GLITCH, name: "Glitch", desc: "Artifact / glitch desire"}
      - {id: I_WARP,   name: "Warp",   desc: "Time/pitch warping desire"}
      - {id: I_SNAP,   name: "Snap",   desc: "Transient / attack desire"}
      - {id: I_GLUE,   name: "Glue",   desc: "Coherence / binding desire"}
      - {id: I_CALM,   name: "Calm",   desc: "Recovery / simplification desire"}

  # ─────────────────────────────────────────────────────────────────────────
  # 2.3 Gesture → Intent Mapping
  # ─────────────────────────────────────────────────────────────────────────
  # Defines how raw gesture becomes semantic intent.
  # mode: "default" uses WSE-STD-GESTURE default_gesture_to_intent()
  # mode: "custom" uses the custom_map expressions below

  intent_mapping:
    mode: "default"   # "default" | "custom"

    # Custom mapping (used when mode == "custom")
    # Expressions reference gesture fields: Gate, Pressure, Position, Speed, Tilt, Chaos, Pitch, Pan
    # All outputs are Q15U [0, 32767]
    custom_map:
      I_TONE:   "Gate ? mul_q15(Pressure, Q15_HALF) + Q15_HALF/4 : 0"
      I_BOW:    "Pressure"
      I_AIR:    "sat16((Speed >> 1) + max(0, Q15_HALF - Pressure) >> 1)"
      I_GLITCH: "sat16(Chaos + (abs(Tilt) >> 1)) >> 1"
      I_WARP:   "sat16(abs(Pitch) + (abs(Position) >> 2))"
      I_SNAP:   "Gate && Speed > Q15_HALF ? Speed : 0"
      I_GLUE:   "max(0, Q15_ONE - Chaos)"
      I_CALM:   "(max(0, Q15_HALF - Pressure) + max(0, Q15_HALF - Speed)) >> 1"

  # ─────────────────────────────────────────────────────────────────────────
  # 2.4 Intent → Family Weight Mapping
  # ─────────────────────────────────────────────────────────────────────────
  # Defines base mapping from intent lanes to operator family weights.
  # Family indices: F_RESERVED=0, F_COLOR=1, F_TRANSPORT=2, F_LOCAL=3,
  #                 F_TOPOLOGY=4, F_GLUE=5, F_EVENTS=6, F_CATASTROPHE=7

  family_mapping:
    mode: "default"   # "default" | "custom"

    # Default mapping (from WSE-STD-GESTURE §8.2):
    #   F_COLOR     = (I_TONE + I_BOW) >> 1
    #   F_TRANSPORT = (I_WARP + I_AIR) >> 1
    #   F_LOCAL     = (I_BOW + I_TONE) >> 1
    #   F_TOPOLOGY  = (I_GLITCH + I_WARP) >> 1
    #   F_GLUE      = I_GLUE
    #   F_EVENTS    = (I_SNAP + I_GLITCH) >> 1
    #   F_CATASTROPHE = (regime == R5) ? Q15_ONE : 0

    custom_map:
      F_COLOR:       "(I_TONE + I_BOW) >> 1"
      F_TRANSPORT:   "(I_WARP + I_AIR) >> 1"
      F_LOCAL:       "(I_BOW + I_TONE) >> 1"
      F_TOPOLOGY:    "(I_GLITCH + I_WARP) >> 1"
      F_GLUE:        "I_GLUE"
      F_EVENTS:      "(I_SNAP + I_GLITCH) >> 1"
      F_CATASTROPHE: "regime == R5_CATASTROPHE ? Q15_ONE : 0"

  # ─────────────────────────────────────────────────────────────────────────
  # 2.5 Macro Semantics
  # ─────────────────────────────────────────────────────────────────────────
  # Maps L0 macro knobs into families, ops, events, trees

  macro_semantics:
    M_GAIN:
      affects:
        - {target: "frame.gain_q15", map: "adsr_vca"}
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
        - {target: "gesture.Chaos", map: "direct"}
        - {target: "overlays.chaos.enable", map: "threshold(0.5)"}
    M_HARM:
      affects:
        - {target: "carrier.eta", map: "harmonic_dial"}
```

---

## 3) L2 RegimeIR (Regime graph)

**Purpose:** the graph becomes an instrument via **modes**. Each regime describes a policy for emissions (families, ops, events, trees), and transitions are driven by **intent lanes** with hysteresis.

**Reference:** WSE-STD-GESTURE v0 (§4-6)

```yaml
L2_regimes:
  # ─────────────────────────────────────────────────────────────────────────
  # 3.1 Regime Nodes (R0-R6)
  # ─────────────────────────────────────────────────────────────────────────
  # Each regime specifies:
  #   - kind: semantic category
  #   - emit_template: reference to L3 template
  #   - regime_spec: per-regime behavior configuration
  #   - dwell_ms: minimum time before transition (optional)

  regime_nodes:
    - id: R0_IDLE
      kind: Idle
      emit_template: T_IDLE
      regime_spec:
        family_enable_mask: 0x00
        event_enable_mask: 0x00
        tower_level: L4_SCALAR

    - id: R1_ATTACK
      kind: Transient
      emit_template: T_ATTACK
      dwell_ms: 20
      regime_spec:
        preferred_tree: 3          # Transient tree
        event_enable_mask: 0x07    # Impulse, noise, pulse
        normalize_curve: "bright"
        tower_level: L1_COARSE

    - id: R2_STABLE
      kind: StableTone
      emit_template: T_STABLE
      regime_spec:
        preferred_tree: 1          # Harmonic tree
        event_enable_mask: 0x00
        normalize_curve: "pink"
        tower_level: L0_FULL

    - id: R3_AIR
      kind: NoiseTexture
      emit_template: T_AIR
      regime_spec:
        preferred_tree: 2          # Texture tree
        event_enable_mask: 0x02    # Noise only
        normalize_curve: "flat"
        tower_level: L1_COARSE

    - id: R4_GLITCH
      kind: GlitchChatter
      emit_template: T_GLITCH
      regime_spec:
        preferred_tree: 2          # Texture tree
        event_enable_mask: 0x2B    # Impulse, noise, gate, swallowtail
        normalize_curve: "pink"
        tower_level: L2_SPARSE

    - id: R5_CATASTROPHE
      kind: Catastrophe
      emit_template: T_CATASTROPHE
      regime_spec:
        family_enable_mask: 0xFF   # All families enabled
        event_enable_mask: 0xF0    # Cusp, swallowtail, fold, sheaf
        morph_rate: 256            # Slow morph during chaos
        tower_level: L2_SPARSE

    - id: R6_RECOVERY
      kind: Recovery
      emit_template: T_RECOVERY
      dwell_ms: 80
      regime_spec:
        preferred_tree: 4          # Drone tree (stable)
        event_enable_mask: 0x80    # Sheaf only
        morph_rate: 1024           # Fast morph to stability
        tower_level: L1_COARSE

  # ─────────────────────────────────────────────────────────────────────────
  # 3.2 Transitions (Intent-Driven)
  # ─────────────────────────────────────────────────────────────────────────
  # Transitions reference intent lanes (I_*) instead of raw gesture.
  # Hysteresis prevents oscillation between regimes.
  # Priority determines evaluation order (higher = checked first).

  transitions:
    # Gate-driven transitions (highest priority)
    - from: R0_IDLE
      to: R1_ATTACK
      when: "gesture.gate_rise"
      priority: 100
      edge_type: HARD
      arc_id: 1                    # attack_burst arc

    - from: any
      to: R0_IDLE
      when: "gesture.gate_fall"
      priority: 99
      edge_type: SOFT

    # Dwell-driven transitions
    - from: R1_ATTACK
      to: R2_STABLE
      when: "dwell_done"
      priority: 50
      edge_type: SOFT

    - from: R6_RECOVERY
      to: R2_STABLE
      when: "dwell_done"
      priority: 50
      edge_type: RECOVERY

    # Intent-driven transitions (use intent lanes)
    - from: R2_STABLE
      to: R3_AIR
      when: "intent.AIR > threshold"
      intent_threshold: {I_AIR: 19660}    # ~0.6 in Q15
      hysteresis: {enter_margin: 1000, exit_margin: 500, debounce_blocks: 4}
      priority: 30
      edge_type: SOFT

    - from: R3_AIR
      to: R2_STABLE
      when: "intent.AIR < threshold"
      intent_threshold: {I_AIR: 14745}    # ~0.45 in Q15
      priority: 30
      edge_type: SOFT

    - from: R2_STABLE
      to: R4_GLITCH
      when: "intent.GLITCH > threshold"
      intent_threshold: {I_GLITCH: 19660}  # ~0.6 in Q15
      hysteresis: {enter_margin: 1000, exit_margin: 500, debounce_blocks: 4}
      priority: 25
      edge_type: CHAOTIC

    - from: R3_AIR
      to: R4_GLITCH
      when: "intent.GLITCH > threshold"
      intent_threshold: {I_GLITCH: 22936}  # ~0.7 in Q15
      priority: 25
      edge_type: CHAOTIC

    - from: R4_GLITCH
      to: R5_CATASTROPHE
      when: "intent.GLITCH > threshold and intent.WARP > threshold"
      intent_threshold: {I_GLITCH: 26214, I_WARP: 19660}  # ~0.8, ~0.6
      priority: 20
      edge_type: CHAOTIC
      arc_id: 2                    # catastrophe_entry arc

    - from: R4_GLITCH
      to: R6_RECOVERY
      when: "intent.CALM > threshold"
      intent_threshold: {I_CALM: 13107}    # ~0.4 in Q15
      priority: 15
      edge_type: RECOVERY
      arc_id: 3                    # recovery_smooth arc

    - from: R5_CATASTROPHE
      to: R6_RECOVERY
      when: "intent.CALM > threshold"
      intent_threshold: {I_CALM: 16384}    # ~0.5 in Q15
      priority: 15
      edge_type: RECOVERY
      arc_id: 3

  # ─────────────────────────────────────────────────────────────────────────
  # 3.3 Arc Specifications
  # ─────────────────────────────────────────────────────────────────────────
  # Arcs are short-duration modulators applied during transitions.

  arcs:
    - id: 1
      name: "attack_burst"
      duration_blocks: 4
      family_boost: {F_COLOR: 8192, F_EVENTS: 16384}
      arc_events: 0x01             # E_IMPULSE
      gain_curve: "ease_in"

    - id: 2
      name: "catastrophe_entry"
      duration_blocks: 16
      family_boost: {F_TOPOLOGY: 16384, F_EVENTS: 16384, F_CATASTROPHE: 32767}
      arc_events: 0x50             # E_CUSP | E_FOLD
      gain_curve: "s_curve"
      morph_rate_override: 256

    - id: 3
      name: "recovery_smooth"
      duration_blocks: 32
      family_boost: {F_GLUE: 16384}
      arc_events: 0x80             # E_SHEAF
      gain_curve: "ease_out"
```

---

## 4) L3–L5 CMMMIR (Controlled Markov Mealy Machine)

**Purpose:** exact runtime semantics. CMMM emits **VoiceFrame fields** (families, toggles, events, tree targets) every block.

**Reference:** WSE-STD-GESTURE v0 (§11, §13)

### 4.1 Core CMMM structure

```yaml
L3_cmmm:
  state:
    # persistent per-voice state
    variables:
      regime_id: {type: u8, init: "R0_IDLE"}
      dwell_counter: {type: u16, init: 0}
      overlay_mask: {type: u8, init: 0}
      arc_id: {type: u8, init: 0}
      arc_counter: {type: u16, init: 0}
      pcg32: {type: u64, init: "seeded(meta.seed_u32, voice_id)"}
      # Slewed family weights (for smooth transitions)
      family_current_q15: {type: "i16[8]", init: "[0]*8"}
      # Slewed tree blend
      tree_blend_current_q15: {type: i16, init: 0}

  inputs:
    # ─────────────────────────────────────────────────────────────────────
    # Per-block inputs from controller + computed intent
    # ─────────────────────────────────────────────────────────────────────
    # Raw gesture basis (8 fields per WSE-STD-GESTURE §1)
    gesture:
      gate: bool
      gate_rise: bool              # True on rising edge
      gate_fall: bool              # True on falling edge
      pressure_q15: i16
      position_q15: i16
      speed_q15: i16
      tilt_q15: i16
      chaos_q15: i16
      pitch_q15: i16
      pan_q15: i16

    # Computed intent vector (8 lanes per WSE-STD-GESTURE §2)
    intent:
      tone_q15: i16                # I_TONE
      bow_q15: i16                 # I_BOW
      air_q15: i16                 # I_AIR
      glitch_q15: i16              # I_GLITCH
      warp_q15: i16                # I_WARP
      snap_q15: i16                # I_SNAP
      glue_q15: i16                # I_GLUE
      calm_q15: i16                # I_CALM

    # Note data
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
      # CMMM Runtime Control (WSE-STD-070)
      - cmmm_node_mask_u32        # Target node mask for K ladder
      - cmmm_mask_ramp_blocks     # Crossfade duration in blocks
      - cmmm_tail_mode_u8         # Tail operator set selection
      - cmmm_traction_q15         # Projection hardness
```

### 4.2 Emission templates (the "Mealy outputs")

Each regime maps to an `EmitTemplate` describing:

- family weights (now derived from **intent vector** per WSE-STD-GESTURE §8)
- enabled ops and params (expressed in "knob space")
- event lane policies (conditions + payload templates)
- tree targets (selected by intent per WSE-STD-GESTURE §10.2)

**Family weight computation:**
Templates can specify family weights in three modes:
1. **Static:** Fixed Q15 value (e.g., `0.5` → 16384)
2. **Intent-derived:** `"intent"` uses the default Intent→Family mapping
3. **Expression:** `"base + intent.GLITCH * 0.5"` for custom blends

```yaml
  emit_templates:

    T_IDLE:
      # Minimal processing - only tail ops
      families:
        mode: "static"
        values: {F_RESERVED: 0, F_COLOR: 3276, F_TRANSPORT: 0, F_LOCAL: 0, F_TOPOLOGY: 0, F_GLUE: 32767, F_EVENTS: 0, F_CATASTROPHE: 0}
      ops_enable: [NormalizeScale, FineClamp]
      trees: {A: 1, B: 1, beta: 0}
      events: []

    T_ATTACK:
      # Intent-driven with regime boost to F_EVENTS and F_COLOR
      families:
        mode: "intent_plus_boost"
        boost: {F_COLOR: 8192, F_EVENTS: 16384}   # Per WSE-STD-GESTURE §8.3
      ops_enable: [Tilt, IntraMix, NormalizeScale, FineClamp]
      trees: {A: 3, B: 3, beta: 0}  # Transient tree
      events:
        - lane: EVT_OP_PULSE
          condition: "gesture.gate_rise"
          payload_template: {variant: 0, op: Impulse, target_kind: pattern_id, pattern_id: 4, p0: 0, p1: 0, p2: 0}
          strength: "intent.snap_q15"

    T_STABLE:
      # Intent-driven with regime boost to tonal families
      families:
        mode: "intent_plus_boost"
        boost: {F_COLOR: 8192, F_LOCAL: 8192, F_GLUE: 8192}
        suppress: {F_TOPOLOGY: 0.5}  # Halve topology
      ops_enable: [Tilt, LowRankMorph, NormalizeScale, FineClamp, SheafProject]
      trees:
        selection: "by_intent"       # Per WSE-STD-GESTURE §10.2
        default: {A: 1, B: 1}
        beta: "intent.warp_q15"
      events:
        - lane: EVT_CARRIER_CFG
          condition: "gesture.gate"
          payload_template: {variant: 0, wave_type: saw, partials_N: 12, pull_q8: 64}
          strength: "macro(M_HARM)"

    T_AIR:
      # Boost transport/breath families
      families:
        mode: "intent_plus_boost"
        boost: {F_TRANSPORT: 16384, F_EVENTS: 8192}
      ops_enable: [DiffuseDown, NormalizeScale, FineClamp]
      trees: {A: 2, B: 2, beta: 0}  # Texture tree
      events:
        - lane: EVT_OP_PULSE
          condition: "macro(M_NOISE) > 328"   # > ~1%
          payload_template: {variant: 0, op: NoiseBand, target_kind: pattern_id, pattern_id: 3, p0: 0, p1: 0, p2: 0}
          strength: "macro(M_NOISE)"

    T_GLITCH:
      # Boost topology and events for artifact generation
      families:
        mode: "intent_plus_boost"
        boost: {F_TOPOLOGY: 16384, F_EVENTS: 16384}
      ops_enable: [RotateNodes, BandShear, IntraMix, NormalizeScale, FineClamp]
      trees:
        A: 1
        B: 0
        beta: "intent.warp_q15"    # Tree morph driven by warp intent
      events:
        - lane: EVT_OP_PULSE
          condition: "intent.glitch_q15 > 16384"   # > 0.5
          payload_template: {variant: 0, op: Impulse, target_kind: pattern_id, pattern_id: 7, p0: 0, p1: 0, p2: 0}
          strength: "intent.glitch_q15"
        - lane: EVT_OP_PULSE
          condition: "intent.glitch_q15 > 22936"   # > 0.7
          payload_template: {variant: 0, op: GlobalPulse, target_kind: global, target_id: 0, p0: 0, p1: 0, p2: 0}
          strength: 13107   # 0.4 in Q15

    T_CATASTROPHE:
      # All families enabled, catastrophe family at max
      families:
        mode: "intent_plus_boost"
        boost: {F_CATASTROPHE: 32767, F_TOPOLOGY: 16384, F_EVENTS: 16384}
      ops_enable: [SwapBands, RotateNodes, BandShear, FoldPressure, EmergencySimplify, NormalizeScale, FineClamp]
      trees: {A: 2, B: 0, beta: "intent.warp_q15"}
      events:
        - lane: EVT_CUSP
          condition: "intent.glitch_q15 > 26214"   # > 0.8
          payload_template: {variant: 0}
          strength: "intent.glitch_q15"
        - lane: EVT_FOLD
          condition: "always"
          payload_template: {variant: 0}
          strength: 16384

    T_RECOVERY:
      # Boost glue/coherence, suppress chaos
      families:
        mode: "intent_plus_boost"
        boost: {F_GLUE: 16384}
        suppress: {F_CATASTROPHE: 0, F_TOPOLOGY: 0.5, F_EVENTS: 0.5}
      ops_enable: [GluePair, SheafProject, NormalizeScale, FineClamp]
      trees: {A: 4, B: 4, beta: 0}  # Drone tree (stable)
      events:
        - lane: EVT_SHEAF
          condition: "always"
          payload_template: {variant: 0}
          strength: "intent.glue_q15"
```

> **Note:** `intent.*` references the computed IntentVector lanes. `gesture.*` references raw GestureBasis fields. `macro(M_*)` references L0 macros. All maps are declared in `compile.map_library` for determinism.

### 4.3 Transition kernel (intent-driven with weighted choices)

Transitions use **intent lanes** (per WSE-STD-GESTURE §6) with optional Markov weighted choices.

```yaml
  transition_kernel:
    # evaluated in priority order; first matching rule applies

    # ─────────────────────────────────────────────────────────────────────
    # Gate-driven transitions (highest priority)
    # ─────────────────────────────────────────────────────────────────────
    - name: "attack"
      from: R0_IDLE
      to: R1_ATTACK
      type: deterministic
      when: "gesture.gate_rise"
      priority: 100

    - name: "idle"
      from: any
      to: R0_IDLE
      type: deterministic
      when: "gesture.gate_fall"
      priority: 99

    # ─────────────────────────────────────────────────────────────────────
    # Dwell-driven transitions
    # ─────────────────────────────────────────────────────────────────────
    - name: "stable_after_attack"
      from: R1_ATTACK
      to: R2_STABLE
      type: deterministic
      when: "dwell_done"
      priority: 50

    - name: "stable_after_recovery"
      from: R6_RECOVERY
      to: R2_STABLE
      type: deterministic
      when: "dwell_done"
      priority: 50

    # ─────────────────────────────────────────────────────────────────────
    # Intent-driven transitions (use intent lanes)
    # ─────────────────────────────────────────────────────────────────────
    - name: "stable_to_air"
      from: R2_STABLE
      to: R3_AIR
      type: deterministic
      when: "intent.air_q15 > 19660"   # > 0.6
      hysteresis: {enter_margin: 1000, exit_margin: 500}
      priority: 30

    - name: "air_to_stable"
      from: R3_AIR
      to: R2_STABLE
      type: deterministic
      when: "intent.air_q15 < 14745"   # < 0.45
      priority: 30

    - name: "glitch_enter"
      from: R2_STABLE
      to: R4_GLITCH
      type: weighted
      when: "intent.glitch_q15 > 19660"   # > 0.6
      hysteresis: {enter_margin: 1000, exit_margin: 500}
      choices:
        - {to: R4_GLITCH, weight: 0.8}
        - {to: R3_AIR,    weight: 0.2}
      prng_draw_site: "CMMM.edge_choice"
      priority: 25

    - name: "glitch_to_catastrophe"
      from: R4_GLITCH
      to: R5_CATASTROPHE
      type: deterministic
      when: "intent.glitch_q15 > 26214 and intent.warp_q15 > 19660"  # > 0.8, > 0.6
      priority: 20

    - name: "recovery_from_glitch"
      from: R4_GLITCH
      to: R6_RECOVERY
      type: deterministic
      when: "intent.calm_q15 > 13107"   # > 0.4
      priority: 15

    - name: "recovery_from_catastrophe"
      from: R5_CATASTROPHE
      to: R6_RECOVERY
      type: deterministic
      when: "intent.calm_q15 > 16384"   # > 0.5
      priority: 15
```

**Determinism rule:** PRNG draws happen **only** when:
1. The transition rule type is `weighted`
2. The rule fires (condition met)
3. The voice is serviced (not frozen)

### 4.4 Overlays (conditional superposition)

Overlays add additional emissions without changing regime identity. Conditions use **intent lanes**.

```yaml
  overlays:
    - id: O_CHAOS
      enable_when: "intent.glitch_q15 > 16384"   # > 0.5
      kind: conditional
      add:
        families: {F_TOPOLOGY: "mul_q15(16384, intent.glitch_q15)"}  # 0.5 * glitch
        events:
          - lane: EVT_OP_PULSE
            condition: "intent.glitch_q15 > 26214"   # > 0.8
            payload_template: {variant: 0, op: SwapBands, target_kind: global, target_id: 0, p0: 0, p1: 0, p2: 0}
            strength: 6553   # 0.2 in Q15
      suppression_rules:
        - when: "transition_type == soft"
          suppress: ["catastrophe_ops", "catastrophe_events"]

    - id: O_WARP
      enable_when: "intent.warp_q15 > 16384"   # > 0.5
      kind: conditional
      add:
        trees:
          beta_mod: "intent.warp_q15"   # Boost tree morphing
```

---

## 5) Compile section (completion + mapping library + emission policy)

### 5.1 Map library (deterministic functions)

All mapping functions are deterministic and operate on Q15 fixed-point values.

**Reference:** WSE-STD-GESTURE v0 (§3, Appendix A)

```yaml
compile:
  map_library:
    # ─────────────────────────────────────────────────────────────────────
    # Q15 constants
    # ─────────────────────────────────────────────────────────────────────
    Q15_ONE:  32767
    Q15_HALF: 16384

    # ─────────────────────────────────────────────────────────────────────
    # Gesture field accessors
    # ─────────────────────────────────────────────────────────────────────
    map(Tilt):        "gesture.tilt_q15"
    map(Chaos):       "gesture.chaos_q15"
    map(Pressure):    "gesture.pressure_q15"
    map(Speed):       "gesture.speed_q15"
    map(Position):    "gesture.position_q15"
    map(Pitch):       "gesture.pitch_q15"

    # ─────────────────────────────────────────────────────────────────────
    # Intent lane accessors
    # ─────────────────────────────────────────────────────────────────────
    intent(TONE):     "intent.tone_q15"
    intent(BOW):      "intent.bow_q15"
    intent(AIR):      "intent.air_q15"
    intent(GLITCH):   "intent.glitch_q15"
    intent(WARP):     "intent.warp_q15"
    intent(SNAP):     "intent.snap_q15"
    intent(GLUE):     "intent.glue_q15"
    intent(CALM):     "intent.calm_q15"

    # ─────────────────────────────────────────────────────────────────────
    # Macro accessors
    # ─────────────────────────────────────────────────────────────────────
    macro(M_GAIN):    "mul_q15(macro.M_GAIN, Q15_ONE)"
    macro(M_BRIGHT):  "mul_q15(macro.M_BRIGHT, Q15_ONE)"
    macro(M_CUTOFF):  "mul_q15(macro.M_CUTOFF, Q15_ONE)"
    macro(M_RESO):    "mul_q15(macro.M_RESO, Q15_ONE)"
    macro(M_NOISE):   "mul_q15(macro.M_NOISE, Q15_ONE)"
    macro(M_GLITCH):  "mul_q15(macro.M_GLITCH, Q15_ONE)"
    macro(M_HARM):    "mul_q15(macro.M_HARM, Q15_ONE)"

    # ─────────────────────────────────────────────────────────────────────
    # Derived mappings
    # ─────────────────────────────────────────────────────────────────────
    tilt_from_brightness: "(macro.M_BRIGHT << 1) - Q15_ONE"   # bipolar
    hf_cap_from_brightness: "Q15_ONE - (macro.M_BRIGHT >> 1)"

    # ─────────────────────────────────────────────────────────────────────
    # Arithmetic primitives (per WSE-STD-GESTURE Appendix A)
    # ─────────────────────────────────────────────────────────────────────
    mul_q15_sat(a, b): "(a * b) >> 15, clamped to [-32768, 32767]"
    sat16(x):          "max(-32768, min(32767, x))"
    abs(x):            "x < 0 ? -x : x"
```

### 5.2 Completion policy (L0 → L5 fill)

If a higher layer is missing, it is generated deterministically.

```yaml
  completion_policy:
    # ─────────────────────────────────────────────────────────────────────
    # L1 completion: derive gesture + intent from defaults
    # ─────────────────────────────────────────────────────────────────────
    if_missing_L1_lift:
      gesture_basis: "use default 8-field GestureBasis (WSE-STD-GESTURE §1)"
      intent_mapping: "use default_gesture_to_intent() (WSE-STD-GESTURE §3)"
      family_mapping: "use default Intent→Family mapping (WSE-STD-GESTURE §8)"
      macro_semantics: "derive from L0 macro definitions"

    # ─────────────────────────────────────────────────────────────────────
    # L2 completion: generate minimal regime skeleton
    # ─────────────────────────────────────────────────────────────────────
    if_missing_L2_regimes:
      regime_nodes: "generate R0_IDLE, R1_ATTACK, R2_STABLE, R6_RECOVERY"
      transitions: "generate gate-driven + dwell-driven transitions only"
      regime_specs: "use default RegimeSpecs from WSE-STD-GESTURE §5.2"

    # ─────────────────────────────────────────────────────────────────────
    # L3 completion: compile regimes into CMMM
    # ─────────────────────────────────────────────────────────────────────
    if_missing_L3_cmmm:
      transition_kernel: "compile L2 transitions, weighted edges disabled"
      emit_templates: "generate from L2 regime_specs"
      overlays: "none"

    # ─────────────────────────────────────────────────────────────────────
    # Always applied
    # ─────────────────────────────────────────────────────────────────────
    always_add_tail_ops: [FineClamp, NormalizeScale, SheafProject]
    default_tree: {A: 1, B: 1, beta: 0}
    default_intent_mapping: "default"   # WSE-STD-GESTURE default_gesture_to_intent()
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

### A.5 CMMM Runtime Control Fields (WSE-STD-070)

Emit templates can include CMMM runtime control fields that get emitted to VoiceFrame for executor latch updates.

#### New VoiceFrame Fields

```yaml
outputs:
  frame_fields:
    # ... existing fields ...
    # CMMM Runtime Control (NEW - WSE-STD-070)
    - cmmm_node_mask_u32      # Target node mask for K ladder
    - cmmm_mask_ramp_blocks   # Crossfade duration in blocks
    - cmmm_tail_mode_u8       # Tail operator set selection
    - cmmm_traction_q15       # Projection hardness
```

#### Emit Template CMMM Fields

Templates can specify CMMM runtime control:

```yaml
emit_templates:
  T_STABLE:
    families:
      mode: "intent_plus_boost"
      boost: {F_COLOR: 8192, F_LOCAL: 8192}
    ops_enable: [Tilt, LowRankMorph, NormalizeScale, FineClamp, SheafProject]
    trees: {A: 1, B: 1, beta: 0}
    # CMMM Runtime Control (NEW)
    cmmm:
      node_mask: 0xFFFFFFFF     # Full K=32 mask
      mask_ramp_blocks: 8       # Smooth crossfade
      tail_mode: FULL           # All tail operators
      traction_q15: 32767       # Hard projection

  T_GLITCH:
    families:
      mode: "intent_plus_boost"
      boost: {F_TOPOLOGY: 16384, F_EVENTS: 16384}
    ops_enable: [RotateNodes, BandShear, IntraMix, NormalizeScale, FineClamp]
    trees: {A: 1, B: 0, beta: "intent.warp_q15"}
    # CMMM Runtime Control (NEW)
    cmmm:
      node_mask: 0x00FFFFFF     # K=24 degraded mask
      mask_ramp_blocks: 4       # Fast transition
      tail_mode: MID            # Reduced tail ops
      traction_q15: 16384       # Medium projection

  T_CATASTROPHE:
    families:
      mode: "intent_plus_boost"
      boost: {F_CATASTROPHE: 32767}
    ops_enable: [SwapBands, RotateNodes, FoldPressure, EmergencySimplify, FineClamp]
    trees: {A: 2, B: 0, beta: "intent.warp_q15"}
    # CMMM Runtime Control (NEW)
    cmmm:
      node_mask: 0x0000FFFF     # K=16 emergency mask
      mask_ramp_blocks: 2       # Rapid degradation
      tail_mode: EMERGENCY      # EmergencySimplify enabled
      traction_q15: 8192        # Soft projection
```

#### Tail Mode Values

| TailMode | ID | Operators Enabled |
|----------|----|--------------------|
| FULL | 0 | FineClamp, NormalizeScale, SheafProject |
| MID | 1 | FineClamp, NormalizeScale |
| MIN | 2 | FineClamp only |
| EMERGENCY | 3 | FineClamp, EmergencySimplify |

#### Pattern Table Integration

Canonical masks from the K ladder can be referenced by pattern_id:

```yaml
cmmm:
  node_mask: "pattern:4"        # K=16 from pattern table
  # Equivalent to: node_mask: 0x0000FFFF
```

| Pattern ID | K | Hex Mask |
|------------|---|----------|
| 0 | 32 | 0xFFFFFFFF |
| 1 | 24 | 0x00FFFFFF |
| 2 | 20 | 0x000FFFFF |
| 3 | 16 | 0x0000FFFF |
| 4 | 12 | 0x00000FFF |
| 5 | 8 | 0x000000FF |
| 6 | 4 | 0x0000000F |
| 7 | 1 | 0x00000001 |

#### Executor Latch State

The executor maintains per-voice CMMM state that persists across blocks:

```python
@dataclass
class ExecutorLatch:
    node_mask_u32: int = 0xFFFFFFFF     # Current active mask
    mask_target_u32: int = 0xFFFFFFFF   # Target for ramping
    mask_ramp_remaining: int = 0         # Blocks left in crossfade
    tail_mode: int = 0                   # Current tail operator set
    traction_q15: int = 32767            # Projection hardness
```

When VoiceFrame contains non-zero `cmmm_node_mask`, the executor:
1. Sets `mask_target_u32` to the new mask
2. Sets `mask_ramp_remaining` to `cmmm_mask_ramp_blocks`
3. Updates `tail_mode` and `traction_q15`
4. Uses `dispatch_coeff_op_crossfade()` for smooth transitions

---

### A.6 Compiler Rules

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