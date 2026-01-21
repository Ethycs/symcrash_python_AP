# WSE-STD-VOICE-PIPELINE v0 — Complete Voice Engine & IR Pipeline Specification

**Standard ID:** WSE-STD-VOICE-PIPELINE-v0
**Status:** Normative (v0 frozen)
**Last Updated:** 2026-01-20
**Scope:** Defines the complete voice engine pipeline from controller input to audio output, including all data structures, state machines, and execution semantics.

**Depends on:**
- WSE-STD-020 (VoiceFrame ABI)
- WSE-STD-070 (Tower Architecture, CMMM Events)
- WSE-STD-210 (CCS Oscillator Modes A/B)
- WSE-STD-VM-HYBRID (Execution Model)
- WSE-STD-GESTURE-v0 (Gesture & Regime FSM)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Data Structures](#3-data-structures)
4. [CMMM State Machine](#4-cmmm-state-machine)
5. [Gesture Processing](#5-gesture-processing)
6. [Intent Computation](#6-intent-computation)
7. [Regime FSM](#7-regime-fsm)
8. [Family Weight Computation](#8-family-weight-computation)
9. [Event System](#9-event-system)
10. [Tree Morphing](#10-tree-morphing)
11. [Arc Modulators](#11-arc-modulators)
12. [LFO System](#12-lfo-system)
13. [Envelope System](#13-envelope-system)
14. [System Macros](#14-system-macros)
15. [Filter System](#15-filter-system)
16. [VoiceFrame Emission](#16-voiceframe-emission)
17. [Execution Model](#17-execution-model)
18. [Degradation & Budgets](#18-degradation--budgets)
19. [Determinism Guarantees](#19-determinism-guarantees)
20. [Conformance Checklist](#20-conformance-checklist)

---

## 1. Overview

The Voice Engine Pipeline is a **Controlled Semi-Markov Mealy Machine (CMMM)** that transforms controller inputs into synthesizer configuration frames.

### 1.1 Formal Definition

```
M = ⟨S, U, P, E⟩
```

Where:
- **S**: State space (regime × phase × energy × arc × envelope)
- **U**: Control input (GestureBasis + MIDI + budget + macros)
- **P**: Transition kernel (stochastic regime edges with hysteresis)
- **E**: Emission function (VoiceFrame output per block)

### 1.2 Design Principles

1. **Block-rate operation**: All processing occurs at block boundaries (64 samples @ 48kHz = ~750 Hz)
2. **Determinism**: Same inputs always produce same outputs
3. **Snapshotability**: All state can be serialized for replay
4. **Graceful degradation**: CPU load triggers progressive simplification
5. **Tower separation**: L0-L5 layers have distinct responsibilities

---

## 2. Pipeline Architecture

### 2.1 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONTROLLER INPUT LAYER                           │
│         MIDI Notes, CC, Pitch Bend, Aftertouch, etc.                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GESTURE BASIS (8 fields)                         │
│   gate, pressure, position, speed, tilt, chaos, pitch, pan          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MACRO APPLICATION                                │
│      SystemMacro[0-12] → gesture modulation                         │
│      BRIGHTNESS → tilt, DENSITY → pressure, RISK → chaos            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GESTURE → INTENT MAPPING                         │
│      IntentVector[8]: tone, bow, air, glitch, warp, snap, glue, calm│
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    REGIME FSM (R0-R6)                               │
│      EdgeSpec evaluation, hysteresis, dwell timing                  │
│      Arc activation on transition                                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ FAMILY WEIGHTS  │  │  TREE MORPH     │  │  EVENT LANES    │
│ F0-F7 (8×Q15)   │  │ tree_a, tree_b  │  │ E0-E7 (8 lanes) │
│ regime + intent │  │ blend_q15       │  │ mask + strength │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ARC MODULATION                                   │
│      family_boost, event_firing, gain_curve                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ENVELOPE & LFO SYSTEM                            │
│      ADSR scheduler, LFO phases (arc_phase_q16[8])                  │
│      Matrix envelopes → FM operators, filter, etc.                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    VOICEFRAME EMISSION                              │
│      VoiceFrame: gain, pan, families, trees, events, filter, FM6   │
│      CMMM events: 0x30 (NODEMASK_SET), 0x31 (TAILMODE_SET)          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DOWNSTREAM PROCESSING                            │
│      Executor → Carrier/IDWPT → Material Engine → Safety → Output   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Processing Order (Per Block, MUST)

1. Receive GestureBasis from controller
2. Apply SystemMacro modulation to gesture
3. Map Gesture → IntentVector
4. Evaluate regime transitions via RegimeGraph
5. Activate Arc if transition occurred
6. Step Arc (if active)
7. Compute family weights from regime + intent + arc
8. Compute event lanes
9. Update tree morph state
10. Tick envelope scheduler (ADSR)
11. Tick LFOs and apply modulation
12. Apply degradation scaling
13. Emit VoiceFrame

---

## 3. Data Structures

### 3.1 VoiceState (Fully Snapshotable)

```python
@dataclass
class VoiceState:
    """Complete per-voice state - deterministically snapshotable."""

    # =========================================================================
    # LIFECYCLE
    # =========================================================================
    phase: VoicePhase               # FREE, ATTACK, DECAY, SUSTAIN, RELEASE
    phase_block: int                # Blocks in current lifecycle phase
    env_phase: VoicePhase           # Internal ADSR phase
    env_level_q15: int              # Current envelope value (Q15)
    env_counter: int                # Envelope block counter

    # =========================================================================
    # GESTURE AND INTENT
    # =========================================================================
    gesture: GestureBasis           # Physical inputs (8 normalized fields)
    intent: IntentVector            # Semantic lanes (8 computed values)
    dI_q15: List[int]               # Intent derivatives (8 values)

    # =========================================================================
    # CMMM FSM STATE
    # =========================================================================
    regime: Regime                  # Current regime (R0-R6)
    regime_blocks: int              # Dwell time in current regime
    previous_regime: Regime         # For arc lookup on transitions

    # =========================================================================
    # ARC MODULATOR STATE
    # =========================================================================
    arc: ArcState                   # Active transition effect
        # active: bool
        # arc_id: int
        # progress_blocks: int
        # total_blocks: int

    # =========================================================================
    # SYNTHESIS STATE
    # =========================================================================
    family_weights: List[int]       # 8 operator family weights (Q15)
    events: EventPayload            # Event lane payloads
    tree_morph: TreeMorphState      # Tree blending state
        # tree_a_id: int (0-4)
        # tree_b_id: int (0-4)
        # blend_q15: int (Q15)
        # morph_rate: int (blocks/transition)
        # is_morphing: bool

    # =========================================================================
    # ENVELOPE AND MODULATION
    # =========================================================================
    gain_q15: int                   # Voice gain envelope
    lfos: List[LFOState]            # LFO definitions
    arc_phase_q16: List[int]        # Unified phase storage (8 slots, Q0.16)
    envelope_matrix: Dict[str, Tuple[EnvelopeScheduler, List[int]]]

    # =========================================================================
    # RNG AND TIMING
    # =========================================================================
    rng_u32: int                    # xorshift32 deterministic RNG
    block_count: int                # Monotonic block counter
    velocity_q15: int               # Note velocity (Q15)

    # =========================================================================
    # NOTE AND PITCH
    # =========================================================================
    note_u8: int                    # MIDI note (0-127)
    cents_i16: int                  # Fine tuning (-2400 to +2400 cents)
    detune_cents: int               # Controller detune
    pitch_offset_cents: int         # Dynamic pitch offset

    # =========================================================================
    # SIMPLE MODE FIELDS (preset-only synthesis)
    # =========================================================================
    engine_mode: int                # EngineMode.SIMPLE or FULL_CMMM
    compiled_preset: Optional[CompiledPreset]
    beta_current_q15: int           # Carrier/Lift blend (current)
    beta_start_q15: int             # Ramp start value
    beta_target_q15: int            # Ramp target value
    beta_ramp_blocks: int           # Ramp duration
```

### 3.2 GestureBasis (8 Fields, Frozen v0)

```python
@dataclass
class GestureBasis:
    """Physical gesture inputs - raw controller state."""
    gate: bool = False              # Note on/off (boolean)
    pressure_q15: int = 0           # Aftertouch / bow pressure (0-32767)
    position_q15: int = 0           # Position on surface (bipolar: -32768 to +32767)
    speed_q15: int = 0              # Movement speed (0-32767)
    tilt_q15: int = 0               # Surface tilt (bipolar: -32768 to +32767)
    chaos_q15: int = 0              # Random/noise input (0-32767)
    pitch_q15: int = 0              # Pitch bend (bipolar: -32768 to +32767)
    pan_q15: int = 0                # Stereo position (bipolar: -32768 to +32767)
```

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `gate` | bool | 0..1 | Note on/off state |
| `pressure_q15` | Q15U | 0..32767 | Aftertouch / bow pressure |
| `position_q15` | Q15S | -32768..+32767 | Position on controller surface |
| `speed_q15` | Q15U | 0..32767 | Movement speed (magnitude) |
| `tilt_q15` | Q15S | -32768..+32767 | Surface tilt angle |
| `chaos_q15` | Q15U | 0..32767 | Random/noise component |
| `pitch_q15` | Q15S | -32768..+32767 | Pitch bend amount |
| `pan_q15` | Q15S | -32768..+32767 | Stereo position |

### 3.3 IntentVector (8 Lanes, Frozen v0)

```python
@dataclass
class IntentVector:
    """Semantic intent lanes - what the voice wants to do."""
    lanes: List[int] = field(default_factory=lambda: [0] * 8)
```

| Lane | Index | Purpose | High Value Indicates |
|------|-------|---------|---------------------|
| I_TONE | 0 | Pitched tone | Strong pitched sustain desired |
| I_BOW | 1 | Sustained pressure | Continuous bowing/breath |
| I_AIR | 2 | Breathiness | Desire for noise, air, texture |
| I_GLITCH | 3 | Artifacts | Desire for glitches, clicks |
| I_WARP | 4 | Warping | Morphing, pitch bending |
| I_SNAP | 5 | Transients | Attack snaps, plucks |
| I_GLUE | 6 | Coherence | Stability and binding |
| I_CALM | 7 | Recovery | Simplification from chaos |

### 3.4 VoiceFrame (Internal Pipeline Frame)

```python
@dataclass
class VoiceFrame:
    """Per-voice coefficient frame for synthesis pipeline."""

    # Identity
    voice_id: int = 0

    # Tree control (Option A)
    tree_a_id: int = 0              # Primary tree ID (0-4)
    tree_b_id: int = 0              # Secondary tree ID (0-4)
    tree_blend_q15: int = 0         # Blend: 0=A, 32767=B

    # Amplitude/Pan
    gain_q15: int = 16384           # Voice gain (Q15)
    pan_q15: int = 0                # Pan (bipolar Q15)

    # Musical envelopes
    tilt_env_q15: int = 16384       # Brightness envelope
    density_env_q15: int = 16384    # Density envelope

    # Voice state
    is_starting: bool = False       # First block after note-on
    is_stopping: bool = False       # Transitioning to FREE
    is_active: bool = True          # Voice producing output

    # Carrier/Lift Architecture
    f0_hz: float = 440.0            # Fundamental frequency
    beta_target_q15: int = 24576    # Dry/wet blend (0=carrier, 32767=wavelet)
    partial_amps_q15: List[int]     # Harmonic amplitudes for leaf gen
    allow_transient: bool = False   # Allow snappy changes

    # Oscillator Mode
    coeff_osc_mode: int = 0         # 0=LUT, 1=Mode B, 2=Sample, 3=FM6
    H_request: int = 16             # Harmonic count request
    waveform_type: int = 0          # 0=sine, 1=square, 2=pulse, 3=saw, etc.

    # PWM/Sync/FM
    pwm_width_q15: int = 16384      # Duty cycle (Q15)
    pwm_enabled: bool = False
    sync_enabled: bool = False
    sync_ratio_q16: int = 0
    fm_depth_q15: int = 0
    fm_source: int = 0

    # FM6 Parameters
    fm_algorithm_id: Optional[int] = None
    fm_op_ratios: Optional[List[int]] = None    # Q16 (6 operators)
    fm_op_levels: Optional[List[int]] = None    # Q15 (6 operators)
    fm_feedback_depth: Optional[int] = None
    fm_topo_a_id: Optional[int] = None
    fm_topo_b_id: Optional[int] = None
    fm_topo_blend_q15: int = 0

    # Matrix Envelope Targets
    env_targets_q15: Optional[dict] = None

    # CMMM Events (WSE-STD-070)
    cmmm_node_mask: Optional[int] = None
    cmmm_mask_ramp_blocks: int = 5
    cmmm_tail_mode: Optional[int] = None
    cmmm_traction_q15: int = 32767

    # Filter Parameters
    filter_cutoff_q15: int = 32767
    filter_reso_q15: int = 0
    filter_mode: int = 0            # 0=LP, 1=HP, 2=BP, 3=Notch
    filter_env_depth_q15: int = 0
```

### 3.5 VoiceFrameABI (Wire Format)

The VoiceFrameABI is the Mealy emission - complete per-voice output per block:

```python
@dataclass
class VoiceFrameABI:
    """Per-voice ABI output (Mealy emission)."""

    # Identity (4 bytes)
    program_id: int
    voice_id: int

    # Regime (1 byte)
    regime_id: int                  # 0-6

    # Amplitude (6 bytes)
    gain_q15: int                   # Voice envelope (Q15)
    pan_q15: int                    # Stereo pan (Q15)

    # Family weights (16 bytes)
    w_family_q15: List[int]         # 8 × Q15

    # Events (18 bytes)
    event_mask: int                 # Bitmask of active events
    event_strengths: List[int]      # 8 × Q15 (per lane)

    # Tree control (4 bytes)
    tree_a_id: int
    tree_b_id: int
    tree_blend_q15: int             # 0=A, 32767=B

    # Processing hints (4 bytes)
    degradation_level: DegradationLevel
    tower_level: TowerLevel

    # Tail policy (8 bytes)
    normalize_curve: str
    fineclamp_cap_ratio: float
    slew_max_delta: int

    # Arc state (6 bytes)
    arc_active: bool
    arc_id: int
    arc_progress_q15: int           # 0=start, 32767=complete

    # Note info (4 bytes)
    note_u8: int
    cents_i16: int

    # Lifecycle flags (1 byte)
    is_starting: bool
    is_stopping: bool
    is_active: bool
```

---

## 4. CMMM State Machine

### 4.1 Formal Model

The CMMM (Controlled Semi-Markov Mealy Machine) is defined as:

```
M = ⟨S, U, P, E⟩

S = Regime × VoicePhase × ArcState × EnvelopeState × TreeMorphState
U = GestureBasis × MacroState × BudgetState
P: S × U → Δ(S)  (transition probability kernel)
E: S × U → VoiceFrame  (emission function)
```

### 4.2 State Components

| Component | Type | Description |
|-----------|------|-------------|
| Regime | R0-R6 | Current FSM state |
| VoicePhase | FREE/ATTACK/DECAY/SUSTAIN/RELEASE | Lifecycle phase |
| ArcState | active, arc_id, progress | Transition modulator |
| EnvelopeState | ADSR phase + level | Amplitude envelope |
| TreeMorphState | tree_a, tree_b, blend | Coefficient tree blend |

### 4.3 Semi-Markov Property

Dwell time in each regime affects transition probability:

```python
def check_transition(regime, intent, dwell_blocks):
    for edge in get_edges_from(regime):
        if dwell_blocks >= edge.min_dwell_blocks:
            if edge.condition.check(intent):
                return edge.to_regime
    return None
```

---

## 5. Gesture Processing

### 5.1 Input Sources

| Source | Maps To | Notes |
|--------|---------|-------|
| MIDI Note On | gate=True, pressure=velocity | Initial attack |
| MIDI Note Off | gate=False | Release trigger |
| Aftertouch | pressure_q15 | Continuous pressure |
| Pitch Bend | pitch_q15 | Bipolar bend |
| Mod Wheel | chaos_q15 or custom | Configurable |
| XY Pad | position_q15, tilt_q15 | Spatial control |

### 5.2 Factory Method

```python
@classmethod
def from_note_on(cls, velocity_q15: int, pitch_q15: int = 0,
                 pan_q15: int = 0) -> 'GestureBasis':
    return cls(
        gate=True,
        pressure_q15=velocity_q15,
        position_q15=0,
        speed_q15=velocity_q15,
        tilt_q15=0,
        chaos_q15=0,
        pitch_q15=pitch_q15,
        pan_q15=pan_q15,
    )
```

---

## 6. Intent Computation

### 6.1 Default Gesture → Intent Mapping (MUST)

```python
def default_gesture_to_intent(g: GestureBasis) -> IntentVector:
    intent = IntentVector()

    # I_TONE: high when gate on and moderate pressure
    if g.gate:
        intent.lanes[I_TONE] = mul_q15_sat(g.pressure_q15, Q15_HALF) + Q15_HALF // 2

    # I_BOW: direct from pressure
    intent.lanes[I_BOW] = g.pressure_q15

    # I_AIR: increases with speed and low pressure
    air_from_speed = g.speed_q15 >> 1
    air_from_low_pressure = max(0, Q15_HALF - g.pressure_q15)
    intent.lanes[I_AIR] = sat16(air_from_speed + (air_from_low_pressure >> 1))

    # I_GLITCH: from chaos and extreme tilt
    glitch_from_chaos = g.chaos_q15
    glitch_from_tilt = abs(g.tilt_q15) >> 1
    intent.lanes[I_GLITCH] = sat16(glitch_from_chaos + glitch_from_tilt) >> 1

    # I_WARP: from pitch bend and position
    intent.lanes[I_WARP] = sat16(abs(g.pitch_q15) + (abs(g.position_q15) >> 2))

    # I_SNAP: high on gate-on with high speed
    if g.gate and g.speed_q15 > Q15_HALF:
        intent.lanes[I_SNAP] = g.speed_q15
    else:
        intent.lanes[I_SNAP] = 0

    # I_GLUE: inverse of chaos
    intent.lanes[I_GLUE] = max(0, Q15_ONE - g.chaos_q15)

    # I_CALM: low pressure and speed
    calm_pressure = max(0, Q15_HALF - g.pressure_q15)
    calm_speed = max(0, Q15_HALF - g.speed_q15)
    intent.lanes[I_CALM] = (calm_pressure + calm_speed) >> 1

    return intent
```

### 6.2 Mapping Rules (MUST)

1. All intent values MUST be clamped to `[0, Q15_ONE]`
2. Mapping MUST be deterministic
3. Custom mappings MAY replace default but MUST provide same interface

---

## 7. Regime FSM

### 7.1 Regime Enum (Frozen v0)

```python
class Regime(IntEnum):
    R0_IDLE           = 0  # Voice silent
    R1_ATTACK_IGNITION = 1  # Note-on transient
    R2_STABLE_TONE    = 2  # Sustained pitched tone
    R3_GRAIN_AIR      = 3  # Breathy texture
    R4_GLITCH_CHATTER = 4  # Intentional artifacts
    R5_CATASTROPHE    = 5  # Extreme chaos
    R6_RECOVERY_HEAL  = 6  # Return to stability
```

### 7.2 Regime State Graph

```
          ┌─────────────────────────────────────────────┐
          │                                             │
          ▼                                             │
    ┌──────────┐     gate_on      ┌──────────────────┐ │
    │ R0_IDLE  │ ─────────────────► R1_ATTACK        │ │
    └──────────┘                  └────────┬─────────┘ │
          ^                                │           │
          │                    dwell > 4   │           │
          │         ┌──────────────────────┘           │
          │         ▼                                  │
          │  ┌──────────────┐                          │
          │  │ R2_STABLE    │◄─────────────────────┐   │
          │  │    TONE      │                      │   │
          │  └──────┬───────┘                      │   │
          │         │                              │   │
          │    air > thr     tone > thr            │   │
          │         │              │               │   │
          │         ▼              │               │   │
          │  ┌──────────────┐      │               │   │
          │  │ R3_GRAIN_AIR │──────┘               │   │
          │  └──────┬───────┘                      │   │
          │         │                              │   │
          │   glitch > thr                         │   │
          │         │                              │   │
          │         ▼                              │   │
          │  ┌──────────────────┐                  │   │
          │  │ R4_GLITCH_CHATTER│                  │   │
          │  └──────┬───────────┘                  │   │
          │         │                              │   │
          │   chaos > thr                          │   │
          │         │                              │   │
          │         ▼                              │   │
          │  ┌──────────────┐     calm > thr       │   │
          │  │R5_CATASTROPHE│─────────────────►────┼───┘
          │  └──────┬───────┘                      │
          │         │                              │
          │   calm > thr                           │
          │         │                              │
          │         ▼                              │
          │  ┌──────────────────┐    glue > thr    │
          │  │ R6_RECOVERY_HEAL │──────────────────┘
          │  └──────┬───────────┘
          │         │
          │   gate_off
          │         │
          └─────────┘
```

### 7.3 RegimeSpec (Per-Regime Configuration)

```python
@dataclass
class RegimeSpec:
    regime: Regime

    # Family Policy
    family_base: List[int]          # 8 base weights (Q15)
    family_cap: List[int]           # 8 caps (Q15)
    family_enable_mask: int         # Bitmask of enabled families

    # Tree Policy
    preferred_tree: int             # 0-4
    tree_lock: bool                 # Lock to preferred tree
    morph_rate: int                 # Blocks per transition

    # Event Policy
    event_enable_mask: int          # Bitmask of enabled event lanes
    event_probability: List[int]   # 8 probabilities (Q15)

    # Tail Policy
    normalize_curve: str            # "pink", "flat", "bright"
    fineclamp_cap_ratio: float
    slew_max_delta: int

    # Tower Level
    tower_level: TowerLevel
```

### 7.4 Default RegimeSpecs (Normative v0)

| Regime | family_enable | preferred_tree | event_enable | tower_level |
|--------|---------------|----------------|--------------|-------------|
| R0_IDLE | 0x00 | - | 0x00 | L4_SCALAR |
| R1_ATTACK | 0xFF | 3 (Transient) | 0x07 | L1_COARSE |
| R2_STABLE | 0xFF | 1 (Harmonic) | 0x00 | L0_FULL |
| R3_GRAIN | 0xFF | 2 (Texture) | 0x02 | L1_COARSE |
| R4_GLITCH | 0xFF | 2 (Texture) | 0x2B | L2_SPARSE |
| R5_CATASTROPHE | 0xFF | 2 (Texture) | 0xF0 | L2_SPARSE |
| R6_RECOVERY | 0xFF | 4 (Drone) | 0x80 | L1_COARSE |

### 7.5 EdgeSpec (Transition Definition)

```python
@dataclass
class EdgeSpec:
    from_regime: Regime
    to_regime: Regime
    edge_type: EdgeType             # HARD, SOFT, CHAOTIC, CORRIDOR_ENTRY, RECOVERY
    condition: ConditionSpec
    hysteresis: HysteresisSpec
    min_dwell_blocks: int           # Minimum time in source regime
    duration_blocks: int            # Transition duration
    priority: int                   # Check order (higher first)
    arc_id: Optional[int]           # Arc to activate
```

### 7.6 HysteresisSpec (Anti-Oscillation)

```python
@dataclass
class HysteresisSpec:
    enter_margin_q15: int = 1000    # ~3% margin to enter
    exit_margin_q15: int = 500      # ~1.5% margin to exit
    debounce_blocks: int = 4        # Min blocks between transitions
    last_transition_block: int = 0
```

---

## 8. Family Weight Computation

### 8.1 Family Enum (8 Families)

```python
F_RESERVED    = 0  # Reserved / identity
F_COLOR       = 1  # Timbral color (Tilt, FineClamp)
F_TRANSPORT   = 2  # Energy transport (DiffuseUp/Down, BandShear)
F_LOCAL       = 3  # Local mixing (IntraMix, LowRankMorph)
F_TOPOLOGY    = 4  # Structural (SwapBands, RotateNodes)
F_GLUE        = 5  # Coherence (GluePair, NormalizeScale)
F_EVENTS      = 6  # Impulses, noise (Impulse, NoiseBand)
F_CATASTROPHE = 7  # Emergency (EmergencySimplify, FoldPressure)
```

### 8.2 Computation Pipeline

```python
def compute_family_weights(intent: IntentVector, regime: Regime,
                           arc: ArcState, program: InstrumentProgram,
                           degradation: DegradationLevel) -> List[int]:
    # Step 1: Start with regime base weights
    weights = regime_spec.family_base.copy()

    # Step 2: Add intent contributions
    weights[F_COLOR] += (intent.tone + intent.bow) >> 1
    weights[F_TRANSPORT] += (intent.warp + intent.air) >> 1
    weights[F_LOCAL] += (intent.bow + intent.tone) >> 1
    weights[F_TOPOLOGY] += (intent.glitch + intent.warp) >> 1
    weights[F_GLUE] += intent.glue
    weights[F_EVENTS] += (intent.snap + intent.glitch) >> 1

    # Step 3: F_CATASTROPHE only in R5
    if regime == Regime.R5_CATASTROPHE:
        weights[F_CATASTROPHE] = Q15_ONE
    else:
        weights[F_CATASTROPHE] = 0

    # Step 4: Apply regime caps
    for i in range(8):
        weights[i] = min(weights[i], regime_spec.family_cap[i])

    # Step 5: Apply enable mask
    for i in range(8):
        if not (regime_spec.family_enable_mask & (1 << i)):
            weights[i] = 0

    # Step 6: Apply program family_mods
    for i in range(8):
        weights[i] = mul_q15_sat(weights[i], program.family_mods[i])

    # Step 7: Apply arc boost
    if arc.active:
        arc_spec = get_arc_spec(arc.arc_id)
        progress = arc.progress_blocks / arc.total_blocks
        curve = apply_gain_curve(progress, arc_spec.gain_curve)
        for i in range(8):
            weights[i] += mul_q15_sat(arc_spec.family_boost[i], curve)

    # Step 8: Apply degradation
    if degradation >= DegradationLevel.LIGHT:
        weights[F_TRANSPORT] = mul_q15_sat(weights[F_TRANSPORT], 9830)  # 30%
    if degradation >= DegradationLevel.HEAVY:
        weights[F_TRANSPORT] = 0
        weights[F_TOPOLOGY] = 0
    if degradation >= DegradationLevel.EMERGENCY:
        for i in range(8):
            if i != F_GLUE:
                weights[i] = 0

    # Step 9: Saturate to Q15
    for i in range(8):
        weights[i] = sat16(weights[i])

    return weights
```

### 8.3 Regime Modifiers (Normative v0)

| Regime | Modifications |
|--------|---------------|
| R1_ATTACK | F_EVENTS += Q15_HALF, F_COLOR += Q15_HALF/2 |
| R2_STABLE | F_COLOR += Q15_HALF/2, F_LOCAL += Q15_HALF/2, F_GLUE += Q15_HALF/2, F_TOPOLOGY >>= 1 |
| R3_GRAIN | F_TRANSPORT += Q15_HALF, F_EVENTS += Q15_HALF/2 |
| R4_GLITCH | F_TOPOLOGY += Q15_HALF, F_EVENTS += Q15_HALF |
| R6_RECOVERY | F_GLUE += Q15_HALF, F_CATASTROPHE = 0, F_TOPOLOGY >>= 1, F_EVENTS >>= 1 |

---

## 9. Event System

### 9.1 Event Lane Enum (8 Lanes)

```python
E_IMPULSE     = 0  # Single-sample impulse
E_NOISE       = 1  # Noise burst
E_PULSE       = 2  # Pressure ramp
E_GATE        = 3  # Gate mask
E_CUSP        = 4  # Cusp injection (catastrophe)
E_SWALLOWTAIL = 5  # Swallowtail mask (catastrophe)
E_FOLD        = 6  # Fold pressure (catastrophe)
E_SHEAF       = 7  # Sheaf projection (recovery)
```

### 9.2 EventPayload Structure

```python
@dataclass
class EventPayload:
    # E_IMPULSE
    impulse_band: int
    impulse_strength_q15: int

    # E_NOISE
    noise_band: int
    noise_amount_q15: int

    # E_PULSE
    pulse_pressure_q15: int

    # E_GATE
    gate_mask: int                  # Bitmask

    # Catastrophe events
    cusp_amount_q15: int
    swallowtail_bands: int          # Bitmask
    fold_pressure_q15: int
    sheaf_threshold_q15: int

    # Active lanes
    active_mask: int                # Which events fire this block
```

### 9.3 Firing Rules by Regime

| Regime | Active Events |
|--------|---------------|
| R1_ATTACK | E_IMPULSE, E_NOISE, E_PULSE |
| R4_GLITCH | E_IMPULSE, E_NOISE, E_GATE, E_SWALLOWTAIL |
| R5_CATASTROPHE | E_CUSP, E_SWALLOWTAIL, E_FOLD, E_SHEAF |
| R6_RECOVERY | E_SHEAF only |

### 9.4 Lane Rule Types

```python
class LaneRuleType(IntEnum):
    DISABLED = 0     # Lane disabled
    GATE_EDGE = 1    # On gate rise/fall
    THRESHOLD = 2    # When intent exceeds threshold
    STOCHASTIC = 3   # Random with probability
    PATTERN = 4      # Pattern-driven by arc
```

---

## 10. Tree Morphing

### 10.1 TreeMorphState

```python
@dataclass
class TreeMorphState:
    tree_a_id: int = 0              # Primary tree (0-4)
    tree_b_id: int = 0              # Secondary tree (during morph)
    blend_q15: int = 0              # 0 = pure A, Q15_ONE = pure B
    morph_rate: int = 512           # Blend increment per block
    is_morphing: bool = False       # Transition in progress
```

### 10.2 Tree Archetypes (5 Trees)

| ID | Name | Use Case |
|----|------|----------|
| 0 | Balanced | Uniform resolution (default) |
| 1 | Harmonic | Pitched, tonal sounds |
| 2 | Texture | Noisy, breathy sounds |
| 3 | Transient | Attack, pluck, percussive |
| 4 | Drone | Stable, ambient, sustained |

### 10.3 Tree Selection Algorithm

```python
def select_tree(intent: IntentVector, regime: Regime) -> int:
    scores = [0] * 5

    # Base scores from intent
    scores[0] = Q15_HALF                           # Balanced: default
    scores[1] = (intent.tone + intent.bow) >> 1    # Harmonic: pitched
    scores[2] = intent.air + (intent.glitch >> 1)  # Texture: air/glitch
    scores[3] = (intent.snap + intent.warp) >> 1   # Transient: percussive
    scores[4] = intent.calm                        # Drone: sustained

    # Regime biases
    if regime == Regime.R1_ATTACK:
        scores[3] += Q15_HALF
    elif regime == Regime.R2_STABLE:
        scores[1] += Q15_HALF
    elif regime == Regime.R3_GRAIN:
        scores[2] += Q15_HALF
    elif regime == Regime.R4_GLITCH:
        scores[2] += Q15_HALF // 2
        scores[3] += Q15_HALF // 2
    elif regime == Regime.R5_CATASTROPHE:
        scores[2] += Q15_HALF
    elif regime == Regime.R6_RECOVERY:
        scores[4] += Q15_HALF

    return argmax(scores)
```

---

## 11. Arc Modulators

### 11.1 ArcSpec Structure

```python
@dataclass
class ArcSpec:
    arc_id: int
    name: str
    duration_blocks: int
    family_boost: List[int]         # 8 Q15 boosts
    arc_events: int                 # Event bitmask
    gain_curve: str                 # "linear", "ease_in", "ease_out", "s_curve"
    morph_rate_override: int        # 0 = use default
```

### 11.2 Built-in Arcs (Normative v0)

| Arc ID | Name | Duration | Family Boost | Events | Curve |
|--------|------|----------|--------------|--------|-------|
| 1 | attack_burst | 4 | F_LOCAL, F_EVENTS | E_IMPULSE | ease_in |
| 2 | catastrophe_entry | 16 | F_TOPOLOGY, F_CATASTROPHE | E_CUSP, E_FOLD | s_curve |
| 3 | recovery_smooth | 32 | F_GLUE | E_SHEAF | ease_out |

### 11.3 Arc Gain Curves

```python
def apply_gain_curve(progress: float, curve: str) -> int:
    """Apply gain curve to progress [0,1], return Q15 weight."""
    if curve == "linear":
        return int(progress * Q15_ONE)
    elif curve == "ease_in":
        return int((progress ** 2) * Q15_ONE)
    elif curve == "ease_out":
        return int((1 - (1 - progress) ** 2) * Q15_ONE)
    elif curve == "s_curve":
        s = 3 * progress ** 2 - 2 * progress ** 3
        return int(s * Q15_ONE)
```

---

## 12. LFO System

### 12.1 LFOState Structure

```python
@dataclass
class LFOState:
    id: str                         # LFO name
    waveform: LFOWaveform           # SINE, TRIANGLE, SAW_UP, SQUARE
    sync_mode: LFOSyncMode          # FREE, GATE_RESET, LEGATO_RESET
    routings: List[LFORouting]      # Target destinations

    # Configuration
    depth_q15: int = 16384          # Modulation depth (Q15)
    arc_slot: int = 0               # Phase storage slot in arc_phase_q16
    rate_hz: float = 5.0            # LFO frequency

    # Timing
    delay_samples: int = 0          # Delay before LFO starts
    ramp_samples: int = 0           # Ramp-in duration
```

### 12.2 Phase Storage (Unified with CMMM)

LFO phases are stored in `VoiceState.arc_phase_q16[arc_slot]`:

- 8 slots available (MAX_ARCS = 8)
- Q0.16 format (0-65535 = one cycle)
- Unified time ownership per WSE-ARCHITECTURE-v1

### 12.3 LFO Targets

```python
class LFOTarget(IntEnum):
    PITCH = 0           # Pitch modulation (vibrato)
    GAIN = 1            # Amplitude modulation (tremolo)
    PWM_WIDTH = 10      # Pulse width modulation
    TILT = 11           # Spectral tilt
    FILTER_CUTOFF = 15  # Filter cutoff frequency
    FILTER_RESO = 16    # Filter resonance
    BAND_SCALE = 20     # Wavelet band scaling
    VFO_RING = 30       # Ring modulation
```

### 12.4 LFO Waveforms

```python
class LFOWaveform(IntEnum):
    SINE = 0
    TRIANGLE = 1
    SAW_UP = 2
    SAW_DOWN = 3
    SQUARE = 4
    SAMPLE_HOLD = 5
```

### 12.5 LFO Routing

```python
@dataclass
class LFORouting:
    lfo_id: int
    target: LFOTarget
    depth_q15: int                  # Modulation depth
```

---

## 13. Envelope System

### 13.1 ADSRSpec

```python
@dataclass
class ADSRSpec:
    attack_ms: float = 10.0
    decay_ms: float = 100.0
    sustain: float = 0.5            # 0.0-1.0
    release_ms: float = 100.0
```

### 13.2 EnvelopeScheduler

```python
@dataclass
class EnvelopeScheduler:
    spec: ADSRSpec

    # State
    phase: EnvelopePhase            # IDLE, ATTACK, DECAY, SUSTAIN, RELEASE
    phase_block: int                # Position in current phase
    current_q15: int                # Current envelope value
    release_level_q15: int          # Level when release started

    def trigger_note(self):
        """Start attack phase."""
        self.phase = EnvelopePhase.ATTACK
        self.phase_block = 0

    def release_note(self):
        """Transition to release phase."""
        self.release_level_q15 = self.current_q15
        self.phase = EnvelopePhase.RELEASE
        self.phase_block = 0

    def tick(self) -> int:
        """Advance one block, return Q15 envelope value."""
        ...
```

### 13.3 Matrix Envelope System

The matrix envelope system routes envelopes to multiple targets:

```python
class EnvelopeTarget(IntEnum):
    FM_OP1 = 0
    FM_OP2 = 1
    FM_OP3 = 2
    FM_OP4 = 3
    FM_OP5 = 4
    FM_OP6 = 5
    GAIN = 10
    FILTER_CUTOFF = 20
    FILTER_RESO = 21
    PITCH = 30
    VIBRATO_DEPTH = 31
```

---

## 14. System Macros

### 14.1 System Macro IDs (Frozen v0)

| ID | Name | Purpose | Set By | Slew |
|----|------|---------|--------|------|
| 0 | ENERGY_STATE | Envelope-owned | Envelope | NONE |
| 1 | ENERGY_INTENT | Attack character | Controller | ONE_POLE |
| 2 | MOTION | Movement/modulation | Controller | ONE_POLE |
| 3 | FOCUS | Spectral concentration | Controller | ONE_POLE |
| 4 | BRIGHTNESS | Harmonic content | Controller | ONE_POLE |
| 5 | DENSITY | Low-mid presence | Controller | ONE_POLE |
| 6 | RISK | Chaos/instability | Controller | NBLOCK |
| 7 | AGE | CMMM integrator | CMMM | NONE |
| 8 | SPACE | Spatial/reverb | Controller | ONE_POLE |
| 9 | MATERIAL | Surface texture | Controller | NBLOCK |
| 10 | UNITY | Voice coherence | Controller | NBLOCK |
| 11 | VIBRATO | LFO depth scaling | Controller | ONE_POLE |
| 12 | LEGATO | Legato/portamento | Controller | NONE |

### 14.2 Macro → Gesture Routing

```python
def apply_macros_to_gesture(gesture: GestureBasis, macros: MacroRuntime):
    # BRIGHTNESS → tilt
    brightness = macros.get_system_macro(SystemMacro.BRIGHTNESS)
    gesture.tilt_q15 = mul_q15_sat(gesture.tilt_q15, brightness)

    # DENSITY → pressure boost
    density = macros.get_system_macro(SystemMacro.DENSITY)
    gesture.pressure_q15 = sat16(gesture.pressure_q15 + (density >> 2))

    # RISK → chaos
    risk = macros.get_system_macro(SystemMacro.RISK)
    gesture.chaos_q15 = sat16(gesture.chaos_q15 + risk)

    # MOTION → speed boost
    motion = macros.get_system_macro(SystemMacro.MOTION)
    gesture.speed_q15 = sat16(gesture.speed_q15 + (motion >> 2))
```

### 14.3 Macro → Audio Routing (Implemented)

| Macro | Target | Implementation |
|-------|--------|----------------|
| BRIGHTNESS (4) | Filter cutoff | `cutoff *= brightness >> 15` |
| VIBRATO (11) | LFO depth | Scales all LFO outputs |
| DENSITY (5) | Body/thickness | Family weight bias |
| RISK (6) | Tail mode | High risk → MIN/EMERGENCY tail |

---

## 15. Filter System

### 15.1 Filter Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `filter_cutoff_q15` | Q15U | 0-32767 | Cutoff (log: 0=20Hz, 32767=20kHz) |
| `filter_reso_q15` | Q15U | 0-32767 | Resonance (0=flat, 32767=self-osc) |
| `filter_mode` | int | 0-3 | 0=LP, 1=HP, 2=BP, 3=Notch |
| `filter_env_depth_q15` | Q15S | -32768-32767 | Envelope modulation amount |

### 15.2 TPT SVF Filter (Material Engine)

The TPT (Topology-Preserving Transform) SVF filter is implemented in `material_engine.py` as `material_id=9`.

### 15.3 Filter Modulation Data Flow

```
Preset defaults → filter_cutoff_q15
                       ↓
BRIGHTNESS macro → × brightness_q15 >> 15
                       ↓
LFO routing → + scaled_lfo (if LFOTarget.FILTER_CUTOFF)
                       ↓
Matrix envelope → + env_value × env_depth
                       ↓
VoiceFrame.filter_cutoff_q15
                       ↓
SynthMaster (aggregate min cutoff across voices)
                       ↓
MaterialEngine._svf_filter() [material_id=9]
```

---

## 16. VoiceFrame Emission

### 16.1 Emission Function (Mealy Output)

```python
def emit_voiceframe(state: VoiceState, preset: CompiledPreset) -> VoiceFrame:
    frame = VoiceFrame()

    # Identity
    frame.voice_id = state.voice_id

    # Pitch
    frame.f0_hz = midi_to_hz(state.note_u8, state.cents_i16)
    frame.note = state.note_u8
    frame.cents = state.cents_i16

    # Amplitude
    frame.gain_q15 = state.gain_q15
    frame.pan_q15 = state.gesture.pan_q15

    # Tree control
    frame.tree_a_id = state.tree_morph.tree_a_id
    frame.tree_b_id = state.tree_morph.tree_b_id
    frame.tree_blend_q15 = state.tree_morph.blend_q15

    # Oscillator mode from preset
    frame.coeff_osc_mode = preset.osc_mode
    frame.H_request = preset.H_request
    frame.waveform_type = preset.waveform_type

    # Beta blend
    frame.beta_target_q15 = state.beta_target_q15

    # Filter (modulated)
    frame.filter_cutoff_q15 = apply_filter_modulation(state, preset)
    frame.filter_reso_q15 = preset.filter_reso_q15
    frame.filter_mode = preset.filter_mode

    # CMMM events (if any)
    if state.cmmm_mask_change_pending:
        frame.cmmm_node_mask = state.target_node_mask
        frame.cmmm_mask_ramp_blocks = 5
    if state.cmmm_tail_change_pending:
        frame.cmmm_tail_mode = state.target_tail_mode
        frame.cmmm_traction_q15 = state.traction_q15

    # FM6 parameters (if FM mode)
    if preset.osc_mode == 3:
        frame.fm_algorithm_id = preset.fm6.algorithm_id
        frame.fm_op_ratios = preset.fm6.op_ratios
        frame.fm_op_levels = apply_fm_envelopes(state, preset)
        frame.fm_feedback_depth = preset.fm6.feedback_depth

    # Matrix envelope targets
    frame.env_targets_q15 = compute_env_targets(state)

    # Lifecycle flags
    frame.is_starting = state.phase_block == 0 and state.phase == VoicePhase.ATTACK
    frame.is_stopping = state.phase == VoicePhase.RELEASE
    frame.is_active = state.phase != VoicePhase.FREE

    return frame
```

### 16.2 CMMM Event Payloads (WSE-STD-070)

| Payload Type | Name | Fields |
|--------------|------|--------|
| 0x30 | NODEMASK_SET | `node_mask_u32`, `ramp_blocks` |
| 0x31 | TAILMODE_SET | `tail_mode`, `traction_q15` |

---

## 17. Execution Model

### 17.1 Fixed Lanes (WSE-STD-VM-HYBRID)

| Lane | Name | Description |
|------|------|-------------|
| 1 | CARRIER_RENDER | Phase-continuous oscillator |
| 2 | INJECT | Event injection → coeff_state |
| 3 | DIAG | Diagonal operators (K-scaling) |
| 4 | BANDED | Banded operators (K-scaling) |
| 5 | PERMUTE | Permutation operators (L-scaling) |
| 6 | LOWRANK | Low-rank operators (L-scaling) |
| 7 | TAIL | Mandatory tail operators |
| 8 | IDWPT_RENDER | Wavelet synthesis → texture |
| 9 | MIX_GAIN | Beta blend + gain ramp → output |

### 17.2 Per-Block Execution Order (MUST)

```python
def vm_step_block(state: VMState, frame: VoiceFrame) -> np.ndarray:
    # Step 0: Process CMMM events
    _process_cmmm_events(state, frame)

    # Decode frame
    cfg = decode_frame(frame)

    # Lane 1: Carrier
    carrier_buf = lane_carrier_render(state, cfg)

    # Lane 2: Inject
    lane_inject(state, cfg)

    # Lanes 3-6: Backbone with crossfade
    for instr in cfg.backbone_instrs:
        dispatch_coeff_op_crossfade(state, instr)

    # Lane 7: Tail
    lane_tail(state, cfg)

    # Lane 8: IDWPT
    texture_buf = lane_idwpt_render(state, cfg, carrier_buf)

    # Lane 9: Mix + Gain
    output = lane_mix_gain(cfg, carrier_buf, texture_buf)

    return output
```

### 17.3 Crossfade-Aware Dispatch

During mask transitions, operators are applied with both masks and blended:

```python
def dispatch_coeff_op_crossfade(state: VMState, instr: BackboneEntry):
    latch = state.executor_latch

    if not latch.is_ramping():
        dispatch_coeff_op(state, instr, latch.node_mask_u32)
        return

    # Get crossfade weights
    start_w, target_w, start_mask, target_mask = latch.get_crossfade_weights()

    # Apply with start mask
    dispatch_coeff_op(state, instr, start_mask)
    coeff_start = state.coeff_state.copy()

    # Apply with target mask
    dispatch_coeff_op(state, instr, target_mask)

    # Blend
    for i in range(len(state.coeff_state)):
        blended = (coeff_start[i] * start_w + state.coeff_state[i] * target_w) >> 15
        state.coeff_state[i] = clamp(blended, -32768, 32767)
```

---

## 18. Degradation & Budgets

### 18.1 Degradation Levels

```python
class DegradationLevel(IntEnum):
    NONE = 0       # Full processing
    LIGHT = 1      # Cap F_TRANSPORT to 30%
    HEAVY = 2      # Disable F_TRANSPORT, F_TOPOLOGY
    EMERGENCY = 3  # Only F_GLUE (normalize + clamp)
```

### 18.2 Tower Levels

```python
class TowerLevel(IntEnum):
    L0_FULL = 0        # 256 coefficients
    L1_COARSE = 1      # 64 coefficients
    L2_SPARSE = 2      # 16 coefficients
    L3_SKELETON = 3    # 4 coefficients
    L4_SCALAR = 4      # 1 coefficient
```

### 18.3 K Ladder (Active Nodes)

| K | Mask | Usage |
|---|------|-------|
| 32 | 0xFFFFFFFF | Full resolution |
| 24 | 0x00FFFFFF | Light degradation |
| 16 | 0x0000FFFF | Moderate degradation |
| 12 | 0x00000FFF | Heavy degradation |
| 8 | 0x000000FF | Severe degradation |
| 4 | 0x0000000F | Emergency |
| 1 | 0x00000001 | Minimal (carrier only) |

### 18.4 Tail Modes

| Mode | Value | Operators |
|------|-------|-----------|
| FULL | 0 | FineClamp, NormalizeScale, SheafProject |
| MID | 1 | FineClamp, NormalizeScale |
| MIN | 2 | FineClamp |
| EMERGENCY | 3 | FineClamp, EmergencySimplify |

---

## 19. Determinism Guarantees

### 19.1 Requirements (MUST)

1. Same `(VoiceState, GestureBasis, MacroState, BudgetState)` → same VoiceFrame
2. Same `(VMState, VoiceFrame)` → identical output samples
3. PRNG advances ONLY when noise/random op executes
4. Unused event lanes do NOT advance PRNG
5. All rounding uses RNE (round-to-nearest-even)

### 19.2 Snapshotability

All VoiceState fields are serializable for deterministic replay:

```python
def snapshot(self) -> Dict:
    """Create serializable snapshot."""
    return {
        'regime': self.regime.value,
        'phase': self.phase.value,
        'rng_u32': self.rng_u32,
        'block_count': self.block_count,
        'arc_phase_q16': self.arc_phase_q16.copy(),
        'family_weights': self.family_weights.copy(),
        'tree_morph': asdict(self.tree_morph),
        'env_level_q15': self.env_level_q15,
        # ... all state fields
    }

@classmethod
def restore(cls, snapshot: Dict) -> 'VoiceState':
    """Restore from snapshot."""
    state = cls()
    state.regime = Regime(snapshot['regime'])
    state.phase = VoicePhase(snapshot['phase'])
    state.rng_u32 = snapshot['rng_u32']
    # ... restore all fields
    return state
```

---

## 20. Conformance Checklist

### 20.1 Core Structures (v0 Frozen)

- [ ] GestureBasis 8-field structure (§3.2)
- [ ] IntentVector 8-lane structure (§3.3)
- [ ] VoiceState fully snapshotable (§3.1)
- [ ] VoiceFrame all fields populated (§3.4)

### 20.2 Pipeline Processing (v0 Frozen)

- [ ] Gesture → Intent mapping (§6)
- [ ] Regime FSM R0-R6 (§7)
- [ ] EdgeSpec with hysteresis (§7.5-7.6)
- [ ] Family weight computation (§8)
- [ ] Event lanes E0-E7 (§9)
- [ ] Tree morphing (§10)
- [ ] Arc modulators (§11)

### 20.3 Modulation Systems

- [ ] LFO using arc_phase_q16 storage (§12)
- [ ] LFOTarget includes FILTER_CUTOFF, FILTER_RESO
- [ ] ADSR envelope scheduler (§13)
- [ ] Matrix envelope routing (§13.3)
- [ ] System macros 0-12 (§14)
- [ ] BRIGHTNESS → filter cutoff (§14.3)
- [ ] VIBRATO → LFO depth (§14.3)

### 20.4 Filter System

- [ ] TPT SVF filter in Material Engine (§15)
- [ ] Filter parameters in VoiceFrame (§15.1)
- [ ] LFO → filter routing (§15.3)
- [ ] Matrix envelope → filter support (§15.3)

### 20.5 Execution Model

- [ ] 9 fixed lanes per WSE-STD-VM-HYBRID (§17.1)
- [ ] Crossfade-aware dispatch during mask ramp (§17.3)
- [ ] CMMM event processing 0x30/0x31 (§16.2)

### 20.6 Determinism

- [ ] Same inputs → same outputs (§19.1)
- [ ] PRNG deterministic (§19.1)
- [ ] VoiceState snapshotable (§19.2)

---

## Appendix A: Q15 Fixed-Point Conventions

```python
Q15_ONE = 32767   # 1.0 in Q0.15
Q15_HALF = 16384  # 0.5 in Q0.15

def mul_q15_sat(a: int, b: int) -> int:
    """Multiply two Q15 values with saturation."""
    result = (a * b) >> 15
    return max(-32768, min(32767, result))

def sat16(x: int) -> int:
    """Saturate to signed 16-bit range."""
    return max(-32768, min(32767, x))
```

---

## Appendix B: Key File Locations

| Component | File | Notes |
|-----------|------|-------|
| VoiceEngine | `voice/voice_engine.py` | Main CMMM implementation |
| VoiceFrame | `frames.py` | Frame data structures |
| Executor | `vm/executor.py` | Block execution |
| ExecutorLatch | `vm/state.py` | CMMM event state |
| LFOState | `vm/harmonic_bank.py` | LFO implementation |
| EnvelopeScheduler | `envelope_scheduler.py` | ADSR |
| MacroRuntime | `macro_runtime.py` | Macro state |
| MaterialEngine | `safety/material_engine.py` | Filter processing |
| Primitives | `primitives.py` | Enums and constants |

---

## Appendix C: Related Standards

- [WSE-STD-020](../architecture/WSE-STD-020.md) - VoiceFrame ABI
- [WSE-STD-070](../architecture/WSE-STD-070.md) - Tower Architecture, CMMM Events
- [WSE-STD-210](../ccs/WSE-STD-210.md) - CCS Oscillator Modes
- [WSE-STD-VM-HYBRID](WSE-STD-VM-HYBRID.md) - Execution Model
- [WSE-STD-GESTURE-v0](WSE-STD-GESTURE-v0.md) - Gesture & Regime FSM
- [OPERATORS.md](../library/OPERATORS.md) - Operator Reference

---

*End of WSE-STD-VOICE-PIPELINE v0*
