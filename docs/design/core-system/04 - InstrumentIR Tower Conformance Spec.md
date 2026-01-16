# InstrumentIR Tower Conformance Specification

**Version:** 2.0
**Status:** Normative
**Date:** 2026-01-14

---

## 1. Overview

This document specifies how `InstrumentIR` implements the Voice Engine Tower architecture. The IR is designed around the **CMMM (Controlled Semi-Markov Mealy Machine)** - a graph-based state machine that controls voice synthesis.

### 1.1 Design Principles

1. **Graph-Based State Machine**: The core abstraction is a regime graph with typed transitions
2. **Computed Values at Runtime**: The runtime receives computed values (family weights, tree selection), not policy matrices
3. **Simple Emit Templates**: Each regime defines what to emit (families, operators, events)
4. **Budgeted Operators**: Operator execution is bounded by the degradation system

### 1.2 Layer Structure

```
InstrumentIR
├── meta: InstrumentMeta
├── io: IOConfig
├── layers: Layers
│   ├── L0_graph: L0Graph          # Conventional DSP graph
│   ├── L1_lift: L1Lift            # Gesture basis + macro semantics
│   ├── L2_regimes: L2Regimes      # Regime graph (state machine)
│   └── L3_cmmm: L3Cmmm            # Emit templates + transitions
├── compile: CompileConfig
├── envelopes: List[EnvelopeBinding]
└── control: ControlConfig
```

---

## 2. L2 Regime System (The Graph-Based State Machine)

The regime system is the heart of the CMMM. It defines **states** (regimes) and **transitions** (edges) that form a graph-based state machine.

### 2.1 EdgeTransitionType

Transitions between regimes have typed behavior:

```python
class EdgeTransitionType(Enum):
    HARD = "hard"           # Immediate switch, no crossfade
    SOFT = "soft"           # Smooth crossfade over duration_blocks
    CHAOTIC = "chaotic"     # Stochastic switching with jitter
    CORRIDOR_ENTRY = "corridor"  # Entry into degradation corridor
    RECOVERY = "recovery"   # Return from instability to stable
```

### 2.2 EdgeSpec

Defines a transition between two regimes:

```python
@dataclass
class EdgeSpec:
    src_regime: str                    # Source regime ID
    dst_regime: str                    # Destination regime ID
    transition_type: str = "soft"      # EdgeTransitionType value
    guard: str = ""                    # Condition expression
    hysteresis: Optional[Hysteresis]   # Enter/exit thresholds
    min_dwell_blocks: int = 0          # Minimum time before allowed
    duration_blocks: int = 4           # Crossfade duration
    overlay: Optional[str] = None      # Overlay during transition
```

### 2.3 RegimeGraphSpec

The full state machine topology:

```python
@dataclass
class RegimeGraphSpec:
    start_regime: str = "R0_IDLE"
    edges: List[EdgeSpec] = field(default_factory=list)
```

### 2.4 YAML Example

```yaml
regime_graph_spec:
  start_regime: R0_IDLE
  edges:
    - src_regime: R0_IDLE
      dst_regime: R1_ATTACK
      transition_type: hard
      guard: "gate_rise"

    - src_regime: R1_ATTACK
      dst_regime: R2_STABLE
      transition_type: soft
      min_dwell_blocks: 15
      duration_blocks: 8

    - src_regime: R2_STABLE
      dst_regime: R3_GLITCH
      transition_type: chaotic
      guard: "intent.glitch > 0.7"
      hysteresis:
        enter: 0.7
        exit: 0.5

    - src_regime: R3_GLITCH
      dst_regime: R2_STABLE
      transition_type: recovery
      guard: "intent.calm > 0.6"

    - src_regime: any
      dst_regime: R4_RELEASE
      transition_type: soft
      guard: "gate_fall"
      duration_blocks: 4
```

---

## 3. L3 CMMM (Emit Templates)

The CMMM defines **what each regime outputs**. This is done through emit templates, not complex policy matrices.

### 3.1 EmitTemplate

Each regime has an emit template that specifies:

```python
@dataclass
class EmitTemplate:
    families: Dict[str, Union[float, str]]  # Family weight distribution
    ops_enable: List[str]                    # Enabled operators
    trees: TreeTarget                        # Tree blend target
    events: List[EventPolicy]                # Conditional events
```

### 3.2 YAML Example

```yaml
emit_templates:
  T_IDLE:
    families:
      CONSTRAINT: 1.0
      CALM: 1.0
    ops_enable: [NormalizeScale, FineClamp]
    trees: { A: T1, B: T1, beta: 0.0 }

  T_ATTACK:
    families:
      COLOR: 0.6
      EXCITE: 1.0
      CONSTRAINT: 1.0
      CALM: 0.5
    ops_enable: [Tilt, IntraMix, NormalizeScale, FineClamp]
    trees: { A: T1, B: T1, beta: 0.0 }
    events:
      - lane: EVT_OP_PULSE
        condition: "gate_rise"
        strength: "0.8"

  T_STABLE:
    families:
      COLOR: 0.5
      COHERENCE: 0.4
      CONSTRAINT: 1.0
      CALM: 0.7
    ops_enable: [Tilt, LowRankMorph, NormalizeScale, FineClamp, SheafProject]
    trees: { A: T1, B: T2, beta: "position" }

  T_GLITCH:
    families:
      GLITCH: 0.8
      EXCITE: 0.6
      CONSTRAINT: 0.5
    ops_enable: [Tilt, BandShear, QuasiRotate, FineClamp]
    trees: { A: T1, B: T3, beta: "chaos" }
    events:
      - lane: EVT_OP_TOGGLE
        condition: "random(0.3)"
        strength: "0.6"
```

---

## 4. Operator Budgeting (Per CMMM Spec)

The CMMM integrates with the budgeting system per [CMMM-2](../../cmmm-infopacket/CMMM-2.md).

### 4.1 Operator Cost Model

```python
@dataclass
class OperatorCost:
    op_id: int
    ticks_base: int               # Fixed overhead
    ticks_per_active_node: int    # Scales with K (active nodes)
    ticks_per_total_node: int     # Scales with L (total nodes)
    is_tail: bool                 # Must always run if possible
    drop_priority: int            # 0=never drop, 255=drop first
```

### 4.2 Degradation Ladder

When over budget, the system degrades deterministically:

1. **Drop events** (highest drop_priority first)
2. **Drop PERMUTE ops** (global touch, expensive)
3. **Drop LOWRANK ops** (global dense)
4. **Reduce K** (32→24→16→12→8→4→1)
5. **Tail downgrade** (FULL→MID→MIN→EMERGENCY)

### 4.3 Tail Modes

```python
class TailMode(IntEnum):
    FULL = 0        # [FineClamp, NormalizeScale, SheafProject]
    MID = 1         # [FineClamp, NormalizeScale]
    MIN = 2         # [FineClamp]
    EMERGENCY = 3   # [EmergencySimplify]
```

---

## 5. Complete Example

```yaml
instrument_ir_v0:
  meta:
    instrument_id: "expressive-strings"
    name: "Expressive Strings"
    version: 1

  io:
    sample_rate_hz: 48000
    block_size: 64

  layers:
    L0_graph:
      nodes:
        - id: osc1
          type: Osc
          params:
            osc_mode: 1  # Mode B

    L1_lift:
      gesture_basis:
        Gate: { range: [0, 1], source: "note_on_off" }
        Pressure: { range: [0, 1], source: "aftertouch" }
        Position: { range: [0, 1], source: "cc74" }

    L2_regimes:
      regime_nodes:
        - { id: R0_IDLE, kind: Idle, emit_template: T_IDLE }
        - { id: R1_ATTACK, kind: Transient, emit_template: T_ATTACK, dwell_ms: 20 }
        - { id: R2_STABLE, kind: StableTone, emit_template: T_STABLE }
        - { id: R3_GLITCH, kind: GlitchChatter, emit_template: T_GLITCH }

      regime_graph_spec:
        start_regime: R0_IDLE
        edges:
          - { src_regime: R0_IDLE, dst_regime: R1_ATTACK, transition_type: hard, guard: "gate_rise" }
          - { src_regime: R1_ATTACK, dst_regime: R2_STABLE, transition_type: soft, min_dwell_blocks: 12 }
          - { src_regime: R2_STABLE, dst_regime: R3_GLITCH, transition_type: chaotic, guard: "chaos > 0.7" }
          - { src_regime: R3_GLITCH, dst_regime: R2_STABLE, transition_type: recovery }

    L3_cmmm:
      emit_templates:
        T_IDLE:
          families: { CONSTRAINT: 1.0, CALM: 1.0 }
          ops_enable: [NormalizeScale, FineClamp]

        T_ATTACK:
          families: { COLOR: 0.6, EXCITE: 1.0, CONSTRAINT: 1.0 }
          ops_enable: [Tilt, IntraMix, NormalizeScale, FineClamp]
          events:
            - lane: EVT_OP_PULSE
              condition: "gate_rise"
              strength: "0.8"

        T_STABLE:
          families: { COLOR: 0.5, COHERENCE: 0.4, CONSTRAINT: 1.0, CALM: 0.7 }
          ops_enable: [Tilt, LowRankMorph, NormalizeScale, FineClamp, SheafProject]
          trees: { A: T1, B: T2, beta: "position" }

        T_GLITCH:
          families: { GLITCH: 0.8, EXCITE: 0.6, CONSTRAINT: 0.5 }
          ops_enable: [Tilt, BandShear, QuasiRotate, FineClamp]
```

---

## 6. Runtime Flow

```
┌─────────────────┐
│   Gesture       │
│   (Gate, etc.)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Regime Graph   │──▶ Current regime based on guards/transitions
│  (L2_regimes)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Emit Template  │──▶ families, ops_enable, trees, events
│  (L3_cmmm)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Budgeting     │──▶ Degrade if over budget (drop ops, reduce K)
│   System        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   VoiceFrame    │──▶ w_family, tree_a, tree_b, blend, events
│   Emission      │
└─────────────────┘
```

---

## 7. Backward Compatibility

All new fields are optional. Existing presets continue to work:

| If Missing | Auto-Completion |
|------------|-----------------|
| `regime_graph_spec` | Use legacy `regime_nodes` + `transitions` |
| `emit_templates` | Auto-generate from default templates |
| `gesture_basis` | Use standard 6-gesture basis |

---

## 8. CMMM Event Bindings

The L3 CMMM layer emits events that control runtime behavior per [WSE-STD-070 §10](../../standards/architecture/WSE-STD-070.md).

### 8.1 Event Types

| Payload Type | Name | Description |
|--------------|------|-------------|
| `0x30` | NODEMASK_SET | Change active node mask with crossfade ramp |
| `0x31` | TAILMODE_SET | Change tail mode with traction |

### 8.2 Runtime Flow

```
EmitTemplate.cmmm
    ↓ VoiceFrameEmitter
VoiceFrame.cmmm_* fields
    ↓ Executor
ExecutorLatch (node_mask, tail_mode)
    ↓ dispatch_coeff_op_crossfade()
Blended operator output (click-free)
```

### 8.3 Implementation

* **ExecutorLatch** (`vm/state.py`): Holds CMMM event state with crossfade weights
* **Crossfade Dispatch** (`vm/lanes.py`): `dispatch_coeff_op_crossfade()` blends during mask ramp
* **CmmmDefaults** (`library/preset_compiler.py`): Preset-level CMMM configuration

---

## 9. References

- [CMMM-2: Budgeting Spec](../../cmmm-infopacket/CMMM-2.md)
- [Voice Engine Tower](../../../symcrash_docs/03%20-%20Audio%20Pipeline/00%20-%20Voice%20Engine/Evolution/02%20-%20The%20Voice%20Engine%20Tower.md)
- [WSE-STD-070: Presets and Tower Completion](../../standards/architecture/WSE-STD-070.md)
- [WSE-STD-CMMM-ROADMAP: Implementation Status](../../standards/voice/WSE-STD-CMM-ROADMAP-v0.md)
