# CMMM Specification (Consolidated)

> Controlled Semi-Markov Mealy Machine for Deterministic Budget-Bounded Synthesis

**Status**: Partially Implemented
**Version**: v0.2 (2026-01-21)

---

## 1. Overview

The CMMM is a per-voice, per-block state machine that:
1. Ingests control deltas and note lifecycle events
2. Updates regime state
3. Chooses a bounded execution plan (operators + K + tail + events)
4. Emits VoiceFrame with CMMM control fields

### 1.1 Design Principles

- **Bit-exact**: Pure integer ops (Q15), deterministic completion
- **FPGA-native**: No dynamic allocation, fixed tables, bounded work
- **Click-free by construction**: Scalar ramps + structural crossfades

### 1.2 Implementation Status

| Component | Location | Status |
|-----------|----------|--------|
| ExecutorLatch | `vm/state.py:44-151` | ✅ Complete |
| Event Processing | `vm/executor.py:43-70` | ✅ Complete |
| Crossfade Dispatch | `vm/lanes.py` | ✅ Complete |
| Operator Costs | `tables/operator_costs.py` | ✅ Complete |
| Tail Modes | `tables/tail_modes.py` | ✅ Complete |
| Pattern Table | `tables/pattern_table.py` | ✅ Complete |
| RegimeBudget | — | ❌ Not implemented |
| apply_degradation() | — | ❌ Not implemented |
| Budget Predictor | — | ❌ Not implemented |

---

## 2. Node Model

### 2.1 Constants

```python
COEFF_LEN = 256      # Total coefficient vector length
NODE_LEN = 8         # Coefficients per node
L_TOTAL = 32         # Total nodes (COEFF_LEN / NODE_LEN)
```

### 2.2 Active Node Mask

- Format: `uint32` bitmask (bit n = 1 means node n is active)
- K = `popcount(mask)` = active node count
- Stored in `ExecutorLatch.node_mask_u32`

### 2.3 Band Layout

Nodes are grouped by band, low-frequency first (for degradation priority):

| Band | Nodes | Count | Description |
|------|-------|-------|-------------|
| 0 | 0 | 1 | Fundamental |
| 1 | 1-2 | 2 | Body |
| 2 | 3-6 | 4 | Mid |
| 3 | 7-12 | 6 | Presence |
| 4 | 13-20 | 8 | Brilliance |
| 5 | 21-31 | 11 | Air |

**Total**: 32 nodes

---

## 3. Operator Cost Model

**Location**: `src/symcrash_python_ap/tables/operator_costs.py`

### 3.1 Cost Formula

```
ticks = base + (ticks_per_active_node × K) + (ticks_per_total_node × L)
```

Where:
- **K** = active nodes (what you budget down)
- **L** = total nodes (constant 32)
- **1 tick** = 1 Q15 multiply-add on 1 coefficient

### 3.2 Operator Categories

**K-scaling ops** (cheap to budget down):
- DIAG: 0, 1, 15, 16, 20, 22
- BANDED: 3, 4, 5

**L-scaling ops** (global touch, expensive):
- PERMUTE: 6, 9, 10, 17
- LOWRANK: 7
- PROJ: 2, 11, 12, 19, 23

**Event ops** (drop first in degradation):
- INJECT: 8, 13, 14, 21

### 3.3 Cost Table

```python
@dataclass(frozen=True)
class OperatorCost:
    op_id: int
    ticks_base: int
    ticks_per_active_node: int  # scales with K
    ticks_per_total_node: int   # scales with L
    nonlinear: bool
    is_tail: bool
    drop_priority: int          # 0=never drop, 255=drop first
```

| ID | Name | Base | /K | /L | Nonlinear | Tail | Drop |
|----|------|------|----|----|-----------|------|------|
| 0 | Identity | 0 | 0 | 0 | - | - | 0 |
| 1 | Tilt | 8 | 1 | 0 | - | - | 80 |
| 2 | FineClamp | 64 | 0 | 2 | - | ✓ | 0 |
| 3 | IntraMix | 16 | 3 | 0 | - | - | 110 |
| 4 | DiffuseUp | 16 | 2 | 0 | - | - | 120 |
| 5 | DiffuseDown | 16 | 2 | 0 | - | - | 120 |
| 6 | BandShear | 32 | 0 | 1 | - | - | 200 |
| 7 | LowRankMorph | 64 | 0 | 4 | - | - | 230 |
| 8 | GlobalPulse | 16 | 1 | 0 | - | - | 240 |
| 9 | SwapBands | 32 | 0 | 1 | - | - | 220 |
| 10 | RotateNodes | 32 | 0 | 1 | - | - | 190 |
| 11 | NormalizeScale | 96 | 0 | 3 | ✓ | ✓ | 0 |
| 12 | GluePair | 48 | 0 | 1 | - | - | 60 |
| 13 | Impulse | 8 | 1 | 0 | - | - | 245 |
| 14 | NoiseBand | 16 | 1 | 0 | - | - | 250 |
| 15 | PressureRamp | 8 | 1 | 0 | - | - | 160 |
| 16 | GateMask | 8 | 1 | 0 | - | - | 40 |
| 17 | QuasiRotate | 32 | 0 | 1 | - | - | 210 |
| 18 | MorphMix | 16 | 0 | 0 | - | - | 140 |
| 19 | EmergencySimplify | 64 | 0 | 2 | - | ✓ | 0 |
| 20 | FoldPressure | 8 | 1 | 0 | - | - | 160 |
| 21 | CuspInjection | 16 | 1 | 0 | - | - | 248 |
| 22 | SwallowtailMask | 8 | 1 | 0 | - | - | 180 |
| 23 | SheafProject | 96 | 0 | 3 | - | ✓ | 10 |

### 3.4 Prediction Function

```python
def predict_ticks(op_ids: List[int], K: int, L: int = 32) -> Tuple[int, bool]:
    ticks = 0
    uses_nonlinear = False
    for op_id in op_ids:
        cost = OPERATOR_COSTS[op_id]
        ticks += (cost.ticks_base +
                  cost.ticks_per_active_node * K +
                  cost.ticks_per_total_node * L)
        uses_nonlinear |= cost.nonlinear
    return ticks, uses_nonlinear
```

---

## 4. Tail Modes

**Location**: `src/symcrash_python_ap/tables/tail_modes.py`

Tail operators guarantee well-formed output:

| Mode | Value | Operators | Description |
|------|-------|-----------|-------------|
| FULL | 0 | [2, 11, 23] | FineClamp → NormalizeScale → SheafProject |
| MID | 1 | [2, 11] | FineClamp → NormalizeScale |
| MIN | 2 | [2] | FineClamp only |
| EMERGENCY | 3 | [19] | EmergencySimplify (carrier-only) |

### 4.1 Voice Class Assignment

- **MAIN voices**: Start at FULL, degrade as needed
- **RESIDUAL voices**: Start at MIN or EMERGENCY

### 4.2 Nonlinear Constraint

If `allow_nonlinear=False`, skip modes containing op 11 (NormalizeScale uses sqrt).

---

## 5. Pattern Table & K Staircase

**Location**: `src/symcrash_python_ap/tables/pattern_table.py`

### 5.1 Degradation Staircase

```
K: 32 → 24 → 16 → 12 → 8 → 4 → 1
```

Node selection: Keep nodes in band order 0→5, within band keep lower node_idx first.

### 5.2 Pattern Table

| Pattern ID | Name | K | Mask |
|------------|------|---|------|
| 0 | FULL | 32 | 0xFFFFFFFF |
| 1 | K24 | 24 | first_k_nodes(24) |
| 2 | K16 | 16 | first_k_nodes(16) |
| 3 | K12 | 12 | first_k_nodes(12) |
| 4 | K8 | 8 | first_k_nodes(8) |
| 5 | K4 | 4 | first_k_nodes(4) |
| 6 | K1 | 1 | first_k_nodes(1) |
| 10 | BANDS_0_1_2 | 7 | bands_to_mask([0,1,2]) |
| 11 | BANDS_0_1 | 3 | bands_to_mask([0,1]) |
| 12 | BAND_0 | 1 | band_to_mask(0) |

---

## 6. Degradation Ladder (Design)

**Status**: ❌ Not yet implemented as a function

### 6.1 Degradation Order

When over budget, apply in this order:

1. **Drop INJECT ops** (highest drop_priority first): {8, 13, 14, 21}
2. **Drop PERMUTE ops**: {6, 9, 10, 17}
3. **Drop LOWRANK ops**: {7}
4. **Reduce K**: 32→24→16→12→8→4→1
5. **Disable nonlinear ops** (if not allowed)
6. **Drop backbone ornaments** (by drop_priority, never tail)
7. **Tail downgrade**: FULL→MID→MIN→EMERGENCY
8. **Emergency**: backbone cleared, K=1, tail=EMERGENCY

### 6.2 Algorithm Sketch

```python
def apply_degradation(backbone_ops, event_ops, node_mask, tail_mode, budget):
    K = min(popcount32(node_mask), budget.max_active_nodes_K)

    while total_cost(backbone_ops, event_ops, K, tail_mode) > budget.max_ticks:
        # Apply ladder steps in order...
        pass

    return (backbone_ops, event_ops, K, node_mask, tail_mode)
```

---

## 7. ExecutorLatch (Runtime State)

**Location**: `src/symcrash_python_ap/vm/state.py:44-151`

### 7.1 Fields

```python
@dataclass
class ExecutorLatch:
    # Node mask control (payload_type 0x30)
    node_mask_u32: int = 0xFFFFFFFF
    mask_target_u32: int = 0xFFFFFFFF
    mask_ramp_remaining: int = 0

    # Tail mode control (payload_type 0x31)
    tail_mode: int = 0              # TailMode enum
    traction_q15: int = Q15_ONE     # Projection hardness

    # Ramp tracking
    _mask_ramp_total: int = 0
    _mask_start_u32: int = 0xFFFFFFFF
```

### 7.2 Methods

| Method | Description |
|--------|-------------|
| `apply_nodemask_event(mask, ramp_blocks)` | Start mask transition |
| `apply_tailmode_event(mode, traction)` | Set tail mode |
| `step_mask_ramp()` | Advance ramp by one block |
| `get_effective_k()` | Return popcount of current mask |
| `get_crossfade_weights()` | Return (start_weight, target_weight, start_mask, target_mask) |
| `is_ramping()` | Check if in transition |

---

## 8. Event Bindings

### 8.1 NODEMASK_SET (payload_type 0x30)

Sets active node mask with crossfade transition.

| VoiceFrame Field | Type | Description |
|------------------|------|-------------|
| `cmmm_node_mask` | u32 | Target node mask |
| `cmmm_mask_ramp_blocks` | u8 | Crossfade duration (default 5) |

### 8.2 TAILMODE_SET (payload_type 0x31)

Sets tail mode policy with traction scaling.

| VoiceFrame Field | Type | Description |
|------------------|------|-------------|
| `cmmm_tail_mode` | u8 | TailMode enum (0-3) |
| `cmmm_traction_q15` | i16 | Projection hardness (0-32767) |

### 8.3 Event Processing

**Location**: `src/symcrash_python_ap/vm/executor.py:43-70`

```python
def _process_cmmm_events(state: VMState, frame: VoiceFrame) -> None:
    latch = state.executor_latch

    # Step 1: Advance mask ramp (every block)
    latch.step_mask_ramp()

    # Step 2: Process node mask event
    if frame.cmmm_node_mask is not None:
        latch.apply_nodemask_event(frame.cmmm_node_mask, frame.cmmm_mask_ramp_blocks)

    # Step 3: Process tail mode event
    if frame.cmmm_tail_mode is not None:
        latch.apply_tailmode_event(frame.cmmm_tail_mode, frame.cmmm_traction_q15)
```

---

## 9. Crossfade Implementation

**Location**: `src/symcrash_python_ap/vm/lanes.py`

### 9.1 Crossfade Weights

```python
def get_crossfade_weights(ramp_remaining, ramp_total):
    if ramp_remaining <= 0:
        return (0, Q15_ONE, 0, current_mask)

    progress = (ramp_total - ramp_remaining) / ramp_total
    target_weight = int(progress * Q15_ONE)
    start_weight = Q15_ONE - target_weight

    return (start_weight, target_weight, start_mask, target_mask)
```

### 9.2 Crossfade Dispatch

`dispatch_coeff_op_crossfade()` blends operator outputs during mask transitions:
- Applies operator with start mask at start_weight
- Applies operator with target mask at target_weight
- Blends results

---

## 10. Voice Scaling Implications

**128 voices is compatible with**:
- K-scaling ops (DIAG/BANDED) — reduce K per voice

**128 voices is NOT compatible with L-scaling ops unless**:
- Restricted to MAIN voices only
- Run at lower update rate
- Use carrier-only policy for residual voices

---

## 11. Test Coverage

| Test File | Purpose |
|-----------|---------|
| `test_crossfade_dispatch.py` | ExecutorLatch weights and ramp behavior |
| `test_cmmm_events.py` | Event roundtrip and K computation |
| `test_cmmm_template_emission.py` | CMMM template emission |
| `test_golden_voiceframe_stream.py` | Deterministic replay verification |

---

## 12. Future Work

### 12.1 Not Yet Implemented

1. **RegimeBudget dataclass** — Per-regime tick limits
2. **apply_degradation() function** — Full ladder implementation
3. **Budget predictor module** — Cost verification before execution
4. **Per-regime operator gating** — IDLE/TRANSIENT/STABLE_TONE budgets

### 12.2 Original Design (CMMM-1) Not Adopted

The original CMMM-1 proposed a more complex cost model with:
- cycles_per_leaf + bram_rd + bram_wr + dsp_mult dimensions
- Separate tail_reserved_{cycles,bram,dsp} budgets

This was simplified to single-dimension ticks in CMMM-2 to avoid premature optimization before FPGA calibration.

---

## Appendix A: Document History

This specification consolidates three design documents:

| Document | Content | Status |
|----------|---------|--------|
| CMMM-1.md | Initial multi-dimensional cost model, regime budgets | Partially superseded |
| CMMM-2.md | K/L scaling refinement, degradation ladder, apply_degradation() | Core concepts adopted |
| CMMM-addenda.md | Integration memo: event bindings, crossfade | Fully implemented |

The original files are retained in `docs/design/cmmm-infopacket/` for historical reference.
