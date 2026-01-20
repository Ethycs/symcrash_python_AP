# Budgets and Degradation Reference

Reference for the performance budget system and degradation ladder in the symcrash synthesis engine.

## Overview

The budget system ensures presets degrade gracefully under load while maintaining musical integrity. Key concepts:

- **Regime Budgets** - Per-state resource allocations
- **Voice Classes** - MAIN (foreground) vs RESIDUAL (background)
- **Degradation Ladder** - Ordered steps to reduce CPU usage
- **Tail Modes** - Safety operator configurations

---

## Regime IDs

Regimes represent instrument states with different resource requirements.

| ID | Name | Description |
|----|------|-------------|
| 0 | IDLE | Quiet/muted state, minimal processing |
| 1 | TRANSIENT | Attack phase, tone building |
| 2 | STABLE_TONE | Steady state, full musical content |
| 3 | GLITCH_CHATTER | Noisy/error states |
| 4 | RECOVERY | Recovery from overload |
| 100 | RESIDUAL | Cloud/background voices (slots 12+) |
| 200 | CARRIER_ONLY | Emergency fallback (single oscillator) |

---

## Voice Classes

Voices are classified based on slot number:

| Class | Slots | Purpose | Budget |
|-------|-------|---------|--------|
| **MAIN** | 0-11 | Foreground voices | Generous |
| **RESIDUAL** | 12+ | Background/cloud | Tight |

```python
from symcrash_python_ap.budgets import get_voice_class, VoiceClass

voice_class = get_voice_class(voice_id=5)   # → VoiceClass.MAIN
voice_class = get_voice_class(voice_id=15)  # → VoiceClass.RESIDUAL
```

---

## Budget Table

Each regime has a `RegimeBudget` with resource limits:

| Regime | Max Ticks | Max K | Nonlinear? | Tail Reserved |
|--------|-----------|-------|------------|---------------|
| STABLE_TONE | 8000 | 32 | Yes | 600 |
| TRANSIENT | 2000 | 16 | No | 400 |
| GLITCH_CHATTER | 4000 | 24 | Yes | 400 |
| RECOVERY | 1500 | 12 | No | 400 |
| IDLE | 1300 | 8 | No | 1152 (EMERGENCY) |
| RESIDUAL | 1300 | 8 | No | 1152 (EMERGENCY) |

### RegimeBudget Structure

```python
@dataclass
class RegimeBudget:
    max_ticks_per_block: int    # Total resource allocation
    max_active_nodes_K: int     # Maximum active nodes (1-32)
    allow_nonlinear: bool       # Whether sqrt/tanh allowed
    tail_reserved_ticks: int    # Guaranteed for tail operators
```

### Getting a Budget

```python
from symcrash_python_ap.budgets import (
    get_regime_budget, RegimeId, VoiceClass
)

# Get budget for a regime and voice class
budget = get_regime_budget(
    regime_id=RegimeId.STABLE_TONE,
    voice_class=VoiceClass.MAIN
)

print(f"Max ticks: {budget.max_ticks_per_block}")
print(f"Max K: {budget.max_active_nodes_K}")
print(f"Nonlinear: {budget.allow_nonlinear}")
```

---

## Node Patterns (K Ladder)

The coefficient vector has 32 nodes (256 coefficients, 8 per node). K represents active nodes.

### K Ladder (Canonical Steps)

```
K=32 → K=24 → K=16 → K=12 → K=8 → K=4 → K=1
```

| Pattern ID | K | Nodes Active | Character |
|------------|---|--------------|-----------|
| 0 | 32 | All | Full quality |
| 1 | 24 | Low-freq prioritized | Good quality |
| 2 | 16 | Bands 0-3 primarily | Reduced detail |
| 3 | 12 | Core bands | Basic |
| 4 | 8 | Low bands | Minimal |
| 5 | 4 | Fundamental only | Very basic |
| 6 | 1 | Single node | Emergency |

### Node Priority

When reducing K, nodes are kept in band order (low-frequency first):

1. Band 0 (1 node) - Always kept
2. Band 1 (2 nodes) - High priority
3. Band 2 (4 nodes) - Medium priority
4. Band 3 (6 nodes) - Lower priority
5. Band 4 (8 nodes) - Low priority
6. Band 5 (11 nodes) - Lowest priority (fine detail)

This preserves tonal character as processing reduces.

### Mask Functions

```python
from symcrash_python_ap.budgets import (
    build_mask_for_k, shrink_mask_to_k, get_pattern_for_k
)

# Build mask for 16 active nodes
mask = build_mask_for_k(16)  # → 32-bit mask

# Reduce an existing mask
new_mask = shrink_mask_to_k(mask, k_target=8)

# Get canonical pattern
pattern = get_pattern_for_k(16)
print(f"Pattern ID: {pattern.pattern_id}")
print(f"Mask: {pattern.mask:032b}")
```

---

## Degradation Ladder

When a preset doesn't fit within budget, the degradation ladder applies steps in order:

### Degradation Rungs (Priority Order)

1. **Drop INJECT ops** - Events (highest priority to drop)
2. **Drop PERMUTE ops** - Global reordering (expensive)
3. **Drop LOWRANK ops** - Global dense operations
4. **Reduce K** - 32→24→16→12→8→4→1
5. **Drop remaining backbone ops** - By drop_priority
6. **Tail downgrade** - FULL→MID→MIN→EMERGENCY
7. **Emergency mode** - K=1, minimal tail, bare carrier

### DegradationResult

```python
@dataclass
class DegradationResult:
    backbone_ops: List[int]      # Core effect chain (after dropping)
    event_ops: List[int]         # Triggered events (often empty)
    K: int                       # Active nodes after degradation
    node_mask: int               # 32-bit mask of which nodes
    tail_mode: TailMode          # Safety tail level
    pattern_id: int              # Canonical pattern for this K
    degradation_steps: int       # How many rungs applied
    final_ticks: int             # Actual predicted cost
    is_emergency: bool           # True if carrier-only fallback
```

### Using apply_degradation()

```python
from symcrash_python_ap.budgets import (
    apply_degradation, TailMode, RegimeBudget, build_mask_for_k
)

result = apply_degradation(
    backbone_ops=[1, 3, 4, 5, 7],   # Your effect chain
    event_ops=[13, 14],              # Triggered events
    node_mask=build_mask_for_k(32),  # Starting mask
    tail_mode=TailMode.FULL,         # Starting tail
    budget=budget                    # RegimeBudget
)

print(f"Final K: {result.K}")
print(f"Backbone: {result.backbone_ops}")
print(f"Tail mode: {result.tail_mode.name}")
print(f"Ticks: {result.final_ticks}/{budget.max_ticks_per_block}")
print(f"Emergency: {result.is_emergency}")
```

---

## Tail Modes

Tail operators guarantee "instrument-safe" output by clamping, normalizing, and projecting coefficients.

### Tail Mode Levels

| Mode | Operators | Cost | When Used |
|------|-----------|------|-----------|
| **FULL** | FineClamp + NormalizeScale + SheafProject | ~1200 ticks | MAIN voices, normal load |
| **MID** | FineClamp + NormalizeScale | ~600 ticks | MAIN voices, moderate load |
| **MIN** | FineClamp only | ~200 ticks | RESIDUAL voices |
| **EMERGENCY** | FineClamp + EmergencySimplify | ~300 ticks | Fallback |

### Tail Operator IDs

| ID | Name | Purpose |
|----|------|---------|
| 2 | FINE_CLAMP | Clamp coefficients to safe range |
| 11 | NORMALIZE_SCALE | Scale by norm (projection-class) |
| 19 | EMERGENCY_SIMPLIFY | Fallback simplification |
| 23 | SHEAF_PROJECT | Sheaf projection |

### Tail Mode Rules

- **MAIN voices**: Start at FULL, can degrade to MID/MIN
- **RESIDUAL voices**: Start at MIN or EMERGENCY
- **If `allow_nonlinear=False`**: Can't use FULL or MID (they require NormalizeScale)

```python
from symcrash_python_ap.budgets import TailMode, get_tail_operators

# Get operators for a tail mode
ops = get_tail_operators(TailMode.FULL)
# → [2, 11, 23]  (FineClamp, NormalizeScale, SheafProject)

ops = get_tail_operators(TailMode.EMERGENCY)
# → [2, 19]  (FineClamp, EmergencySimplify)
```

---

## Cost Prediction

The cost model predicts operator ticks based on K (active nodes) and L (total nodes).

### Cost Formula

```
ticks = base + (K * ticks_per_active_node) + (L * ticks_per_total_node)
```

Where:
- `K` = active nodes (degradable, 1-32)
- `L` = total nodes (constant = 32)

### Operator Cost Categories

| Category | K-Scaling | L-Scaling | Notes |
|----------|-----------|-----------|-------|
| DIAG | Yes | No | Cheap to budget down |
| BANDED | Yes | No | Cheap to budget down |
| PERMUTE | No | Yes | Global, expensive |
| LOWRANK | No | Yes | Global, expensive |
| PROJ | Partial | Yes | Mostly global |
| INJECT | Yes | No | Sparse writes |

### Predicting Costs

```python
from symcrash_python_ap.budgets import (
    predict_chain_cost, check_budget_fit, PredictedCost
)

# Predict cost for a chain
predicted = predict_chain_cost(
    backbone_ops=[1, 3, 4, 5],
    event_ops=[13],
    K=24,
    tail_mode=TailMode.FULL
)

print(f"Total: {predicted.total_ticks}")
print(f"Backbone: {predicted.backbone_ticks}")
print(f"Events: {predicted.event_ticks}")
print(f"Tail: {predicted.tail_ticks}")
print(f"Uses nonlinear: {predicted.uses_nonlinear}")

# Check if it fits
fits, violations = check_budget_fit(predicted, budget)
if not fits:
    print(f"Over budget: {violations}")
```

### PredictedCost Structure

```python
@dataclass
class PredictedCost:
    total_ticks: int        # Sum of all operations
    backbone_ticks: int     # Core effect chain
    event_ticks: int        # Triggered events
    tail_ticks: int         # Safety tail
    active_nodes_K: int     # K used in prediction
    total_nodes_L: int      # L (always 32)
    uses_nonlinear: bool    # Any sqrt/tanh?
```

---

## Practical Guidelines

### For MAIN Voices (slots 0-11)

- Start with STABLE_TONE budget (8000 ticks, K=32)
- Use FULL tail for maximum quality
- Design with ~4-6 effects in backbone
- 1-2 optional triggered events (often dropped under load)
- Can use nonlinear operations

### For RESIDUAL Voices (slots 12+)

- Use RESIDUAL budget (1300 ticks, K=8)
- Expect MIN or EMERGENCY tail
- Keep to 1-2 simple effects
- Avoid nonlinear operations
- Degrade gracefully when many voices active

### For TRANSIENT/ATTACK

- Budget: 2000 ticks, K=16
- Nonlinear forbidden
- For impact/snappiness - avoid complex ops
- Tail: 400 ticks (not EMERGENCY)

### Load Headroom

- Design for 60-70% of max ticks
- Allows degradation room for CPU spikes
- Leaves tail reservation untouched

### For FM6 Mode (osc_mode=3)

FM6 synthesis has different budget characteristics:

| Component | Cost (ticks) | Notes |
|-----------|--------------|-------|
| FM6 physics kernel | ~800-1200 | Differential equation per operator |
| WPT projection | ~400 | Transform to coefficient space |
| Standard operators | Variable | Same as other modes |
| IDWPT reconstruction | ~400 | Transform back to time domain |

**FM6 Budget Guidelines:**
- MAIN voices: Use STABLE_TONE budget (8000 ticks)
- Algorithm complexity matters: 32 (parallel) is cheaper than 1 (stacked)
- Feedback level increases cost (more iterations)
- Topology morphing adds ~200 ticks for blend computation
- HQ mode (diff-eq kernel) costs ~50% more than LQ mode

**FM6 Degradation Path:**
1. Reduce operator count (disable ops 5, 6 first)
2. Switch to LQ kernel mode
3. Disable topology morphing
4. Standard K-ladder degradation
5. Emergency: Single carrier oscillator

---

## Authoring Workflow

### Step 1: Understand Your Regime

```python
from symcrash_python_ap.budgets import (
    get_regime_budget, RegimeId, VoiceClass
)

budget = get_regime_budget(RegimeId.STABLE_TONE, VoiceClass.MAIN)
print(f"Budget: {budget.max_ticks_per_block} ticks")
print(f"Max K: {budget.max_active_nodes_K}")
```

### Step 2: Design Your Operator Chain

```python
backbone_ops = [1, 3, 4, 7]  # Tilt, IntraMix, DiffuseUp, LowRankMorph
event_ops = [13]              # Impulse on attack
K = 24
tail_mode = TailMode.FULL
```

### Step 3: Predict and Check

```python
from symcrash_python_ap.budgets import predict_chain_cost, check_budget_fit

predicted = predict_chain_cost(backbone_ops, event_ops, K, tail_mode)
fits, violations = check_budget_fit(predicted, budget)

if not fits:
    print(f"Need to degrade: {violations}")
```

### Step 4: Apply Degradation

```python
from symcrash_python_ap.budgets import apply_degradation

result = apply_degradation(
    backbone_ops, event_ops,
    build_mask_for_k(K),
    tail_mode,
    budget
)

print(f"Degraded K: {result.K}")
print(f"Degraded ops: {result.backbone_ops}")
print(f"Final ticks: {result.final_ticks}")
```

---

## Key Functions Reference

### Budget Functions

```python
get_regime_budget(regime_id, voice_class) -> RegimeBudget
get_voice_class(voice_id) -> VoiceClass
budget_allows_operator(budget, op_cost, uses_nonlinear) -> bool
```

### Degradation Functions

```python
apply_degradation(backbone_ops, event_ops, node_mask, tail_mode, budget) -> DegradationResult
degrade_to_fit(ops, K, budget, tail_mode) -> DegradationResult
simulate_degradation(backbone_ops, event_ops, K, tail_mode, budget) -> DegradationResult
```

### Cost Prediction Functions

```python
predict_chain_cost(backbone_ops, event_ops, K, tail_mode) -> PredictedCost
check_budget_fit(predicted, budget) -> Tuple[bool, List[str]]
predict_operator_ticks(op_id, K) -> Tuple[int, bool]
estimate_max_backbone_ticks(budget, K, tail_mode) -> int
can_add_operator(current_ticks, op_id, K, budget, tail_mode) -> bool
```

### Pattern Functions

```python
get_pattern_for_k(k) -> PatternEntry
build_mask_for_k(k) -> int
shrink_mask_to_k(mask, k_target) -> int
node_mask_to_coeff_mask(node_mask) -> List[bool]
popcount32(mask) -> int
```

### Tail Functions

```python
get_tail_operators(tail_mode) -> List[int]
get_tail_cost(tail_mode) -> int
```

---

## CMMM Runtime Integration (WSE-STD-070)

The budget system integrates with the CMMM runtime via event payloads that control node masks and tail modes dynamically.

### Node Mask Events (0x30 NODEMASK_SET)

CMMM can change K at runtime using the `0x30` event payload:

```python
# VoiceFrame fields for node mask control
frame.cmmm_node_mask = 0x0000FFFF    # Target mask (16 nodes)
frame.cmmm_mask_ramp_blocks = 5      # Crossfade duration
```

The executor latch receives this and crossfades between masks over the specified blocks to prevent clicks.

### Tail Mode Events (0x31 TAILMODE_SET)

CMMM can change tail mode at runtime:

```python
# VoiceFrame fields for tail mode control
frame.cmmm_tail_mode = TailMode.MIN  # New tail mode
frame.cmmm_traction_q15 = 16384      # Projection hardness (0.5)
```

### Crossfade-Aware Operator Dispatch

During mask transitions, operators are applied using `dispatch_coeff_op_crossfade()`:

1. Save coefficient state
2. Apply operator with **start** mask
3. Restore and apply with **target** mask
4. Blend outputs using crossfade weights

This ensures K changes are click-free even during complex operator chains.

### ExecutorLatch State

The per-voice `ExecutorLatch` tracks:

```python
@dataclass
class ExecutorLatch:
    node_mask_u32: int = 0xFFFFFFFF    # Current mask
    mask_target_u32: int = 0xFFFFFFFF  # Target for ramping
    mask_ramp_remaining: int = 0        # Blocks left in ramp
    tail_mode: int = 0                  # TailMode enum
    traction_q15: int = 32767           # Projection hardness
```

### Integration with Pattern Table

The pattern table provides canonical masks for the K ladder:

```python
from symcrash_python_ap.budgets import get_pattern_for_k, PATTERN_TABLE

# Get mask for K=16
pattern = get_pattern_for_k(16)
frame.cmmm_node_mask = pattern.leaf_mask_u32
frame.cmmm_mask_ramp_blocks = 5
```

Pattern IDs 0-6 correspond to the K ladder rungs. Pattern IDs 7+ are extended patterns for artistic control.

---

## Related Documentation

- [PRESET-AUTHORING-GUIDE.md](PRESET-AUTHORING-GUIDE.md) - Main authoring guide
- [OPERATORS.md](OPERATORS.md) - Operator reference
- [MACROS-AND-CONTROLS.md](MACROS-AND-CONTROLS.md) - Control system
- [../architecture/WSE-STD-070.md](../architecture/WSE-STD-070.md) - Tower completion and CMMM events
