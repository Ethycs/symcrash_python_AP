# WSE-STD-DEGRADE v0 — Deterministic Budgeting + Degradation Ladder (K/Tail/Op Plan)

**Status:** FROZEN (normative)
**Applies to:** Python reference VM + FPGA executor planning logic
**Depends on:** `WSE-STD-020` (VoiceFrame ABI), Operator Library IDs 0..23, Q15 conventions

---

## 0. Purpose

This standard defines a **bit-exact, bounded-work planning layer** that converts a candidate operator plan into a **budget-fitting execution plan** by applying a deterministic degradation ladder.

**Output of this standard (per voice, per block):**

* `backbone_ops[]` (ordered)
* `event_ops[]` (unordered or ordered; executor-defined)
* `node_mask_u32` (active nodes mask)
* `K = popcount(node_mask_u32)`
* `tail_mode` (FULL/MID/MIN/EMERGENCY)
* `pattern_id` (canonical rung ID)
* `is_emergency` (bool)

This is the “degradation tree” contract.

---

## 1. Design Invariants (MUST)

1. **Bounded Work:** For every `(VoiceFrame, voice_state)` there exists a **finite** plan that fits the regime budget (possibly Emergency).
2. **Deterministic:** Same inputs → same plan and same outputs, across runs and platforms.
3. **Monotone Degradation:** Each ladder step strictly reduces predicted cost or reduces allowed feature set; no oscillation.
4. **Tail Safety Dominance:** Tail operators are guaranteed to run in some form; Emergency tail is always available.
5. **K Canonicality:** Active node selection is deterministic and ordered (low-bands-first).

---

## 2. Fixed Constants (v0)

These are frozen for v0 planning semantics:

* `COEFF_LEN = 256`
* `NODE_LEN = 8`
* Total nodes `L = 32`
* Bands `NUM_BANDS = 6`
* `BAND_SLICES = [(0,8),(8,24),(24,56),(56,104),(104,168),(168,256)]`
* Node indexing is canonical low→high frequency: nodes are traversed in band order 0→5, and within a band node_idx increasing.

**Node mask representation (MUST):** `uint32 node_mask`, where bit `i` selects canonical node `i` (0..31).

---

## 3. K Ladder (v0)

K is constrained to the following **rungs**:

`K_RUNG_LIST = [32, 24, 16, 12, 8, 4, 1]`

**Rule (MUST):** Any K reduction MUST move to the next lower rung in this list.

### 3.1 pattern_id mapping (v0 frozen)

`pattern_id` MUST map directly to the rung used:

| pattern_id |  K |
| ---------: | -: |
|          0 | 32 |
|          1 | 24 |
|          2 | 16 |
|          3 | 12 |
|          4 |  8 |
|          5 |  4 |
|          6 |  1 |

---

## 4. Tail Modes (v0)

Tail mode is a small enum:

```
TailMode:
  FULL      # [2, 11, 23]   FineClamp, NormalizeScale, SheafProject
  MID       # [2, 11]       FineClamp, NormalizeScale
  MIN       # [2]           FineClamp only
  EMERGENCY # [2, 19]       FineClamp + EmergencySimplify
```

### 4.1 Tail feasibility rule (MUST)

A regime budget MUST guarantee at least **EMERGENCY tail at K=1** is feasible, or the engine MUST hard-fail configuration at boot (not at runtime).

(Your current code implements `_tail_fits_at_emergency(budget)`; that becomes normative.)

### 4.2 Nonlinear constraint rule (MUST)

If `budget.allow_nonlinear == false`, then:

* `FULL` and `MID` MUST be downgraded to `MIN` **before** running the main loop (or equivalently during Step 0b).

---

## 5. Operator Taxonomy (v0)

Operators are categorized into plan-degradation groups:

* `INJECT`: event/injection operators (e.g., 8, 13, 14, 21)
* `PERMUTE`: topology reorder ops (6, 9, 10, 17)
* `LOWRANK`: global dense-ish ops (7)
* `OTHER BACKBONE`: remaining DIAG/BANDED/PROJ/MIX
* `TAIL`: as defined in TailModes

Each operator has:

* `drop_priority` (0..255) where **higher = drop earlier**
* `nonlinear` flag
* category

**Rule (MUST):** The degradation ladder MUST use `drop_priority` ordering within each drop step.

---

## 6. Cost Model Interface (v0)

A predicted cost model MUST exist with this interface:

* `compute_ticks(K, L)` (integer)
* `uses_nonlinear` aggregation
* optional breakdown fields (ticks_backbone / ticks_events / ticks_tail)

**Rule (MUST):** Planning uses **integer-only** arithmetic. No floats.

---

## 7. Regime Budgets (v0)

A `RegimeBudget` MUST define:

* `max_ticks_per_block`
* `max_active_nodes_K`
* `allow_nonlinear`
* (optional) memory/DSP counters if you later enforce them; v0 requires ticks + K + nonlinear gate.

Budgets may be selected by `(regime_id, voice_class)`.

---

## 8. Deterministic Node Mask Construction (v0)

### 8.1 Canonical “shrink to K” (MUST)

`shrink_mask_to_k(mask, K_target)` MUST select the **lowest-frequency** nodes first.

Normative behavior:

* Enumerate nodes in canonical priority order (band 0→5, node_idx increasing)
* Keep nodes that are already set, but if mask has >K bits, drop the highest-priority-loss nodes first (i.e., retain the earliest nodes)
* If mask has <K bits, fill using the canonical order.

### 8.2 popcount (MUST)

`popcount32()` MUST be deterministic and bit-exact (any standard integer popcount is fine).

---

## 9. The Degradation Ladder Algorithm (normative)

`apply_degradation(backbone_ops, event_ops, node_mask, tail_mode, budget) -> DegradationResult`

### 9.1 Inputs

* `backbone_ops`: ordered list (semantic order preserved)
* `event_ops`: list (ordering not required unless your executor depends on it; if it does, preserve order)
* `node_mask`: uint32
* `tail_mode`: TailMode
* `budget`: RegimeBudget

### 9.2 Pre-steps (MUST)

**Step 0: enforce hard K cap**

* `K = popcount(node_mask)`
* If `K > budget.max_active_nodes_K`: set `K = budget.max_active_nodes_K` and `mask = shrink_mask_to_k(mask, K)`

**Step 0b: nonlinear tail constraint**

* If `budget.allow_nonlinear == false` and `tail in {FULL, MID}` → `tail = MIN`

**Step 0c: emergency feasibility**

* If Emergency tail at K=1 cannot fit → return `DegradationResult(is_emergency=true, backbone=[], events=[], K=1, tail=EMERGENCY, mask=pattern(K=1))` AND set a global diagnostic fault. (This is a configuration error.)

### 9.3 Main loop (MUST)

Iterate up to `max_iterations` (default 128):

1. Predict cost for `(backbone, events, K, tail)`
2. If predicted uses nonlinear ops and `allow_nonlinear==false`:

   * Strip nonlinear ops from backbone/events (except tail ops as defined by `tail_mode`)
   * `steps += 1`, continue
3. If the plan fits budget: **return success** (`is_emergency=false`)
4. Else apply ladder steps in order below; each successful mutation `continue` to next iteration.

### 9.4 Ladder step order (FROZEN v0)

1. **Drop event ops (INJECT)**

   * Remove the single droppable event op with highest `drop_priority`
2. **Drop PERMUTE ops**

   * Remove the single droppable backbone op in PERMUTE category with highest `drop_priority`
3. **Drop LOWRANK ops**

   * Remove the single droppable backbone op in LOWRANK category with highest `drop_priority`
4. **Reduce K**

   * Set `K` to next lower rung in `K_RUNG_LIST`, update `mask = shrink_mask_to_k(mask, K)`
5. **Drop remaining backbone ops by priority**

   * Remove one droppable backbone op (excluding tail ops) with highest `drop_priority`
6. **Tail downgrade**

   * FULL → MID (if nonlinear allowed) else FULL → MIN
   * MID → MIN
   * MIN → EMERGENCY
7. **Emergency mode**

   * `backbone=[]; events=[]; K=1; mask=pattern(K=1); tail=EMERGENCY; break`

After loop or break: return result with `is_emergency=true`.

### 9.5 “Droppable” definition (MUST)

An operator is droppable iff:

* it is not required by current `tail_mode`
* it is not `Identity` if you treat Identity as structural (dropping Identity is a no-op anyway)
* it is present in the relevant list (events or backbone)

---

## 10. Output Result (v0)

`DegradationResult` MUST contain:

* `backbone_ops_final[]`
* `event_ops_final[]`
* `K_final`
* `node_mask_final_u32`
* `tail_mode_final`
* `pattern_id` (per §3.1)
* `steps` (integer count of ladder mutations)
* `ticks` (predicted ticks of final plan)
* `is_emergency` (bool)

---

## 11. Conformance Checklist

An implementation conforms to `WSE-STD-DEGRADE v0` iff it:

* Implements K ladder and `pattern_id` mapping exactly (§3)
* Implements TailModes exactly (§4)
* Enforces nonlinear constraints exactly (§4.2, §9.2 Step 0b)
* Uses integer-only deterministic prediction and popcount (§6, §8)
* Applies the ladder step order exactly (§9.4)
* Preserves backbone operator order (semantic) (§9.1)
* Produces `DegradationResult` fields exactly (§10)

---

## 12. Notes on “Degradation Tree” vs “Ladder”

This standard calls it a “tree” colloquially, but the normative execution is a **linear priority ladder**. Branching happens only because the *presence/absence* of categories changes which step triggers.

That is intentional: it prevents “planner ambiguity” and keeps FPGA behavior straightforward.

---

## 13. v0 Example Outcomes (informative)

* **STABLE_TONE, K=32, FULL tail:** fits without changes (steps=0)
* **TRANSIENT, nonlinear disallowed:** strip nonlinear, reduce K, tail->MIN
* **IDLE/RESIDUAL extreme budgets:** immediate Emergency (K=1, tail=EMERGENCY, no ops)

(Your reported scenarios are consistent with this contract.)

---

If you want, next we can “freeze the sibling standard” that sits above this:

* **WSE-STD-BUDGET v0** (regime table + voice class routing + tick calibration)

…but the degradation tree itself is now locked.
# WSE-STD-DEGRADE v0 — Deterministic Budgeting + Degradation Ladder (K/Tail/Op Plan)

**Status:** FROZEN (normative)
**Applies to:** Python reference VM + FPGA executor planning logic
**Depends on:** `WSE-STD-020` (VoiceFrame ABI), Operator Library IDs 0..23, Q15 conventions

---

## 0. Purpose

This standard defines a **bit-exact, bounded-work planning layer** that converts a candidate operator plan into a **budget-fitting execution plan** by applying a deterministic degradation ladder.

**Output of this standard (per voice, per block):**

* `backbone_ops[]` (ordered)
* `event_ops[]` (unordered or ordered; executor-defined)
* `node_mask_u32` (active nodes mask)
* `K = popcount(node_mask_u32)`
* `tail_mode` (FULL/MID/MIN/EMERGENCY)
* `pattern_id` (canonical rung ID)
* `is_emergency` (bool)

This is the “degradation tree” contract.

---

## 1. Design Invariants (MUST)

1. **Bounded Work:** For every `(VoiceFrame, voice_state)` there exists a **finite** plan that fits the regime budget (possibly Emergency).
2. **Deterministic:** Same inputs → same plan and same outputs, across runs and platforms.
3. **Monotone Degradation:** Each ladder step strictly reduces predicted cost or reduces allowed feature set; no oscillation.
4. **Tail Safety Dominance:** Tail operators are guaranteed to run in some form; Emergency tail is always available.
5. **K Canonicality:** Active node selection is deterministic and ordered (low-bands-first).

---

## 2. Fixed Constants (v0)

These are frozen for v0 planning semantics:

* `COEFF_LEN = 256`
* `NODE_LEN = 8`
* Total nodes `L = 32`
* Bands `NUM_BANDS = 6`
* `BAND_SLICES = [(0,8),(8,24),(24,56),(56,104),(104,168),(168,256)]`
* Node indexing is canonical low→high frequency: nodes are traversed in band order 0→5, and within a band node_idx increasing.

**Node mask representation (MUST):** `uint32 node_mask`, where bit `i` selects canonical node `i` (0..31).

---

## 3. K Ladder (v0)

K is constrained to the following **rungs**:

`K_RUNG_LIST = [32, 24, 16, 12, 8, 4, 1]`

**Rule (MUST):** Any K reduction MUST move to the next lower rung in this list.

### 3.1 pattern_id mapping (v0 frozen)

`pattern_id` MUST map directly to the rung used:

| pattern_id |  K |
| ---------: | -: |
|          0 | 32 |
|          1 | 24 |
|          2 | 16 |
|          3 | 12 |
|          4 |  8 |
|          5 |  4 |
|          6 |  1 |

---

## 4. Tail Modes (v0)

Tail mode is a small enum:

```
TailMode:
  FULL      # [2, 11, 23]   FineClamp, NormalizeScale, SheafProject
  MID       # [2, 11]       FineClamp, NormalizeScale
  MIN       # [2]           FineClamp only
  EMERGENCY # [2, 19]       FineClamp + EmergencySimplify
```

### 4.1 Tail feasibility rule (MUST)

A regime budget MUST guarantee at least **EMERGENCY tail at K=1** is feasible, or the engine MUST hard-fail configuration at boot (not at runtime).

(Your current code implements `_tail_fits_at_emergency(budget)`; that becomes normative.)

### 4.2 Nonlinear constraint rule (MUST)

If `budget.allow_nonlinear == false`, then:

* `FULL` and `MID` MUST be downgraded to `MIN` **before** running the main loop (or equivalently during Step 0b).

---

## 5. Operator Taxonomy (v0)

Operators are categorized into plan-degradation groups:

* `INJECT`: event/injection operators (e.g., 8, 13, 14, 21)
* `PERMUTE`: topology reorder ops (6, 9, 10, 17)
* `LOWRANK`: global dense-ish ops (7)
* `OTHER BACKBONE`: remaining DIAG/BANDED/PROJ/MIX
* `TAIL`: as defined in TailModes

Each operator has:

* `drop_priority` (0..255) where **higher = drop earlier**
* `nonlinear` flag
* category

**Rule (MUST):** The degradation ladder MUST use `drop_priority` ordering within each drop step.

---

## 6. Cost Model Interface (v0)

A predicted cost model MUST exist with this interface:

* `compute_ticks(K, L)` (integer)
* `uses_nonlinear` aggregation
* optional breakdown fields (ticks_backbone / ticks_events / ticks_tail)

**Rule (MUST):** Planning uses **integer-only** arithmetic. No floats.

---

## 7. Regime Budgets (v0)

A `RegimeBudget` MUST define:

* `max_ticks_per_block`
* `max_active_nodes_K`
* `allow_nonlinear`
* (optional) memory/DSP counters if you later enforce them; v0 requires ticks + K + nonlinear gate.

Budgets may be selected by `(regime_id, voice_class)`.

---

## 8. Deterministic Node Mask Construction (v0)

### 8.1 Canonical “shrink to K” (MUST)

`shrink_mask_to_k(mask, K_target)` MUST select the **lowest-frequency** nodes first.

Normative behavior:

* Enumerate nodes in canonical priority order (band 0→5, node_idx increasing)
* Keep nodes that are already set, but if mask has >K bits, drop the highest-priority-loss nodes first (i.e., retain the earliest nodes)
* If mask has <K bits, fill using the canonical order.

### 8.2 popcount (MUST)

`popcount32()` MUST be deterministic and bit-exact (any standard integer popcount is fine).

---

## 9. The Degradation Ladder Algorithm (normative)

`apply_degradation(backbone_ops, event_ops, node_mask, tail_mode, budget) -> DegradationResult`

### 9.1 Inputs

* `backbone_ops`: ordered list (semantic order preserved)
* `event_ops`: list (ordering not required unless your executor depends on it; if it does, preserve order)
* `node_mask`: uint32
* `tail_mode`: TailMode
* `budget`: RegimeBudget

### 9.2 Pre-steps (MUST)

**Step 0: enforce hard K cap**

* `K = popcount(node_mask)`
* If `K > budget.max_active_nodes_K`: set `K = budget.max_active_nodes_K` and `mask = shrink_mask_to_k(mask, K)`

**Step 0b: nonlinear tail constraint**

* If `budget.allow_nonlinear == false` and `tail in {FULL, MID}` → `tail = MIN`

**Step 0c: emergency feasibility**

* If Emergency tail at K=1 cannot fit → return `DegradationResult(is_emergency=true, backbone=[], events=[], K=1, tail=EMERGENCY, mask=pattern(K=1))` AND set a global diagnostic fault. (This is a configuration error.)

### 9.3 Main loop (MUST)

Iterate up to `max_iterations` (default 128):

1. Predict cost for `(backbone, events, K, tail)`
2. If predicted uses nonlinear ops and `allow_nonlinear==false`:

   * Strip nonlinear ops from backbone/events (except tail ops as defined by `tail_mode`)
   * `steps += 1`, continue
3. If the plan fits budget: **return success** (`is_emergency=false`)
4. Else apply ladder steps in order below; each successful mutation `continue` to next iteration.

### 9.4 Ladder step order (FROZEN v0)

1. **Drop event ops (INJECT)**

   * Remove the single droppable event op with highest `drop_priority`
2. **Drop PERMUTE ops**

   * Remove the single droppable backbone op in PERMUTE category with highest `drop_priority`
3. **Drop LOWRANK ops**

   * Remove the single droppable backbone op in LOWRANK category with highest `drop_priority`
4. **Reduce K**

   * Set `K` to next lower rung in `K_RUNG_LIST`, update `mask = shrink_mask_to_k(mask, K)`
5. **Drop remaining backbone ops by priority**

   * Remove one droppable backbone op (excluding tail ops) with highest `drop_priority`
6. **Tail downgrade**

   * FULL → MID (if nonlinear allowed) else FULL → MIN
   * MID → MIN
   * MIN → EMERGENCY
7. **Emergency mode**

   * `backbone=[]; events=[]; K=1; mask=pattern(K=1); tail=EMERGENCY; break`

After loop or break: return result with `is_emergency=true`.

### 9.5 “Droppable” definition (MUST)

An operator is droppable iff:

* it is not required by current `tail_mode`
* it is not `Identity` if you treat Identity as structural (dropping Identity is a no-op anyway)
* it is present in the relevant list (events or backbone)

---

## 10. Output Result (v0)

`DegradationResult` MUST contain:

* `backbone_ops_final[]`
* `event_ops_final[]`
* `K_final`
* `node_mask_final_u32`
* `tail_mode_final`
* `pattern_id` (per §3.1)
* `steps` (integer count of ladder mutations)
* `ticks` (predicted ticks of final plan)
* `is_emergency` (bool)

---

## 11. Conformance Checklist

An implementation conforms to `WSE-STD-DEGRADE v0` iff it:

* Implements K ladder and `pattern_id` mapping exactly (§3)
* Implements TailModes exactly (§4)
* Enforces nonlinear constraints exactly (§4.2, §9.2 Step 0b)
* Uses integer-only deterministic prediction and popcount (§6, §8)
* Applies the ladder step order exactly (§9.4)
* Preserves backbone operator order (semantic) (§9.1)
* Produces `DegradationResult` fields exactly (§10)

---

## 12. Notes on “Degradation Tree” vs “Ladder”

This standard calls it a “tree” colloquially, but the normative execution is a **linear priority ladder**. Branching happens only because the *presence/absence* of categories changes which step triggers.

That is intentional: it prevents “planner ambiguity” and keeps FPGA behavior straightforward.

---

## 13. v0 Example Outcomes (informative)

* **STABLE_TONE, K=32, FULL tail:** fits without changes (steps=0)
* **TRANSIENT, nonlinear disallowed:** strip nonlinear, reduce K, tail->MIN
* **IDLE/RESIDUAL extreme budgets:** immediate Emergency (K=1, tail=EMERGENCY, no ops)

(Your reported scenarios are consistent with this contract.)

---

If you want, next we can “freeze the sibling standard” that sits above this:

* **WSE-STD-BUDGET v0** (regime table + voice class routing + tick calibration)

…but the degradation tree itself is now locked.
# WSE-STD-DEGRADE v0 — Deterministic Budgeting + Degradation Ladder (K/Tail/Op Plan)

**Status:** FROZEN (normative)
**Applies to:** Python reference VM + FPGA executor planning logic
**Depends on:** `WSE-STD-020` (VoiceFrame ABI), Operator Library IDs 0..23, Q15 conventions

---

## 0. Purpose

This standard defines a **bit-exact, bounded-work planning layer** that converts a candidate operator plan into a **budget-fitting execution plan** by applying a deterministic degradation ladder.

**Output of this standard (per voice, per block):**

* `backbone_ops[]` (ordered)
* `event_ops[]` (unordered or ordered; executor-defined)
* `node_mask_u32` (active nodes mask)
* `K = popcount(node_mask_u32)`
* `tail_mode` (FULL/MID/MIN/EMERGENCY)
* `pattern_id` (canonical rung ID)
* `is_emergency` (bool)

This is the “degradation tree” contract.

---

## 1. Design Invariants (MUST)

1. **Bounded Work:** For every `(VoiceFrame, voice_state)` there exists a **finite** plan that fits the regime budget (possibly Emergency).
2. **Deterministic:** Same inputs → same plan and same outputs, across runs and platforms.
3. **Monotone Degradation:** Each ladder step strictly reduces predicted cost or reduces allowed feature set; no oscillation.
4. **Tail Safety Dominance:** Tail operators are guaranteed to run in some form; Emergency tail is always available.
5. **K Canonicality:** Active node selection is deterministic and ordered (low-bands-first).

---

## 2. Fixed Constants (v0)

These are frozen for v0 planning semantics:

* `COEFF_LEN = 256`
* `NODE_LEN = 8`
* Total nodes `L = 32`
* Bands `NUM_BANDS = 6`
* `BAND_SLICES = [(0,8),(8,24),(24,56),(56,104),(104,168),(168,256)]`
* Node indexing is canonical low→high frequency: nodes are traversed in band order 0→5, and within a band node_idx increasing.

**Node mask representation (MUST):** `uint32 node_mask`, where bit `i` selects canonical node `i` (0..31).

---

## 3. K Ladder (v0)

K is constrained to the following **rungs**:

`K_RUNG_LIST = [32, 24, 16, 12, 8, 4, 1]`

**Rule (MUST):** Any K reduction MUST move to the next lower rung in this list.

### 3.1 pattern_id mapping (v0 frozen)

`pattern_id` MUST map directly to the rung used:

| pattern_id |  K |
| ---------: | -: |
|          0 | 32 |
|          1 | 24 |
|          2 | 16 |
|          3 | 12 |
|          4 |  8 |
|          5 |  4 |
|          6 |  1 |

---

## 4. Tail Modes (v0)

Tail mode is a small enum:

```
TailMode:
  FULL      # [2, 11, 23]   FineClamp, NormalizeScale, SheafProject
  MID       # [2, 11]       FineClamp, NormalizeScale
  MIN       # [2]           FineClamp only
  EMERGENCY # [2, 19]       FineClamp + EmergencySimplify
```

### 4.1 Tail feasibility rule (MUST)

A regime budget MUST guarantee at least **EMERGENCY tail at K=1** is feasible, or the engine MUST hard-fail configuration at boot (not at runtime).

(Your current code implements `_tail_fits_at_emergency(budget)`; that becomes normative.)

### 4.2 Nonlinear constraint rule (MUST)

If `budget.allow_nonlinear == false`, then:

* `FULL` and `MID` MUST be downgraded to `MIN` **before** running the main loop (or equivalently during Step 0b).

---

## 5. Operator Taxonomy (v0)

Operators are categorized into plan-degradation groups:

* `INJECT`: event/injection operators (e.g., 8, 13, 14, 21)
* `PERMUTE`: topology reorder ops (6, 9, 10, 17)
* `LOWRANK`: global dense-ish ops (7)
* `OTHER BACKBONE`: remaining DIAG/BANDED/PROJ/MIX
* `TAIL`: as defined in TailModes

Each operator has:

* `drop_priority` (0..255) where **higher = drop earlier**
* `nonlinear` flag
* category

**Rule (MUST):** The degradation ladder MUST use `drop_priority` ordering within each drop step.

---

## 6. Cost Model Interface (v0)

A predicted cost model MUST exist with this interface:

* `compute_ticks(K, L)` (integer)
* `uses_nonlinear` aggregation
* optional breakdown fields (ticks_backbone / ticks_events / ticks_tail)

**Rule (MUST):** Planning uses **integer-only** arithmetic. No floats.

---

## 7. Regime Budgets (v0)

A `RegimeBudget` MUST define:

* `max_ticks_per_block`
* `max_active_nodes_K`
* `allow_nonlinear`
* (optional) memory/DSP counters if you later enforce them; v0 requires ticks + K + nonlinear gate.

Budgets may be selected by `(regime_id, voice_class)`.

---

## 8. Deterministic Node Mask Construction (v0)

### 8.1 Canonical “shrink to K” (MUST)

`shrink_mask_to_k(mask, K_target)` MUST select the **lowest-frequency** nodes first.

Normative behavior:

* Enumerate nodes in canonical priority order (band 0→5, node_idx increasing)
* Keep nodes that are already set, but if mask has >K bits, drop the highest-priority-loss nodes first (i.e., retain the earliest nodes)
* If mask has <K bits, fill using the canonical order.

### 8.2 popcount (MUST)

`popcount32()` MUST be deterministic and bit-exact (any standard integer popcount is fine).

---

## 9. The Degradation Ladder Algorithm (normative)

`apply_degradation(backbone_ops, event_ops, node_mask, tail_mode, budget) -> DegradationResult`

### 9.1 Inputs

* `backbone_ops`: ordered list (semantic order preserved)
* `event_ops`: list (ordering not required unless your executor depends on it; if it does, preserve order)
* `node_mask`: uint32
* `tail_mode`: TailMode
* `budget`: RegimeBudget

### 9.2 Pre-steps (MUST)

**Step 0: enforce hard K cap**

* `K = popcount(node_mask)`
* If `K > budget.max_active_nodes_K`: set `K = budget.max_active_nodes_K` and `mask = shrink_mask_to_k(mask, K)`

**Step 0b: nonlinear tail constraint**

* If `budget.allow_nonlinear == false` and `tail in {FULL, MID}` → `tail = MIN`

**Step 0c: emergency feasibility**

* If Emergency tail at K=1 cannot fit → return `DegradationResult(is_emergency=true, backbone=[], events=[], K=1, tail=EMERGENCY, mask=pattern(K=1))` AND set a global diagnostic fault. (This is a configuration error.)

### 9.3 Main loop (MUST)

Iterate up to `max_iterations` (default 128):

1. Predict cost for `(backbone, events, K, tail)`
2. If predicted uses nonlinear ops and `allow_nonlinear==false`:

   * Strip nonlinear ops from backbone/events (except tail ops as defined by `tail_mode`)
   * `steps += 1`, continue
3. If the plan fits budget: **return success** (`is_emergency=false`)
4. Else apply ladder steps in order below; each successful mutation `continue` to next iteration.

### 9.4 Ladder step order (FROZEN v0)

1. **Drop event ops (INJECT)**

   * Remove the single droppable event op with highest `drop_priority`
2. **Drop PERMUTE ops**

   * Remove the single droppable backbone op in PERMUTE category with highest `drop_priority`
3. **Drop LOWRANK ops**

   * Remove the single droppable backbone op in LOWRANK category with highest `drop_priority`
4. **Reduce K**

   * Set `K` to next lower rung in `K_RUNG_LIST`, update `mask = shrink_mask_to_k(mask, K)`
5. **Drop remaining backbone ops by priority**

   * Remove one droppable backbone op (excluding tail ops) with highest `drop_priority`
6. **Tail downgrade**

   * FULL → MID (if nonlinear allowed) else FULL → MIN
   * MID → MIN
   * MIN → EMERGENCY
7. **Emergency mode**

   * `backbone=[]; events=[]; K=1; mask=pattern(K=1); tail=EMERGENCY; break`

After loop or break: return result with `is_emergency=true`.

### 9.5 “Droppable” definition (MUST)

An operator is droppable iff:

* it is not required by current `tail_mode`
* it is not `Identity` if you treat Identity as structural (dropping Identity is a no-op anyway)
* it is present in the relevant list (events or backbone)

---

## 10. Output Result (v0)

`DegradationResult` MUST contain:

* `backbone_ops_final[]`
* `event_ops_final[]`
* `K_final`
* `node_mask_final_u32`
* `tail_mode_final`
* `pattern_id` (per §3.1)
* `steps` (integer count of ladder mutations)
* `ticks` (predicted ticks of final plan)
* `is_emergency` (bool)

---

## 11. Conformance Checklist

An implementation conforms to `WSE-STD-DEGRADE v0` iff it:

* Implements K ladder and `pattern_id` mapping exactly (§3)
* Implements TailModes exactly (§4)
* Enforces nonlinear constraints exactly (§4.2, §9.2 Step 0b)
* Uses integer-only deterministic prediction and popcount (§6, §8)
* Applies the ladder step order exactly (§9.4)
* Preserves backbone operator order (semantic) (§9.1)
* Produces `DegradationResult` fields exactly (§10)

---

## 12. Notes on “Degradation Tree” vs “Ladder”

This standard calls it a “tree” colloquially, but the normative execution is a **linear priority ladder**. Branching happens only because the *presence/absence* of categories changes which step triggers.

That is intentional: it prevents “planner ambiguity” and keeps FPGA behavior straightforward.

---

## 13. v0 Example Outcomes (informative)

* **STABLE_TONE, K=32, FULL tail:** fits without changes (steps=0)
* **TRANSIENT, nonlinear disallowed:** strip nonlinear, reduce K, tail->MIN
* **IDLE/RESIDUAL extreme budgets:** immediate Emergency (K=1, tail=EMERGENCY, no ops)

(Your reported scenarios are consistent with this contract.)

---

If you want, next we can “freeze the sibling standard” that sits above this:

* **WSE-STD-BUDGET v0** (regime table + voice class routing + tick calibration)

…but the degradation tree itself is now locked.
# WSE-STD-DEGRADE v0 — Deterministic Budgeting + Degradation Ladder (K/Tail/Op Plan)

**Status:** FROZEN (normative)
**Applies to:** Python reference VM + FPGA executor planning logic
**Depends on:** `WSE-STD-020` (VoiceFrame ABI), Operator Library IDs 0..23, Q15 conventions

---

## 0. Purpose

This standard defines a **bit-exact, bounded-work planning layer** that converts a candidate operator plan into a **budget-fitting execution plan** by applying a deterministic degradation ladder.

**Output of this standard (per voice, per block):**

* `backbone_ops[]` (ordered)
* `event_ops[]` (unordered or ordered; executor-defined)
* `node_mask_u32` (active nodes mask)
* `K = popcount(node_mask_u32)`
* `tail_mode` (FULL/MID/MIN/EMERGENCY)
* `pattern_id` (canonical rung ID)
* `is_emergency` (bool)

This is the “degradation tree” contract.

---

## 1. Design Invariants (MUST)

1. **Bounded Work:** For every `(VoiceFrame, voice_state)` there exists a **finite** plan that fits the regime budget (possibly Emergency).
2. **Deterministic:** Same inputs → same plan and same outputs, across runs and platforms.
3. **Monotone Degradation:** Each ladder step strictly reduces predicted cost or reduces allowed feature set; no oscillation.
4. **Tail Safety Dominance:** Tail operators are guaranteed to run in some form; Emergency tail is always available.
5. **K Canonicality:** Active node selection is deterministic and ordered (low-bands-first).

---

## 2. Fixed Constants (v0)

These are frozen for v0 planning semantics:

* `COEFF_LEN = 256`
* `NODE_LEN = 8`
* Total nodes `L = 32`
* Bands `NUM_BANDS = 6`
* `BAND_SLICES = [(0,8),(8,24),(24,56),(56,104),(104,168),(168,256)]`
* Node indexing is canonical low→high frequency: nodes are traversed in band order 0→5, and within a band node_idx increasing.

**Node mask representation (MUST):** `uint32 node_mask`, where bit `i` selects canonical node `i` (0..31).

---

## 3. K Ladder (v0)

K is constrained to the following **rungs**:

`K_RUNG_LIST = [32, 24, 16, 12, 8, 4, 1]`

**Rule (MUST):** Any K reduction MUST move to the next lower rung in this list.

### 3.1 pattern_id mapping (v0 frozen)

`pattern_id` MUST map directly to the rung used:

| pattern_id |  K |
| ---------: | -: |
|          0 | 32 |
|          1 | 24 |
|          2 | 16 |
|          3 | 12 |
|          4 |  8 |
|          5 |  4 |
|          6 |  1 |

---

## 4. Tail Modes (v0)

Tail mode is a small enum:

```
TailMode:
  FULL      # [2, 11, 23]   FineClamp, NormalizeScale, SheafProject
  MID       # [2, 11]       FineClamp, NormalizeScale
  MIN       # [2]           FineClamp only
  EMERGENCY # [2, 19]       FineClamp + EmergencySimplify
```

### 4.1 Tail feasibility rule (MUST)

A regime budget MUST guarantee at least **EMERGENCY tail at K=1** is feasible, or the engine MUST hard-fail configuration at boot (not at runtime).

(Your current code implements `_tail_fits_at_emergency(budget)`; that becomes normative.)

### 4.2 Nonlinear constraint rule (MUST)

If `budget.allow_nonlinear == false`, then:

* `FULL` and `MID` MUST be downgraded to `MIN` **before** running the main loop (or equivalently during Step 0b).

---

## 5. Operator Taxonomy (v0)

Operators are categorized into plan-degradation groups:

* `INJECT`: event/injection operators (e.g., 8, 13, 14, 21)
* `PERMUTE`: topology reorder ops (6, 9, 10, 17)
* `LOWRANK`: global dense-ish ops (7)
* `OTHER BACKBONE`: remaining DIAG/BANDED/PROJ/MIX
* `TAIL`: as defined in TailModes

Each operator has:

* `drop_priority` (0..255) where **higher = drop earlier**
* `nonlinear` flag
* category

**Rule (MUST):** The degradation ladder MUST use `drop_priority` ordering within each drop step.

---

## 6. Cost Model Interface (v0)

A predicted cost model MUST exist with this interface:

* `compute_ticks(K, L)` (integer)
* `uses_nonlinear` aggregation
* optional breakdown fields (ticks_backbone / ticks_events / ticks_tail)

**Rule (MUST):** Planning uses **integer-only** arithmetic. No floats.

---

## 7. Regime Budgets (v0)

A `RegimeBudget` MUST define:

* `max_ticks_per_block`
* `max_active_nodes_K`
* `allow_nonlinear`
* (optional) memory/DSP counters if you later enforce them; v0 requires ticks + K + nonlinear gate.

Budgets may be selected by `(regime_id, voice_class)`.

---

## 8. Deterministic Node Mask Construction (v0)

### 8.1 Canonical “shrink to K” (MUST)

`shrink_mask_to_k(mask, K_target)` MUST select the **lowest-frequency** nodes first.

Normative behavior:

* Enumerate nodes in canonical priority order (band 0→5, node_idx increasing)
* Keep nodes that are already set, but if mask has >K bits, drop the highest-priority-loss nodes first (i.e., retain the earliest nodes)
* If mask has <K bits, fill using the canonical order.

### 8.2 popcount (MUST)

`popcount32()` MUST be deterministic and bit-exact (any standard integer popcount is fine).

---

## 9. The Degradation Ladder Algorithm (normative)

`apply_degradation(backbone_ops, event_ops, node_mask, tail_mode, budget) -> DegradationResult`

### 9.1 Inputs

* `backbone_ops`: ordered list (semantic order preserved)
* `event_ops`: list (ordering not required unless your executor depends on it; if it does, preserve order)
* `node_mask`: uint32
* `tail_mode`: TailMode
* `budget`: RegimeBudget

### 9.2 Pre-steps (MUST)

**Step 0: enforce hard K cap**

* `K = popcount(node_mask)`
* If `K > budget.max_active_nodes_K`: set `K = budget.max_active_nodes_K` and `mask = shrink_mask_to_k(mask, K)`

**Step 0b: nonlinear tail constraint**

* If `budget.allow_nonlinear == false` and `tail in {FULL, MID}` → `tail = MIN`

**Step 0c: emergency feasibility**

* If Emergency tail at K=1 cannot fit → return `DegradationResult(is_emergency=true, backbone=[], events=[], K=1, tail=EMERGENCY, mask=pattern(K=1))` AND set a global diagnostic fault. (This is a configuration error.)

### 9.3 Main loop (MUST)

Iterate up to `max_iterations` (default 128):

1. Predict cost for `(backbone, events, K, tail)`
2. If predicted uses nonlinear ops and `allow_nonlinear==false`:

   * Strip nonlinear ops from backbone/events (except tail ops as defined by `tail_mode`)
   * `steps += 1`, continue
3. If the plan fits budget: **return success** (`is_emergency=false`)
4. Else apply ladder steps in order below; each successful mutation `continue` to next iteration.

### 9.4 Ladder step order (FROZEN v0)

1. **Drop event ops (INJECT)**

   * Remove the single droppable event op with highest `drop_priority`
2. **Drop PERMUTE ops**

   * Remove the single droppable backbone op in PERMUTE category with highest `drop_priority`
3. **Drop LOWRANK ops**

   * Remove the single droppable backbone op in LOWRANK category with highest `drop_priority`
4. **Reduce K**

   * Set `K` to next lower rung in `K_RUNG_LIST`, update `mask = shrink_mask_to_k(mask, K)`
5. **Drop remaining backbone ops by priority**

   * Remove one droppable backbone op (excluding tail ops) with highest `drop_priority`
6. **Tail downgrade**

   * FULL → MID (if nonlinear allowed) else FULL → MIN
   * MID → MIN
   * MIN → EMERGENCY
7. **Emergency mode**

   * `backbone=[]; events=[]; K=1; mask=pattern(K=1); tail=EMERGENCY; break`

After loop or break: return result with `is_emergency=true`.

### 9.5 “Droppable” definition (MUST)

An operator is droppable iff:

* it is not required by current `tail_mode`
* it is not `Identity` if you treat Identity as structural (dropping Identity is a no-op anyway)
* it is present in the relevant list (events or backbone)

---

## 10. Output Result (v0)

`DegradationResult` MUST contain:

* `backbone_ops_final[]`
* `event_ops_final[]`
* `K_final`
* `node_mask_final_u32`
* `tail_mode_final`
* `pattern_id` (per §3.1)
* `steps` (integer count of ladder mutations)
* `ticks` (predicted ticks of final plan)
* `is_emergency` (bool)

---

## 11. Conformance Checklist

An implementation conforms to `WSE-STD-DEGRADE v0` iff it:

* Implements K ladder and `pattern_id` mapping exactly (§3)
* Implements TailModes exactly (§4)
* Enforces nonlinear constraints exactly (§4.2, §9.2 Step 0b)
* Uses integer-only deterministic prediction and popcount (§6, §8)
* Applies the ladder step order exactly (§9.4)
* Preserves backbone operator order (semantic) (§9.1)
* Produces `DegradationResult` fields exactly (§10)

---

## 12. Notes on “Degradation Tree” vs “Ladder”

This standard calls it a “tree” colloquially, but the normative execution is a **linear priority ladder**. Branching happens only because the *presence/absence* of categories changes which step triggers.

That is intentional: it prevents “planner ambiguity” and keeps FPGA behavior straightforward.

---

## 13. v0 Example Outcomes (informative)

* **STABLE_TONE, K=32, FULL tail:** fits without changes (steps=0)
* **TRANSIENT, nonlinear disallowed:** strip nonlinear, reduce K, tail->MIN
* **IDLE/RESIDUAL extreme budgets:** immediate Emergency (K=1, tail=EMERGENCY, no ops)

(Your reported scenarios are consistent with this contract.)

---

If you want, next we can “freeze the sibling standard” that sits above this:

* **WSE-STD-BUDGET v0** (regime table + voice class routing + tick calibration)

…but the degradation tree itself is now locked.
# WSE-STD-DEGRADE v0 — Deterministic Budgeting + Degradation Ladder (K/Tail/Op Plan)

**Status:** FROZEN (normative)
**Applies to:** Python reference VM + FPGA executor planning logic
**Depends on:** `WSE-STD-020` (VoiceFrame ABI), Operator Library IDs 0..23, Q15 conventions

---

## 0. Purpose

This standard defines a **bit-exact, bounded-work planning layer** that converts a candidate operator plan into a **budget-fitting execution plan** by applying a deterministic degradation ladder.

**Output of this standard (per voice, per block):**

* `backbone_ops[]` (ordered)
* `event_ops[]` (unordered or ordered; executor-defined)
* `node_mask_u32` (active nodes mask)
* `K = popcount(node_mask_u32)`
* `tail_mode` (FULL/MID/MIN/EMERGENCY)
* `pattern_id` (canonical rung ID)
* `is_emergency` (bool)

This is the “degradation tree” contract.

---

## 1. Design Invariants (MUST)

1. **Bounded Work:** For every `(VoiceFrame, voice_state)` there exists a **finite** plan that fits the regime budget (possibly Emergency).
2. **Deterministic:** Same inputs → same plan and same outputs, across runs and platforms.
3. **Monotone Degradation:** Each ladder step strictly reduces predicted cost or reduces allowed feature set; no oscillation.
4. **Tail Safety Dominance:** Tail operators are guaranteed to run in some form; Emergency tail is always available.
5. **K Canonicality:** Active node selection is deterministic and ordered (low-bands-first).

---

## 2. Fixed Constants (v0)

These are frozen for v0 planning semantics:

* `COEFF_LEN = 256`
* `NODE_LEN = 8`
* Total nodes `L = 32`
* Bands `NUM_BANDS = 6`
* `BAND_SLICES = [(0,8),(8,24),(24,56),(56,104),(104,168),(168,256)]`
* Node indexing is canonical low→high frequency: nodes are traversed in band order 0→5, and within a band node_idx increasing.

**Node mask representation (MUST):** `uint32 node_mask`, where bit `i` selects canonical node `i` (0..31).

---

## 3. K Ladder (v0)

K is constrained to the following **rungs**:

`K_RUNG_LIST = [32, 24, 16, 12, 8, 4, 1]`

**Rule (MUST):** Any K reduction MUST move to the next lower rung in this list.

### 3.1 pattern_id mapping (v0 frozen)

`pattern_id` MUST map directly to the rung used:

| pattern_id |  K |
| ---------: | -: |
|          0 | 32 |
|          1 | 24 |
|          2 | 16 |
|          3 | 12 |
|          4 |  8 |
|          5 |  4 |
|          6 |  1 |

---

## 4. Tail Modes (v0)

Tail mode is a small enum:

```
TailMode:
  FULL      # [2, 11, 23]   FineClamp, NormalizeScale, SheafProject
  MID       # [2, 11]       FineClamp, NormalizeScale
  MIN       # [2]           FineClamp only
  EMERGENCY # [2, 19]       FineClamp + EmergencySimplify
```

### 4.1 Tail feasibility rule (MUST)

A regime budget MUST guarantee at least **EMERGENCY tail at K=1** is feasible, or the engine MUST hard-fail configuration at boot (not at runtime).

(Your current code implements `_tail_fits_at_emergency(budget)`; that becomes normative.)

### 4.2 Nonlinear constraint rule (MUST)

If `budget.allow_nonlinear == false`, then:

* `FULL` and `MID` MUST be downgraded to `MIN` **before** running the main loop (or equivalently during Step 0b).

---

## 5. Operator Taxonomy (v0)

Operators are categorized into plan-degradation groups:

* `INJECT`: event/injection operators (e.g., 8, 13, 14, 21)
* `PERMUTE`: topology reorder ops (6, 9, 10, 17)
* `LOWRANK`: global dense-ish ops (7)
* `OTHER BACKBONE`: remaining DIAG/BANDED/PROJ/MIX
* `TAIL`: as defined in TailModes

Each operator has:

* `drop_priority` (0..255) where **higher = drop earlier**
* `nonlinear` flag
* category

**Rule (MUST):** The degradation ladder MUST use `drop_priority` ordering within each drop step.

---

## 6. Cost Model Interface (v0)

A predicted cost model MUST exist with this interface:

* `compute_ticks(K, L)` (integer)
* `uses_nonlinear` aggregation
* optional breakdown fields (ticks_backbone / ticks_events / ticks_tail)

**Rule (MUST):** Planning uses **integer-only** arithmetic. No floats.

---

## 7. Regime Budgets (v0)

A `RegimeBudget` MUST define:

* `max_ticks_per_block`
* `max_active_nodes_K`
* `allow_nonlinear`
* (optional) memory/DSP counters if you later enforce them; v0 requires ticks + K + nonlinear gate.

Budgets may be selected by `(regime_id, voice_class)`.

---

## 8. Deterministic Node Mask Construction (v0)

### 8.1 Canonical “shrink to K” (MUST)

`shrink_mask_to_k(mask, K_target)` MUST select the **lowest-frequency** nodes first.

Normative behavior:

* Enumerate nodes in canonical priority order (band 0→5, node_idx increasing)
* Keep nodes that are already set, but if mask has >K bits, drop the highest-priority-loss nodes first (i.e., retain the earliest nodes)
* If mask has <K bits, fill using the canonical order.

### 8.2 popcount (MUST)

`popcount32()` MUST be deterministic and bit-exact (any standard integer popcount is fine).

---

## 9. The Degradation Ladder Algorithm (normative)

`apply_degradation(backbone_ops, event_ops, node_mask, tail_mode, budget) -> DegradationResult`

### 9.1 Inputs

* `backbone_ops`: ordered list (semantic order preserved)
* `event_ops`: list (ordering not required unless your executor depends on it; if it does, preserve order)
* `node_mask`: uint32
* `tail_mode`: TailMode
* `budget`: RegimeBudget

### 9.2 Pre-steps (MUST)

**Step 0: enforce hard K cap**

* `K = popcount(node_mask)`
* If `K > budget.max_active_nodes_K`: set `K = budget.max_active_nodes_K` and `mask = shrink_mask_to_k(mask, K)`

**Step 0b: nonlinear tail constraint**

* If `budget.allow_nonlinear == false` and `tail in {FULL, MID}` → `tail = MIN`

**Step 0c: emergency feasibility**

* If Emergency tail at K=1 cannot fit → return `DegradationResult(is_emergency=true, backbone=[], events=[], K=1, tail=EMERGENCY, mask=pattern(K=1))` AND set a global diagnostic fault. (This is a configuration error.)

### 9.3 Main loop (MUST)

Iterate up to `max_iterations` (default 128):

1. Predict cost for `(backbone, events, K, tail)`
2. If predicted uses nonlinear ops and `allow_nonlinear==false`:

   * Strip nonlinear ops from backbone/events (except tail ops as defined by `tail_mode`)
   * `steps += 1`, continue
3. If the plan fits budget: **return success** (`is_emergency=false`)
4. Else apply ladder steps in order below; each successful mutation `continue` to next iteration.

### 9.4 Ladder step order (FROZEN v0)

1. **Drop event ops (INJECT)**

   * Remove the single droppable event op with highest `drop_priority`
2. **Drop PERMUTE ops**

   * Remove the single droppable backbone op in PERMUTE category with highest `drop_priority`
3. **Drop LOWRANK ops**

   * Remove the single droppable backbone op in LOWRANK category with highest `drop_priority`
4. **Reduce K**

   * Set `K` to next lower rung in `K_RUNG_LIST`, update `mask = shrink_mask_to_k(mask, K)`
5. **Drop remaining backbone ops by priority**

   * Remove one droppable backbone op (excluding tail ops) with highest `drop_priority`
6. **Tail downgrade**

   * FULL → MID (if nonlinear allowed) else FULL → MIN
   * MID → MIN
   * MIN → EMERGENCY
7. **Emergency mode**

   * `backbone=[]; events=[]; K=1; mask=pattern(K=1); tail=EMERGENCY; break`

After loop or break: return result with `is_emergency=true`.

### 9.5 “Droppable” definition (MUST)

An operator is droppable iff:

* it is not required by current `tail_mode`
* it is not `Identity` if you treat Identity as structural (dropping Identity is a no-op anyway)
* it is present in the relevant list (events or backbone)

---

## 10. Output Result (v0)

`DegradationResult` MUST contain:

* `backbone_ops_final[]`
* `event_ops_final[]`
* `K_final`
* `node_mask_final_u32`
* `tail_mode_final`
* `pattern_id` (per §3.1)
* `steps` (integer count of ladder mutations)
* `ticks` (predicted ticks of final plan)
* `is_emergency` (bool)

---

## 11. Conformance Checklist

An implementation conforms to `WSE-STD-DEGRADE v0` iff it:

* Implements K ladder and `pattern_id` mapping exactly (§3)
* Implements TailModes exactly (§4)
* Enforces nonlinear constraints exactly (§4.2, §9.2 Step 0b)
* Uses integer-only deterministic prediction and popcount (§6, §8)
* Applies the ladder step order exactly (§9.4)
* Preserves backbone operator order (semantic) (§9.1)
* Produces `DegradationResult` fields exactly (§10)

---

## 12. Notes on “Degradation Tree” vs “Ladder”

This standard calls it a “tree” colloquially, but the normative execution is a **linear priority ladder**. Branching happens only because the *presence/absence* of categories changes which step triggers.

That is intentional: it prevents “planner ambiguity” and keeps FPGA behavior straightforward.

---

## 13. v0 Example Outcomes (informative)

* **STABLE_TONE, K=32, FULL tail:** fits without changes (steps=0)
* **TRANSIENT, nonlinear disallowed:** strip nonlinear, reduce K, tail->MIN
* **IDLE/RESIDUAL extreme budgets:** immediate Emergency (K=1, tail=EMERGENCY, no ops)

(Your reported scenarios are consistent with this contract.)

---

If you want, next we can “freeze the sibling standard” that sits above this:

* **WSE-STD-BUDGET v0** (regime table + voice class routing + tick calibration)

…but the degradation tree itself is now locked.
