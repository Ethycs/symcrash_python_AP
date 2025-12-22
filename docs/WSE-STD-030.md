## WSE-STD-030 — Executor Scheduling, Tail Ordering, and Budget Accounting (v0)

**Status:** FREEZE v0
**Scope:** Symbol Engine execution order + tail splice semantics + compute-budget hooks and deterministic degradation behavior *as observed by the executor*.
**Non-scope:** UI intent semantics, voice allocation policy, operator math internals (those live in WSE-STD-001 and operator specs).

### 0. Normative keywords

The key words **MUST**, **MUST NOT**, **SHOULD**, **MAY** are normative.

---

# 1) Concepts and objects

### 1.1 Block time

All execution happens at **BlockTime**. One “tick” = one audio block (e.g., 64 samples).

### 1.2 Inputs

Per active voice per block, the executor receives a **VoiceFrame** containing:

* `backbone[]` : ordered list of `(op_id, strength, params...)`
* `event_mask` + `event_strength[lanes]`
* `event_payload` (v0a: common payload) **or** `event_payload[lanes]` (v0b)
* `traction` / clamp hints (used by tail and/or Q-family ops)

*(Exact ABI lives in WSE-STD-020; STD-030 defines how it is used.)*

### 1.3 Symbol state

The executor maintains a per-voice **SymbolState**:

* `c` : coefficient state vector (e.g., 256 int16)
* `op_enable_mask` : persistent per-op toggles (if supported by event lane)
* `internal` : per-op scratch, per-voice PRNG state handle (per WSE-STD-040)

**Time awareness:** the executor is **time-blind**: it does not maintain musical counters. It only processes the current block’s frame + current symbol state.

---

# 2) Execution pipeline (normative order)

For each voice `v` serviced on block `t`, the executor MUST execute in exactly this order:

1. **Load/prepare**
2. **Backbone stage** (ordered list; no reordering)
3. **Event stage** (lane splice point)
4. **Mandatory Tail stage** (always last; cannot be dropped)
5. **Commit** (write back coefficient state, emit output coeffs to wavelet engine)

This order is **normative** and MUST be identical across Python reference and FPGA implementation.

---

## 2.1 Prepare stage

The executor MUST:

* Obtain `c_in := SymbolState[v].c`
* Obtain `enable_mask := SymbolState[v].op_enable_mask` (if used)
* Initialize a local working buffer `c_work := c_in`

The executor MAY precompute any indexing metadata (leaf masks, pattern resolution) but MUST NOT reorder operators.

---

## 2.2 Backbone stage (author order)

Let `backbone = [(op0, s0, p0), (op1, s1, p1), ...]`.

**Rule B1 (author order):** Backbone operators MUST be applied **in the given list order**.
**Rule B2 (no executor sorting):** The executor MUST NOT sort by `op_id`, cost, family, or any other property.
**Rule B3 (toggle interaction):** If `op_enable_mask` is supported, then an op marked “disabled” MUST be skipped as if it were not present. This skip MUST NOT advance PRNG draw order for that op.

Backbone operators are applied as:

* `c_work ← APPLY(op_i, c_work, s_i, p_i, context)`

where `APPLY()` semantics are defined per operator. (Operator math is out of scope here.)

---

## 2.3 Event stage (lane splice)

Events are applied after all backbone ops and before tail.

### 2.3.1 Lane processing order

**Rule E1 (lane order):** Event lanes MUST be processed in strictly increasing lane index order: `0..L-1`.

### 2.3.2 Event activation

A lane `ℓ` is active for this block iff `(event_mask >> ℓ) & 1 == 1`.

If active, its event strength is `e = event_strength[ℓ]` (Q-format per WSE-STD-020).

### 2.3.3 Payload resolution (v0a vs v0b)

* **v0a (COMMON payload):** all active lanes use `payload_common`. Per-lane targeting is not available.
* **v0b (PER_LANE payload):** lane ℓ uses `payload[ℓ]`. If `payload[ℓ]` is absent/zero and the format supports “defaulting”, the implementation MAY fall back to `payload_common`. If you implement fallback, it MUST be deterministic.

### 2.3.4 Event application

Each lane maps to exactly one “event operator” or “event action” (some lanes may mutate executor state rather than applying a library op).

**Rule E2 (event op isolation):** Event lanes MUST NOT retroactively change the already-run backbone stage for this block. (They may toggle masks for future blocks.)

Typical event actions (examples; final lane map is WSE-STD-020):

* Toggle `op_enable_mask` bits
* Apply an injection operator using `payload` targeting (impulse/noise/pattern)
* Apply a short permutation burst (rotate/swap on targeted bands)
* Apply a “material tag” forwarded downstream (if your architecture forwards lanes)

**Rule E3 (persistent toggles):** If a lane toggles `op_enable_mask`, the mutation MUST persist in SymbolState until toggled again or explicitly reset by a well-defined reset condition (reset conditions MUST be specified in WSE-STD-020 or WSE-STD-050). STD-030 requires that **some** reset rule exists; it does not choose it.

---

# 3) Mandatory Tail stage (always last, never dropped)

The tail stage is a fixed sequence of **tail ops** that MUST run after backbone + events, and MUST run even under compute pressure.

## 3.1 Tail reservation

**Rule T0 (reserved overhead):** Tail ops are “mandatory overhead and reserved outside budget.”

* Budget accounting MUST subtract/allocate tail cost first (or treat tail as non-budgeted).
* Budget dropping logic MUST NOT drop tail ops.

## 3.2 Tail sequence

Tail MUST run in the following conceptual order:

1. **Projection / clamp / legality** (`Q-family`)
2. **Emergency simplify (optional but deterministic trigger)**
3. **Normalization / scale discipline** (if part of tail in your op library)
4. **State coercion / final invariants** (ensures postconditions before commit)

You may implement these as distinct ops or one fused kernel, but the semantics MUST match.

## 3.3 Tail safety predicate (normative)

After tail completes, the following predicate MUST hold:

### `TAIL_SAFE(v, t)` postconditions

Let `c_out` be the coefficient state after tail.

1. **Bounded magnitude**

* For all coefficient indices `k`:
  `|c_out[k]| ≤ C_MAX` (C_MAX is a fixed constant per WSE-STD-001 / build config)

2. **Bounded energy (optional but recommended if you have an energy model)**

* `ENERGY(c_out) ≤ E_MAX(traction, preset)`
  If you do not implement energy, you MUST still guarantee (1).

3. **No NaN / undefined / wrap**

* All operations MUST be defined for all input values; no overflow wrap is allowed. Saturation MUST be used where needed (per WSE-STD-001).

4. **Deterministic coercion**

* If any upstream stage produced an illegal intermediate state (e.g., exceeded intermediate bounds), tail MUST coerce to a deterministic legal state using a deterministic rule (e.g., clamp + renormalize + optional simplify).
* Coercion MUST NOT depend on non-deterministic ordering or unseeded randomness.

5. **Idempotence-ish**

* Reapplying tail to an already tail-safe state MUST NOT introduce new structure; it may only further clamp/normalize. (Exact mathematical idempotence is not required, but “no new artifacts” is the intent.)

This predicate is what lets you safely time-slice and degrade without the synth “exploding.”

---

# 4) Budget model and dropping rules (executor hooks)

STD-030 defines *how budgets interact with execution*, not the global policy for distributing budget across voices (that policy is WSE-STD-050).

## 4.1 Cost accounting interface

Each operator `op_id` MUST have an integer **cost unit** `cost(op_id)`.

* Costs MAY be rough. They MUST be stable across builds for determinism.
* Tail ops MUST have a defined cost too, even if reserved.

## 4.2 Budget variables

Per block, executor is given:

* `B_total` : total available cost units for non-tail work (or total including tail, if you prefer, but then you must reserve tail first)
* `B_tail` : implicit reserved cost (sum of tail ops, if counted)
* `B_voice_max` (optional): per-voice cap

## 4.3 Drop policy (within a single voice)

If a voice begins execution and runs out of budget mid-backbone, the executor MUST apply the following deterministic drop rule:

**Rule D1 (drop remainder, not reorder):**

* Complete the operators already started for this block.
* Skip remaining backbone ops in list order until enough budget remains to run events and tail.
* Events MAY be partially dropped, but lane processing order must remain deterministic.

**Rule D2 (tail is sacred):**

* Tail MUST still execute fully.

**Rule D3 (no “partial op” semantics):**

* An operator is atomic at block level. If it starts, it must complete; if budget is insufficient to start it, it must be skipped.

## 4.4 Event dropping under budget pressure

Events are processed after backbone. If budget is tight:

**Rule D4 (drop lowest priority events first):**

* You MUST define a lane priority order for dropping.
* Default v0 recommendation (can be overridden, but must be frozen somewhere):

  1. Drop cosmetic/permutation bursts first
  2. Then drop injections
  3. Then drop toggles last (since toggles affect future behavior)
  4. Never drop emergency triggers (if any)

If you do not want priorities, you MAY adopt a simpler rule:

* Process lanes in order until budget remaining is below a threshold, then ignore the rest.
  But whichever you choose MUST be explicit and deterministic.

---

# 5) Determinism requirements (executor-visible)

Even though PRNG details are in WSE-STD-040, STD-030 imposes executor-level determinism constraints:

## 5.1 Deterministic draw sites

If an operator would draw randomness, then:

* It MUST draw only when the operator is actually executed (not when skipped).
* The draw count MUST NOT depend on data-dependent branches unless those branches are themselves deterministic and identical across implementations.

## 5.2 No hidden counters

The executor MUST NOT create timing by itself. All “every N blocks” patterns must arrive via VoiceFrame events, not internal `%` counters.

---

# 6) State commit and outputs

After tail:

* `SymbolState[v].c ← c_out`
* `SymbolState[v].op_enable_mask ← updated_mask` (if mutated by events)
* The executor MUST emit `c_out` to the wavelet engine for this voice/block, or write it to the appropriate buffer.

**Rule C1 (commit atomicity):** Commit MUST be atomic w.r.t. partial execution. If a voice is serviced, it either commits a fully tail-safe `c_out`, or (if not serviced) commits nothing.

---

# 7) Reference pseudocode (normative structure)

```python
def execute_voice_block(v, frame, sym_state, budget):
    c = sym_state[v].c
    mask = sym_state[v].op_enable_mask

    c_work = c

    # Backbone
    for (op_id, strength, params) in frame.backbone:
        if mask.disables(op_id):
            continue
        if budget.remaining < cost(op_id) + budget.reserve_for_events_and_tail():
            break
        c_work = APPLY(op_id, c_work, strength, params)
        budget.consume(cost(op_id))

    # Events (lanes 0..L-1)
    for lane in range(L):
        if not frame.event_mask & (1 << lane):
            continue

        if budget.remaining < cost_event_lane(lane) + budget.reserve_for_tail():
            if should_drop_lane(lane):   # deterministic rule
                continue
            else:
                # if lane is non-droppable and budget insufficient,
                # force it into a "no-op" deterministic behavior
                pass

        payload = resolve_payload(frame, lane)  # v0a/v0b rule
        c_work, mask = APPLY_EVENT_LANE(lane, c_work, mask, frame.event_strength[lane], payload)
        budget.consume(cost_event_lane(lane))

    # Tail (mandatory, reserved)
    c_out = APPLY_TAIL(c_work, frame.traction, preset_limits)
    assert TAIL_SAFE(c_out)

    sym_state[v].c = c_out
    sym_state[v].op_enable_mask = mask
    return c_out
```

---

# 8) What is “frozen” by STD-030

This standard freezes:

* The executor stage order (Backbone → Events → Tail)
* Author-order semantics (no reordering)
* Lane processing order (0..L-1)
* Tail being mandatory and budget-reserved
* Tail safety predicate (postconditions)
* Deterministic budget dropping semantics (atomic ops, skip remainder, tail always runs)

---

# 9) What remains open (explicitly not frozen here)

* Exact operator list, families, and math (operator specs)
* Exact lane mapping and payload bit layout (WSE-STD-020)
* Global scheduling policy across voices (WSE-STD-050)
* Exact numeric constants (C_MAX, E_MAX, priorities) unless you choose to freeze them now

---

## Quick questions I need answered to finalize the *few* remaining blanks inside STD-030

Answer in bullets; no need for long explanations:

1. **Tail contents:** which ops are tail in v0? (e.g., `Q`, `NORMALIZE_SCALE`, `EMERGENCY_SIMPLIFY`, anything else?)
2. **Event dropping:** do you want the simple rule “process lanes in order until budget runs out,” or a frozen lane-priority list?
3. **Toggle reset:** should `op_enable_mask` reset on **voice allocation** only, or also on **program/preset change**?

If you answer those 3, STD-030 becomes fully closed with no remaining ambiguity.
