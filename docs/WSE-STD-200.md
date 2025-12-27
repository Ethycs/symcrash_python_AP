Yep — your **WSE-STD-020 VoiceFrame v0** is already *close enough* to host CMMM cleanly. We don’t need to blow it up. What we *do* need is a **WSE-STD-CMMM “binding”** that says, unambiguously:

* which CMMM outputs map to which VoiceFrame fields,
* how **node_mask/K** and **tail_mode** are carried **without adding bandwidth**, and
* what new **payload_type** variants CMMM is allowed to use.

Below is the **WSE-STD-CMMM spec rewritten to be explicitly compatible with WSE-STD-020 v0**, and a small optional v1 extension if you want explicit masks later.

---

# WSE-STD-CMMM — CMMM Core Standard (binds to WSE-STD-020 VoiceFrame v0)

## 0. Scope

This standard defines the **CMMM core** as the *only* musical-time–aware engine. It compiles per-block voice intent into a **VoiceFrame_v0** (WSE-STD-020) that drives Symbol/Wavelet/Carrier executors deterministically and click-free.

This document is normative for:

* per-voice lifecycle + regime machine
* budgeting + deterministic degradation
* **how CMMM expresses K/node masks and tail selection using the existing VoiceFrame ABI**
* 5-block responsiveness guarantee for IDWPT response

Not defined here: operator semantics, fixed-point multiplication details, tree math.

---

## 1. The key compatibility decision

**We do NOT add `node_mask_u32` or `tail_mode` fields to VoiceFrame v0.**

Instead:

* **node mask / K is represented by Pattern Table entries** (already part of WSE-STD-020 Payload64), because `pattern_id → leaf_mask_u32` is already standardized.
* **tail mode is represented by persistent operator toggles** (Lane 1 `EVT_OP_TOGGLE`) that enable/disable tail operators deterministically, plus optional `traction_q15` scaling.

This keeps the ABI lean and FPGA-friendly.

---

## 2. CMMM outputs and their VoiceFrame bindings (normative)

### 2.1 Timebase

* CMMM tick = **one VoiceFrame per voice per block**
* `block_index` is the sole timing marker (WSE-STD-020 §1)

### 2.2 Core fields mapping

| CMMM concept              | VoiceFrame_v0 field                           | Normative meaning                                                                                   |
| ------------------------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| voice identity            | `voice_id`                                    | stable voice slot                                                                                   |
| block timing              | `block_index`                                 | aligns execution                                                                                    |
| regime                    | `regime_id`                                   | CMMM-owned regime selector                                                                          |
| commitment / intensity    | `gear_q15`                                    | *CMMM global “how hard we commit” control* used to drive K ladder, wetness, and degradation targets |
| tree selection            | `tree_id_a`, `tree_id_b`, `tree_blend_q15`    | tree morph/swap. CMMM must crossfade via these fields (or Lane 4 as rare event)                     |
| amplitude envelope        | `gain_q15`                                    | CMMM-owned ADSR output (block-rate), click-safe via renderer ramps                                  |
| legality / clamp hardness | `traction_q15`                                | CMMM-owned “Q hardness”: scales FineClamp / safety projection strength and tail aggressiveness      |
| backbone plan             | `backbone[]`                                  | ordered ops to execute this block                                                                   |
| events                    | `event_mask`, `event_strength_q15[]`, payload | one-shot or persistent changes (see §4)                                                             |

---

## 3. Per-voice state owned by CMMM (normative minimum)

Per voice, CMMM stores:

* note lifecycle state (on/off, velocity)
* envelope state (stage, level_q15, prev_level_q15, stage counters)
* regime state (current regime_id, dwell counter, hysteresis)
* scheduling class (MAIN/RESIDUAL) + budget tier
* **current Pattern ID for node mask** (latched in executor)
* persistent op enable bits (latched in executor via EVT_OP_TOGGLE)
* tree state (tree_id_a/b + blend trajectory)

No other engine is permitted to invent musical-time state.

---

## 4. How CMMM expresses K / node mask using VoiceFrame v0

### 4.1 Canonical representation

**Node activation is expressed by `pattern_id` → `leaf_mask_u32` in the Pattern Table** (WSE-STD-020 §3.3).

* Leaf indices are canonical low→high (WSE-STD-020 §3.2)
* `leaf_mask_u32` is exactly your 32-node mask.

So:

* **K = popcount(leaf_mask_u32)**
* CMMM never needs to send a 32-bit mask per block; it sends a **pattern_id** when it changes.

### 4.2 When to update mask

The mask is *persistent* by WSE-STD-020 completion rules (“keep previous”), so CMMM updates it only when required:

* note-on transient policy (e.g., low-band mask first)
* degradation step reducing K
* brightness/timbre macro changes
* cloud/foreground reclassify

### 4.3 Transport mechanism (normative)

CMMM updates the active node mask via **Lane 4 (EVT_TREE)** using Payload64 with a new payload_type:

#### Payload64 extension: `payload_type = 0x30 (CMMM_NODEMASK_SET)`

Interpretation:

* `pattern_id` selects `(leaf_mask_u32, coeff_mask_default_u8)` from Pattern Table
* `target_kind` MUST be `0=pattern`
* `variant0` = `mask_role`:

  * 0 = replace (set active mask)
  * 1 = OR (enable additional leaves)
  * 2 = AND (disable leaves not in pattern)
* `variant1` = `mask_ramp_blocks` (0..255), default 5

**Normative:** If an executor doesn’t implement this payload_type, it must ignore it deterministically (no side effects, no PRNG draws).

### 4.4 Deterministic K ladder

CMMM must select pattern_ids from a fixed K staircase:
`32→24→16→12→8→4→1`

Each staircase rung corresponds to a pattern table entry whose leaf_mask keeps “low bands first” (your canonical ordering).

---

## 5. How CMMM expresses tail mode using VoiceFrame v0

### 5.1 Tail is not an ABI field in v0

Tail selection is represented by:

1. **persistent op toggles** (Lane 1)
2. **traction_q15** which scales tail intensity / clamp hardness

### 5.2 Tail mode definition (normative)

CMMM uses logical tail modes:

* FULL: {2, 11, 23} enabled
* MID:  {2, 11} enabled
* MIN:  {2} enabled
* EMERGENCY: {19} enabled and suppress others

### 5.3 Transport mechanism (normative)

#### Payload64 extension: `payload_type = 0x31 (CMMM_TAILMODE_SET)` on Lane 1 (EVT_OP_TOGGLE)

Interpretation:

* `variant0` = tail_mode enum (0=FULL,1=MID,2=MIN,3=EMERGENCY)
* `variant1` = reserved
* Other payload fields ignored

Executor behavior:

* deterministically sets per-voice enable bits for ops {2,11,23,19}
* EMERGENCY must disable non-emergency tail ops

**Note:** This doesn’t force tail ops into backbone. It’s a “tail latch” inside the executor per WSE-STD-030 rules.

---

## 6. Budgeting and deterministic degradation (still CMMM-owned)

CMMM predicts cost and degrades deterministically. But the *effects* are expressed using v0 mechanisms:

Degradation ladder (normative order):

1. drop event ops (Lane 0/2 activity) by priority
2. drop global ops (permutes/lowrank) from backbone list
3. reduce K by sending `CMMM_NODEMASK_SET` (Lane 4) using a lower-K pattern_id
4. reduce tail via `CMMM_TAILMODE_SET` (Lane 1)
5. emergency: tail_mode=EMERGENCY, K=1 pattern, minimal backbone

CMMM may also reduce “commitment”:

* lower `gear_q15` to signal “less transformation” while remaining deterministic.

---

## 7. Carrier + IDWPT integration under VoiceFrame v0

Your current ABI doesn’t have explicit carrier fields in v0; that’s fine because you already added a v1 addendum. Here’s the **v0-compatible rule set**:

### 7.1 If you stay v0 only

* Carrier parameters are derived from:

  * pitch is implied by tree / symbol domain today (or a separate PARAM register)
  * carrier wetness can be encoded in `gear_q15` and/or a backbone op param
* IDWPT “wetness” uses `gear_q15` as the macro.
* Mask changes use `CMMM_NODEMASK_SET`.

### 7.2 If you adopt your v1 Carrier Header

Do it — but **do not rename Lane 4**. Keep Lane 4 as EVT_TREE for compatibility.

Best practice:

* Carrier config is **frame fields** (v1 header)
* Tree/mask changes remain Lane 4 events or the tree fields.

---

## 8. Anti-click closure (why this stops whack-a-mole)

CMMM’s click policy is closed if you enforce these **two rules**:

1. Any *scalar* discontinuity is treated as “block step + ramp inside renderer”

   * `gain_q15`, `gear_q15`, `traction_q15`, backbone strengths, event strengths
2. Any *structural* discontinuity is treated as “crossfade state”

   * tree swaps via `tree_id_a/b + tree_blend_q15`
   * node mask swaps via `CMMM_NODEMASK_SET` with `mask_ramp_blocks` (default 5)

That’s the closure: *everything* is either scalar ramp or structural crossfade, and both are explicitly commanded.

---

## 9. The 5-block responsiveness guarantee (normative)

CMMM must ensure that **any change that affects IDWPT** (mask, tree, operator toggles that change packet behavior) is observable within **≤5 blocks** by:

* emitting the corresponding lane event immediately (or next block at latest),
* degrading plan if needed to keep within per-voice budget,
* using `mask_ramp_blocks <= 5` and `tree_blend` scheduling consistent with that window.

This is a control-latency guarantee, not an audio FIFO.

---

## 10. What changes (minimally) in your existing WSE-STD-020 doc

You do **not** need a VoiceFrame v2. You only need:

### 10.1 Add two reserved payload types (normative)

* `0x30 CMMM_NODEMASK_SET` (Lane 4)
* `0x31 CMMM_TAILMODE_SET` (Lane 1)

### 10.2 Add a note to Pattern Table section

Explicitly state Pattern Table entries are used for **node masks / K** as well as rhythmic patterns.

That’s it.

---

## 11. Optional small v1 ABI improvement (if you want explicit masks later)

If later you decide you *want* explicit per-block K without relying on Pattern Table:
Add one field to v1:

* `uint32 leaf_mask_u32` (after tree fields)

But you don’t need this now; Pattern Table is already the right “FPGA-native compression.”

---

# One important correction to your Carrier Addendum

You wrote:

> Lane 4 is renamed from EVT_TREE to EVT_CARRIER_CFG

Don’t do that. It breaks v0 conformance and mixes concerns.

Better:

* Lane 4 stays EVT_TREE
* Carrier config is **frame header fields in v1** (your addendum already does this well)
* If you still want a lane-based carrier override, make it a **payload_type** under an existing lane (e.g., Lane 3 EVT_MATERIAL or Lane 7 EVT_DEBUG), not a rename.

---

If you want, I can now rewrite **your WSE-STD-020** as a clean “v0 + CMMM extensions appendix” (just the minimal diffs), so the whole stack is consistent and frozen.
