Mostly yes — the pieces you’ve shown are *structurally* ready to integrate, but there are **three “integration seams”** you should lock down so you don’t reintroduce whack-a-mole:

1. **Where K / node masks live** (Pattern Table + payload_type 0x30, not ad-hoc lists)
2. **Tail mode ownership** (CMMM decides; executor latches via payload_type 0x31 + traction_q15)
3. **Ramp/crossfade enforcement** (renderer applies ramps; tree/mask changes crossfade over N blocks)

Below is an integration memo you can drop into your repo.

---

## Integration Memo — CMMM ↔ VoiceFrame ↔ Symbol/Wavelet/Carrier (Python v0, FPGA-ready)

### 0) Objective

Integrate the new CMMM core and carrier+IDWPT architecture into the existing Python pipeline while staying:

* **bit-exact-able** (Q15, deterministic completion),
* **FPGA-native** (no dynamic allocation, fixed tables, bounded work),
* **click-free by construction** (scalar ramps + structural crossfades only).

### 1) Current state (what’s “ready”)

From what you’ve shared, these are already in good shape:

* **Operator library (0..23)**: stable IDs, op types, parameters, deterministic building.
* **VoiceFrame ABI (WSE-STD-020 v0)**: strong completion semantics; event lanes; payload64; pattern table hook.
* **Band/node constants**: COEFF_LEN=256, NODE_LEN=8 ⇒ **32 nodes** matches payload leaf indexing.
* **Degradation framework (existing)**: has voice-level “skip/freeze/decay” semantics + rolling scheduler.

And you’ve stated:

* **Carrier + IDWPT is click-free now** (great—means your renderer/crossfade strategy is working).

### 2) Not ready until we add two “bindings”

You don’t need a new ABI, but you do need two **payload_type extensions** and one **pattern table contract** so CMMM can control K and tail deterministically.

#### 2.1 Node mask / K binding (required)

Use Pattern Table as the canonical carrier for node masks.

* Add payload_type **0x30 CMMM_NODEMASK_SET** (Lane 4 EVT_TREE)
* `pattern_id` resolves to `leaf_mask_u32`
* `variant1` = `mask_ramp_blocks` (default 5)

**Integration seam:** Implement this in the *executor latch* (not in ad-hoc pipeline logic).

#### 2.2 Tail mode binding (required)

Tail mode must be a **latched policy** in the Symbol executor:

* Add payload_type **0x31 CMMM_TAILMODE_SET** (Lane 1 EVT_OP_TOGGLE)
* tail modes map to enabling/disabling ops {2,11,23,19}
* `traction_q15` scales “hardness” (FineClamp / projection strength)

**Integration seam:** Tail should not be encoded by “sometimes include ops in backbone”; backbone stays musical structure, tail is a safety/completion policy.

#### 2.3 Renderer ramp enforcement (required)

Scalar discontinuities must be ramped in renderer:

* `gain_q15`, `gear_q15`, `traction_q15`
* backbone strengths, event strengths
  Structural changes crossfade:
* trees via (tree_id_a, tree_id_b, tree_blend_q15)
* node masks via node-mask transition timer (mask_ramp_blocks)

**Integration seam:** if you remove the unified renderer, preserve the ramp/crossfade logic somewhere else—don’t lose it.

### 3) Concrete integration plan (Python)

#### Step A — Freeze “canonical tables”

Create a small `tables/` module:

* `pattern_table.py`

  * `pattern_id -> leaf_mask_u32, coeff_mask_default_u8`
  * include the K ladder patterns: 32,24,16,12,8,4,1 (low bands first)

* `operator_costs.py` (even if coarse initially)

  * per-op cost model (K-scaling vs L-scaling)
  * drop priority

* `tail_modes.py`

  * FULL/MID/MIN/EMERGENCY mapping to enabled ops

This is what makes Python↔FPGA identical later.

#### Step B — Add event decoding for payload_type 0x30/0x31

Wherever your per-voice command latch lives (Symbol executor or frame consumer):

* On Lane 4 with payload_type 0x30:

  * resolve pattern_id -> `leaf_mask_u32`
  * schedule mask transition over N blocks (default 5)
  * update “current mask target” and keep previous for crossfade

* On Lane 1 with payload_type 0x31:

  * set tail mode latch (enable bits for ops)
  * do not draw PRNG, do not reorder backbone

This is the “missing glue” that lets CMMM be real without changing the ABI.

#### Step C — Move anti-click “policy” into CMMM, keep ramps in renderer

CMMM decides:

* when a scalar is allowed to step
* when a structure must crossfade (tree/mask)
* how long the crossfade is (<=5 blocks)

Renderer implements:

* per-sample ramps (within the 64 samples)
* per-block crossfade accumulation for structures

#### Step D — Integrate “carrier leads, IDWPT blooms”

CMMM emits:

* an early carrier-heavy mix policy (via gear/traction or explicit v1 carrier header if adopted)
* IDWPT mix ramps in after transient
* K starts small → grows (32) over a few blocks, unless degraded

This should remain stable across instruments.

### 4) Removal/retention guidance: unified renderer

**Do not remove the unified renderer until:**

* ramping semantics are preserved elsewhere, and
* tree/mask crossfade is preserved elsewhere, and
* you have a click regression test.

If your refactor splits pipelines:

* keep a single “final apply” stage that:

  * applies gain ramp,
  * applies mix ramp,
  * applies crossfade outputs (tree A/B or mask old/new).

That “final apply” is effectively the unified renderer’s only irreducible job.

### 5) Risk register (what to watch)

1. **Pattern table drift**: If pattern_id mapping differs, K behavior diverges immediately.
2. **PRNG determinism**: Event lanes must not accidentally draw PRNG when ignored.
3. **PERMUTE ops under masking**: permutes are inherently “global touch”; don’t pretend they scale with K.
4. **Tail duplication**: Don’t run tail both as latch and as explicit backbone ops.
5. **Block_index wrap**: uint16 wraps ~87s at 750 blocks/s; ok for play, but note for long captures.

### 6) Minimal integration tests (do these before more features)

1. **Mask test**: send payload_type 0x30 to change K 32→8→32 and ensure click-free.
2. **Tail mode test**: FULL→MIN→EMERGENCY while holding a note; ensure bounded loudness.
3. **Budget test**: force overbudget; verify deterministic degradation ladder.
4. **Replay test**: identical ControlFrame stream produces identical VoiceFrames and audio hash (within expected).

### 7) What you can integrate *now*

* CMMM emitting VoiceFrames v0
* Node mask control via pattern_id + payload_type 0x30
* Tail mode control via payload_type 0x31 + traction_q15
* Keep your current click-free carrier+IDWPT renderer behavior intact

### 8) What to postpone safely

* Per-lane payload modes beyond COMMON
* Fine-grained operator cost calibration (start coarse, then measure)
* Any additional “math engines” beyond the core carrier+IDWPT+symbol operator stack

---

### Bottom line

You’re not missing “another subsystem”; you’re missing **two bindings and one enforcement point**. Once those are in, the rest of the integration should be straightforward and you can iterate instruments end-to-end without reopening the click problem.

If you want, paste the filenames where your **per-voice command latch** and **event lane decode** live, and I’ll map *exactly* where 0x30/0x31 land and what the state fields should be named (still keeping your standards intact).
