# WSE-STD-CMMM v0 — CMMM Core State Machine

## 0) Purpose

CMMM is the **only time-owning** module. Per voice, per block, it deterministically compiles:

**(note lifecycle + controls + instrument preset + symbol hints + budget constraints) → VoiceFrame**

### Invariants (must hold)

1. **Bounded work:** emitted VoiceFrame always fits the regime budget after degradation.
2. **Deterministic:** same inputs + same initial state → same VoiceFrames (bit-exact).
3. **Click-closed:** all discontinuities are represented as ramps/crossfades managed by CMMM.
4. **Responsiveness:** any requested “meaningful change” must take effect within **≤ 5 blocks** (your guarantee).

---

## 1) Interfaces

### Inputs (per block)

* `block_index`
* `note_event` (optional): NOTE_ON/NOTE_OFF with pitch, vel, voice_id
* `macro_updates`: pots/CC (already quantized, Q15 or int)
* `symbol_hints`: optional (e.g., “prefer brighter”, “glitch”, “cloudify”, etc.)
* `budget_context`: regime budgets + voice class (MAIN/RESIDUAL)

### Output (per block)

* `VoiceFrame` (your WSE-STD-020 ABI; ideally v1 with carrier header)

---

## 2) Per-voice state (the whole machine)

This is what you store per voice (Python struct / FPGA regs). Everything else is tables.

### 2.1 Core state variables

* `voice_id`
* `active: bool`
* `age_blocks: uint32` (since note-on)
* `block_index_last`

### 2.2 Note lifecycle FSM (envelope time owner)

* `env_stage ∈ {IDLE, ATTACK, DECAY, SUSTAIN, RELEASE}`
* `gain_prev_q15`, `gain_curr_q15` (block endpoints for intra-block ramp)
* `activity_q15` (optional “voice alive” meter)

### 2.3 Regime FSM (mode/hysteresis)

* `regime_id ∈ {IDLE, TRANSIENT, STABLE_TONE, GLITCH_CHATTER, RECOVERY, RESIDUAL, CARRIER_ONLY}`
* `dwell_blocks`, `hysteresis_accum` (simple counters are enough)

### 2.4 Structural continuity manager (anti-click closure)

* `tree_id_a`, `tree_id_b`, `tree_blend_q15`
* `mask_u32` (canonical node mask)
* `pattern_id` (the rung id you already use)
* `tail_mode`
* `transition_timer_blocks` (for long swaps 1–5s)
* `fast_response_timer` (enforces ≤5-block response)

### 2.5 Carrier header state (stable pitch anchor)

* `carrier_model`, `carrier_role`, `transition_kind`
* `detune_cents`, `pitchbend_cents`
* `eta_prev_q15`, `eta_curr_q15` (carrier↔IDWPT blend ramp endpoints)

### 2.6 Executor latch mirrors (completion rules)

Keep the last-emitted backbone/events/payload in state so **VF_DELTA** and omission rules remain deterministic.

---

## 3) CMMM “one block” step order (deterministic)

This is the canonical order. Do not reorder.

### Step A — Ingest & quantize inputs (no audio math)

* Apply macro updates to local registers (Q15, clamped)
* Apply note events (update pitch, set active, reset timers)
* Decide voice class (MAIN/RESIDUAL) if you do cloud voices

### Step B — Update timeful states (envelope + regime)

* Update envelope stage → produce `gain_curr_q15`
* Set `gain_prev_q15 = last gain_curr_q15` (for interpolation)
* Update regime FSM using hysteresis + thresholds (gear/activity/controls)

**Rule:** no downstream module is allowed to derive time from anything but these block updates.

### Step C — Propose an *intended plan* (unconstrained)

From InstrumentIR + regime + symbol hints:

* desired backbone operator list
* desired event ops this block (if any)
* desired `K_target` (via desired pattern rung)
* desired `tail_mode_target`
* desired tree ids / blend target
* desired carrier params + eta target

This is where “musical intent” lives.

### Step D — Enforce click closure (turn jumps into ramps)

For any field that can click, convert to a ramp target:

* `gain_q15` ramps intra-block (renderer does linear interp)
* `gear_q15`, `traction_q15`, `eta_q15` ramp over N blocks
* `tree_blend_q15` ramps over N blocks (long: 1–5s)
* `pattern_id / mask` changes either:

  * crossfade via `tree_blend` (preferred), or
  * “mask morph” using a controlled rung schedule over time

**Hard rule:** CMMM is the only place allowed to schedule these ramps.

### Step E — Apply budget system (the bounded-work closure)

Call your implemented pipeline:

`predict_chain_cost(backbone, events, K, tail) → check_budget_fit → apply_degradation`

Outputs:

* degraded `backbone_ops`
* degraded `event_ops`
* degraded `pattern_id`/`mask` (via K rung)
* degraded `tail_mode`

**Policy rule:** budget fit happens *after* musical intent, *before* VoiceFrame emission.

### Step F — Emit VoiceFrame (ABI)

Fill VoiceFrame fields from the *final* plan:

* `regime_id`, `gear_q15`
* `tree_id_a/b`, `tree_blend_q15`
* `gain_q15`, `traction_q15`
* `backbone[]` in semantic order
* `event_mask`, `event_strength[]`, `payload_common`
* carrier header fields (if using v1)

Then latch output as “previous command” for deterministic completion.

---

## 4) The actual state machine (minimal but complete)

### 4.1 Note lifecycle FSM

* `IDLE → ATTACK` on NOTE_ON
* `ATTACK → DECAY → SUSTAIN` by countdown
* `* → RELEASE` on NOTE_OFF
* `RELEASE → IDLE` when gain hits 0

Outputs: `gain_curr_q15`, `activity_q15`

### 4.2 Regime FSM (example rules)

* `IDLE` when `activity_q15 ~ 0`
* `TRANSIENT` for first N blocks after NOTE_ON or sharp macro change
* `STABLE_TONE` after dwell timer
* `GLITCH_CHATTER` when symbol asks + budget allows events
* `RECOVERY` after emergency or large clipping risk
* `RESIDUAL` when voice is cloud-class
* `CARRIER_ONLY` when budget forces it (or intentional)

### 4.3 Structural transition FSM

Two timers:

* **Fast timer (≤5 blocks):** ensures audible response quickly (eta ramp, gear ramp, small K change).
* **Slow timer (1–5s):** amortizes heavy structural swaps (tree changes, large mask topology change).

Rule: **carrier leads, wave packet may come late**
So when doing big changes:

1. update carrier pitch/timbre immediately (≤5 blocks)
2. bring in IDWPT texture gradually (slow timer)
3. only then do expensive topology swaps if needed

---

## 5) What you should fix before you freeze it (small!)

You’re ready, but two tiny things will save pain:

1. **Ensure MIN tail is linear** in your cost flags (FineClamp shouldn’t force nonlinear disallow loops).
2. Add a **baseline per-voice tick cost** so “empty ops” doesn’t masquerade as a valid audio plan in RESIDUAL/CARRIER_ONLY.

Neither blocks you from writing CMMM; they just prevent pathological outcomes.

---

## 6) CMMM kernel signature (Python-first, FPGA-ready)

```python
def cmmm_step_block(
    st: VoiceState,
    inputs: BlockInputs,
    instrument: InstrumentIR,
    tables: Tables,          # costs, budgets, patterns, tail defs
) -> tuple[VoiceState, VoiceFrame]:
    ...
```

Everything you’ve built (budgets module) plugs into **Step E**.

---

If you want the next concrete deliverable: I can write the **exact `VoiceState` dataclass + `cmmm_step_block()` pseudocode** with your current fields (VoiceFrame v0/v1, pattern_id mapping, tail modes, and the ≤5 block response guarantee) so you can drop it into your repo as `cmmm_core.py`.
