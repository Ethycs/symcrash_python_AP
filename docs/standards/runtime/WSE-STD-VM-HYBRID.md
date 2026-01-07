# WSE-STD-VM-HYBRID v0 — Microcoded Block VM with Fixed Lanes

**Status:** Draft
**Version:** 0.1.0
**Date:** 2025-12-26

## 0. Scope

Defines a **per-block, per-voice** execution machine ("hybrid VM") for the Symbol Engine synthesizer:

* Executes **VoiceFrame** as microcode
* Runs a fixed set of **hardware-friendly lanes**
* Guarantees **deterministic, bounded work** (when paired with CMMM budget/degradation)
* Supports **carrier oscillator + IDWPT lift/effect** in one unified pipeline

This standard does **not** define CMMM policy (regimes, budgets) beyond the hooks it requires.

---

## 1. Core Concepts

### 1.1 Microcode Contract

A `VoiceFrame` is interpreted as a **microprogram** for one voice for one block.

* Backbone entries = ordered instructions
* Event lanes = conditional "sideband instructions"
* Tree/mask/tail/carrier fields = block-scoped execution configuration

### 1.2 Fixed Lanes

Instead of an open-ended instruction set, the VM provides **fixed execution lanes** that map cleanly to FPGA datapaths and Python reference code.

**Lane Set (v0):**

| Lane | Name | Description |
|------|------|-------------|
| 1 | `LANE_CARRIER_RENDER` | Phase-continuous oscillator |
| 2 | `LANE_INJECT` | Event injection → coeff_state |
| 3 | `LANE_DIAG` | Diagonal operators (K-scaling) |
| 4 | `LANE_BANDED` | Banded operators (K-scaling) |
| 5 | `LANE_PERMUTE` | Permutation operators (L-scaling) |
| 6 | `LANE_LOWRANK` | Low-rank operators (L-scaling) |
| 7 | `LANE_TAIL` | Mandatory tail operators |
| 8 | `LANE_IDWPT_RENDER` | Wavelet synthesis → texture_buf |
| 9 | `LANE_MIX_GAIN` | Eta blend + gain ramp → output |

Backbone ops select among lanes 3–6 (and optionally 2). Tail is always lane 7. Render is lanes 1 & 8.

---

## 2. Determinism + Boundedness Invariants

### 2.1 Determinism Rules

The VM MUST satisfy:

* Same `(initial state, VoiceFrame sequence, tables)` → identical output blocks (bit-exact in fixed-point)
* Unsupported event lanes MUST be ignored **without side effects** (no PRNG draws, no state changes)
* No data-dependent loops: runtime depends only on (op list, K, tail mode), not coefficient values

### 2.2 Boundedness Rule

The VM MUST execute exactly:

* `backbone_len ≤ N_BACKBONE_MAX` (typically 16)
* At most one invocation per event lane bit set
* Exactly one tail mode (including EMERGENCY)

CMMM + budgets ensure the emitted frame fits; VM assumes the frame is valid and executes it.

---

## 3. VM State (Per Voice)

### 3.1 Persistent State (Required)

```
VMState {
    voice_id: u16

    # Carrier oscillator
    phase: u32                    # Phase accumulator (0..2^32-1)
    phase_inc: u32                # Phase increment per sample
    target_phase_inc: u32         # Target for pitch slew
    pitch_slew_rate: u32          # Per-sample delta for portamento
    harmonic_amps_q15: [i16; 16]  # Per-harmonic amplitudes
    envelope: EnvelopeState       # ADSR state
    dc_block_state: i32           # DC blocking filter state

    # Coefficients
    coeff_state: [i16; COEFF_LEN] # Q15 coefficient vector (256 or 64)

    # PRNG
    prng_state: u32               # xorshift32 state

    # Tree (optional)
    tree_state: Optional<TreeState>
}
```

### 3.2 Block-Local Scratch (Not Persistent)

```
carrier_buf: [i16; BLOCK_SAMPLES]   # Carrier output
texture_buf: [i16; BLOCK_SAMPLES]   # IDWPT output
tmp_coeff: [i16; COEFF_LEN]         # Optional ping-pong buffer
```

FPGA note: scratch can be time-multiplexed; Python can allocate arrays.

---

## 4. Tables (Shared, Versioned)

The VM depends on immutable tables (must match Python + FPGA):

| Table | Purpose | Key → Value |
|-------|---------|-------------|
| Operator Library | Op definitions | `op_id → (lane, params, flags)` |
| Pattern Table | K-ladder masks | `pattern_id → mask_u32` |
| Tail Definition | Tail op lists | `tail_mode → [op_id]` |
| Wavelet Tree | IDWPT topology | `tree_id → lifting_network` |
| Carrier | Oscillator models | `model_id → wavetable/params` |

**Table Version:** All tables must be versioned. VM rejects frames with mismatched table versions.

---

## 5. Operator Lane Assignment Table

All 24 operators with their lane assignment, scaling mode, and degradation priority:

| ID | Name | Lane | K/L | Nonlinear | Tail | Drop Priority |
|----|------|------|-----|-----------|------|---------------|
| 0 | Identity | DIAG | - | No | No | 0 (never) |
| 1 | Tilt | DIAG | K | No | No | 80 |
| 2 | FineClamp | PROJ | L | No | **Yes** | 0 (never) |
| 3 | IntraMix | BANDED | K | No | No | 110 |
| 4 | DiffuseUp | BANDED | K | No | No | 120 |
| 5 | DiffuseDown | BANDED | K | No | No | 120 |
| 6 | BandShear | PERMUTE | L | No | No | 140 |
| 7 | LowRankMorph | LOWRANK | L | No | No | 200 |
| 8 | GlobalPulse | INJECT | K | No | No | 220 |
| 9 | SwapBands | PERMUTE | L | No | No | 150 |
| 10 | RotateNodes | PERMUTE | L | No | No | 130 |
| 11 | NormalizeScale | PROJ | L | **Yes** | **Yes** | 0 (never) |
| 12 | GluePair | PROJ | L | No | No | 60 |
| 13 | Impulse | INJECT | K | No | No | 230 |
| 14 | NoiseBand | INJECT | K | No | No | 240 |
| 15 | PressureRamp | DIAG | K | No | No | 160 |
| 16 | GateMask | DIAG | K | No | No | 40 |
| 17 | QuasiRotate | PERMUTE | L | No | No | 180 |
| 18 | MorphMix | MIX | - | No | No | 100 |
| 19 | EmergencySimplify | PROJ | L | No | **Yes** | 0 (never) |
| 20 | FoldPressure | DIAG | K | No | No | 160 |
| 21 | CuspInjection | INJECT | K | No | No | 248 |
| 22 | SwallowtailMask | DIAG | K | No | No | 180 |
| 23 | SheafProject | PROJ | L | No | **Yes** | 10 |

### 5.1 Scaling Modes

* **K-scaling**: Cost scales with active nodes `K` (1-32). DIAG, BANDED, INJECT lanes.
* **L-scaling**: Cost scales with total nodes `L` (always 32). PERMUTE, LOWRANK, PROJ lanes.

### 5.2 Tail Operators

Tail operators (2, 11, 19, 23) have special rules:

* **MUST NOT** appear in backbone unless part of tail sequence
* **drop_priority = 0** means they cannot be dropped by degradation
* Tail mode selects which subset executes:
  * `FULL` = [2, 11, 23]
  * `MID` = [2, 11]
  * `MIN` = [2]
  * `EMERGENCY` = [19]

### 5.3 Nonlinear Operators

Only op 11 (NormalizeScale) is marked nonlinear (uses sqrt-ish normalization).

When `budget.allow_nonlinear = False`:
* Op 11 is stripped from backbone
* Tail mode is forced to MIN or EMERGENCY

---

## 6. Instruction Decoding

### 6.1 BackboneEntry Decoding

Each `BackboneEntry(op_id, strength_q15, param0_q15, param1_q15)` decodes to:

```python
lane = get_lane(op_id)  # From operator table
exec(lane, op_id, strength_q15, param0_q15, param1_q15, mask_u32, coeff_state)
```

**Normative:** Backbone operator order is semantic; VM MUST execute in listed order.

### 6.2 Events Decoding

Event lanes are decoded using:

* `event_mask: u8` (8 lanes)
* `event_strength_q15[lane]`
* `payload_mode` + `payload64` selectors

Events MUST execute **after carrier render but before tail**, unless explicitly categorized as "render-side" events.

---

## 7. Canonical Per-Block Execution Order

For each voice, for each block:

### Step 0 — Resolve Configuration

```python
DecodedPlan {
    mask_u32: int           # From pattern_id or direct
    K: int                  # popcount(mask_u32)
    tail_ops: List[int]     # From tail_mode
    tree_a, tree_b: int     # Tree IDs
    tree_blend_q15: int     # Morph position
    carrier_params: ...     # Pitch, harmonics, model
    eta_prev_q15: int       # Previous block's eta
    eta_curr_q15: int       # Current block's eta target
    gain_prev_q15: int      # Previous block's gain
    gain_curr_q15: int      # Current block's gain target
    backbone_instrs: [...]  # Backbone entries
    event_instrs: [...]     # Active event lanes
    carrier_role: enum      # OFF, EXCITER_ONLY, AUDIBLE
    idwpt_enable: bool      # Whether to render IDWPT
}
```

### Step 1 — LANE_CARRIER_RENDER

Render block `carrier_buf` with phase continuity.

* If `carrier_role = OFF`: `carrier_buf = zeros`
* If `carrier_role = EXCITER_ONLY`: render but used only for IDWPT excitation
* If `carrier_role = AUDIBLE`: render for mixing

Must be click-free internally (phase accumulator; glide handled by pitch_slew_rate).

### Step 2 — LANE_INJECT

Apply injections into `coeff_state` under `mask_u32`:

* Process active event lanes (impulse/noise/pulse/cusp seeds)
* Uses payload selectors to pick target leaves/coeffs

**Determinism:** PRNG state advances ONLY if an op requiring PRNG executes.

### Step 3 — Coefficient Backbone Lanes (In Order)

For each backbone entry:

```python
match lane:
    case DIAG:    lane_diag(state, instr, mask)
    case BANDED:  lane_banded(state, instr, mask)
    case PERMUTE: lane_permute(state, instr, mask)
    case LOWRANK: lane_lowrank(state, instr, mask)
    case PROJ:    lane_proj(state, instr, mask)
    case MIX:     lane_mix(state, instr, mask)
```

**Normative:** Operations apply to coefficient state, not audio buffers.

### Step 4 — LANE_TAIL

Execute the tail mode:

* `FULL`: FineClamp → NormalizeScale → SheafProject
* `MID`: FineClamp → NormalizeScale
* `MIN`: FineClamp
* `EMERGENCY`: EmergencySimplify

Tail is where you enforce "legal/pleasant" bounds.

### Step 5 — LANE_IDWPT_RENDER

Render texture from `coeff_state` using IDWPT:

* If `idwpt_enable = False`: `texture_buf = zeros`
* Optionally excited/conditioned by `carrier_buf`

**Important:** IDWPT render MUST NOT reset phases per block unless explicitly commanded.

### Step 6 — LANE_MIX_GAIN

Compute final output block:

```python
# Intra-block eta ramp
for i in range(BLOCK_SAMPLES):
    alpha = i / BLOCK_SAMPLES
    eta = lerp(eta_prev_q15, eta_curr_q15, alpha)
    dry_w, wet_w = equal_power_weights(eta)
    mixed[i] = dry_w * carrier_buf[i] + wet_w * texture_buf[i]

# Intra-block gain ramp
for i in range(BLOCK_SAMPLES):
    alpha = i / BLOCK_SAMPLES
    gain = lerp(gain_prev_q15, gain_curr_q15, alpha)
    output[i] = (mixed[i] * gain) >> 15
```

This step is where "click-free output" is guaranteed even if parameters step at block boundaries.

---

## 8. Latency as Microcode Contract

The 5-block IDWPT latency is **NOT** an audio FIFO trick. It's a **microcode scheduling contract**:

1. **CMMM emits frames 5 blocks ahead** for the wavelet path
2. **Carrier frames are emitted at block N**, wavelet frames at block N-5
3. **VM executes both in lockstep** — no internal buffering needed
4. **Beta ramping is in the microcode** via `eta_prev_q15 → eta_curr_q15`

This means:

* No carrier FIFO in VM (that's CMMM's job to schedule)
* VM sees aligned frames and just executes
* "Wet ready" is a microcode state bit, not a runtime condition
* First 5 blocks have `eta_curr_q15 = 0` (pure carrier)

---

## 9. Budget Integration Hook

The hybrid VM itself is dumb: it executes the frame.

Budgeting happens upstream (CMMM), but the VM requires:

* `K` and `tail_mode` are **explicit** in the microcode
* op IDs are bounded (backbone_len max, events max)

Predictor maps to VM cost naturally:

```
ticks ≈ Σ lane_cost(op_id, K, L) + render_cost + tail_cost(tail_mode, K, L)
```

**Normative:** If a VoiceFrame violates the budget, the VM still executes it (for debugging), but FPGA builds MAY optionally assert and enter emergency for safety.

---

## 10. Python Reference Implementation

```python
def vm_step_block(state: VMState, frame: VoiceFrame, tables: Tables) -> Tuple[VMState, np.ndarray]:
    """Execute one block for one voice."""
    cfg = decode_frame(frame, tables)

    # Lane 1: Carrier
    if cfg.carrier_role != CarrierRole.OFF:
        carrier_buf = lane_carrier_render(state, cfg)
    else:
        carrier_buf = np.zeros(BLOCK_SAMPLES, dtype=np.int16)

    # Lane 2: Inject
    lane_inject(state, cfg)

    # Lanes 3-6: Backbone (in order)
    for instr in cfg.backbone_instrs:
        dispatch_coeff_op(state, instr, cfg.mask_u32)

    # Lane 7: Tail
    lane_tail(state, cfg)

    # Lane 8: IDWPT
    if cfg.idwpt_enable:
        texture_buf = lane_idwpt_render(state, cfg, carrier_buf)
    else:
        texture_buf = np.zeros(BLOCK_SAMPLES, dtype=np.int16)

    # Lane 9: Mix + Gain
    output = lane_mix_gain(cfg, carrier_buf, texture_buf)

    return state, output
```

FPGA mapping is the same schedule, with each lane as a pipelined unit and the dispatcher as a microsequencer.

---

## 11. PRNG Determinism

The VM maintains per-voice PRNG state using xorshift32:

```python
def xorshift32(state: int) -> Tuple[int, int]:
    """Returns (new_state, random_value)."""
    x = state
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17)
    x ^= (x << 5) & 0xFFFFFFFF
    return x, x
```

**Rules:**
* PRNG advances ONLY when a noise/random op executes
* Unused event lanes do NOT advance PRNG
* Initial state is set at voice allocation
* Snapshotable for deterministic replay

---

## 12. VM Done Criteria

A conforming VM implementation must satisfy:

- [ ] Runs one voice for one block: `(state, frame) -> (state', block)`
- [ ] Carrier is phase-continuous across blocks
- [ ] IDWPT is deterministic (no hidden phase resets)
- [ ] Tail modes execute in fixed order
- [ ] Gain and eta ramps applied intra-block (click-free)
- [ ] PRNG advances only when noise op executes
- [ ] Same frames produce bit-identical output every run

---

## Appendix A: Lane Cost Model

For CMMM budget prediction:

| Lane | Cost Formula |
|------|--------------|
| CARRIER | `BASE_CARRIER + BLOCK_SAMPLES * harmonics` |
| INJECT | `BASE_INJECT + K * COEFF_LEN_PER_NODE` |
| DIAG | `ticks_base + K * ticks_per_active_node` |
| BANDED | `ticks_base + K * ticks_per_active_node` |
| PERMUTE | `ticks_base + L * ticks_per_total_node` |
| LOWRANK | `ticks_base + L * ticks_per_total_node` |
| TAIL | Sum of tail op costs |
| IDWPT | `BASE_IDWPT + L * LIFTING_OPS` |
| MIX_GAIN | `BLOCK_SAMPLES * 4` |

---

## Appendix B: Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2025-12-26 | Initial draft |
