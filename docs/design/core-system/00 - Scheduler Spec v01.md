# Scheduler Spec v0 — OrangeCrab (ECP5)

**Canonical, normative, one-page** (block-synchronous, deterministic, “maximal instrument”)

**Goal:** Each audio block, run the highest effective Tower level (L5→L0) the FPGA can afford **without clicks, without divergence**, and with **tail always enforced**.

---

## 0) Symbols and constants

- Block size: $B = 64$ samples (block tick rate = $F_s/B$)
- Voices: $V = 0..V_{MAX}-1$ (e.g. 128)
- Tiles: fixed partition of voices into tiles of size $T$ (recommended $T=16$)
    - $tile\_id(v) = v \gg \log_2(T)$
    - $tile\_off(v) = v \& (T-1)$
- Budgets per block:
    - $B_{compute\_block}$ (units)
    - $B_{mem\_block}$ (units)
- Tail set (mandatory, fixed order):  
  FineClamp(2) → NormalizeScale(11) → (GluePair(12) optional) → SheafProject(23) → EmergencySimplify(19 if triggered)
- Phases: Phase A preconditioning, Phase B creative, Phase C tail

---

## 1) Inputs per block

For each voice $v$, we have $VoiceFramePacket[v]$ (136 bytes) already parsed.

- If $reject\_code \neq 0$: treat as **no-op frame** for scheduling and execution.
- Extract for eligibility:
    - $gain_{q15}$, $activity_{q15}$, $flags.gate$, $program\_id$, $voice\_id$
    - $degradation\_level$ (advisory only; scheduler decides global D)
    - requested features implied by frame (families/events)

---

## 2) Health telemetry (scheduler-owned, observational)

Scheduler maintains per block:

- $used\_compute$, $used\_mem$ (actual executed costs)
- $cpu\_bar = used\_compute / B_{compute\_block}$ (fixed-point output)
- $mem\_bar = used\_mem / B_{mem\_block}$
- $stab\_bar$ (derived from tail metrics, see §8)
- $gear = 5 - D$ (or any monotone mapping; normative: derived from D)

**Rule:** health bars MUST NOT affect decisions in the same block except via $D$ computed at block start (see §4). Bars are outputs.

---

## 3) Determinism rules (hard)

If a voice $v$ is **not serviced** this block:

- $c[v]$ (coeff state) holds (or cloud-decays if configured)
- **do not** advance: PRNG, regime, overlays, dwell timers, emission counters
- wavelet synthesis still renders from last $c[v]$ (audio continuous)

If a voice’s frame is rejected ($reject\_code \neq 0$):

- it is treated as inactive and not serviced
- no state advances

---

## 4) Global degrade state D (maximal instrument policy)

Compute combined pressure from the **previous** block telemetry:

- $p = \max(cpu\_bar, mem\_bar)$ ($q15$)

Map to degrade state $D \in \{0,1,2,3,4\}$ with hysteresis:

Thresholds ($q15$):

- $D=0$ if $p < 0.70$
- $D=1$ if $0.70 \leq p < 0.85$
- $D=2$ if $0.85 \leq p < 1.00$
- $D=3$ if $1.00 \leq p < 1.15$
- $D=4$ if $p \geq 1.15$

Hysteresis rule:

- $D$ may change by at most $\pm1$ per block.
- $D$ only increases immediately; it decreases only after $H_{down}$ consecutive blocks below the lower threshold (recommend $H_{down}=4$).

**This makes the instrument “maximal” but stable.**

---

## 5) Weighted frequency basis via tiled window counters (Top/Mid)

Define emission groups:

- **MID** = L2–L3 work (BANDED + Diffuse, LOWRANK, richer events)
- **TOP** = L4–L5 work (chaos overlays, catastrophe corridor features, dense patterning)

Tail is not throttled; it is service-gated only.

### Degrade→period table (normative)

Select periods $K_{mid}$, $K_{top}$ from $D$:

| D | K_top | K_mid |
|--:|--:|--:|
| 0 | 1 | 1 |
| 1 | 2 | 1 |
| 2 | 4 | 2 |
| 3 | 8 | 4 |
| 4 | 16 (or suppressed) | 8 |

**Suppression at $D=4$:** TOP may be suppressed entirely if $stab\_bar$ is red (see §8). Suppression must be deterministic.

### Tile counters

Per tile $t$, maintain:

- $ctr_{mid}[t]$ modulo $K_{mid}$
- $ctr_{top}[t]$ modulo $K_{top}$

Update rule:

- A tile counter advances **only if at least one serviced voice in that tile exists this block**:
    - $ctr_X[t] \leftarrow (ctr_X[t] + 1) \mod K_X$

Phase rule (per voice, deterministic):

- $phase_X(v) = tile\_off(v) \& (K_X - 1)$ (K is power of two)

Emission due rule for a serviced voice $v$ in tile $t$:

- $MID_{due}(v) = ((ctr_{mid}[t] + phase_{mid}(v)) \& (K_{mid}-1)) == 0$
- $TOP_{due}(v) = ((ctr_{top}[t] + phase_{top}(v)) \& (K_{top}-1)) == 0$

---

## 6) Voice eligibility and service selection (ring scheduler)

Define a voice as **eligible foreground** if:

- frame accepted ($reject\_code==0$) AND
- ($flags.gate==1$ OR $activity_{q15} > \theta_{active}$)  
  Recommended $\theta_{active} = 0.05$ ($q15$)

Maintain a ring cursor $cursor_{fg}$ over $[0..V_{MAX}-1]$.

### Tail reservation rule

Let $C_{tail\_voice}$ be fixed cost for mandatory tail (and GluePair if enabled by policy).

Select $V_{serviced}$ such that:

- $V_{serviced} \geq 0$
- $V_{serviced} \cdot C_{tail\_voice} \leq \min(B_{compute\_block}, B_{mem\_block})$ **(tail must fit)**
- choose maximal $V_{serviced}$ subject to this constraint and a configured max $V_{serviced\_max}$

### Selection algorithm (deterministic)

At BLOCK_START:

1. Initialize empty $serviced\_set$
2. Scan voices starting at $cursor_{fg}$, wrapping, until either:
    - $len(serviced\_set) == V_{serviced}$, or
    - scanned $V_{MAX}$ voices
3. Add a voice if it is eligible foreground.
4. Set $cursor_{fg}$ to the next index after the last scanned voice.

---

## 7) Per-voice micro-scheduler (what runs when serviced)

For each serviced voice $v$, the symbol executor runs:

### Phase A (always for serviced voices)

Preconditioning, fixed order:

1. PERMUTE $\Pi$: $\{9,10,17,6\}$ in ascending $op\_id$
2. DIAG/S: $\{1,15,20,22\}$ in ascending $op\_id$

### Phase B (budgeted, gated by due flags)

Compute:

- if $TOP_{due}(v)$ then allow TOP features (L4–L5 set)
- if $MID_{due}(v)$ then allow MID features (L2–L3 set)
- otherwise Phase B may still run **minimal** injections required for continuity (L0/L1)

Normative degradation ladder inside Phase B (applies when Phase B cost exceeds per-voice core budget):

1. LOWRANK down-rank (2→1→0)
2. Diffuse frequency already handled by $K_{mid}$; additionally may mask leaves
3. Reduce BANDED scope (mask bands)
4. Reduce injection density (fewer targets)
5. Omit creative Phase B entirely

### Phase C Tail (always for serviced voices)

Fixed, no exceptions:  
2 → 11 → (12 optional) → 23 → (19 if triggered)

**EmergencySimplify trigger** is based on tail stability metrics (energy/clamp conditions) and is deterministic.

---

## 8) Stability metrics and TOP suppression rule

Each serviced voice reports tail stats (simple counters):

- $clamp\_hits$
- $norm\_gain\_extreme$ (boolean)
- $emergency\_simplify\_fired$ (boolean)

Scheduler aggregates:

- $stab\_bar$ decreases with $clamp\_hits$ and emergency events.

**TOP suppression (deterministic):**

- If $D==4$ AND $stab\_bar < \theta_{stab\_red}$ then $TOP_{due}(v)=false$ for all voices this block (TOP suppressed).
- This suppression MUST NOT advance any TOP-related PRNG or regime state.

---

## 9) Accounting (compute + mem)

Maintain running totals this block:

- $used\_compute += cost(op\ executed)$
- $used\_mem += cost\_mem(op\ executed)$

Costs use the v0 model (unit-based) and are counted for:

- Phase A ops executed
- Phase B ops executed (after degradations)
- Phase C tail ops executed

At BLOCK_END:

- compute health bars from used totals
- update D hysteresis counters
- advance tile counters only for tiles with $\geq1$ serviced voice

---

## 10) Outputs

Per block:

- $serviced\_set$ (implicit; for debug)
- $cpu\_bar$, $mem\_bar$, $stab\_bar$, $gear$
- updated $cursor_{fg}$, $ctr_{mid}[t]$, $ctr_{top}[t]$
- updated per-voice coeff state for serviced voices only

**Non-serviced voices retain prior coeff state and are still rendered by wavelet.**

---

### Implementation notes (non-normative)

- The above is compatible with 1-lane or multi-lane symbol pipelines.
- If multi-lane, $serviced\_set$ may be processed in parallel, but ordering must not affect results; enforce deterministic per-voice cost accounting.

---

If you want, next I can produce the **matching FSM diagram** (BLOCK_START → SELECT → SERVICE_NEXT → BLOCK_END) and a compact “register file” list (what state lives per voice vs per tile) tailored to OrangeCrab BRAM/LUT expectations.