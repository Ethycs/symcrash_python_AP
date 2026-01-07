
# Symbol Executor v0 — Phased Schedule + Tail + Budget (Normative)

Locked. This is consistent with the packet ABI, the tower IR compiler, and the “instrument never dies” constraint. Here’s how I’d fold what you wrote into the **canonical spec** (normative), with just enough tightening so Python SIM + FPGA can’t diverge.

---

## 1) Three-phase execution model

Every serviced voice, every block, runs **exactly**:

- **Phase A — Preconditioning (always run, cheap)**
- **Phase B — Creative core (budgeted / degradable)**
- **Phase C — Tail (always run, fixed order, no exceptions)**

This schedule is independent of author order. Authoring tools may _present_ order, but the executor enforces the phases.

---

## 2) Operator classification (normative tags)

Each operator declares:

- $class \in \{ \text{DIAG}, \text{BANDED}, \text{PERMUTE}, \text{LOWRANK}, \text{INJECT}, \text{PROJ} \}$
- $tail\_rank \in \{ \text{NONE}, \text{SOFT\_TAIL}, \text{HARD\_TAIL} \}$
- $cost\_model$ (see §4)

### Tail set (hard requirement)

**HARD_TAIL** (must run in Phase C, fixed order):

- 2 FineClamp
- 11 NormalizeScale
- 23 SheafProject
- 19 EmergencySimplify (conditional trigger, but if triggered it runs here)

**SOFT_TAIL** (Phase C, placed before SheafProject; may be omitted under policy):

- 12 GluePair (if enabled / requested)
- 16 GateMask _only if used as constraint; if used musically it stays in creative (see below)_

Everything else is **NON-TAIL**.

> Your earlier directive stands: tail set applies for Safety, **no exceptions**.

---

## 3) Normative execution order

### Phase A — Preconditioning (always)

Order is fixed:

1. **PERMUTE group $\Pi$**  
   SwapBands (9), RotateNodes (10), QuasiRotate (17), BandShear (6)
   - applied in **ascending op_id** (or a fixed $\Pi$-order table; pick one and keep it forever)

2. **DIAG / S group**  
   Tilt (1), PressureRamp (15), FoldPressure (20), SwallowtailMask (22)
   - applied in **ascending op_id**

_(If you later add more DIAG “coordinate setting” ops, they belong here.)_

### Phase B — Creative core (budgeted)

Default stable order:

3. **BANDED mixing/transport**  
   IntraMix (3) then DiffuseUp (4) / DiffuseDown (5) (ascending op_id within class)

4. **LOWRANK coherence**  
   LowRankMorph (7) (and any future lowrank ops)

5. **INJECT events**  
   Event ops: GlobalPulse (8), Impulse (13), NoiseBand (14), CuspInjection (21)
   - applied **after** BANDED/LOWRANK so impacts “hit” a stable regime

### Phase C — Tail (always; fixed; no exceptions)

6. FineClamp (2)
7. NormalizeScale (11)
8. GluePair (12) _(if enabled and not dropped by policy)_
9. SheafProject (23)
10. EmergencySimplify (19) _(only if triggered; may override output deterministically)_

---

## 3.1 Event lane splice (from the packet)

Event lane dispatch maps into the phases:

- toggles ($EVT\_OP\_TOGGLE$) apply **before Phase A** (update enable mask)
- injections ($EVT\_INJECT\_NOTE$) contribute to $b_t$ and are applied in **Phase B INJECT**
- pulses ($EVT\_OP\_PULSE$) are INJECT-class and also run in **Phase B INJECT**
- tree morph and K mode are routed out of the Symbol executor (Wavelet/Material)

---

## 4) Budget model (two budgets, tail reserved)

Track two budgets per block:

- $B_{compute}$ (MAC-ish budget)
- $B_{mem}$ (read/write bandwidth budget)

Each operator has a cost estimate $(c_{compute}, c_{mem})$ computed from:

- $N = 256$ coeff (32 leaves × 8)
- rank $r$ for lowrank
- target count for inject

### v0 cost units (starting point)

Use these as the normative “accounting units” (tunable, but fixed for a given build):

- **DIAG**: $(1N, 1N)$
- **PERMUTE**: $(0.5N, 2N)$ _(more memory than compute)_
- **BANDED (tri-diag)**: $(3N, 3N)$
- **DiffuseUp/Down**: $(4N, 4N)$
- **LOWRANK$\,r$**: $(2rN, 2rN)$
- **FineClamp**: $(1N, 1N)$
- **NormalizeScale**: $(2N, 2N)$ _(energy calc + scale)_
- **SheafProject**: $(3N, 3N)$ _(start here)_
- **INJECT**: $(k, k)$ where $k = \#$target writes (small)

### Tail reservation (normative)

For each serviced voice, reserve tail cost **first**:

$$
C_{tail} = cost(2) + cost(11) + cost(23) + \text{optional } cost(12)
$$

Compute maximum core allowed:

$$
B_{core} = B_{total} - V_{serviced} \times C_{tail}
$$

Tail must always fit. If it can’t, you must reduce $V_{serviced}$ (service fewer voices), not drop tail.

---

## 5) Degradation strategy (deterministic priority)

Under budget pressure degrade in this exact order:

1. **LOWRANK down-rank** ($r=2 \rightarrow 1 \rightarrow 0$)
2. **Reduce diffusion frequency** (run DiffuseUp/Down every 2 or 4 blocks)
3. **Reduce banded scope** (mask bands/leaves)
4. **Reduce injection density** (fewer targets, lower event rate)
5. **Freeze voices** (foreground freeze; cloud decay/freeze)
6. **Never drop Phase C tail**
7. **EmergencySimplify triggers** if stability metrics exceed threshold

---

## 6) Determinism rule when skipping voices (critical)

If a voice is not serviced for budget reasons:

- do **not** advance its PRNG
- do **not** advance its regime graph
- do **not** advance timing counters / envelopes (unless you explicitly choose “envelope continues while coeff frozen” — pick one)
- apply only a deterministic hold/decay policy (per instrument category)

This ensures playback is stable under variable load and matches your earlier CMMM determinism requirement.

---

## 7) What I still need to finalize numbers (optional)

You mentioned ECP5 + 64-sample blocks @ 48k (750 Hz). To give you a realistic initial $B_{compute}/B_{mem}$ and “max voices serviced per block” policy, I’d need only one implementation fact:

- Are you **time-multiplexing** a single symbol pipeline across voices, or do you have **multiple parallel lanes** (e.g., 2–4 voices processed concurrently)?

If you don’t answer, the spec above still stands; the budget constants just remain implementation-chosen.

---

If you want the “one-page” version of _this_ (like the packet diagram), I can compress the three-phase schedule + tail order + degradation ladder into a single diagram block you can paste into the master doc right next to the VoiceFrame packet layout.