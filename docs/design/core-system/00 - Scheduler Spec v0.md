# Scheduler Spec v0 — Graceful Degrade by Tower Level and Weighted Frequency Basis

## 1) Principle

When compute/memory pressure exceeds budget, the system MUST degrade behavior by **reducing the execution frequency of higher-level (L5) features first**, while preserving:

- continuous audio output (wavelet always renders last state)
- mandatory tail legality (Phase C always runs on serviced voices)
- determinism (no hidden state changes on skipped work)

Degradation is expressed as a **frequency throttle** (“run this feature every $K$ blocks”) chosen from a **weighted basis** rather than a binary enable/disable.

---

## 2) Tower Feature Groups (execution domains)

Define five degradable feature groups, mapped to Tower levels:

| Group | Tower Level | Examples | Scheduler lever |
|---|--:|---|---|
| G0 | L0 | ADSR gain, basic families, minimal injection | never degraded (unless voice skipped) |
| G1 | L1 | simple regime switching, basic events | throttle lightly |
| G2 | L2 | diffusion/banded mixing, simple topology ops | throttle/modulate scope |
| G3 | L3 | lowrank coherence, richer event patterns | down-rank + throttle |
| G4 | L4/L5 | chaos overlays, catastrophe corridor, dense patterning | throttle heavily, then suppress |

**Mandatory tail** is not degradable and is outside this table.

---

## 3) Weighted frequency basis

Each group $G_k$ has an execution frequency selector $K_k \in \{1,2,4,8,16\}$ blocks.

Interpretation:

- $K=1$ means execute every block
- $K=2$ means execute on even blocks
- $K=4$ means execute when $(block\_index \bmod 4)==phase$
- etc.

### Deterministic phase assignment

Each group has a fixed phase offset per voice:

$$
phase_{v,k} = \text{hash}(voice\_id, k) \bmod K_k
$$

This prevents all voices aligning and causing bursty load.

---

## 4) Degradation decision is global and monotone per block

Define global pressure metrics computed by scheduler after budget accounting:

- $cpu\_pressure = used\_compute / B\_{compute\_block}$
- $mem\_pressure = used\_mem / B\_{mem\_block}$

Define combined pressure:

$$
p = \max(cpu\_pressure, mem\_pressure)
$$

Then compute a **degrade state** $D \in \{0,1,2,3,4\}$:

- $D=0$ if $p < 0.70$
- $D=1$ if $0.70 \leq p < 0.85$
- $D=2$ if $0.85 \leq p < 1.00$
- $D=3$ if $1.00 \leq p < 1.15$
- $D=4$ if $p \geq 1.15$

$D$ may be smoothed for UI, but the scheduling decision uses the instantaneous $D$ (deterministic).

---

## 5) Level downgrade table (L5→L0) via frequency throttles

Map degrade state $D$ to throttles $K_k$:

| Degrade D | G4 (L4/L5) | G3 (L3) | G2 (L2) | G1 (L1) | G0 (L0) |
|--:|--:|--:|--:|--:|--:|
| 0 | 1 | 1 | 1 | 1 | 1 |
| 1 | 2 | 1 | 1 | 1 | 1 |
| 2 | 4 | 2 | 1 | 1 | 1 |
| 3 | 8 | 4 | 2 | 1 | 1 |
| 4 | 16 (or 0*) | 8 | 4 | 2 | 1 |

*At $D=4$, you MAY set $G4$ to **suppressed** (never executed) if stability is at risk, but suppression must be explicit and deterministic (see §7).

This is your “graceful degrade L5→L0 on weighted freq basis.”

---

## 6) What “executing a group” means (per voice, per block)

When a voice is serviced this block, the executor decides per group:

A group $G_k$ executes iff:

1. voice is serviced, AND
2. $(block\_index \bmod K_k) == phase_{v,k}$

### Group-to-operator mapping (normative v0)

- **G2**: BANDED + DiffuseUp/Down
- **G3**: LowRankMorph (rank may also be reduced under D)
- **G4**: chaos overlay effects, catastrophe corridor ops/events, topology-heavy permutations beyond Phase A defaults
- **G1**: event emission (non-critical), regime transitions beyond minimal
- **G0**: gain/ADSR + base families + minimal injection required for note-on continuity

**Phase A** (preconditioning) is not “a group”: it runs whenever a voice is serviced (cheap).

**Phase C tail** always runs whenever a voice is serviced (non-negotiable).

---

## 7) Suppression rules (hard safety gates)

Even if a group is scheduled to execute, it can be suppressed deterministically:

- If SOFT overlay active → catastrophe-related ops/events suppressed (your rule)
- If Stability margin is red (e.g., clamp hits exceed threshold) → suppress **G4 first**, then **G3** if needed
- Tail still runs regardless

Suppression must not advance PRNG/regime state that would have been advanced by those features.

---

## 8) Interaction with voice skipping (service scheduler)

The frequency throttles apply **only to voices that are serviced**.

If a voice is not serviced:

- No group executes (obvious)
- No PRNG/regime/timer advance (your determinism rule)
- Coefficients hold/decay per category
- Wavelet still renders from last coeffs

So the scheduler has two knobs:

1. how many voices are serviced ($V_{serviced}$)
2. for those voices, which groups run by frequency throttles ($K_k$)

This gives very smooth degradation.

---

## 9) Bonus: “weighted” basis using actual cost (optional refinement)

If you want the basis to be truly cost-aware, define group weights:

- $w_4 > w_3 > w_2 > w_1 > w_0$

and choose throttles by minimizing:  
$$
\sum_k w_k \cdot \mathbf{1}[\text{group executes}]
$$  
subject to budgets.

But the table in §5 is already good and FPGA-simple.

---

If you want, I’ll now write the **full one-page Scheduler Spec v0** in the same canonical style as the packet diagram, with this degradation policy embedded and with explicit FSM states (BLOCK_START → SELECT_VOICES → SERVICE_VOICE → DONE).