# WSE-STD-070A — CCS Polyphase IDWPT (Coefficient-Driven Synthesis)
**Status:** Draft v0.2 (gap-closure spec)
**Scope:** Voice synthesis path (CCS), coefficient representation, polyphase IDWPT rendering
**Applies to:** Python reference + FPGA mapping

---

## 0. Problem this spec closes

We kept "running into problems" because we were mixing two incompatible mental models:

1) **Block-analysis model (old):**
- Take a carrier audio block → FWPT → per-block coefficients → modify → IDWPT.
- This is *not* shift/phase invariant, so the same "sound" depends on block alignment.
- Masking/modulating coefficients produces **phase-dependent combing** and **block-edge artifacts**.

2) **Coefficient-synthesis model (CCS):**
- Coefficients are **first-class signals** that evolve continuously.
- IDWPT is used as a **synthesis decoder**, not an "undo analysis" step.

The missing technical detail was: **how a block IDWPT can be used in a streaming voice engine** without resetting state or introducing block-rate artifacts.

This spec defines the *approved* streaming-compatible method:

> **Polyphase-interleaved block IDWPT** driven by **K-slot coefficient state**.

---

## 1. Definitions

- **B (block size):** 64 audio samples.
- **L (leaf count):** 64 leaves (balanced depth-6 WPT).
- **K (polyphase slot count):** VM default **K = 4**.
- **Coeff state:** `c[l, k]` where `l ∈ [0..63]`, `k ∈ [0..K-1]`.

### 1.1 Coeff state layout (normative)

The VM stores coefficient state flattened leaf-major:

```
coeff_state_flat[l*K + k]  <=>  c[l, k]
```

So with K=4, `coeff_state_flat` has 256 entries.

---

## 2. Why IDWPT can "produce all audio" in CCS

In CCS, the forward transform is **not** the source of coefficients.

- The coefficient oscillator (CoeffLUT) and operators generate/shape `c[l, k]` directly.
- **IDWPT is the synthesis filterbank**: it maps coefficient space → time-domain samples.

So it is expected—and *intended*—that IDWPT produces complete audio even if no FWPT ran.

---

## 3. Polyphase-interleaved block IDWPT (normative)

### 3.1 Rendering rule

For each audio block, render K reconstructions:

```
for k in 0..K-1:
    y_k[0..B-1] = IDWPT( c[:, k] )
```

Then interleave:

```
output[n] = y_{n mod K}[n]
```

Equivalently: for each k, take only the samples aligned to that phase slot:

```
output[k], output[k+K], output[k+2K], ... are copied from y_k[k], y_k[k+K], y_k[k+2K], ...
```

### 3.2 Interpretation

This is a constrained polyphase synthesis network:
- each `c[:, k]` represents the coefficient "view" at sub-phase position `k`
- selecting `y_{n mod K}` eliminates systematic "bad boundary samples" that appear when using a single block-local reconstruction

This provides streaming-compatible behavior **without** implementing a full sample-by-sample lifting pipeline.

---

## 4. CoeffLUT runtime determinism (normative)

The runtime lookup `get_subband_samples(phase_32)` MUST be:
- fixed-point only
- deterministic across platforms

Phase interpolation MUST use integer math:

```
val = v0 + ((v1 - v0) * frac + 2^(F-1)) >> F
```

Where `frac` is an F-bit fraction (VM uses F=26 for phase interpolation).

(Offline table generation may use floating point.)

---

## 5. Pipeline contract (normative)

The CCS voice path is:

1. **Carrier injection into coeff_state** (optional blend with prior coeff_state)
2. **Event injection** (lane_inject)
3. **Operators** (backbone)
4. **Tail** (clamp/normalize/project)
5. **CCS render** (polyphase-interleaved IDWPT)
6. **Mix** (residual blend or latency-aligned blend)

Dry/wet must be latency-aligned or computed as residual:

```
output = dry + eta * (wet - dry)
```

---

## 6. Acceptance tests (normative)

An implementation MUST pass:

1) **Boundary ratio test**
Compute RMS of (samples at block boundaries) / (samples within block). Must be < 3.0x.

2) **Static coefficient test**
Holding `c[l,k]` constant yields steady output (no block-rate AM).

3) **Slow ramp test**
Linearly ramping coefficients does not zipper at block rate.

4) **Determinism test**
Same input frames → bit-identical output across runs.

---

## 7. Non-goals

- This spec does **not** claim the WPT is globally shift-invariant.
- This spec does **not** require overcomplete/undecimated WPT.
- This spec does **not** require a true single-sample streaming lifting network (that would be WSE-STD-070B).

---

## 8. Reference implementations

- CoeffLUT fixed-point runtime interpolation: `coeff_lut_streaming.StreamingCoeffLUT.get_subband_samples`
- Polyphase IDWPT rendering: `lanes.lane_ccs_render` and `wavelet.idwpt_streaming.idwpt_polyphase_block`
- VM coefficient generation: `coeff_lut_streaming.ccs_generate_coeffs_polyphase`
