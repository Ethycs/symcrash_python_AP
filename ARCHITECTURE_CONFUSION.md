# Architecture Confusion Analysis

## The Fundamental Question

**What does "coefficient-domain carrier" actually mean?**

## Three Possible Interpretations

### Interpretation 1: Coefficients Oscillate (Mode A as currently implemented)
```
coeff[i] = mag[i] * sin(phi[i])
phi[i] advances each block
→ IDWPT(coeffs) → audio
```

**Problem**: IDWPT of oscillating coefficients ≠ oscillating audio!
- Single coeff[0] oscillating → produces varying DC offset (not sinusoid!)
- Test shows: coeff[0]=10000 → audio = constant 20000 (DC)
- Coefficient space != audio space

**Status**: TESTED, DOESN'T WORK

### Interpretation 2: 2D Table Lookup (phase → coefficients)
```
c_base = CoeffTable[note][phase]  ← Lookup 64 coeffs
→ blend with c_state
→ IDWPT(blended) → audio
```

**Problem**: Block discontinuities!
- Each phase_idx gives a different coefficient snapshot
- No continuity between snapshots
- Tested: 57k sample jumps at boundaries!

**Status**: TESTED, DOESN'T WORK

### Interpretation 3: Additive Synthesis (current working code)
```
For each audio sample:
    audio[s] = Σ mag[h] * sin(phase * h)
    phase advances per sample
```

**Works**: Phase-continuous, no discontinuities
**Problem**: Generates AUDIO, not coefficients
**Status**: WORKING but violates "coefficient-domain" requirement

## What the Specs Actually Say

### WSE-STD-120.md (lines 67-68):
> "IDWPT SHALL be used as a synthesis operator over persistent coefficient state.
> IDWPT SHALL NOT be driven by per-block reanalysis of a carrier"

**Key word: PERSISTENT**

Coefficients are **persistent state** that evolves via operators, not regenerated fresh each block!

### STANDARD-PATCH.MD (lines 30-34):
> For each coefficient i:
>     phi_i += phase_inc * stride_i
>     coeff_base[i] = (mag[i] * sin_lut(phi_i)) >> 15

This suggests **per-coefficient oscillation**, but...

### WSE-ARCH-CCS.md (lines 116-120):
> c^(0)_k = CoeffTable[wave_id](φ_k)
> c_k = Tail(Q(O(c^(0)_k; VoiceFrame_k)))
> x_k = IDWPT(c_k)

This suggests **table lookup**, not oscillation!

## The Real Problem

**These specs are CONTRADICTORY or I'm fundamentally misunderstanding what "wavelet coefficient" means in this context.**

## Critical Questions

1. **What IS a "wavelet coefficient" in this system?**
   - Is it a value in the WPT tree (what FWPT produces)?
   - Or is it some other representation?

2. **How do we get audio from oscillating coefficients?**
   - If coeff[0] oscillates between -10000 and +10000
   - IDWPT should output what?
   - Current behavior: varying DC offset (not sinusoid!)

3. **What was working before?**
   - Old `coeff_osc_render()` generated audio via additive synthesis
   - It was phase-continuous and worked perfectly
   - But it bypassed operators and IDWPT
   - Was this intentionally "wrong" or is there a different understanding?

## Hypothesis: Maybe "Coefficients" ≠ "Wavelet Tree Coefficients"

Maybe in this architecture:
- "coeff_state[256]" is an abstract state space (not WPT tree coefficients)
- Operators work on this abstract space
- IDWPT is misnamed - it's actually a "render function" not inverse DWT
- The 256 values represent something else (harmonic parameters? spectral envelope?)

## What We Need

**User clarification on:**

1. Should we keep the working additive synthesis and just route it through operators somehow?
2. Is there a working example/test that shows the coefficient-domain carrier producing correct audio?
3. What should IDWPT(oscillating_coefficients) actually output?
4. Are the old working tests using a different architecture that we should preserve?

## Immediate Action

Before implementing further, we need to:
1. Find and run ANY working test that uses the coefficient-domain approach successfully
2. Understand what the working legacy path actually does
3. Clarify if "no FWPT" means "no FWPT in voice path" or "no FWPT anywhere including table gen"
