# Implementation Reality Check

## Current Status: CCS Not Working Correctly

### The Core Problem

**Coefficients from FWPT of a sinusoid are NOT suitable for persistent state!**

FFT shows:
- DC signal (stuck at 13k-15k, no oscillation)
- Perfect 2044-sample jumps every block (phase discontinuity)
- 750 Hz combing artifacts (same as legacy!)

### Why FWPT Coefficients Don't Work for Persistence

FWPT analyzes a 64-sample AUDIO block and extracts frequency content.
- Input: Time-domain sinusoid (64 samples)
- Output: 64 frequency-band coefficients (snapshot)

These coefficients represent **"how much energy is in each band for THIS specific 64-sample window"**.

When we regenerate them each block:
- Block N: FWPT(sin(phase))
- Block N+1: FWPT(sin(phase + delta))
- Coefficients jump discontinuously!

### What WSE-STD-120 Actually Requires

Line 97-109: `coeff_state[v][leaf][k]` where k=0..K-1 are **time samples in that subband**!

Not frequency snapshots, but **time-series data**!

Example for leaf 5 (some frequency band):
```
coeff[5][0] = subband_signal[t=0]
coeff[5][1] = subband_signal[t=1]
...
coeff[5][7] = subband_signal[t=7]
```

These should vary smoothly as the subband signal evolves!

### The Correct Architecture (Not Yet Implemented)

**Per STANDARD-PATCH.MD Mode A** (lines 30-34):
```
For each coefficient i:
    phi_i += phase_inc * stride_i
    coeff[i] = mag[i] * sin(phi_i)
```

This generates SUBBAND OSCILLATORS, not full-band snapshots!

Each coefficient oscillates at its own rate, representing that frequency band's time evolution.

## What We Have vs. What We Need

### Current (WRONG):
```
Block N:
  audio_ref = generate_sine(phase_N)
  coeffs = FWPT(audio_ref)  ← Snapshot!
  IDWPT(coeffs) → audio
```
Coefficients jump each block!

### Required (CORRECT):
```
Initialize:
  For each leaf i:
    phi[i] = initial_phase
    mag[i], stride[i] from table

Each block:
  For each leaf i:
    For each time sample k in [0..3]:
      phi[i] += delta_phi * stride[i]
      coeff[i][k] = mag[i] * sin(phi[i])

  IDWPT_streaming(coeffs[leaf][k]) → audio
```
Coefficients evolve smoothly via phase!

## Recommendation

**Option A: Revert to Legacy Path (Short-term)**
- Legacy path works (has combing but produces sound)
- Fix the combing with better OLA/windowing
- Accept FWPT in path as necessary cost

**Option B: Implement True Mode A (Medium-term)**
- Per-leaf phase accumulators (64 or 32+8)
- Generate coefficient TIME-SERIES, not snapshots
- Implement true polyphase IDWPT
- This is substantial work (weeks)

**Option C: Hybrid Approach (Pragmatic)**
- Keep legacy path as default
- Mark CCS as experimental/incomplete
- Document what's needed for full implementation
- Revisit when streaming polyphase is prioritized

## My Recommendation

**Disable CCS (use_coeff_osc=False by default) and document the gap.**

The current CCS produces worse artifacts than legacy! It has:
- DC bias
- Block discontinuities (2044 samples!)
- Same 750Hz combing
- Plus instability from tail normalization

Legacy path is actually BETTER right now - it at least produces oscillating audio!

**Should I:**
1. Disable CCS and revert to legacy as default?
2. Continue implementing true Mode A with per-leaf oscillators?
3. Document current state and move to other priorities?
