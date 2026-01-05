# CCS Implementation Status

**Last Updated**: 2026-01-05
**Status**: ❌ NOT FUNCTIONAL

---

## Current State

### Verified Test Results (2026-01-05)

```
LUT target frequency: 440.00 Hz
Single block peak frequency: 750.0 Hz    ← WRONG!
Multi-block peak frequency: 307.50 Hz    ← WRONG!
VM output fundamental: 311.25 Hz         ← WRONG!
Non-harmonic peaks: 1431                 ← SEVERE CONTAMINATION
Energy in ALL 64 leaves (should be concentrated)
```

**The CCS system produces WRONG FREQUENCIES with massive spectral contamination.**

---

## Compliance Against WSE-STD-ARCH-CCS.md

| Requirement | Section | Status | Notes |
|-------------|---------|--------|-------|
| No FWPT in voice path | 2.1 | ❌ | `coeff_lut_streaming.py` uses `fwpt_jit()` for table generation |
| CoeffTable maps phase → 256 coeffs | 3.2 | ❌ | Current tables map phase → 64 coeffs (wrong dimension) |
| IDWPT reconstructs target waveform | 2.1 | ❌ | Produces 750 Hz instead of target frequency |
| Phase-continuous operation | A3 | ✅ | Phase accumulator is continuous |
| Operators modify coefficients | 2.1 | ⚠️ | Operators exist but operate on wrong coefficients |

---

## Compliance Against WSE-STD-120.md

| Requirement | Section | Status | Notes |
|-------------|---------|--------|-------|
| IDWPT as synthesis operator | 2.2 | ✅ | `lane_ccs_render()` calls `idwpt_jit()` |
| Persistent coefficient state | 4.1 | ⚠️ | State persists but carrier regenerates each block |
| Coefficient state continuous | Invariant 2 | ❌ | Coefficients derived from broken FWPT tables |
| No per-block reanalysis | Invariant 4 | ❌ | Tables built via FWPT analysis |
| No safety window for correctness | Invariant 5 | ⚠️ | Block continuity passes (ratio < 3.0) but wrong frequency |

---

## Root Cause Analysis

### The Problem

FWPT decomposes audio into subband **ENERGY**, not frequency content. When IDWPT reconstructs from these coefficients, it produces audio at the filterbank's intrinsic center frequencies (750 Hz = 48000/64), **not** the original signal frequency.

### Current (BROKEN) Implementation

```
Audio(440 Hz) → FWPT → coefficients → table[phase] → IDWPT → Audio(750 Hz!)
```

**What happens:**
1. FWPT decomposes 440 Hz audio into subband energy distribution
2. Energy spreads across multiple subbands
3. When IDWPT reconstructs, it produces audio at filterbank center frequencies
4. The 750 Hz = 48000/64 is the filterbank's fundamental frequency

### The Key Insight

From WSE-STD-120 and the original gap analysis:

> **Wavelet coefficients are NOT spectral snapshots. They are TIME-DOMAIN SUBBAND SIGNALS.**

The specification says:
- "IDWPT SHALL NOT be driven by per-block reanalysis of a carrier"
- Using FWPT for offline table generation IS still "reanalysis" - it doesn't preserve frequency

### The Correct Approach

Compute coefficients that, when fed to IDWPT, produce the target waveform directly. Do NOT use FWPT for this. Methods include:

1. **Iterative Optimization**: Find coefficients C such that `IDWPT(C) ≈ target_audio` via gradient descent
2. **Analytic Filterbank Response**: Compute impulse response of each IDWPT leaf, set coefficients for constructive interference at target frequency
3. **True Streaming Subband Oscillator**: Per CCS-ARCHITECTURE-GAP.md maximal solution

---

## Recommended Fix Options

### Option A: Disable CCS (Short Term)
- Set `use_coeff_osc = False` in decode.py
- Legacy path has combing but produces CORRECT frequencies
- Immediate fix, audible

### Option B: Fix Table Generation (Medium Term)
- Replace FWPT-based table generation with iterative optimization
- Compute coefficients that reconstruct to target waveforms
- Requires understanding IDWPT synthesis filterbank behavior

### Option C: Full Streaming Rewrite (Long Term)
- Implement "Maximal Solution": true streaming coefficient-domain oscillator
- Polyphase IDWPT (streaming synthesis filterbank)
- Offline subband LUT generation (NOT FWPT-based)
- True streaming coefficient evolution

---

## Implementation Files

### Core CCS Files
- `src/symcrash_python_ap/vm/coeff_lut_streaming.py` - **BROKEN** table generation using FWPT
- `src/symcrash_python_ap/vm/lanes.py` - `lane_ccs_render()` polyphase IDWPT
- `src/symcrash_python_ap/vm/executor.py` - CCS path in VM execution

### Related Files
- `src/symcrash_python_ap/vm/coeff_osc.py` - Coefficient oscillator state
- `src/symcrash_python_ap/wavelet/fast_lifting.py` - `fwpt_jit()` and `idwpt_jit()`
- `tools/verify_sine_output.py` - Verification script that exposed the problem

---

## Specification Documents

These WSE-STD documents define the correct CCS architecture:

- **WSE-STD-ARCH-CCS.md** - Main CCS architecture specification
- **WSE-STD-120.md** - Polyphase upgrade specification

---

## History

| Date | Event |
|------|-------|
| 2026-01-03 | CCS implementation claimed "complete" |
| 2026-01-05 | Verification reveals wrong frequencies (750 Hz instead of 440 Hz) |
| 2026-01-05 | 1431 non-harmonic artifacts detected |
| 2026-01-05 | Root cause confirmed: FWPT table generation doesn't preserve frequency |
| 2026-01-05 | Documentation consolidated into this status document |

---

## Bottom Line

**CCS is NOT working correctly.** The implementation produces audio at the wrong frequencies with severe spectral contamination.

Until the table generation is fixed, **disable CCS and use the legacy path** which produces correct frequencies (with combing artifacts that are far less severe than wrong pitch).
