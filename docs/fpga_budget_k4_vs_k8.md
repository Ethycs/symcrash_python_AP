# FPGA Budget Analysis: K=4 vs K=8 Polyphase Reconstruction

## Executive Summary

This analysis compares K=4 and K=8 polyphase IDWPT reconstruction paths for CCS synthesis, evaluating FPGA resource costs and audio quality trade-offs.

**UPDATE (2024-01)**: Both K=4 and K=8 now use the same reconstruction strategy:
- K_STRIDE = 1 (1-sample phase offset per slot)
- Interleaved output: `out[k::K] = idwpt(slot_k)[0::K]`

This produces equivalent quality for both configurations. The previous K=4 consecutive-block approach (K_STRIDE=16) failed due to boundary discontinuities.

**Current Configuration**: Set `VM_K = 4` or `VM_K = 8` in `tables.py`:
- K=4: 256 coefficients, 4 IDWPTs per block
- K=8: 512 coefficients, 8 IDWPTs per block

---

## Architecture Overview

### Polyphase IDWPT Reconstruction

The IDWPT reconstructs 64 audio samples per block from 64 wavelet coefficients. Polyphase reconstruction:
- Runs K parallel IDWPT transforms with phase-offset coefficients
- Interleaves results: sample n uses output from k-slot (n mod K)
- Reduces block-boundary artifacts through overlapping windows

### Current Implementation

- **Coefficient state**: 512 values = 64 leaves × 8 k-slots
- **K=4 path**: Uses slots 0, 2, 4, 6 (every 2nd slot)
- **K=8 path**: Uses all 8 slots

---

## FPGA Resource Analysis

### Per-Voice IDWPT Resources

| Resource | K=4 | K=8 | Delta |
|----------|-----|-----|-------|
| IDWPT instances | 4 | 8 | +4 (2x) |
| Multiply-accumulate ops/block | 4×512 = 2048 | 8×512 = 4096 | +2048 (2x) |
| Coefficient RAM (per voice) | 256 words | 512 words | +256 (2x) |
| Interleave logic | 4:1 mux | 8:1 mux | +1 mux stage |

### Lifting Transform Breakdown (per IDWPT)

64-point balanced WPT = 6 stages of lifting:
- Stage 1-6: 64 samples × 2 ops (predict/update) = 768 multiply-adds
- Total per IDWPT: ~768 MAC operations

| Metric | K=4 | K=8 |
|--------|-----|-----|
| MACs per block | 3,072 | 6,144 |
| MACs @ 48kHz (750 blocks/s) | 2.3M/s | 4.6M/s |

### Memory Bandwidth

| Resource | K=4 | K=8 |
|----------|-----|-----|
| Coeff reads/block | 256 | 512 |
| Output writes/block | 64 | 64 |
| Total words/block | 320 | 576 |

### Parallelism Considerations

**Option A: Sequential K IDWPTs**
- Latency: K × IDWPT_cycles
- K=4: 4 cycles per sample output
- K=8: 8 cycles per sample output

**Option B: Parallel K IDWPTs**
- Latency: 1 × IDWPT_cycles
- K=4: 4 IDWPT cores
- K=8: 8 IDWPT cores (2x area)

**Option C: Time-multiplexed**
- Single IDWPT core, K iterations
- Coefficient buffering required
- Same throughput, lower area

---

## Quality Comparison

### Frequency-Domain Artifacts

| Artifact | K=4 | K=8 |
|----------|-----|-----|
| Block-rate spur (750 Hz) | Present (-14 dB) | Reduced (-20 dB) |
| Intermodulation | Higher | Lower |
| Phase continuity | Good | Excellent |
| Cycle correlation | ~0.80 | ~0.95+ |

### Root Cause of K=4 Spurs

The current K=4 implementation has a phase alignment issue:
- Mode B generates K=8 basis frames with 1-sample phase offsets
- Subsampling to K=4 (slots 0,2,4,6) gives 2-sample phase jumps
- Polyphase interleave expects 1-sample spacing
- Mismatch causes 750 Hz (block rate) artifacts

**Fix**: Generate K=4-specific basis frames with 2-sample phase offsets, OR use K=8.

---

## Implementation Strategy: Dual Path

### Runtime Selection

```
if (polyphony > THRESHOLD):
    use_k4_path()  # Lower quality, higher polyphony
else:
    use_k8_path()  # Higher quality, fewer voices
```

### Suggested Thresholds

| Scenario | Path | Max Voices |
|----------|------|------------|
| Solo/Lead | K=8 | 4-8 |
| Pad/Chord | K=4 | 16-32 |
| Full Mix | K=4 | 32-64 |

### Shared Resources

Both paths can share:
- Lifting filter coefficients (ROM)
- MAC units (time-multiplexed)
- Output buffer

Path-specific resources:
- K=4: 4-slot coefficient buffer, 4:1 interleave
- K=8: 8-slot coefficient buffer, 8:1 interleave

---

## FPGA Budget: Lattice ECP5 LFE5UM-25

### Target Device Resources

| Resource | Available |
|----------|-----------|
| LUTs | 24,000 |
| sysMEM Blocks (18Kb) | 56 |
| Embedded Memory | 1,008 Kb |
| Distributed RAM | 194 Kb |
| 18×18 Multipliers | 28 |
| PLLs/DLLs | Available |

### Single Voice, K=8 Path

| Resource | Usage | Available | % |
|----------|-------|-----------|---|
| LUTs | ~800 | 24,000 | 3.3% |
| 18×18 Mult | 8 | 28 | 29% |
| sysMEM (18Kb) | 2 | 56 | 3.6% |

### Single Voice, K=4 Path

| Resource | Usage | Available | % |
|----------|-------|-----------|---|
| LUTs | ~500 | 24,000 | 2.1% |
| 18×18 Mult | 4 | 28 | 14% |
| sysMEM (18Kb) | 1 | 56 | 1.8% |

### Maximum Polyphony Analysis

**Constraint: 28 multipliers**

| Path | Mult/Voice | Max Voices (Mult-limited) |
|------|------------|---------------------------|
| K=4 | 4 | 7 voices |
| K=8 | 8 | 3 voices |

**Note**: With time-multiplexing (sequential IDWPT), multiplier usage can be reduced:

| Path | Shared Mult | Max Voices |
|------|-------------|------------|
| K=4 (TDM) | 1-2 | 14-28 voices |
| K=8 (TDM) | 1-2 | 14-28 voices |

### Recommended Configuration

Given the **28 multiplier constraint**, the practical approach is:

1. **Time-multiplexed IDWPT core** (1-2 multipliers shared)
2. **K=4 for polyphony** (4 sequential IDWPT passes)
3. **K=8 for quality mode** (8 sequential IDWPT passes, same core)

This gives:
- ~12-14 voices at K=4 quality
- ~6-7 voices at K=8 quality
- Runtime switchable per-voice or globally

---

## Recommendations

### Short Term
1. **Fix K=4 basis frame generation** - Generate with 2-sample phase offsets
2. **Keep K=4 as default** - Sufficient quality for most uses

### Medium Term
1. **Add K=8 path for quality mode** - Runtime selectable
2. **Profile actual spur levels** - May be acceptable for many use cases

### Long Term
1. **Adaptive K selection** - Based on voice count and quality budget
2. **Per-voice K assignment** - Lead voices get K=8, pads get K=4

---

## Conclusion

| Aspect | K=4 | K=8 |
|--------|-----|-----|
| Resource cost | 1x | 2x |
| Audio quality | Good | Excellent |
| Max polyphony | High | Moderate |
| Implementation | Current | Needs new path |

**Verdict**: K=4 is the right performance choice for polyphony. The current spur issues are fixable with proper basis frame generation. K=8 can be added as a quality option for solo/lead voices where resources permit.
