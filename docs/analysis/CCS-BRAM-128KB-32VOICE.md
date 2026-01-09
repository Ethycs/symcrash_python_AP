# Full Synthesis Pipeline BRAM Analysis: 128KB for 32 Voices

## Executive Summary

**Target:** 32 full-featured voices in 128KB BRAM
**Verdict:** Achievable with aggressive optimization. Requires eliminating Python-specific overhead.

| Configuration | BRAM Usage | Fits 128KB? | Notes |
|---------------|------------|-------------|-------|
| Python Reference (full) | **250 KB** | NO (1.95×) | All subsystems |
| FPGA Baseline (K=8) | **106 KB** | YES (83%) | Tight fit |
| FPGA Optimized (K=4) | **74 KB** | YES (58%) | **Recommended** |
| FPGA Minimal (K=2) | **58 KB** | YES (45%) | Maximum headroom |

**Critical insight:** The Python implementation includes per-voice state in WaveletEngine (3.1 KB/voice) and Orchestrator (15 B/voice) that the initial analysis missed. An FPGA can eliminate floats and redundant buffers, but all pipeline components must be accounted for.

---

## Full Pipeline Memory Breakdown

The synthesis pipeline consists of multiple subsystems, each with per-voice and shared state:

```
┌─────────────────────────────────────────────────────────────────────┐
│  FULL PIPELINE MEMORY MAP                                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  VoiceFrameEmitter → BlockScheduler → BlockOrchestrator             │
│        ↓                   ↓                  ↓                      │
│  Control state      Ring cursor       Voice slot state              │
│  (140 B/voice)      (shared 48 B)     (15 B/voice)                  │
│                                                                      │
│  VMState → CCSState/ModeBState → coeff_state                        │
│    ↓              ↓                    ↓                             │
│  Envelope     Phase accum         64L × K × int16                   │
│  PRNG         Wave ID             (512-1024 B/voice)                │
│                                                                      │
│  WaveletEngine (IDWPT Lift)                                          │
│        ↓                                                             │
│  voice_phase_state: 256 floats/voice (2048 B Python, 0 FPGA)        │
│  prev_coeffs: 64L × 8K × int16/voice (1024 B)                       │
│  regularity: 20 B/voice                                              │
│  boundary_state: 20 B/voice                                          │
│                                                                      │
│  Shared Buffers                                                      │
│        ↓                                                             │
│  Waveform LUTs (4-64 KB)                                            │
│  IDWPT lifting (320 B)                                              │
│  Output mixing (772 B)                                              │
│  Sine quarter-wave (2 KB)                                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 1. Per-Voice State (32 voices)

#### Python Reference (Complete Pipeline)

| Subsystem | Component | Size/Voice | 32 Voices | Notes |
|-----------|-----------|------------|-----------|-------|
| **VMState** | coeff_state | 1,024 B | 32 KB | 64L × 8K × int16 |
| | CCSState | 40 B | 1.3 KB | Phase acc, inc, wave_id |
| | ModeBState | 200 B | 6.4 KB | H_active=16, weights, PWM |
| | EnvelopeState | 28 B | 0.9 KB | ADSR tracking |
| | mask_state_q15 | 128 B | 4 KB | 64 leaves × int16 |
| | PRNG state | 4 B | 128 B | xorshift32 seed |
| | wet_ola_tail | 16 B | 512 B | Legacy OLA buffer |
| | prev_fm_mod_buffer | 16 B | 512 B | FM feedback |
| | misc flags/scalars | 52 B | 1.6 KB | Booleans, tracking |
| | **VMState Subtotal** | **1,508 B** | **48.3 KB** | |
| **WaveletEngine** | voice_phase_state | 2,048 B | 64 KB | 256 floats (Python only!) |
| | prev_coeffs | 1,024 B | 32 KB | Change detection |
| | regularity | 20 B | 640 B | Slew limiter state |
| | boundary_state | 20 B | 640 B | Tree transition |
| | voice_tree_state | 12 B | 384 B | Prev tree A/B/blend |
| | **WaveletEngine Subtotal** | **3,124 B** | **100 KB** | |
| **Orchestrator** | VoiceSlot | 15 B | 480 B | State, owner, note, etc. |
| | voice_frames ref | 8 B | 256 B | Pointer (Python) |
| | **Orchestrator Subtotal** | **23 B** | **736 B** | |
| **Emitter** | _envelope | 32 B | 1 KB | EnvelopeScheduler |
| | control modifiers | 48 B | 1.5 KB | Pitch, gain, family |
| | ADSR overrides | 16 B | 512 B | Optional params |
| | misc | 44 B | 1.4 KB | State, counters |
| | **Emitter Subtotal** | **140 B** | **4.5 KB** | |
| **Scheduler** | (shared only) | 0 B | 0 B | Per-tile, not per-voice |
| | **TOTAL PER-VOICE** | **4,795 B** | **153.4 KB** | Python reference |

#### FPGA Target (Optimized Pipeline)

| Subsystem | Component | Size/Voice | 32 Voices | Optimization |
|-----------|-----------|------------|-----------|--------------|
| **VMState** | coeff_state (K=8) | 1,024 B | 32 KB | Core buffer |
| | coeff_state (K=4) | 512 B | 16 KB | 50% reduction |
| | coeff_state (K=2) | 256 B | 8 KB | 75% reduction |
| | phase_acc (32-bit) | 4 B | 128 B | Single accumulator |
| | phase_inc (32-bit) | 4 B | 128 B | Per-voice pitch |
| | envelope (packed) | 4 B | 128 B | Phase+level packed |
| | PRNG seed | 4 B | 128 B | xorshift32 |
| | flags (packed) | 2 B | 64 B | All booleans packed |
| | note + wave_id | 2 B | 64 B | Packed bytes |
| | **VMState K=8** | **1,044 B** | **33.4 KB** | |
| | **VMState K=4** | **532 B** | **17.0 KB** | |
| | **VMState K=2** | **276 B** | **8.8 KB** | |
| **WaveletEngine** | prev_coeffs (K=8) | 1,024 B | 32 KB | Needed for R operator |
| | prev_coeffs (K=4) | 512 B | 16 KB | Reduced |
| | prev_coeffs (K=2) | 256 B | 8 KB | Minimal |
| | regularity | 8 B | 256 B | prev_sample + counter |
| | (phase_state) | **0 B** | **0 B** | **ELIMINATED** (carrier handles pitch) |
| | **WaveletEngine K=8** | **1,032 B** | **33.0 KB** | |
| | **WaveletEngine K=4** | **520 B** | **16.6 KB** | |
| | **WaveletEngine K=2** | **264 B** | **8.5 KB** | |
| **Orchestrator** | VoiceSlot (packed) | 4 B | 128 B | State+flags packed |
| **Emitter** | (merged into VM) | 0 B | 0 B | No separate emitter |
| | **TOTAL K=8** | **2,080 B** | **66.6 KB** | |
| | **TOTAL K=4** | **1,056 B** | **33.8 KB** | **Recommended** |
| | **TOTAL K=2** | **544 B** | **17.4 KB** | Ultra-compact |

### 2. Waveform LUT Storage (Shared)

| Config | Formula | 4 Waves | Notes |
|--------|---------|---------|-------|
| Full 16-bit P=64 | L×K×P×2 | 256 KB | 64×8×64×2 per wave |
| Half-phase 16-bit | L×K×(P/2)×2 | 128 KB | Sign reconstruction |
| Half-phase 8-bit | L×K×(P/2)×1 | **64 KB** | 4× reduction |
| K=4 half-phase 8-bit | L×4×32×1 | **32 KB** | FPGA-fit |
| K=2 half-phase 8-bit | L×2×32×1 | **16 KB** | Minimal |
| Single wave K=2 | L×2×32×1 | **4 KB** | Ultra-minimal |

**LUT Parameters:**
- L = 64 leaves (fixed by WPT depth 6)
- K = polyphase time slots (8 → 4 → 2)
- P = 64 phase steps with cubic interpolation

### 3. IDWPT Lifting Engine (Shared)

| Component | Size | Notes |
|-----------|------|-------|
| Lifting coefficients | 64 B | CDF 5/3 predict/update |
| Temp coefficient buffer | 128 B | 64 × int16 working space |
| Temp audio buffer | 128 B | 64 × int16 output |
| **IDWPT Total** | **320 B** | Per-block, reused |

### 4. Output Mixing (Shared)

| Component | Size | Notes |
|-----------|------|-------|
| L accumulator | 256 B | 64 × int32 (prevent overflow) |
| R accumulator | 256 B | 64 × int32 |
| L output | 128 B | 64 × int16 final |
| R output | 128 B | 64 × int16 final |
| Mix gain state | 4 B | Q15 ramping |
| **Mix Total** | **772 B** | Per-block |

### 5. Fixed Infrastructure

| Component | Size | Notes |
|-----------|------|-------|
| Sine quarter-wave LUT | 2 KB | 1024 × int16, shared |
| Voice slot array | 64 B | 32 × 2 bytes (active/note) |
| Scheduler state | 32 B | Cursor, counters |
| Control registers | 64 B | Master gain, globals |
| **Fixed Total** | **2.2 KB** |

---

## Configuration Options (Full Pipeline)

### Option A: FPGA Baseline K=8 (Fits, Tight)

```
32 voices × K=8 polyphase (FULL PIPELINE)

Per-Voice State (complete):
  VMState (K=8):       33.4 KB  (32 × 1,044 B)
  WaveletEngine:       33.0 KB  (32 × 1,032 B) - includes prev_coeffs
  Orchestrator:         0.1 KB  (32 × 4 B)
  ───────────────────────────
  Per-voice subtotal:  66.5 KB

Shared State:
  Waveform LUTs:       32.0 KB  (4 × 8 KB, K=8 half-phase 8-bit)
  Scheduler state:      0.3 KB  (cursors, counters, 8 tiles × 4 B)
  IDWPT engine:         0.3 KB  (lifting coeffs, temp buffers)
  Output mixing:        0.8 KB  (L/R accumulators + output)
  Sine LUT:             2.0 KB  (quarter-wave, shared)
  Fixed overhead:       4.0 KB  (control registers, misc)
  ───────────────────────────
  Shared subtotal:     39.4 KB

─────────────────────────────
TOTAL:               105.9 KB  ✓ FITS (83%)
Margin:               22.1 KB  (limited expansion room)
```

### Option B: FPGA Optimized K=4 (RECOMMENDED)

```
32 voices × K=4 polyphase (FULL PIPELINE)

Per-Voice State (complete):
  VMState (K=4):       17.0 KB  (32 × 532 B)
  WaveletEngine:       16.6 KB  (32 × 520 B) - includes prev_coeffs
  Orchestrator:         0.1 KB  (32 × 4 B)
  ───────────────────────────
  Per-voice subtotal:  33.7 KB

Shared State:
  Waveform LUTs:       16.0 KB  (4 × 4 KB, K=4 half-phase 8-bit)
  Scheduler state:      0.3 KB
  IDWPT engine:         0.3 KB
  Output mixing:        0.8 KB
  Sine LUT:             2.0 KB
  Fixed overhead:       4.0 KB
  ───────────────────────────
  Shared subtotal:     23.4 KB

─────────────────────────────
TOTAL:                57.1 KB  ✓ FITS (45%)
Margin:               70.9 KB  (room for Mode B, samples, effects)
```

### Option C: FPGA Minimal K=2 (Maximum Margin)

```
32 voices × K=2 polyphase (FULL PIPELINE)

Per-Voice State (complete):
  VMState (K=2):        8.8 KB  (32 × 276 B)
  WaveletEngine:        8.5 KB  (32 × 264 B)
  Orchestrator:         0.1 KB
  ───────────────────────────
  Per-voice subtotal:  17.4 KB

Shared State:
  Waveform LUTs:        8.0 KB  (4 × 2 KB, K=2 half-phase 8-bit)
  Scheduler state:      0.3 KB
  IDWPT engine:         0.3 KB
  Output mixing:        0.8 KB
  Sine LUT:             2.0 KB
  Fixed overhead:       4.0 KB
  ───────────────────────────
  Shared subtotal:     15.4 KB

─────────────────────────────
TOTAL:                32.8 KB  ✓ FITS (26%)
Margin:               95.2 KB  (massive headroom for features)
```

### Option D: K=4 with Mode B Support (FM/PWM)

```
32 voices × K=4 + Mode B capability (FULL PIPELINE)

Per-Voice State:
  VMState (K=4):       17.0 KB
  WaveletEngine:       16.6 KB
  Orchestrator:         0.1 KB
  Mode B weights:       4.1 KB  (32 × 128 B: H_max=32, a_h + b_h)
  FM feedback:          0.5 KB  (8 voices × 64 B buffer)
  ───────────────────────────
  Per-voice subtotal:  38.3 KB

Shared State:
  Waveform LUTs:       16.0 KB  (Mode A fallback)
  Harmonic basis (12): 24.6 KB  (12 anchors × (2H+1) × 64 × 8 × 2)
                                (Reduced: 4 anchors = 8.2 KB)
  Scheduler + IDWPT:    0.6 KB
  Output mixing:        0.8 KB
  Sine LUT:             2.0 KB
  Fixed overhead:       4.0 KB
  ───────────────────────────
  Shared subtotal:     31.4 KB  (with 4 Mode B anchors)

─────────────────────────────
TOTAL:                69.7 KB  ✓ FITS (54%)
Margin:               58.3 KB  (enables PWM, FM synthesis)
```

---

## K Parameter Trade-offs

The K parameter (polyphase time samples) determines coefficient resolution:

| K | Coeffs/Voice | Audio Quality | Use Case |
|---|--------------|---------------|----------|
| 8 | 512 | Best | Python reference, high-end FPGA |
| 4 | 256 | Good | **Recommended for 128KB** |
| 2 | 128 | Acceptable | Ultra-low-resource |
| 1 | 64 | Minimal | Single-frame only |

**K=4 Impact:**
- 50% memory reduction vs K=8
- Slightly reduced time resolution within block
- Still maintains full 64-leaf spectral resolution
- Imperceptible difference for most sounds

---

## Implementation Notes

### 8-bit Coefficient Encoding

```python
# Store: 8-bit signed, range [-128, 127]
stored = (coeff_16bit >> 8).astype(np.int8)

# Reconstruct: Scale back to 16-bit
restored = stored.astype(np.int16) << 8

# Error: ±128 LSB (0.4% of full scale)
```

### Half-Phase Sign Reconstruction

```python
# Periodic waveforms: f(θ + 180°) = -f(θ)
# Store only phases 0° to 180°

def get_coeff(phase_32bit, half_lut):
    if phase_32bit >= 0x80000000:  # >= 180°
        phase_32bit -= 0x80000000
        return -half_lut[phase_32bit]
    else:
        return half_lut[phase_32bit]
```

### Mode B Runtime Generation

```python
# Instead of pre-computed basis atlas:
# 1. Use 2KB sine quarter-wave LUT
# 2. Compute Ucos_h, Usin_h per-voice per-block
# 3. Trade compute for memory

def mode_b_generate_coeffs(state, out_512):
    for h in range(state.H_active):
        # Compute sine/cosine from shared LUT
        cos_h = cosine_q15(state.phase * h)
        sin_h = sine_q15(state.phase * h)

        # Accumulate weighted harmonics
        for leaf in range(64):
            for k in range(K):
                out_512[leaf * K + k] += (
                    state.a_h[h] * cos_h +
                    state.b_h[h] * sin_h
                ) * basis_weight(leaf, h, k)
```

---

## Recommended Configuration for 128KB

```
┌────────────────────────────────────────────────────────┐
│  FULL SYNTHESIS PIPELINE - 128KB BRAM Config           │
├────────────────────────────────────────────────────────┤
│  Voices:           32                                  │
│  Polyphase K:      4                                   │
│  Waveform LUTs:    4 (8-bit half-phase, K=4)           │
│  Mode B:           Runtime generation (4 anchors)     │
│  LUT Phase Steps:  64 (cubic interpolation)           │
├────────────────────────────────────────────────────────┤
│  Per-voice state:  33.7 KB (VM + Wavelet + Orch)      │
│  Shared state:     23.4 KB (LUTs + buffers + fixed)   │
│  ──────────────────────────────────────────────────── │
│  TOTAL USAGE:      57.1 KB (45%)                      │
│  MARGIN:           70.9 KB (55%)                      │
└────────────────────────────────────────────────────────┘
```

**Margin enables:**
- Additional waveform LUTs (+4 KB each at K=4)
- Mode B harmonic basis frames (+8 KB for 4 anchors)
- Per-voice effects state (reverb sends, delay feedback)
- Sample playback buffers
- Extended control registers
- Runtime operator parameter LUTs

---

## Python vs FPGA: Key Eliminations

| Python Overhead | FPGA Elimination | Savings |
|-----------------|------------------|---------|
| `voice_phase_state` (256 floats/voice) | Carrier handles pitch | **64 KB** |
| `VoiceFrameEmitter` (140 B/voice) | Merged into VMState | **4.5 KB** |
| Python dict overhead | Static arrays | ~10 KB |
| Float operations | Fixed-point only | N/A |
| Object references | Direct addressing | ~2 KB |

**Total Python overhead eliminated: ~80 KB**

---

## Code Changes Required

To implement K=4 mode:

1. **Add K parameter to VMState:**
   ```python
   # state.py
   VM_K = 4  # Reduced from 8
   coeff_state: np.ndarray  # Shape (64*VM_K,) = (256,)
   ```

2. **Update COEFF_LEN:**
   ```python
   # tables.py
   COEFF_LEN = 256  # Was 512
   ```

3. **Modify LUT generation:**
   ```python
   # coeff_lut_streaming.py
   DEFAULT_K = 4  # Reduced polyphase lanes
   ```

4. **Adjust IDWPT to match:**
   ```python
   # lanes.py - lane_ccs_render
   coeff_2d = state.coeff_state[:COEFF_LEN].reshape(64, VM_K)
   # K=4 means 4 IDWPTs per block (was 8)
   ```

5. **Update WaveletEngine prev_coeffs:**
   ```python
   # wavelet_engine.py
   self.prev_coeffs[voice_id] = [0] * (64 * VM_K)  # 256 instead of 512
   ```

---

## Conclusion

**32 voices in 128KB BRAM is achievable** with the full synthesis pipeline using:

| Optimization | Impact |
|-------------|--------|
| K=4 polyphase | 50% coeff_state reduction |
| 8-bit half-phase LUTs | 4× LUT size reduction |
| Eliminate voice_phase_state | 64 KB saved (carrier handles pitch) |
| Merge emitter into VM | 4.5 KB saved |
| Packed flags/enums | Minor savings |

**Recommended configuration (Option B):**
- Uses **57.1 KB** (45% of 128KB budget)
- Leaves **70.9 KB margin** (55%) for:
  - Mode B FM/PWM synthesis
  - Additional waveforms
  - Effects state
  - Sample playback

**Critical insight:** The Python WaveletEngine's `voice_phase_state` (256 floats × 32 voices = 64 KB) can be completely eliminated in FPGA because the CCS carrier handles pitch/phase - the IDWPT lift doesn't need continuous phase tracking.
