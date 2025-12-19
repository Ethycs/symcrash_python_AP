# Ultimate Wavelet Engine Exploration Plan

## Goal

Validate the "Ultimate Wavelet Engine" spec against our 24K LUT budget and determine the optimal implementation strategy.

---

## Phase 1: Core Kernel Sizing (Priority: Critical)

### 1.1 Lifting Core Variants

Build and measure three lifting implementations:

| Variant | Description | Target LUTs |
|---------|-------------|-------------|
| A) Combinational | Current approach, 6 levels unrolled | ~3,300 |
| B) Sequential + RAM | Single lift unit, 31 iterations | ~800-1,200 |
| C) Shift-register stencil | Streaming pipeline | ~600-800 |

**Deliverable:** Python model + Verilog for each, with cycle counts.

### 1.2 Forward vs Inverse

The spec requires BOTH directions:
- **Inverse** (synthesis): coeffs → audio (Generator mode)
- **Forward** (analysis): audio → coeffs (Transformer/Decomposer modes)

**Question:** Can we share hardware? The lifting steps are:
```
Forward:  predict then update
Inverse:  undo-update then undo-predict
```

Same adders/shifters, different order. Explore a **bidirectional kernel**.

**Deliverable:** Bidirectional lifting module, measure overhead vs dedicated.

---

## Phase 2: Baby Fourier Integration (Priority: High)

### 2.1 Goertzel vs Oscillator Bank

| Approach | LUTs | Cycles | Accuracy |
|----------|------|--------|----------|
| Goertzel (8 bins) | ~200-400 | 64/bin | Exact for bin center |
| Oscillator bank (8 osc) | ~300-500 | 1/sample | Phase-continuous |
| Hybrid | ~400-600 | Mixed | Best of both |

**Key question:** How many harmonic bins (H) do we actually need?
- Spec says 8-32
- Start with H=8, measure quality

**Deliverable:** Goertzel module, oscillator bank module, compare.

### 2.2 Harmonic/Exotic Blend

The blend equation:
```
x_gen = (1-η) · R_τ(c) + η · F(h)
```

This requires **two synthesis paths running simultaneously** when 0 < η < 1.

**Options:**
1. Always run both, mux output (2× resources)
2. Time-multiplex (2× cycles)
3. Approximate: blend coefficients before synthesis (cheaper, less accurate)

**Deliverable:** Evaluate option 3 accuracy in Python first.

---

## Phase 3: Multi-Tree Architecture (Priority: High)

### 3.1 Tree Representation

From earlier discussion, two approaches:

| Approach | LUTs | Flexibility |
|----------|------|-------------|
| Hardcoded per-tree datapaths | 3,300 × N trees | None |
| Pointer/schedule from RAM | ~1,000 | Unlimited |

**Decision point:** The spec wants 2+ tree families (harmonic vs exotic).
RAM-based scheduling is the clear winner.

### 3.2 Tree Crossfade

Spec requires crossfade over M_xf blocks when switching trees.

**Options:**
1. Run both trees simultaneously during crossfade (2× resources)
2. Coefficient-space interpolation (cheaper, may artifact)
3. Pre-compute crossfade in software, send blended coefficients

**Deliverable:** Prototype option 2 in Python, measure artifacts.

### 3.3 Tree Schedule Format

Define the RAM format for loadable trees:
```
TreeSchedule {
    leaf_map[32]: initial coefficient placement
    combine_ops[31]: (src_a, src_b, dst, depth) tuples
    complex_mask[32]: which leaves use atom transform
}
```

~256 bytes per tree. Fits easily in BRAM or loads from SD.

**Deliverable:** Tree schedule format spec, Python encoder/decoder.

---

## Phase 4: Three-Mode Operation (Priority: Medium)

### 4.1 Mode Switching

| Mode | Forward | Inverse | Baby Fourier |
|------|---------|---------|--------------|
| Generator | No | Yes | Optional |
| Transformer | Yes | Yes | Optional |
| Decomposer | Yes | No | Analysis only |

**Resource sharing:** All modes use same lifting kernel, just different data flow.

### 4.2 Transformer Pipeline

```
audio_in → [Condition] → [Forward WPT] → [Symbol Ops] → [Inverse WPT] → audio_out
```

**Latency:** 2 blocks minimum (1 analysis + 1 synthesis).

**Question:** Can Symbol Ops happen inside wavelet engine, or strictly external?
Spec suggests "restricted local operator subset" for low latency.

**Deliverable:** Define which operators can be "wavelet-local" vs external.

### 4.3 Decomposer Output

Tiles (cheap): per-node coefficients + energy
Flows (expensive): multi-block tracking, top-N extraction

**Question:** Does flow extraction happen in wavelet engine or downstream?

**Deliverable:** Define interface boundary for flow extraction.

---

## Phase 5: Conditioning Frontend (Priority: Medium)

### 5.1 DC Blocker

Standard 1st-order high-pass:
```
y[n] = x[n] - x[n-1] + α·y[n-1]
```

~20 LUTs, trivial.

### 5.2 Noise Gate

Threshold + hysteresis:
```
if (rms < threshold) gain = 0
else gain = 1
```

With slew limiting. ~50 LUTs.

### 5.3 Anti-Pop Slew Limiter

```
delta = x[n] - x[n-1]
if (|delta| > max_slew) x[n] = x[n-1] + sign(delta) * max_slew
```

~30 LUTs.

**Total conditioning:** ~100-150 LUTs. Negligible.

**Deliverable:** Conditioning module with configurable enables.

---

## Phase 6: Resource Budget Validation (Priority: Critical)

### 6.1 Component Budget

| Component | LUTs (est) | Notes |
|-----------|------------|-------|
| Bidirectional lifting kernel | 1,200 | Sequential, shared F/I |
| Tree schedule RAM | 100 | BRAM preferred |
| Baby Fourier (8 bins) | 400 | Goertzel |
| Harmonic blend mux | 100 | Output mixing |
| Conditioning frontend | 150 | DC block, gate, slew |
| Crossfade logic | 200 | Coefficient interpolation |
| Control/scheduling | 300 | State machines |
| **Wavelet Engine Total** | **~2,450** | **~10% of 24K** |

### 6.2 Remaining Budget

```
24,000 - 2,450 = 21,550 LUTs remaining
```

Plenty for:
- Symbol Engine
- Material Engine
- Safety Engine
- Mixer
- I2S / Audio IO
- USB/MIDI interface

### 6.3 Voice Multiplexing

Spec requires time-multiplexing across V voices.

At 48kHz, 64-sample blocks = 750 cycles/block at 50MHz.

Sequential lifting = ~40 cycles/voice.

**Max voices:** 750 / 40 = **~18 voices** per block.

If we need more, options:
1. Increase clock (100MHz → 36 voices)
2. Pipeline more aggressively
3. Accept fewer voices

**Deliverable:** Cycle-accurate simulation of voice budget.

---

## Phase 7: Implementation Sequence

### Step 1: Bidirectional Lifting Module
- [ ] Python reference with forward + inverse
- [ ] Verilog sequential implementation
- [ ] Testbench with bit-exact verification

### Step 2: Tree Schedule System
- [ ] Define schedule format
- [ ] Python encoder (tree → schedule)
- [ ] Verilog schedule executor
- [ ] Test with Trees 0-4

### Step 3: Baby Fourier Integration
- [ ] Goertzel module (8 bins)
- [ ] Phase accumulator for continuous tracking
- [ ] Blend logic with slewed η

### Step 4: Three-Mode Controller
- [ ] Mode state machine
- [ ] Generator datapath
- [ ] Transformer datapath (forward → ops → inverse)
- [ ] Decomposer output formatting

### Step 5: Conditioning Frontend
- [ ] DC blocker
- [ ] Noise gate
- [ ] Slew limiter
- [ ] Integration with input path

### Step 6: Crossfade System
- [ ] Tree switch detection
- [ ] Coefficient interpolation
- [ ] Output blend during transition

### Step 7: Integration & Validation
- [ ] Full Python model
- [ ] Full Verilog implementation
- [ ] Bit-exact comparison
- [ ] Resource utilization report

---

## Open Questions

1. **Stereo policy:** Independent trees per channel, or shared with M/S?
   - Recommendation: Shared tree, M/S processing upstream

2. **Coefficient precision:** Q1.15 or Q3.13?
   - Recommendation: Q1.15 for coefficients, Q1.23 for audio

3. **Flow extraction location:** Inside wavelet engine or downstream?
   - Recommendation: Downstream (symbol engine territory)

4. **Catastrophe integration:** How do catastrophe presets arm from wavelet features?
   - Recommendation: Export energy/flux metrics, arm logic in material engine

5. **Cache/packet system:** RAM format for stored wavelet blocks?
   - Recommendation: Same as tree schedule format, extends naturally

---

## Success Criteria

1. **Fits budget:** < 3,000 LUTs for complete wavelet engine
2. **Meets timing:** All modes complete within block interval
3. **Bit-exact:** Verilog matches Python reference
4. **Mode switching:** Seamless transitions without artifacts
5. **Tree flexibility:** Load arbitrary trees from SD card
