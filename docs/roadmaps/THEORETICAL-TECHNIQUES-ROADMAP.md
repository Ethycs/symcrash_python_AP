# Theoretical Techniques Implementation Roadmap

**Source:** symcrash_docs/00 - Theory/07 - Music Sound Board/Theoretical Techniques Dump.md
**Status:** Active
**Date:** 2026-01-15
**Cross-ref:** [WSE-STD-CMM-ROADMAP-v0](../standards/voice/WSE-STD-CMM-ROADMAP-v0.md)

---

## Overview

This roadmap maps 23 theoretical synthesis techniques to the WSE architecture, prioritized by:
- **Feasibility**: How well it fits existing infrastructure
- **Impact**: Musical value and uniqueness
- **Cost**: FPGA/CPU budget requirements

### Current CMMM Phase Status (from WSE-STD-CMM-ROADMAP)

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0 — Substrate | ✅ Complete | ABI/VM/budgets/voice engine locked |
| Phase 1 — USS/Preset | ✅ Complete | InstrumentIR v0 + PresetLibraryV0 operational |
| Phase 2 — SUBTRACTIVE | 🟡 Partial | Carrier + Amp ADSR done; filter audio path + LFO missing |
| Phase 3 — FM/PM | 🟢 Largely Complete | FM6 engine + matrix envelopes working |
| Phase 4+ — Wavetable/Sampler/etc | 🔲 Not started | |

---

## Tier 1: Already Implemented or Near-Complete

These techniques are already in the codebase or require minimal work.

| Technique | WSE Location | Status | Notes |
|-----------|--------------|--------|-------|
| **Catastrophe/bifurcation synthesis** | Material Engine (programs 4-5) | ✅ Done | Fold (A2) and Cusp (A3) catastrophes |
| **Symbolic dynamics / Sturmian words** | GateMask operator (16) | ✅ Done | Sturmian/Thue-Morse sequences |
| **Tropical additive synthesis** | Material Engine (program 3) | ✅ Done | Tropical Cut (min/max threshold) |
| **Rank/order-statistic nonlinearities** | Material Engine (program 2) | ✅ Done | Rank Grit (trimmed mean) |
| **Chaos as oscillator** | PRNG + GateMask | ✅ Done | PCG-32 per-voice xorshift32, deterministic |
| **Topological/sheaf-based (partial)** | SheafProject operator (23) | ✅ Done | Local-to-global consistency enforcement |
| **Steel/structural hysteresis** | Material Engine (programs 6-7) | 📋 Spec'd | Bouc-Wen + damage events (WSE-STD-MATERIAL-EXT) |

### Related Infrastructure Complete

| System | Status | Location |
|--------|--------|----------|
| Voice Engine (CMMM) | ✅ Complete | `voice_engine.py` - 7 regimes, hysteresis, dwell |
| Matrix Envelope System | ✅ Complete | `envelope_targets.py` - FM_OP1-6, GAIN, FILTER targets |
| FM6 Engine | ✅ Complete | `fm6_engine.py` - 32 DX7 algorithms + feedback |
| Degradation Ladder | ✅ Complete | `degradation_ladder.py` - deterministic budget shedding |
| InstrumentIR v0 | ✅ Complete | `instrument_ir.py` - L0-L3 layered tower |
| PresetLibraryV0 | ✅ Complete | `library/preset_library.py` - YAML + versioning |

**Immediate actions:** Document existing capabilities in user guide.

---

## Tier 2: Low-Hanging Fruit (1-2 weeks each)

These fit naturally into existing systems with moderate effort.

### 2.1 Quasiperiodic Tiling Rhythms
**Math:** Penrose tilings, golden-ratio rotations
**WSE Fit:** Event scheduling in Voice Engine
**Implementation:**
1. Add `QuasiperiodicScheduler` to Voice Engine
2. Use φ (golden ratio) rotation for event timing
3. Expose via `quasi_period_q15` in VoiceFrame

**Files:** `voice_engine.py`, `frames.py`
**Cost:** ~50 lines, negligible CPU

### 2.2 Fractional-Calculus Filters
**Math:** Fractional derivatives (α-order differentiators/integrators)
**WSE Fit:** Material Engine or new operator
**Implementation:**
1. Grünwald-Letnikov discrete approximation
2. Short history buffer (8-16 samples)
3. Fractional order α ∈ [0, 2] as parameter

**Algorithm:**
```
D^α[x] ≈ (1/h^α) * Σ_{k=0}^{N} (-1)^k * C(α,k) * x[n-k]
where C(α,k) = α*(α-1)*...*(α-k+1) / k!
```

**Files:** `material_engine.py` (program 8)
**Cost:** ~100 lines, ~20 ticks/sample

### 2.3 Symbolic Block-Reordering Textures
**Math:** Combinatorics on words, permutation sequences
**WSE Fit:** New PERMUTE operator or Material program
**Implementation:**
1. Block-based sample reordering (not coefficient)
2. Permutation sequence from Sturmian/PRNG
3. Micro-cut granular feel

**Files:** `operators.py` (op 24) or `material_engine.py` (program 9)
**Cost:** ~80 lines, L-scaling (64 samples)

### 2.4 Integer/OEIS-Driven Partial Sets
**Math:** Beatty sequences, prime sequences for partials
**WSE Fit:** Harmonic Bank (Mode B) partial selection
**Implementation:**
1. OEIS sequence lookup tables (precomputed)
2. Map sequence to partial enable/disable mask
3. Expose `oeis_sequence_id` parameter

**Sequences to include:**
- A000040 (primes) → metallic
- A000045 (Fibonacci) → organic inharmonic
- A001950 (Beatty φ) → golden partials
- A002024 (triangular) → bell-like

**Files:** `harmonic_bank.py`, new `oeis_tables.py`
**Cost:** ~200 lines + 2KB lookup tables

---

## Tier 3: Medium Effort (2-4 weeks each)

These require new subsystems but have clear implementation paths.

### 3.1 Topological/Sheaf-Based Synthesis — EXTENSION
**Math:** Simplicial complexes, sheaf consistency
**WSE Fit:** CMMM operator constraints
**Status:** ✅ Base done, extension possible

**Already implemented:**
- SheafProject operator (op 23) enforces local-to-global consistency
- Tree family morphing via `tree_blend_q15`
- Band coherence rules in tail stage

**Extension opportunities:**
1. Multiple sheaf "structures" (switchable topologies)
2. Topology morph = which bands must agree changes over time
3. Expose topology_id as compositional parameter

**Files:** `operators.py` (SheafProject), `tree_library.py`
**Cost:** ~150 lines for extension (base already exists)

### 3.2 Jet-Space-Driven Control
**Math:** Jet spaces = (value, 1st deriv, 2nd deriv, ...)
**WSE Fit:** Voice Engine gesture processing
**Implementation:**
1. Track gesture derivatives: position, velocity, acceleration, jerk
2. Map higher derivatives to "complexity burst" triggers
3. Singularity detection: when derivatives diverge

**Algorithm:**
```
jet = (x, dx/dt, d²x/dt², d³x/dt³)
singularity_score = |d³x/dt³| / (|dx/dt| + ε)
if singularity_score > threshold:
    trigger_complexity_burst()
```

**Files:** `gesture_basis.py`, `voice_engine.py`
**Cost:** ~150 lines, minimal CPU (block-rate)

### 3.3 Differential-Geometric Chord Space
**Math:** Riemannian geometry, geodesics in pitch space
**WSE Fit:** Multi-voice coordination
**Implementation:**
1. Chord = point in torus (pitch class space)
2. Geodesic = shortest path respecting voice leading
3. Expose as "chord glide" mode in Voice Engine

**Files:** New `chord_geometry.py`, `multi_voice_emitter.py`
**Cost:** ~400 lines, block-rate computation

### 3.4 Space-Filling Curves in Timbre Space
**Math:** Hilbert/Peano curves
**WSE Fit:** Macro/control automation
**Implementation:**
1. Precompute Hilbert curve through N-D parameter space
2. Single "position" parameter scans entire space
3. Locally smooth, globally exhaustive

**Files:** New `hilbert_scanner.py`, `cockpit_engine.py`
**Cost:** ~200 lines + lookup tables

### 3.5 Chaotic + Filtered Hybrids
**Math:** Chua circuit, Lorenz with bandlimiting
**WSE Fit:** New oscillator mode or Material program
**Implementation:**
1. Discrete chaotic map (logistic, Hénon)
2. Post-filter with resonant LP
3. Chaos depth parameter: blend chaotic/clean

**Files:** `chaos_osc.py`, `vm/state.py`
**Cost:** ~250 lines, ~30 ticks/sample

---

## Tier 4: Significant Effort (1-2 months each)

These require substantial new infrastructure.

### 4.1 Operadic Phrase/Waveform Composition
**Math:** Operads = algebraic composition rules
**WSE Fit:** L4-L5 tower layers, regime composition
**Implementation:**
1. Define "phrase operads" with composition laws
2. Phrases can nest hierarchically
3. Fractal-like but musically coherent structure

**Requires:** New `operad_composer.py`, regime graph extensions
**Cost:** ~800 lines, architectural change

### 4.2 Noncommutative Modulation Structures
**Math:** Order-dependent operations (A*B ≠ B*A)
**WSE Fit:** Operator ordering in Symbol Engine
**Implementation:**
1. Make operator order musically significant
2. Different orders = different timbres
3. Expose ordering as compositional parameter

**Requires:** Symbol Engine refactor, order-aware dispatch
**Cost:** ~500 lines, CPU overhead for order tracking

### 4.3 Knot-Theoretic Resonant Paths
**Math:** Braid groups, knot invariants
**WSE Fit:** Multi-parameter automation paths
**Implementation:**
1. Define "knotted" paths through parameter space
2. Knot topology imprints on timbral evolution
3. Same start/end, different journey

**Requires:** New `knot_paths.py`, complex math
**Cost:** ~600 lines, precomputation heavy

### 4.4 Fuzzy-Logic Parameter Control
**Math:** Fuzzy sets, rule-based inference
**WSE Fit:** Cockpit Engine, macro runtime
**Implementation:**
1. Define fuzzy membership functions for "bright", "harsh", etc.
2. Rule engine: IF bright AND harsh THEN reduce_highs
3. Organic, linguistic morphing

**Requires:** New `fuzzy_control.py`, rule parser
**Cost:** ~500 lines, moderate CPU

### 4.5 Sound-Texture Statistics Synthesis
**Math:** Multi-scale texture statistics (McDermott/Simoncelli)
**WSE Fit:** Analysis-synthesis mode (requires FWPT path)
**Implementation:**
1. Extract statistics from reference texture
2. Synthesize matching statistics from noise
3. Iterative refinement

**Requires:** FWPT analysis path (currently not in voice path)
**Cost:** ~1000 lines, CPU-heavy (offline)

---

## Tier 5: Research Projects (3+ months)

These are ambitious and may require fundamental changes.

### 5.1 Chaos as Terrain (Wave-Terrain)
**Math:** Strange attractors as height maps
**WSE Fit:** New synthesis mode (Mode C?)
**Implementation:**
1. Generate Lorenz/Rössler attractor
2. Use as 2D wavetable surface
3. Scan with phase + modulator

**Requires:** New synthesis engine, 2D interpolation
**Cost:** Major new subsystem

### 5.2 Topological Timbre Modeling/Clustering
**Math:** Manifold learning, persistent homology
**WSE Fit:** Preset organization, timbre morphing
**Implementation:**
1. Embed presets in learned timbre manifold
2. Navigate via geodesics
3. Automatic "similar preset" discovery

**Requires:** Offline ML pipeline, manifold representation
**Cost:** Research project scope

### 5.3 Sturmian-Based Melodic Skeleton Morphing
**Math:** Sturmian morphisms, Christoffel words
**WSE Fit:** Sequencer/arpeggiator layer (not yet exists)
**Implementation:**
1. Represent melodies as Sturmian words
2. Morph between words via slope changes
3. Structurally coherent melodic transforms

**Requires:** New sequencer subsystem
**Cost:** New layer beyond current scope

---

## Implementation Priority Matrix

```
                    IMPACT
                Low    Med    High
           ┌────────┬────────┬────────┐
     Low   │ 2.3    │ 2.2    │ 2.4    │
EFFORT     ├────────┼────────┼────────┤
     Med   │ 3.4    │ 3.2    │ 3.1    │
           │ 3.5    │ 3.3    │        │
           ├────────┼────────┼────────┤
     High  │ 4.4    │ 4.1    │ 5.1    │
           │        │ 4.2    │ 5.2    │
           └────────┴────────┴────────┘
```

**Recommended Order:**
1. **2.4 OEIS Partials** - High impact, low effort, unique sound
2. **2.2 Fractional Filters** - Fills "viscous" texture gap
3. **3.1 Sheaf Extension** - Builds on existing SheafProject
4. **3.2 Jet-Space Control** - Adds gesture intelligence
5. **2.1 Quasiperiodic Rhythms** - Unique rhythmic feel

---

## Mapping to Existing Architecture

| Technique | Symbol Engine | Material Engine | Voice Engine | New Subsystem | Status |
|-----------|---------------|-----------------|--------------|---------------|--------|
| Catastrophe | - | ✅ (4,5) | - | - | ✅ Done |
| Sturmian | ✅ GateMask (16) | - | - | - | ✅ Done |
| Tropical | - | ✅ (3) | - | - | ✅ Done |
| Rank Stats | - | ✅ (2) | - | - | ✅ Done |
| Sheaf/Topo | ✅ SheafProject (23) | - | - | - | ✅ Done |
| Steel Hysteresis | - | 📋 (6,7) | - | - | 📋 Spec'd |
| Quasiperiodic | - | - | Scheduler | - | 🔲 Planned |
| Fractional | - | (8) | - | - | 🔲 Planned |
| Block Reorder | (24) | - | - | - | 🔲 Planned |
| OEIS Partials | - | - | - | Harmonic Bank | 🔲 Planned |
| Jet-Space | - | - | Gesture | - | 🔲 Planned |
| Chord Geometry | - | - | - | Multi-voice | 🔲 Planned |
| Hilbert Scan | - | - | - | Cockpit | 🔲 Planned |
| Chaos+Filter | - | (10) | - | - | 🔲 Planned |
| Operadic | - | - | - | L4-L5 | 🔲 Future |

### Alignment with CMMM Roadmap Phases

| Technique | CMMM Phase | Dependency |
|-----------|------------|------------|
| OEIS Partials | Phase 2 (SUBTRACTIVE) | Mode B harmonic bank |
| Fractional Filters | Phase 2 (SUBTRACTIVE) | Filter audio path needed |
| Jet-Space Control | Phase 2+ | GestureBasis complete |
| Chaos+Filter | Phase 4 (WAVETABLE) | New osc mode |
| Wave-Terrain | Phase 4 (WAVETABLE) | 2D table scanning |
| Granular/Texture | Phase 5 (SAMPLER) | Sample playback |

---

## Resource Estimates

| Phase | Techniques | Lines of Code | Dev Time | FPGA Impact |
|-------|------------|---------------|----------|-------------|
| Phase 1 | 2.1-2.4 | ~450 | 2 weeks | Minimal |
| Phase 2 | 3.1-3.3 | ~850 | 4 weeks | Moderate |
| Phase 3 | 3.4-3.5, 4.4 | ~950 | 6 weeks | Moderate |
| Phase 4 | 4.1-4.3 | ~1900 | 8 weeks | Significant |
| Research | 5.1-5.3 | ~3000+ | 3+ months | TBD |

---

## Next Steps

1. **Immediate:** Document existing catastrophe/Sturmian/tropical capabilities
2. **This sprint:** Implement 2.4 (OEIS partials) - highest ROI
3. **Next sprint:** Implement 2.2 (fractional filters)
4. **Following:** Extend SheafProject for topological morphing

---

**End of Roadmap**
