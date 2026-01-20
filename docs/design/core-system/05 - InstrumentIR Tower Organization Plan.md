# InstrumentIR Tower Organization Plan

**Status:** DRAFT
**Date:** 2026-01-19
**Related:** WSE-STD-070, InstrumentIR Tower Conformance Spec, PIPELINE-ARCHITECTURE-GAP.md

---

## Executive Summary

This document defines how all synthesis parameters, control axes, and runtime configuration should be organized within the InstrumentIR Tower. The goal is to eliminate out-of-band patterns where values are hardcoded in code instead of flowing through the canonical IR → Capsule → VoiceFrame pipeline.

---

## Current InstrumentIR Structure

```
InstrumentIR
├── meta: InstrumentMeta
│   ├── instrument_id, name, version, author
│   ├── seed_u32 (PRNG seed)
│   └── voice_model: VoiceModel (max_voices, tile_size)
│
├── io: IOConfig
│   ├── sample_rate_hz: 48000
│   └── block_size: 64
│
├── layers: Layers
│   ├── L0_graph: L0Graph (DSP nodes, edges, macros)
│   ├── L1_lift: L1Lift (gesture basis, intent mapping, control config)
│   ├── L2_regimes: L2Regimes (regime graph, envelopes)
│   └── L3_cmmm: L3Cmmm (emit templates, transitions, overlays)
│
├── compile: CompileConfig
│   ├── map_library (expression mappings)
│   ├── completion_policy (tower completion rules)
│   ├── degrade_profile (budget degradation)
│   ├── frame_emitter_policy (VoiceFrame emission)
│   └── material: MaterialConfig (beta blend)
│
├── envelopes: List[EnvelopeBinding] (backwards compat → L2)
└── control: ControlConfig (backwards compat → L1)
```

---

## Out-of-Band Patterns to Consolidate

### 1. ADSR Defaults (Currently in synth_master.py)

**Current Location:** `synth_master.py:163-167`
```python
self._attack_blocks = 7
self._decay_blocks = 75
self._sustain = 0.7
self._release_blocks = 150
```

**Target Location:** `InstrumentIR.layers.L0_graph.nodes[].params.adsr`

Already defined in IR:
```python
@dataclass
class ADSR:
    a_ms: float = 8.0
    d_ms: float = 80.0
    s: float = 0.7
    r_ms: float = 180.0
```

**Action:** Remove hardcoded values from synth_master, read from compiled preset.

---

### 2. Harmonic Profile (Currently in synth_master.py)

**Current Location:** `synth_master.py:169`
```python
self._harmonic_amps = [32767, 16383, 10922, 8191, 6553, 5461, 4681, 4096]
```

**Target Location:** `InstrumentIR.layers.L0_graph.nodes[].params.harmonics`

Already defined in IR:
```python
@dataclass
class HarmonicProfile:
    amplitudes: List[float] = field(default_factory=lambda: [1.0])
```

**Action:** Remove hardcoded profile, read from NodeParams.harmonics.

---

### 3. Beta Target (Currently hardcoded in synth_master.py)

**Current Location:** `synth_master.py:441`
```python
beta_target_q15=24576  # Hardcoded 75%
```

**Target Location:** `InstrumentIR.compile.material.beta_target_q15`

Already defined in IR:
```python
@dataclass
class MaterialConfig:
    beta_start_q15: int = 0
    beta_target_q15: int = 16384
    beta_ramp_blocks: int = 8
```

**Action:** Read from compiled preset's MaterialConfig.

---

### 4. Material ID (Currently instance state in synth_master.py)

**Current Location:** `synth_master.py:140, 608`
```python
self._material_id = 0
def set_material(self, material_id: int): ...
```

**Target Location:** NEW - Add to CompileConfig

**Proposed Addition:**
```python
@dataclass
class MaterialConfig:
    beta_start_q15: int = 0
    beta_target_q15: int = 16384
    beta_ramp_blocks: int = 8
    material_id: int = 0              # NEW: Material program ID
    material_params: Dict[str, int] = field(default_factory=dict)  # NEW
```

---

### 5. Safety Configuration (Currently scattered)

**Current Locations:**
- `safety_engine.py:153-165` - Energy caps
- `synth_master.py:137-138` - Enable flags

**Target Location:** NEW - Add SafetyConfig to CompileConfig

**Proposed Addition:**
```python
@dataclass
class SafetyConfig:
    """Safety engine configuration."""
    enabled: bool = True
    dc_removal: bool = False          # Disabled for CCS path
    master_ola: bool = False          # OLA in lane_idwpt_render
    limiter_threshold_q15: int = 32000
    slew_limit_q15: int = 2000
    energy_cap_soft: int = 0x30000000
    energy_cap_hard: int = 0x40000000
```

---

### 6. Oscillator Mode (Currently instance state)

**Current Location:** `synth_master.py:175`
```python
self._osc_mode = OSC_MODE_A
```

**Target Location:** `InstrumentIR.layers.L0_graph.nodes[].params.osc_mode`

Already defined:
```python
@dataclass
class NodeParams:
    osc_mode: int = 0  # 0=A/LUT, 1=B/Rotator, 2=Sample, 3=FM6
```

**Action:** Read from compiled preset.

---

### 7. Q15/Block Constants (Currently in 38+ files)

**Current Locations:** Defined locally in many files
```python
Q15_ONE = 32767
Q15_HALF = 16384
BLOCK_SIZE = 64
```

**Target Location:** `primitives.py` (already partially there)

**Proposed Addition to primitives.py:**
```python
# =============================================================================
# FIXED-POINT CONSTANTS
# =============================================================================

Q15_ONE = 32767
Q15_HALF = 16384
Q15_QUARTER = 8192
Q15_SCALE = 32768

Q16_ONE = 65536
Q8_ONE = 256

# =============================================================================
# BLOCK CONSTANTS
# =============================================================================

BLOCK_SIZE = 64
BLOCK_SAMPLES = 64
COEFF_LEN = 512      # 64 leaves × 8 k-slots
VM_K = 8             # K time slots per block
```

**Action:** All files import from primitives.py instead of defining locally.

---

### 8. Control Axes (Cockpit → VoiceEngine)

**Current Problem:** ControlFrame produced but discarded in synth_master.py:400-404

**Target Flow:**
```
InstrumentIR.control (ControlConfig)
    ↓ Preset Load
MacroRuntime.apply_control_program()
    ↓ Per-Block
CockpitEngine.tick() → ControlFrame
    ↓
VoiceEngine.apply_control_frame(control_frame)
    ↓
VoiceFrame emission with updated parameters
```

**ControlConfig Already Defined:**
```python
@dataclass
class ControlConfig:
    energy_intent: float = 0.5
    motion: float = 0.0
    focus: float = 0.5
    brightness: float = 0.5
    density: float = 0.5
    risk: float = 0.0
    space: float = 0.25
    material: float = 0.5
    unity: float = 0.5
```

**Action:** Wire ControlFrame consumption in synth_master.render_block().

---

## Proposed InstrumentIR Additions

### Add to CompileConfig

```python
@dataclass
class SafetyConfig:
    """Safety engine configuration per WSE-STD-SAFETY."""
    enabled: bool = True
    dc_removal: bool = False
    master_ola: bool = False
    limiter_threshold_q15: int = 32000
    slew_limit_q15: int = 2000


@dataclass
class MaterialConfig:
    """Material engine configuration per WSE-STD-MATERIAL."""
    material_id: int = 0              # Material program (0-5)
    beta_start_q15: int = 0
    beta_target_q15: int = 16384
    beta_ramp_blocks: int = 8
    # Per-program parameters
    tape_saturation_q15: int = 8192
    catastrophe_threshold_q15: int = 24576


@dataclass
class CompileConfig:
    """Compile section configuration."""
    map_library: Dict[str, str] = field(default_factory=dict)
    completion_policy: CompletionPolicy = field(default_factory=CompletionPolicy)
    degrade_profile: DegradeProfile = field(default_factory=DegradeProfile)
    frame_emitter_policy: FrameEmitterPolicy = field(default_factory=FrameEmitterPolicy)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)  # NEW
```

---

## Tower Layer Responsibilities

| Layer | Name | What It Contains | Runtime Consumer |
|-------|------|------------------|------------------|
| **L0** | Patch | DSP graph, ADSR, harmonics, osc_mode | VoiceFrame fields |
| **L1** | Family | Gesture basis, intent mapping, control axes, FM6 config | MacroRuntime, Cockpit |
| **L2** | Instrument | Regime graph, envelopes, CMMM control | VoiceEngine regime machine |
| **L3** | Cockpit | Emit templates, overlays, transitions | VoiceFrame emission |
| **compile** | Config | Material, safety, degrade, emission policy | SynthMaster post-processing |

---

## Migration Path

### Phase 1: Consolidate Constants
1. Add Q15/block constants to `primitives.py`
2. Update all files to import from primitives
3. Remove local constant definitions

### Phase 2: Add Missing IR Sections
1. Add `SafetyConfig` to `instrument_ir.py`
2. Extend `MaterialConfig` with material_id and params
3. Update `CompileConfig` to include safety

### Phase 3: Wire Control Plane
1. Connect ControlFrame → VoiceEngine in synth_master
2. Populate ParamFrame from ControlFrame for Material/Safety
3. Remove direct material_id instance state

### Phase 4: Remove Hardcoded Defaults
1. Remove ADSR hardcoding from synth_master
2. Remove harmonic profile hardcoding
3. Remove beta_target hardcoding
4. All values come from compiled preset (InstrumentIR → Capsule)

---

## Validation

After migration, verify:

1. **No local Q15 constants** - All files import from primitives
2. **No hardcoded ADSR/harmonics** - Read from compiled preset
3. **ControlFrame consumed** - Not discarded in render loop
4. **Material/Safety configurable** - Via InstrumentIR, not instance state
5. **Tower projection works** - L3 → L2 → L1 → L0 without breaking

---

## References

- [WSE-STD-070](../../standards/architecture/WSE-STD-070.md) - Tower completion rules
- [InstrumentIR Tower Conformance Spec](04%20-%20InstrumentIR%20Tower%20Conformance%20Spec.md)
- [PIPELINE-ARCHITECTURE-GAP](../../status-reports/implementation/PIPELINE-ARCHITECTURE-GAP.md)
- [primitives.py](../../../src/symcrash_python_ap/primitives.py)
- [instrument_ir.py](../../../src/symcrash_python_ap/instrument_ir.py)
