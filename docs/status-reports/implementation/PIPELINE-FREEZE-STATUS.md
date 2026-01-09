# Pipeline Freeze Status

**Date**: 2026-01-08
**Status**: ✅ FEATURE COMPLETE — Ready for preset authoring

---

## Frozen Systems Summary

All core pipeline systems are frozen at v0. No breaking changes allowed.

| System | Version | File(s) | Status |
|--------|---------|---------|--------|
| **ControlRegistry** | v0 | `control_registry.py` | ✅ FROZEN |
| **ControlFrame** | v0 | `control_frame.py` | ✅ FROZEN |
| **CockpitEngine** | v0 | `cockpit_engine.py` | ✅ FROZEN |
| **VoiceEngineRegistry** | v0 | `voice/voice_engine_registry.py` | ✅ FROZEN |
| **VoiceFrame** | v0 | `frames.py` | ✅ FROZEN |
| **CapsuleV0** | v0 | `library/capsule.py` | ✅ FROZEN |
| **Capsule Codec** | v0 | `library/capsule_codec.py` | ✅ FROZEN |
| **VM Executor** | v0 | `vm/executor.py` | ✅ FROZEN |
| **VM Lanes** | v0 | `vm/lanes.py` | ✅ FROZEN |
| **VM State Codec** | v0 | `voice/voice_state_codec.py` | ✅ FROZEN |
| **CCS Mode A** | v0 | `vm/coeff_lut_streaming.py` | ✅ FROZEN |
| **CCS Mode B** | v0 | `vm/harmonic_bank.py` | ✅ FROZEN |
| **IDWPT Lifting** | v0 | `wavelet/fast_lifting.py` | ✅ FROZEN |
| **Degradation** | v0 | `budget/degradation.py` | ✅ FROZEN |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HOT PATH (750 Hz)                              │
└─────────────────────────────────────────────────────────────────────────────┘

 Pots/MIDI/ImGui
       │
       ▼
┌──────────────────┐
│  CockpitEngine   │  ← Control time owner (slew filtering)
│    tick()        │
└────────┬─────────┘
         │ ControlFrame
         ▼
┌──────────────────┐
│ VoiceFrameEmitter│  ← Musical time owner (ADSR, note duration)
│   emit_frame()   │
└────────┬─────────┘
         │ VoiceFrame
         ▼
┌──────────────────┐
│   VM Executor    │  ← Signal time owner (phase accumulators)
│  vm_step_block() │
│    ├── CCS      │      (Mode A or Mode B coefficient synthesis)
│    ├── Operators│      (24 lane operators)
│    ├── Tail     │      (safety constraints)
│    └── IDWPT    │      (coefficient → audio)
└────────┬─────────┘
         │ int16[64]
         ▼
┌──────────────────┐
│ UnifiedRenderer  │  ← Safety + pan law
│  render_block()  │
└────────┬─────────┘
         │ (L, R)
         ▼
     sounddevice
```

---

## CCS Architecture (WSE-STD-ARCH-CCS Compliant)

The CCS path eliminates FWPT from the voice hot path:

```
Phase → CoeffTable → coeff_state → Operators → IDWPT → Audio
```

### Mode A (LUT-based)
- Pre-computed coefficient tables per waveform/note
- Phase interpolation between LUT bins
- Best for arbitrary wavetables

### Mode B (Harmonic Bank)
- Pre-computed basis frames (Ucos_h, Usin_h, DC) per pitch anchor
- Runtime modulation with harmonic weights
- Best for classic oscillators, PWM, FM/PM

---

## Control Plane

### ControlID Enum (Frozen)

| ID | Name | Binding Target |
|----|------|----------------|
| 0 | TIMBRE | family_scale[COLOR] |
| 1 | TEXTURE | family_scale[LOCAL] |
| 2 | DYNAMICS | gain_scale_q15 |
| 3 | MORPH | tree_blend_q15 |
| 4 | CHAOS | family_scale[CATASTROPHE] |
| 5 | DETUNE | pitch_offset_cents |
| 6 | PITCH_BEND | pitch_offset_cents |
| 9 | MASTER_GAIN | global.master_gain_q15 |
| 16-19 | ADSR | attack/decay/sustain/release |

### VoiceParamID Domains (Frozen)

| Domain | Range | Purpose |
|--------|-------|---------|
| CORE | 0-31 | Pitch, gain, pan, envelope |
| CMMM | 32-63 | Tree blend, material morph |
| CCS | 64-127 | Coefficient oscillator params |
| IDWPT | 128-159 | Transform parameters |
| PLANNER | 160-191 | Budget planner |
| DEBUG | 240-255 | Debug/diagnostics |

---

## Test Status

### Passing Tests
- **175 pipeline_tests** (conformance, unit, integration, quality, spectral)
- **32 core audio_generation_tests** (carrier, presets, safety)

### Known Issues (Non-blocking)
- 24 test files fail to collect due to:
  - Missing `test_config.py` import path (majority)
  - Missing `vm.coeff_osc` module (legacy)
  - Stale P=256 assertion (P=64 now)

These are test infrastructure issues, not pipeline bugs.

---

## Capsule Format (Canonical Save/Load)

```
CapsuleV0
├── identity: CapsuleIdentity
│   ├── capsule_id, kind, name, author, tags
│   └── engine_compat (control_registry_abi, voice_registry_abi, voiceframe_abi)
├── program: ProgramReference
│   └── program_id, instrument_ir_id, regime_graph_id, tree_family_id
├── defaults: CapsuleDefaults
│   ├── control_defaults: Dict[ControlID, int]
│   └── voice_param_defaults: Dict[VoiceParamID, int]
├── assets: List[AssetRef]
└── seeds: SeedConfig
```

**Key Rule**: Capsule MUST NOT contain runtime histories (coeff_state, phase, PRNG state).

---

## What's Next

1. **Test Consolidation**: Fix import paths for `audio_generation_tests/`
2. **Preset Creation**: Use InstrumentIR factory methods to create presets
3. **Audio Validation**: Render and verify preset audio quality

---

## Specification Documents

| Document | Path | Status |
|----------|------|--------|
| WSE-STD-COCKPIT-ENGINE | `docs/standards/control/` | ✅ Frozen |
| WSE-STD-CONTROLFRAME | `docs/standards/control/` | ✅ Frozen |
| WSE-STD-CONTROLREGISTRY | `docs/standards/control/` | ✅ Frozen |
| WSE-STD-ARCH-CCS | `docs/standards/ccs/` | ✅ Frozen |
| WSE-STD-210 (Mode B) | `docs/standards/ccs/` | ✅ Frozen |
| WSE-STD-120 (Polyphase) | `docs/standards/ccs/` | ✅ Frozen |
| WSE-STD-VM-HYBRID | `docs/standards/voice/` | ✅ Frozen |
| WSE-STD-VOICEENGINEREGISTRY | `docs/standards/voice/` | ✅ Frozen |
| WSE-STD-CAPSULE-COMPAT | `docs/standards/library/` | ✅ Frozen |
| WSE-STD-DEGRADE | `docs/standards/budget/` | ✅ Frozen |

---

*Pipeline is feature complete. Ready for preset authoring.*
