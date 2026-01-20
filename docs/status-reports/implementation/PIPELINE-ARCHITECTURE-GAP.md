# Pipeline Architecture Gap Analysis

**Status:** CURRENT
**Date:** 2026-01-16
**Related:** WSE-ARCHITECTURE-v1.md, primitives.py

---

## Executive Summary

This document captures the gap between the canonical 10-stage pipeline defined in WSE-ARCHITECTURE-v1.md and the current implementation. The execution layer (Symbol Engine → IDWPT → Material → Safety) is now unified and working correctly. The control plane (Cockpit → VoiceEngine) has a disconnect that prevents dynamic parameter routing.

---

## Canonical Pipeline (WSE-ARCHITECTURE-v1.md)

```
IntentFrame (GestureTime - async)
         │
         ▼
┌─────────────────────────────────┐
│   Stage 1: COCKPIT ENGINE       │  Slew bank, macro binding
│   (WSE-STD-COCKPIT-ENGINE)      │  Produces: ControlFrame
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Stage 2: VOICE ENGINE         │  Time owner, ADSR, regime graph
│   (WSE-STD-060)                 │  Consumes: ControlFrame (MISSING)
│   CMMM: L2 Regimes → L3 Emit    │  Produces: VoiceFrame per voice
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Stage 3-6: SYMBOL ENGINE      │  Coefficient-domain operators
│   (Embedded in vm_step_block)   │
│   • lane_inject (Events)        │  From primitives.py:
│   • backbone (24 Operators)     │  - 8 Families, 24 Operators
│   • lane_tail (4 TailModes)     │  - K Ladder degradation masks
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Stage 7: WAVELET ENGINE       │  Polyphase IDWPT reconstruction
│   lane_ccs_render()             │  64 leaves × 8 k-slots → 64 samples
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Stage 8: VOICE MIX            │  Sum voices → stereo (L, R)
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Stage 9: MATERIAL ENGINE      │  Analog processing layer
│   (WSE-STD-MATERIAL-v0)         │  6 programs: tape, catastrophe, etc.
│                                 │  Consumes: ParamFrame (EMPTY)
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Stage 10: SAFETY ENGINE       │  Audio-space tail
│   OLA, limiter, click prevent   │  Consumes: ParamFrame (EMPTY)
└─────────────────────────────────┘
         │
         ▼
    Audio Output (64 samples)
```

---

## Current Implementation Status

### Working Correctly (Unified)

| Stage | Component | Status | Notes |
|-------|-----------|--------|-------|
| 3-6 | Symbol Engine | ✅ UNIFIED | Single path through vm_step_block |
| 7 | Wavelet Engine | ✅ WORKING | lane_ccs_render with polyphase IDWPT |
| 8 | Voice Mix | ✅ WORKING | Inline in SynthMaster |
| 9 | Material Engine | ✅ CALLED | But receives empty ParamFrame |
| 10 | Safety Engine | ✅ CALLED | But receives empty ParamFrame |

### Gaps (Control Plane Disconnect)

| Gap | Severity | Description |
|-----|----------|-------------|
| **Gap 1: ControlFrame Discarded** | HIGH | Cockpit.tick() produces ControlFrame but it's never passed downstream |
| **Gap 2: ParamFrame Never Populated** | HIGH | Material/Safety receive empty defaults, no dynamic control |
| **Gap 3: InstrumentIR → VoiceEngine Incomplete** | MEDIUM | IR defines emit_templates but VoiceEngine doesn't fully consume them |

---

## Gap 1: ControlFrame Discarded

### Location
`synth_master.py:400-404`

### Current Code
```python
def render_block(self, ...):
    if self._enable_cockpit:
        control_frame = self.cockpit.tick()  # PRODUCED

    return self._render_block_voice_engine()  # control_frame NOT PASSED
```

### Expected Flow
```
Cockpit.tick() → ControlFrame → VoiceEngine.update_controls() → VoiceFrame
```

### Impact
- Voice control changes (ADSR retrigger, pitch bend, CC) only work through VoiceEngine's direct API
- No unified control path from gesture input to voice parameters
- Macro runtime updates don't reach the execution layer

---

## Gap 2: ParamFrame Never Populated

### Location
`synth_master.py:466-471`

### Current Code
```python
self._param_frame = ParamFrame()  # Created once, never updated

if self._enable_material:
    L, R = self.material.step_block(L, R, self._param_frame, ...)  # Empty params

if self._enable_safety:
    L, R = self.safety.step_block(L, R, self._param_frame)  # Empty params
```

### Expected Flow
```
ControlFrame.globals → ParamFrame → Material/Safety
```

### Impact
- Material engine effects (tape, catastrophe, rank, tropical) cannot be dynamically controlled
- Safety parameters (limiter threshold, OLA strength) are fixed at defaults
- No block-rate modulation of post-processing

---

## Gap 3: InstrumentIR → VoiceEngine Incomplete

### Location
`instrument_ir.py`, `voice_engine.py`, `emitter_bridge.py`

### Current State
InstrumentIR defines:
- `L2_regimes`: Regime graph with transitions
- `L3_cmmm`: Emit templates, overlays, CMMM control
- `control`: ControlConfig with macro defaults
- `envelopes`: Matrix envelope bindings

VoiceEngine consumes:
- GestureBasis (raw controller inputs)
- VoiceState (ADSR, regime)

### Missing Connection
- InstrumentIR.control → MacroRuntime initialization
- InstrumentIR.L3_cmmm.emit_templates → VoiceFrame field population
- InstrumentIR.envelopes → Matrix envelope routing

---

## Unified Execution Layer (Completed)

The following duplicate paths have been removed:

| Removed | Location | Replacement |
|---------|----------|-------------|
| FM6 Path A (direct bypass) | executor.py | Unified FWPT→ops→IDWPT path |
| vm_step_block_raw() | executor.py | Use vm_step_block() only |
| use_coeff_osc=False legacy | executor.py | CCS path is now unconditional |
| USE_VM_EXECUTOR=False legacy | unified_renderer.py | VM path is now unconditional |

### Current Unified Flow
```
VoiceFrame
    │
    ▼
vm_step_block(state, frame, tables)
    │
    ├─► decode_frame() → DecodedPlan
    │
    ├─► Coefficient Synthesis (Mode A/B/Sample/FM6)
    │   └─► All modes produce carrier_512 coefficients
    │
    ├─► Polyphase Injection (carrier_512 → coeff_state)
    │
    ├─► lane_inject() [Events]
    │
    ├─► dispatch_coeff_op_crossfade() [Backbone Operators]
    │   └─► CMMM-aware crossfade with ExecutorLatch
    │
    ├─► lane_tail() [Tail Operators per TailMode]
    │
    └─► lane_ccs_render() [IDWPT Reconstruction]
            │
            ▼
        output[64] audio samples
```

---

## Primitives Reference (primitives.py)

The canonical primitives define the vocabulary:

| Primitive | Count | Used By |
|-----------|-------|---------|
| Gesture | 8 | Cockpit input (GATE, PRESSURE, POSITION, etc.) |
| Intent | 8 | Family weight computation (I_TONE, I_BOW, etc.) |
| SystemMacro | 13 | CMMM control axes (ENERGY_STATE, BRIGHTNESS, etc.) |
| Family | 8 | Operator grouping (F_COLOR, F_TRANSPORT, etc.) |
| Operator | 24 | Coefficient transforms (TILT, DIFFUSE_UP, etc.) |
| Regime | 7 | Voice states (R0_IDLE through R6_RECOVERY) |
| EventLane | 8 | Event channels (EVT_OP_PULSE, EVT_TREE_MORPH, etc.) |
| TailMode | 4 | Safety operator sets (FULL, MID, MIN, EMERGENCY) |
| KLadder | 8 | Degradation masks (K32 through K1) |

---

## Recommended Fixes

### Priority 1: Wire ControlFrame → VoiceEngine

```python
# In SynthMaster.render_block():
if self._enable_cockpit:
    control_frame = self.cockpit.tick()
    if self._voice_engine:
        self._voice_engine.apply_control_frame(control_frame)
```

### Priority 2: Populate ParamFrame from ControlFrame

```python
# In SynthMaster._render_block_voice_engine():
if control_frame:
    self._param_frame.material_id = control_frame.globals.get('material_id', 0)
    self._param_frame.safety_threshold = control_frame.globals.get('limiter_threshold', 32767)
```

### Priority 3: Connect InstrumentIR to MacroRuntime

```python
# At preset load time:
if ir.control:
    control_program = ir.control.to_control_program()
    macro_runtime.apply_control_program(control_program)
```

---

## Test Verification

After fixes, verify with:

```bash
# Existing tests should still pass
pixi run python -m pytest tests/test_fm6_pipeline.py tests/test_crossfade_dispatch.py tests/test_cmmm_events.py -v

# Add integration test for control flow
pixi run python -m pytest tests/integration/test_control_pipeline.py -v
```

---

## References

- [WSE-ARCHITECTURE-v1.md](../../WSE-ARCHITECTURE-v1.md) - Canonical architecture
- [primitives.py](../../../src/symcrash_python_ap/primitives.py) - Named primitives
- [instrument_ir.py](../../../src/symcrash_python_ap/instrument_ir.py) - IR specification
- [synth_master.py](../../../src/symcrash_python_ap/synth_master.py) - Pipeline orchestrator
- [executor.py](../../../src/symcrash_python_ap/vm/executor.py) - VM execution
