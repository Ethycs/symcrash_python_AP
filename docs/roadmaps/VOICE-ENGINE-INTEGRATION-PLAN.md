# Voice Engine Integration Plan

**Version:** 1.0
**Date:** 2026-01-16
**Status:** Planning

---

## 1. Problem Statement

Two parallel voice systems exist:

| System | Purpose | Status |
|--------|---------|--------|
| **VoiceEngine** | Full CMMM implementation (7 regimes, hysteresis, intent vectors) | Complete but unused |
| **VoiceFrameEmitter** | Simple preset→VoiceFrame converter | Used by all render paths |

VoiceFrameEmitter was accidentally created as a "quick and dirty" path and accumulated features that belong in VoiceEngine:
- LFO modulation with `_arc_phase_q16` storage
- Matrix envelope system
- ControlFrame consumption
- ADSR overrides

This creates:
1. **Redundant phase storage** - Both have `arc_phase_q16`
2. **Orphaned code** - VoiceState LFO methods never called
3. **Wrong layer** - LFO lives in emitter instead of engine
4. **No regime support** - Simple path can't use CMMM features

---

## 2. Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      VoiceEngine                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  VoiceState                          │    │
│  │  - arc_phase_q16[8]     (LFO phase storage)         │    │
│  │  - regime: Regime       (R0-R6 state machine)       │    │
│  │  - intent: IntentVector (8 semantic lanes)          │    │
│  │  - gesture: GestureBasis (8 normalized inputs)      │    │
│  │  - lfos: List[LFOState] (from preset)               │    │
│  │  - envelope state       (ADSR)                      │    │
│  │  - tree_morph           (blend state)               │    │
│  │  - family_weights       (computed per block)        │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │               step_voice()                           │    │
│  │  1. Update gesture from ControlFrame                │    │
│  │  2. Compute intent from gesture                     │    │
│  │  3. Evaluate regime transitions                     │    │
│  │  4. Tick LFOs (using arc_phase_q16)                 │    │
│  │  5. Compute family weights from intent+regime       │    │
│  │  6. Apply LFO modulation                            │    │
│  │  7. Build VoiceFrame                                │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           ▼                                  │
│                      VoiceFrame                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Integration Phases

### Phase 1-2: LFO in VoiceEngine ✅ COMPLETE

**Status:** Already implemented in VoiceEngine.

VoiceEngine already has:
- `state.tick_lfos()` method called in `step_voice()` at line 2729
- `state.handle_note_on_lfos()` called in `note_on()` at line 2348
- `VoiceState.arc_phase_q16` for phase storage

VoiceFrameEmitter has optional delegation via `voice_state` parameter.

---

### Phase 3: Add "Simple Mode" to VoiceEngine ✅ COMPLETE

**Goal:** VoiceEngine can run in simple mode (no regimes) for preset-only synthesis

**Implemented (2026-01-16):**

```python
# New enum in voice_engine.py
class EngineMode(IntEnum):
    SIMPLE = 0      # Preset-only, no regime transitions
    FULL_CMMM = 1   # Full regime graph with hysteresis

# New VoiceState fields for Simple mode
compiled_preset: Optional[CompiledPreset] = None
engine_mode: int = 1  # Default FULL_CMMM
beta_current_q15, beta_start_q15, beta_target_q15, beta_ramp_blocks
velocity_q15, pitch_offset_cents
envelope_matrix: Dict[str, Tuple[EnvelopeScheduler, List[int]]]
amp_envelope: Optional[EnvelopeScheduler]

# New VoiceEngine methods:
allocate_voice_from_preset(preset, voice_id)  # Allocate in Simple mode
note_on_simple(voice_id, note, velocity_q15, cents)
note_off_simple(voice_id)
step_voice_simple(voice_id) -> VoiceFrame  # Returns VoiceFrame (not VoiceFrameABI)
```

**Simple mode behavior:**
- Regime fixed at R2_STABLE
- No regime transitions
- Sound parameters from CompiledPreset
- LFOs, envelopes, matrix routing initialized from preset
- Returns VoiceFrame (compatible with existing VM)

**Files modified:**
- `voice/voice_engine.py` - Added EngineMode, Simple mode methods
- `voice/__init__.py` - Export EngineMode

---

### Phase 4: Deprecate VoiceFrameEmitter ✅ COMPLETE

**Status:** Implemented (2026-01-16)

**Goal:** VoiceFrameEmitter becomes a thin wrapper around VoiceEngine

**Implementation:**
- `VoiceFrameEmitter` now creates internal `VoiceEngine` and delegates all calls
- All state management (LFOs, envelopes, beta ramp) handled by VoiceEngine
- Control frame consumption still handled by emitter (applies modifiers to frames)
- Added `set_eta()`, `get_eta()`, `set_detune()`, `get_detune()` methods
- `MultiVoiceEmitter` also wraps VoiceEngine

**API preserved:**
- `note_on(note, velocity_q15, cents)` → `VoiceEngine.note_on_simple()`
- `note_off()` → `VoiceEngine.note_off_simple()`
- `emit_frame()` → `VoiceEngine.step_voice_simple()` + control modifiers
- `is_active`, `is_releasing` → check VoiceState.phase
- `reset()` → re-allocate voice from preset
- `consume_control_frame()` → local handling (Phase 5 topic)

**Files modified:**
- `voiceframe_emitter.py` - Complete rewrite as wrapper
- `tests/unit/test_preset_compiler.py` - Updated tests for public API

---

### Phase 5: Full Migration

**Goal:** All render paths use VoiceEngine directly

**Files to update:**
- `audio/multiprocess_synth.py`
- `tools/synth/standalone_synth.py`
- `tools/synth/standalone_synth_simple.py`
- `tools/synth/pico2_live_synth.py`
- `tools/synth/pico2_synth_test.py`
- `tools/benchmarks/bench_vm.py`
- `tools/audio/save_and_play.py`
- `symcrash_imgui_controls/synth_bridge.py`

**Migration pattern:**
```python
# OLD
emitter = VoiceFrameEmitter(preset)
emitter.note_on(60, 32767)
frame = emitter.emit_frame()

# NEW
engine = VoiceEngine(config=VoiceEngineConfig(mode=EngineMode.SIMPLE))
state = VoiceState(voice_id=0)
engine.allocate_voice(state, preset)
engine.note_on(state, 60, 32767)
frame = engine.step_voice(state)
```

---

### Phase 6: Delete VoiceFrameEmitter

**Goal:** Remove deprecated code

**Actions:**
1. Delete `voiceframe_emitter.py`
2. Update `__init__.py` exports
3. Update all imports
4. Update documentation

---

## 4. Feature Mapping

What moves where:

| Feature | VoiceFrameEmitter | → VoiceEngine |
|---------|-------------------|---------------|
| LFO phase storage | `_arc_phase_q16` | `VoiceState.arc_phase_q16` |
| LFO ticking | `emit_frame()` | `step_voice()` |
| LFO note-on sync | `note_on()` | `allocate_voice()` |
| Matrix envelopes | `_envelope_matrix` | `VoiceState.envelope_matrix` |
| Control consumption | `consume_control_frame()` | `update_gesture()` |
| ADSR overrides | `_attack_blocks`, etc | `VoiceState.adsr_override` |
| Beta ramping | `_beta_current_q15` | `VoiceState.beta_q15` |
| Pitch offset | `_pitch_offset_cents` | `VoiceState.detune_cents` |
| Family scales | `_family_scale_q15` | `VoiceState.family_weights` |

---

## 5. New VoiceState Fields

Fields to add to VoiceState:

```python
@dataclass
class VoiceState:
    # ... existing fields ...

    # Matrix envelope system (from Phase 2)
    envelope_matrix: Dict[str, Tuple[EnvelopeScheduler, List[int]]] = field(
        default_factory=dict
    )

    # Beta ramping (from VoiceFrameEmitter)
    beta_current_q15: int = 0
    beta_start_q15: int = 0
    beta_target_q15: int = 24576
    beta_ramp_blocks: int = 10

    # ADSR overrides (from ControlFrame)
    adsr_override: Optional[ADSRSpec] = None

    # Compiled preset reference (for emit)
    preset: Optional[CompiledPreset] = None
```

---

## 6. Compatibility Notes

### Standalone Mode Support

VoiceFrameEmitter's standalone mode (no VoiceState) is used by:
- Unit tests
- Simple tools

**Solution:** Create `VoiceEngine.create_standalone_voice()`:
```python
def create_standalone_voice(self, preset: CompiledPreset) -> Tuple[VoiceState, VoiceFrame]:
    """Create a voice for standalone/test use."""
    state = VoiceState(voice_id=0, preset=preset)
    state.lfos = preset.lfos
    return state
```

### ControlFrame Integration

VoiceFrameEmitter consumes ControlFrame directly. VoiceEngine uses GestureBasis.

**Solution:** Add ControlFrame→GestureBasis converter:
```python
def gesture_from_control_frame(frame: ControlFrame, voice_id: int) -> GestureBasis:
    """Extract gesture for a voice from ControlFrame."""
    gesture = GestureBasis()
    for patch in frame.voice_patches:
        if patch.voice_id == voice_id or patch.voice_id == 0:
            for item in patch.items:
                if item.control_id == ControlID.DYNAMICS:
                    gesture.pressure_q15 = item.value
                # ... etc
    return gesture
```

---

## 7. Timeline Estimate

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1 | 1 hour | None |
| Phase 2 | 2 hours | Phase 1 |
| Phase 3 | 3 hours | Phase 2 |
| Phase 4 | 2 hours | Phase 3 |
| Phase 5 | 4 hours | Phase 4 |
| Phase 6 | 1 hour | Phase 5 |

**Total:** ~13 hours

---

## 8. Verification Checklist

After each phase:

- [ ] All existing tests pass
- [ ] LFO modulation works correctly
- [ ] Matrix envelopes work correctly
- [ ] FM6 presets render correctly
- [ ] Pico2 live synth works
- [ ] No duplicate phase storage
- [ ] VoiceState.arc_phase_q16 is the single source of truth

---

## 9. References

- [WSE-ARCHITECTURE-v1](../standards/architecture/WSE-ARCHITECTURE-v1.md) - Voice Engine owns time
- [WSE-STD-060](../standards/testing/WSE-STD-060.md) - Deterministic replay
- [WSE-STD-070](../standards/architecture/WSE-STD-070.md) - Tower completion
- [Voice Engine Spec](../voice/voice_engine_spec.md) - CMMM specification
