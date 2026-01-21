# WSE-STD-SUBTRACTIVE-PROFILE v0 — Virtual Analog Synthesis Profile

**Standard ID:** WSE-STD-SUBTRACTIVE-PROFILE v0
**Title:** CMMM Profile for Virtual Analog (Subtractive) Synthesis
**Status:** Normative (v0 draft)
**Applies to:** Voice Engine, Preset Compiler, Material Engine
**Depends on:**
* WSE-STD-020 v1 (VoiceFrame ABI)
* WSE-STD-VM-HYBRID v0 (VM execution semantics)
* WSE-STD-ENVELOPE-SUITE v0 (Envelope generators)
* WSE-STD-001 (Fixed-Point arithmetic)

---

## 0. Scope

This document defines the **SUBTRACTIVE** CMMM profile for virtual analog synthesis. This profile covers:

- Classic oscillators (saw, pulse, triangle, sine)
- Resonant filters (lowpass, highpass, bandpass, notch)
- ADSR envelope generators for amplitude and filter
- LFO modulation to pitch, filter cutoff, and pulse width
- Unison/detune for thickness

This standard does **not** define:
* FM/PM synthesis (see WSE-STD-FM-PROFILE)
* Wavetable scanning (see future WSE-STD-WAVETABLE-PROFILE)
* Sample playback (see future WSE-STD-SAMPLER-PROFILE)

---

## 1. Profile Header

| Property | Value |
|----------|-------|
| Profile ID | `SUBTRACTIVE` |
| VM Lanes Used | CARRIER_RENDER, INJECT, BACKBONE, TAIL, IDWPT_RENDER, MIX_GAIN |
| Oscillator Modes | MODE_A (LUT), MODE_B (Harmonic Bank) |
| Budget Regimes | MAIN, RESIDUAL |
| Determinism | PRNG for noise; phase continuous for pitched oscillators |

---

## 2. InstrumentIR Fields

### 2.1 Required L0 Fields

```yaml
instrument_ir_v0:
  meta:
    instrument_id: "subtractive-example"
    name: "Virtual Analog Synth"

  layers:
    L0_graph:
      nodes:
        - id: "carrier"
          type: "OSCILLATOR"
          params:
            osc_mode: 1          # MODE_B (Harmonic Bank)
            waveform: "SAW"      # SAW, PULSE, TRIANGLE, SINE
            detune_cents: 0

        - id: "filter"
          type: "FILTER"
          params:
            filter_type: "LP"    # LP, HP, BP, NOTCH
            cutoff_hz: 2000
            resonance: 0.5

  envelopes:
    - id: "amp_env"
      adsr: { a_ms: 10, d_ms: 100, s: 0.7, r_ms: 200 }
      targets: [GAIN]

    - id: "filter_env"
      adsr: { a_ms: 5, d_ms: 200, s: 0.3, r_ms: 300 }
      targets: [FILTER_CUTOFF]
```

### 2.2 Oscillator Specification

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `osc_mode` | U8 | 0-1 | 0=MODE_A (LUT), 1=MODE_B (Harmonic) |
| `waveform` | Enum | SAW, PULSE, TRI, SINE | Base waveform shape |
| `detune_cents` | I16 | -1200 to +1200 | Pitch offset in cents |
| `pulsewidth_q15` | Q15U | 0-32767 | Pulse width (PULSE only) |
| `unison_voices` | U8 | 1-8 | Number of unison voices |
| `unison_detune_cents` | U16 | 0-100 | Spread between unison voices |

### 2.3 Filter Specification

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `filter_type` | Enum | LP, HP, BP, NOTCH | Filter mode |
| `cutoff_hz` | U16 | 20-20000 | Base cutoff frequency |
| `resonance_q15` | Q15U | 0-32767 | Resonance (Q) |
| `env_depth_q15` | Q15S | -32768 to +32767 | Envelope modulation depth |
| `keytrack_q15` | Q15U | 0-32767 | Keyboard tracking amount |

### 2.4 LFO Specification

```yaml
lfo:
  rate_hz: 5.0           # 0.01 to 50 Hz
  shape: "SINE"          # SINE, TRIANGLE, SAW_UP, SAW_DOWN, SQUARE, S&H
  targets:
    - target: "PITCH"
      depth_q15: 1000    # Vibrato depth
    - target: "FILTER_CUTOFF"
      depth_q15: 8000    # Filter wobble
```

---

## 3. Control Manifold

### 3.1 Macro Controls (Producer-Facing)

| Macro | Range | Default | Mapping |
|-------|-------|---------|---------|
| BRIGHTNESS | 0-32767 | 32767 | filter_cutoff_hz |
| DYNAMICS | 0-32767 | 32767 | gain_scale_q15 |
| VIBRATO | 0-32767 | 0 | LFO depth to pitch |
| ATTACK | 0-65535 | 10 | amp_env.attack_ms |
| RELEASE | 0-65535 | 200 | amp_env.release_ms |

### 3.2 Internal Mapping

```python
# In voice_engine.py:
if macro_id == SystemMacro.BRIGHTNESS:
    # Scale filter cutoff: 0 = minimum cutoff, 32767 = maximum
    filter_cutoff_q15 = (base_cutoff_q15 * brightness_q15) >> 15

if macro_id == SystemMacro.VIBRATO:
    # Scale LFO output: vibrato controls depth
    lfo_to_pitch = (lfo_output * vibrato_q15) >> 15
```

---

## 4. State Machine

### 4.1 Regime Graph

```
IDLE ──note_on──> ATTACK ──complete──> SUSTAIN ──note_off──> RELEASE ──complete──> IDLE
                    │
                    └──retrigger──> ATTACK (from current level)
```

### 4.2 Regime Definitions

| Regime | Budget | Description |
|--------|--------|-------------|
| IDLE | Minimal | Voice inactive, ready for allocation |
| ATTACK | Full | Rising envelope, all features active |
| SUSTAIN | Normal | Steady state, envelope at sustain level |
| RELEASE | Normal→Minimal | Falling envelope, prepare for reclaim |

### 4.3 Hysteresis Rules

- **ATTACK→SUSTAIN**: After attack+decay time, level reaches sustain
- **SUSTAIN→RELEASE**: On note_off event
- **RELEASE→IDLE**: When envelope reaches zero (< threshold)
- **Retrigger**: Re-enter ATTACK from any active state

---

## 5. Compiler Rules

### 5.1 IR → VoiceFrame Compilation

```python
def compile_subtractive(ir: InstrumentIR) -> VoiceParamDefaults:
    defaults = {}

    # Oscillator
    osc_node = ir.get_node("carrier")
    defaults[VoiceParamID.OSC_MODE] = osc_node.params.osc_mode
    defaults[VoiceParamID.WAVEFORM] = waveform_to_int(osc_node.params.waveform)

    # Filter
    filter_node = ir.get_node("filter")
    defaults[VoiceParamID.FILTER_MODE] = filter_mode_to_int(filter_node.params.filter_type)
    defaults[VoiceParamID.FILTER_CUTOFF] = hz_to_q15(filter_node.params.cutoff_hz)
    defaults[VoiceParamID.FILTER_RESO] = filter_node.params.resonance_q15

    # LFO
    if ir.lfo:
        defaults[VoiceParamID.LFO_RATE] = hz_to_phase_inc(ir.lfo.rate_hz)
        defaults[VoiceParamID.LFO_SHAPE] = shape_to_int(ir.lfo.shape)

    return defaults
```

### 5.2 K/Pattern Selection

| Complexity | K | Pattern | Notes |
|------------|---|---------|-------|
| Full quality | 32 | K32_FULL | Default for MAIN voices |
| Normal | 16 | K16_HARMONIC | Sustain phase |
| Reduced | 8 | K8_CORE | Under load |
| Emergency | 4 | K4_MINIMAL | Heavy degradation |

### 5.3 Tail Mode Selection

| Condition | Tail Mode | Description |
|-----------|-----------|-------------|
| Attack phase | FULL | Maximum quality |
| Sustain phase | MID | Balanced |
| Release phase | MIN | Acceptable quality |
| Over budget | EMERGENCY | Preserve pitch only |

---

## 6. Degradation Semantics

### 6.1 Degradation Priority (First to Drop)

1. **Unison voices** → Reduce to mono
2. **Filter resonance** → Flatten response
3. **LFO modulation** → Disable modulation
4. **K reduction** → 32→16→8→4
5. **Tail mode** → FULL→MID→MIN→EMERGENCY

### 6.2 Preserved Under Degradation

- **Pitch accuracy**: Always maintained
- **Envelope shape**: Attack/release contour preserved
- **Filter cutoff**: Base cutoff maintained (modulation may drop)

---

## 7. Implementation

### 7.1 Oscillator (Mode A / Mode B)

**Mode A (LUT-based):**
- 256-sample half-period tables
- BLEP anti-aliasing for discontinuities
- Sub-sample interpolation

**Mode B (Harmonic Bank):**
- Additive synthesis via harmonic weights
- Bandlimited by construction
- Smooth morphing between waveforms

```python
# In harmonic_bank.py:
def get_waveform_harmonics(waveform: str, num_harmonics: int) -> List[int]:
    if waveform == "SAW":
        return [Q15_ONE // (n + 1) for n in range(num_harmonics)]
    elif waveform == "SQUARE":
        return [Q15_ONE // (n + 1) if n % 2 == 0 else 0 for n in range(num_harmonics)]
    # ...
```

### 7.2 Filter (TPT SVF)

**Implementation:** `material_engine.py` (material_id=9)

```python
# Topology-Preserving Transform State Variable Filter
# Per-sample processing:
hp = (input - 2*R*bp - lp) * g_factor
bp = hp * g + bp_z1
lp = bp * g + lp_z1

# Mode selection:
if mode == LP: output = lp
elif mode == HP: output = hp
elif mode == BP: output = bp
elif mode == NOTCH: output = lp + hp
```

### 7.3 LFO

**Implementation:** `envelope_suite.py` (LFO class)

- Block-rate operation (750 Hz)
- 6 waveform shapes
- Phase sync on note-on (optional)
- Unipolar/bipolar output modes

---

## 8. Tests

### 8.1 Golden Trace Requirements

- [ ] Saw oscillator sweep (C2 to C6)
- [ ] Filter cutoff sweep (100 Hz to 10 kHz)
- [ ] LFO to pitch (vibrato)
- [ ] LFO to filter (auto-wah)
- [ ] Envelope attack transient
- [ ] Envelope release tail

### 8.2 Stress Trace Requirements

- [ ] K degradation under polyphony
- [ ] Filter resonance at high Q
- [ ] Unison detune spread
- [ ] LFO rate extremes (0.1 Hz to 50 Hz)
- [ ] Rapid note on/off (retriggering)

### 8.3 Existing Test Coverage

| Test File | Coverage |
|-----------|----------|
| `test_envelope_suite.py` | ADSR, LFO, Slew |
| `test_material_engine.py` | TPT SVF filter |
| `test_composite_osc.py` | Mode A/B oscillators |
| `test_harmonic_bank.py` | Waveform harmonics |

---

## 9. Conformance Checklist

A conforming SUBTRACTIVE profile implementation MUST:

- [x] Support MODE_A and MODE_B oscillators
- [x] Implement TPT SVF filter with LP/HP/BP/NOTCH modes
- [x] Provide ADSR envelope for amplitude
- [x] Provide ADSR envelope for filter (optional depth)
- [x] Support LFO with routing to pitch and filter
- [x] Map BRIGHTNESS macro to filter cutoff
- [x] Map VIBRATO macro to LFO→pitch depth
- [x] Degrade gracefully under budget pressure
- [ ] Produce hash-frozen golden traces
- [ ] Pass stress tests under polyphony

A conforming implementation MAY:

- Implement additional filter types (ladder, comb)
- Support pulse width modulation
- Add unison/detune modes
- Provide additional LFO shapes

---

## 10. References

- [envelope_suite.py](../../../src/symcrash_python_ap/envelope_suite.py) - ADSR, LFO implementation
- [material_engine.py](../../../src/symcrash_python_ap/material/material_engine.py) - TPT SVF filter
- [harmonic_bank.py](../../../src/symcrash_python_ap/vm/harmonic_bank.py) - Mode B waveforms
- [composite_osc.py](../../../src/symcrash_python_ap/vm/composite_osc.py) - Mode A/B blending
- [voice_engine.py](../../../src/symcrash_python_ap/voice/voice_engine.py) - Macro routing
- [WSE-STD-ENVELOPE-SUITE-v0.md](../control/WSE-STD-ENVELOPE-SUITE-v0.md) - Envelope specification

---

**Document Version:** 0.1
**Last Updated:** 2026-01-20
**Author:** CCS Development Team
**Status:** NORMATIVE DRAFT
