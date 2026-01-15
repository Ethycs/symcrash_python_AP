# WSE-STD-ENVELOPE-MATRIX v0 — Matrix Envelope System

**Standard ID:** WSE-STD-ENVELOPE-MATRIX v0
**Title:** Matrix Envelope Routing System for Multi-Target ADSR Control
**Status:** Normative (v0 draft)
**Applies to:** InstrumentIR, VoiceFrameEmitter, Executor
**Depends on:**

* WSE-STD-020 v1 (VoiceFrame ABI)
* WSE-STD-VM-HYBRID v0 (VM execution semantics)
* WSE-STD-FM6 (FM6 synthesis)

---

## 0. Scope

This standard defines:

1. A **matrix envelope system** allowing named envelopes to route to multiple targets
2. The **EnvelopeTarget** registry of valid routing destinations
3. **YAML schema** for defining envelope bindings in InstrumentIR
4. **VoiceFrame extension** (`env_targets_q15`) for per-block target values
5. **Executor behavior** for applying envelope targets to synthesis engines

This standard does **not** define:

* ADSR envelope shape computation (see EnvelopeScheduler)
* Individual target semantics beyond routing (see WSE-STD-FM6 for FM operators)

---

## 1. Motivation

### 1.1 Problem: Static FM Operator Levels

Traditional FM synthesis uses static operator levels, but **musical FM** requires:

- **Brightness decay**: Modulator envelope decays faster than carrier
- **Attack transients**: Modulator peaks during attack ("blat" on brass)
- **Per-operator control**: Different envelopes for different operator roles

### 1.2 Solution: Matrix Envelope Routing

Instead of one-envelope-per-operator, the matrix system allows:

- **Named envelopes** with semantic meaning ("carrier_env", "mod_env")
- **One-to-many routing**: Single envelope controls multiple targets
- **Target abstraction**: Same system works for FM6, filters, pitch, etc.

---

## 2. EnvelopeTarget Registry

### 2.1 Target IDs

All envelope routing uses integer target IDs from the `EnvelopeTarget` enum:

| Target Name | ID | Description |
|------------|-----|-------------|
| `FM_OP1` | 0 | FM6 Operator 1 `env_value_q15` |
| `FM_OP2` | 1 | FM6 Operator 2 `env_value_q15` |
| `FM_OP3` | 2 | FM6 Operator 3 `env_value_q15` |
| `FM_OP4` | 3 | FM6 Operator 4 `env_value_q15` |
| `FM_OP5` | 4 | FM6 Operator 5 `env_value_q15` |
| `FM_OP6` | 5 | FM6 Operator 6 `env_value_q15` |
| `GAIN` | 10 | Voice output gain multiplier |
| `FILTER_CUTOFF` | 20 | Filter cutoff frequency (reserved) |
| `FILTER_RESO` | 21 | Filter resonance (reserved) |
| `PITCH` | 30 | Pitch modulation (reserved) |
| `VIBRATO_DEPTH` | 31 | Vibrato depth (reserved) |
| `MATERIAL_MIX` | 40 | Material (dry/wet) blend (reserved) |
| `MATERIAL_DRIVE` | 41 | Material drive amount (reserved) |

### 2.2 Target Categories

```python
FM_OPERATOR_TARGETS = {FM_OP1, FM_OP2, FM_OP3, FM_OP4, FM_OP5, FM_OP6}
VOICE_TARGETS = {GAIN}
FILTER_TARGETS = {FILTER_CUTOFF, FILTER_RESO}  # Reserved
PITCH_TARGETS = {PITCH, VIBRATO_DEPTH}          # Reserved
```

---

## 3. YAML Schema

### 3.1 EnvelopeBinding Format

Envelope bindings are defined at the InstrumentIR top level:

```yaml
instrument_ir_v0:
  # ... meta, io sections ...

  # Matrix Envelope System
  envelopes:
    - id: "envelope_name"      # Required: unique identifier
      adsr:                    # Required: ADSR envelope shape
        a_ms: 8.0              # Attack time in milliseconds
        d_ms: 150.0            # Decay time in milliseconds
        s: 0.7                 # Sustain level (0.0 - 1.0)
        r_ms: 100.0            # Release time in milliseconds
      targets:                 # Required: list of EnvelopeTarget names
        - FM_OP1
        - FM_OP2
        - FM_OP3
```

### 3.2 Complete Example: FM Brass

```yaml
instrument_ir_v0:
  meta:
    instrument_id: "fm-brass-v0"
    name: "FM Brass"
    version: 0
    author: "symcrash"
    seed_u32: 0x464D4252

  # Matrix envelopes for proper brass voicing
  envelopes:
    # Carrier envelope: standard ADSR for amplitude
    - id: "carrier_env"
      adsr:
        a_ms: 8.0           # Fast attack
        d_ms: 150.0         # Medium decay
        s: 0.7              # Strong sustain
        r_ms: 100.0         # Quick release
      targets:
        - FM_OP1            # All 4 carriers share this envelope
        - FM_OP2
        - FM_OP3
        - FM_OP4

    # Modulator envelope: fast decay for brightness transient
    - id: "mod_env"
      adsr:
        a_ms: 8.0           # Same fast attack
        d_ms: 50.0          # Much faster decay (brightness fades)
        s: 0.3              # Lower sustain (less brightness in sustain)
        r_ms: 100.0         # Same release
      targets:
        - FM_OP5            # Both modulators share this envelope
        - FM_OP6

  io:
    sample_rate_hz: 48000
    block_size: 64

  layers:
    L0_graph:
      nodes:
        - id: "fm6"
          type: "FM6"
          params:
            algorithm: 19       # Split to 3 Carriers
            feedback: 1
            op_ratios: [65536, 65726, 65346, 65916, 65536, 65536]
            op_levels: [32767, 32767, 32767, 28671, 10922, 8191]
```

### 3.3 Multiple Envelopes Targeting Same Operator

If multiple envelopes target the same operator, the **last one wins** (no multiplication):

```yaml
envelopes:
  - id: "attack_boost"
    adsr: { a_ms: 0, d_ms: 20, s: 0.5, r_ms: 0 }
    targets: [FM_OP5]
  - id: "mod_env"
    adsr: { a_ms: 8, d_ms: 50, s: 0.3, r_ms: 100 }
    targets: [FM_OP5]  # This one wins
```

Future versions may support envelope multiplication or modulation.

---

## 4. VoiceFrame Extension

### 4.1 New Field: env_targets_q15

```python
@dataclass
class VoiceFrame:
    # ... existing fields ...

    # Matrix Envelope System (WSE-STD-ENVELOPE-MATRIX-v0)
    # Envelope routing targets: EnvelopeTarget ID → Q15 envelope value
    env_targets_q15: Optional[Dict[int, int]] = None
```

### 4.2 Semantics

- `None` = No envelope routing (use defaults)
- `{}` = Empty dict (no targets = all FM ops at Q15_ONE)
- `{0: 16384, 1: 16384, 4: 8192, 5: 8192}` = FM_OP1-2 at 0.5, FM_OP5-6 at 0.25

---

## 5. Executor Behavior

### 5.1 FM6 Operator Application

When `frame.env_targets_q15` is set, the executor applies values to FM6 operators:

```python
env_targets = getattr(frame, 'env_targets_q15', None)
if env_targets:
    for target_id, value in env_targets.items():
        if 0 <= target_id <= 5:  # FM_OP1 through FM_OP6
            fm6_state.operators[target_id].env_value_q15 = value
else:
    # No envelope targets: set all operators to full output
    for op in fm6_state.operators:
        op.env_value_q15 = Q15_ONE
```

### 5.2 GAIN Target (Future)

When `GAIN` (target_id=10) is present, it should multiply with existing gain:

```python
if 10 in env_targets:  # GAIN target
    gain_q15 = (gain_q15 * env_targets[10]) >> 15
```

---

## 6. VoiceFrameEmitter Integration

### 6.1 Initialization

```python
def _init_envelope_matrix(self, bindings):
    for binding in bindings:
        spec = ADSRSpec(
            attack_ms=binding.adsr.a_ms,
            decay_ms=binding.adsr.d_ms,
            sustain=binding.adsr.s,
            release_ms=binding.adsr.r_ms,
        )
        target_ids = [int(EnvelopeTarget[t]) for t in binding.targets]
        self._envelope_matrix[binding.id] = (
            EnvelopeScheduler(spec=spec),
            target_ids
        )
```

### 6.2 Note-On/Off

All matrix envelopes trigger/release together with the main voice envelope:

```python
def note_on(self, note, velocity_q15, cents):
    # ... existing note_on logic ...
    for scheduler, _ in self._envelope_matrix.values():
        scheduler.trigger_note()

def note_off(self):
    # ... existing note_off logic ...
    for scheduler, _ in self._envelope_matrix.values():
        scheduler.release_note()
```

### 6.3 Frame Emission

```python
def emit_frame(self):
    # Tick all matrix envelopes and collect target values
    env_targets_q15 = None
    if self._envelope_matrix:
        env_targets_q15 = {}
        for env_id, (scheduler, target_ids) in self._envelope_matrix.items():
            value = scheduler.tick()
            for target_id in target_ids:
                env_targets_q15[target_id] = value

    return VoiceFrame(
        # ... other fields ...
        env_targets_q15=env_targets_q15,
    )
```

---

## 7. Design Rationale

### 7.1 Why Named Envelopes?

Named envelopes provide semantic clarity:

- `"carrier_env"` vs `[32767, 32767, 32767, 32767, 0, 0]`
- Easier to understand preset intent
- Supports future envelope modulation by name

### 7.2 Why One-to-Many Routing?

Reduces envelope instances and CPU:

- 2 envelopes (carrier + mod) vs 6 envelopes (per-op)
- Single envelope parameter change affects all targets
- Matches DX7 mental model (envelope groups)

### 7.3 Why Target Registry?

Abstraction enables future expansion:

- Same YAML format for FM6, filter, pitch
- New targets added without schema changes
- Target semantics defined by engine (FM6, filter, etc.)

---

## 8. Conformance

A conforming implementation MUST:

1. Parse `envelopes` section from InstrumentIR YAML
2. Create EnvelopeScheduler for each binding
3. Route envelope outputs to targets via `env_targets_q15`
4. Apply FM_OPx targets to `fm6_state.operators[x].env_value_q15`
5. Trigger all matrix envelopes on note-on
6. Release all matrix envelopes on note-off

A conforming implementation MAY:

- Ignore unknown target names (skip with warning)
- Apply target-specific scaling or clamping
- Optimize by sharing scheduler instances for identical ADSRs

---

## 9. References

- [envelope_targets.py](../../../src/symcrash_python_ap/envelope_targets.py) - EnvelopeTarget enum
- [instrument_ir.py](../../../src/symcrash_python_ap/instrument_ir.py) - EnvelopeBinding dataclass
- [voiceframe_emitter.py](../../../src/symcrash_python_ap/voiceframe_emitter.py) - Matrix integration
- [executor.py](../../../src/symcrash_python_ap/vm/executor.py) - Target application

---

## Appendix A: Common Envelope Patterns

### A.1 Brass (Brightness Decay)

```yaml
envelopes:
  - id: "carrier_env"
    adsr: { a_ms: 8, d_ms: 150, s: 0.7, r_ms: 100 }
    targets: [FM_OP1, FM_OP2, FM_OP3, FM_OP4]
  - id: "mod_env"
    adsr: { a_ms: 8, d_ms: 50, s: 0.3, r_ms: 100 }
    targets: [FM_OP5, FM_OP6]
```

### A.2 Electric Piano (Bell-like Attack)

```yaml
envelopes:
  - id: "carrier_env"
    adsr: { a_ms: 0, d_ms: 800, s: 0.0, r_ms: 200 }
    targets: [FM_OP1, FM_OP3]
  - id: "mod_env"
    adsr: { a_ms: 0, d_ms: 200, s: 0.0, r_ms: 50 }
    targets: [FM_OP2, FM_OP4, FM_OP5, FM_OP6]
```

### A.3 Organ (No Decay)

```yaml
envelopes:
  - id: "organ_env"
    adsr: { a_ms: 2, d_ms: 0, s: 1.0, r_ms: 10 }
    targets: [FM_OP1, FM_OP2, FM_OP3, FM_OP4, FM_OP5, FM_OP6]
```

### A.4 Pad (Slow Attack, Long Release)

```yaml
envelopes:
  - id: "pad_env"
    adsr: { a_ms: 500, d_ms: 1000, s: 0.6, r_ms: 2000 }
    targets: [FM_OP1, FM_OP2, FM_OP3, FM_OP4, FM_OP5, FM_OP6]
```

---

**End of WSE-STD-ENVELOPE-MATRIX v0**
