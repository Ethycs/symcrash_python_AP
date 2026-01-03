# WSE-STD-CONTROLREGISTRY v0 — Control Registry (Truth Table)

**Status:** Normative (v0 frozen)
**Applies to:** CockpitEngine, VoiceFrameEmitter, presets
**Scope:** Control definitions, ranges, defaults, slew, and binding targets

---

## 0. Purpose

The **ControlRegistry** is the **single source of truth** for all control IDs:

* Legal range
* Default value
* Value type
* Slew policy
* Binding target (what field in VoiceFrame or emitter does it affect)

CockpitEngine and VoiceFrameEmitter both use this registry. Presets may override defaults but not ranges or types.

---

## 1. Registry Entry Format

```python
@dataclass
class ControlEntry:
    """Single control definition."""
    control_id: int          # ControlID enum value
    name: str                # Human-readable name
    value_type: ValueType    # Q15U, I16, U16, BOOL, ENUM8
    min_value: int           # Inclusive minimum
    max_value: int           # Inclusive maximum
    default_value: int       # Initial value
    slew_policy: SlewPolicy  # INSTANT, NBLOCK, ONE_POLE
    slew_param: int          # N for NBLOCK, alpha_q15 for ONE_POLE
    binding_target: str      # What it affects (see §3)
```

---

## 2. Value Types

```python
class ValueType(IntEnum):
    Q15U = 0      # Q15 unsigned: 0..32767
    I16 = 1       # Signed int16: -32768..32767
    U16 = 2       # Unsigned int16: 0..65535
    BOOL = 3      # Boolean: 0 or 1
    ENUM8 = 4     # Enumeration: 0..255
```

---

## 3. Binding Target Strings (Frozen v0)

| Binding Target | Description |
|----------------|-------------|
| `family_scale[COLOR]` | Multiply family[0] weight |
| `family_scale[LOCAL]` | Multiply family[1] weight |
| `family_scale[TRANSPORT]` | Multiply family[2] weight |
| `family_scale[GLUE]` | Multiply family[3] weight |
| `family_scale[TOPOLOGY]` | Multiply family[4] weight |
| `family_scale[CATASTROPHE]` | Multiply family[5] weight |
| `family_scale[INJECT]` | Multiply family[6] weight |
| `family_scale[TAIL]` | Multiply family[7] weight |
| `tree_blend_q15` | VoiceFrame tree_blend_q15 field |
| `gain_scale_q15` | Post-envelope gain multiplier |
| `pitch_offset_cents` | Added to note pitch (cumulative) |
| `attack_blocks` | ADSR attack time in blocks |
| `decay_blocks` | ADSR decay time in blocks |
| `sustain_level_q15` | ADSR sustain level |
| `release_blocks` | ADSR release time in blocks |
| `global.master_gain_q15` | Global master gain |
| `global.tempo_q8_8` | Global tempo (Q8.8) |

---

## 4. Default Registry Table (Frozen v0)

| ControlID | Name | Type | Min | Max | Default | Slew | Param | Binding |
|-----------|------|------|-----|-----|---------|------|-------|---------|
| 0 | TIMBRE | Q15U | 0 | 32767 | 32767 | ONE_POLE | 2048 | family_scale[COLOR] |
| 1 | TEXTURE | Q15U | 0 | 32767 | 32767 | ONE_POLE | 2048 | family_scale[LOCAL] |
| 2 | DYNAMICS | Q15U | 0 | 32767 | 32767 | ONE_POLE | 2048 | gain_scale_q15 |
| 3 | MORPH | Q15U | 0 | 32767 | 0 | NBLOCK | 16 | tree_blend_q15 |
| 4 | CHAOS | Q15U | 0 | 32767 | 0 | NBLOCK | 32 | family_scale[CATASTROPHE] |
| 5 | DETUNE | I16 | -2400 | 2400 | 0 | INSTANT | 0 | pitch_offset_cents |
| 6 | PITCH_BEND | I16 | -200 | 200 | 0 | INSTANT | 0 | pitch_offset_cents |
| 7 | RESERVED_7 | Q15U | 0 | 32767 | 0 | INSTANT | 0 | (none) |
| 8 | RESERVED_8 | Q15U | 0 | 32767 | 0 | INSTANT | 0 | (none) |
| 9 | MASTER_GAIN | Q15U | 0 | 32767 | 32767 | ONE_POLE | 4096 | global.master_gain_q15 |
| 10-15 | RESERVED | - | - | - | 0 | INSTANT | 0 | (none) |
| 16 | ATK_BLOCKS | U16 | 1 | 500 | 8 | NBLOCK | 4 | attack_blocks |
| 17 | DCY_BLOCKS | U16 | 1 | 1000 | 32 | NBLOCK | 4 | decay_blocks |
| 18 | SUS_Q15 | Q15U | 0 | 32767 | 24576 | NBLOCK | 4 | sustain_level_q15 |
| 19 | REL_BLOCKS | U16 | 1 | 1500 | 64 | NBLOCK | 4 | release_blocks |

### Notes on Default Values

* **TIMBRE, TEXTURE, DYNAMICS**: Default to 32767 (1.0) = no attenuation
* **MORPH, CHAOS**: Default to 0 = no effect
* **MASTER_GAIN**: Default to 32767 (1.0) = full volume
* **ADSR**: Default to classic synth values (~10ms attack, ~42ms decay, 0.75 sustain, ~85ms release)

---

## 5. Slew Computation (Integer-Only)

### 5.1 INSTANT

```python
current = target
```

No filtering. Value changes immediately.

### 5.2 NBLOCK(n)

Linear ramp over n blocks:

```python
def slew_nblock(current: int, target: int, n: int) -> int:
    if current == target:
        return current

    delta = target - current
    step = (abs(delta) + n - 1) // n  # ceiling division

    if delta > 0:
        return min(current + step, target)
    else:
        return max(current - step, target)
```

**Guaranteed**: Reaches target in exactly `n` blocks (worst case).

### 5.3 ONE_POLE(alpha_q15)

Exponential smoothing (integer-only):

```python
def slew_one_pole(current: int, target: int, alpha_q15: int, min_val: int, max_val: int) -> int:
    delta = target - current
    increment = (alpha_q15 * delta) >> 15

    # Ensure progress even for small deltas
    if increment == 0 and delta != 0:
        increment = 1 if delta > 0 else -1

    return max(min_val, min(max_val, current + increment))
```

**Time constant**: Approximately `(32767 / alpha_q15)` blocks to reach ~63% of target.

For `alpha_q15 = 2048`: ~16 blocks (~21ms) time constant.
For `alpha_q15 = 4096`: ~8 blocks (~11ms) time constant.

---

## 6. Cumulative Bindings

For binding targets marked as **cumulative**, multiple controls contribute:

```
pitch_offset_cents = sum of all controls bound to pitch_offset_cents
                   = DETUNE + PITCH_BEND
```

**Normative:** Cumulative bindings are computed as the sum of all contributing control values.

---

## 7. Python Reference Implementation

```python
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List


class ValueType(IntEnum):
    Q15U = 0
    I16 = 1
    U16 = 2
    BOOL = 3
    ENUM8 = 4


class SlewPolicy(IntEnum):
    INSTANT = 0
    NBLOCK = 1
    ONE_POLE = 2


@dataclass
class ControlEntry:
    control_id: int
    name: str
    value_type: ValueType
    min_value: int
    max_value: int
    default_value: int
    slew_policy: SlewPolicy
    slew_param: int
    binding_target: str


# Frozen v0 registry
CONTROL_REGISTRY_V0: List[ControlEntry] = [
    ControlEntry(0, "TIMBRE", ValueType.Q15U, 0, 32767, 32767,
                 SlewPolicy.ONE_POLE, 2048, "family_scale[COLOR]"),
    ControlEntry(1, "TEXTURE", ValueType.Q15U, 0, 32767, 32767,
                 SlewPolicy.ONE_POLE, 2048, "family_scale[LOCAL]"),
    ControlEntry(2, "DYNAMICS", ValueType.Q15U, 0, 32767, 32767,
                 SlewPolicy.ONE_POLE, 2048, "gain_scale_q15"),
    ControlEntry(3, "MORPH", ValueType.Q15U, 0, 32767, 0,
                 SlewPolicy.NBLOCK, 16, "tree_blend_q15"),
    ControlEntry(4, "CHAOS", ValueType.Q15U, 0, 32767, 0,
                 SlewPolicy.NBLOCK, 32, "family_scale[CATASTROPHE]"),
    ControlEntry(5, "DETUNE", ValueType.I16, -2400, 2400, 0,
                 SlewPolicy.INSTANT, 0, "pitch_offset_cents"),
    ControlEntry(6, "PITCH_BEND", ValueType.I16, -200, 200, 0,
                 SlewPolicy.INSTANT, 0, "pitch_offset_cents"),
    ControlEntry(9, "MASTER_GAIN", ValueType.Q15U, 0, 32767, 32767,
                 SlewPolicy.ONE_POLE, 4096, "global.master_gain_q15"),
    ControlEntry(16, "ATK_BLOCKS", ValueType.U16, 1, 500, 8,
                 SlewPolicy.NBLOCK, 4, "attack_blocks"),
    ControlEntry(17, "DCY_BLOCKS", ValueType.U16, 1, 1000, 32,
                 SlewPolicy.NBLOCK, 4, "decay_blocks"),
    ControlEntry(18, "SUS_Q15", ValueType.Q15U, 0, 32767, 24576,
                 SlewPolicy.NBLOCK, 4, "sustain_level_q15"),
    ControlEntry(19, "REL_BLOCKS", ValueType.U16, 1, 1500, 64,
                 SlewPolicy.NBLOCK, 4, "release_blocks"),
]


class ControlRegistry:
    """Single source of truth for control definitions."""

    def __init__(self, entries: List[ControlEntry] = None):
        self._entries: Dict[int, ControlEntry] = {}
        entries = entries or CONTROL_REGISTRY_V0
        for entry in entries:
            self._entries[entry.control_id] = entry

    def get_entry(self, control_id: int) -> ControlEntry:
        """Get control entry by ID. Raises KeyError if not found."""
        return self._entries[control_id]

    def get_default(self, control_id: int) -> int:
        """Get default value for control."""
        return self._entries[control_id].default_value

    def clamp(self, control_id: int, value: int) -> int:
        """Clamp value to legal range for control."""
        entry = self._entries[control_id]
        return max(entry.min_value, min(entry.max_value, value))

    def all_entries(self) -> List[ControlEntry]:
        """Get all registry entries."""
        return list(self._entries.values())

    def has_control(self, control_id: int) -> bool:
        """Check if control ID is defined."""
        return control_id in self._entries
```

---

## 8. Relationship to Other Standards

| Standard | How it uses ControlRegistry |
|----------|----------------------------|
| **WSE-STD-COCKPIT-ENGINE** | Uses registry for slew policies and binding targets |
| **WSE-STD-CONTROLFRAME** | Uses ControlID enum from registry |
| **WSE-STD-PRESET** | May override default values, but not ranges or types |
| **WSE-STD-020 VoiceFrame** | Family weights and other fields are binding targets |

---

## 9. Preset Interaction

Presets may define custom default values for controls, but:

* **MUST NOT** change min/max ranges
* **MUST NOT** change value types
* **MUST NOT** change slew policies (slew is runtime behavior)
* **MAY** override default values

Example preset override:

```yaml
# In preset YAML
control_defaults:
  TIMBRE: 16384    # Start at 50% instead of 100%
  CHAOS: 8192      # Add some chaos by default
  ATK_BLOCKS: 24   # Longer attack
```

---

## 10. Block Timing Reference

| Blocks | Time (ms) @ 48kHz/64 samples |
|--------|------------------------------|
| 1 | 1.33 ms |
| 8 | 10.7 ms |
| 16 | 21.3 ms |
| 32 | 42.7 ms |
| 64 | 85.3 ms |
| 128 | 170.7 ms |
| 256 | 341.3 ms |
| 500 | 666.7 ms |
| 750 | 1000 ms (1 second) |

---

## 11. Conformance Checklist

An implementation conforms to WSE-STD-CONTROLREGISTRY v0 iff:

- [ ] Uses frozen ControlID values (0-19 defined)
- [ ] All control values are clamped to [min_value, max_value]
- [ ] Default values are used for uninitialized controls
- [ ] Slew policies are correctly implemented (INSTANT, NBLOCK, ONE_POLE)
- [ ] Slew computation is integer-only (no floats)
- [ ] Cumulative bindings (pitch_offset_cents) are computed as sum
- [ ] Presets cannot change ranges or types

---

*WSE-STD-CONTROLREGISTRY v0 — Frozen*
