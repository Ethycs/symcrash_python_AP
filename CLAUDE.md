when running python files use pixi run python

## Known Issues

- 5 FM6 presets failed to compile due to a serialization issue with FM6 operator ratios exceeding int16 range

## Tower Implementation (WSE-STD-070)

The InstrumentIR Tower is now runtime-meaningful with all layers compiling to executable behavior.

### Key Components

- **ExecutorLatch** (`vm/state.py`): Holds CMMM event state (node mask, tail mode) with crossfade support
- **CMMM Events** (`vm/executor.py`): Decodes 0x30 (NODEMASK_SET) and 0x31 (TAILMODE_SET) payloads
- **Crossfade Dispatch** (`vm/lanes.py`): `dispatch_coeff_op_crossfade()` blends operator outputs during mask transitions
- **Registries** (`library/regime_registry.py`, `library/tree_registry.py`): Runtime storage for regime graphs and tree families
- **Canonical Tables** (`tables/`): Pattern table, operator costs, tail modes

### Conformance Profiles

| Profile | Requires | Behavior |
|---------|----------|----------|
| P0 | L0 + envelopes | Static synthesis |
| P1 | P0 + L2 regimes | State machine drives emission |
| P2 | P1 + L3 CMMM + 0x30/0x31 | Full expressive synthesis |
| P3 | P2 + budgeting | Deterministic degradation |

### Test Coverage

- `tests/test_crossfade_dispatch.py` - ExecutorLatch weights and ramp behavior
- `tests/test_cmmm_events.py` - Event roundtrip and K computation
- `tests/test_golden_voiceframe_stream.py` - Deterministic replay verification