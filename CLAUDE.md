when running python files use pixi run python

## Roadmap-led work (user direction, 2026-10-31)

- Work from `symcrash_docs/06 - Roadmaps/TOWER-CONFORMANCE-ROADMAP.md`
  and the linked roadmap for the selected milestone. Follow documented
  dependencies and acceptance criteria; standards define required behavior.
- Before implementation, identify the roadmap item, its specification,
  bounded next step, and correctness/realtime/resource checks. Work on one
  bounded item at a time. Record necessary prerequisite work in that roadmap
  before expanding scope; status reports are evidence, not a separate backlog.
- Preserve documented N/K/P, stored widths and Q formats, RNE/SAT ordering,
  headroom, capacities and deadlines. Any required change must trace to its
  specification and include the tradeoff and hardware/resource consequences.
- Update the same roadmap at checkpoints with evidence and remaining gates.
  Distinguish a qualified candidate, production adoption, correctness, measured
  timing, and hardware fit. Partial improvements do not complete a milestone.
- Reuse existing tests and benchmark tools. Add machinery only for a concrete
  coverage gap; repeat checks when changes, failures or required gates warrant
  them. Do not start an unrelated optimization while a milestone is open.
- Engine maturity gates precede further hardware-port expansion. Preserve the
  existing RTL evidence. Follow the user's current pause and Git instructions;
  a roadmap update does not resume implementation or authorize a commit.

## Known Issues

- ~~5 FM6 presets failed to compile due to a serialization issue with FM6 operator ratios exceeding int16 range~~ **FIXED** (2026-01-21): Added `ValueType.I32` to voice_engine_registry.py

## Tower Implementation (WSE-STD-070)

The InstrumentIR Tower is now runtime-meaningful with all layers compiling to executable behavior.

### Key Components

- **ExecutorLatch** (`vm/state.py`): Holds CMMM event state (node mask, tail mode)
- **CMMM Events** (`vm/executor.py`): Decodes 0x30 (NODEMASK_SET) and 0x31 (TAILMODE_SET) payloads
- **Mask-Aware Dispatch** (`vm/lanes.py`): `dispatch_coeff_op_crossfade()` dispatches operators with current node mask
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

- `tests/test_crossfade_dispatch.py` - ExecutorLatch mask and dispatch behavior
- `tests/test_cmmm_events.py` - Event roundtrip and K computation
- `tests/test_golden_voiceframe_stream.py` - Deterministic replay verification