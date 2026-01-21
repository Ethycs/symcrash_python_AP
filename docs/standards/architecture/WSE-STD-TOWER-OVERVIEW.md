# Tower Architecture Quick Reference

**Standard ID:** WSE-STD-TOWER-OVERVIEW
**Status:** Normative
**Date:** 2026-01-20
**Purpose:** Entry point and quick reference for Tower architecture documentation

---

## Tower Levels

The Tower is a functorial lifting chain in the category of Controlled Semi-Markov Mealy Machines (CMMM). Each level adds structure via left adjoints.

| Level | Name | What It Adds | Authoritative Spec |
|-------|------|--------------|-------------------|
| **L0** | Patch | Static DSP graph (OSC → FILTER → AMP) | [WSE-STD-070](WSE-STD-070.md) §1-2 |
| **L1** | Family | Gesture semantics, arc modulators | [WSE-STD-070](WSE-STD-070.md) §3 |
| **L2** | Instrument | Regime state machine (7 states) | [WSE-STD-070](WSE-STD-070.md) §3 |
| **L3** | Cockpit | CMMM emission, emit templates | [WSE-STD-070](WSE-STD-070.md) §3, [InstrumentIR Tower Conformance](../../design/core-system/04%20-%20InstrumentIR%20Tower%20Conformance%20Spec.md) |
| **L4** | Forge | Graph editing operations | [WSE-STD-070](WSE-STD-070.md) §3 |
| **L5** | Lab | Custom ops/trees, research tools | [WSE-STD-070](WSE-STD-070.md) §3 (TBD) |

---

## Where to Start

| Goal | Read This |
|------|-----------|
| **Using presets** | [WSE-STD-070](WSE-STD-070.md) §1-2 (Preset format, editor primitives) |
| **Writing instruments** | [PRESET-AUTHORING-GUIDE](../library/PRESET-AUTHORING-GUIDE.md), [InstrumentIR Conformance](../../design/core-system/04%20-%20InstrumentIR%20Tower%20Conformance%20Spec.md) |
| **Understanding theory** | [Full Tower v0](../../../symcrash_docs/03%20-%20Audio%20Pipeline/01%20-%20Full%20Tower%20v0.md), [Lifting the Tower](../../../symcrash_docs/00%20-%20Theory/04%20-%20Voice%20Engine%20Background/04%20-%20Lifting%20the%20Tower.md) |
| **Implementation status** | [TOWER-CONFORMANCE-ROADMAP](../../roadmaps/TOWER-CONFORMANCE-ROADMAP.md) |
| **FM6 synthesis** | [FM6_PATHWAY_GUIDE](../../FM6_PATHWAY_GUIDE.md) |
| **Control/macros** | [MACROS-AND-CONTROLS](../library/MACROS-AND-CONTROLS.md) |

---

## Key Concepts

### Completion Rules
Tower levels compile downward. For most presets, define L0 and let the compiler auto-complete L1/L2/L3.

See: [WSE-STD-070 §4 - Tower Completion Rules](WSE-STD-070.md)

### CMMM Definition
Controlled Semi-Markov Mealy Machine: $\mathcal{M} = (S, U, P_\theta, E_\phi, M, Q)$

See: [Voice Engine Tower - CMMM](../../../symcrash_docs/03%20-%20Audio%20Pipeline/00%20-%20Voice%20Engine/Evolution/02%20-%20The%20Voice%20Engine%20Tower.md)

### Conformance Profiles

| Profile | Requirements | Description |
|---------|--------------|-------------|
| **P0** | L0 + envelopes | Static synthesis |
| **P1** | P0 + L2 regimes | State machine drives emission |
| **P2** | P1 + L3 CMMM + 0x30/0x31 | Full expressive synthesis |
| **P3** | P2 + budgeting | Deterministic degradation |

See: [TOWER-CONFORMANCE-ROADMAP §3](../../roadmaps/TOWER-CONFORMANCE-ROADMAP.md)

---

## Document Authority Matrix

| Topic | Authoritative Document | Category |
|-------|----------------------|----------|
| Tower level definitions | WSE-STD-070 | Standard |
| Preset format | WSE-STD-PRESET-v0 | Standard |
| Editor primitives | WSE-STD-070 §2 | Standard |
| Completion rules | WSE-STD-070 §4 | Standard |
| CMMM definition | Voice Engine Tower | Theory |
| Functorial structure | Lifting the Tower | Theory |
| IR layer schemas | InstrumentIR Conformance | Design |
| Preset authoring | PRESET-AUTHORING-GUIDE | Guide |
| Implementation status | TOWER-CONFORMANCE-ROADMAP | Roadmap |

---

## Related Standards

- [WSE-STD-001](../core/WSE-STD-001.md) - Fixed-Point Arithmetic
- [WSE-STD-020](../core/WSE-STD-020.md) - UART Protocol
- [WSE-STD-070](WSE-STD-070.md) - Anti-Click, Tower Architecture
- [WSE-STD-PRESET-v0](../control/WSE-STD-PRESET-v0.md) - Preset Format
- [WSE-STD-ENVELOPE-SUITE-v0](../control/WSE-STD-ENVELOPE-SUITE-v0.md) - Envelope System

---

**Document Version:** 1.0
**Created:** 2026-01-20
**Author:** Generated from TOWER-DOCUMENTATION-CONSOLIDATION-PLAN
