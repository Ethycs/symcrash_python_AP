# Tower/L5 Documentation Consolidation Plan

**Status:** ✅ COMPLETE (All Phases)
**Date:** 2026-01-19
**Updated:** 2026-01-20
**Cross-ref:** TOWER-CONFORMANCE-ROADMAP.md, WSE-STD-070.md, Full Tower v0.md

---

## Executive Summary

This plan proposes consolidating L5/Tower documentation and functionality currently scattered across `docs/` and `symcrash_docs/` into a coherent, maintainable structure. The goal is to eliminate duplication, clarify authoritative sources, and ensure implementation code references a single source of truth.

---

## 1. Current State Analysis

### 1.1 Documentation Locations

| Directory | Purpose | Authority Level |
|-----------|---------|-----------------|
| `docs/standards/` | Frozen normative specifications | **Primary** |
| `docs/roadmaps/` | Implementation status tracking | **Primary** |
| `docs/design/` | Design decisions and conformance specs | **Secondary** |
| `symcrash_docs/` | Theory, background, detailed derivations | **Reference** |

### 1.2 Tower Documentation Inventory

#### Primary Normative Documents (Keep as Authoritative)

| Document | Location | Content |
|----------|----------|---------|
| WSE-STD-070.md | `docs/standards/architecture/` | Preset format, completion rules, projections |
| TOWER-CONFORMANCE-ROADMAP.md | `docs/roadmaps/` | Master status tracker |
| 04 - InstrumentIR Tower Conformance Spec.md | `docs/design/core-system/` | IR layer definitions, YAML schemas |

#### Theory/Background Documents (Keep as Reference)

| Document | Location | Content |
|----------|----------|---------|
| 01 - Full Tower v0.md | `symcrash_docs/03 - Audio Pipeline/` | Definitive theory with editor/IR/engine primitives |
| 04 - Lifting the Tower.md | `symcrash_docs/00 - Theory/04 - Voice Engine Background/` | Functorial lifting diagrams |
| 02 - The Voice Engine Tower.md | `symcrash_docs/03 - Audio Pipeline/00 - Voice Engine/Evolution/` | CMMM definition, VoiceFrame ABI |

#### Duplicated/Overlapping Content (Needs Resolution)

| Content | Locations | Resolution |
|---------|-----------|------------|
| Tower level table (L0-L5) | WSE-STD-070 §3, Full Tower v0 | Merge into single normative table in WSE-STD-070 |
| CMMM definition | Voice Engine Tower, Lifting the Tower | Keep both (different perspectives) |
| Completion rules | WSE-STD-070 §4, implied in Full Tower | WSE-STD-070 is authoritative |
| Editor primitives | WSE-STD-070 §2, Full Tower v0 | WSE-STD-070 is authoritative |

---

## 2. Identified Issues

### 2.1 Duplication Issues

1. **Tower level definitions** appear in multiple documents with slight variations:
   - `symcrash_docs/03 - Audio Pipeline/01 - Full Tower v0.md` uses L0-L5 with detailed primitives
   - `symcrash_docs/03 - Audio Pipeline/00 - Voice Engine/Evolution/02 - The Voice Engine Tower.md` uses L0-L4 (missing L5)
   - `docs/standards/architecture/WSE-STD-070.md` uses L0-L5 with normative rules

2. **No single entry point** for developers - unclear which document to read first

3. **Cross-references are fragile** - roadmaps reference symcrash_docs paths that may change

### 2.2 L5 Specific Gaps

1. **L5 is defined but not implemented** - only enum exists in `instrument_ir.py`
2. **No L5 specification document** - Full Tower v0 describes intent but no formal spec
3. **NewOpDef/NewTreeDef formats undefined** - mentioned but not specified
4. **Compilation validators ($C_{5\to3}$) missing** - required but not designed

### 2.3 Inconsistencies

| Topic | Document A | Document B | Resolution |
|-------|------------|------------|------------|
| L4 name | "Forge" (WSE-STD-070) | "Forge" (Full Tower) | Consistent |
| L5 name | "Lab" (WSE-STD-070) | "Lab" (Full Tower) | Consistent |
| Tower levels count | 6 levels (L0-L5) | 5 levels (L0-L4) in Voice Engine Tower | Normalize to 6 |
| Event payload | 0x30/0x31 in WSE-STD-070 | Not in Full Tower | Add to Full Tower |

---

## 3. Consolidation Recommendations

### 3.1 Document Hierarchy (Proposed)

```
docs/
├── standards/
│   └── architecture/
│       └── WSE-STD-070.md        # FROZEN: Normative tower spec
│       └── WSE-STD-TOWER-OVERVIEW.md  # NEW: Quick reference
├── design/
│   └── core-system/
│       └── 04 - InstrumentIR Tower Conformance Spec.md  # KEEP: IR details
│       └── 05 - InstrumentIR Tower Organization Plan.md  # KEEP: Migration
└── roadmaps/
    └── TOWER-CONFORMANCE-ROADMAP.md  # KEEP: Master status

symcrash_docs/  (Theory Reference - Read-Only)
├── 03 - Audio Pipeline/
│   └── 01 - Full Tower v0.md     # KEEP: Definitive theory
└── 00 - Theory/
    └── 04 - Voice Engine Background/
        └── 04 - Lifting the Tower.md  # KEEP: Category theory
```

### 3.2 Proposed Actions

#### Phase 1: Create Entry Point (Priority: HIGH)

Create `docs/standards/architecture/WSE-STD-TOWER-OVERVIEW.md`:
- 1-page quick reference for developers
- Links to authoritative sources for each layer
- Clear "start here" guidance

#### Phase 2: Normalize Tower Levels (Priority: MEDIUM)

Update `symcrash_docs/03 - Audio Pipeline/00 - Voice Engine/Evolution/02 - The Voice Engine Tower.md`:
- Add L5 (Lab) section to match other documents
- Cross-reference WSE-STD-070 as authoritative

#### Phase 3: L5 Specification (Priority: LOW - Deferred)

When L5 implementation begins, create:
- `docs/standards/architecture/WSE-STD-L5-LAB.md`
- Define NewOpDef, NewTreeDef formats
- Specify $C_{5\to3}$ compilation rules

#### Phase 4: Consolidate Roadmap References (Priority: MEDIUM)

Update TOWER-CONFORMANCE-ROADMAP.md:
- Use stable relative paths for cross-references
- Add document authority matrix
- Remove duplicate content, link instead

---

## 4. L5 Functionality Consolidation

### 4.1 Current L5 State

```python
# src/symcrash_python_ap/instrument_ir.py
class TowerLevel(IntEnum):
    L5_LAB = 5  # Enum only, no implementation
```

### 4.2 L5 Required Components (When Implemented)

| Component | Spec Location | Implementation | Status |
|-----------|---------------|----------------|--------|
| NewOpDef format | TBD | `library/op_definitions.py` | Not started |
| NewTreeDef format | TBD | `library/tree_definitions.py` | Not started |
| Lab validation ($V_5$) | TBD | `library/lab_validator.py` | Not started |
| Compilation ($C_{5\to3}$) | TBD | `library/lab_compiler.py` | Not started |

### 4.3 L5 Design Principles (From Full Tower v0)

1. **Must compile down** - Lab objects must produce valid Cockpit instruments
2. **Proof obligation** - New operators must prove they satisfy tail predicate
3. **Offline only** - Lab features never exposed to runtime without compilation
4. **Research mode** - Analysis, regression, mimic tools live here

---

## 5. Implementation Checklist

### Immediate Actions (This Sprint)

- [x] Create `WSE-STD-TOWER-OVERVIEW.md` quick reference (2026-01-20)
- [x] Update Voice Engine Tower doc to include L5 (2026-01-20)
- [x] Add document authority matrix to TOWER-CONFORMANCE-ROADMAP.md (2026-01-20)

### Near-Term (Next Sprint)

- [x] Review all symcrash_docs for L5 references, ensure consistency (2026-01-20)
  - Tower levels (L0-L5) are consistent across Full Tower v0, WSE-STD-070, Voice Engine Tower
  - Note: Different "level" taxonomies exist (Op Levels, Voice Engine Internal Layers, CMMM mapping) - these are distinct from Tower levels
- [x] Add L5 placeholder section to InstrumentIR Tower Conformance Spec (2026-01-20)
- [x] Create issue tracker items for L5 component implementation (2026-01-20) - see below

### L5 Implementation Tracker

| Component | Priority | Spec | Implementation | Test |
|-----------|----------|------|----------------|------|
| NewOpDef schema (YAML) | 🟢 LOW | 🔲 | 🔲 | 🔲 |
| NewOpDef dataclass (Python) | 🟢 LOW | 🔲 | 🔲 | 🔲 |
| NewTreeDef schema (YAML) | 🟢 LOW | 🔲 | 🔲 | 🔲 |
| NewTreeDef dataclass (Python) | 🟢 LOW | 🔲 | 🔲 | 🔲 |
| $V_5$ Lab validator | 🟢 LOW | 🔲 | 🔲 | 🔲 |
| $C_{5\to3}$ compilation | 🟢 LOW | 🔲 | 🔲 | 🔲 |
| WSE-STD-L5-LAB spec | 🟢 LOW | 🔲 | - | - |
| L5 conformance tests | 🟢 LOW | - | 🔲 | 🔲 |

### When L5 Implementation Begins

- [ ] Write WSE-STD-L5-LAB specification
- [ ] Design NewOpDef schema (YAML + Python dataclass)
- [ ] Design NewTreeDef schema
- [ ] Implement $C_{5\to3}$ compilation pathway
- [ ] Add L5 tests to conformance suite

---

## 6. Document Cross-Reference Matrix

| Topic | Authoritative Doc | Supporting Docs |
|-------|-------------------|-----------------|
| Tower level definitions | WSE-STD-070 §3 | Full Tower v0, Voice Engine Tower |
| Preset format | WSE-STD-070 §1 | - |
| Editor primitives | WSE-STD-070 §2 | Full Tower v0 |
| Completion rules | WSE-STD-070 §4 | - |
| Projection maps | WSE-STD-070 §5 | - |
| CMMM definition | Voice Engine Tower §2 | Lifting the Tower |
| Functorial structure | Lifting the Tower | Full Tower v0 |
| IR layer schemas | InstrumentIR Conformance §1-3 | - |
| Emit templates | InstrumentIR Conformance §3 | - |
| Budgeting | InstrumentIR Conformance §4 | CMMM-2 |
| Implementation status | TOWER-CONFORMANCE-ROADMAP | - |
| L5 Lab (when spec'd) | WSE-STD-L5-LAB (future) | Full Tower v0 |

---

## 7. Success Criteria

1. **Single source of truth** for each Tower concept
2. **Clear entry point** for new developers
3. **No duplicate definitions** requiring synchronized updates
4. **L5 ready for implementation** when needed
5. **Stable cross-references** that don't break on file moves

---

## Appendix A: Proposed WSE-STD-TOWER-OVERVIEW.md Outline

```markdown
# Tower Architecture Quick Reference

## Tower Levels
[Link to WSE-STD-070 §3]

| Level | Name | What it adds | Authoritative Spec |
|-------|------|--------------|-------------------|
| L0 | Patch | Static DSP graph | WSE-STD-070 |
| L1 | Family | Gesture semantics | WSE-STD-070 |
| L2 | Instrument | Regime state machine | WSE-STD-070 |
| L3 | Cockpit | CMMM emission | WSE-STD-070, InstrumentIR Conformance |
| L4 | Forge | Graph editing | WSE-STD-070 |
| L5 | Lab | Custom ops/trees | WSE-STD-070, TBD |

## Where to Start
- **Using presets:** Read WSE-STD-070 §1-2
- **Writing instruments:** Read InstrumentIR Conformance
- **Understanding theory:** Read Full Tower v0, Lifting the Tower
- **Implementation status:** See TOWER-CONFORMANCE-ROADMAP

## Key Concepts
- [Completion Rules](WSE-STD-070.md#4-tower-completion-rules)
- [CMMM Definition](../../../symcrash_docs/03%20-%20Audio%20Pipeline/00%20-%20Voice%20Engine/Evolution/02%20-%20The%20Voice%20Engine%20Tower.md)
- [Conformance Profiles](TOWER-CONFORMANCE-ROADMAP.md#3-conformance-profiles)
```

---

## Changelog

### v1.2 (2026-01-20)
- **Plan Execution Complete** - All phases verified and working
- Phase 1: WSE-STD-TOWER-OVERVIEW.md exists with proper cross-references
- Phase 2: Voice Engine Tower updated with L5 section and cross-reference
- Phase 2: TOWER-CONFORMANCE-ROADMAP has Document Authority Matrix (§12)
- Phase 2: InstrumentIR Conformance Spec has L4/L5 placeholder (§9)
- Full Tower v0 verified consistent with L0-L5 definitions
- All cross-references use stable relative paths

### v1.1 (2026-01-20)
- Created WSE-STD-REGISTRY-MASTER-v0.md consolidating all registries
- Updated WSE-STD-CONTROLREGISTRY.md to reference master
- Updated WSE-STD-VOICEENGINEREGISTRY-v0.md to reference master
- Registry consolidation complete

### v1.0 (2026-01-19)
- Initial consolidation plan
- Identified duplication and inconsistencies
- Proposed document hierarchy
- Created L5 implementation checklist
