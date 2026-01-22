# Merge Plan: docs/ into symcrash_docs/

## Status: PHASE 1 COMPLETE

**Goal**: Consolidate `docs/` into `symcrash_docs/` (git submodule), extending its numbered hierarchy.

---

## Completed: Duplicate Resolution

### Files Updated (newer/expanded versions merged into symcrash_docs)

| File | Location | Change |
|------|----------|--------|
| `WSE-STD-070.md` | `symcrash_docs/` | v0 (299 lines) → v1 (1,114 lines) with 19 complexity axes, CustomDef schemas, CMMM bindings |
| `03 - InstrumentIR v0` | `symcrash_docs/02 - Implementation/03 - Core System/` | 514 lines → 1,209 lines with Intent Vector, Gesture Basis, Carrier-First Semantics, CMMM Runtime Control |

### Files Added (new specs from docs/)

| File | Location | Description |
|------|----------|-------------|
| `04 - InstrumentIR Tower Conformance Spec.md` | `symcrash_docs/02 - Implementation/03 - Core System/` | CMMM graph-based state machine, emit templates, budgeting integration |
| `04 - Wire Contract v0b, VoiceFrame Packet v0b.md` | `symcrash_docs/02 - Implementation/03 - Core System/` | Final ABI closure, per-lane payload semantics, carrier header v1 |
| `05 - InstrumentIR Tower Organization Plan.md` | `symcrash_docs/02 - Implementation/03 - Core System/` | Parameter consolidation plan for eliminating hardcoded values |

### Files Unchanged (already identical)

- WSE-STD-001, 010, 020, 030, 040, 050, 060
- Scheduler Spec v0, v01
- Closure Appendix v0.1
- Packet Byte Order v0
- Symbol Executor v0

---

## Remaining: Extend symcrash_docs with [05-09] Sections

Migrate remaining `docs/` content into new numbered sections in `symcrash_docs/`.

---

## Current symcrash_docs Structure

```
symcrash_docs/
├── 00 - Theory/           (22 files - mathematical foundations)
├── 01 - Design/           (31 files - system design specs)
├── 02 - Implementation/   (15 files - dev & integration)
├── 03 - Audio Pipeline/   (64 files - synthesis engines)
├── 04 - Reference/        (6 files - project metadata)
└── WSE-STD-*.md           (8 root-level standards: 001, 010-070)
```

---

## Phase 1: Create New Sections [05-09]

Extend symcrash_docs with new numbered directories for docs/ content:

```
symcrash_docs/
├── [00-04 existing sections unchanged]
│
├── 05 - Standards/                    # FROM: docs/standards/
│   ├── 00 - Core/                     # WSE-STD-001, 020, 030, 200, 201
│   ├── 01 - Architecture/             # WSE-STD-070, 080, 090, 100, 110
│   ├── 02 - Runtime/                  # WSE-STD-040, 050
│   ├── 03 - Voice/                    # FM-PROFILE, SUBTRACTIVE-PROFILE, VOICE-PIPELINE, etc.
│   ├── 04 - Control/                  # COCKPIT-ENGINE, CONTROLFRAME, ENVELOPE-*, MACRO, PRESET
│   ├── 05 - Library/                  # OPERATORS, PRESETLIB, REGISTRY-MASTER, CAPSULE-*
│   ├── 06 - CCS/                      # FM6, BRAID, POLYPHASE-IDWPT, ARCH-CCS
│   ├── 07 - Budget/                   # DEGRADE-v0, MATERIAL-v0, MATERIAL-v0-EXT
│   ├── 08 - Synthesis/                # WAVETABLE-PROFILE
│   ├── 09 - Wavelet/                  # WSE-STD-010
│   └── 10 - Testing/                  # WSE-STD-060, GOLDEN-TEST
│
├── 06 - Roadmaps/                     # FROM: docs/roadmaps/
│   ├── GAP-ANALYSIS-SUMMARY.md
│   ├── THEORETICAL-TECHNIQUES-ROADMAP.md
│   ├── TOWER-CONFORMANCE-ROADMAP.md
│   ├── TOWER-DOCUMENTATION-CONSOLIDATION-PLAN.md
│   ├── VOICE-ENGINE-INTEGRATION-PLAN.md
│   └── WSE-STD-CMM-ROADMAP-v0.md
│
├── 07 - Status Reports/               # FROM: docs/status-reports/
│   ├── 00 - Implementation/           # CCS-STATUS, INTEGRATION-*, PIPELINE-*
│   ├── 01 - Reviews/                  # BUDGET-*, MATERIAL-STANDARD-SUMMARY
│   └── 02 - Testing/                  # CLEANUP-SUMMARY, PIPELINE-TESTS-MIGRATION
│
├── 08 - Analysis/                     # FROM: docs/analysis/
│   └── CCS-BRAM-128KB-32VOICE.md
│
└── 09 - Archived/                     # FROM: docs/archived/
    ├── WSE-ARCHITECTURE-v0.md
    └── WSE-ARCHITECTURE-v1.md
```

---

## Phase 2: Handle Duplicates

### Root-Level Standards (symcrash_docs root → 05 - Standards/)

Move existing root WSE-STD files into the new `05 - Standards/` structure:

| Current Location | New Location |
|------------------|--------------|
| `symcrash_docs/WSE-STD-001.md` | `05 - Standards/00 - Core/WSE-STD-001.md` |
| `symcrash_docs/WSE-STD-010.md` | `05 - Standards/09 - Wavelet/WSE-STD-010.md` |
| `symcrash_docs/WSE-STD-020.md` | `05 - Standards/00 - Core/WSE-STD-020.md` |
| `symcrash_docs/WSE-STD-030.md` | `05 - Standards/00 - Core/WSE-STD-030.md` |
| `symcrash_docs/WSE-STD-040.md` | `05 - Standards/02 - Runtime/WSE-STD-040.md` |
| `symcrash_docs/WSE-STD-050.md` | `05 - Standards/02 - Runtime/WSE-STD-050.md` |
| `symcrash_docs/WSE-STD-060.md` | `05 - Standards/10 - Testing/WSE-STD-060.md` |
| `symcrash_docs/WSE-STD-070.md` | `05 - Standards/01 - Architecture/WSE-STD-070.md` |

**Action**: Compare with docs/standards/ versions. Keep the newer/more complete version.

### Core System Duplicates

Files exist in both:
- `symcrash_docs/02 - Implementation/03 - Core System/`
- `docs/design/core-system/`

| File | Action |
|------|--------|
| Scheduler Spec v0.md | Keep in `02 - Implementation/`, delete from docs/ |
| Scheduler Spec v01.md | Keep in `02 - Implementation/`, delete from docs/ |
| Closure Appendix v0.1.md | Keep in `02 - Implementation/`, delete from docs/ |
| Packet Byte Order v0.md | Keep in `02 - Implementation/`, delete from docs/ |
| InstrumentIR v0.md | Keep in `02 - Implementation/`, delete from docs/ |

**Note**: Verify content is identical before deleting. If docs/ version is newer, copy to symcrash_docs first.

---

## Phase 3: Execution Steps

### Step 1: Work in the Submodule
```bash
cd symcrash_docs
```

### Step 2: Create New Directory Structure
```bash
# Standards hierarchy
mkdir -p "05 - Standards/00 - Core"
mkdir -p "05 - Standards/01 - Architecture"
mkdir -p "05 - Standards/02 - Runtime"
mkdir -p "05 - Standards/03 - Voice"
mkdir -p "05 - Standards/04 - Control"
mkdir -p "05 - Standards/05 - Library"
mkdir -p "05 - Standards/06 - CCS"
mkdir -p "05 - Standards/07 - Budget"
mkdir -p "05 - Standards/08 - Synthesis"
mkdir -p "05 - Standards/09 - Wavelet"
mkdir -p "05 - Standards/10 - Testing"

# Other new sections
mkdir -p "06 - Roadmaps"
mkdir -p "07 - Status Reports/00 - Implementation"
mkdir -p "07 - Status Reports/01 - Reviews"
mkdir -p "07 - Status Reports/02 - Testing"
mkdir -p "08 - Analysis"
mkdir -p "09 - Archived"
```

### Step 3: Move Root Standards into Structure
```bash
mv WSE-STD-001.md "05 - Standards/00 - Core/"
mv WSE-STD-020.md "05 - Standards/00 - Core/"
mv WSE-STD-030.md "05 - Standards/00 - Core/"
mv WSE-STD-010.md "05 - Standards/09 - Wavelet/"
mv WSE-STD-040.md "05 - Standards/02 - Runtime/"
mv WSE-STD-050.md "05 - Standards/02 - Runtime/"
mv WSE-STD-060.md "05 - Standards/10 - Testing/"
mv WSE-STD-070.md "05 - Standards/01 - Architecture/"
```

### Step 4: Copy docs/ Content
```bash
cd ..  # back to repo root

# Standards (copy unique files only)
cp docs/standards/core/WSE-STD-200.md "symcrash_docs/05 - Standards/00 - Core/"
cp docs/standards/core/WSE-STD-201.md "symcrash_docs/05 - Standards/00 - Core/"
cp docs/standards/architecture/WSE-STD-080.md "symcrash_docs/05 - Standards/01 - Architecture/"
cp docs/standards/architecture/WSE-STD-090.md "symcrash_docs/05 - Standards/01 - Architecture/"
cp docs/standards/architecture/WSE-STD-100.md "symcrash_docs/05 - Standards/01 - Architecture/"
cp docs/standards/architecture/WSE-STD-110.md "symcrash_docs/05 - Standards/01 - Architecture/"
cp docs/standards/voice/*.md "symcrash_docs/05 - Standards/03 - Voice/"
cp docs/standards/control/*.md "symcrash_docs/05 - Standards/04 - Control/"
cp docs/standards/library/*.md "symcrash_docs/05 - Standards/05 - Library/"
cp docs/standards/ccs/*.md "symcrash_docs/05 - Standards/06 - CCS/"
cp docs/standards/budget/*.md "symcrash_docs/05 - Standards/07 - Budget/"
cp docs/standards/synthesis/*.md "symcrash_docs/05 - Standards/08 - Synthesis/"
cp docs/standards/testing/WSE-STD-GOLDEN-TEST.md "symcrash_docs/05 - Standards/10 - Testing/"

# Roadmaps
cp docs/roadmaps/*.md "symcrash_docs/06 - Roadmaps/"

# Status Reports
cp docs/status-reports/implementation/*.md "symcrash_docs/07 - Status Reports/00 - Implementation/"
cp docs/status-reports/reviews/*.md "symcrash_docs/07 - Status Reports/01 - Reviews/"
cp docs/status-reports/testing/*.md "symcrash_docs/07 - Status Reports/02 - Testing/"

# Analysis
cp docs/analysis/*.md "symcrash_docs/08 - Analysis/"

# Archived
cp docs/archived/*.md "symcrash_docs/09 - Archived/"
cp docs/WSE-ARCHITECTURE-v1.md "symcrash_docs/09 - Archived/"

# Root guides (add to Reference or create new section)
cp docs/FM6_PATHWAY_GUIDE.md "symcrash_docs/04 - Reference/"
cp docs/fpga_budget_k4_vs_k8.md "symcrash_docs/04 - Reference/"
```

### Step 5: Commit in Submodule
```bash
cd symcrash_docs
git add .
git commit -m "Merge docs/ content into symcrash_docs

- Add 05 - Standards/ with categorized WSE-STD specs
- Add 06 - Roadmaps/ with planning documents
- Add 07 - Status Reports/ with implementation tracking
- Add 08 - Analysis/ with resource budgeting
- Add 09 - Archived/ with historical versions
- Reorganize root WSE-STD files into 05 - Standards/"
git push
```

### Step 6: Update Main Repo
```bash
cd ..
git add symcrash_docs
rm -rf docs/  # Remove after verification
git add -A
git commit -m "Complete merge of docs/ into symcrash_docs submodule"
```

---

## Phase 4: Post-Merge Tasks

1. **Update CLAUDE.md** - Remove references to `docs/` paths, update to `symcrash_docs/` paths
2. **Update internal links** - Search codebase for `docs/` references and update
3. **Create README.md** in `symcrash_docs/05 - Standards/` explaining the authority hierarchy
4. **Verify .obsidian config** - Move `docs/roadmaps/.obsidian/` if needed

---

## Files Requiring Manual Comparison

~~Before executing, diff these files to determine which version to keep:~~ **COMPLETED**

- [x] `WSE-STD-001.md` - identical
- [x] `WSE-STD-010.md` - identical
- [x] `WSE-STD-020.md` - identical
- [x] `WSE-STD-030.md` - identical
- [x] `WSE-STD-040.md` - identical
- [x] `WSE-STD-050.md` - identical
- [x] `WSE-STD-060.md` - identical
- [x] `WSE-STD-070.md` - **updated** (v0 → v1, +815 lines)
- [x] `03 - InstrumentIR v0.md` - **updated** (+695 lines)
- [x] Scheduler Spec v0, v01 - identical
- [x] Closure Appendix v0.1 - identical
- [x] Packet Byte Order v0 - identical
- [x] Symbol Executor v0 - identical

---

## Final Structure Summary

After merge, symcrash_docs will contain:

| Section | Files | Content |
|---------|-------|---------|
| 00 - Theory | 22 | Mathematical foundations |
| 01 - Design | 31 | System design specs |
| 02 - Implementation | 15 | Development & integration |
| 03 - Audio Pipeline | 64 | Synthesis engine docs |
| 04 - Reference | 8 | Project metadata + guides |
| 05 - Standards | ~50 | All WSE-STD specifications |
| 06 - Roadmaps | 6 | Planning documents |
| 07 - Status Reports | 12 | Implementation tracking |
| 08 - Analysis | 1 | Resource budgeting |
| 09 - Archived | 2 | Historical versions |

**Total: ~210 files** in unified symcrash_docs structure
