# Instrument Optimization Workflow — Handoff

## Goal
Author presets by **fitting them to real instrument waveforms** with a closed,
numerical loop (no listening required). Parse a target recording → extract its
harmonic spectrum + envelope → render a candidate preset *through the real
engine* → measure the distance → optimize the preset parameters to minimize it.
The residual loss is also an honest measure of the engine's expressive ceiling.

## What already exists (validated)
- **Harness:** `tools/optimize_instrument.py` — `analyze_target → render_steady
  (through vm_step_block) → extract_harmonics → harmonic_loss → optimize (Powell)
  → write_preset`. Runs end-to-end; writes a preset YAML with a loss header.
- **Target:** `violin.wav` (repo root): 44.1 kHz, 2.5 s, A=440 (MIDI 69).
  Measured harmonics H1..H12: `1.000 0.093 0.085 0.177 0.275 0.136 0.021 0.093
  0.024 0.012 0.015 0.004` (dominant fundamental, weak H2/H3, formant bump at
  H4–H6). More targets: `audio_out/sampler_tests/violin_*.wav`.
- **The loop is proven to close:** with NO operators and `partials = target`,
  the engine reproduces the spectrum to **loss 0.001** (near-perfect). With
  `color+body` operators the spectrum is colored to **loss 0.198** — genuine
  operator coloration (e.g. H3 0.08→0.23) that the optimizer must deconvolve.

## Critical engine facts (learned the hard way — honor these)
1. **`waveform_type=0` is REQUIRED for partials to take effect.** Any non-zero
   `waveform_type` forces a built-in waveform that OVERRIDES `custom_harmonics`
   (priority in `vm/executor.py:_setup_mode_b_harmonic_weights` is
   `OEIS > custom_harmonics > waveform_type`). The harness already sets 0.
2. `VoiceFrame.partial_amps_q15` → `params.custom_harmonics` only when the list
   is non-empty (`vm/decode.py:310`).
3. **Operators reach audio via the emit template.** `render_steady` derives
   operators from the axis dict via `derive_operators_from_emphasis` (continuous
   7-axis model — see `docs/jargon.md`). Axes: color, body, spread, warp, glue,
   snap, chaos. NormalizeScale+FineClamp are always-on safety.

## Known bugs to fix (found during setup — fix as part of this work)
- `tools/render/render_preset.py` renders `waveform: partials` presets as a
  **sawtooth**: `extract_waveform` now returns 0 for "partials" (fixed), but
  `render_preset_note` still uses a **hardcoded** `harmonic_amps` (line ~151)
  instead of the preset's partials.
- **The preset's `partials:` are not loaded into the IR** — `Osc.params.harmonics`
  is `None` for `violin_ir_v1` despite the YAML listing them. Trace the L0 graph
  loader in `instrument_ir.py` and wire `partials:` → the Osc node so render and
  capsule paths see them. (The optimizer bypasses this by passing partials
  directly, so it is unaffected, but presets it writes won't render correctly
  until this is fixed.)
- `violin_ir_v1.yaml`'s "from real analysis" partials are wrong (H2 = 1.000 vs
  real 0.093). Re-fit it with this loop as the first deliverable.

## Tasks (in priority order)
1. **Make the optimizer actually converge.** Powell stalls with operators on
   (12-dim, noisy objective). Try: better init (start from target), Nelder-Mead
   with restarts, coordinate descent over partials, or `cma` (CMA-ES) if
   available. Target: with `color+body` operators, drive loss from 0.198 toward
   ~0.02. Add an envelope term to the loss (attack/sustain/release) if you also
   optimize ADSR.
2. **Produce a re-fitted `violin_opt_v0.yaml`** that beats `violin_ir_v1` on
   harmonic loss; render it and report before/after loss + a harmonic table.
3. **Generalize:** run the loop on the other `audio_out/sampler_tests/violin_*`
   targets; confirm the workflow is target-agnostic. Document any instrument the
   engine *cannot* match well (its expressive ceiling).
4. **Fix the two render/loader bugs above** so optimized presets render
   correctly in `render_preset.py` and load their partials.
5. Keep everything reproducible: no `Math.random`/wall-clock seeds; deterministic
   runs. Use `pixi run python ...`.

## How to run
```
pixi run python tools/optimize_instrument.py violin.wav --note 69 \
    --harmonics 12 --iters 400 --axes "color:0.3,body:0.2" \
    --out presets/instruments/violin_opt_v0.yaml --name "Violin Opt"
```

## Deliverables
- A converging optimizer (loss curve in the run output).
- `presets/instruments/violin_opt_v0.yaml` beating the current violin on loss.
- The two bug fixes (render partials + IR loader partials).
- A short report: per-target before/after loss, and the engine's matching limits.
- Do NOT touch the frozen WSE-STD specs or commit the `symcrash_docs` submodule.
```
