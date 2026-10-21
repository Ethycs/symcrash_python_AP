# Symcrash Open Stage Control workbench

The workbench is one generated Open Stage Control session with seven screens:

- **PLAY** — preset browser, transport, keyboard, amp, filter, noise, unison.
- **PATCH** — the preset's authored L0 node/output graph.
- **FM6** — algorithm selector, visible operator cords, operator controls.
- **MOD MAP** — named envelopes patched to FM, gain, filter, pitch, and material targets.
- **BEHAVIOR** — state-transition graph plus attack/stable authoring axes.
- **SYSTEM** — the eleven live controller-owned hidden axes.
- **SOURCE** — diagnostics and a full comment-preserving YAML editor.

## Launch

In the repository root, start the synth backend:

```powershell
pixi run workbench dx_marimba_v0
```

Then start Open Stage Control in a second terminal. This Pixi task downloads
the verified portable Windows build into `.pixi/` on first use, so no PATH or
administrator installation is required:

```powershell
pixi run workbench-ui
```

Regenerate the session after changing the control registry or layout code:

```powershell
pixi run python -m symcrash_workbench --write-layout
```

The generated JSON is ignored; the backend creates it automatically when it
is missing. Its local path is `tools/workbench/symcrash-workbench.json`.

## Editing model

Knobs and accepted cords change the in-memory `InstrumentIR`. **SAVE** applies
all pending scalar and structural edits to the backing YAML in one transaction
and increments `meta.version`. **RELOAD** discards pending edits and rebuilds
the surface from disk. Invalid cords are rejected, reported in SOURCE, and
immediately snapped back to the last accepted state.

The PATCH screen uses eight stable UI slots that remap whenever a preset is
loaded. The sound-design YAML has one `output` per node, so the backend accepts
one downstream node per source and rejects cycles. This graph is currently an
authoring/serialization view: the simple live engine still chooses its DSP
stages by node type, not by arbitrary L0 edge traversal.

The FM6 and MOD MAP screens are audible at the next 64-sample block. The FM6
cords are read-only because the algorithm number is the canonical topology;
selecting a new algorithm redraws them. Envelope inputs may fan out to many
targets. Behavior topology is used by the full/offline regime renderer; the
streaming voice currently maps note phases directly to the behavior templates.

Presets without an explicit behavior state graph show empty state slots.
**MAKE DEFAULT GRAPH** materializes the standard idle → attack → stable
skeleton plus the any-state gate-release edge. It remains unsaved until SAVE.

The SOURCE editor is served on loopback port 7410 and embedded as an Open Stage
Control frame. **VALIDATE** runs the exact `InstrumentIR` loader without
changing anything. **APPLY + RELOAD** validates, atomically replaces the YAML,
increments `meta.version`, and retriggers a streaming voice at the next block.
Use `--source-port 0` to disable the embedded editor. If a different source port
is selected, update the frame URL in the generated layout as well.
