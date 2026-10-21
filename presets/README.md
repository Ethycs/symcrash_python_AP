# Presets

This is the repository's single preset root.

## Tracked source

- `instruments/` — human-editable `instrument_ir_v0` YAML.
- `macros/` — human-editable `macro_ir_v0` YAML.
- `synths/` — reserved for human-editable `synth_ir_v0` YAML.

These YAML files are the source of truth. Edit, review, and commit them.

## Generated library

`presets/.build/` is disposable and ignored by Git. It contains compiled
capsule bundles, the runtime index, caches, and optional WAV previews. Build
it from the YAML sources with:

```powershell
pixi run python tools/render/compile_presets.py --clean
```

Runtime and workbench code obtain these locations from
`symcrash_python_ap.paths`; tools must not invent another top-level preset or
capsule directory.

The former top-level `preset_library/` and `capsules/` paths are retired.
