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

`presets/.build/resources/` also holds the exact Mode B basis atlas. Generate
it with `pixi run python tools/prepare_mode_b_basis.py` before audio startup.
The current pitch domain uses 301,500,612 bytes (about 288 MiB).
Hosts prepare it at initialization when missing. It is independent of preset
capsule compilation, and is mapped read-only during playback. Stop running
hosts before rebuilding an existing image with `--force`, especially on Windows.

Mode A's companion image contains 643 tables (21,072,476 bytes). Prepare it
with `pixi run python tools/prepare_mode_a_tables.py`; hosts also prepare it
at initialization. Both images are disposable build outputs, independent of
musical checkpoints. The Mode A image preserves the existing native and legacy
waveform recipes, including their distinct C4 tables.

The separate legacy `library.preset_model` API supports WSE-STD-070 tower
objects. Its `save_preset`/`load_preset` functions use a 16-byte header plus the
[versioned tower body codec](../symcrash_docs/05%20-%20Standards/05%20-%20Library/WSE-IMPL-TOWER-CODEC-v1.md).
That compatibility format preserves explicit tower layers; current YAML source
and compiled capsule bundles continue to use the formats described above.
