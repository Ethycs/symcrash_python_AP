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

`presets/.build/resources/` also holds the fixed-anchor Mode B basis atlas. Generate
it with `pixi run python tools/prepare_mode_b_basis.py` before audio startup.
Its 128 source anchors use 3,124,940 bytes and cover the full integer-cent
pitch domain by fixed-point interpolation. The accepted Q32 phase is unchanged.
The former 301,500,612-byte exhaustive image is superseded and cannot be
adopted by the v3 loader. See the [arithmetic/resource profile](../symcrash_docs/05%20-%20Standards/06%20-%20CCS/WSE-IMPL-BASIS-RESOURCE-v3.md).
Hosts prepare it at initialization when missing. It is independent of preset
capsule compilation, and is mapped read-only during playback. Stop running
hosts before rebuilding an existing image with `--force`, especially on Windows.

Mode A's companion image contains 63,508 tables (2,081,284,256 bytes). Prepare it
with `pixi run python tools/prepare_mode_a_tables.py`; hosts also prepare it
at initialization. Both images are disposable build outputs, independent of
musical checkpoints. Mode A v2 covers every shared integer-cent Q32 pitch,
with a pitch-dependent harmonic cap and three distinct legacy C4 tables.
The old 21 MB v1 image is superseded; its fingerprint cannot be adopted as v2.
The offline builder uses up to eight workers; `--workers 1` selects serial
generation with identical bytes.

The separate legacy `library.preset_model` API supports WSE-STD-070 tower
objects. Its `save_preset`/`load_preset` functions use a 16-byte header plus the
[versioned tower body codec](../symcrash_docs/05%20-%20Standards/05%20-%20Library/WSE-IMPL-TOWER-CODEC-v1.md).
That compatibility format preserves explicit tower layers; current YAML source
and compiled capsule bundles continue to use the formats described above.
