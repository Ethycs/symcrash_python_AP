#!/usr/bin/env python3
"""
Compile authoring presets (YAML) to capsule bundles.

This script transforms human-editable YAML preset files into compiled
capsule bundles stored in the preset_library.

Usage:
    python scripts/compile_presets.py              # Compile all presets
    python scripts/compile_presets.py brass_stab   # Compile matching presets
    python scripts/compile_presets.py --clean      # Clean and recompile all

Directory structure:
    presets/
        instruments/   <- Instrument preset YAML files
        macros/        <- Macro preset YAML files (vibrato, legato, etc.)

    preset_library/
        capsules/      <- Compiled capsule bundles (output)
        index/         <- Library index
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from symcrash_python_ap.instrument_ir import load_instrument
from symcrash_python_ap.library.compiler import CapsuleCompiler
from symcrash_python_ap.library.preset_library import PresetLibraryV0
from symcrash_python_ap.library.capsule_codec import capsule_to_bytes

PRESETS_DIR = Path(__file__).parent.parent / "presets"
LIBRARY_DIR = Path(__file__).parent.parent / "preset_library"


def compile_preset(yaml_path: Path, library: PresetLibraryV0, verbose: bool = True) -> bool:
    """
    Compile a single preset YAML to a capsule bundle.

    Args:
        yaml_path: Path to the YAML preset file
        library: PresetLibrary to import the compiled capsule into
        verbose: Print progress messages

    Returns:
        True if compilation and import succeeded
    """
    if verbose:
        print(f"Compiling {yaml_path.name}...")

    try:
        # Load InstrumentIR from YAML
        ir = load_instrument(str(yaml_path))
    except Exception as e:
        if verbose:
            print(f"  LOAD ERROR: {e}")
        return False

    # Compile to capsule
    compiler = CapsuleCompiler()
    result = compiler.compile_from_ir(ir)

    if not result.success:
        if verbose:
            print(f"  COMPILE ERROR: {result.errors}")
        return False

    if result.warnings and verbose:
        for warning in result.warnings:
            print(f"  WARNING: {warning}")

    # Convert to binary bundle
    try:
        bundle = capsule_to_bytes(result.capsule)
    except Exception as e:
        if verbose:
            print(f"  SERIALIZE ERROR: {e}")
        return False

    # Import into library
    import_result = library.import_capsule(bundle)

    if not import_result.success:
        if verbose:
            print(f"  IMPORT ERROR: {import_result.errors}")
        return False

    if verbose:
        print(f"  OK -> {result.capsule_id} (rev {import_result.revision_id})")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Compile preset YAML files to capsule bundles"
    )
    parser.add_argument(
        "pattern",
        nargs="?",
        default=None,
        help="Optional pattern to filter preset names (e.g., 'brass_stab')"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean existing capsules before compiling"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    args = parser.parse_args()

    verbose = not args.quiet

    # Clean if requested
    if args.clean:
        import shutil
        capsules_dir = LIBRARY_DIR / "capsules"
        if capsules_dir.exists():
            if verbose:
                print(f"Cleaning {capsules_dir}...")
            shutil.rmtree(capsules_dir)
        index_file = LIBRARY_DIR / "index" / "library_index.json"
        if index_file.exists():
            index_file.unlink()

    # Initialize library
    library = PresetLibraryV0(str(LIBRARY_DIR))

    # Collect YAML files from both instruments/ and macros/
    yaml_files = []
    for subdir in ["instruments", "macros"]:
        subpath = PRESETS_DIR / subdir
        if subpath.exists():
            yaml_files.extend(subpath.glob("*.yaml"))

    if not yaml_files:
        print(f"No YAML files found in {PRESETS_DIR}")
        return 1

    # Filter if pattern provided
    if args.pattern:
        yaml_files = [f for f in yaml_files if args.pattern in f.stem]
        if not yaml_files:
            print(f"No presets matching '{args.pattern}'")
            return 1

    # Compile each preset
    success = 0
    failed = 0
    for yaml_path in sorted(yaml_files):
        if compile_preset(yaml_path, library, verbose=verbose):
            success += 1
        else:
            failed += 1

    # Summary
    if verbose:
        print(f"\nDone: {success} compiled, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
