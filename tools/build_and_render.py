#!/usr/bin/env python3
"""
Build and Render - Unified Preset Pipeline
==========================================

Compiles presets to capsules and renders audio in a single workflow.
Reuses existing scripts:
  - tools/render/compile_presets.py
  - tools/render/render_all_presets.py

Usage:
    pixi run python tools/build_and_render.py                    # Compile + render modified
    pixi run python tools/build_and_render.py brass_stab         # Compile + render matching
    pixi run python tools/build_and_render.py --compile-only     # Just compile
    pixi run python tools/build_and_render.py --render-only      # Just render
    pixi run python tools/build_and_render.py --clean            # Clean everything, rebuild
    pixi run python tools/build_and_render.py --all              # Force all
    pixi run python tools/build_and_render.py --preview          # Also generate preset_library/previews

Output locations:
    preset_library/capsules/       <- Compiled capsule bundles
    preset_library/index/          <- Library index
    preset_library/previews/       <- Quick audio previews (--preview)
    audio_out/presets/             <- Full rendered audio samples
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
COMPILE_SCRIPT = PROJECT_ROOT / "tools" / "render" / "compile_presets.py"
RENDER_SCRIPT = PROJECT_ROOT / "tools" / "render" / "render_all_presets.py"


def run_compile(pattern=None, clean=False, force_all=False, preview=False, quiet=False):
    """
    Run the compile_presets.py script.

    Args:
        pattern: Optional pattern to filter preset names
        clean: If True, clean existing capsules and cache first
        force_all: If True, force recompile all presets
        preview: If True, render audio previews to preset_library/previews
        quiet: If True, suppress progress output

    Returns:
        True if compilation succeeded
    """
    cmd = [sys.executable, str(COMPILE_SCRIPT)]

    if pattern:
        cmd.append(pattern)
    if clean:
        cmd.append("--clean")
    if force_all:
        cmd.append("--all")
    if preview:
        cmd.append("--preview")
    if quiet:
        cmd.append("--quiet")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode == 0


def run_render(pattern=None, clean=False, force_all=False, quiet=False):
    """
    Run the render_all_presets.py script.

    Args:
        pattern: Optional pattern to filter preset names
        clean: If True, clean existing renders and cache first
        force_all: If True, force re-render all presets
        quiet: If True, suppress progress output

    Returns:
        True if rendering succeeded
    """
    cmd = [sys.executable, str(RENDER_SCRIPT)]

    if pattern:
        cmd.append(pattern)
    if clean:
        cmd.append("--clean")
    if force_all:
        cmd.append("--all")
    if quiet:
        cmd.append("--quiet")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Build and render presets in a unified workflow"
    )
    parser.add_argument(
        "pattern",
        nargs="?",
        default=None,
        help="Optional pattern to filter preset names (e.g., 'brass')"
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile presets to capsules, skip rendering"
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Only render audio, skip compilation"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean everything (capsules, renders, caches) and rebuild"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Force recompile/re-render all presets (ignore modification dates)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Also render quick previews to preset_library/previews"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    args = parser.parse_args()

    verbose = not args.quiet

    # Validate mutually exclusive options
    if args.compile_only and args.render_only:
        print("Error: Cannot use both --compile-only and --render-only")
        return 1

    # Track overall success
    compile_ok = True
    render_ok = True

    if args.render_only:
        # Skip compilation
        if verbose:
            print("=" * 60)
            print("RENDERING AUDIO (compile skipped)")
            print("=" * 60)
        render_ok = run_render(args.pattern, args.clean, args.all, args.quiet)

    elif args.compile_only:
        # Skip rendering
        if verbose:
            print("=" * 60)
            print("COMPILING PRESETS (render skipped)")
            print("=" * 60)
        compile_ok = run_compile(args.pattern, args.clean, args.all, args.preview, args.quiet)

    else:
        # Full workflow: compile then render
        if verbose:
            print("=" * 60)
            print("STEP 1: COMPILING PRESETS TO CAPSULES")
            print("=" * 60)
        compile_ok = run_compile(args.pattern, args.clean, args.all, args.preview, args.quiet)

        if verbose:
            print()
            print("=" * 60)
            print("STEP 2: RENDERING AUDIO SAMPLES")
            print("=" * 60)
        render_ok = run_render(args.pattern, args.clean, args.all, args.quiet)

    # Summary
    if verbose:
        print()
        print("=" * 60)
        print("BUILD + RENDER COMPLETE")
        print("=" * 60)
        if not args.render_only:
            status = "OK" if compile_ok else "FAILED"
            print(f"  Compile: {status}")
        if not args.compile_only:
            status = "OK" if render_ok else "FAILED"
            print(f"  Render:  {status}")

        # Show output locations
        print()
        print("Output locations:")
        if not args.render_only:
            print(f"  Capsules: {PROJECT_ROOT / 'preset_library' / 'capsules'}")
            if args.preview:
                print(f"  Previews: {PROJECT_ROOT / 'preset_library' / 'previews'}")
        if not args.compile_only:
            print(f"  Audio:    {PROJECT_ROOT / 'audio_out' / 'presets'}")

    success = compile_ok and render_ok
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
