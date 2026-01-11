#!/usr/bin/env python3
"""
Compile authoring presets (YAML) to capsule bundles.

This script transforms human-editable YAML preset files into compiled
capsule bundles stored in the preset_library.

Usage:
    python scripts/compile_presets.py              # Compile modified presets only
    python scripts/compile_presets.py brass_stab   # Compile matching presets
    python scripts/compile_presets.py --clean      # Clean and recompile all
    python scripts/compile_presets.py --all        # Force recompile all
    python scripts/compile_presets.py --preview    # Render audio previews

Directory structure:
    presets/
        instruments/   <- Instrument preset YAML files (instrument_ir_v0)
        synths/        <- Synth preset YAML files (synth_ir_v0)
        macros/        <- Macro preset YAML files (macro_ir_v0)

    preset_library/
        capsules/
            instruments/   <- Compiled instrument capsules (CapsuleKind.INSTRUMENT)
            synths/        <- Compiled synth capsules (CapsuleKind.SYNTH)
            assets/        <- Compiled asset capsules (CapsuleKind.ASSET)
            macros/        <- Compiled macro capsules (CapsuleKind.MACRO)
        index/         <- Library index
        .compile_cache.json <- Modification date tracking
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from symcrash_python_ap.instrument_ir import load_instrument
from symcrash_python_ap.macro_ir import load_macro, is_macro_preset
from symcrash_python_ap.library.compiler import CapsuleCompiler
from symcrash_python_ap.library.preset_library import PresetLibraryV0
from symcrash_python_ap.library.capsule_codec import capsule_to_bytes

PRESETS_DIR = Path(__file__).parent.parent / "presets"
LIBRARY_DIR = Path(__file__).parent.parent / "preset_library"
CACHE_FILE = LIBRARY_DIR / ".compile_cache.json"


# =============================================================================
# MODIFICATION DATE CACHE
# =============================================================================

def load_cache() -> Dict[str, float]:
    """
    Load the modification date cache.

    Returns:
        Dictionary mapping preset path (relative) -> last compiled mtime.
    """
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_cache(cache: Dict[str, float]) -> None:
    """Save the modification date cache."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def get_relative_path(yaml_path: Path) -> str:
    """Get path relative to PRESETS_DIR for cache key."""
    try:
        return str(yaml_path.relative_to(PRESETS_DIR))
    except ValueError:
        return str(yaml_path)


def needs_recompile(yaml_path: Path, cache: Dict[str, float]) -> bool:
    """
    Check if a preset needs recompilation.

    Args:
        yaml_path: Path to the YAML preset file
        cache: Modification date cache

    Returns:
        True if the file has been modified since last compilation.
    """
    rel_path = get_relative_path(yaml_path)
    current_mtime = yaml_path.stat().st_mtime

    if rel_path not in cache:
        return True

    return current_mtime > cache[rel_path]


def update_cache_entry(yaml_path: Path, cache: Dict[str, float]) -> None:
    """Update the cache with the current file's modification time."""
    rel_path = get_relative_path(yaml_path)
    cache[rel_path] = yaml_path.stat().st_mtime


# =============================================================================
# AUDIO PREVIEW RENDERING
# =============================================================================

def apply_adsr(audio: 'np.ndarray', sample_rate: int,
               a_ms: float, d_ms: float, s: float, r_ms: float) -> 'np.ndarray':
    """
    Apply ADSR envelope to audio (from render_brass_presets.py).

    Args:
        audio: Input audio samples
        sample_rate: Sample rate in Hz
        a_ms: Attack time in milliseconds
        d_ms: Decay time in milliseconds
        s: Sustain level (0.0 to 1.0)
        r_ms: Release time in milliseconds

    Returns:
        Audio with envelope applied
    """
    import numpy as np

    total_samples = len(audio)

    a_samples = int((a_ms / 1000) * sample_rate)
    d_samples = int((d_ms / 1000) * sample_rate)
    r_samples = int((r_ms / 1000) * sample_rate)

    # Sustain duration = total - attack - decay - release
    s_samples = max(0, total_samples - a_samples - d_samples - r_samples)

    envelope = np.zeros(total_samples, dtype=np.float64)
    idx = 0

    # Attack
    if a_samples > 0 and idx < total_samples:
        attack_end = min(idx + a_samples, total_samples)
        envelope[idx:attack_end] = np.linspace(0, 1, attack_end - idx)
        idx = attack_end

    # Decay
    if d_samples > 0 and idx < total_samples:
        decay_end = min(idx + d_samples, total_samples)
        envelope[idx:decay_end] = np.linspace(1, s, decay_end - idx)
        idx = decay_end

    # Sustain
    if s_samples > 0 and idx < total_samples:
        sustain_end = min(idx + s_samples, total_samples)
        envelope[idx:sustain_end] = s
        idx = sustain_end

    # Release
    if r_samples > 0 and idx < total_samples:
        release_end = min(idx + r_samples, total_samples)
        envelope[idx:release_end] = np.linspace(s, 0, release_end - idx)
        idx = release_end

    # Apply envelope
    return (audio.astype(np.float64) * envelope).astype(np.int16)


def render_preview(
    ir,
    capsule_id: str,
    duration_sec: float = 2.0,
    verbose: bool = True
) -> Optional[str]:
    """
    Render an audio preview for a compiled preset.

    Uses vm_step_block with VoiceFrame for both FM6 and Mode B synthesis.
    Extracts envelope and waveform parameters from the IR's L0_graph.

    Args:
        ir: InstrumentIR object
        capsule_id: Capsule ID for the output filename
        duration_sec: Preview duration in seconds
        verbose: Print progress messages

    Returns:
        Path to the rendered WAV file, or None if failed.
    """
    try:
        from symcrash_python_ap.audio_io import AudioWriter
        from symcrash_python_ap.vm.executor import vm_step_block
        from symcrash_python_ap.vm.harmonic_bank import clear_basis_atlas
        from symcrash_python_ap.vm.state import VMState, OSC_MODE_B, OSC_MODE_FM6
        from symcrash_python_ap.frames import VoiceFrame
        import numpy as np

        sample_rate = 48000
        block_size = 64
        blocks_per_sec = sample_rate // block_size

        def ms_to_blocks(ms):
            return max(1, int(ms / 1000.0 * blocks_per_sec))

        # Clear basis atlas for clean render
        clear_basis_atlas()

        # Extract envelope and synthesis params from L0 graph
        env = {'attack_blocks': 7, 'decay_blocks': 75, 'sustain': 0.7, 'release_blocks': 150}
        waveform = 0  # sine default
        use_fm6 = False
        fm6_params = None

        if hasattr(ir, 'layers') and hasattr(ir.layers, 'L0_graph'):
            for node in ir.layers.L0_graph.nodes:
                if node.type == "Amp" and hasattr(node.params, 'adsr') and node.params.adsr:
                    adsr = node.params.adsr
                    env = {
                        'attack_blocks': ms_to_blocks(adsr.a_ms),
                        'decay_blocks': ms_to_blocks(adsr.d_ms),
                        'sustain': adsr.s,
                        'release_blocks': ms_to_blocks(adsr.r_ms),
                    }
                if node.type == "Osc" and hasattr(node.params, 'waveform') and node.params.waveform:
                    wf = node.params.waveform.lower()
                    # WaveformType: SINE=0, SQUARE=1, PULSE=2, SAW=3, SAW_DOWN=4, TRIANGLE=5
                    waveform = {'sine': 0, 'square': 1, 'pulse': 2, 'saw': 3, 'tri': 5, 'triangle': 5}.get(wf, 0)
                if node.type == "FM6":
                    use_fm6 = True
                    fm6_params = {
                        'algorithm': getattr(node.params, 'algorithm', 1),
                        'feedback': getattr(node.params, 'feedback', 0),
                        'op_ratios': getattr(node.params, 'op_ratios', [65536] * 6),
                        'op_levels': getattr(node.params, 'op_levels', [32767, 0, 0, 0, 0, 0]),
                    }

        # Choose appropriate note based on preset name
        name = capsule_id.lower()
        if 'bass' in name or 'sub' in name:
            note = 36  # C2
        elif 'bell' in name:
            note = 72  # C5
        else:
            note = 60  # C4

        attack_blocks = env['attack_blocks']
        decay_blocks = env['decay_blocks']
        sustain = env['sustain']
        release_blocks = env['release_blocks']

        sustain_blocks = int(duration_sec * blocks_per_sec) - attack_blocks
        total_blocks = attack_blocks + sustain_blocks + release_blocks

        attack_level_q15 = 32767
        sustain_level_q15 = int(sustain * 32767)

        state = VMState.create(voice_id=0)
        state.reset_for_note_on(note=note)

        # Default harmonic amps for non-FM6 presets
        harmonic_amps = [32767, 16383, 10922, 8191, 6553, 5461, 4681, 4096]

        samples = []

        for block_idx in range(total_blocks):
            # Calculate envelope gain
            if block_idx < attack_blocks:
                t = block_idx / attack_blocks if attack_blocks > 0 else 1
                gain_q15 = int(attack_level_q15 * t)
            elif block_idx < attack_blocks + decay_blocks:
                t = (block_idx - attack_blocks) / decay_blocks if decay_blocks > 0 else 1
                gain_q15 = int(attack_level_q15 - (attack_level_q15 - sustain_level_q15) * t)
            elif block_idx < attack_blocks + sustain_blocks:
                gain_q15 = sustain_level_q15
            else:
                release_idx = block_idx - (attack_blocks + sustain_blocks)
                t = release_idx / release_blocks if release_blocks > 0 else 1
                gain_q15 = int(sustain_level_q15 * (1.0 - t))

            gain_q15 = max(0, min(32767, gain_q15))

            if use_fm6 and fm6_params:
                # FM6 synthesis mode
                # Pass DX7 feedback level (0-7) directly - executor converts to Q15
                feedback_level = min(7, max(0, fm6_params['feedback']))

                frame = VoiceFrame(
                    note=note,
                    coeff_osc_mode=OSC_MODE_FM6,
                    gain_q15=gain_q15,
                    beta_target_q15=0,
                    fm_algorithm_id=fm6_params['algorithm'],
                    fm_op_ratios=fm6_params['op_ratios'],
                    fm_op_levels=fm6_params['op_levels'],
                    fm_feedback_depth=feedback_level,
                )
            else:
                # Standard Mode B synthesis
                frame = VoiceFrame(
                    note=note,
                    coeff_osc_mode=OSC_MODE_B,
                    H_request=len(harmonic_amps),
                    gain_q15=gain_q15,
                    beta_target_q15=0,
                    partial_amps_q15=harmonic_amps,
                    waveform_type=waveform,
                )

            state, output = vm_step_block(state, frame)
            samples.extend(output.tolist())

        audio = np.array(samples, dtype=np.int16)
        L = audio
        R = audio  # Mono duplicated to stereo

        # Write using audio_io
        writer = AudioWriter(
            output_dir=str(LIBRARY_DIR / "previews"),
            sample_rate=sample_rate
        )

        metadata = writer.write_audio(
            filename=capsule_id,
            L=L,
            R=R,
            timestamp=False,
            verbose=verbose
        )

        return metadata.filepath

    except ImportError as e:
        if verbose:
            print(f"  Preview skipped (missing dependency): {e}")
        return None
    except Exception as e:
        if verbose:
            print(f"  Preview failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# PRESET COMPILATION
# =============================================================================

def compile_instrument(yaml_path: Path, library: PresetLibraryV0, verbose: bool = True, render_audio: bool = False) -> bool:
    """
    Compile an instrument preset YAML to a capsule bundle.

    Args:
        yaml_path: Path to the YAML preset file
        library: PresetLibrary to import the compiled capsule into
        verbose: Print progress messages
        render_audio: If True, render an audio preview using audio_io

    Returns:
        True if compilation and import succeeded
    """
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

    # Render audio preview if requested
    if render_audio:
        preview_path = render_preview(ir, result.capsule_id, verbose=verbose)
        if preview_path and verbose:
            print(f"  Preview: {preview_path}")

    return True


def compile_macro(yaml_path: Path, library: PresetLibraryV0, verbose: bool = True) -> bool:
    """
    Compile a macro preset YAML to a capsule bundle.

    Args:
        yaml_path: Path to the YAML preset file
        library: PresetLibrary to import the compiled capsule into
        verbose: Print progress messages

    Returns:
        True if compilation and import succeeded
    """
    try:
        # Load MacroIR from YAML
        macro_ir = load_macro(str(yaml_path))
    except Exception as e:
        if verbose:
            print(f"  LOAD ERROR: {e}")
        return False

    # Validate
    is_valid, error = macro_ir.validate()
    if not is_valid:
        if verbose:
            print(f"  VALIDATION ERROR: {error}")
        return False

    # Convert to capsule
    try:
        capsule = macro_ir.to_capsule()
        capsule_id = capsule.identity.capsule_id
    except Exception as e:
        if verbose:
            print(f"  COMPILE ERROR: {e}")
        return False

    # Convert to binary bundle
    try:
        bundle = capsule_to_bytes(capsule)
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
        print(f"  OK -> {capsule_id} (rev {import_result.revision_id})")

    return True


def compile_preset(yaml_path: Path, library: PresetLibraryV0, verbose: bool = True, render_audio: bool = False) -> bool:
    """
    Compile a single preset YAML to a capsule bundle.

    Automatically detects whether the preset is an instrument or macro.

    Args:
        yaml_path: Path to the YAML preset file
        library: PresetLibrary to import the compiled capsule into
        verbose: Print progress messages
        render_audio: If True, render an audio preview using audio_io (instruments only)

    Returns:
        True if compilation and import succeeded
    """
    if verbose:
        print(f"Compiling {yaml_path.name}...")

    # Detect preset type and route to appropriate compiler
    try:
        if is_macro_preset(str(yaml_path)):
            return compile_macro(yaml_path, library, verbose=verbose)
        else:
            # Default to instrument
            return compile_instrument(yaml_path, library, verbose=verbose, render_audio=render_audio)
    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")
        return False


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
        help="Clean existing capsules and cache, then recompile all"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Force recompile all presets (ignore modification dates)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Render audio previews for compiled presets (uses audio_io)"
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
        # Also clean the cache
        if CACHE_FILE.exists():
            if verbose:
                print(f"Cleaning {CACHE_FILE}...")
            CACHE_FILE.unlink()
        # Clean previews
        previews_dir = LIBRARY_DIR / "previews"
        if previews_dir.exists():
            if verbose:
                print(f"Cleaning {previews_dir}...")
            shutil.rmtree(previews_dir)

    # Load modification date cache
    cache = load_cache()

    # Initialize library
    library = PresetLibraryV0(str(LIBRARY_DIR))

    # Collect YAML files from instruments/, synths/, and macros/
    yaml_files = []
    for subdir in ["instruments", "synths", "macros"]:
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

    # Filter to only modified files (unless --all or --clean)
    force_all = args.all or args.clean
    if not force_all:
        yaml_files_to_compile = [f for f in yaml_files if needs_recompile(f, cache)]
        skipped = len(yaml_files) - len(yaml_files_to_compile)
        if skipped > 0 and verbose:
            print(f"Skipping {skipped} unchanged preset(s)")
        yaml_files = yaml_files_to_compile

    if not yaml_files:
        if verbose:
            print("All presets up to date.")
        return 0

    # Compile each preset
    success = 0
    failed = 0
    for yaml_path in sorted(yaml_files):
        if compile_preset(yaml_path, library, verbose=verbose, render_audio=args.preview):
            success += 1
            # Update cache on successful compilation
            update_cache_entry(yaml_path, cache)
        else:
            failed += 1

    # Save updated cache
    save_cache(cache)

    # Summary
    if verbose:
        print(f"\nDone: {success} compiled, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
