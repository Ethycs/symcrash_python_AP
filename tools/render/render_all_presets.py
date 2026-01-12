#!/usr/bin/env python3
"""
Render All Library Presets
==========================

Generates audio samples for all presets in the library.

Usage:
    pixi run python tools/render/render_all_presets.py              # Render modified presets only
    pixi run python tools/render/render_all_presets.py brass        # Render matching presets
    pixi run python tools/render/render_all_presets.py --clean      # Clean and re-render all
    pixi run python tools/render/render_all_presets.py --all        # Force re-render all

Directory structure:
    presets/
        instruments/   <- Instrument preset YAML files

    audio_out/
        presets/       <- Rendered audio files
        .render_cache.json <- Modification date tracking
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from symcrash_python_ap.instrument_ir import load_instrument
from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.harmonic_bank import clear_basis_atlas
from symcrash_python_ap.vm.state import VMState, OSC_MODE_B, OSC_MODE_FM6
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import AudioWriter

FS = 48000
BLOCK_SIZE = 64
BLOCKS_PER_SEC = FS // BLOCK_SIZE

PRESETS_DIR = Path(__file__).parent.parent.parent / "presets" / "instruments"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "audio_out"
CACHE_FILE = OUTPUT_DIR / ".render_cache.json"


# =============================================================================
# MODIFICATION DATE CACHE
# =============================================================================

def load_cache() -> Dict[str, float]:
    """
    Load the modification date cache.

    Returns:
        Dictionary mapping preset path (relative) -> last rendered mtime.
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


def needs_rerender(yaml_path: Path, cache: Dict[str, float]) -> bool:
    """
    Check if a preset needs re-rendering.

    Args:
        yaml_path: Path to the YAML preset file
        cache: Modification date cache

    Returns:
        True if the file has been modified since last render.
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
# RENDERING
# =============================================================================

def ms_to_blocks(ms):
    return max(1, int(ms / 1000.0 * BLOCKS_PER_SEC))


def render_preset(preset_path, note, duration_sec):
    """Render a single preset to audio."""
    clear_basis_atlas()
    ir = load_instrument(str(preset_path))

    # Extract envelope and synthesis params from L0 graph
    env = {'attack_blocks': 7, 'decay_blocks': 75, 'sustain': 0.7, 'release_blocks': 150}
    waveform = 1  # saw default
    use_fm6 = False
    fm6_params = None

    for node in ir.layers.L0_graph.nodes:
        if node.type == "Amp" and node.params.adsr:
            adsr = node.params.adsr
            env = {
                'attack_blocks': ms_to_blocks(adsr.a_ms),
                'decay_blocks': ms_to_blocks(adsr.d_ms),
                'sustain': adsr.s,
                'release_blocks': ms_to_blocks(adsr.r_ms),
            }
        if node.type == "Osc" and node.params.waveform:
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

    attack_blocks = env['attack_blocks']
    decay_blocks = env['decay_blocks']
    sustain = env['sustain']
    release_blocks = env['release_blocks']

    sustain_blocks = int(duration_sec * BLOCKS_PER_SEC) - attack_blocks
    total_blocks = attack_blocks + sustain_blocks + release_blocks

    attack_level_q15 = 32767
    sustain_level_q15 = int(sustain * 32767)

    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=note)

    # Default harmonic amps for non-FM6 presets
    harmonic_amps = [32767, 16383, 10922, 8191, 6553, 5461, 4681, 4096]
    if hasattr(ir, 'carrier') and ir.carrier and hasattr(ir.carrier, 'harmonic_amps_q15'):
        harmonic_amps = ir.carrier.harmonic_amps_q15

    samples = []

    for block_idx in range(total_blocks):
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
            # Note: fm_feedback_depth expects DX7 level 0-7 (executor converts to Q15)
            feedback_level = fm6_params['feedback']

            frame = VoiceFrame(
                note=note,
                coeff_osc_mode=OSC_MODE_FM6,
                gain_q15=gain_q15,
                beta_target_q15=0,
                fm_algorithm_id=fm6_params['algorithm'],
                fm_op_ratios=fm6_params['op_ratios'],
                fm_op_levels=fm6_params['op_levels'],
                fm_feedback_depth=feedback_level,  # DX7 level 0-7, not Q15
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

    return np.array(samples, dtype=np.int16), ir.meta.name or preset_path.stem


def normalize_to_rms(samples: np.ndarray, target_rms_dbfs: float = -12.0) -> np.ndarray:
    """
    Normalize audio to target RMS level in dBFS.

    Args:
        samples: Audio samples (int16)
        target_rms_dbfs: Target RMS level in dBFS (default -12 dBFS)

    Returns:
        Normalized samples (int16)
    """
    # Convert to float for calculation
    samples_f = samples.astype(np.float64)

    # Calculate current RMS
    rms = np.sqrt(np.mean(samples_f ** 2))
    if rms < 1.0:
        return samples  # Too quiet, don't amplify noise

    # Convert target dBFS to linear
    # 0 dBFS = 32767 RMS for full-scale sine
    # For int16: 0 dBFS peak = 32767, so -12 dBFS RMS ≈ 8211
    target_rms = 32767 * (10 ** (target_rms_dbfs / 20))

    # Calculate gain
    gain = target_rms / rms

    # Apply gain with peak limiting
    normalized = samples_f * gain

    # Soft clip to prevent harsh clipping
    peak = np.max(np.abs(normalized))
    if peak > 32000:  # Leave ~1 dB headroom
        # Apply gentle compression above threshold
        normalized = np.clip(normalized, -32000, 32000)

    return normalized.astype(np.int16)


def main():
    parser = argparse.ArgumentParser(
        description="Render preset YAML files to audio samples"
    )
    parser.add_argument(
        "pattern",
        nargs="?",
        default=None,
        help="Optional pattern to filter preset names (e.g., 'brass')"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean existing renders and cache, then re-render all"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Force re-render all presets (ignore modification dates)"
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
        presets_out_dir = OUTPUT_DIR / "presets"
        if presets_out_dir.exists():
            if verbose:
                print(f"Cleaning {presets_out_dir}...")
            shutil.rmtree(presets_out_dir)
        if CACHE_FILE.exists():
            if verbose:
                print(f"Cleaning {CACHE_FILE}...")
            CACHE_FILE.unlink()

    # Load modification date cache
    cache = load_cache()

    # Collect YAML files
    if not PRESETS_DIR.exists():
        print(f"Presets directory not found: {PRESETS_DIR}")
        return 1

    presets = sorted(PRESETS_DIR.glob("*.yaml"))

    if not presets:
        print(f"No YAML files found in {PRESETS_DIR}")
        return 1

    # Filter if pattern provided
    if args.pattern:
        presets = [f for f in presets if args.pattern.lower() in f.stem.lower()]
        if not presets:
            print(f"No presets matching '{args.pattern}'")
            return 1

    # Filter to only modified files (unless --all or --clean)
    force_all = args.all or args.clean
    if not force_all:
        presets_to_render = [f for f in presets if needs_rerender(f, cache)]
        skipped = len(presets) - len(presets_to_render)
        if skipped > 0 and verbose:
            print(f"Skipping {skipped} unchanged preset(s)")
        presets = presets_to_render

    if not presets:
        if verbose:
            print("All presets up to date.")
        return 0

    # Render each preset
    audio_io = AudioWriter(output_dir=str(OUTPUT_DIR))

    if verbose:
        print(f"Rendering {len(presets)} preset(s)...")
        print("=" * 70)

    success_count = 0
    fail_count = 0

    for preset_path in presets:
        try:
            # Choose appropriate note for each preset type
            name = preset_path.stem
            if 'bass' in name.lower() or 'sub' in name.lower():
                note = 36  # C2
            elif 'pad' in name.lower() or 'string' in name.lower():
                note = 60  # C4
            elif 'bell' in name.lower():
                note = 72  # C5
            else:
                note = 60  # C4 default

            duration = 2.0 if 'pad' in name.lower() or 'string' in name.lower() else 1.5

            samples, preset_name = render_preset(preset_path, note, duration)

            # Calculate pre-normalization RMS for logging
            pre_rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))

            # Normalize to -12 dBFS RMS
            samples = normalize_to_rms(samples, target_rms_dbfs=-12.0)

            # Calculate post-normalization RMS
            post_rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))

            output_name = f"{preset_path.stem}_note{note}"

            metadata = audio_io.write_audio(
                output_name,
                samples.tolist(),
                subdirectory="presets",
                timestamp=False
            )

            if verbose:
                print(f"  OK: {preset_path.name:35} -> {output_name}.wav (RMS: {int(pre_rms):5} -> {int(post_rms):5})")

            success_count += 1
            update_cache_entry(preset_path, cache)

        except Exception as e:
            if verbose:
                print(f"FAIL: {preset_path.name:35} -> {e}")
            fail_count += 1

    # Save updated cache
    save_cache(cache)

    if verbose:
        print("=" * 70)
        print(f"Done: {success_count} rendered, {fail_count} failed")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
