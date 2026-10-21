#!/usr/bin/env python3
"""
Unified Preset Test Harness
============================

Compiles, renders, and verifies presets in a single workflow.
Only processes presets that have been modified since last run.

Usage:
    pixi run python tools/preset_harness.py              # Test modified presets only
    pixi run python tools/preset_harness.py --all        # Test all presets
    pixi run python tools/preset_harness.py brass        # Test matching pattern
    pixi run python tools/preset_harness.py --compile    # Only compile modified
    pixi run python tools/preset_harness.py --render     # Only render modified
    pixi run python tools/preset_harness.py --verify     # Only verify modified
    pixi run python tools/preset_harness.py --clean      # Clean cache, test all

    # Render instruments with macro applied:
    pixi run python tools/preset_harness.py --with-macro vibrato_classic_v0
    pixi run python tools/preset_harness.py violin --with-macro tremolo_fast_v0

Output locations:
    presets/.build/capsules/    <- Compiled capsule bundles
    audio_out/presets/          <- Rendered audio samples
"""

import argparse
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

# Add src and tools to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from lib.preset_cache import (
    get_modified_presets,
    get_all_presets,
    mark_compiled,
    mark_rendered,
    mark_tested,
    clear_cache,
)
from symcrash_python_ap.paths import (
    PRESETS_DIR,
    PRESET_MACROS_DIR as MACROS_DIR,
    PRESET_BUILD_DIR as LIBRARY_DIR,
)

# Paths
AUDIO_OUT_DIR = PROJECT_ROOT / "audio_out" / "presets"


# =============================================================================
# COMPILATION (using compile_presets.py logic directly)
# =============================================================================

@dataclass
class CompileResult:
    """Result of preset compilation."""
    capsule_id: Optional[str] = None
    revision: Optional[int] = None
    budget_ticks: int = 0
    active_nodes: int = 0
    success: bool = False


def do_compile_preset(yaml_path: Path, verbose: bool = True) -> CompileResult:
    """
    Compile a single preset and return capsule_id, revision, and budget info.

    Returns:
        CompileResult with capsule_id, revision, and budget cost
    """
    from symcrash_python_ap.instrument_ir import load_instrument
    from symcrash_python_ap.macro_ir import load_macro, is_macro_preset
    from symcrash_python_ap.library.compiler import CapsuleCompiler
    from symcrash_python_ap.library.preset_library import PresetLibraryV0
    from symcrash_python_ap.library.capsule_codec import capsule_to_bytes

    # Get or create library
    library = PresetLibraryV0(str(LIBRARY_DIR))
    result_obj = CompileResult()

    try:
        # Load and compile
        if is_macro_preset(str(yaml_path)):
            # Macro preset
            macro_ir = load_macro(str(yaml_path))
            compiler = CapsuleCompiler()
            result = compiler.compile_from_macro(macro_ir)
        else:
            # Instrument preset
            ir = load_instrument(str(yaml_path))
            compiler = CapsuleCompiler()
            result = compiler.compile_from_ir(ir)

        if not result.success:
            if verbose:
                print(f"  COMPILE ERROR: {result.errors}")
            return result_obj

        if result.warnings and verbose:
            for warning in result.warnings:
                print(f"  WARNING: {warning}")

        # Import into library directly (no bytes round-trip)
        import_result = library.import_capsule(result.capsule)

        if not import_result.success:
            if verbose:
                print(f"  IMPORT ERROR: {import_result.errors}")
            return result_obj

        # Try to estimate budget cost from capsule
        budget_ticks = 0
        active_nodes = 32  # Default to full
        try:
            from symcrash_python_ap.budgets.budget_predictor import predict_chain_cost
            from symcrash_python_ap.budgets.tail_modes import TailMode

            # Get operators from capsule program if available
            capsule = result.capsule
            if hasattr(capsule, 'program') and capsule.program:
                prog = capsule.program
                backbone_ops = getattr(prog, 'backbone_ops', []) or []
                event_ops = getattr(prog, 'event_ops', []) or []

                if backbone_ops or event_ops:
                    predicted = predict_chain_cost(
                        backbone_ops=backbone_ops,
                        event_ops=event_ops,
                        K=32,  # Full resolution
                        tail_mode=TailMode.FULL,
                    )
                    budget_ticks = predicted.total_ticks
                    active_nodes = predicted.active_nodes_K
        except Exception:
            pass  # Budget estimation is optional

        result_obj.capsule_id = result.capsule_id
        result_obj.revision = import_result.revision_id
        result_obj.budget_ticks = budget_ticks
        result_obj.active_nodes = active_nodes
        result_obj.success = True
        return result_obj

    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")
        return result_obj


# =============================================================================
# RENDERING (replicating render_all_presets.py logic)
# =============================================================================

from symcrash_python_ap.primitives import (
    SAMPLE_RATE_HZ as FS,
    BLOCK_SIZE,
    BLOCK_RATE_HZ as BLOCKS_PER_SEC,
    ms_to_blocks,
)


def normalize_to_rms(samples: list, target_rms_dbfs: float = -12.0) -> list:
    """
    Normalize audio to target RMS level in dBFS.

    Args:
        samples: Audio samples (int16 values as list)
        target_rms_dbfs: Target RMS level in dBFS (default -12 dBFS)

    Returns:
        Normalized samples as list
    """
    import numpy as np

    samples_arr = np.array(samples, dtype=np.float64)

    # Calculate current RMS
    rms = np.sqrt(np.mean(samples_arr ** 2))
    if rms < 1.0:
        return samples  # Too quiet, don't amplify noise

    # Convert target dBFS to linear
    target_rms = 32767 * (10 ** (target_rms_dbfs / 20))

    # Calculate gain
    gain = target_rms / rms

    # Apply gain with peak limiting
    normalized = samples_arr * gain

    # Soft clip to prevent harsh clipping
    peak = np.max(np.abs(normalized))
    if peak > 32000:
        normalized = np.clip(normalized, -32000, 32000)

    return normalized.astype(np.int16).tolist()


def do_render_preset(
    yaml_path: Path,
    output_dir: Path,
    verbose: bool = True,
    macro_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Render a preset to a WAV file.

    Replicates the logic from render_all_presets.py including:
    - FM6 synthesis support
    - RMS normalization to -12 dBFS
    - Proper envelope calculation
    - Waveform type handling
    - Carrier harmonic extraction
    - Optional macro application (vibrato, tremolo)

    Args:
        yaml_path: Path to instrument preset YAML
        output_dir: Output directory for WAV files
        verbose: Print progress messages
        macro_path: Optional path to macro preset to apply during rendering

    Returns:
        Path to the rendered WAV file, or None on failure
    """
    import math
    from symcrash_python_ap.instrument_ir import load_instrument
    from symcrash_python_ap.macro_ir import load_macro, is_macro_preset
    from symcrash_python_ap.vm.state import VMState, OSC_MODE_B, OSC_MODE_FM6
    from symcrash_python_ap.vm.executor import vm_step_block
    from symcrash_python_ap.vm.harmonic_bank import clear_basis_atlas
    from symcrash_python_ap.frames import VoiceFrame
    from symcrash_python_ap.audio_io import AudioWriter
    from symcrash_python_ap.library.capsule import MacroType
    from symcrash_python_ap.macro_runtime import MacroRuntime, SystemMacro

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear basis atlas before each render (prevents stale cache issues)
    clear_basis_atlas()

    try:
        # Load preset
        ir = load_instrument(str(yaml_path))

        # Determine note and duration based on preset name
        name_lower = yaml_path.stem.lower()
        if "bass" in name_lower or "sub" in name_lower:
            note = 36  # C2
        elif "bell" in name_lower:
            note = 72  # C5
        else:
            note = 60  # C4

        duration_sec = 2.0 if ("pad" in name_lower or "string" in name_lower) else 1.5

        # Extract envelope and synthesis params from L0 graph
        env = {'attack_blocks': 7, 'decay_blocks': 75, 'sustain': 0.7, 'release_blocks': 150}
        waveform = 3  # saw default (WaveformType: SINE=0, SQUARE=1, PULSE=2, SAW=3, SAW_DOWN=4, TRIANGLE=5)
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
                from symcrash_python_ap.instrument_ir import WAVEFORM_NAME_TO_ID
                waveform = WAVEFORM_NAME_TO_ID.get(wf, 3)
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

        vm_state = VMState.create(voice_id=0)
        vm_state.reset_for_note_on(note=note)

        # Initialize MacroRuntime for cockpit control integration
        macro_runtime = MacroRuntime()

        # Apply control defaults from preset if present
        if ir.control is not None:
            control_program = ir.control.to_control_program()
            macro_runtime.apply_control_defaults(control_program)
            if verbose:
                brightness = macro_runtime.get_system_macro(SystemMacro.BRIGHTNESS)
                density = macro_runtime.get_system_macro(SystemMacro.DENSITY)
                print(f"    Control: brightness={brightness/327.67:.1f}%, density={density/327.67:.1f}%")

        # Load and configure macro if specified
        macro_program = None
        if macro_path and macro_path.exists():
            try:
                macro_ir = load_macro(str(macro_path))
                macro_program = macro_ir.macro
                # Configure vibrato/tremolo on VMState
                if macro_program.macro_type in (MacroType.VIBRATO, MacroType.TREMOLO):
                    vm_state.mode_b_state.vibrato.configure(
                        rate_hz=macro_program.rate_hz,
                        depth_pitch_cents=macro_program.depth_pitch_cents,
                        depth_amp_q15=macro_program.depth_amp_q15,
                        depth_timbre_q15=macro_program.depth_timbre_q15,
                        delay_ms=macro_program.delay_ms,
                        ramp_ms=macro_program.ramp_ms,
                    )
                    if verbose:
                        print(f"    Applied macro: {macro_path.stem} ({macro_program.macro_type.name})")
            except Exception as e:
                if verbose:
                    print(f"    Warning: Failed to load macro {macro_path}: {e}")

        # Default harmonic amps for non-FM6 presets
        base_harmonic_amps = [32767, 16383, 10922, 8191, 6553, 5461, 4681, 4096]
        if hasattr(ir, 'carrier') and ir.carrier and hasattr(ir.carrier, 'harmonic_amps_q15'):
            base_harmonic_amps = ir.carrier.harmonic_amps_q15

        # Extract partials from Osc node if present
        for node in ir.layers.L0_graph.nodes:
            if node.type == "Osc" and hasattr(node.params, 'partials') and node.params.partials:
                base_harmonic_amps = list(node.params.partials)
                break

        samples = []

        for block_idx in range(total_blocks):
            # Tick macro runtime for slew processing
            macro_runtime.tick()

            # Get current system macro values for synthesis modulation
            brightness_q15 = macro_runtime.get_system_macro(SystemMacro.BRIGHTNESS)
            density_q15 = macro_runtime.get_system_macro(SystemMacro.DENSITY)

            # Apply BRIGHTNESS to harmonic rolloff (higher brightness = more highs)
            # BRIGHTNESS controls a tilt filter: high brightness boosts upper harmonics
            # At brightness=0, upper harmonics are attenuated by 50%
            # At brightness=0.5 (16384), no change
            # At brightness=1.0 (32767), upper harmonics are boosted by 50%
            harmonic_amps = list(base_harmonic_amps)
            if len(harmonic_amps) > 1:
                brightness_norm = (brightness_q15 - 16384) / 16384.0  # -1 to +1
                for i in range(1, len(harmonic_amps)):
                    # Linear tilt based on harmonic index
                    tilt_factor = 1.0 + brightness_norm * 0.5 * (i / (len(harmonic_amps) - 1))
                    harmonic_amps[i] = int(harmonic_amps[i] * max(0.1, min(2.0, tilt_factor)))
                    harmonic_amps[i] = max(0, min(32767, harmonic_amps[i]))

            # Apply DENSITY to low-mid emphasis (higher density = more body)
            # DENSITY boosts fundamental and first few harmonics
            # At density=0.5 (16384), no change
            # At density=1.0 (32767), fundamental boosted by 25%, 2nd by 12%
            if len(harmonic_amps) > 0:
                density_norm = (density_q15 - 16384) / 16384.0  # -1 to +1
                # Boost fundamental and first few harmonics based on density
                density_boost = [0.25, 0.12, 0.06, 0.03]  # Falloff for body emphasis
                for i in range(min(len(density_boost), len(harmonic_amps))):
                    boost_factor = 1.0 + density_norm * density_boost[i]
                    harmonic_amps[i] = int(harmonic_amps[i] * max(0.5, min(1.5, boost_factor)))
                    harmonic_amps[i] = max(0, min(32767, harmonic_amps[i]))

            # Calculate envelope (matching render_all_presets.py logic)
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

            # Apply vibrato/tremolo amplitude modulation if macro is active
            if macro_program and macro_program.macro_type in (MacroType.VIBRATO, MacroType.TREMOLO):
                vib = vm_state.mode_b_state.vibrato
                if vib.enabled:
                    # Update vibrato LFO timing
                    vib.elapsed_samples += BLOCK_SIZE

                    # Update depth envelope (delay + ramp)
                    if vib.elapsed_samples < vib.delay_samples:
                        vib.current_depth_scale = 0
                    elif vib.elapsed_samples < vib.delay_samples + vib.ramp_samples:
                        elapsed_after_delay = vib.elapsed_samples - vib.delay_samples
                        vib.current_depth_scale = int(32767 * elapsed_after_delay / max(1, vib.ramp_samples))
                    else:
                        vib.current_depth_scale = 32767

                    # Apply amplitude modulation from LFO
                    if vib.current_depth_scale > 0 and vib.depth_amp_q15 > 0:
                        # Advance LFO phase
                        vib.phase = (vib.phase + vib.phase_inc * BLOCK_SIZE) & 0xFFFFFFFF
                        phase_norm = vib.phase / (1 << 32)
                        lfo_sin = math.sin(2.0 * math.pi * phase_norm)

                        # Scale by envelope
                        depth_scale = vib.current_depth_scale / 32767.0
                        amp_mod = 1.0 + (vib.depth_amp_q15 / 32767.0) * depth_scale * lfo_sin
                        gain_q15 = int(gain_q15 * amp_mod)
                        gain_q15 = max(0, min(32767, gain_q15))

            if use_fm6 and fm6_params:
                # FM6 synthesis mode
                frame = VoiceFrame(
                    note=note,
                    coeff_osc_mode=OSC_MODE_FM6,
                    gain_q15=gain_q15,
                    beta_target_q15=0,
                    fm_algorithm_id=fm6_params['algorithm'],
                    fm_op_ratios=fm6_params['op_ratios'],
                    fm_op_levels=fm6_params['op_levels'],
                    fm_feedback_depth=fm6_params['feedback'],
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

            vm_state, output = vm_step_block(vm_state, frame)
            samples.extend(output.tolist())

        # Normalize to -12 dBFS RMS
        samples = normalize_to_rms(samples, target_rms_dbfs=-12.0)

        # Save (mono - matches render_all_presets.py output)
        writer = AudioWriter(output_dir=str(output_dir))
        # Include macro name in output if macro was applied
        if macro_path:
            wav_name = f"{yaml_path.stem}_note{note}_{macro_path.stem}"
        else:
            wav_name = f"{yaml_path.stem}_note{note}"
        meta = writer.write_audio(wav_name, samples, subdirectory="", timestamp=False)

        return Path(meta.filepath)

    except Exception as e:
        if verbose:
            print(f"  RENDER ERROR: {e}")
        return None


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_audio(wav_path: Path, min_peak: int = 100, min_duration_sec: float = 0.5) -> Tuple[bool, str]:
    """
    Verify that rendered audio is valid.

    Args:
        wav_path: Path to WAV file
        min_peak: Minimum peak amplitude to pass
        min_duration_sec: Minimum duration in seconds

    Returns:
        (passed, message)
    """
    if not wav_path.exists():
        return False, f"File not found: {wav_path}"

    try:
        with wave.open(str(wav_path), "rb") as wf:
            n_frames = wf.getnframes()
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()

            duration_sec = n_frames / sample_rate

            # Check duration
            if duration_sec < min_duration_sec:
                return False, f"Duration too short: {duration_sec:.2f}s < {min_duration_sec}s"

            # Read audio to check peak
            raw_data = wf.readframes(n_frames)

            # Convert to samples (assuming 16-bit)
            if sample_width == 2:
                import struct
                samples = struct.unpack(f"<{n_frames * n_channels}h", raw_data)
                peak = max(abs(min(samples)), max(samples))
            else:
                peak = 0

            # Check peak
            if peak < min_peak:
                return False, f"Peak too low: {peak} < {min_peak}"

            peak_pct = peak / 32767 * 100
            return True, f"peak={peak} ({peak_pct:.1f}%), duration={duration_sec:.2f}s"

    except Exception as e:
        return False, f"Error reading WAV: {e}"


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_harness(
    pattern: Optional[str] = None,
    force_all: bool = False,
    clean: bool = False,
    compile_only: bool = False,
    render_only: bool = False,
    verify_only: bool = False,
    quiet: bool = False,
    with_macro: Optional[str] = None,
) -> int:
    """
    Run the test harness.

    Args:
        pattern: Optional pattern to filter preset names
        force_all: Process all presets (ignore cache)
        clean: Clean cache and process all
        compile_only: Only compile presets
        render_only: Only render presets
        verify_only: Only verify audio
        quiet: Suppress output
        with_macro: Optional macro preset name to apply during rendering

    Returns:
        0 on success, 1 on failure
    """
    verbose = not quiet

    # Resolve macro path if specified
    macro_path = None
    if with_macro:
        # Try to find the macro file
        macro_name = with_macro
        if not macro_name.endswith('.yaml'):
            macro_name = f"{macro_name}.yaml"
        macro_path = MACROS_DIR / macro_name
        if not macro_path.exists():
            # Try without _v0 suffix
            for candidate in MACROS_DIR.glob("*.yaml"):
                if with_macro in candidate.stem:
                    macro_path = candidate
                    break
        if macro_path and not macro_path.exists():
            macro_path = None

    if verbose:
        print("=" * 70)
        print("PRESET TEST HARNESS")
        if macro_path:
            print(f"  with macro: {macro_path.stem}")
        print("=" * 70)

    # Clean cache if requested
    if clean:
        if verbose:
            print("Cleaning cache...")
        clear_cache(LIBRARY_DIR)
        force_all = True

    # Get presets to process
    if force_all:
        presets = get_all_presets(PRESETS_DIR, pattern)
        if verbose:
            print(f"Processing all {len(presets)} preset(s)")
    else:
        presets = get_modified_presets(
            PRESETS_DIR,
            LIBRARY_DIR,
            pattern,
            check_compiled=not render_only and not verify_only,
            check_rendered=not compile_only and not verify_only,
        )
        if verbose:
            if presets:
                print(f"Found {len(presets)} modified preset(s)")
            else:
                print("All presets up to date")
                return 0

    if not presets:
        if verbose:
            print("No presets to process")
        return 0

    # Track results
    compiled = 0
    rendered = 0
    verified = 0
    failed = 0

    # Determine what to do
    do_compile = not render_only and not verify_only
    do_render = not compile_only and not verify_only
    do_verify = not compile_only and not render_only

    # Compile phase
    max_ticks = 0
    if do_compile:
        if verbose:
            print()
            print("-" * 70)
            print(f"COMPILING ({len(presets)} presets)")
            print("-" * 70)

        for preset_path in presets:
            result = do_compile_preset(preset_path, verbose=verbose)
            if result.success:
                mark_compiled(LIBRARY_DIR, preset_path, PRESETS_DIR, result.capsule_id)
                compiled += 1
                max_ticks = max(max_ticks, result.budget_ticks)
                if verbose:
                    budget_str = f", {result.budget_ticks} ticks/block" if result.budget_ticks > 0 else ""
                    print(f"  {preset_path.name} -> {result.capsule_id} (rev {result.revision}{budget_str}) OK")
            else:
                failed += 1
                if verbose:
                    print(f"  {preset_path.name} FAILED")

        if verbose and max_ticks > 0:
            print(f"  Max ticks/block: {max_ticks}")

    # Render phase
    if do_render:
        from symcrash_python_ap.macro_ir import is_macro_preset

        # Filter presets for rendering with macro
        render_presets = presets
        if macro_path:
            # Only render instrument presets when a macro is specified
            render_presets = [p for p in presets if not is_macro_preset(str(p))]

        if verbose:
            print()
            print("-" * 70)
            macro_str = f" + {macro_path.stem}" if macro_path else ""
            print(f"RENDERING ({len(render_presets)} presets{macro_str})")
            print("-" * 70)

        for preset_path in render_presets:
            wav_path = do_render_preset(
                preset_path,
                AUDIO_OUT_DIR,
                verbose=verbose,
                macro_path=macro_path,
            )
            if wav_path and wav_path.exists():
                mark_rendered(LIBRARY_DIR, preset_path, PRESETS_DIR)
                rendered += 1
                if verbose:
                    print(f"  {preset_path.name} -> {wav_path.name} OK")
            else:
                failed += 1
                if verbose:
                    print(f"  {preset_path.name} FAILED")

    # Verify phase
    if do_verify:
        if verbose:
            print()
            print("-" * 70)
            print(f"VERIFICATION ({len(presets)} presets)")
            print("-" * 70)

        for preset_path in presets:
            # Find the rendered WAV
            wav_name = f"{preset_path.stem}_note60.wav"
            wav_path = AUDIO_OUT_DIR / wav_name

            # Also check for note36 (bass) and note72 (bell)
            if not wav_path.exists():
                wav_name = f"{preset_path.stem}_note36.wav"
                wav_path = AUDIO_OUT_DIR / wav_name
            if not wav_path.exists():
                wav_name = f"{preset_path.stem}_note72.wav"
                wav_path = AUDIO_OUT_DIR / wav_name

            passed, message = verify_audio(wav_path)
            if passed:
                mark_tested(LIBRARY_DIR, preset_path, PRESETS_DIR)
                verified += 1
                if verbose:
                    print(f"  {wav_path.name}: {message} OK")
            else:
                failed += 1
                if verbose:
                    print(f"  {wav_path.name}: {message} FAIL")

    # Summary
    if verbose:
        print()
        print("=" * 70)
        parts = []
        if do_compile:
            parts.append(f"{compiled} compiled")
        if do_render:
            parts.append(f"{rendered} rendered")
        if do_verify:
            parts.append(f"{verified} verified")
        parts.append(f"{failed} failed")
        print(f"RESULTS: {', '.join(parts)}")
        print("=" * 70)

    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="Unified preset test harness"
    )
    parser.add_argument(
        "pattern",
        nargs="?",
        default=None,
        help="Optional pattern to filter preset names"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all presets (ignore modification cache)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean cache and process all"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Only compile presets"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Only render presets"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify rendered audio"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output"
    )
    parser.add_argument(
        "--with-macro",
        type=str,
        default=None,
        help="Apply macro preset during rendering (e.g., vibrato_classic_v0)"
    )
    args = parser.parse_args()

    return run_harness(
        pattern=args.pattern,
        force_all=args.all,
        clean=args.clean,
        compile_only=args.compile,
        render_only=args.render,
        verify_only=args.verify,
        quiet=args.quiet,
        with_macro=args.with_macro,
    )


if __name__ == "__main__":
    sys.exit(main())
