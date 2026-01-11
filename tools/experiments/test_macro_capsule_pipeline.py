"""
Test Macro Capsule Pipeline
============================

Proves the MacroProgram capsule pipeline by:
1. Loading vibrato capsule from YAML
2. Applying MacroProgram to VibratoState
3. Rendering violin with capsule-configured vibrato

This demonstrates that MACRO capsules can be loaded and applied
independently of instrument presets.
"""

import sys
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("WARNING: PyYAML not installed, using fallback values")

from symcrash_python_ap.library.capsule import (
    MacroProgram, MacroType, CapsuleKind, CapsuleV0, CapsuleIdentity
)
from symcrash_python_ap.vm.harmonic_bank import (
    ModeBState, VibratoState, mode_b_generate_coeffs, clear_basis_atlas
)
from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import AudioWriter
from symcrash_python_ap.macro_runtime import MacroRuntime, SystemMacro

# Constants
FS = 48000
BLOCK_SIZE = 64
BLOCKS_PER_SEC = FS // BLOCK_SIZE

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_generation_tests" / "audio_out")
VIBRATO_YAML = Path(__file__).parent.parent / "presets/macros/vibrato_classic_v0.yaml"
VIOLIN_YAML = Path(__file__).parent.parent / "presets/instruments/violin_ir_v1.yaml"


def load_macro_from_yaml(path: Path) -> MacroProgram:
    """Load a MacroProgram from a YAML macro capsule file."""
    if not HAS_YAML:
        # Fallback: return classic vibrato
        return MacroProgram(
            macro_type=MacroType.VIBRATO,
            rate_hz=5.5,
            depth_pitch_cents=25,
            depth_amp_q15=1966,
            depth_timbre_q15=2621,
            delay_ms=40.0,
            ramp_ms=30.0,
        )

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    macro_data = data.get('macro', {})
    macro_type_str = macro_data.get('macro_type', 'VIBRATO')
    macro_type = MacroType[macro_type_str]

    return MacroProgram(
        macro_type=macro_type,
        rate_hz=macro_data.get('rate_hz', 5.5),
        depth_pitch_cents=macro_data.get('depth_pitch_cents', 0),
        depth_amp_q15=macro_data.get('depth_amp_q15', 0),
        depth_timbre_q15=macro_data.get('depth_timbre_q15', 0),
        delay_ms=macro_data.get('delay_ms', 0.0),
        ramp_ms=macro_data.get('ramp_ms', 0.0),
        glide_time_ms=macro_data.get('glide_time_ms', 0.0),
        glide_curve=macro_data.get('glide_curve', 0),
        preserve_phase=macro_data.get('preserve_phase', False),
    )


def apply_macro_to_vibrato_state(macro: MacroProgram, vibrato: VibratoState) -> None:
    """
    Apply a MacroProgram to a VibratoState instance.

    This is the key function that bridges the capsule system to the runtime.
    """
    if macro.macro_type != MacroType.VIBRATO:
        print(f"  Warning: MacroProgram type is {macro.macro_type.name}, expected VIBRATO")
        return

    vibrato.configure(
        rate_hz=macro.rate_hz,
        depth_pitch_cents=macro.depth_pitch_cents,
        depth_amp_q15=macro.depth_amp_q15,
        depth_timbre_q15=macro.depth_timbre_q15,
        delay_ms=macro.delay_ms,
        ramp_ms=macro.ramp_ms,
    )

    print(f"  Applied MacroProgram to VibratoState:")
    print(f"    rate_hz={macro.rate_hz}")
    print(f"    depth_pitch_cents={macro.depth_pitch_cents}")
    print(f"    depth_amp_q15={macro.depth_amp_q15}")
    print(f"    depth_timbre_q15={macro.depth_timbre_q15}")
    print(f"    delay_ms={macro.delay_ms}")
    print(f"    ramp_ms={macro.ramp_ms}")


def load_violin_harmonics() -> list:
    """Load violin harmonic profile from YAML."""
    if HAS_YAML and VIOLIN_YAML.exists():
        with open(VIOLIN_YAML, 'r') as f:
            data = yaml.safe_load(f)
        return data.get('carrier', {}).get('harmonic_amps_q15', [32767])
    # Fallback
    return [30890, 32767, 6926, 23969, 3673, 6138, 1164, 1228]


def render_violin_with_macro_vibrato(
    macro: MacroProgram,
    note: int = 69,
    duration_sec: float = 3.0,
) -> np.ndarray:
    """
    Render a violin note with vibrato configured via MacroProgram capsule.

    This proves the pipeline: YAML -> MacroProgram -> MacroRuntime + VibratoState -> audio
    """
    # Load violin harmonic profile
    base_harmonics = load_violin_harmonics()

    # Setup envelope
    attack_blocks = int(0.03 * BLOCKS_PER_SEC)  # 30ms
    decay_blocks = int(0.25 * BLOCKS_PER_SEC)   # 250ms
    sustain = 0.80
    release_blocks = int(0.55 * BLOCKS_PER_SEC)  # 550ms

    sustain_blocks = int(duration_sec * BLOCKS_PER_SEC) - attack_blocks
    total_blocks = attack_blocks + sustain_blocks + release_blocks

    attack_level_q15 = 32767
    sustain_level_q15 = int(sustain * 32767)

    # Create VM state with vibrato
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=note)

    # Create MacroRuntime and apply capsule to BOTH systems
    macro_runtime = MacroRuntime()

    # 1. Register macro in MacroRuntime (for controller access)
    macro_runtime.apply_macro_capsule(macro)
    print(f"  Registered macro in MacroRuntime: VIBRATO={macro_runtime.get_system_macro(SystemMacro.VIBRATO)}")

    # 2. Apply macro to VibratoState (for actual modulation)
    apply_macro_to_vibrato_state(macro, state.mode_b_state.vibrato)

    samples = []

    print(f"\n  Rendering {total_blocks} blocks...")

    for block_idx in range(total_blocks):
        t_sec = block_idx * BLOCK_SIZE / FS

        # Compute gain envelope (ADSR)
        if block_idx < attack_blocks:
            t = block_idx / attack_blocks
            gain_q15 = int(attack_level_q15 * t)
        elif block_idx < attack_blocks + decay_blocks:
            t = (block_idx - attack_blocks) / decay_blocks
            gain_q15 = int(attack_level_q15 - (attack_level_q15 - sustain_level_q15) * t)
        elif block_idx < attack_blocks + sustain_blocks:
            gain_q15 = sustain_level_q15
        else:
            release_idx = block_idx - (attack_blocks + sustain_blocks)
            t = release_idx / release_blocks
            gain_q15 = int(sustain_level_q15 * (1.0 - t))

        gain_q15 = max(0, min(32767, gain_q15))

        # Get vibrato modulation from VibratoState
        vib = state.mode_b_state.vibrato
        if vib.enabled:
            # Tick the vibrato LFO
            vib.elapsed_samples += BLOCK_SIZE

            # Update depth envelope
            if vib.elapsed_samples < vib.delay_samples:
                vib.current_depth_scale = 0
            elif vib.elapsed_samples < vib.delay_samples + vib.ramp_samples:
                elapsed_after_delay = vib.elapsed_samples - vib.delay_samples
                vib.current_depth_scale = int(32767 * elapsed_after_delay / max(1, vib.ramp_samples))
            else:
                vib.current_depth_scale = 32767

            # Apply LFO
            if vib.current_depth_scale > 0:
                # Advance LFO phase
                vib.phase = (vib.phase + vib.phase_inc * BLOCK_SIZE) & 0xFFFFFFFF

                # Compute sine from phase
                phase_norm = vib.phase / (1 << 32)  # 0.0 to 1.0
                lfo_sin = math.sin(2.0 * math.pi * phase_norm)

                # Scale by envelope
                depth_scale = vib.current_depth_scale / 32767.0

                # Apply amplitude modulation
                if vib.depth_amp_q15 > 0:
                    amp_mod = 1.0 + (vib.depth_amp_q15 / 32767.0) * depth_scale * lfo_sin
                    gain_q15 = int(gain_q15 * amp_mod)
                    gain_q15 = max(0, min(32767, gain_q15))

        # Copy base harmonics (vibrato doesn't modify them in this simple demo)
        harmonic_amps = list(base_harmonics)

        # Create frame
        frame = VoiceFrame(
            note=note,
            coeff_osc_mode=1,  # Mode B
            H_request=len(harmonic_amps),
            gain_q15=gain_q15,
            beta_target_q15=0,
            partial_amps_q15=harmonic_amps,
            ops_enable=[],
            families={},
            waveform_type=0,
        )

        # Render block
        state, output = vm_step_block(state, frame)
        samples.extend(output.tolist())

    return np.array(samples, dtype=np.int16)


def render_comparison():
    """Render violin with and without macro-based vibrato for comparison."""
    print("\n" + "=" * 60)
    print("MACRO CAPSULE PIPELINE TEST")
    print("=" * 60)

    # Clear cached basis frames
    clear_basis_atlas()
    print("\nCleared basis atlas cache")

    # 1. Load the vibrato macro capsule
    print(f"\n1. Loading vibrato capsule from: {VIBRATO_YAML}")
    macro = load_macro_from_yaml(VIBRATO_YAML)
    print(f"   Loaded: {macro.macro_type.name}")
    print(f"   Rate: {macro.rate_hz} Hz")
    print(f"   Pitch depth: {macro.depth_pitch_cents} cents")
    print(f"   Amp depth: {macro.depth_amp_q15} (Q15)")

    # 2. Render with vibrato
    print("\n2. Rendering violin A4 with MACRO VIBRATO...")
    samples_with_vibrato = render_violin_with_macro_vibrato(
        macro=macro,
        note=69,  # A4
        duration_sec=3.0,
    )

    # 3. Render without vibrato (baseline)
    print("\n3. Rendering violin A4 WITHOUT vibrato (baseline)...")
    no_vibrato = MacroProgram(macro_type=MacroType.VIBRATO)  # All depths = 0
    samples_no_vibrato = render_violin_with_macro_vibrato(
        macro=no_vibrato,
        note=69,
        duration_sec=3.0,
    )

    # 4. Save both
    audio_io = AudioWriter(output_dir=OUTPUT_DIR)

    meta_with = audio_io.write_audio(
        "violin_with_macro_vibrato",
        samples_with_vibrato.tolist(),
        subdirectory="macro_test",
        timestamp=False
    )

    meta_without = audio_io.write_audio(
        "violin_no_vibrato_baseline",
        samples_no_vibrato.tolist(),
        subdirectory="macro_test",
        timestamp=False
    )

    print("\n" + "=" * 60)
    print("PIPELINE TEST COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  1. {meta_with.filepath} (with capsule vibrato)")
    print(f"  2. {meta_without.filepath} (baseline)")

    # Analyze difference
    diff = np.abs(samples_with_vibrato.astype(np.float32) - samples_no_vibrato.astype(np.float32))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"\nDifference analysis:")
    print(f"  Max sample difference: {max_diff:.1f}")
    print(f"  Mean sample difference: {mean_diff:.1f}")

    if mean_diff > 100:
        print(f"\n  SUCCESS: Vibrato modulation detected (mean diff > 100)")
    else:
        print(f"\n  WARNING: Vibrato effect may be too subtle or not applied")

    return samples_with_vibrato, samples_no_vibrato


if __name__ == "__main__":
    render_comparison()
