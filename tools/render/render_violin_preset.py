"""
Render Violin Preset from YAML Specification
=============================================

Loads and renders the violin_ir_v1.yaml preset with:
- Expressive bowed-string character
- Bow pressure vibrato (5.5 Hz)
- Body resonance warble
- Attack brightness that settles
"""

import sys
import math
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Try to import yaml, fall back to JSON
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.harmonic_bank import clear_basis_atlas
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import AudioWriter

# Constants
FS = 48000
BLOCK_SIZE = 64
BLOCKS_PER_SEC = FS // BLOCK_SIZE

OUTPUT_DIR = str(Path(__file__).parent.parent.parent / "audio_out")
PRESET_PATH = Path(__file__).parent.parent.parent / "presets/instruments/violin_ir_v1.yaml"
PRESET_PATH_JSON = None  # JSON duplicates removed - use YAML only


def load_violin_preset() -> dict:
    """Load the violin preset from YAML or JSON file."""
    # Try YAML first if available
    if HAS_YAML and PRESET_PATH.exists():
        with open(PRESET_PATH, 'r') as f:
            return yaml.safe_load(f)
    # Fall back to JSON
    elif PRESET_PATH_JSON.exists():
        with open(PRESET_PATH_JSON, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(
            f"Neither {PRESET_PATH} nor {PRESET_PATH_JSON} found. "
            f"PyYAML installed: {HAS_YAML}"
        )


def render_violin_note_from_preset(
    preset: dict,
    note: int = 60,
    duration_sec: float = 2.0,
) -> np.ndarray:
    """
    Render a violin note using parameters from the YAML preset.

    Reads all modulation parameters from the preset 'modulation' section:
    - vibrato: rate_hz, depth_amp, depth_timbre, delay_ms, ramp_ms
    - body_warble: rate_hz, depth, target_harmonic
    - attack_texture: bright_ms, bright_amount, harmonics
    """
    # Extract envelope params
    env = preset.get('env', {}).get('amp', {})
    attack_ms = env.get('attack_ms', 30.0)
    decay_ms = env.get('decay_ms', 250.0)
    sustain = env.get('sustain', 0.80)
    release_ms = env.get('release_ms', 550.0)

    # Convert to blocks
    attack_blocks = max(1, int(attack_ms / 1000.0 * BLOCKS_PER_SEC))
    decay_blocks = max(1, int(decay_ms / 1000.0 * BLOCKS_PER_SEC))
    release_blocks = max(1, int(release_ms / 1000.0 * BLOCKS_PER_SEC))
    sustain_blocks = int(duration_sec * BLOCKS_PER_SEC) - attack_blocks
    total_blocks = attack_blocks + sustain_blocks + release_blocks

    # Extract material params
    material = preset.get('material', {})
    beta_start = material.get('beta_start_q15', 0)
    beta_target = material.get('beta_target_q15', 0)
    beta_ramp_blocks = material.get('beta_ramp_blocks', 1)

    # Extract carrier harmonic amps
    carrier = preset.get('carrier', {})
    base_harmonics = carrier.get('harmonic_amps_q15', [32767])

    # Extract modulation params
    mod = preset.get('modulation', {})

    # Vibrato params
    vib = mod.get('vibrato', {})
    vibrato_rate_hz = vib.get('rate_hz', 5.5)
    vibrato_depth_amp = vib.get('depth_amp', 0.0)
    vibrato_depth_timbre = vib.get('depth_timbre', 0.0)
    vibrato_delay_ms = vib.get('delay_ms', 40.0)
    vibrato_ramp_ms = vib.get('ramp_ms', 27.0)
    vibrato_delay_blocks = int(vibrato_delay_ms / 1000.0 * BLOCKS_PER_SEC)
    vibrato_ramp_blocks = max(1, int(vibrato_ramp_ms / 1000.0 * BLOCKS_PER_SEC))

    # Body warble params
    body = mod.get('body_warble', {})
    body_rate_hz = body.get('rate_hz', 1.2)
    body_depth = body.get('depth', 0.0)
    body_target_harmonic = body.get('target_harmonic', 4) - 1  # Convert to 0-indexed

    # Attack texture params
    attack_tex = mod.get('attack_texture', {})
    attack_bright_ms = attack_tex.get('bright_ms', 60.0)
    attack_bright_amount = attack_tex.get('bright_amount', 0.0)
    attack_harmonics = attack_tex.get('harmonics', [5, 6])
    attack_bright_blocks = int(attack_bright_ms / 1000.0 * BLOCKS_PER_SEC)

    # ADSR envelope params
    attack_level_q15 = 32767
    sustain_level_q15 = int(sustain * 32767)

    # Create VM state
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=note)

    # No CMMM operators
    ops_enable = []
    families = {}

    samples = []

    print(f"  Rendering {total_blocks} blocks with modulation:")
    print(f"    Vibrato: {vibrato_rate_hz} Hz, amp={vibrato_depth_amp}, timbre={vibrato_depth_timbre}")
    print(f"    Body warble: {body_rate_hz} Hz, depth={body_depth}")
    print(f"    Attack texture: {attack_bright_amount} brightness over {attack_bright_ms}ms")

    for block_idx in range(total_blocks):
        # Time in seconds for LFO calculations
        t_sec = block_idx * BLOCK_SIZE / FS

        # Compute beta with ramp
        if block_idx < beta_ramp_blocks:
            t = block_idx / beta_ramp_blocks
            beta_q15 = int(beta_start + (beta_target - beta_start) * t)
        else:
            beta_q15 = beta_target

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

        # =====================================================================
        # Modulate harmonic amplitudes for organic character
        # =====================================================================

        # Copy base harmonics
        harmonic_amps = list(base_harmonics)

        # 1. Vibrato: amplitude + timbre wobble (delayed onset)
        if block_idx > vibrato_delay_blocks and (vibrato_depth_amp > 0 or vibrato_depth_timbre > 0):
            # Ramp in vibrato
            vib_ramp = min(1.0, (block_idx - vibrato_delay_blocks) / vibrato_ramp_blocks)
            vib_phase = 2.0 * math.pi * vibrato_rate_hz * t_sec

            # Amplitude wobble (affects gain)
            if vibrato_depth_amp > 0:
                amp_mod = 1.0 + vibrato_depth_amp * vib_ramp * math.sin(vib_phase)
                gain_q15 = int(gain_q15 * amp_mod)
                gain_q15 = max(0, min(32767, gain_q15))

            # Timbre wobble: shift energy between H2 (string) and H4 (body)
            if vibrato_depth_timbre > 0 and len(harmonic_amps) >= 4:
                timbre_shift = vibrato_depth_timbre * vib_ramp * math.sin(vib_phase + 0.3)
                harmonic_amps[1] = int(base_harmonics[1] * (1.0 + timbre_shift))  # H2
                harmonic_amps[3] = int(base_harmonics[3] * (1.0 - timbre_shift * 0.5))  # H4

        # 2. Body warble: slow harmonic drift (body resonance movement)
        if body_depth > 0 and body_target_harmonic < len(harmonic_amps):
            body_phase = 2.0 * math.pi * body_rate_hz * t_sec
            body_mod = 1.0 + body_depth * math.sin(body_phase)
            harmonic_amps[body_target_harmonic] = int(harmonic_amps[body_target_harmonic] * body_mod)

        # 3. Attack brightness: extra upper harmonics that settle
        if block_idx < attack_bright_blocks and attack_bright_amount > 0:
            bright_env = 1.0 - (block_idx / attack_bright_blocks)
            bright_boost = 1.0 + attack_bright_amount * bright_env
            # Boost specified harmonics on attack
            for h in attack_harmonics:
                h_idx = h - 1  # Convert to 0-indexed
                if h_idx < len(harmonic_amps):
                    harmonic_amps[h_idx] = int(base_harmonics[h_idx] * bright_boost * (1.5 if h == 5 else 1.3))

        # Clamp all harmonics to Q15 range
        harmonic_amps = [max(0, min(32767, h)) for h in harmonic_amps]

        # Create frame for this block
        frame = VoiceFrame(
            note=note,
            coeff_osc_mode=1,  # Mode B for harmonic bank
            H_request=len(harmonic_amps),
            gain_q15=gain_q15,
            beta_target_q15=beta_q15,
            partial_amps_q15=harmonic_amps,
            ops_enable=ops_enable,
            families=families,
            waveform_type=0,
        )

        # Render block
        state, output = vm_step_block(state, frame)
        samples.extend(output.tolist())

    return np.array(samples, dtype=np.int16)


def render_violin_phrase(preset: dict):
    """Render a short violin phrase to demonstrate the preset."""
    print("\n" + "=" * 60)
    print("VIOLIN IR PRESET - Audio Rendering")
    print("=" * 60)

    preset_info = preset.get('preset', {})
    print(f"\nPreset: {preset_info.get('preset_id', 'Unknown')}")
    print(f"Description: {preset_info.get('description', 'N/A')}")

    # Render a simple phrase: C4 - E4 - G4 - C5
    notes = [
        (60, 1.5),   # C4, 1.5 seconds
        (64, 1.5),   # E4, 1.5 seconds
        (67, 1.5),   # G4, 1.5 seconds
        (72, 2.0),   # C5, 2.0 seconds (longer final note)
    ]

    print(f"\nRendering phrase: C4 -> E4 -> G4 -> C5")

    all_samples = []
    for note, duration in notes:
        print(f"\n  Note {note} ({duration}s)...")
        samples = render_violin_note_from_preset(preset, note=note, duration_sec=duration)
        all_samples.extend(samples.tolist())
        # Small gap between notes (50ms)
        gap = np.zeros(int(0.05 * FS), dtype=np.int16)
        all_samples.extend(gap.tolist())

    final_samples = np.array(all_samples, dtype=np.int16)

    # Save to file (mono)
    audio_io = AudioWriter(output_dir=OUTPUT_DIR)
    L = final_samples.tolist()

    metadata = audio_io.write_audio(
        "violin_ir_phrase",
        L,  # Mono output
        subdirectory="presets",
        timestamp=False
    )

    print(f"\nSaved: {metadata.filepath}")
    print(f"Duration: {len(final_samples) / FS:.2f}s")
    print(f"Samples: {len(final_samples)}")

    return final_samples, metadata.filepath


def render_single_note(preset: dict):
    """Render a single sustained violin note."""
    print("\n" + "=" * 60)
    print("VIOLIN IR - Single Note Demo")
    print("=" * 60)

    note = 69  # A4 (440 Hz) to match reference violin.wav
    duration = 3.0

    print(f"\nRendering single note: A4 (MIDI {note}), {duration}s")

    samples = render_violin_note_from_preset(preset, note=note, duration_sec=duration)

    # Save to file (mono)
    audio_io = AudioWriter(output_dir=OUTPUT_DIR)
    L = samples.tolist()

    metadata = audio_io.write_audio(
        "violin_ir_single_note",
        L,  # Mono output
        subdirectory="presets",
        timestamp=False
    )

    print(f"\nSaved: {metadata.filepath}")
    print(f"Duration: {len(samples) / FS:.2f}s")

    return samples, metadata.filepath


if __name__ == "__main__":
    print("=" * 60)
    print("VIOLIN IR PRESET RENDERER")
    print("=" * 60)

    # Clear cached basis frames
    clear_basis_atlas()
    print("Cleared basis atlas cache")

    # Load preset from YAML
    print(f"\nLoading preset from: {PRESET_PATH}")
    preset = load_violin_preset()

    preset_info = preset.get('preset', {})
    print(f"Preset ID: {preset_info.get('preset_id')}")
    print(f"Version: {preset_info.get('preset_version')}")

    # Show modulation params
    mod = preset.get('modulation', {})
    print("\nModulation parameters from YAML:")
    print(f"  Vibrato: {mod.get('vibrato', {})}")
    print(f"  Body warble: {mod.get('body_warble', {})}")
    print(f"  Attack texture: {mod.get('attack_texture', {})}")

    # Render single note
    single_samples, single_path = render_single_note(preset)

    # Render phrase
    phrase_samples, phrase_path = render_violin_phrase(preset)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  1. {single_path}")
    print(f"  2. {phrase_path}")
