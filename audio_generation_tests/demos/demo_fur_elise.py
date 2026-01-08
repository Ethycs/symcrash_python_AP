"""
Für Elise Demo - Full Pipeline with Presets
============================================

Generates the opening theme of Beethoven's Für Elise using the full
CarrierLiftPipeline with various presets loaded from JSON.

Demonstrates:
1. Melodic sequencing through the full synthesis pipeline
2. Loading and applying preset configurations from JSON files
3. Different timbres for the same melody
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))  # for test_config

from test_config import FS, BLOCK_SIZE, BLOCKS_PER_SEC

from symcrash_python_ap.safety.unified_renderer import CarrierLiftPipeline
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import AudioWriter
from symcrash_python_ap.library.preset_data import list_presets, load_preset
from symcrash_python_ap.wavelet.harmonic_injector import WaveType, get_partial_amplitude

# Constants
Q15_ONE = 32767
Q15_HALF = 16384

# Waveform name to WaveType mapping
WAVEFORM_MAP = {
    "sine": WaveType.SINE,
    "saw": WaveType.SAW,
    "sawtooth": WaveType.SAW,
    "square": WaveType.PULSE,
    "pulse": WaveType.PULSE,
    "triangle": WaveType.TRIANGLE,
    "tri": WaveType.TRIANGLE,
}


def generate_partial_amps(wave_type: WaveType, num_partials: int = 16) -> List[int]:
    """Generate partial amplitude list for a given waveform type."""
    return [get_partial_amplitude(wave_type, k + 1) for k in range(num_partials)]


def blend_partial_amps(partial_amps_list: List[List[int]]) -> List[int]:
    """Blend multiple oscillator partial amplitudes together."""
    if not partial_amps_list:
        return [0] * 16
    if len(partial_amps_list) == 1:
        return partial_amps_list[0]

    # Average the amplitudes and renormalize
    num_partials = len(partial_amps_list[0])
    blended = []
    for k in range(num_partials):
        total = sum(pa[k] for pa in partial_amps_list)
        blended.append(total // len(partial_amps_list))
    return blended

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
audio_io = AudioWriter(output_dir=OUTPUT_DIR)


def midi_to_freq(note: int) -> float:
    """Convert MIDI note to frequency in Hz."""
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


# Für Elise opening theme (public domain - composed 1810)
# Format: (midi_note, duration_in_eighth_notes)
# 0 = rest

FUR_ELISE_MELODY: List[Tuple[int, float]] = [
    # Opening motif - E5 D#5 E5 D#5 E5 B4 D5 C5 A4
    (76, 1),   # E5
    (75, 1),   # D#5
    (76, 1),   # E5
    (75, 1),   # D#5
    (76, 1),   # E5
    (71, 1),   # B4
    (74, 1),   # D5
    (72, 1),   # C5
    (69, 2),   # A4 (quarter note)

    # C4 E4 A4 B4
    (0, 1),    # rest
    (60, 1),   # C4
    (64, 1),   # E4
    (69, 1),   # A4
    (71, 2),   # B4 (quarter note)

    # E4 G#4 B4 C5
    (0, 1),    # rest
    (64, 1),   # E4
    (68, 1),   # G#4
    (71, 1),   # B4
    (72, 2),   # C5 (quarter note)

    # E4 E5 D#5 E5 D#5 E5 B4 D5 C5 A4
    (0, 1),    # rest
    (64, 1),   # E4
    (76, 1),   # E5
    (75, 1),   # D#5
    (76, 1),   # E5
    (75, 1),   # D#5
    (76, 1),   # E5
    (71, 1),   # B4
    (74, 1),   # D5
    (72, 1),   # C5
    (69, 2),   # A4 (quarter note)

    # C4 E4 A4 B4
    (0, 1),    # rest
    (60, 1),   # C4
    (64, 1),   # E4
    (69, 1),   # A4
    (71, 2),   # B4 (quarter note)

    # E4 C5 B4 A4
    (0, 1),    # rest
    (64, 1),   # E4
    (72, 1),   # C5
    (71, 1),   # B4
    (69, 3),   # A4 (dotted quarter)

    # Repeat opening motif
    (76, 1),   # E5
    (75, 1),   # D#5
    (76, 1),   # E5
    (75, 1),   # D#5
    (76, 1),   # E5
    (71, 1),   # B4
    (74, 1),   # D5
    (72, 1),   # C5
    (69, 2),   # A4 (quarter note)

    # Final resolution
    (0, 1),    # rest
    (60, 1),   # C4
    (64, 1),   # E4
    (69, 1),   # A4
    (71, 2),   # B4
    (0, 1),    # rest
    (64, 1),   # E4
    (72, 1),   # C5
    (71, 1),   # B4
    (69, 4),   # A4 (half note - final)
]


def get_preset_config(preset_name: str) -> Dict[str, Any]:
    """
    Load preset and extract configuration.

    Returns dict with:
        - name: Display name
        - adsr: ADSR parameters
        - ops_enable: List of enabled operators
        - families: Operator strength values
        - beta: Dry/wet mix (0=carrier, Q15_ONE=wet)
        - wave_type: WaveType enum for harmonic generation
        - partial_amps: List of Q15 partial amplitudes
    """
    ir = load_preset(preset_name)

    # Default ADSR
    adsr = {"a_ms": 8, "d_ms": 80, "s": 0.7, "r_ms": 180}

    # Collect all oscillator waveforms
    waveform_names = []

    # Extract ADSR and waveforms from L0 graph nodes
    for node in ir.layers.L0_graph.nodes:
        if node.type == "Amp" and node.params.adsr:
            adsr = {
                "a_ms": node.params.adsr.a_ms,
                "d_ms": node.params.adsr.d_ms,
                "s": node.params.adsr.s,
                "r_ms": node.params.adsr.r_ms,
            }
        if node.type == "Osc" and hasattr(node.params, 'waveform') and node.params.waveform:
            waveform_names.append(node.params.waveform)

    # Default to saw if no oscillators found
    if not waveform_names:
        waveform_names = ["saw"]

    # Convert waveform names to WaveTypes and generate partials for each
    wave_types = [WAVEFORM_MAP.get(wn.lower(), WaveType.SAW) for wn in waveform_names]
    partial_amps_list = [generate_partial_amps(wt) for wt in wave_types]

    # Blend multiple oscillators together
    partial_amps = blend_partial_amps(partial_amps_list)

    # For display, show the primary waveform(s)
    waveform_name = "+".join(waveform_names)

    # Extract ops from L3 CMMM emit_template
    stable = ir.layers.L3_cmmm.emit_templates.get("Stable")
    ops_enable = stable.ops_enable if stable else []
    families = stable.families if stable else {}

    # Read beta from preset compile section, or use defaults based on type
    preset_beta = None
    if hasattr(ir, 'compile') and ir.compile and hasattr(ir.compile, 'completion_policy'):
        policy = ir.compile.completion_policy
        if hasattr(policy, 'default_tree') and policy.default_tree:
            preset_beta = getattr(policy.default_tree, 'beta', None)

    if preset_beta is not None and preset_beta > 0:
        # Use preset's beta value (convert from 0-1 to Q15)
        beta = int(preset_beta * Q15_ONE)
    elif "pad" in preset_name or "string" in preset_name:
        beta = Q15_HALF  # 50% blend for pads/strings
    elif "bell" in preset_name:
        beta = Q15_ONE // 3  # 33% wet for bells
    elif ops_enable:
        # If operators are defined, enable wet path at 50%
        beta = Q15_HALF
    else:
        beta = 0  # Pure carrier if no ops

    return {
        "name": ir.meta.name,
        "adsr": adsr,
        "ops_enable": ops_enable,
        "families": families,
        "beta": beta,
        "wave_types": wave_types,
        "waveform": waveform_name,
        "partial_amps": partial_amps,
    }


def render_melody_with_preset(preset_name: str, bpm: float = 70) -> Tuple[List[int], List[int], Dict]:
    """
    Render Für Elise melody with a specific preset using full pipeline.

    Uses legato-style rendering to avoid clicks between notes:
    - voice_on only at start and after rests
    - Note changes via VoiceFrame pitch (carrier maintains phase continuity)
    - voice_off only at rests or end

    Args:
        preset_name: Name of the preset to use
        bpm: Tempo in beats per minute

    Returns:
        Tuple of (samples_L, samples_R, preset_info)
    """
    config = get_preset_config(preset_name)
    adsr = config["adsr"]
    beta = config["beta"]
    partial_amps = config["partial_amps"]
    ops_enable = config["ops_enable"]
    families = config["families"]

    # Calculate timing
    eighth_note_sec = 60.0 / bpm / 2
    eighth_note_blocks = int(eighth_note_sec * BLOCKS_PER_SEC)

    # Convert ADSR to block counts
    attack_blocks = max(1, int((adsr["a_ms"] / 1000) * BLOCKS_PER_SEC))
    decay_blocks = max(1, int((adsr["d_ms"] / 1000) * BLOCKS_PER_SEC))
    sustain_level = adsr["s"]

    # Release crossfade blocks for smooth note-to-note transitions
    release_blocks = max(1, int((adsr["r_ms"] / 1000) * BLOCKS_PER_SEC))
    crossfade_blocks = min(release_blocks, 4)  # ~5ms crossfade between notes

    # Create pipeline
    pipeline = CarrierLiftPipeline()

    samples_L = []
    samples_R = []

    voice_active = False
    prev_note = None  # Track last note for release tail

    for i, (midi_note, duration) in enumerate(FUR_ELISE_MELODY):
        note_blocks = int(duration * eighth_note_blocks)

        # Look ahead to see if next event is a rest
        next_is_rest = (i + 1 >= len(FUR_ELISE_MELODY) or
                        FUR_ELISE_MELODY[i + 1][0] == 0)

        if midi_note == 0:
            # Rest - render release tail then silence
            if voice_active:
                # Render release tail to fade out smoothly (avoid click)
                last_env = sustain_level * 0.85  # Approximate last envelope level
                for rel_idx in range(release_blocks):
                    release_progress = (rel_idx + 1) / release_blocks
                    env = last_env * (1.0 - release_progress)

                    frame = VoiceFrame(
                        voice_id=0,
                        note=prev_note if prev_note else 60,
                        f0_hz=midi_to_freq(prev_note if prev_note else 60),
                        gain_q15=int(Q15_ONE * 0.85 * env),
                        beta_target_q15=beta,
                        partial_amps_q15=partial_amps,
                        ops_enable=ops_enable,
                        families=families,
                        coeff_osc_mode=1,  # Mode B: Harmonic Bank/Rotator
                        H_request=16,
                    )
                    L, R = pipeline.render_block(frame)
                    samples_L.extend(L)
                    samples_R.extend(R)

                pipeline.voice_off(0)
                voice_active = False

            # Render remaining silence (subtract release tail duration)
            silence_blocks = max(0, note_blocks - release_blocks)
            for _ in range(silence_blocks):
                samples_L.extend([0] * BLOCK_SIZE)
                samples_R.extend([0] * BLOCK_SIZE)
        else:
            # Start voice only if not already active
            if not voice_active:
                pipeline.voice_on(0, note=midi_note)
                voice_active = True
                is_new_phrase = True
            else:
                is_new_phrase = False

            # Render with ADSR envelope
            for block_idx in range(note_blocks):
                # Calculate envelope level
                if is_new_phrase and block_idx < attack_blocks:
                    # Ramp from ~50% to 100% during attack phase for immediate punch
                    env = 0.5 + 0.5 * (block_idx + 1) / attack_blocks
                elif is_new_phrase and block_idx < attack_blocks + decay_blocks:
                    decay_progress = (block_idx - attack_blocks) / decay_blocks
                    env = 1.0 - (1.0 - sustain_level) * decay_progress
                else:
                    # Sustain with slight decay over note duration
                    if is_new_phrase:
                        sustain_start_block = attack_blocks + decay_blocks
                    else:
                        sustain_start_block = 0
                    sustain_progress = (block_idx - sustain_start_block) / max(1, note_blocks - sustain_start_block)
                    env = sustain_level * (1.0 - sustain_progress * 0.15)

                # Apply release crossfade at end of note if next note is also a note (not rest)
                # This smooths the transition to the next pitch
                if not next_is_rest and block_idx >= note_blocks - crossfade_blocks:
                    crossfade_progress = (block_idx - (note_blocks - crossfade_blocks)) / crossfade_blocks
                    # Slight dip before next note attack
                    env *= (1.0 - 0.2 * crossfade_progress)

                # Create voice frame with waveform-specific partial amplitudes
                # Use 85% gain for more headroom (stereo pan at center is 0.707)
                # Mode B (coeff_osc_mode=1) uses WSE-STD-210 harmonic bank rotator
                frame = VoiceFrame(
                    voice_id=0,
                    note=midi_note,
                    f0_hz=midi_to_freq(midi_note),
                    gain_q15=int(Q15_ONE * 0.85 * env),
                    beta_target_q15=beta,
                    partial_amps_q15=partial_amps,
                    ops_enable=ops_enable,
                    families=families,
                    coeff_osc_mode=1,  # Mode B: Harmonic Bank/Rotator (WSE-STD-210)
                    H_request=16,      # Up to 16 harmonics
                )

                # Render block
                L, R = pipeline.render_block(frame)
                samples_L.extend(L)
                samples_R.extend(R)

            # Track last note for release tail
            prev_note = midi_note

    # End any remaining voice with release tail
    if voice_active:
        # Render release tail for final note
        last_env = sustain_level * 0.85
        for rel_idx in range(release_blocks):
            release_progress = (rel_idx + 1) / release_blocks
            env = last_env * (1.0 - release_progress)

            frame = VoiceFrame(
                voice_id=0,
                note=prev_note if prev_note else 60,
                f0_hz=midi_to_freq(prev_note if prev_note else 60),
                gain_q15=int(Q15_ONE * 0.85 * env),
                beta_target_q15=beta,
                partial_amps_q15=partial_amps,
                ops_enable=ops_enable,
                families=families,
                coeff_osc_mode=1,  # Mode B: Harmonic Bank/Rotator
                H_request=16,
            )
            L, R = pipeline.render_block(frame)
            samples_L.extend(L)
            samples_R.extend(R)

        pipeline.voice_off(0)

    return samples_L, samples_R, config


def save_audio(name: str, L: List[int], R: List[int]) -> str:
    """Save stereo audio to WAV file."""
    metadata = audio_io.write_audio(
        name, L, R,
        subdirectory="fur_elise",
        timestamp=False
    )
    return metadata.filepath


def main():
    print("=" * 60)
    print("FÜR ELISE - Full Pipeline Preset Showcase")
    print("=" * 60)
    print(f"\nSample rate: {FS} Hz")
    print(f"Block size: {BLOCK_SIZE}")
    print(f"\nAvailable presets: {list_presets()}")
    print()

    # Use all available presets
    selected_presets = list_presets()

    results = []

    for preset_name in selected_presets:
        print(f"\n{'=' * 50}")
        print(f"Rendering with preset: {preset_name}")
        print("=" * 50)

        L, R, config = render_melody_with_preset(preset_name, bpm=120)

        # Analyze
        arr = np.array(L, dtype=np.float64)
        duration = len(L) / FS
        peak = int(np.max(np.abs(arr)))
        rms = np.sqrt(np.mean(arr ** 2))

        print(f"  Preset: {config['name']}")
        print(f"  Waveform: {config['waveform']}")
        print(f"  ADSR: A={config['adsr']['a_ms']}ms D={config['adsr']['d_ms']}ms "
              f"S={config['adsr']['s']} R={config['adsr']['r_ms']}ms")
        ops_str = ", ".join(f"{op}@{config['families'].get(op, 1.0):.2f}" for op in config['ops_enable']) if config['ops_enable'] else "none"
        print(f"  Ops: {ops_str}")
        wet_pct = int(config['beta'] * 100 / Q15_ONE)
        print(f"  Beta: {config['beta']} ({wet_pct}% wet)")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Peak: {peak}, RMS: {rms:.0f}")

        # Save
        filepath = save_audio(f"fur_elise_{preset_name}", L, R)
        print(f"  Saved: {filepath}")

        results.append({
            "preset": preset_name,
            "duration": duration,
            "peak": peak,
            "rms": rms,
            "filepath": filepath,
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nGenerated {len(results)} versions of Für Elise:\n")

    for r in results:
        print(f"  {r['preset']:15s} - Peak: {r['peak']:5d}, RMS: {r['rms']:5.0f}")

    print(f"\nOutput directory: audio_out/fur_elise/")
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
