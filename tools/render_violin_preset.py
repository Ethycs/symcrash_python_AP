"""
Render Violin Preset from YAML Specification
=============================================

Builds and renders the violin_ir preset with:
- Expressive bowed-string character
- Noisy bow bite on attack
- Warm body texture via beta ramp
- TILT, INTRA_MIX, DIFFUSE_DOWN, LOWRANK_MORPH, IMPULSE, NOISE_BAND operators
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from symcrash_python_ap.instrument_ir import (
    InstrumentIR,
    L0Node, L0Edge, L0Graph,
    NodeParams, ADSR, Macro,
    ms_to_blocks, pct_to_q15,
)
from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.harmonic_bank import clear_basis_atlas
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import AudioWriter

# Constants
FS = 48000
BLOCK_SIZE = 64
BLOCKS_PER_SEC = FS // BLOCK_SIZE

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_generation_tests" / "audio_out")


def build_violin_ir() -> InstrumentIR:
    """
    Build the violin IR from the YAML specification.

    preset:
      preset_id: "VIOLIN_IR_V1"
      description: "Expressive bowed-string with noisy bow bite and warm body."

    env:
      amp: attack=30ms, decay=220ms, sustain=0.75, release=550ms

    material:
      beta_start=0.10 (mostly dry bow attack)
      beta_target=0.45 (add body texture)
      beta_ramp_blocks=24

    backbone ops: TILT, INTRA_MIX, DIFFUSE_DOWN, LOWRANK_MORPH, IMPULSE, NOISE_BAND
    """
    ir = InstrumentIR()

    # Metadata
    ir.meta.instrument_id = "violin-ir-v1"
    ir.meta.name = "Violin IR V1"
    ir.meta.author = "preset-author"

    # L0 Graph: osc_string + osc_body + noise_bow -> mix -> filter -> amp -> fx_send
    ir.layers.L0_graph.nodes = [
        # Primary string oscillator (saw for rich harmonics)
        L0Node(
            id="osc_string",
            type="Osc",
            params=NodeParams(
                waveform="saw",
                detune_cents=2.0,
                role="audible",
                archetype="harmonic"
            )
        ),
        # Body resonance oscillator (tri for hollow body)
        L0Node(
            id="osc_body",
            type="Osc",
            params=NodeParams(
                waveform="tri",
                detune_cents=-3.0,
                role="audible",
                archetype="hollow"
            )
        ),
        # Bow noise (pink noise for rosiny texture)
        L0Node(
            id="noise_bow",
            type="Noise",
            params=NodeParams(color="pink")
        ),
        # Mixer
        L0Node(id="mix", type="Mixer", params=NodeParams()),
        # Low-pass filter for body warmth
        L0Node(
            id="filt",
            type="Filter",
            params=NodeParams(mode="lp")
        ),
        # Amplitude with ADSR
        L0Node(
            id="amp",
            type="Amp",
            params=NodeParams(
                adsr=ADSR(
                    a_ms=30.0,    # Slow attack for bowed string
                    d_ms=220.0,   # Medium decay
                    s=0.75,       # High sustain
                    r_ms=550.0    # Long release
                )
            )
        ),
        # FX Send (reverb for room)
        L0Node(
            id="fx_send",
            type="FXSend",
            params=NodeParams(send_a=0.18, send_b=0.0)
        ),
    ]

    # Edges
    ir.layers.L0_graph.edges = [
        L0Edge(src="osc_string", dst="mix"),
        L0Edge(src="osc_body", dst="mix"),
        L0Edge(src="noise_bow", dst="mix"),
        L0Edge(src="mix", dst="filt"),
        L0Edge(src="filt", dst="amp"),
        L0Edge(src="amp", dst="fx_send"),
    ]

    # Macros
    ir.layers.L0_graph.macros = [
        Macro(id="M_BRIGHT", name="Brightness", default=0.42),
        Macro(id="M_TEXTURE", name="Bow Texture", default=0.35),
        Macro(id="M_MORPH", name="Body Morph", default=0.15),
        Macro(id="M_ATTACK", name="Attack", default=0.30),
        Macro(id="M_RELEASE", name="Release", default=0.45),
        Macro(id="M_DETUNE", name="Detune", default=0.02),
    ]

    return ir


def render_violin_note(
    note: int = 60,
    duration_sec: float = 2.0,
    attack_blocks: int = 23,  # ~30ms
    release_blocks: int = 413,  # ~550ms
) -> np.ndarray:
    """
    Render a violin note with proper ADSR envelope and beta ramp.

    Material params (warmer variant):
    - beta_start_q15: 1638 (0.05, very dry bow attack)
    - beta_target_q15: 9830 (0.30, subtle body texture - warmer)
    - beta_ramp_blocks: 32 (slower bloom)
    """
    # Calculate blocks
    sustain_blocks = int(duration_sec * BLOCKS_PER_SEC) - attack_blocks
    total_blocks = attack_blocks + sustain_blocks + release_blocks

    # Material beta ramp - TESTING: pure carrier to isolate spurs
    # The 750 Hz spur = 48000/64 block rate suggests IDWPT seam issues
    beta_start = 0         # Q15 for 0.0 (pure carrier - test)
    beta_target = 0        # Q15 for 0.0 (pure carrier throughout)
    beta_ramp_blocks = 1   # No ramp needed

    # ADSR envelope params (slightly slower attack for warmth)
    # attack=40ms, decay=250ms, sustain=0.80, release=600ms
    attack_level_q15 = 32767
    sustain_level_q15 = int(0.80 * 32767)  # Higher sustain
    decay_blocks = 188  # ~250ms (slower decay)

    # Create VM state and frame
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=note)
    # Use CCS path with Mode B for custom harmonics (now supported!)

    # Operators for organic sound - MINIMAL to avoid aliasing artifacts
    # The spurs at high frequencies suggest operator processing is creating artifacts
    # Let's try with NO operators first to verify clean synthesis
    ops_enable = []  # Empty - pure Mode B synthesis

    # Operator strengths (not used when ops_enable is empty)
    families = {}

    # Carrier harmonic amps (REVERSE-ENGINEERED from violin.wav via FWPT)
    # The reference violin has H2 strongest, with body resonance at H4
    harmonic_amps = [
        30890,  # H1: 94.3% - strong fundamental
        32767,  # H2: 100.0% - STRONGEST (characteristic violin sound)
        6926,   # H3: 21.1% - moderate
        23969,  # H4: 73.2% - STRONG body resonance
        3673,   # H5: 11.2% - weak
        6138,   # H6: 18.7% - moderate
        1164,   # H7: 3.6% - very weak
        1228,   # H8: 3.7% - very weak
    ]

    samples = []

    for block_idx in range(total_blocks):
        # Compute beta with ramp
        if block_idx < beta_ramp_blocks:
            # Ramp from start to target
            t = block_idx / beta_ramp_blocks
            beta_q15 = int(beta_start + (beta_target - beta_start) * t)
        else:
            beta_q15 = beta_target

        # Compute gain envelope (ADSR)
        if block_idx < attack_blocks:
            # Attack phase
            t = block_idx / attack_blocks
            gain_q15 = int(attack_level_q15 * t)
        elif block_idx < attack_blocks + decay_blocks:
            # Decay phase
            t = (block_idx - attack_blocks) / decay_blocks
            gain_q15 = int(attack_level_q15 - (attack_level_q15 - sustain_level_q15) * t)
        elif block_idx < attack_blocks + sustain_blocks:
            # Sustain phase
            gain_q15 = sustain_level_q15
        else:
            # Release phase
            release_idx = block_idx - (attack_blocks + sustain_blocks)
            t = release_idx / release_blocks
            gain_q15 = int(sustain_level_q15 * (1.0 - t))

        gain_q15 = max(0, min(32767, gain_q15))

        # Create frame for this block
        # Mode B now supports custom harmonics via partial_amps_q15
        frame = VoiceFrame(
            note=note,
            coeff_osc_mode=1,  # Mode B for harmonic bank with custom harmonics
            H_request=8,
            gain_q15=gain_q15,
            beta_target_q15=beta_q15,
            partial_amps_q15=harmonic_amps,
            ops_enable=ops_enable,
            families=families,
            waveform_type=0,  # Ignored when partial_amps_q15 is provided
        )

        # Render block
        state, output = vm_step_block(state, frame)
        samples.extend(output.tolist())

    return np.array(samples, dtype=np.int16)


def render_violin_phrase():
    """Render a short violin phrase to demonstrate the preset."""
    print("\n" + "=" * 60)
    print("VIOLIN IR PRESET - Audio Rendering")
    print("=" * 60)

    # Build the IR (for reference)
    ir = build_violin_ir()
    print(f"\nPreset: {ir.meta.name}")
    print(f"ID: {ir.meta.instrument_id}")
    print(f"L0 Nodes: {[n.id for n in ir.layers.L0_graph.nodes]}")
    print(f"Macros: {[m.name for m in ir.layers.L0_graph.macros]}")

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
        print(f"  Note {note} ({duration}s)...")
        samples = render_violin_note(note=note, duration_sec=duration)
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


def render_single_note():
    """Render a single sustained violin note."""
    print("\n" + "=" * 60)
    print("VIOLIN IR - Single Note Demo")
    print("=" * 60)

    note = 69  # A4 (440 Hz) to match reference violin.wav
    duration = 3.0

    print(f"\nRendering single note: A4 (MIDI {note}), {duration}s")

    samples = render_violin_note(note=note, duration_sec=duration)

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

    # Clear cached basis frames to use K=8 reconstruction
    clear_basis_atlas()
    print("Cleared basis atlas cache (using K=8 reconstruction for clean signal)")

    # Render single note
    single_samples, single_path = render_single_note()

    # Render phrase
    phrase_samples, phrase_path = render_violin_phrase()

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  1. {single_path}")
    print(f"  2. {phrase_path}")
