"""
Generic Preset Renderer
=======================

Renders audio from any InstrumentIR preset (YAML or JSON).
Supports:
- Custom L2 regime graphs with state transitions
- Envelope parameters
- Backbone operators
- Material (beta) ramps

Usage:
    python tools/render_preset.py [preset_path] [--note N] [--duration S]

Examples:
    python tools/render_preset.py  # Renders glitch_lead_v0 by default
    python tools/render_preset.py presets/instruments/brass_stab_v0.yaml
    python tools/render_preset.py presets/instruments/violin_ir_v1.yaml --note 69
"""

import sys
import argparse
import math
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from symcrash_python_ap.instrument_ir import load_instrument
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

# Default preset
DEFAULT_PRESET = Path(__file__).parent.parent / "presets/instruments/glitch_lead_v0.yaml"


def ms_to_blocks(ms: float) -> int:
    """Convert milliseconds to blocks."""
    return max(1, int(ms / 1000.0 * BLOCKS_PER_SEC))


def extract_envelope_params(ir) -> dict:
    """Extract envelope parameters from InstrumentIR."""
    # Try to find ADSR from L0 graph amp nodes
    for node in ir.layers.L0_graph.nodes:
        if node.type == "Amp" and node.params.adsr:
            adsr = node.params.adsr
            return {
                'attack_blocks': ms_to_blocks(adsr.a_ms),
                'decay_blocks': ms_to_blocks(adsr.d_ms),
                'sustain': adsr.s,
                'release_blocks': ms_to_blocks(adsr.r_ms),
            }

    # Default envelope
    return {
        'attack_blocks': ms_to_blocks(10),
        'decay_blocks': ms_to_blocks(100),
        'sustain': 0.7,
        'release_blocks': ms_to_blocks(200),
    }


def extract_waveform(ir) -> int:
    """Extract waveform type from InstrumentIR."""
    for node in ir.layers.L0_graph.nodes:
        if node.type == "Osc" and node.params.waveform:
            wf = node.params.waveform.lower()
            if wf == "sine":
                return 0
            elif wf == "saw":
                return 1
            elif wf == "pulse" or wf == "square":
                return 2
            elif wf == "tri":
                return 3
    return 1  # Default to saw


def get_regime_info(ir) -> list:
    """Get regime info from L2 graph."""
    if ir.layers.L2_regimes.regime_nodes:
        return [(n.id, n.kind, n.emit_template, n.dwell_ms) for n in ir.layers.L2_regimes.regime_nodes]
    return []


def get_emit_template(ir, template_name: str) -> dict:
    """Get emit template by name."""
    templates = ir.layers.L3_cmmm.emit_templates
    if template_name in templates:
        t = templates[template_name]
        return {
            'families': dict(t.families),
            'ops_enable': list(t.ops_enable),
        }
    return {'families': {}, 'ops_enable': []}


def render_preset_note(
    ir,
    note: int = 60,
    duration_sec: float = 2.0,
    risk_profile: list = None,
) -> np.ndarray:
    """
    Render a single note using the preset parameters.

    Args:
        ir: InstrumentIR object
        note: MIDI note number
        duration_sec: Duration in seconds
        risk_profile: Optional list of (block_idx, risk_value) tuples for regime transitions
    """
    # Extract parameters
    env = extract_envelope_params(ir)
    waveform = extract_waveform(ir)

    attack_blocks = env['attack_blocks']
    decay_blocks = env['decay_blocks']
    sustain = env['sustain']
    release_blocks = env['release_blocks']

    sustain_blocks = int(duration_sec * BLOCKS_PER_SEC) - attack_blocks
    total_blocks = attack_blocks + sustain_blocks + release_blocks

    # Regime info
    regimes = get_regime_info(ir)
    current_regime = "R2_STABLE"  # Start in stable after attack
    regime_dwell = 0

    # ADSR levels
    attack_level_q15 = 32767
    sustain_level_q15 = int(sustain * 32767)

    # Create VM state
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=note)

    # Default harmonic profile (saw-like)
    harmonic_amps = [32767, 16383, 10922, 8191, 6553, 5461, 4681, 4096]

    samples = []

    print(f"  Rendering {total_blocks} blocks...")
    print(f"  Envelope: A={attack_blocks} D={decay_blocks} S={sustain:.2f} R={release_blocks}")
    print(f"  Waveform type: {waveform}")

    if regimes:
        print(f"  Regime graph: {len(regimes)} states")
        for r_id, r_kind, r_template, r_dwell in regimes:
            print(f"    {r_id} ({r_kind}) -> {r_template}, dwell={r_dwell}ms")

    for block_idx in range(total_blocks):
        # Compute gain envelope (ADSR)
        if block_idx < attack_blocks:
            t = block_idx / attack_blocks
            gain_q15 = int(attack_level_q15 * t)
            phase = "ATTACK"
        elif block_idx < attack_blocks + decay_blocks:
            t = (block_idx - attack_blocks) / decay_blocks
            gain_q15 = int(attack_level_q15 - (attack_level_q15 - sustain_level_q15) * t)
            phase = "DECAY"
        elif block_idx < attack_blocks + sustain_blocks:
            gain_q15 = sustain_level_q15
            phase = "SUSTAIN"
        else:
            release_idx = block_idx - (attack_blocks + sustain_blocks)
            t = release_idx / release_blocks
            gain_q15 = int(sustain_level_q15 * (1.0 - t))
            phase = "RELEASE"

        gain_q15 = max(0, min(32767, gain_q15))

        # Handle risk profile for regime transitions (if provided)
        risk_value = 0.0
        if risk_profile:
            for (ridx, rval) in risk_profile:
                if block_idx >= ridx:
                    risk_value = rval

        # Determine current regime based on risk (simplified state machine)
        ops_enable = []
        families = {}

        if regimes:
            # Simple state machine: STABLE <-> GLITCH based on risk
            if risk_value > 0.7 and "GLITCH" not in current_regime:
                current_regime = "R3_GLITCH"
                regime_dwell = 0
                print(f"  Block {block_idx}: -> GLITCH (risk={risk_value:.2f})")
            elif current_regime == "R3_GLITCH":
                regime_dwell += 1
                # Find dwell time for glitch
                glitch_dwell_ms = 80
                for r_id, r_kind, r_template, r_dwell in regimes:
                    if "GLITCH" in r_id and r_dwell:
                        glitch_dwell_ms = r_dwell
                if regime_dwell > ms_to_blocks(glitch_dwell_ms):
                    current_regime = "R4_RECOVERY"
                    regime_dwell = 0
                    print(f"  Block {block_idx}: -> RECOVERY")
            elif current_regime == "R4_RECOVERY":
                regime_dwell += 1
                recovery_dwell_ms = 60
                for r_id, r_kind, r_template, r_dwell in regimes:
                    if "RECOVERY" in r_id and r_dwell:
                        recovery_dwell_ms = r_dwell
                if regime_dwell > ms_to_blocks(recovery_dwell_ms):
                    current_regime = "R2_STABLE"
                    regime_dwell = 0
                    print(f"  Block {block_idx}: -> STABLE")

            # Get template for current regime
            template_name = None
            for r_id, r_kind, r_template, r_dwell in regimes:
                if r_id == current_regime or r_kind.upper() in current_regime.upper():
                    template_name = r_template
                    break

            if template_name:
                template = get_emit_template(ir, template_name)
                ops_enable = template['ops_enable']
                families = template['families']

        # Create frame
        frame = VoiceFrame(
            note=note,
            coeff_osc_mode=1,  # Mode B
            H_request=len(harmonic_amps),
            gain_q15=gain_q15,
            beta_target_q15=0,
            partial_amps_q15=harmonic_amps,
            ops_enable=ops_enable,
            families=families,
            waveform_type=waveform,
        )

        # Render block
        state, output = vm_step_block(state, frame)
        samples.extend(output.tolist())

    return np.array(samples, dtype=np.int16)


def render_preset(preset_path: str, note: int = 60, duration: float = 2.0, with_glitch: bool = False):
    """Render a preset to audio."""
    print("\n" + "=" * 60)
    print("PRESET RENDERER")
    print("=" * 60)

    # Clear cached basis frames
    clear_basis_atlas()
    print("Cleared basis atlas cache")

    # Load preset
    print(f"\nLoading: {preset_path}")
    ir = load_instrument(preset_path)

    print(f"  Name: {ir.meta.name}")
    print(f"  ID: {ir.meta.instrument_id}")

    # Create risk profile if requested
    risk_profile = None
    if with_glitch:
        # Ramp up risk during sustain, then drop
        risk_profile = [
            (ms_to_blocks(500), 0.0),     # Start low
            (ms_to_blocks(800), 0.75),    # Trigger glitch
            (ms_to_blocks(1200), 0.2),    # Drop risk
        ]
        print(f"\n  Risk profile: {risk_profile}")

    # Render note
    print(f"\nRendering note {note} for {duration}s...")
    samples = render_preset_note(ir, note=note, duration_sec=duration, risk_profile=risk_profile)

    # Generate output filename
    preset_name = Path(preset_path).stem
    suffix = "_glitch" if with_glitch else ""
    output_name = f"{preset_name}_note{note}{suffix}"

    # Save to file
    audio_io = AudioWriter(output_dir=OUTPUT_DIR)
    metadata = audio_io.write_audio(
        output_name,
        samples.tolist(),
        subdirectory="presets",
        timestamp=False
    )

    print(f"\nSaved: {metadata.filepath}")
    print(f"Duration: {len(samples) / FS:.2f}s")
    print(f"Samples: {len(samples)}")

    return samples, metadata.filepath


def main():
    parser = argparse.ArgumentParser(description="Render audio from InstrumentIR preset")
    parser.add_argument("preset", nargs="?", default=str(DEFAULT_PRESET),
                       help="Path to preset YAML/JSON file")
    parser.add_argument("--note", "-n", type=int, default=60,
                       help="MIDI note number (default: 60 = C4)")
    parser.add_argument("--duration", "-d", type=float, default=2.0,
                       help="Duration in seconds (default: 2.0)")
    parser.add_argument("--glitch", "-g", action="store_true",
                       help="Enable glitch/risk transitions during render")

    args = parser.parse_args()

    # Check preset exists
    if not Path(args.preset).exists():
        print(f"Error: Preset not found: {args.preset}")
        sys.exit(1)

    samples, filepath = render_preset(
        args.preset,
        note=args.note,
        duration=args.duration,
        with_glitch=args.glitch
    )

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
