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
import copy
import math
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from symcrash_python_ap.instrument_ir import (
    load_instrument, extract_partials_q15, extract_phase_spread_q15,
    extract_filter_params, extract_oeis_params, extract_noise_params,
    extract_unison_params,
)
from symcrash_python_ap.envelope_targets import EnvelopeTarget
from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.harmonic_bank import clear_basis_atlas, LFOTarget
from symcrash_python_ap.vm.state import VMState, OSC_MODE_FM6
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import AudioWriter

# Constants (canonical conversion from primitives; re-exported here for
# the existing workbench/scripts import surface)
from symcrash_python_ap.primitives import (
    SAMPLE_RATE_HZ as FS,
    BLOCK_SIZE,
    BLOCK_RATE_HZ as BLOCKS_PER_SEC,
    ms_to_blocks,
)

OUTPUT_DIR = str(Path(__file__).parent.parent.parent / "audio_out")

# Default preset
DEFAULT_PRESET = Path(__file__).parent.parent.parent / "presets/instruments/glitch_lead_v0.yaml"


def extract_envelope_params(ir) -> dict:
    """Extract envelope parameters from InstrumentIR (shared ir_extract)."""
    from symcrash_python_ap.library.ir_extract import extract_amp

    amp = extract_amp(ir)
    if amp is not None:
        return {
            'attack_blocks': ms_to_blocks(amp.a_ms),
            'decay_blocks': ms_to_blocks(amp.d_ms),
            'sustain': amp.s,
            'release_blocks': ms_to_blocks(amp.r_ms),
        }

    # Default envelope
    return {
        'attack_blocks': ms_to_blocks(10),
        'decay_blocks': ms_to_blocks(100),
        'sustain': 0.7,
        'release_blocks': ms_to_blocks(200),
    }


# Canonical waveform-name -> id extraction (single source in instrument_ir;
# kept under the old name for the workbench/scripts import surface)
from symcrash_python_ap.instrument_ir import (
    extract_waveform_type as extract_waveform,
)


# Canonical binding-envelope evaluator (single owner in envelope_scheduler;
# old name kept for the workbench streaming import surface)
from symcrash_python_ap.envelope_scheduler import BindingEnvelope as _EnvelopeState

# Shared frame builder (single owner of per-block frame construction);
# helper re-exports keep the old local names for the streaming import
# surface.
from symcrash_python_ap.voice.frame_builder import (
    IRFrameBuilder,
    NoteClock,
    GOLDEN_Q32 as _GOLDEN_Q32,
    equal_power_pan,
    filtered_partials as _filtered_partials,
    get_emit_template,
    unison_positions,
)


def get_regime_info(ir) -> list:
    """Get regime info from L2 graph."""
    if ir.layers.L2_regimes.regime_nodes:
        return [(n.id, n.kind, n.emit_template, n.dwell_ms) for n in ir.layers.L2_regimes.regime_nodes]
    return []


def _render_voice(
    ir,
    note: int = 60,
    duration_sec: float = 2.0,
    risk_profile: list = None,
    velocity_q15: int = 32767,
    *,
    cents_offset: int = 0,
    phase_offset_q32: int = 0,
    noise_seed_xor: int = 0,
    verbose: bool = True,
) -> np.ndarray:
    """
    Render one (mono) voice of a note using the preset parameters.

    With the keyword-only offsets at their defaults this is the bit-exact
    legacy mono render path. Unison rendering (render_preset_note) calls
    this N times with per-copy detune/phase/noise offsets.

    Args:
        ir: InstrumentIR object
        note: MIDI note number
        duration_sec: Duration in seconds
        risk_profile: Optional list of (block_idx, risk_value) tuples for regime transitions
        velocity_q15: Note velocity (0-32767). Scales gain (as in the live
            engine) and any envelope binding with velocity_sens > 0 — e.g.
            velocity -> FM modulation index for mallet dynamics.
        cents_offset: Constant detune added to every frame's cents (unison).
        phase_offset_q32: Q32 oscillator start phase (Mode B; unison
            decorrelation via the golden-angle convention).
        noise_seed_xor: XORed into the noise exciter seed so unison copies
            get decorrelated attack noise.
        verbose: Print render header info.
    """
    # One shared frame builder (voice/frame_builder.py) owns all
    # per-block frame construction; this function keeps only the offline
    # timeline (total blocks), the L2 regime/risk state machine, and the
    # VM stepping loop.
    builder = IRFrameBuilder(
        ir, note, velocity_q15,
        cents_offset=cents_offset, noise_seed_xor=noise_seed_xor,
    )
    attack_blocks = builder.attack_blocks
    decay_blocks = builder.decay_blocks
    release_blocks = builder.release_blocks

    sustain_blocks = int(duration_sec * BLOCKS_PER_SEC) - attack_blocks
    total_blocks = attack_blocks + sustain_blocks + release_blocks
    sustain_until = attack_blocks + sustain_blocks

    # Regime info
    regimes = get_regime_info(ir)
    current_regime = "R2_STABLE"  # Start in stable after attack
    regime_dwell = 0

    # Create VM state
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=note)

    # Unison phase decorrelation: the Mode B oscillator applies this Q32
    # start phase on its first-time-init reset (vm/harmonic_bank.py).
    # 0 (default) leaves the legacy zero-phase reset bit-exact.
    if phase_offset_q32:
        state.mode_b_state.phase_offset_q32 = phase_offset_q32 & 0xFFFFFFFF

    samples = []

    if verbose:
        print(f"  Rendering {total_blocks} blocks...")
        print(f"  Envelope: A={attack_blocks} D={decay_blocks} "
              f"S={builder.sustain_q15 / 32767:.2f} R={release_blocks}")
        print(f"  Waveform type: {builder.waveform_type}")

        if regimes:
            print(f"  Regime graph: {len(regimes)} states")
            for r_id, r_kind, r_template, r_dwell in regimes:
                print(f"    {r_id} ({r_kind}) -> {r_template}, dwell={r_dwell}ms")

    for block_idx in range(total_blocks):
        clock = NoteClock(block_idx, sustain_until_block=sustain_until)

        # Handle risk profile for regime transitions (if provided)
        risk_value = 0.0
        if risk_profile:
            for (ridx, rval) in risk_profile:
                if block_idx >= ridx:
                    risk_value = rval

        # L2 regime machine drives emission when present; otherwise the
        # builder maps the ADSR phase to an emit template (simple form).
        ops_enable = None
        families = None
        if regimes:
            ops_enable = []
            families = {}

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

        frame = builder.build(clock, ops_enable=ops_enable, families=families)

        # Render block
        state, output = vm_step_block(state, frame)
        samples.extend(output.tolist())

    return np.array(samples, dtype=np.int16)


def render_preset_note(
    ir,
    note: int = 60,
    duration_sec: float = 2.0,
    risk_profile: list = None,
    velocity_q15: int = 32767,
) -> np.ndarray:
    """
    Render a single note using the preset parameters.

    No unison authored (the default): bit-exact legacy mono path —
    returns a 1-D int16 array.

    Unison authored (`unison: {voices, spread_cents, width}` on the Osc
    node, Packet E5): renders `voices` detuned copies of the note and
    returns a stereo int16 array of shape (2, n_samples) = (L, R).

    Unison law:
    - Detune: copy k at position t_k = (2k - (N-1)) / (N-1) in [-1, +1]
      gets a constant detune of round(spread_cents * t_k) — symmetric
      across +/-spread_cents, center voice (odd N) undetuned.
    - Phase: copy k starts at Q32 phase (k * 0x9E3779B9) mod 2^32
      (golden-angle sequence, same constant as the harmonic bank's
      partial phase spread) so copies don't attack phase-coherent.
    - Stereo: equal-power pan at position width * t_k
      (L = cos(theta), R = sin(theta), theta = (pan+1) * pi/4), then the
      mix is scaled by 1/sqrt(N) (decorrelated copies sum in power).
    """
    unison = extract_unison_params(ir)
    if not unison:
        return _render_voice(ir, note=note, duration_sec=duration_sec,
                             risk_profile=risk_profile, velocity_q15=velocity_q15)

    n_voices = unison['voices']
    spread = unison['spread_cents']
    width = unison['width']
    print(f"  Unison: {n_voices} voices, +/-{spread} cents, width={width:.2f}")

    mix_l = None
    mix_r = None
    for k, (t, phase_q32) in enumerate(unison_positions(n_voices)):
        voice = _render_voice(
            ir, note=note, duration_sec=duration_sec,
            risk_profile=risk_profile, velocity_q15=velocity_q15,
            cents_offset=round(spread * t),
            phase_offset_q32=phase_q32,
            noise_seed_xor=phase_q32,
            verbose=(k == 0),
        ).astype(np.float64)
        pan_l, pan_r = equal_power_pan(t, width)
        if mix_l is None:
            mix_l = voice * pan_l
            mix_r = voice * pan_r
        else:
            mix_l += voice * pan_l
            mix_r += voice * pan_r

    # Decorrelated copies sum in power -> 1/sqrt(N), plus a deterministic
    # peak guard (copies are still partly correlated early in the note).
    norm = 1.0 / math.sqrt(n_voices)
    peak = max(float(np.abs(mix_l).max()), float(np.abs(mix_r).max())) * norm
    if peak > 32000.0:
        norm *= 32000.0 / peak
    left = np.clip(np.rint(mix_l * norm), -32768, 32767).astype(np.int16)
    right = np.clip(np.rint(mix_r * norm), -32768, 32767).astype(np.int16)
    return np.stack([left, right])


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

    # Save to file (stereo when the preset authors unison, mono otherwise)
    audio_io = AudioWriter(output_dir=OUTPUT_DIR)
    if samples.ndim == 2:
        metadata = audio_io.write_audio(
            output_name,
            samples[0].tolist(),
            samples[1].tolist(),
            subdirectory="presets",
            timestamp=False
        )
        n_samples = samples.shape[1]
    else:
        metadata = audio_io.write_audio(
            output_name,
            samples.tolist(),
            subdirectory="presets",
            timestamp=False
        )
        n_samples = len(samples)

    print(f"\nSaved: {metadata.filepath}")
    print(f"Duration: {n_samples / FS:.2f}s")
    print(f"Samples: {n_samples} ({metadata.channels} ch)")

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
