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
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from symcrash_python_ap.instrument_ir import (
    load_instrument, extract_partials_q15, extract_phase_spread_q15,
    extract_filter_params,
)
from symcrash_python_ap.envelope_targets import EnvelopeTarget
from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.harmonic_bank import clear_basis_atlas, LFOTarget
from symcrash_python_ap.vm.state import VMState, OSC_MODE_FM6
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import AudioWriter

# Constants
FS = 48000
BLOCK_SIZE = 64
BLOCKS_PER_SEC = FS // BLOCK_SIZE

OUTPUT_DIR = str(Path(__file__).parent.parent.parent / "audio_out")

# Default preset
DEFAULT_PRESET = Path(__file__).parent.parent.parent / "presets/instruments/glitch_lead_v0.yaml"


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
    """Extract waveform type from InstrumentIR (canonical WaveformType values:
    0=SINE, 1=SQUARE, 2=PULSE, 3=SAW, 4=SAW_DOWN, 5=TRIANGLE)."""
    for node in ir.layers.L0_graph.nodes:
        if node.type == "Osc" and node.params.waveform:
            wf = node.params.waveform.lower()
            if wf == "sine":
                return 0
            elif wf in ("square",):
                return 1
            elif wf == "pulse":
                return 2
            elif wf == "saw":
                return 3
            elif wf in ("tri", "triangle"):
                return 5
            elif wf == "partials":
                # 0 lets the custom harmonic profile (partials) drive the
                # spectrum via the harmonic bank.
                return 0
    return 3  # Default to saw


class _EnvelopeState:
    """ADSR state for one matrix-envelope binding (mirrors voice_engine)."""

    def __init__(self, binding, note: int = 60):
        adsr = binding.adsr
        # DX7-style EG key scaling: higher notes get shorter envelopes
        ks = float(getattr(binding, 'key_scale_rate', 0.0))
        mult = 2.0 ** (-ks * (note - 60) / 24.0) if ks else 1.0
        self.attack_blocks = ms_to_blocks(adsr.a_ms * mult)
        self.decay_blocks = ms_to_blocks(adsr.d_ms * mult)
        self.sustain = adsr.s
        self.release_blocks = ms_to_blocks(adsr.r_ms * mult)
        self.target_ids = []
        for name in binding.targets:
            try:
                self.target_ids.append(int(EnvelopeTarget[name]))
            except KeyError:
                pass
        self.velocity_sens_q15 = int(getattr(binding, 'velocity_sens', 0.0) * 32767)

    def value_q15(self, block_idx, sustain_blocks):
        a, d, s, r = (self.attack_blocks, self.decay_blocks,
                      self.sustain, self.release_blocks)
        s_q15 = int(s * 32767)
        if block_idx < a:
            v = int(32767 * (block_idx / a)) if a > 0 else 32767
        elif block_idx < a + d:
            t = (block_idx - a) / d if d > 0 else 1
            v = int(32767 - (32767 - s_q15) * t)
        elif block_idx < sustain_blocks:
            v = s_q15
        else:
            t = (block_idx - sustain_blocks) / r if r > 0 else 1
            v = int(s_q15 * (1.0 - t))
        return max(0, min(32767, v))


def _filtered_partials(base_amps_q15, f0_hz, cutoff_hz, resonance):
    """
    Reshape a harmonic profile through a 24 dB/oct LPF magnitude curve with
    a resonance bump at the cutoff. Authoring-side float math; the engine
    consumes the resulting Q15 ints per block (phase-continuously).
    """
    out = []
    for i, amp in enumerate(base_amps_q15):
        f = (i + 1) * f0_hz
        ratio = f / max(cutoff_hz, 20.0)
        gain = 1.0 / (1.0 + ratio ** 4)
        if resonance > 0:
            # Peak around the cutoff, ~1.5x at resonance=1
            x = math.log2(max(ratio, 1e-6))
            gain += resonance * 1.5 * math.exp(-(x * x) * 8.0)
        out.append(max(0, min(32767, round(amp * gain))))
    return out


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
    velocity_q15: int = 32767,
) -> np.ndarray:
    """
    Render a single note using the preset parameters.

    Args:
        ir: InstrumentIR object
        note: MIDI note number
        duration_sec: Duration in seconds
        risk_profile: Optional list of (block_idx, risk_value) tuples for regime transitions
        velocity_q15: Note velocity (0-32767). Scales gain (as in the live
            engine) and any envelope binding with velocity_sens > 0 — e.g.
            velocity -> FM modulation index for mallet dynamics.
    """
    # Extract parameters
    env = extract_envelope_params(ir)
    waveform = extract_waveform(ir)

    attack_blocks = env['attack_blocks']
    decay_blocks = env['decay_blocks']
    sustain = env['sustain']
    release_blocks = env['release_blocks']

    # Amp dynamics (WSE-STD-PRESET-AUTHORING 3.3): EG key scaling and the
    # velocity->gain curve, mirroring voice_engine semantics
    amp_velocity_sens_q15 = 32767
    for node in ir.layers.L0_graph.nodes:
        if node.type == "Amp" and node.params.adsr:
            ks = float(getattr(node.params, 'key_scale_rate', 0.0))
            if ks:
                mult = 2.0 ** (-ks * (note - 60) / 24.0)
                attack_blocks = max(1, int(attack_blocks * mult))
                decay_blocks = max(1, int(decay_blocks * mult))
                release_blocks = max(1, int(release_blocks * mult))
            amp_velocity_sens_q15 = int(
                getattr(node.params, 'velocity_sens', 1.0) * 32767)
            break
    vel_scale_q15 = (32767 - amp_velocity_sens_q15
                     + ((amp_velocity_sens_q15 * velocity_q15) >> 15))

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

    # Harmonic profile: use the preset's partials when defined; otherwise pass
    # no custom harmonics so waveform_type drives the spectrum (custom
    # harmonics always win over waveform_type in the executor).
    harmonic_amps = extract_partials_q15(ir) or []
    phase_spread_q15 = extract_phase_spread_q15(ir)

    # Filter node (Hz-domain): applied as per-block partials reshaping,
    # swept by any envelope binding targeting FILTER_CUTOFF.
    filter_params = extract_filter_params(ir) if harmonic_amps else None

    # Matrix-envelope bindings (FILTER_CUTOFF sweeps etc.)
    envelope_states = [
        _EnvelopeState(b, note) for b in (getattr(ir, 'envelopes', None) or [])
    ]
    filter_envs = [e for e in envelope_states
                   if int(EnvelopeTarget.FILTER_CUTOFF) in e.target_ids]

    # LFO modulators (vibrato / PWM motion / tremolo) — same routing
    # semantics as voice_engine.step_voice_simple
    lfos = list(getattr(ir, 'modulators', None) or [])

    # PWM base width from the Osc node (pulse waveform)
    pwm_width_base = 16384
    pwm_enabled = False
    for node in ir.layers.L0_graph.nodes:
        if node.type == "Osc":
            if getattr(node.params, 'pwm_width_q15', None) not in (None, 16384):
                pwm_width_base = node.params.pwm_width_q15
            if getattr(node.params, 'pw', 0.5) != 0.5:
                pwm_width_base = int(node.params.pw * 32767)
            pwm_enabled = bool(getattr(node.params, 'pwm_enable', False))
            break

    # FM6 synthesis (DX7-style) — frames carry the operator config and the
    # matrix-envelope targets (modulator decay is what makes FM breathe)
    use_fm6 = False
    fm6_params = None
    for node in ir.layers.L0_graph.nodes:
        if node.type == "FM6":
            use_fm6 = True
            fm6_params = {
                'algorithm': getattr(node.params, 'algorithm', 1),
                'feedback': getattr(node.params, 'feedback', 0),
                'op_ratios': getattr(node.params, 'op_ratios', [65536] * 6),
                'op_levels': getattr(node.params, 'op_levels', [32767, 0, 0, 0, 0, 0]),
                'post_filter_hz': getattr(node.params, 'post_filter_hz', None),
            }
            break
    # A PWM-targeted LFO implies PWM (only meaningful without custom partials)
    if not harmonic_amps and any(
        r.target == LFOTarget.PWM_WIDTH for lfo in lfos for r in lfo.routings
    ):
        pwm_enabled = True

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
        else:
            # No explicit L2 regime graph (simple-form preset): drive the emit
            # templates by envelope phase so the behaviour-state operators still
            # reach the audio.  Without this, simple-form presets render with no
            # operators applied and emphasis edits are inaudible.
            phase_to_template = {
                'ATTACK': 'T_ATTACK', 'DECAY': 'T_STABLE',
                'SUSTAIN': 'T_STABLE', 'RELEASE': 'T_RECOVERY',
            }
            templates = ir.layers.L3_cmmm.emit_templates
            for cand in (phase_to_template.get(phase, 'T_STABLE'), 'T_STABLE', 'T_IDLE'):
                if cand in templates:
                    tmpl = get_emit_template(ir, cand)
                    ops_enable = tmpl['ops_enable']
                    families = tmpl['families']
                    break

        # --- Per-block modulation: LFOs + filter envelope ---
        total_cents = 0
        gain_scale_q15 = 32767
        pwm_width = pwm_width_base
        for lfo in lfos:
            lfo.advance_block()
            lfo_val = lfo.get_lfo_value()
            for routing in lfo.routings:
                scaled = (lfo_val * routing.amount_q15 + 16384) >> 15
                if routing.target == LFOTarget.PITCH:
                    total_cents += scaled
                elif routing.target == LFOTarget.GAIN:
                    gain_scale_q15 = max(0, min(32767, 32767 + scaled))
                elif routing.target == LFOTarget.PWM_WIDTH:
                    pwm_width = max(1638, min(31130, pwm_width + scaled))

        block_amps = harmonic_amps
        if filter_params and harmonic_amps:
            env_v = 32767
            if filter_envs:
                env_v = max(e.value_q15(block_idx, attack_blocks + sustain_blocks)
                            for e in filter_envs)
            sweep_oct = 4.0 * filter_params['env_amount'] * (env_v / 32767.0)
            eff_cutoff = filter_params['cutoff_hz'] * (2.0 ** sweep_oct)
            f0 = 440.0 * 2 ** ((note - 69) / 12)
            block_amps = _filtered_partials(
                harmonic_amps, f0, eff_cutoff, filter_params['resonance'])

        # Matrix-envelope targets (FM operator decay etc.)
        env_targets_q15 = None
        if envelope_states:
            env_targets_q15 = {}
            for env_state in envelope_states:
                value = env_state.value_q15(block_idx, attack_blocks + sustain_blocks)
                if env_state.velocity_sens_q15:
                    # velocity_sens: value *= (1 - sens + sens*velocity)
                    scale = (32767 - env_state.velocity_sens_q15
                             + ((env_state.velocity_sens_q15 * velocity_q15) >> 15))
                    value = (value * scale) >> 15
                for target_id in env_state.target_ids:
                    env_targets_q15[target_id] = value

        # Create frame
        if use_fm6 and fm6_params:
            frame = VoiceFrame(
                note=note,
                cents=int(total_cents),
                coeff_osc_mode=OSC_MODE_FM6,
                gain_q15=((((gain_q15 * gain_scale_q15 + 16384) >> 15) * vel_scale_q15 + 16384) >> 15),
                beta_target_q15=0,
                fm_algorithm_id=fm6_params['algorithm'],
                fm_op_ratios=fm6_params['op_ratios'],
                fm_op_levels=fm6_params['op_levels'],
                fm_feedback_depth=fm6_params['feedback'],
                fm_post_filter_enabled=bool(fm6_params['post_filter_hz']) or None,
                fm_post_filter_cutoff_hz=fm6_params['post_filter_hz'],
                env_targets_q15=env_targets_q15,
                ops_enable=ops_enable,
                families=families,
            )
        else:
            frame = VoiceFrame(
                note=note,
                cents=int(total_cents),
                coeff_osc_mode=1,  # Mode B
                H_request=len(harmonic_amps) or 16,
                gain_q15=((((gain_q15 * gain_scale_q15 + 16384) >> 15) * vel_scale_q15 + 16384) >> 15),
                beta_target_q15=0,
                partial_amps_q15=block_amps,
                partial_phase_spread_q15=phase_spread_q15,
                ops_enable=ops_enable,
                families=families,
                waveform_type=waveform,
                pwm_width_q15=pwm_width,
                pwm_enabled=pwm_enabled,
                env_targets_q15=env_targets_q15,
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
