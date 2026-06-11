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

    # Unison phase decorrelation: the Mode B oscillator applies this Q32
    # start phase on its first-time-init reset (vm/harmonic_bank.py).
    # 0 (default) leaves the legacy zero-phase reset bit-exact.
    if phase_offset_q32:
        state.mode_b_state.phase_offset_q32 = phase_offset_q32 & 0xFFFFFFFF

    # Harmonic profile: use the preset's partials when defined; otherwise pass
    # no custom harmonics so waveform_type drives the spectrum (custom
    # harmonics always win over waveform_type in the executor).
    harmonic_amps = extract_partials_q15(ir) or []
    phase_spread_q15 = extract_phase_spread_q15(ir)

    # OEIS partial selection (outranks custom partials and waveform_type
    # in the executor)
    oeis_params = extract_oeis_params(ir)

    # Noise exciter (Packet E4): deterministic broadband attack transient.
    # No noise authored -> kwargs stay empty and the frame keeps its
    # level-0 defaults (bit-exact no-op in the VM).
    noise_params = extract_noise_params(ir)
    noise_frame_kwargs = {}
    if noise_params:
        noise_frame_kwargs = dict(
            noise_color=noise_params['color'],
            noise_level_q15=noise_params['level_q15'],
            noise_attack_blocks=ms_to_blocks(noise_params['attack_ms']),
            noise_decay_blocks=ms_to_blocks(noise_params['decay_ms']),
            noise_seed_u32=(noise_params['seed_u32'] ^ noise_seed_xor) & 0xFFFFFFFF,
        )

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
    # semantics as voice_engine.step_voice_simple. Deep-copied so each
    # unison voice copy advances its own LFO state from the authored
    # initial phase (and the IR itself is never mutated).
    lfos = [copy.deepcopy(m) for m in (getattr(ir, 'modulators', None) or [])]

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

    if verbose:
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
                cents=int(total_cents) + cents_offset,
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
                **noise_frame_kwargs,
            )
        else:
            frame = VoiceFrame(
                note=note,
                cents=int(total_cents) + cents_offset,
                coeff_osc_mode=1,  # Mode B
                H_request=len(harmonic_amps) or 16,
                gain_q15=((((gain_q15 * gain_scale_q15 + 16384) >> 15) * vel_scale_q15 + 16384) >> 15),
                beta_target_q15=0,
                partial_amps_q15=block_amps,
                partial_phase_spread_q15=phase_spread_q15,
                ops_enable=ops_enable,
                families=families,
                waveform_type=waveform,
                oeis_sequence_id=oeis_params['sequence_id'] if oeis_params else -1,
                oeis_rolloff_db=oeis_params['rolloff_db'] if oeis_params else -3.0,
                pwm_width_q15=pwm_width,
                pwm_enabled=pwm_enabled,
                env_targets_q15=env_targets_q15,
                **noise_frame_kwargs,
            )

        # Render block
        state, output = vm_step_block(state, frame)
        samples.extend(output.tolist())

    return np.array(samples, dtype=np.int16)


# Golden-ratio conjugate scaled to 2^32 — the repo's golden-angle phase
# convention (see vm/harmonic_bank.py partial phase spread).
_GOLDEN_Q32 = 0x9E3779B9


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
    for k in range(n_voices):
        t = (2 * k - (n_voices - 1)) / (n_voices - 1)  # -1..+1, left -> right
        phase_q32 = (k * _GOLDEN_Q32) & 0xFFFFFFFF
        voice = _render_voice(
            ir, note=note, duration_sec=duration_sec,
            risk_profile=risk_profile, velocity_q15=velocity_q15,
            cents_offset=round(spread * t),
            phase_offset_q32=phase_q32,
            noise_seed_xor=phase_q32,
            verbose=(k == 0),
        ).astype(np.float64)
        theta = (width * t + 1.0) * (math.pi / 4.0)  # equal-power pan
        if mix_l is None:
            mix_l = voice * math.cos(theta)
            mix_r = voice * math.sin(theta)
        else:
            mix_l += voice * math.cos(theta)
            mix_r += voice * math.sin(theta)

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
