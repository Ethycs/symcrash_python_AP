"""
IR-driven per-block VoiceFrame builder (synthesis-path dedup, step 8).

Single owner of authoring-side frame construction: the offline renderer
(tools/render/render_preset.py), workbench streaming
(symcrash_workbench/streaming.py) and the imgui workbench renderer all
build their per-block VoiceFrames here. The live/performance twin is
voice_engine.step_voice_simple (CompiledPreset-driven); frame
construction is the Voice Engine's mandate (WSE-STD-CAPSULE-COMPAT §5.3).

Two latch modes:
- relatch_per_block=False (offline): all IR parameters are extracted once
  at construction; LFO modulators are deep-copied so unison copies advance
  private LFO state and the IR is never mutated. Amp key-scaling is
  applied to the already-converted block counts (historical offline law).
- relatch_per_block=True (streaming): every build() re-extracts from the
  live IR (knob edits latch at block boundaries) and advances the IR's
  own stateful modulators. Amp key-scaling multiplies the ms values
  before block conversion (historical streaming law).

DELIBERATE RESIDUAL DUPLICATION: this module carries the offline rounding
law (`(x * y + 16384) >> 15`); voice_engine.step_voice_simple uses
RSHIFT_RNE/mul_q15_sat. The two laws are NOT unified - doing so would
change ear-tuned output. Parity between them is enforced at tolerance by
tests/test_envelope_bindings_live.py.
"""

import copy as _copylib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..primitives import Q15_HALF, Q15_ONE, Q16_ONE, ms_to_blocks
from ..envelope_scheduler import GATE_HELD, adsr_value_q15, BindingEnvelope
from ..envelope_targets import EnvelopeTarget
from ..frames import VoiceFrame
from ..instrument_ir import (
    extract_filter_params,
    extract_noise_params,
    extract_oeis_params,
    extract_partials_q15,
    extract_phase_spread_q15,
    extract_waveform_type,
)
from ..library.ir_extract import extract_amp, extract_fm6
from ..vm.harmonic_bank import LFOTarget
from ..vm.state import OSC_MODE_FM6

# Golden-ratio conjugate scaled to 2^32 - the repo's golden-angle phase
# convention (single home; also used by vm/harmonic_bank partial spread).
GOLDEN_Q32 = 0x9E3779B9

_PHASE_TO_TEMPLATE = {
    'ATTACK': 'T_ATTACK', 'DECAY': 'T_STABLE',
    'SUSTAIN': 'T_STABLE', 'RELEASE': 'T_RECOVERY',
}


# ============================================================ shared helpers

def velocity_curve_q15(velocity_sens_q15: int, velocity_q15: int) -> int:
    """value scale = 1 - sens + sens*velocity (voice_engine semantics)."""
    return (Q15_ONE - velocity_sens_q15
            + ((velocity_sens_q15 * velocity_q15) >> 15))


def unison_positions(n: int) -> List[Tuple[float, int]]:
    """
    Unison copy positions: [(t_k, phase_q32_k)].

    t_k = (2k - (N-1)) / (N-1) in [-1, +1] (0.0 for a single voice);
    phase_q32_k = (k * GOLDEN_Q32) mod 2^32 (golden-angle sequence).
    """
    out = []
    for k in range(n):
        t = 0.0 if n == 1 else (2 * k - (n - 1)) / (n - 1)
        out.append((t, (k * GOLDEN_Q32) & 0xFFFFFFFF))
    return out


def equal_power_pan(t: float, width: float) -> Tuple[float, float]:
    """Equal-power pan gains (L, R) for position width*t in [-1, +1]."""
    theta = (width * t + 1.0) * (math.pi / 4.0)
    return math.cos(theta), math.sin(theta)


def filtered_partials(base_amps_q15, f0_hz, cutoff_hz, resonance):
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
        out.append(max(0, min(Q15_ONE, round(amp * gain))))
    return out


def get_emit_template(ir, template_name: str) -> dict:
    """Get an L3 emit template's {families, ops_enable} by name."""
    templates = ir.layers.L3_cmmm.emit_templates
    if template_name in templates:
        t = templates[template_name]
        return {
            'families': dict(t.families),
            'ops_enable': list(t.ops_enable),
        }
    return {'families': {}, 'ops_enable': []}


# ================================================================ note clock

@dataclass
class NoteClock:
    """
    Where we are in the note.

    sustain_until_block: block where release begins. Offline law:
        attack_blocks + sustain_blocks (timeline known up front).
        Streaming law: GATE_HELD while the gate is down, then the
        note-off block.
    release_from_q15: level release decays from; None = sustain level
        (offline law). Streaming latches the last raw amp gain at
        note-off.
    """
    block_idx: int
    gate: bool = True
    sustain_until_block: int = GATE_HELD
    release_from_q15: Optional[int] = None


# ============================================================= frame builder

class IRFrameBuilder:
    """
    Builds one VoiceFrame per block from an InstrumentIR.

    After each build() the raw (pre-LFO, pre-velocity) amp gain is
    available as `last_amp_gain_q15` - streaming latches it at note-off
    as the release-from level.
    """

    def __init__(
        self,
        ir,
        note: int,
        velocity_q15: int = Q15_ONE,
        *,
        cents_offset: int = 0,
        noise_seed_xor: int = 0,
        relatch_per_block: bool = False,
    ):
        self.ir = ir
        self.note = note
        self.velocity_q15 = velocity_q15
        self.cents_offset = cents_offset
        self.noise_seed_xor = noise_seed_xor
        self.relatch = relatch_per_block
        self.last_amp_gain_q15 = 0

        if not self.relatch:
            # Offline: latch once; deep-copy LFOs (private state per voice
            # copy, IR untouched)
            self._lfos = [
                _copylib.deepcopy(m)
                for m in (getattr(ir, 'modulators', None) or [])
            ]
            self._latch()
        else:
            self._lfos = None  # live IR modulators, fetched per build

    # ------------------------------------------------------------ latching

    def _latch(self):
        """Extract all frame parameters from the IR (one latch)."""
        ir = self.ir
        note = self.note

        # --- Amp envelope + dynamics ---
        amp = extract_amp(ir)
        if self.relatch:
            # Streaming law: no-Amp defaults are 1-block ramps at full
            # sustain; key scaling multiplies ms before block conversion.
            a, d, r = 1, 1, 1
            s_q15 = Q15_ONE
            vel_sens_q15 = Q15_ONE
            if amp is not None:
                ks = amp.key_scale_rate
                mult = 2.0 ** (-ks * (note - 60) / 24.0) if ks else 1.0
                a = ms_to_blocks(amp.a_ms * mult)
                d = ms_to_blocks(amp.d_ms * mult)
                r = ms_to_blocks(amp.r_ms * mult)
                s_q15 = int(amp.s * Q15_ONE)
                vel_sens_q15 = int(amp.velocity_sens * Q15_ONE)
        else:
            # Offline law: legacy defaults; key scaling scales the
            # already-converted block counts.
            if amp is not None:
                a = ms_to_blocks(amp.a_ms)
                d = ms_to_blocks(amp.d_ms)
                r = ms_to_blocks(amp.r_ms)
                sustain = amp.s
                ks = amp.key_scale_rate
                if ks:
                    mult = 2.0 ** (-ks * (note - 60) / 24.0)
                    a = max(1, int(a * mult))
                    d = max(1, int(d * mult))
                    r = max(1, int(r * mult))
                vel_sens_q15 = int(amp.velocity_sens * Q15_ONE)
            else:
                a = ms_to_blocks(10)
                d = ms_to_blocks(100)
                sustain = 0.7
                r = ms_to_blocks(200)
                vel_sens_q15 = Q15_ONE
            s_q15 = int(sustain * Q15_ONE)
        self.attack_blocks = a
        self.decay_blocks = d
        self.sustain_q15 = s_q15
        self.release_blocks = r
        self._vel_scale_q15 = velocity_curve_q15(vel_sens_q15,
                                                 self.velocity_q15)

        # --- Spectrum sources ---
        self._harmonic_amps = extract_partials_q15(ir) or []
        self._phase_spread_q15 = extract_phase_spread_q15(ir)
        self._oeis_params = extract_oeis_params(ir)
        self.waveform_type = extract_waveform_type(ir)

        # --- Noise exciter (empty kwargs = bit-exact no-op in the VM) ---
        noise_params = extract_noise_params(ir)
        self._noise_kwargs = {}
        if noise_params:
            self._noise_kwargs = dict(
                noise_color=noise_params['color'],
                noise_level_q15=noise_params['level_q15'],
                noise_attack_blocks=ms_to_blocks(noise_params['attack_ms']),
                noise_decay_blocks=ms_to_blocks(noise_params['decay_ms']),
                noise_seed_u32=((noise_params['seed_u32']
                                 ^ self.noise_seed_xor) & 0xFFFFFFFF),
            )

        # --- Filter (partials reshaping, swept by FILTER_CUTOFF bindings) ---
        self._filter_params = (extract_filter_params(ir)
                               if self._harmonic_amps else None)

        # --- Matrix-envelope bindings ---
        self._env_states = [
            BindingEnvelope(b, note)
            for b in (getattr(ir, 'envelopes', None) or [])
        ]
        self._filter_envs = [
            e for e in self._env_states
            if int(EnvelopeTarget.FILTER_CUTOFF) in e.target_ids
        ]

        # --- PWM base width from the Osc node ---
        pwm_width = 16384
        pwm_enabled = False
        for node in ir.layers.L0_graph.nodes:
            if node.type == "Osc":
                if getattr(node.params, 'pwm_width_q15', None) not in (None, Q15_HALF):
                    pwm_width = node.params.pwm_width_q15
                if getattr(node.params, 'pw', 0.5) != 0.5:
                    pwm_width = int(node.params.pw * Q15_ONE)
                pwm_enabled = bool(getattr(node.params, 'pwm_enable', False))
                break
        # A PWM-targeted LFO implies PWM (only without custom partials)
        lfos = self._live_lfos()
        if not self._harmonic_amps and any(
            r.target == LFOTarget.PWM_WIDTH for lfo in lfos
            for r in lfo.routings
        ):
            pwm_enabled = True
        self._pwm_width_base = pwm_width
        self._pwm_enabled = pwm_enabled

        # --- FM6 (first FM6 node wins; tool-side defaults) ---
        fm6 = extract_fm6(ir)
        self._fm6 = None
        if fm6 is not None:
            self._fm6 = {
                'algorithm': fm6.algorithm if fm6.algorithm is not None else 1,
                'feedback': fm6.feedback if fm6.feedback is not None else 0,
                'op_ratios': fm6.op_ratios or [Q16_ONE] * 6,
                'op_levels': fm6.op_levels or [Q15_ONE, 0, 0, 0, 0, 0],
                'post_filter_hz': fm6.post_filter_hz,
                'mod_index_q16': (round(fm6.mod_index * Q16_ONE)
                                  if fm6.mod_index is not None else None),
            }

    def _live_lfos(self):
        if self._lfos is not None:
            return self._lfos
        return list(getattr(self.ir, 'modulators', None) or [])

    # ------------------------------------------------------------ per block

    def phase_at(self, clock: NoteClock) -> str:
        """ADSR phase name at this block (drives emit-template choice)."""
        if not clock.gate:
            return 'RELEASE'
        if clock.block_idx < self.attack_blocks:
            return 'ATTACK'
        if clock.block_idx < self.attack_blocks + self.decay_blocks:
            return 'DECAY'
        if clock.block_idx < clock.sustain_until_block:
            return 'SUSTAIN'
        return 'RELEASE'

    def set_fm6_op_levels(self, levels) -> None:
        """Override FM6 operator levels (workbench character transforms)."""
        if self._fm6 is not None:
            self._fm6 = dict(self._fm6, op_levels=list(levels))

    def build(
        self,
        clock: NoteClock,
        ops_enable: Optional[list] = None,
        families: Optional[dict] = None,
        amps_override: Optional[list] = None,
    ) -> VoiceFrame:
        """
        Build the VoiceFrame for this block.

        ops_enable/families: pass explicitly when an L2 regime machine
        drives emission (offline renderer); None selects the emit template
        from the ADSR phase (simple-form law, shared by both paths).
        amps_override: replaces the extracted harmonic profile for this
        block (workbench character transforms).
        """
        if self.relatch:
            self._latch()  # knob edits latch at block boundaries

        b = clock.block_idx

        # --- Amp gain (raw, pre-LFO/velocity) ---
        if not clock.gate:
            # Gate law: note-off forces release regardless of where the
            # block index sits in the attack/decay timeline (streaming
            # latches the last raw gain as the release-from level).
            start = (clock.release_from_q15
                     if clock.release_from_q15 is not None
                     else self.sustain_q15)
            t = ((b - clock.sustain_until_block) / self.release_blocks
                 if self.release_blocks > 0 else 1)
            gain_q15 = max(0, min(Q15_ONE, int(start * (1.0 - t))))
        else:
            gain_q15 = adsr_value_q15(
                self.attack_blocks, self.decay_blocks, self.sustain_q15,
                self.release_blocks, b, clock.sustain_until_block,
                release_from_q15=clock.release_from_q15,
            )
        self.last_amp_gain_q15 = gain_q15
        phase = self.phase_at(clock)

        # --- Emit template by phase when the caller has no regime machine ---
        if ops_enable is None:
            ops_enable = []
            families = {}
            templates = self.ir.layers.L3_cmmm.emit_templates
            for cand in (_PHASE_TO_TEMPLATE.get(phase, 'T_STABLE'),
                         'T_STABLE', 'T_IDLE'):
                if cand in templates:
                    tmpl = get_emit_template(self.ir, cand)
                    ops_enable = tmpl['ops_enable']
                    families = tmpl['families']
                    break
        elif families is None:
            families = {}

        # --- LFO modulators ---
        total_cents = 0
        gain_scale_q15 = Q15_ONE
        pwm_width = self._pwm_width_base
        for lfo in self._live_lfos():
            lfo.advance_block()
            lfo_val = lfo.get_lfo_value()
            for routing in lfo.routings:
                scaled = (lfo_val * routing.amount_q15 + Q15_HALF) >> 15
                if routing.target == LFOTarget.PITCH:
                    total_cents += scaled
                elif routing.target == LFOTarget.GAIN:
                    gain_scale_q15 = max(0, min(Q15_ONE, Q15_ONE + scaled))
                elif routing.target == LFOTarget.PWM_WIDTH:
                    pwm_width = max(1638, min(31130, pwm_width + scaled))

        # --- Filter sweep (partials reshaping) ---
        harmonics = (amps_override if amps_override is not None
                     else self._harmonic_amps)
        block_amps = harmonics
        if self._filter_params and harmonics:
            env_v = Q15_ONE
            if self._filter_envs:
                env_v = max(
                    e.value_q15(b, clock.sustain_until_block)
                    for e in self._filter_envs
                )
            sweep_oct = (4.0 * self._filter_params['env_amount']
                         * (env_v / 32767.0))
            eff_cutoff = self._filter_params['cutoff_hz'] * (2.0 ** sweep_oct)
            f0 = 440.0 * 2 ** ((self.note - 69) / 12)
            block_amps = filtered_partials(
                harmonics, f0, eff_cutoff,
                self._filter_params['resonance'])

        # --- Matrix-envelope targets ---
        env_targets_q15 = None
        if self._env_states:
            env_targets_q15 = {}
            for es in self._env_states:
                value = es.value_q15(b, clock.sustain_until_block)
                if es.velocity_sens_q15:
                    scale = velocity_curve_q15(es.velocity_sens_q15,
                                               self.velocity_q15)
                    value = (value * scale) >> 15
                for target_id in es.target_ids:
                    env_targets_q15[target_id] = value

        final_gain = ((((gain_q15 * gain_scale_q15 + Q15_HALF) >> 15)
                       * self._vel_scale_q15 + Q15_HALF) >> 15)
        cents = int(total_cents) + self.cents_offset

        # --- FM6 path ---
        if self._fm6 is not None:
            post_hz = self._fm6['post_filter_hz']
            return VoiceFrame(
                note=self.note,
                cents=cents,
                coeff_osc_mode=OSC_MODE_FM6,
                gain_q15=final_gain,
                beta_target_q15=0,
                fm_algorithm_id=self._fm6['algorithm'],
                fm_op_ratios=self._fm6['op_ratios'],
                fm_op_levels=self._fm6['op_levels'],
                fm_feedback_depth=self._fm6['feedback'],
                fm_mod_index_q16=self._fm6['mod_index_q16'],
                fm_post_filter_enabled=bool(post_hz) or None,
                fm_post_filter_cutoff_hz=post_hz,
                env_targets_q15=env_targets_q15,
                ops_enable=ops_enable,
                families=families,
                **self._noise_kwargs,
            )

        # --- Mode B path ---
        oeis = self._oeis_params
        return VoiceFrame(
            note=self.note,
            cents=cents,
            coeff_osc_mode=1,
            H_request=len(harmonics) or 16,
            gain_q15=final_gain,
            beta_target_q15=0,
            partial_amps_q15=block_amps,
            partial_phase_spread_q15=self._phase_spread_q15,
            ops_enable=ops_enable,
            families=families,
            waveform_type=self.waveform_type,
            oeis_sequence_id=oeis['sequence_id'] if oeis else -1,
            oeis_rolloff_db=oeis['rolloff_db'] if oeis else -3.0,
            pwm_width_q15=pwm_width,
            pwm_enabled=self._pwm_enabled,
            env_targets_q15=env_targets_q15,
            **self._noise_kwargs,
        )


__all__ = [
    'GOLDEN_Q32', 'GATE_HELD',
    'NoteClock', 'IRFrameBuilder',
    'velocity_curve_q15', 'unison_positions', 'equal_power_pan',
    'filtered_partials', 'get_emit_template',
]
