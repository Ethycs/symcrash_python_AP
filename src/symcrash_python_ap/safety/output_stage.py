"""Shared stereo output ownership for SynthMaster and live renderers.

VMs supply mono voice blocks. Pan/mix, consumer events, Material/aux returns,
master gain and final Safety run once per output block, including empty blocks.
"""
from ..audio_routing import PostMixControls, AuxiliaryReturns
from ..fixed import mul_q15_sat, RSHIFT_RNE
from ..frames import ParamFrame
from ..material.material_engine import MaterialEngine
from ..primitives import Q15_ONE, Q15_HALF, Q15_SCALE, Q15_NEG_ONE
from ..wavelet.sine_core import sine_q15, cosine_q15
from .safety_engine import SafetyEngine, PEAK_MAX, SLEW_MAX, DC_MAX, OLA_OVERLAP
from .output_projection import output_feasible


class StereoOutputMixin:
    """Output DSP and configuration, independent of voice clocks or device I/O."""

    def _init_output_stage(self, block_size):
        from ..instrument_ir import SafetyConfig
        self.output_block_size = block_size
        self.material = MaterialEngine(block_size)
        self.safety = SafetyEngine(block_size)
        self._material_id = 0
        self._beta_target_q15 = Q15_HALF
        self._beta_start_q15 = 0
        self._beta_ramp_blocks = 8
        self._enable_material = True
        self._enable_safety = True
        self._safety_override = None
        self._output_preset = None
        self._emitted_tail_L = []
        self._emitted_tail_R = []
        self.post_mix_controls = PostMixControls()
        self.auxiliary_returns = AuxiliaryReturns(block_size)
        self.last_post_mix_params = ParamFrame()
        self.configure_safety(SafetyConfig())

    def configure_preset(self, preset):
        """Apply selected global defaults once, without resetting DSP histories."""
        if preset is not self._output_preset:
            self._apply_preset_output(preset)

    def configure_aux(self, name, material_id, params=None, return_gain_q15=Q15_ONE):
        self.auxiliary_returns.configure(name, material_id, params, return_gain_q15)

    def set_material(self, material_id):
        if isinstance(material_id, bool) or not isinstance(material_id, int) or not 0 <= material_id <= 10:
            raise ValueError('Material program ID must be an integer in 0..10')
        self._material_id = material_id

    def disable_output_stage(self, stage):
        if stage == 'material':
            self._enable_material = False
        elif stage == 'safety':
            self._safety_override = False
            self._enable_safety = False
        else:
            raise ValueError('Unknown output stage: ' + stage)

    def enable_output_stage(self, stage):
        if stage == 'material':
            self._enable_material = True
        elif stage == 'safety':
            previous = self._safety_override
            self._safety_override = True
            try:
                self.configure_safety(self._safety_config)
            except (TypeError, ValueError):
                self._safety_override = previous
                raise
        else:
            raise ValueError('Unknown output stage: ' + stage)

    def configure_safety(self, config: 'SafetyConfig') -> None:
        """Latch authored output settings without resetting DSP or emitted history.

        Hard peak and slew requests may tighten, but never relax, the normative
        bounds. Infeasible changes fail before changing the active configuration.
        Explicit enable/disable debug overrides persist across preset changes.
        """
        from dataclasses import asdict
        from ..instrument_ir import SafetyConfig
        if not isinstance(config, SafetyConfig):
            raise TypeError('Expected SafetyConfig')
        config = SafetyConfig.from_dict(asdict(config))  # Validate even mutated dataclasses.
        peak = min(PEAK_MAX, config.limiter_threshold_q15)
        slew = min(SLEW_MAX, config.slew_limit_q15)
        enabled = config.enabled if self._safety_override is None else self._safety_override
        anchor = bool(config.master_ola and self._emitted_tail_L)
        previous = (self._emitted_tail_L[-1] if self._emitted_tail_L else 0,
                    self._emitted_tail_R[-1] if self._emitted_tail_R else 0)
        if enabled and not all(output_feasible(p, self.output_block_size, peak, slew,
                                DC_MAX if config.dc_removal else -1, anchor) for p in previous):
            raise ValueError('Safety settings are infeasible for the previous emitted output')
        self._safety_config = config
        self._enable_safety = enabled
        self._safety_dc_removal = config.dc_removal
        self._safety_master_ola = config.master_ola
        self._safety_limiter_threshold = config.limiter_threshold_q15
        self._safety_slew_limit = config.slew_limit_q15
        # This authored field is a hard threshold, not the standalone soft knee.
        self.safety.LIMIT_CEILING = self.safety.LIMIT_THRESHOLD = peak
        self.safety.max_slew = slew


    def _apply_preset_output(self, preset: 'CompiledPreset') -> None:
        """One output owner for YAML, capsules, compiled and cockpit presets."""
        material_id = preset.material.material_id
        if isinstance(material_id, bool) or not isinstance(material_id, int) or not 0 <= material_id <= 10:
            raise ValueError('Material program ID must be an integer in 0..10')
        from ..instrument_ir import SafetyConfig
        self.configure_safety(getattr(preset, "safety", SafetyConfig()))
        self._material_id = material_id
        self._beta_start_q15 = preset.material.beta_start_q15
        self._beta_target_q15 = preset.material.beta_target_q15
        self._beta_ramp_blocks = preset.material.beta_ramp_blocks
        self._output_preset = preset


    def _mix_output(self, rendered, param_frame, consumer_events=(), *, mono=False):
        """Mix (VoiceFrame, mono samples) pairs, then run the shared final tail."""
        rendered = list(rendered)
        if not isinstance(param_frame, ParamFrame):
            raise TypeError('Output parameters must be a ParamFrame')
        block_size = self.output_block_size
        if any(len(samples) != block_size for _, samples in rendered):
            raise ValueError('Output voices must contain complete audio blocks')
        L, R = [0] * block_size, [0] * block_size
        send_inputs = self.auxiliary_returns.empty_inputs()
        rendered_frames = [frame for frame, _ in rendered]
        for frame, samples in rendered:
            phase = (max(Q15_NEG_ONE, min(Q15_ONE, frame.pan_q15)) + Q15_SCALE) << 14
            left_gain, right_gain = cosine_q15(phase), sine_q15(phase)
            left = [mul_q15_sat(int(sample), left_gain) for sample in samples]
            right = [mul_q15_sat(int(sample), right_gain) for sample in samples]
            for i in range(block_size):
                L[i] += left[i]
                R[i] += right[i]
            self.auxiliary_returns.add_voice(send_inputs, left, right,
                                            frame.sendA_q15, frame.sendB_q15)

        # Clip to int16 range
        L = [max(Q15_NEG_ONE, min(Q15_ONE, s)) for s in L]
        R = [max(Q15_NEG_ONE, min(Q15_ONE, s)) for s in R]

        # Aggregate filter parameters from active voices
        # Use minimum cutoff (darkest voice wins) for musical coherence
        filter_enabled = param_frame.filter_enabled
        filter_cutoff_q15 = param_frame.filter_cutoff_q15 if filter_enabled else Q15_ONE
        filter_reso_q15 = param_frame.filter_reso_q15 if filter_enabled else 0
        filter_mode = param_frame.filter_mode if filter_enabled else 0

        for frame in rendered_frames:
            if frame is not None:
                fc = getattr(frame, 'filter_cutoff_q15', Q15_ONE)
                fr = getattr(frame, 'filter_reso_q15', 0)
                fm = getattr(frame, 'filter_mode', 0)
                # Use minimum cutoff (darkest voice)
                if fc < filter_cutoff_q15:
                    filter_cutoff_q15 = fc
                    filter_mode = fm
                # Use maximum resonance
                if fr > filter_reso_q15:
                    filter_reso_q15 = fr
                # Enable filter if any voice has cutoff below max
                if fc < Q15_ONE:
                    filter_enabled = True

        # Consumer time belongs to the output block, including blocks with no
        # serviced voices. Existing material slew applies burst targets.
        param_frame = self.post_mix_controls.step(param_frame, consumer_events)
        self.last_post_mix_params = param_frame

        # Apply post-processing if enabled (Stages 9-10)
        if self._enable_material:
            # Use material_id=9 (SVF filter) if filter is enabled, otherwise use configured material
            material_id = 9 if filter_enabled else self._material_id
            L, R = self.material.step_block(
                L, R, param_frame, material_id, 0,
                filter_cutoff_q15=filter_cutoff_q15,
                filter_reso_q15=filter_reso_q15,
                filter_mode=filter_mode
            )
            aux_l, aux_r = self.auxiliary_returns.process(send_inputs)
            L = [max(Q15_NEG_ONE, min(Q15_ONE, x + y)) for x, y in zip(L, aux_l)]
            R = [max(Q15_NEG_ONE, min(Q15_ONE, x + y)) for x, y in zip(R, aux_r)]
        else:
            # The debug bypass freezes effect DSP state but still exposes buses.
            self.auxiliary_returns.last_inputs = {
                name: ([max(Q15_NEG_ONE, min(Q15_ONE, x)) for x in left],
                       [max(Q15_NEG_ONE, min(Q15_ONE, x)) for x in right])
                for name, (left, right) in send_inputs.items()}
        # Device mono conversion precedes Safety, so rounding cannot invalidate
        # the final slew/DC guarantees. The stereo effects still run only once.
        if mono:
            L = [RSHIFT_RNE(int(left) + int(right), 1) for left, right in zip(L, R)]
            R = list(L)
        # Master gain belongs to the final output path, including effect tails.
        # Material/Safety do not consume ParamFrame.gate_q15 themselves.
        if param_frame.gate_q15 != Q15_ONE:
            L = [mul_q15_sat(int(s), param_frame.gate_q15) for s in L]
            R = [mul_q15_sat(int(s), param_frame.gate_q15) for s in R]
        if self._enable_safety:
            # A debug bypass freezes DSP state, but output history belongs to
            # the stream. Re-enabling must connect to what was actually emitted.
            if self._emitted_tail_L:
                self.safety.prev_sample_L = self._emitted_tail_L[-1]
                self.safety.prev_sample_R = self._emitted_tail_R[-1]
                self.safety.master_ola_L = list(self._emitted_tail_L)
                self.safety.master_ola_R = list(self._emitted_tail_R)
            L, R = self.safety.step_block(L, R, param_frame,
                apply_master_ola=self._safety_master_ola,
                apply_dc_removal=self._safety_dc_removal)
        self._emitted_tail_L = list(L[-OLA_OVERLAP:])
        self._emitted_tail_R = list(R[-OLA_OVERLAP:])

        return L, R
