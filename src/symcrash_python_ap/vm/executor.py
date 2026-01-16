"""
VM Executor (WSE-STD-VM-HYBRID v0)
==================================

Main execution entry point for the hybrid VM.
Executes one block for one voice with fixed lane schedule.
"""

from typing import Tuple, Optional, TYPE_CHECKING
import numpy as np

from .state import (
    VMState, OSC_MODE_A, OSC_MODE_B, OSC_MODE_SAMPLE, OSC_MODE_FM6,
    PAYLOAD_TYPE_NODEMASK_SET, PAYLOAD_TYPE_TAILMODE_SET,
)
from .decode import DecodedPlan, CarrierRole, decode_frame, FM_SOURCE_FEEDBACK
from .tables import Tables, DEFAULT_TABLES, BLOCK_SAMPLES
from .lanes import (
    lane_carrier_render,
    lane_inject,
    dispatch_coeff_op,
    dispatch_coeff_op_crossfade,  # WSE-STD-070: crossfade-aware dispatch
    lane_tail,
    lane_idwpt_render,
    lane_ccs_render,
    lane_mix_gain,
)
from ..fixed import RSHIFT_RNE

if TYPE_CHECKING:
    from ..frames import VoiceFrame


# =============================================================================
# LATENCY CONSTANTS
# =============================================================================

IDWPT_LATENCY_BLOCKS = 5  # Blocks until wet signal is ready


# =============================================================================
# CMMM EVENT PROCESSING
# =============================================================================

def _process_cmmm_events(state: VMState, frame: 'VoiceFrame') -> None:
    """
    Process CMMM event payloads from VoiceFrame.

    Updates the ExecutorLatch based on CMMM events:
    - 0x30 (NODEMASK_SET): Changes node mask with optional ramp
    - 0x31 (TAILMODE_SET): Changes tail mode with traction

    Args:
        state: Per-voice VM state (modified in place)
        frame: VoiceFrame containing CMMM event fields
    """
    latch = state.executor_latch

    # Step 1: Advance mask ramp (must happen every block)
    latch.step_mask_ramp()

    # Step 2: Process node mask event (payload_type 0x30)
    node_mask = getattr(frame, 'cmmm_node_mask', None)
    if node_mask is not None:
        ramp_blocks = getattr(frame, 'cmmm_mask_ramp_blocks', 5)
        latch.apply_nodemask_event(node_mask, ramp_blocks)

    # Step 3: Process tail mode event (payload_type 0x31)
    tail_mode = getattr(frame, 'cmmm_tail_mode', None)
    if tail_mode is not None:
        traction = getattr(frame, 'cmmm_traction_q15', 32767)
        latch.apply_tailmode_event(tail_mode, traction)


# =============================================================================
# MAIN EXECUTOR
# =============================================================================

def vm_step_block(
    state: VMState,
    frame: 'VoiceFrame',
    tables: Tables = DEFAULT_TABLES,
) -> Tuple[VMState, np.ndarray]:
    """
    Execute one block for one voice.

    This is the main entry point for the hybrid VM. It executes the
    fixed lane schedule deterministically.

    Args:
        state: Per-voice VM state (modified in place)
        frame: VoiceFrame microcode for this block
        tables: Immutable tables for lookups

    Returns:
        (state, output_block) where:
            state: Updated VM state
            output_block: int16 array of BLOCK_SAMPLES
    """
    # -------------------------------------------------------------------------
    # Step 0: Process CMMM events (before decode)
    # -------------------------------------------------------------------------
    _process_cmmm_events(state, frame)

    # -------------------------------------------------------------------------
    # Step 1: Decode configuration
    # -------------------------------------------------------------------------
    cfg = decode_frame(
        frame,
        tables,
        prev_eta_q15=state.eta_prev_q15,
        prev_gain_q15=state.gain_prev_q15,
    )

    # Track blocks since start (for latency model)
    state.blocks_since_start += 1

    # Check if wet is ready (microcode contract)
    if state.blocks_since_start >= IDWPT_LATENCY_BLOCKS:
        state.wet_ready = True
    else:
        # Force eta to 0 (pure carrier) during latency period
        cfg.eta_curr_q15 = 0
        cfg.idwpt_enable = False

    # Update tree state
    state.tree_a_id = cfg.tree_a
    state.tree_b_id = cfg.tree_b

    # =========================================================================
    # CCS PATH (Coefficient Synthesis) - WSE-STD-ARCH-CCS COMPLIANT
    # =========================================================================
    # When use_coeff_osc is True, we use the TRUE CCS path per spec section 2.1:
    #   Phase → CoeffTable → Coefficients → Operators → IDWPT → Audio
    #
    # This eliminates FWPT from the voice path and ensures operators modify
    # the carrier coefficients before IDWPT reconstruction.

    if cfg.use_coeff_osc and state.use_coeff_osc:
        # -----------------------------------------------------------------
        # CCS PATH: Coefficient-domain architecture (WSE-STD-121 COMPLIANT)
        # -----------------------------------------------------------------
        # Uses StreamingCoeffLUT with per-note tables that have frequency
        # baked into the FWPT coefficients. No FWPT in runtime voice path.
        #
        # WSE-STD-210: Mode A (LUT) vs Mode B (Rotator/Harmonic Bank)
        # - Mode A: Phase → LUT lookup → Coefficients (existing)
        # - Mode B: Phase → sin/cos weights × basis frames → Coefficients

        from .tables import COEFF_LEN

        # Get oscillator params
        params = cfg.coeff_osc_params
        osc_mode = params.osc_mode

        # Update state if pitch changed
        pitch_changed = state.last_note != params.note or state.last_cents != params.cents
        if pitch_changed:
            state.ccs_state.note = params.note
            state.ccs_state.wave_id = params.table_id
            state.last_note = params.note
            state.last_cents = params.cents

        # Update Mode B state if using rotator (always check for first-time init)
        if osc_mode == OSC_MODE_B:
            mb_state = state.mode_b_state
            first_time_init = mb_state.phase_inc == 0

            # Decide whether to reset (retrigger) or use legato (smooth pitch change)
            if first_time_init:
                # First-time initialization: always do full reset
                mb_state.reset(note=params.note, phase_inc=params.phase_inc)
                mb_state.pwm_width_q15 = params.pwm_width_q15
                mb_state.pwm_enabled = params.pwm_enabled
                mb_state.H_active = min(params.H_request, mb_state.H_bandlimit)
                # Set harmonic weights: custom harmonics override waveform_type
                if params.custom_harmonics:
                    mb_state.weights.set_custom(params.custom_harmonics)
                    mb_state.H_active = min(len(params.custom_harmonics), mb_state.H_bandlimit)
                else:
                    waveform_type = getattr(params, 'waveform_type', 0)
                    mb_state.weights.set_from_waveform_type(
                        waveform_type, mb_state.H_active, params.pwm_width_q15
                    )
            elif pitch_changed or mb_state.note != params.note:
                # Pitch changed: use legato_mode to decide how to handle
                if params.legato_mode:
                    # Legato: preserve phase for smooth transition
                    mb_state.set_pitch_legato(note=params.note, phase_inc=params.phase_inc)
                    mb_state.H_active = min(params.H_request, mb_state.H_bandlimit)
                else:
                    # Non-legato: full reset (may cause click, but musically intentional)
                    mb_state.reset(note=params.note, phase_inc=params.phase_inc)
                    mb_state.pwm_width_q15 = params.pwm_width_q15
                    mb_state.pwm_enabled = params.pwm_enabled
                    mb_state.H_active = min(params.H_request, mb_state.H_bandlimit)
                    # Set harmonic weights: custom harmonics override waveform_type
                    if params.custom_harmonics:
                        mb_state.weights.set_custom(params.custom_harmonics)
                        mb_state.H_active = min(len(params.custom_harmonics), mb_state.H_bandlimit)
                    else:
                        waveform_type = getattr(params, 'waveform_type', 0)
                        mb_state.weights.set_from_waveform_type(
                            waveform_type, mb_state.H_active, params.pwm_width_q15
                        )

            # Update FM depth from VoiceFrame (modulator signal comes from fm_mod_buffer)
            # Phase 3 FM/PM will provide the modulator via operator graph
            mb_state.fm_depth_q15 = params.fm_depth_q15

        if state.ccs_state.phase_inc != params.phase_inc:
            state.ccs_state.set_pitch(params.phase_inc, slew_blocks=8)
            if osc_mode == OSC_MODE_B:
                state.mode_b_state.phase_inc = params.phase_inc

        # Generate polyphase carrier: 64 leaves × K k-slots = 512 values (K=8)
        carrier_512 = np.zeros(COEFF_LEN, dtype=np.int16)

        if osc_mode == OSC_MODE_B:
            # -----------------------------------------------------------------
            # MODE B: Coefficient synthesis
            # -----------------------------------------------------------------
            from .harmonic_bank import (
                mode_b_generate_coeffs, mode_b_generate_coeffs_matrix,
                get_basis_for_note
            )
            from .waveform_matrix import WaveformType

            # Determine which synthesis method to use:
            # - Matrix method: PL waveforms (SQUARE, SAW, TRIANGLE) without FM/PWM
            # - Harmonic bank: SINE, PWM-modulated, FM-modulated signals
            waveform_type = getattr(params, 'waveform_type', WaveformType.SINE)
            use_matrix_method = (
                waveform_type in (WaveformType.SQUARE, WaveformType.SAW,
                                  WaveformType.SAW_DOWN, WaveformType.TRIANGLE)
                and not params.pwm_enabled
                and params.fm_depth_q15 == 0
            )

            if use_matrix_method:
                # Matrix method: precomputed coefficient LUT (no runtime FWPT)
                mode_b_generate_coeffs_matrix(
                    state.mode_b_state, waveform_type, carrier_512
                )
            else:
                # Harmonic bank: basis frames + runtime sin/cos modulation
                basis = get_basis_for_note(params.note)

                # Update PWM if enabled
                if params.pwm_enabled:
                    state.mode_b_state.set_pwm_weights(params.pwm_width_q15)

                # === FM MODULATOR WIRING (Phase 3 - WSE-STD-210 Section 8.3) ===
                # Wire previous block's modulator feedback to carrier.
                # z⁻¹ block delay ensures deterministic, bounded-runtime feedback.
                if params.fm_depth_q15 != 0 and params.fm_source == FM_SOURCE_FEEDBACK:
                    state.mode_b_state.fm_mod_buffer = state.prev_fm_mod_buffer
                else:
                    state.mode_b_state.fm_mod_buffer = None

                # Generate coefficients using harmonic bank (with FM if active)
                mode_b_generate_coeffs(state.mode_b_state, basis, carrier_512)

            # Advance Mode B phase for next block
            state.mode_b_state.advance_block()
        elif osc_mode == OSC_MODE_SAMPLE:
            # -----------------------------------------------------------------
            # MODE SAMPLE: Audio-native via FWPT gateway (Phase 3)
            # -----------------------------------------------------------------
            from .sample_source import sample_generate_coeffs, compute_playback_rate

            # Update playback rate from VoiceFrame parameters
            # If sample has a root note, pitch-shift based on played note
            sample_state = state.sample_state
            if sample_state.buffer is not None:
                # Get sample rate from frame (default 1.0x)
                sample_rate = getattr(frame, 'sample_rate_q16', 65536) or 65536

                # If sample has root note metadata, compute pitch shift
                root_note = getattr(sample_state, 'root_note', None)
                if root_note is not None and root_note != params.note:
                    # Pitch shift to match requested note
                    sample_rate = compute_playback_rate(root_note, params.note)

                sample_state.rate_q16 = sample_rate

            # Generate coefficients from sample playback
            # Note: Sample buffer must be loaded via Voice Engine before playback
            sample_generate_coeffs(sample_state, carrier_512)
        elif osc_mode == OSC_MODE_FM6:
            # -----------------------------------------------------------------
            # MODE FM6: 6-Operator FM Synthesis (WSE-STD-FM-TOPOLOGY-v0)
            # -----------------------------------------------------------------
            # FM6 uses direct sample-rate audio generation.
            # FWPT is an "adapter" to re-enter coefficient domain ONLY when
            # effects processing is needed.
            #
            # Pipeline options:
            # A) NO effects: audio -> gain -> output (bypass coefficients)
            # B) WITH effects: audio -> FWPT -> coeff_ops -> IDWPT -> output
            #
            from .fm6_engine import fm6_generate_block_full

            # Get or create FM6 state
            fm6_state = state.fm6_state

            # Configure FM6 from VoiceFrame parameters if provided
            fm_algorithm_id = getattr(frame, 'fm_algorithm_id', None)
            fm_op_ratios = getattr(frame, 'fm_op_ratios', None)
            fm_op_levels = getattr(frame, 'fm_op_levels', None)
            fm_feedback_depth = getattr(frame, 'fm_feedback_depth', None)

            # Update algorithm if specified
            if fm_algorithm_id is not None and fm_algorithm_id != fm6_state.algorithm_id:
                fm6_state.set_algorithm(fm_algorithm_id)

            # Update operator ratios if specified
            if fm_op_ratios is not None:
                for i, ratio in enumerate(fm_op_ratios[:6]):
                    fm6_state.operators[i].ratio_q16 = ratio

            # Update operator levels if specified
            if fm_op_levels is not None:
                for i, level in enumerate(fm_op_levels[:6]):
                    fm6_state.operators[i].level_q15 = level

            # Update feedback depth if specified
            if fm_feedback_depth is not None:
                from .fm6_state import dx7_feedback_to_q15
                from .fm_algorithms import get_algorithm
                alg = get_algorithm(fm6_state.algorithm_id)
                feedback_q15 = dx7_feedback_to_q15(fm_feedback_depth)
                if alg.feedback_op >= 0:
                    fm6_state.operators[alg.feedback_op].feedback_q15 = feedback_q15

            # Apply matrix envelope targets to FM6 operators
            # env_targets_q15 maps EnvelopeTarget ID -> Q15 value
            # FM_OP1-FM_OP6 = 0-5
            env_targets = getattr(frame, 'env_targets_q15', None)
            if env_targets:
                from .fm6_state import Q15_ONE as FM6_Q15_ONE
                for target_id, value in env_targets.items():
                    if 0 <= target_id <= 5:  # FM_OP1 through FM_OP6
                        fm6_state.operators[target_id].env_value_q15 = value
            else:
                # No envelope targets: set all operators to full output
                from .fm6_state import Q15_ONE as FM6_Q15_ONE
                for op in fm6_state.operators:
                    op.env_value_q15 = FM6_Q15_ONE

            # FM6 generates audio directly at sample rate
            base_phase_inc = params.phase_inc
            fm6_audio = fm6_generate_block_full(fm6_state, base_phase_inc)

            # Check if we have coefficient-domain effects to apply
            # Effects are requested via backbone_instrs (non-empty = has effects)
            has_coeff_effects = len(cfg.backbone_instrs) > 0

            if has_coeff_effects:
                # ---------------------------------------------------------
                # PATH B: FM6 audio -> FWPT -> coeff ops -> IDWPT -> output
                # ---------------------------------------------------------
                # Use FWPT as adapter to re-enter coefficient domain for effects
                from ..wavelet.fast_lifting import fwpt_jit

                # FWPT analysis: 64 audio samples -> 64 wavelet coefficients
                coeffs_64 = fwpt_jit(fm6_audio.tolist(), headroom_bits=1)

                # Replicate across K time slots for CCS processing
                # Layout: coeff_state[leaf * K + k] for each leaf and k-slot
                from .tables import VM_K
                for k in range(VM_K):
                    for leaf in range(64):
                        state.coeff_state[leaf * VM_K + k] = int(np.clip(coeffs_64[leaf], -32768, 32767))

                # Apply coefficient-domain operators
                # Use crossfade-aware dispatch for smooth mask transitions (WSE-STD-070)
                lane_inject(state, cfg)
                for instr in cfg.backbone_instrs:
                    dispatch_coeff_op_crossfade(state, instr)
                lane_tail(state, cfg)

                # IDWPT reconstruction back to audio
                output = lane_ccs_render(state, cfg)
            else:
                # ---------------------------------------------------------
                # PATH A: Direct audio output (no effects, bypass FWPT/IDWPT)
                # ---------------------------------------------------------
                output = fm6_audio.copy()

            # Apply gain interpolation
            gain_curr = cfg.gain_curr_q15
            gain_prev = cfg.gain_prev_q15
            if gain_curr != 32767 or gain_prev != 32767:
                for i in range(BLOCK_SAMPLES):
                    alpha = (i * 32767 + (BLOCK_SAMPLES >> 1)) // BLOCK_SAMPLES
                    gain = gain_prev + RSHIFT_RNE((gain_curr - gain_prev) * alpha, 15)
                    output[i] = RSHIFT_RNE(int(output[i]) * gain, 15)

            # Update state and return
            state.eta_prev_q15 = cfg.eta_curr_q15
            state.gain_prev_q15 = cfg.gain_curr_q15
            return state, output
        else:
            # -----------------------------------------------------------------
            # MODE A: LUT-based (existing path)
            # -----------------------------------------------------------------
            from .coeff_lut_streaming import ccs_generate_coeffs_polyphase
            ccs_generate_coeffs_polyphase(state.ccs_state, carrier_512)

        # PERSISTENT EVOLUTION per WSE-STD-120 section 4.1:
        # Coefficients persist and evolve, carrier provides stabilization

        # Drone mode: reduce base_mix for slow-evolving textures (Phase 4)
        # 5% carrier refresh means operators dominate sound evolution
        if params.drone_mode:
            base_mix = 1638  # ~5% carrier, 95% persistence
        else:
            base_mix = params.base_mix_q15

        # POLYPHASE INJECTION: Each k-slot gets its own carrier value
        # This is the key to eliminating block-boundary discontinuities
        for idx in range(COEFF_LEN):
            carrier_val = int(carrier_512[idx])
            old_val = int(state.coeff_state[idx])

            # Blend: new = old * (1 - base_mix) + carrier * base_mix
            # base_mix = 32767 → pure carrier (full oscillation)
            # base_mix = 16384 → 50/50 blend
            # base_mix = 0 → pure persistence
            inv_mix = 32767 - base_mix
            blended = RSHIFT_RNE(old_val * inv_mix + carrier_val * base_mix, 15)

            state.coeff_state[idx] = np.clip(blended, -32768, 32767)

        # Step 1: LANE_INJECT (events modify carrier-seeded coeff_state)
        lane_inject(state, cfg)

        # Steps 2-5: Backbone lanes (operators modify carrier)
        # Use crossfade-aware dispatch for smooth mask transitions (WSE-STD-070)
        for instr in cfg.backbone_instrs:
            dispatch_coeff_op_crossfade(state, instr)

        # === FM MODULATOR EXTRACTION (Phase 3 - WSE-STD-210 Section 8.3) ===
        # Extract modulator from operator output for NEXT block's FM.
        # This creates z⁻¹ block-delayed feedback loop.
        if osc_mode == OSC_MODE_B and params.fm_source == FM_SOURCE_FEEDBACK:
            from .harmonic_bank import extract_modulator_signal
            state.prev_fm_mod_buffer = extract_modulator_signal(
                state.coeff_state, method=0
            )

        # Step 6: LANE_TAIL (safety constraints)
        lane_tail(state, cfg)

        # Step 7: LANE_CCS_RENDER (all oscillator modes use standard CCS path)
        # FM6 now generates true coefficients via basis frames, so it goes through
        # the same IDWPT polyphase reconstruction as Mode A/B
        output = lane_ccs_render(state, cfg)

        # Apply gain with proper rounding (WSE-STD-001 compliant)
        gain_curr = cfg.gain_curr_q15
        gain_prev = cfg.gain_prev_q15
        if gain_curr != 32767 or gain_prev != 32767:
            # Interpolate gain across block using round-to-nearest
            for i in range(BLOCK_SAMPLES):
                # Round-to-nearest for alpha: (i * 32767 + 32) / 64
                alpha = (i * 32767 + (BLOCK_SAMPLES >> 1)) // BLOCK_SAMPLES
                gain = gain_prev + RSHIFT_RNE((gain_curr - gain_prev) * alpha, 15)
                output[i] = RSHIFT_RNE(int(output[i]) * gain, 15)

        # Advance CCS phase for next block (Mode A only - Mode B advances in generate)
        if osc_mode == OSC_MODE_A:
            state.ccs_state.advance_block()

    else:
        # -----------------------------------------------------------------
        # LEGACY PATH: Old architecture (has combing when wet enabled)
        # -----------------------------------------------------------------

        # Step 1: LANE_CARRIER_RENDER
        if cfg.carrier_role != CarrierRole.OFF:
            carrier_buf = lane_carrier_render(state, cfg)
        else:
            carrier_buf = np.zeros(BLOCK_SAMPLES, dtype=np.int16)

        # Step 2: LANE_INJECT (events)
        lane_inject(state, cfg)

        # Steps 3-6: Backbone lanes (in order)
        # Use crossfade-aware dispatch for smooth mask transitions (WSE-STD-070)
        for instr in cfg.backbone_instrs:
            dispatch_coeff_op_crossfade(state, instr)

        # Step 7: LANE_TAIL
        lane_tail(state, cfg)

        # Step 8: LANE_IDWPT_RENDER
        if cfg.idwpt_enable:
            texture_buf = lane_idwpt_render(state, cfg, carrier_buf)
        else:
            texture_buf = np.zeros(BLOCK_SAMPLES, dtype=np.int16)

        # Step 9: LANE_MIX_GAIN
        output = lane_mix_gain(cfg, carrier_buf, texture_buf)

    # -------------------------------------------------------------------------
    # Update state for next block
    # -------------------------------------------------------------------------
    state.eta_prev_q15 = cfg.eta_curr_q15
    state.gain_prev_q15 = cfg.gain_curr_q15

    return state, output


def vm_step_block_raw(
    state: VMState,
    cfg: DecodedPlan,
) -> Tuple[VMState, np.ndarray]:
    """
    Execute one block with a pre-decoded plan.

    Lower-level API for when you want to bypass decode_frame().

    Args:
        state: Per-voice VM state (modified in place)
        cfg: Pre-decoded execution plan

    Returns:
        (state, output_block)
    """
    # Track blocks
    state.blocks_since_start += 1
    if state.blocks_since_start >= IDWPT_LATENCY_BLOCKS:
        state.wet_ready = True

    if cfg.use_coeff_osc and state.use_coeff_osc:
        # CCS PATH: WSE-STD-121 compliant with TRUE POLYPHASE INJECTION
        from .coeff_lut_streaming import ccs_generate_coeffs_polyphase
        from .tables import COEFF_LEN

        # Generate polyphase carrier: 64 leaves × K k-slots = 512 values (K=8)
        carrier_512 = np.zeros(COEFF_LEN, dtype=np.int16)
        ccs_generate_coeffs_polyphase(state.ccs_state, carrier_512)

        # Inject with base_mix blending (polyphase: each k-slot has own value)
        params = cfg.coeff_osc_params
        base_mix = params.base_mix_q15

        for idx in range(COEFF_LEN):
            carrier_val = int(carrier_512[idx])
            old_val = int(state.coeff_state[idx])
            inv_mix = 32767 - base_mix
            blended = RSHIFT_RNE(old_val * inv_mix + carrier_val * base_mix, 15)
            state.coeff_state[idx] = np.clip(blended, -32768, 32767)

        lane_inject(state, cfg)

        # Use crossfade-aware dispatch for smooth mask transitions (WSE-STD-070)
        for instr in cfg.backbone_instrs:
            dispatch_coeff_op_crossfade(state, instr)

        lane_tail(state, cfg)

        output = lane_ccs_render(state, cfg)

        # Apply gain with proper rounding (WSE-STD-001 compliant)
        gain_curr = cfg.gain_curr_q15
        gain_prev = cfg.gain_prev_q15
        if gain_curr != 32767 or gain_prev != 32767:
            for i in range(BLOCK_SAMPLES):
                alpha = (i * 32767 + (BLOCK_SAMPLES >> 1)) // BLOCK_SAMPLES
                gain = gain_prev + RSHIFT_RNE((gain_curr - gain_prev) * alpha, 15)
                output[i] = RSHIFT_RNE(int(output[i]) * gain, 15)

        # Advance CCS phase for next block
        state.ccs_state.advance_block()
    else:
        # LEGACY PATH
        if cfg.carrier_role != CarrierRole.OFF:
            carrier_buf = lane_carrier_render(state, cfg)
        else:
            carrier_buf = np.zeros(BLOCK_SAMPLES, dtype=np.int16)

        lane_inject(state, cfg)

        # Use crossfade-aware dispatch for smooth mask transitions (WSE-STD-070)
        for instr in cfg.backbone_instrs:
            dispatch_coeff_op_crossfade(state, instr)

        lane_tail(state, cfg)

        if cfg.idwpt_enable and state.wet_ready:
            texture_buf = lane_idwpt_render(state, cfg, carrier_buf)
        else:
            texture_buf = np.zeros(BLOCK_SAMPLES, dtype=np.int16)

        output = lane_mix_gain(cfg, carrier_buf, texture_buf)

    # Update state
    state.eta_prev_q15 = cfg.eta_curr_q15
    state.gain_prev_q15 = cfg.gain_curr_q15

    return state, output


# =============================================================================
# MULTI-VOICE EXECUTOR
# =============================================================================

class VMExecutor:
    """
    Multi-voice VM executor.

    Manages state for multiple voices and provides batch execution.
    """

    def __init__(self, max_voices: int = 128, tables: Optional[Tables] = None):
        """
        Initialize VM executor.

        Args:
            max_voices: Maximum number of simultaneous voices
            tables: Tables instance (default: DEFAULT_TABLES)
        """
        self.max_voices = max_voices
        self.tables = tables or DEFAULT_TABLES
        self.states: dict[int, VMState] = {}

    def get_or_create_state(self, voice_id: int) -> VMState:
        """Get existing state or create new one for a voice."""
        if voice_id not in self.states:
            self.states[voice_id] = VMState.create(voice_id)
        return self.states[voice_id]

    def step_voice(
        self,
        voice_id: int,
        frame: 'VoiceFrame',
    ) -> np.ndarray:
        """
        Execute one block for a voice.

        Args:
            voice_id: Voice slot ID
            frame: VoiceFrame microcode

        Returns:
            output_block: int16 array of BLOCK_SAMPLES
        """
        state = self.get_or_create_state(voice_id)
        state, output = vm_step_block(state, frame, self.tables)
        self.states[voice_id] = state
        return output

    def release_voice(self, voice_id: int) -> None:
        """Release a voice slot."""
        if voice_id in self.states:
            del self.states[voice_id]

    def reset_voice(self, voice_id: int) -> None:
        """Reset a voice to initial state."""
        if voice_id in self.states:
            self.states[voice_id] = VMState.create(voice_id)

    def active_voices(self) -> list[int]:
        """Get list of active voice IDs."""
        return [vid for vid, state in self.states.items() if state.is_active()]

    def snapshot_all(self) -> dict[int, dict]:
        """Snapshot all voice states for deterministic replay."""
        return {vid: state.snapshot() for vid, state in self.states.items()}

    def restore_all(self, snapshots: dict[int, dict]) -> None:
        """Restore all voice states from snapshots."""
        self.states = {vid: VMState.restore(snap) for vid, snap in snapshots.items()}


__all__ = [
    'vm_step_block',
    'vm_step_block_raw',
    'VMExecutor',
    'IDWPT_LATENCY_BLOCKS',
    '_process_cmmm_events',
]
