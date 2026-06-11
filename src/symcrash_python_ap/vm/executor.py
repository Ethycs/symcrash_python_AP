"""
VM Executor (WSE-STD-VM-HYBRID v0)
==================================

Main execution entry point for the hybrid VM.
Executes one block for one voice with fixed lane schedule.
"""

from typing import Tuple, Optional, TYPE_CHECKING
import numpy as np

from .state import (
    VMState, OSC_MODE_A, OSC_MODE_B, OSC_MODE_SAMPLE, OSC_MODE_FM6, OSC_MODE_WAVETABLE,
    OSC_MODE_GRANULAR, PAYLOAD_TYPE_NODEMASK_SET, PAYLOAD_TYPE_TAILMODE_SET,
)
from .decode import DecodedPlan, decode_frame, FM_SOURCE_FEEDBACK
from .tables import Tables, DEFAULT_TABLES, BLOCK_SAMPLES, VM_K
from .lanes import (
    lane_inject,
    dispatch_coeff_op,
    dispatch_coeff_op_crossfade,  # WSE-STD-070: mask-aware dispatch
    lane_tail,
    lane_ccs_render,
)
from ..fixed import RSHIFT_RNE
from ..primitives import Q15_ONE, Q15_HALF, Q15_NEG_ONE, DRONE_MODE_CARRIER_MIX

# JIT hot-path kernels (bit-exact, switchable; see vm/hotpath_jit.py)
from . import hotpath_jit as _hot

if TYPE_CHECKING:
    from ..frames import VoiceFrame


def _apply_block_gain(output: np.ndarray, gain_prev: int, gain_curr: int) -> None:
    """
    Apply interpolated per-sample gain to one block (WSE-STD-001 compliant).

    Pure-Python reference with a bit-exact JIT fast path. The caller skips
    the call entirely when both gains are Q15_ONE (same as the reference).
    """
    if _hot.jit_enabled():
        _hot.gain_interp(output, gain_prev, gain_curr)
        return
    # Reference implementation (conformance path)
    for i in range(BLOCK_SAMPLES):
        # Round-to-nearest for alpha: (i * Q15_ONE + 32) / 64
        alpha = (i * Q15_ONE + (BLOCK_SAMPLES >> 1)) // BLOCK_SAMPLES
        gain = gain_prev + RSHIFT_RNE((gain_curr - gain_prev) * alpha, 15)
        output[i] = RSHIFT_RNE(int(output[i]) * gain, 15)


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
    - 0x30 (NODEMASK_SET): Changes node mask (immediate)
    - 0x31 (TAILMODE_SET): Changes tail mode with traction

    Also advances CMMM enrichment state:
    - Symbol mode dwell counter
    - Timbral mode ramp
    - CMMMemory (tension, fatigue)

    Args:
        state: Per-voice VM state (modified in place)
        frame: VoiceFrame containing CMMM event fields
    """
    latch = state.executor_latch

    # Step 1: Process node mask event (payload_type 0x30)
    node_mask = getattr(frame, 'cmmm_node_mask', None)
    if node_mask is not None:
        ramp_blocks = getattr(frame, 'cmmm_mask_ramp_blocks', 5)
        latch.apply_nodemask_event(node_mask, ramp_blocks)

    # Step 3: Process tail mode event (payload_type 0x31)
    tail_mode = getattr(frame, 'cmmm_tail_mode', None)
    if tail_mode is not None:
        traction = getattr(frame, 'cmmm_traction_q15', 32767)
        latch.apply_tailmode_event(tail_mode, traction)

    # -------------------------------------------------------------------------
    # CMMM Enrichment: Symbol mode and memory (Gaps G1, G2)
    # -------------------------------------------------------------------------
    # Step 4: Advance symbol mode dwell counter
    latch.step_symbol_mode()

    # Step 5: Advance timbral mode ramp
    latch.step_timbral_mode()

    # Step 6: Update CMMM memory
    # Compute intent energy from frame (average of 8 intent lanes)
    intent_vector = getattr(frame, 'intent_vector', None)
    if intent_vector is not None and len(intent_vector) >= 8:
        intent_energy = sum(intent_vector) >> 3  # Average of 8 lanes
    else:
        intent_energy = Q15_HALF  # Default mid-level intent

    is_transition = latch.is_timbral_ramping()
    state.cmmm_memory.step(intent_energy, is_transition)


# =============================================================================
# NOISE EXCITER (Packet E4 - Acoustic Attack Transients)
# =============================================================================

def _mix_noise_exciter(state: VMState, cfg: DecodedPlan,
                       output: np.ndarray) -> None:
    """
    Sum the deterministic noise exciter into the voice output.

    Mix point: post-IDWPT, pre-gain - the exciter is part of the voice
    signal, so the voice gain envelope (velocity, de-click ramps) applies
    uniformly while the exciter keeps its own attack/decay shape.

    Bit-exact no-op: when the frame authors no noise (noise_level_q15 == 0,
    the default) this returns before touching the output buffer, and after
    the exciter envelope reaches zero the path is skipped entirely.
    """
    if cfg.noise_level_q15 <= 0:
        return
    from .noise_exciter import render_noise_block, noise_seed_base

    # blocks_since_start was already incremented this block; the exciter
    # envelope is anchored at note-on (block_index 0).
    block_index = state.blocks_since_start - 1
    noise = render_noise_block(
        state.noise_state,
        noise_seed_base(cfg.noise_seed_u32, state.voice_id),
        block_index,
        cfg.noise_color,
        cfg.noise_level_q15,
        cfg.noise_attack_blocks,
        cfg.noise_decay_blocks,
    )
    if noise is None:
        return
    mixed = output.astype(np.int32) + noise
    np.clip(mixed, Q15_NEG_ONE, Q15_ONE, out=mixed)
    output[:] = mixed.astype(np.int16)


# =============================================================================
# MODE B HELPER FUNCTIONS
# =============================================================================

def _setup_mode_b_harmonic_weights(mb_state, params) -> None:
    """
    Set up Mode B harmonic weights based on params.

    Priority order: OEIS sequence > custom_harmonics > waveform_type

    Args:
        mb_state: ModeBState instance
        params: Oscillator params with weight configuration
    """
    oeis_id = getattr(params, 'oeis_sequence_id', -1)
    if oeis_id >= 0:
        rolloff = getattr(params, 'oeis_rolloff_db', -3.0)
        mb_state.weights.set_oeis(oeis_id, mb_state.H_active, rolloff)
    elif params.custom_harmonics:
        spread = getattr(params, 'phase_spread_q15', 0)
        mb_state.weights.set_custom(params.custom_harmonics, spread)
        mb_state.H_active = min(len(params.custom_harmonics), mb_state.H_bandlimit)
        # Latch for per-block spectral motion detection
        mb_state.last_custom_harmonics = (list(params.custom_harmonics), spread)
    else:
        waveform_type = getattr(params, 'waveform_type', 0)
        mb_state.weights.set_from_waveform_type(
            waveform_type, mb_state.H_active, params.pwm_width_q15
        )


# =============================================================================
# FM6 HELPER FUNCTIONS
# =============================================================================

def _update_fm6_operators(fm6_state, fm_op_ratios, fm_op_levels, env_targets) -> None:
    """
    Update FM6 operators with ratios, levels, and envelope targets.

    Args:
        fm6_state: FM6State or FM6MorphState instance
        fm_op_ratios: Optional list of 6 operator ratios (Q16)
        fm_op_levels: Optional list of 6 operator levels (Q15)
        env_targets: Optional dict of envelope target values (Q15)
    """
    from .fm6_state import Q15_ONE as FM6_Q15_ONE

    # Update operator ratios if specified
    if fm_op_ratios is not None:
        for i, ratio in enumerate(fm_op_ratios[:6]):
            fm6_state.operators[i].ratio_q16 = ratio

    # Update operator levels if specified
    if fm_op_levels is not None:
        for i, level in enumerate(fm_op_levels[:6]):
            fm6_state.operators[i].level_q15 = level

    # Apply envelope targets to operators
    if env_targets:
        for target_id, value in env_targets.items():
            if 0 <= target_id <= 5:
                fm6_state.operators[target_id].env_value_q15 = value
    else:
        # Default: full envelope value
        for op in fm6_state.operators:
            op.env_value_q15 = FM6_Q15_ONE


# =============================================================================
# DYNAMIC OPERATOR SELECTION (CMMM Enrichment Gap G4)
# =============================================================================

def _get_dynamic_backbone(state: VMState, frame: 'VoiceFrame', cfg) -> list:
    """
    Get backbone instructions with optional dynamic operator selection.

    This implements CMMM Enrichment Gap G4 (Timbral Sub-states):
    - If intent vector is present and dynamic_timbral_mode enabled,
      computes timbral mode from intent and looks up operators in REGIME_SYMBOL_OPS
    - Falls back to static backbone_instrs from cfg

    Args:
        state: Per-voice VM state
        frame: VoiceFrame for this block
        cfg: Decoded plan with static backbone

    Returns:
        List of BackboneEntry instructions to execute
    """
    from .decode import BackboneEntry
    from ..primitives import (
        get_operators_for_state,
        compute_timbral_mode_from_intent,
        REGIME_SYMBOL_OPS,
    )

    # Check if dynamic timbral mode is enabled (opt-in via frame flag)
    dynamic_enabled = getattr(frame, 'dynamic_timbral_mode', False)

    if not dynamic_enabled:
        # Use static backbone from preset
        return cfg.backbone_instrs

    # Get regime from frame
    regime_id = getattr(frame, 'regime_id', 2)  # Default R2_STABLE

    # Get intent vector for timbral mode computation
    intent_vector = getattr(frame, 'intent_vector', None)
    if intent_vector is not None and len(intent_vector) >= 4:
        # Compute timbral mode from intent
        timbral_mode = compute_timbral_mode_from_intent(list(intent_vector))

        # Update executor latch timbral mode (with ramp)
        if timbral_mode != state.executor_latch.timbral_mode:
            state.executor_latch.set_timbral_mode(timbral_mode, ramp_blocks=4)
    else:
        # Use current latched timbral mode
        timbral_mode = state.executor_latch.timbral_mode

    # Look up dynamic operators
    key = (regime_id, timbral_mode)
    if key in REGIME_SYMBOL_OPS:
        op_ids = REGIME_SYMBOL_OPS[key]

        # Get family strengths from frame if available
        families = getattr(frame, 'families', {})

        # Build backbone entries from operator IDs
        backbone = []
        for op_id in op_ids:
            # Get strength from families dict (keyed by op_id or op_name)
            strength = families.get(op_id, Q15_HALF)  # Default 50% strength
            if isinstance(strength, float):
                strength = int(strength * 32767)
            backbone.append(BackboneEntry(op_id=op_id, strength_q15=strength))

        return backbone

    # Fallback: use static backbone from preset
    return cfg.backbone_instrs


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
    # TRUE CCS path per spec section 2.1:
    #   Phase → CoeffTable → Coefficients → Operators → IDWPT → Audio
    #
    # This eliminates FWPT from the voice path and ensures operators modify
    # the carrier coefficients before IDWPT reconstruction.
    #
    # WSE-STD-210: Mode A (LUT) vs Mode B (Rotator/Harmonic Bank)
    # - Mode A: Phase → LUT lookup → Coefficients
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
            # Set harmonic weights using helper
            _setup_mode_b_harmonic_weights(mb_state, params)
        elif pitch_changed or mb_state.note != params.note:
            # Pitch changed: use legato_mode to decide how to handle.
            # A cents-only change (same note) is vibrato or pitch bend, not a
            # retrigger — always take the phase-continuous path, otherwise
            # per-block LFO vibrato hard-resets phase every block (a ~750 Hz
            # discontinuity comb that swamps the spectrum with inharmonics).
            cents_only = mb_state.note == params.note
            if params.legato_mode or cents_only:
                # Legato: preserve phase for smooth transition
                mb_state.set_pitch_legato(note=params.note, phase_inc=params.phase_inc)
                mb_state.H_active = min(params.H_request, mb_state.H_bandlimit)
            else:
                # Non-legato: full reset (may cause click, but musically intentional)
                mb_state.reset(note=params.note, phase_inc=params.phase_inc)
                mb_state.pwm_width_q15 = params.pwm_width_q15
                mb_state.pwm_enabled = params.pwm_enabled
                mb_state.H_active = min(params.H_request, mb_state.H_bandlimit)
                # Set harmonic weights using helper
                _setup_mode_b_harmonic_weights(mb_state, params)

        # Per-block spectral motion: re-latch custom harmonics when the
        # frame's partials (or phase spread) change mid-note. Weights-only —
        # the phase accumulator is untouched, so this is phase-continuous.
        if params.custom_harmonics:
            spread = getattr(params, 'phase_spread_q15', 0)
            latch = (params.custom_harmonics, spread)
            prev = getattr(mb_state, 'last_custom_harmonics', None)
            if prev is None or list(prev[0]) != list(latch[0]) or prev[1] != latch[1]:
                mb_state.weights.set_custom(params.custom_harmonics, spread)
                mb_state.H_active = min(len(params.custom_harmonics), mb_state.H_bandlimit)
                mb_state.last_custom_harmonics = (list(params.custom_harmonics), spread)

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
            mode_b_generate_coeffs, mode_b_generate_coeffs_vectorized,
            mode_b_generate_coeffs_matrix, get_basis_for_note
        )
        from .waveform_matrix import WaveformType

        # Determine which synthesis method to use:
        # - Matrix method: PL waveforms (SQUARE, SAW, TRIANGLE) without FM/PWM
        # - Harmonic bank: SINE, custom/OEIS harmonics, PWM/FM-modulated signals
        #
        # The matrix LUT is keyed only by waveform_type and never reads
        # mb_state.weights, so custom harmonics and OEIS selections must
        # force the harmonic-bank path or they are silently ignored.
        waveform_type = getattr(params, 'waveform_type', WaveformType.SINE)
        use_matrix_method = (
            waveform_type in (WaveformType.SQUARE, WaveformType.SAW,
                              WaveformType.SAW_DOWN, WaveformType.TRIANGLE)
            and not params.pwm_enabled
            and params.fm_depth_q15 == 0
            and not params.custom_harmonics
            and getattr(params, 'oeis_sequence_id', -1) < 0
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
            mode_b_generate_coeffs_vectorized(state.mode_b_state, basis, carrier_512)

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
        # FM6 generates audio at sample rate, then uses FWPT as adapter
        # to re-enter the unified coefficient domain pipeline.
        #
        # Pipeline: audio -> FWPT -> coeff_ops -> IDWPT -> output
        #
        # This unified path ensures FM6 participates in CMMM control,
        # matrix routing, and tail mode enforcement like Mode A/B.

        # Check for topology morphing (L1Lift.fm6_morph)
        fm_topo_a_id = getattr(frame, 'fm_topo_a_id', None)
        fm_topo_b_id = getattr(frame, 'fm_topo_b_id', None)
        fm_topo_blend_q15 = getattr(frame, 'fm_topo_blend_q15', 0)
        use_morphing = (
            fm_topo_a_id is not None and
            fm_topo_b_id is not None and
            (fm_topo_a_id != fm_topo_b_id or fm_topo_blend_q15 > 0)
        )

        # Configure FM6 from VoiceFrame parameters if provided
        fm_algorithm_id = getattr(frame, 'fm_algorithm_id', None)
        fm_op_ratios = getattr(frame, 'fm_op_ratios', None)
        fm_op_levels = getattr(frame, 'fm_op_levels', None)
        fm_feedback_depth = getattr(frame, 'fm_feedback_depth', None)

        if use_morphing:
            # -----------------------------------------------------------------
            # FM6 MORPHING PATH: Blend between two topologies
            # -----------------------------------------------------------------
            from .fm6_morph_engine import fm6_generate_block_morphed

            # Get FM6 morph state (lazy-initialized via property)
            fm6_morph_state = state.fm6_morph_state

            # Set morphing parameters
            fm6_morph_state.set_morph(fm_topo_a_id, fm_topo_b_id, fm_topo_blend_q15)

            # Update operators using helper
            env_targets = getattr(frame, 'env_targets_q15', None)
            _update_fm6_operators(fm6_morph_state, fm_op_ratios, fm_op_levels, env_targets)

            # FM6 morphed audio generation
            base_phase_inc = params.phase_inc
            fm6_audio = fm6_generate_block_morphed(fm6_morph_state, base_phase_inc)
        else:
            # -----------------------------------------------------------------
            # FM6 DIRECT PATH: Single topology (no morphing)
            # -----------------------------------------------------------------
            from .fm6_engine import (
                fm6_generate_block_full,
                fm6_gateway_level_overrides,
            )

            # Get or create FM6 state
            fm6_state = state.fm6_state

            # Update algorithm if specified
            if fm_algorithm_id is not None and fm_algorithm_id != fm6_state.algorithm_id:
                fm6_state.set_algorithm(fm_algorithm_id)

            # Update operators using helper
            env_targets = getattr(frame, 'env_targets_q15', None)
            _update_fm6_operators(fm6_state, fm_op_ratios, fm_op_levels, env_targets)

            # Update feedback depth if specified (algorithm-specific)
            if fm_feedback_depth is not None:
                from .fm6_state import dx7_feedback_to_q15
                from .fm_algorithms import get_algorithm
                alg = get_algorithm(fm6_state.algorithm_id)
                feedback_q15 = dx7_feedback_to_q15(fm_feedback_depth)
                if alg.feedback_op >= 0:
                    fm6_state.operators[alg.feedback_op].feedback_q15 = feedback_q15

            # Configure post-filter if specified
            fm_post_filter_enabled = getattr(frame, 'fm_post_filter_enabled', None)
            fm_post_filter_cutoff = getattr(frame, 'fm_post_filter_cutoff_hz', None)
            if fm_post_filter_enabled is not None:
                cutoff = fm_post_filter_cutoff if fm_post_filter_cutoff else 8000.0
                fm6_state.set_post_filter(fm_post_filter_enabled, cutoff)

            # FM6 generates audio directly at sample rate.
            # Gateway headroom (Packet E2): when summed CARRIER levels
            # exceed 0.5 the FWPT below saturates (~23% H3 at sum 1.0), so
            # carrier levels are auto-normalized to sum 0.5. The IDWPT
            # render stage's built-in 2x makeup gain (headroom_bits 1->2)
            # restores full-scale loudness after the nonlinear stage.
            # No-op (None) when carriers already sum <= 0.5.
            base_phase_inc = params.phase_inc
            level_overrides = fm6_gateway_level_overrides(fm6_state)
            fm6_audio = fm6_generate_block_full(fm6_state, base_phase_inc,
                                                level_overrides)

        # -----------------------------------------------------------------
        # UNIFIED PATH: FM6 audio -> FWPT -> coeff ops -> IDWPT -> output
        # -----------------------------------------------------------------
        # Always go through coefficient domain for unified pipeline.
        # This ensures FM6 can use CMMM operators, matrix routing, and
        # tail mode enforcement consistently with Mode A/B.
        from ..wavelet.fast_lifting import fwpt_jit

        # FWPT analysis: 64 audio samples -> 64 wavelet coefficients
        if _hot.jit_enabled():
            # Bit-exact array-native port (no list round-trips)
            coeffs_64 = np.empty(64, dtype=np.int64)
            _hot.fwpt_64(fm6_audio, 1, coeffs_64)
        else:
            coeffs_64 = fwpt_jit(fm6_audio.tolist(), headroom_bits=1)

        # Replicate across K time slots for CCS processing
        # Layout: coeff_state[leaf * K + k] for each leaf and k-slot
        # np.repeat replicates each leaf K times in leaf-major order, which is
        # bit-exact with the per-element loop it replaces.
        from .tables import VM_K
        clipped_64 = np.clip(
            np.asarray(coeffs_64, dtype=np.int64), Q15_NEG_ONE, Q15_ONE
        ).astype(np.int16)
        state.coeff_state[:64 * VM_K] = np.repeat(clipped_64[:64], VM_K)

        # Apply coefficient-domain operators (may be empty, but lane_inject/tail still apply)
        # Use mask-aware dispatch (WSE-STD-070)
        lane_inject(state, cfg)
        for instr in cfg.backbone_instrs:
            dispatch_coeff_op_crossfade(state, instr)
        lane_tail(state, cfg)

        # IDWPT reconstruction back to audio
        output = lane_ccs_render(state, cfg)

        # Noise exciter (Packet E4): pre-gain broadband transient
        _mix_noise_exciter(state, cfg, output)

        # Apply gain interpolation
        gain_curr = cfg.gain_curr_q15
        gain_prev = cfg.gain_prev_q15
        if gain_curr != Q15_ONE or gain_prev != Q15_ONE:
            _apply_block_gain(output, gain_prev, gain_curr)

        # Update state and return
        state.eta_prev_q15 = cfg.eta_curr_q15
        state.gain_prev_q15 = cfg.gain_curr_q15
        return state, output
    elif osc_mode == OSC_MODE_WAVETABLE:
        # -----------------------------------------------------------------
        # MODE WAVETABLE: 2D Wavetable Synthesis (Phase 4 - WSE-STD-210)
        # -----------------------------------------------------------------
        # Wavetable generates audio via atlas interpolation, then uses FWPT
        # to re-enter the unified coefficient domain pipeline.
        #
        # Pipeline: atlas interpolation -> audio -> FWPT -> coeff_ops -> IDWPT
        from .wavetable_engine import wavetable_generate_coeffs, create_default_bank

        # Get wavetable state
        wt_state = state.wavetable_state

        # Configure wavetable from VoiceFrame parameters if provided
        wt_atlas_name = getattr(frame, 'wt_atlas_name', None)
        wt_position = getattr(frame, 'wt_position', None)
        wt_frame = getattr(frame, 'wt_frame', None)
        wt_frame_rate = getattr(frame, 'wt_frame_rate_hz', None)

        # Initialize atlas if needed (lazy bank creation)
        if wt_state.atlas is None:
            bank = create_default_bank()
            if wt_atlas_name:
                atlas = bank.get_by_name(wt_atlas_name)
            else:
                atlas = bank.get(0)  # Default to BasicWaves
            wt_state.atlas = atlas
            wt_state.reset()

        # Update pitch from note
        if pitch_changed or wt_state.phase_inc_q32 == 0:
            wt_state.set_pitch_note(params.note)

        # Update position and frame if specified
        if wt_position is not None:
            wt_state.set_position(wt_position)
        if wt_frame is not None:
            wt_state.set_frame(wt_frame)
        if wt_frame_rate is not None:
            wt_state.set_frame_rate(wt_frame_rate)

        # Ensure playing
        if not wt_state.is_playing:
            wt_state.is_playing = True

        # Generate coefficients from wavetable
        wavetable_generate_coeffs(wt_state, carrier_512)

        # Inject into coeff_state for unified pipeline
        for idx in range(COEFF_LEN):
            state.coeff_state[idx] = int(carrier_512[idx])

        # Apply coefficient-domain operators (unified path)
        lane_inject(state, cfg)
        for instr in cfg.backbone_instrs:
            dispatch_coeff_op_crossfade(state, instr)
        lane_tail(state, cfg)

        # IDWPT reconstruction back to audio
        output = lane_ccs_render(state, cfg)

        # Noise exciter (Packet E4): pre-gain broadband transient
        _mix_noise_exciter(state, cfg, output)

        # Apply gain interpolation
        gain_curr = cfg.gain_curr_q15
        gain_prev = cfg.gain_prev_q15
        if gain_curr != Q15_ONE or gain_prev != Q15_ONE:
            _apply_block_gain(output, gain_prev, gain_curr)

        # Update state and return
        state.eta_prev_q15 = cfg.eta_curr_q15
        state.gain_prev_q15 = cfg.gain_curr_q15
        return state, output
    elif osc_mode == OSC_MODE_GRANULAR:
        # -----------------------------------------------------------------
        # MODE GRANULAR: Coefficient-Domain Granular Synthesis (Phase 5)
        # -----------------------------------------------------------------
        # Granular operates entirely in coefficient domain:
        # - Pre-computed coefficient buffers (FWPT done once at load time)
        # - Up to 8 simultaneous grains per voice
        # - Grain scheduling, spray, pitch shifting
        #
        # Pipeline: grain mixing (coeff) -> coeff_ops -> IDWPT
        from .granular_engine import granular_generate_coeffs, granular_set_parameters

        # Get granular state
        gr_state = state.granular_state

        # Configure granular parameters from VoiceFrame if provided
        gr_density = getattr(frame, 'granular_density', None)
        gr_grain_size_ms = getattr(frame, 'granular_grain_size_ms', None)
        gr_spray = getattr(frame, 'granular_spray', None)
        gr_pitch_ratio = getattr(frame, 'granular_pitch_ratio', None)
        gr_scan_rate = getattr(frame, 'granular_scan_rate', None)

        # Update parameters if any are specified
        if any(p is not None for p in [gr_density, gr_grain_size_ms, gr_spray, gr_pitch_ratio, gr_scan_rate]):
            granular_set_parameters(
                gr_state,
                density=gr_density if gr_density is not None else 0.3,
                grain_size_ms=gr_grain_size_ms if gr_grain_size_ms is not None else 80.0,
                spray=gr_spray if gr_spray is not None else 0.2,
                pitch_ratio=gr_pitch_ratio if gr_pitch_ratio is not None else 1.0,
                scan_rate=gr_scan_rate if gr_scan_rate is not None else 1.0,
            )

        # Generate coefficients from granular synthesis
        # Note: Source buffer must be loaded separately via gr_state.set_source()
        granular_generate_coeffs(gr_state, carrier_512)

        # Inject into coeff_state for unified pipeline
        for idx in range(COEFF_LEN):
            state.coeff_state[idx] = int(carrier_512[idx])

        # Apply coefficient-domain operators (unified path)
        lane_inject(state, cfg)
        for instr in cfg.backbone_instrs:
            dispatch_coeff_op_crossfade(state, instr)
        lane_tail(state, cfg)

        # IDWPT reconstruction back to audio
        output = lane_ccs_render(state, cfg)

        # Noise exciter (Packet E4): pre-gain broadband transient
        _mix_noise_exciter(state, cfg, output)

        # Apply gain interpolation
        gain_curr = cfg.gain_curr_q15
        gain_prev = cfg.gain_prev_q15
        if gain_curr != Q15_ONE or gain_prev != Q15_ONE:
            _apply_block_gain(output, gain_prev, gain_curr)

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
        base_mix = DRONE_MODE_CARRIER_MIX  # ~5% carrier, 95% persistence
    else:
        base_mix = params.base_mix_q15

    # POLYPHASE INJECTION: Each k-slot gets its own carrier value
    # This is the key to eliminating block-boundary discontinuities
    if _hot.jit_enabled():
        _hot.inject_blend(state.coeff_state, carrier_512, base_mix)
    else:
        # Reference implementation (conformance path)
        for idx in range(COEFF_LEN):
            carrier_val = int(carrier_512[idx])
            old_val = int(state.coeff_state[idx])

            # Blend: new = old * (1 - base_mix) + carrier * base_mix
            # base_mix = Q15_ONE → pure carrier (full oscillation)
            # base_mix = Q15_HALF → 50/50 blend
            # base_mix = 0 → pure persistence
            inv_mix = Q15_ONE - base_mix
            blended = RSHIFT_RNE(old_val * inv_mix + carrier_val * base_mix, 15)

            state.coeff_state[idx] = np.clip(blended, Q15_NEG_ONE, Q15_ONE)

    # Step 1: LANE_INJECT (events modify carrier-seeded coeff_state)
    lane_inject(state, cfg)

    # -------------------------------------------------------------------------
    # CMMM Enrichment: Capture pre-operator state for evolution rate scaling (Gap G1)
    # -------------------------------------------------------------------------
    evolution_rate = state.executor_latch.get_evolution_rate_q15()
    if evolution_rate < Q15_ONE:
        # Capture state before operators for scaling
        prev_coeffs = state.coeff_state.copy()

    # -------------------------------------------------------------------------
    # CMMM Enrichment: Dynamic Operator Selection (Gap G4 - Timbral Sub-states)
    # -------------------------------------------------------------------------
    # Get backbone instructions - either from dynamic REGIME_SYMBOL_OPS or static backbone
    backbone_instrs = _get_dynamic_backbone(state, frame, cfg)

    # Steps 2-5: Backbone lanes (operators modify carrier)
    # Use mask-aware dispatch (WSE-STD-070)
    for instr in backbone_instrs:
        dispatch_coeff_op_crossfade(state, instr)

    # -------------------------------------------------------------------------
    # CMMM Enrichment: Apply evolution rate scaling (Gap G1)
    # -------------------------------------------------------------------------
    # Scale coefficient changes by evolution_rate:
    # - EVOLVE (32767): Full evolution (100%)
    # - HOLD (1024): Minimal evolution (~3%)
    # - FREEZE (0): No evolution (0%)
    if evolution_rate < Q15_ONE:
        if _hot.jit_enabled():
            _hot.evolution_scale(state.coeff_state, prev_coeffs, evolution_rate)
        else:
            # Reference implementation (conformance path)
            for idx in range(COEFF_LEN):
                new_val = int(state.coeff_state[idx])
                old_val = int(prev_coeffs[idx])
                delta = new_val - old_val
                # Scale delta by evolution rate
                scaled_delta = RSHIFT_RNE(delta * evolution_rate, 15)
                state.coeff_state[idx] = np.clip(old_val + scaled_delta, Q15_NEG_ONE, Q15_ONE)

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

    # Noise exciter (Packet E4): pre-gain broadband transient
    _mix_noise_exciter(state, cfg, output)

    # Apply gain with proper rounding (WSE-STD-001 compliant)
    gain_curr = cfg.gain_curr_q15
    gain_prev = cfg.gain_prev_q15
    if gain_curr != Q15_ONE or gain_prev != Q15_ONE:
        # Interpolate gain across block using round-to-nearest
        _apply_block_gain(output, gain_prev, gain_curr)

    # Advance CCS phase for next block (Mode A only - Mode B advances in generate)
    if osc_mode == OSC_MODE_A:
        state.ccs_state.advance_block()

    # -------------------------------------------------------------------------
    # Update state for next block
    # -------------------------------------------------------------------------
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
    'VMExecutor',
    'IDWPT_LATENCY_BLOCKS',
    '_process_cmmm_events',
]
