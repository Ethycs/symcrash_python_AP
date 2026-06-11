"""
VM Frame Decoder (WSE-STD-VM-HYBRID v0)
=======================================

Decodes VoiceFrame into DecodedPlan for VM execution.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, TYPE_CHECKING

from .tables import (
    Tables, DEFAULT_TABLES, TailMode,
    get_tail_ops, get_pattern_for_k, popcount32,
    BLOCK_SAMPLES,
)

if TYPE_CHECKING:
    from ..frames import VoiceFrame


# =============================================================================
# CARRIER ROLE
# =============================================================================

class CarrierRole(IntEnum):
    """Carrier oscillator role in the mix."""
    OFF = 0           # No carrier (IDWPT only)
    EXCITER_ONLY = 1  # Carrier used only to excite IDWPT, not mixed
    AUDIBLE = 2       # Carrier mixed with IDWPT output


# =============================================================================
# BACKBONE INSTRUCTION
# =============================================================================

@dataclass
class BackboneEntry:
    """A single backbone operator instruction."""
    op_id: int
    strength_q15: int = 32767  # Full strength by default
    param0_q15: int = 0
    param1_q15: int = 0


# =============================================================================
# EVENT INSTRUCTION
# =============================================================================

@dataclass
class EventEntry:
    """A single event lane instruction."""
    lane: int               # Event lane (0-7)
    op_id: int              # Which inject op to use
    strength_q15: int = 32767
    payload_mode: int = 0   # Payload interpretation mode
    payload64: int = 0      # 64-bit payload


# =============================================================================
# CARRIER PARAMS
# =============================================================================

@dataclass
class CarrierParams:
    """Carrier oscillator parameters for a block."""
    f0_hz: float = 440.0
    phase_inc: int = 0       # Computed from note/cents via LUT
    note: int = 60           # MIDI note number
    cents: int = 0           # Fine tuning in cents
    harmonic_amps_q15: List[int] = field(default_factory=lambda: [32767] + [0] * 15)
    model_id: int = 0        # Oscillator model (0 = additive sine)


@dataclass
class CoeffOscParams:
    """
    Coefficient Oscillator parameters for CCS (WSE-STD-ARCH-CCS v0).

    These parameters control coefficient synthesis, which replaces
    the carrier→FWPT→mask→IDWPT path to eliminate combing artifacts.

    Extended for WSE-STD-210 Mode B support.
    """
    table_id: int = 0         # Which coefficient table (0=sine, 1=saw, etc.)
    base_mix_q15: int = 32767  # Blend factor: 0=pure CMMM state, 32767=pure osc
    phase_inc: int = 0        # Base phase increment (from MIDI note)
    note: int = 60            # MIDI note number
    cents: int = 0            # Fine tuning in cents

    # WSE-STD-210 Section 10.2 - Mode B parameters
    osc_mode: int = 0             # 0=Mode A (LUT), 1=Mode B (Rotator)
    H_request: int = 16           # Requested harmonic count (bandlimited at runtime)
    waveform_type: int = 0        # 0=sine, 1=square, 2=pulse, 3=saw, 4=saw_down, 5=triangle
    pwm_width_q15: int = 16384    # PWM duty cycle (Q15: 0=0%, 32767≈100%)
    pwm_enabled: bool = False     # Enable PWM modulation
    sync_enabled: bool = False    # Enable hard sync
    fm_depth_q15: int = 0         # FM modulation depth (Q15)
    fm_source: int = 0            # FM modulator source (0=off, 1=feedback, 2=lfo)
    unison_voices: int = 1        # Number of unison voices (1=mono)
    unison_spread_cents: int = 0  # Unison detune spread in cents
    drone_mode: bool = False      # Drone/Pad mode (reduces base_mix for slow evolution)
    legato_mode: bool = False     # Legato mode (preserve phase/coefficients on pitch change)
    custom_harmonics: list = None # Custom harmonic amplitudes (Q15) - overrides waveform_type
    phase_spread_q15: int = 0     # Per-harmonic phase dispersion for custom harmonics
    oeis_sequence_id: int = -1    # OEIS sequence for partial selection (-1=disabled, 0-7=sequence)
    oeis_rolloff_db: float = -3.0 # OEIS rolloff (dB/octave, negative = decay)


# FM modulator source constants (Phase 3)
FM_SOURCE_OFF = 0       # No FM modulation
FM_SOURCE_FEEDBACK = 1  # Operator feedback (DX-style)
FM_SOURCE_LFO = 2       # LFO (future)


# =============================================================================
# DECODED PLAN
# =============================================================================

@dataclass
class DecodedPlan:
    """
    Fully decoded execution plan for one block.

    Created from VoiceFrame by decode_frame().
    """
    # Node mask and K
    mask_u32: int = 0xFFFFFFFF
    K: int = 32

    # Tail operators
    tail_mode: TailMode = TailMode.FULL
    tail_ops: List[int] = field(default_factory=lambda: [2, 11, 23])

    # Tree parameters
    tree_a: int = 0
    tree_b: int = 0
    tree_blend_q15: int = 0

    # Carrier parameters (legacy, used when use_coeff_osc=False)
    carrier_params: CarrierParams = field(default_factory=CarrierParams)
    carrier_role: CarrierRole = CarrierRole.AUDIBLE

    # CoeffOsc parameters (CCS path, used when use_coeff_osc=True)
    coeff_osc_params: CoeffOscParams = field(default_factory=CoeffOscParams)
    use_coeff_osc: bool = True  # Enable CCS path by default

    # Dry/wet blend (eta) - for intra-block ramping
    eta_prev_q15: int = 0
    eta_curr_q15: int = 0

    # Gain - for intra-block ramping
    gain_prev_q15: int = 32767
    gain_curr_q15: int = 32767

    # Backbone instructions (in execution order)
    backbone_instrs: List[BackboneEntry] = field(default_factory=list)

    # Event instructions (active event lanes)
    event_instrs: List[EventEntry] = field(default_factory=list)

    # IDWPT enable
    idwpt_enable: bool = True

    # Pattern ID (for reference)
    pattern_id: int = 0

    # Noise exciter (Packet E4): deterministic seeded broadband transient
    # mixed into the voice output pre-gain. level 0 = disabled.
    noise_color: int = 0            # 0=white, 1=pink, 2=brown
    noise_level_q15: int = 0        # Mix level (Q15; 0 = off)
    noise_attack_blocks: int = 1    # Envelope attack (blocks)
    noise_decay_blocks: int = 45    # Envelope decay (blocks)
    noise_seed_u32: int = 0         # Preset meta.seed_u32


# =============================================================================
# EVENT LANE MAPPING
# =============================================================================

# Map event lane to default inject operator
EVENT_LANE_TO_OP: dict[int, int] = {
    0: 13,  # Impulse
    1: 13,  # Impulse
    2: 14,  # NoiseBand
    3: 8,   # GlobalPulse
    4: 16,  # GateMask
    5: 13,  # Impulse
    6: 13,  # Impulse
    7: 21,  # CuspInjection
}


# =============================================================================
# DECODE FUNCTION
# =============================================================================

def decode_frame(
    frame: 'VoiceFrame',
    tables: Tables = DEFAULT_TABLES,
    prev_eta_q15: int = 0,
    prev_gain_q15: int = 32767,
    prev_partial_amps: Optional[List[int]] = None,
) -> DecodedPlan:
    """
    Decode a VoiceFrame into a DecodedPlan.

    Args:
        frame: VoiceFrame to decode
        tables: Tables instance for lookups
        prev_eta_q15: Previous block's eta (for ramping)
        prev_gain_q15: Previous block's gain (for ramping)
        prev_partial_amps: Previous block's partial amplitudes (for latching)

    Returns:
        DecodedPlan ready for VM execution
    """
    plan = DecodedPlan()

    # -------------------------------------------------------------------------
    # Pattern / Mask / K
    # -------------------------------------------------------------------------
    if hasattr(frame, 'pattern_id') and frame.pattern_id is not None:
        pattern = tables.get_pattern(frame.pattern_id)
        plan.pattern_id = pattern.pattern_id
        plan.mask_u32 = pattern.mask_u32
        plan.K = pattern.k_value
    elif hasattr(frame, 'node_mask') and frame.node_mask is not None:
        plan.mask_u32 = frame.node_mask
        plan.K = popcount32(frame.node_mask)
    else:
        # Default: full mask
        plan.mask_u32 = 0xFFFFFFFF
        plan.K = 32

    # -------------------------------------------------------------------------
    # Tail mode
    # -------------------------------------------------------------------------
    if hasattr(frame, 'tail_mode') and frame.tail_mode is not None:
        plan.tail_mode = TailMode(frame.tail_mode)
    else:
        plan.tail_mode = TailMode.FULL
    plan.tail_ops = tables.get_tail_ops(plan.tail_mode)

    # -------------------------------------------------------------------------
    # Tree parameters
    # -------------------------------------------------------------------------
    plan.tree_a = getattr(frame, 'tree_a_id', 0) or 0
    plan.tree_b = getattr(frame, 'tree_b_id', 0) or 0
    plan.tree_blend_q15 = getattr(frame, 'tree_blend_q15', 0) or 0

    # -------------------------------------------------------------------------
    # Carrier parameters
    # -------------------------------------------------------------------------
    # Use fixed-point LUT for bit-exact pitch conversion (no floating-point noise)
    from ..fixed import midi_to_phase_inc_fixed

    note = getattr(frame, 'note', 60)
    cents = getattr(frame, 'cents', 0)
    phase_inc = midi_to_phase_inc_fixed(note, cents)

    # Keep f0_hz for compatibility/display (but don't use for phase_inc)
    f0_hz = getattr(frame, 'f0_hz', 440.0) or 440.0

    # Harmonic amplitudes: use frame value, latch from previous, or default
    if hasattr(frame, 'partial_amps_q15') and frame.partial_amps_q15:
        # Frame provides explicit values
        harmonic_amps = [32767] + [0] * 15
        for i, amp in enumerate(frame.partial_amps_q15[:16]):
            harmonic_amps[i] = amp
    elif prev_partial_amps:
        # Latch from previous block (keep previous harmonics)
        harmonic_amps = list(prev_partial_amps[:16])
        # Pad to 16 if needed
        while len(harmonic_amps) < 16:
            harmonic_amps.append(0)
    else:
        # Default: fundamental only
        harmonic_amps = [32767] + [0] * 15

    plan.carrier_params = CarrierParams(
        f0_hz=f0_hz,
        phase_inc=phase_inc,
        note=note,
        cents=cents,
        harmonic_amps_q15=harmonic_amps,
        model_id=0,
    )

    # -------------------------------------------------------------------------
    # CoeffOsc parameters (CCS path)
    # -------------------------------------------------------------------------
    # Get table ID from frame (default 0 = sine)
    coeff_table_id = getattr(frame, 'coeff_table_id', 0) or 0

    # base_mix controls how much oscillator vs CMMM evolved state
    # For CCS path: base_mix should default to full carrier (32767) for normal synthesis
    # beta_target_q15 (eta) controls wet/dry mixing, NOT carrier injection
    base_mix = getattr(frame, 'base_mix_q15', None)
    if base_mix is None:
        base_mix = 32767  # Full oscillator by default for CCS path

    # Extract Mode B parameters from VoiceFrame (WSE-STD-210 Section 10.2)
    osc_mode = getattr(frame, 'coeff_osc_mode', 0) or 0
    H_request = getattr(frame, 'H_request', 16) or 16
    waveform_type = getattr(frame, 'waveform_type', 0) or 0
    pwm_width_q15 = getattr(frame, 'pwm_width_q15', 16384) or 16384
    pwm_enabled = getattr(frame, 'pwm_enabled', False)
    sync_enabled = getattr(frame, 'sync_enabled', False)
    fm_depth_q15 = getattr(frame, 'fm_depth_q15', 0) or 0
    fm_source = getattr(frame, 'fm_source', 0) or 0
    unison_voices = getattr(frame, 'unison_voices', 1) or 1
    unison_spread_cents = getattr(frame, 'unison_spread_cents', 0) or 0
    drone_mode = getattr(frame, 'drone_mode', False)
    legato_mode = getattr(frame, 'legato_mode', False)

    # Custom harmonics for Mode B (overrides waveform_type if provided)
    custom_harmonics = None
    if hasattr(frame, 'partial_amps_q15') and frame.partial_amps_q15:
        # Use custom harmonics from frame
        custom_harmonics = list(frame.partial_amps_q15)
    phase_spread_q15 = getattr(frame, 'partial_phase_spread_q15', 0) or 0

    # OEIS sequence for partial selection (-1 = disabled)
    oeis_sequence_id = getattr(frame, 'oeis_sequence_id', -1)
    oeis_rolloff_db = getattr(frame, 'oeis_rolloff_db', -3.0)

    plan.coeff_osc_params = CoeffOscParams(
        table_id=coeff_table_id,
        base_mix_q15=base_mix,
        phase_inc=phase_inc,
        note=note,
        cents=cents,
        # WSE-STD-210 Mode B parameters
        osc_mode=osc_mode,
        H_request=H_request,
        waveform_type=waveform_type,
        pwm_width_q15=pwm_width_q15,
        pwm_enabled=pwm_enabled,
        sync_enabled=sync_enabled,
        fm_depth_q15=fm_depth_q15,
        fm_source=fm_source,
        unison_voices=unison_voices,
        unison_spread_cents=unison_spread_cents,
        drone_mode=drone_mode,
        legato_mode=legato_mode,
        custom_harmonics=custom_harmonics,
        phase_spread_q15=phase_spread_q15,
        oeis_sequence_id=oeis_sequence_id,
        oeis_rolloff_db=oeis_rolloff_db,
    )

    # Check if we should use CCS path (using fast on-demand table generation)
    plan.use_coeff_osc = getattr(frame, 'use_coeff_osc', True)

    # Carrier role
    # NOTE: Don't use `or 24576` here - 0 is a valid value (pure carrier)
    beta_target = getattr(frame, 'beta_target_q15', None)
    if beta_target is None:
        beta_target = 24576
    if beta_target >= 32000:
        plan.carrier_role = CarrierRole.EXCITER_ONLY
    elif beta_target <= 1000:
        plan.carrier_role = CarrierRole.AUDIBLE  # Carrier-only, but still audible
    else:
        plan.carrier_role = CarrierRole.AUDIBLE

    # -------------------------------------------------------------------------
    # Eta (dry/wet) ramping
    # -------------------------------------------------------------------------
    plan.eta_prev_q15 = prev_eta_q15
    # Current eta target (beta in VoiceFrame)
    plan.eta_curr_q15 = beta_target

    # -------------------------------------------------------------------------
    # Gain ramping
    # -------------------------------------------------------------------------
    plan.gain_prev_q15 = prev_gain_q15
    # NOTE: Don't use `or 32767` here - 0 is a valid gain value (silence)
    gain = getattr(frame, 'gain_q15', None)
    plan.gain_curr_q15 = 32767 if gain is None else gain

    # -------------------------------------------------------------------------
    # Backbone instructions
    # -------------------------------------------------------------------------
    if hasattr(frame, 'ops_enable') and frame.ops_enable:
        families = getattr(frame, 'families', {}) or {}
        for op_name in frame.ops_enable:
            # Get operator ID from name
            op_id = _op_name_to_id(op_name)
            if op_id is not None:
                strength = int(max(0.0, min(1.0, _op_strength(op_name, families))) * 32767)
                plan.backbone_instrs.append(BackboneEntry(
                    op_id=op_id,
                    strength_q15=strength,
                ))

    # Alternative: backbone_ops as list of IDs
    if hasattr(frame, 'backbone_ops') and frame.backbone_ops:
        for op_id in frame.backbone_ops:
            plan.backbone_instrs.append(BackboneEntry(op_id=op_id))

    # -------------------------------------------------------------------------
    # Event instructions
    # -------------------------------------------------------------------------
    if hasattr(frame, 'event_mask') and frame.event_mask:
        event_mask = frame.event_mask
        event_strengths = getattr(frame, 'event_strength_q15', [32767] * 8) or [32767] * 8

        for lane in range(8):
            if event_mask & (1 << lane):
                op_id = EVENT_LANE_TO_OP.get(lane, 13)
                strength = event_strengths[lane] if lane < len(event_strengths) else 32767
                plan.event_instrs.append(EventEntry(
                    lane=lane,
                    op_id=op_id,
                    strength_q15=strength,
                ))

    # -------------------------------------------------------------------------
    # Noise exciter (Packet E4) - level 0 keeps the path disabled
    # -------------------------------------------------------------------------
    plan.noise_level_q15 = getattr(frame, 'noise_level_q15', 0) or 0
    if plan.noise_level_q15 > 0:
        plan.noise_color = getattr(frame, 'noise_color', 0) or 0
        plan.noise_attack_blocks = getattr(frame, 'noise_attack_blocks', 1) or 1
        plan.noise_decay_blocks = getattr(frame, 'noise_decay_blocks', 45) or 45
        plan.noise_seed_u32 = getattr(frame, 'noise_seed_u32', 0) or 0

    # -------------------------------------------------------------------------
    # IDWPT enable
    # -------------------------------------------------------------------------
    plan.idwpt_enable = plan.eta_curr_q15 > 0

    return plan


# =============================================================================
# OPERATOR NAME MAPPING
# =============================================================================

_OP_NAME_TO_ID: dict[str, int] = {
    'identity': 0,
    'tilt': 1,
    'fineclamp': 2,
    'fine_clamp': 2,
    'intramix': 3,
    'intra_mix': 3,
    'diffuseup': 4,
    'diffuse_up': 4,
    'diffusedown': 5,
    'diffuse_down': 5,
    'bandshear': 6,
    'band_shear': 6,
    'lowrankmorph': 7,
    'lowrank_morph': 7,
    'globalpulse': 8,
    'global_pulse': 8,
    'swapbands': 9,
    'swap_bands': 9,
    'rotatenodes': 10,
    'rotate_nodes': 10,
    'normalizescale': 11,
    'normalize_scale': 11,
    'gluepair': 12,
    'glue_pair': 12,
    'impulse': 13,
    'noiseband': 14,
    'noise_band': 14,
    'pressureramp': 15,
    'pressure_ramp': 15,
    'gatemask': 16,
    'gate_mask': 16,
    'quasirotate': 17,
    'quasi_rotate': 17,
    'morphmix': 18,
    'morph_mix': 18,
    'emergencysimplify': 19,
    'emergency_simplify': 19,
    'foldpressure': 20,
    'fold_pressure': 20,
    'cuspinjection': 21,
    'cusp_injection': 21,
    'swallowtailmask': 22,
    'swallowtail_mask': 22,
    'sheafproject': 23,
    'sheaf_project': 23,
}


def _op_name_to_id(name: str) -> Optional[int]:
    """Convert operator name to ID."""
    return _OP_NAME_TO_ID.get(name.lower().replace('-', '_').replace(' ', '_'))


# Family-vector slot for axis / legacy emphasis / legacy family names
# (docs/jargon.md; slot 0 = TAIL, always-on safety, never author-weighted).
_FAMILY_NAME_SLOT = {
    'color': 1, 'spread': 2, 'body': 3, 'warp': 4,
    'glue': 5, 'snap': 6, 'chaos': 7,
    # legacy emphasis / character names
    'tone': 1, 'brightness': 1, 'slip': 2, 'density': 3, 'dream': 4,
    'clarity': 5, 'stability': 5, 'settling': 5, 'energy': 6, 'risk': 7,
    # legacy emit-template family names (pre 7-axis collapse)
    'COLOR': 1, 'TRANSPORT': 2, 'LOCAL': 3, 'TOPOLOGY': 4,
    'COHERENCE': 5, 'GLUE': 5, 'EXCITE': 6, 'EVENTS': 6, 'CATASTROPHE': 7,
}

# Operator -> owning family slot (AXIS_OPERATORS in instrument_ir.py, plus
# escape-hatch TOP ops with a clear family). Unmapped ops (safety/TAIL,
# LowRankMorph, NoiseBand, ...) run at full strength.
_OP_FAMILY_SLOT = {
    'Tilt': 1,
    'DiffuseDown': 2, 'DiffuseUp': 2,
    'IntraMix': 3,
    'BandShear': 4, 'SwapBands': 4, 'RotateNodes': 4,
    'QuasiRotate': 4, 'GateMask': 4,
    'GluePair': 5, 'SheafProject': 5,
    'Impulse': 6, 'GlobalPulse': 6,
    'FoldPressure': 7, 'CuspInjection': 7, 'SwallowtailMask': 7,
}


def _op_strength(op_name: str, families) -> float:
    """
    Resolve an enabled operator's strength from the frame's family weights.

    families may be a dict keyed by axis / legacy emphasis / legacy family /
    operator name (weights 0.0-1.0), or an 8-slot family vector (floats or
    Q15 ints). An operator is scaled only when its family carries an explicit
    weight; otherwise it runs at full strength (preserves curated presets
    whose lists predate the 7-axis weights).
    """
    if isinstance(families, dict):
        if op_name in families:
            return float(families[op_name])  # direct per-op override
        slot = _OP_FAMILY_SLOT.get(op_name)
        if slot is not None:
            total = None
            for name, weight in families.items():
                if _FAMILY_NAME_SLOT.get(name) == slot:
                    total = (total or 0.0) + float(weight)
            if total is not None:
                return min(1.0, total)
        return 1.0
    if isinstance(families, (list, tuple)) and families:
        slot = _OP_FAMILY_SLOT.get(op_name)
        if slot is not None and slot < len(families):
            value = families[slot]
            if isinstance(value, int) and value > 1:
                return value / 32767.0  # Q15 family vector
            return float(value)
        return 1.0
    return 1.0


__all__ = [
    'CarrierRole',
    'BackboneEntry',
    'EventEntry',
    'CarrierParams',
    'CoeffOscParams',
    'DecodedPlan',
    'decode_frame',
    # FM modulator source constants (Phase 3)
    'FM_SOURCE_OFF',
    'FM_SOURCE_FEEDBACK',
    'FM_SOURCE_LFO',
]
