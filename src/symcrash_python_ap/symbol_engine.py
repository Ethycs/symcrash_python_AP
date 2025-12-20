"""
Symbol Engine Executor v1.0
===========================

Consumes VoiceFrame[] and updates per-voice coefficient state deterministically.

Invariants:
1. Total machine: For every (state, VoiceFrame) there is a well-defined output state.
2. Deterministic: Same inputs → same outputs (no hidden RNG; seeds are explicit/fixed).
3. Bounded work: Per voice per block cost is predictable and capped.
4. Stateless operators: All ops are pure w.r.t. coefficients this block.

Symbol Engine owns:
- Per-voice coefficient state c_q15[] (and auxiliary buffers for tree morph)
- Deterministic execution of operator pipeline per block per voice
- Legality enforcement (masks, tail constraints, safety-only kill)
- Coefficient slew/stability stage

Symbol Engine does NOT own:
- Tempo, beat, phrase timing (Voice Engine)
- Note lifecycle (Voice Engine)
- Audio reconstruction (Wavelet Engine)
- OLA/declick/limiting (Safety Engine)
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np

from fixed import (
    sat16, mul_q15_sat, shift_round_sat16, slew_q15,
    Q15_ONE, Q15_SHIFT, mac_q15
)
from frames import ParamFrame, VoiceFrame

# Import operator library
from operator_library import (
    build_operator_library, COEFF_LEN, BAND_SLICES, NUM_BANDS,
    OpType, Operator, from_q15, to_q15
)


# =============================================================================
# CONSTANTS
# =============================================================================

NUM_FAMILIES = 8
NUM_EVENT_LANES = 8
MAX_TREES = 5

# Family indices (F0-F7)
F_IDENTITY = 0    # Reserved / Identity
F_COLOR = 1       # Tilt, spectral shape
F_TRANSPORT = 2   # DiffuseUp/Down, LowRankMorph
F_LOCAL = 3       # IntraMix
F_TOPOLOGY = 4    # BandShear, RotateNodes, QuasiRotate, SwapBands
F_GLUE = 5        # GluePair, SheafProject
F_RESERVED = 6    # Reserved for future
F_CATASTROPHE = 7 # PressureRamp, FoldPressure, GateMask, SwallowtailMask

# Event lane indices (E0-E7)
E_IMPULSE_0 = 0
E_IMPULSE_1 = 1
E_NOISE = 2
E_PULSE = 3
E_GATE = 4
E_IMPULSE_2 = 5
E_IMPULSE_3 = 6
E_CUSP = 7

# Degradation levels
DEGRADE_NONE = 0
DEGRADE_LIGHT = 1
DEGRADE_HEAVY = 2
DEGRADE_EMERGENCY = 3


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class VoiceCoeffState:
    """
    Per-voice coefficient state with dual-buffer morph support (Policy S1).

    Maintains cA for tree A and cB for tree B during morph windows.
    """
    c_q15: List[int] = field(default_factory=lambda: [0] * COEFF_LEN)
    cB_q15: Optional[List[int]] = None  # Secondary buffer (only during morph)

    primary_tree_id: int = 0
    secondary_tree_id: int = 0
    has_secondary: bool = False

    # Previous state for slewing
    c_prev: List[int] = field(default_factory=lambda: [0] * COEFF_LEN)


@dataclass
class EventPayload:
    """Event payload for injection operators."""
    target_group: int = 3       # 0=low, 1=mid, 2=high, 3=all
    seed: int = 0               # Combined with voice_id for deterministic RNG
    param0_q8: int = 0x80       # Lane-specific param (Q8.8)
    param1_q8: int = 0x80       # Lane-specific param (Q8.8)


@dataclass
class FamilySpec:
    """
    Maps family weight to concrete operator parameters.

    For each family, defines:
    - Which operators are enabled
    - Parameter scaling from weight
    - Regime-specific overrides
    """
    enabled_ops: List[int] = field(default_factory=list)
    param_scale: float = 1.0
    param_bias: float = 0.0
    max_param: float = 1.0


@dataclass
class TailSpec:
    """Tail stage configuration (always executed)."""
    normalize_curve: str = "pink"      # "flat", "pink", "bright"
    fineclamp_cap_ratio: float = 0.5
    sheaf_sections: int = 4
    slew_max_delta: int = 512


@dataclass
class RegimeOverride:
    """Per-regime parameter overrides."""
    family_enable_mask: int = 0xFF     # Bitmask of enabled families
    normalize_curve: str = "pink"
    fineclamp_cap_ratio: float = 0.5
    sheaf_sections: int = 4
    glue_pairs: List[Tuple[int, int]] = field(default_factory=lambda: [(2, 3)])
    # Family-specific biases
    tilt_bias: float = 0.0
    transport_scale: float = 1.0
    catastrophe_enable: bool = False


@dataclass
class DegradePolicy:
    """Degradation policy: which ops/families to drop at each level."""
    # Level 1: Light degradation
    level1_disable_families: List[int] = field(default_factory=lambda: [])
    level1_cap_transport: float = 0.3
    level1_disable_swapbands: bool = True

    # Level 2: Heavy degradation
    level2_disable_families: List[int] = field(default_factory=lambda: [F_TRANSPORT, F_TOPOLOGY])

    # Level 3: Emergency - only tail stages
    # (Handled specially in executor)


@dataclass
class InstrumentProgram:
    """
    Compiled program table for an instrument.

    Symbol does not "interpret vibes"; it executes a compiled plan.
    """
    program_id: int = 0
    name: str = "Default"

    # Family configuration
    family_specs: Dict[int, FamilySpec] = field(default_factory=dict)

    # Tail configuration (always executed)
    tail: TailSpec = field(default_factory=TailSpec)

    # Per-regime overrides
    regime_overrides: Dict[int, RegimeOverride] = field(default_factory=dict)

    # Degradation policy
    degrade: DegradePolicy = field(default_factory=DegradePolicy)

    # Activity thresholds for voice skipping
    activity_min_q15: int = 256
    gain_min_q15: int = 256
    skip_interval_blocks: int = 4


@dataclass
class ExecutorInput:
    """
    Per-voice input to the executor for one block.

    Decoded from VoiceFrame with additional control fields.
    """
    voice_id: int = 0

    # Family weights (Q1.15)
    w_family: List[int] = field(default_factory=lambda: [0] * NUM_FAMILIES)

    # Control state
    regime_id: int = 0
    degradation_level: int = DEGRADE_NONE

    # Event control
    event_mask: int = 0                # Bitmask of active event lanes
    event_strength: List[int] = field(default_factory=lambda: [0] * NUM_EVENT_LANES)
    event_payload: EventPayload = field(default_factory=EventPayload)

    # Tree control (Option A)
    tree_a_id: int = 0
    tree_b_id: int = 0
    blend_q15: int = 0                 # 0 = pure A, Q15_ONE = pure B

    # Activity (for skip decisions)
    activity_q15: int = Q15_ONE
    gain_q15: int = 0x4000


# =============================================================================
# OPERATOR ENGINE (unchanged from original)
# =============================================================================

class OperatorEngine:
    """
    Applies operators from the library with bit-accurate arithmetic.
    """

    def __init__(self, coeff_len: int = COEFF_LEN):
        self.coeff_len = coeff_len
        self.library = build_operator_library()

        # Slew state for coefficients
        self.slew_max_delta = 512  # Max change per block

    def apply_diag(self, c: List[int], diag: np.ndarray) -> List[int]:
        """Apply diagonal operator (bit-accurate)."""
        result = []
        for i in range(len(c)):
            val = mul_q15_sat(c[i], int(diag[i]))
            result.append(val)
        return result

    def apply_permute(self, c: List[int], perm: np.ndarray) -> List[int]:
        """Apply permutation operator."""
        result = [0] * len(c)
        for i, j in enumerate(perm):
            result[i] = c[int(j)]
        return result

    def apply_lowrank(self, c: List[int], U: np.ndarray, V: np.ndarray,
                      strength_q15: int = Q15_ONE) -> List[int]:
        """Apply low-rank operator: c + strength * Σ U_k (V_k · c)."""
        result = list(c)
        rank = U.shape[0]

        for k in range(rank):
            # Compute V_k · c (dot product in int32)
            dot = 0
            for i in range(len(c)):
                dot = mac_q15(dot, int(V[k, i]), c[i])
            dot = shift_round_sat16(dot, Q15_SHIFT)

            # Scale by strength
            dot = mul_q15_sat(dot, strength_q15)

            # Add U_k * dot to result
            for i in range(len(c)):
                contrib = mul_q15_sat(int(U[k, i]), dot)
                result[i] = sat16(result[i] + contrib)

        return result

    def apply_banded_tridiag(self, c: List[int], op: Operator,
                             strength_q15: int = Q15_ONE) -> List[int]:
        """Apply tridiagonal banded operator with strength scaling."""
        result = [0] * len(c)

        for band_id, (start, end) in enumerate(BAND_SLICES):
            kernel_info = op.bands_kernel.get(band_id)
            if kernel_info is None or "kernel" not in kernel_info:
                for i in range(start, end):
                    result[i] = c[i]
                continue

            kernel = kernel_info["kernel"]
            # Scale kernel by strength
            a = mul_q15_sat(int(kernel[0]), strength_q15)
            b_orig = int(kernel[1])
            a2 = mul_q15_sat(int(kernel[2]), strength_q15)

            # Adjust center weight to maintain energy
            b = b_orig + (Q15_ONE - strength_q15) if strength_q15 < Q15_ONE else b_orig

            for i in range(start, end):
                i_left = max(start, i - 1)
                i_right = min(end - 1, i + 1)

                acc = 0
                acc = mac_q15(acc, a, c[i_left])
                acc = mac_q15(acc, b, c[i])
                acc = mac_q15(acc, a2, c[i_right])
                result[i] = shift_round_sat16(acc, Q15_SHIFT)

        return result

    def apply_inject(self, c: List[int], inject: np.ndarray,
                     amplitude: int = Q15_ONE) -> List[int]:
        """Apply injection (add scaled vector)."""
        result = list(c)
        for i in range(len(c)):
            if inject[i] != 0:
                contrib = mul_q15_sat(int(inject[i]), amplitude)
                result[i] = sat16(result[i] + contrib)
        return result

    def apply_proj_energy_cap(self, c: List[int], op: Operator,
                              cap_ratio: float = 0.5) -> List[int]:
        """Apply energy capping projection per band."""
        result = list(c)
        targets = op.proj_params.get("targets_q15")

        if targets is None:
            return result

        for band_id, (start, end) in enumerate(BAND_SLICES):
            # Measure band energy (in int32)
            energy = 0
            for i in range(start, end):
                energy += c[i] * c[i]

            # Target energy scaled by cap_ratio
            target_energy = 0
            for i in range(start, end):
                t = int(targets[i] * cap_ratio)
                target_energy += t * t

            # Scale if over budget
            if energy > target_energy and energy > 0:
                scale = int(np.sqrt(target_energy / energy) * Q15_ONE)
                scale = min(Q15_ONE, scale)

                for i in range(start, end):
                    result[i] = mul_q15_sat(result[i], scale)

        return result

    def apply_operator(self, c: List[int], op_id: int,
                       strength_q15: int = Q15_ONE,
                       params: Optional[Dict] = None) -> List[int]:
        """Apply single operator by ID with optional strength scaling."""
        if op_id not in self.library:
            return c

        op = self.library[op_id]
        params = params or {}

        if op.op_type == OpType.DIAG:
            if strength_q15 >= Q15_ONE:
                return self.apply_diag(c, op.diag)
            # Blend with identity for partial strength
            result = self.apply_diag(c, op.diag)
            inv_strength = Q15_ONE - strength_q15
            return [sat16(mul_q15_sat(r, strength_q15) + mul_q15_sat(c[i], inv_strength))
                    for i, r in enumerate(result)]

        elif op.op_type == OpType.PERMUTE:
            if strength_q15 >= Q15_ONE:
                return self.apply_permute(c, op.perm)
            # Permutation doesn't interpolate cleanly; threshold at 0.5
            if strength_q15 > Q15_ONE // 2:
                return self.apply_permute(c, op.perm)
            return c

        elif op.op_type == OpType.LOWRANK:
            return self.apply_lowrank(c, op.U, op.V, strength_q15)

        elif op.op_type == OpType.BANDED:
            if op.bands_kernel and "kernel" in list(op.bands_kernel.values())[0]:
                return self.apply_banded_tridiag(c, op, strength_q15)
            return c

        elif op.op_type == OpType.INJECT:
            amp = params.get("amplitude", strength_q15)
            return self.apply_inject(c, op.inject_vec, amp)

        elif op.op_type == OpType.PROJ:
            cap = params.get("cap_ratio", 0.5)
            return self.apply_proj_energy_cap(c, op, cap)

        elif op.op_type == OpType.MIX:
            if op.mix_ops:
                c_a = self.apply_operator(c, op.mix_ops[0], strength_q15, params)
                c_b = self.apply_operator(c, op.mix_ops[1], strength_q15, params)
                w = int(op.mix_weight * Q15_ONE)
                w_inv = Q15_ONE - w
                return [sat16(mul_q15_sat(c_a[i], w_inv) + mul_q15_sat(c_b[i], w))
                        for i in range(len(c))]
            return c

        return c

    def slew_coeffs(self, c_old: List[int], c_new: List[int],
                    max_delta: int = 512) -> List[int]:
        """Slew coefficients to prevent discontinuities."""
        return [slew_q15(old, new, max_delta) for old, new in zip(c_old, c_new)]


# =============================================================================
# SYMBOL ENGINE EXECUTOR
# =============================================================================

class SymbolExecutor:
    """
    Symbol Engine Executor v1.0

    Executes compiled instrument programs on per-voice coefficient state.
    Deterministic, bounded, and total.
    """

    def __init__(self, n_voices: int = 128, coeff_len: int = COEFF_LEN):
        self.n_voices = n_voices
        self.coeff_len = coeff_len

        # Per-voice coefficient state
        self.voice_state: Dict[int, VoiceCoeffState] = {}

        # Operator engine
        self.ops = OperatorEngine(coeff_len)

        # Program table (loaded externally)
        self.programs: Dict[int, InstrumentProgram] = {
            0: self._build_default_program()
        }

        # Debug counters
        self.stats = {
            "ops_executed": 0,
            "clamps_hit": 0,
            "simplify_triggered": 0,
            "voices_skipped": 0,
        }

    def _build_default_program(self) -> InstrumentProgram:
        """Build default instrument program."""
        prog = InstrumentProgram(
            program_id=0,
            name="Default",
        )

        # Family specs
        prog.family_specs = {
            F_COLOR: FamilySpec(enabled_ops=[1], param_scale=3.0, max_param=6.0),
            F_TRANSPORT: FamilySpec(enabled_ops=[4, 5, 7], param_scale=0.2, max_param=0.4),
            F_LOCAL: FamilySpec(enabled_ops=[3], param_scale=0.15, max_param=0.3),
            F_TOPOLOGY: FamilySpec(enabled_ops=[6, 10, 17], param_scale=1.0, max_param=3.0),
            F_GLUE: FamilySpec(enabled_ops=[12, 23], param_scale=0.3, max_param=0.5),
            F_CATASTROPHE: FamilySpec(enabled_ops=[15, 20, 16, 22], param_scale=0.1, max_param=0.3),
        }

        # Regime overrides
        prog.regime_overrides = {
            0: RegimeOverride(normalize_curve="pink", fineclamp_cap_ratio=0.5),
            1: RegimeOverride(normalize_curve="flat", fineclamp_cap_ratio=0.4,
                            family_enable_mask=0xFF, catastrophe_enable=True),
            2: RegimeOverride(normalize_curve="bright", fineclamp_cap_ratio=0.3,
                            catastrophe_enable=True),
        }

        return prog

    def get_voice_state(self, voice_id: int) -> VoiceCoeffState:
        """Get or create voice coefficient state."""
        if voice_id not in self.voice_state:
            self.voice_state[voice_id] = VoiceCoeffState()
        return self.voice_state[voice_id]

    def load_program(self, program: InstrumentProgram):
        """Load an instrument program."""
        self.programs[program.program_id] = program

    # =========================================================================
    # §5.1 PRE-STEP: DECODE / NORMALIZE CONTROLS
    # =========================================================================

    def decode_input(self, vf: VoiceFrame, program_id: int = 0) -> ExecutorInput:
        """
        Decode VoiceFrame into ExecutorInput.

        Maps VoiceFrame fields to executor control structure.
        """
        inp = ExecutorInput(voice_id=vf.voice_id)

        # Tree control (Option A)
        # VoiceFrame uses tree_a, tree_b (floats 0-1) for beta values
        # We map to tree IDs based on beta thresholds
        inp.tree_a_id = self._beta_to_tree_id(vf.tree_a)
        inp.tree_b_id = self._beta_to_tree_id(vf.tree_b)
        inp.blend_q15 = int(vf.tree_blend * Q15_ONE)

        # Activity/gain for skip decisions
        inp.activity_q15 = vf.density_env_q15
        inp.gain_q15 = vf.gain_q15

        # Derive family weights from VoiceFrame envelopes
        # F1 (Color) from tilt envelope
        inp.w_family[F_COLOR] = vf.tilt_env_q15
        # F3 (Local) from density
        inp.w_family[F_LOCAL] = vf.density_env_q15
        # Others default to moderate values
        inp.w_family[F_TRANSPORT] = Q15_ONE // 4
        inp.w_family[F_TOPOLOGY] = Q15_ONE // 8
        inp.w_family[F_GLUE] = Q15_ONE // 2

        # Regime from material_id (placeholder mapping)
        inp.regime_id = min(vf.material_id, 2)

        # Event mask and strength from aux fields
        inp.event_mask = vf.aux_flags & 0xFF
        # Distribute aux_param across event strengths
        for i in range(NUM_EVENT_LANES):
            if inp.event_mask & (1 << i):
                inp.event_strength[i] = vf.aux_param_q15

        return inp

    def _beta_to_tree_id(self, beta: float) -> int:
        """Map beta (0-1) to tree ID (0-4)."""
        if beta < 0.2:
            return 0  # Balanced
        elif beta < 0.4:
            return 1  # Harmonic
        elif beta < 0.6:
            return 2  # Texture
        elif beta < 0.8:
            return 3  # Transient
        else:
            return 4  # Drone

    # =========================================================================
    # §5.2 ENSURE MORPH BUFFERS CONSISTENT
    # =========================================================================

    def ensure_morph_buffers(self, state: VoiceCoeffState, inp: ExecutorInput):
        """
        Ensure morph buffers are consistent with tree control.

        Policy S1: Dual-state during morph.
        """
        # Hard retarget check
        if inp.tree_a_id != state.primary_tree_id:
            state.primary_tree_id = inp.tree_a_id
            # Keep coefficients as-is (unified layout assumption)

        # Secondary buffer management
        blend = inp.blend_q15

        if 0 < blend < Q15_ONE:
            # In morph window
            if inp.tree_b_id != state.secondary_tree_id or not state.has_secondary:
                # Allocate/refresh secondary buffer
                state.cB_q15 = list(state.c_q15)  # Copy from primary
                state.secondary_tree_id = inp.tree_b_id
                state.has_secondary = True

        elif blend == 0:
            # Pure A - can drop secondary
            if state.has_secondary:
                state.cB_q15 = None
                state.has_secondary = False

        elif blend >= Q15_ONE:
            # Pure B - promote secondary to primary
            if state.has_secondary and state.cB_q15 is not None:
                state.c_q15 = list(state.cB_q15)
                state.primary_tree_id = state.secondary_tree_id
                state.cB_q15 = None
                state.has_secondary = False

    # =========================================================================
    # §5.3 COMPUTE EFFECTIVE FAMILY ENABLE MASK
    # =========================================================================

    def compute_family_enable(self, inp: ExecutorInput,
                              program: InstrumentProgram) -> Tuple[int, List[int]]:
        """
        Compute effective family enable mask and weights.

        Returns: (enable_mask, w_eff[8])
        """
        # Start from program defaults
        enable_mask = 0xFF
        w_eff = list(inp.w_family)

        # Apply regime overrides
        regime = program.regime_overrides.get(inp.regime_id)
        if regime:
            enable_mask &= regime.family_enable_mask

            # Apply biases
            if regime.tilt_bias != 0:
                bias_q15 = int(regime.tilt_bias * Q15_ONE)
                w_eff[F_COLOR] = sat16(w_eff[F_COLOR] + bias_q15)

            if regime.transport_scale != 1.0:
                w_eff[F_TRANSPORT] = int(w_eff[F_TRANSPORT] * regime.transport_scale)

            # Catastrophe enable
            if not regime.catastrophe_enable:
                enable_mask &= ~(1 << F_CATASTROPHE)

        # Apply degradation
        degrade = program.degrade
        level = inp.degradation_level

        if level >= DEGRADE_LIGHT:
            # Level 1
            if inp.event_mask == 0:
                # Disable event-only execution
                pass  # Events handled separately
            # Cap transport
            max_transport = int(degrade.level1_cap_transport * Q15_ONE)
            w_eff[F_TRANSPORT] = min(w_eff[F_TRANSPORT], max_transport)

        if level >= DEGRADE_HEAVY:
            # Level 2 - disable transport and topology
            for fam in degrade.level2_disable_families:
                enable_mask &= ~(1 << fam)

        if level >= DEGRADE_EMERGENCY:
            # Level 3 - only tail stages
            enable_mask = 0

        # Zero weights for disabled families
        for i in range(NUM_FAMILIES):
            if not (enable_mask & (1 << i)):
                w_eff[i] = 0

        return enable_mask, w_eff

    # =========================================================================
    # §5.5 APPLY OPERATOR CHAIN
    # =========================================================================

    def apply_chain(self, c: List[int], inp: ExecutorInput,
                    program: InstrumentProgram,
                    enable_mask: int, w_eff: List[int]) -> List[int]:
        """
        Apply canonical operator chain (Stages 0-9).

        Fixed operator order for predictable runtime.
        """
        # Get regime override
        regime = program.regime_overrides.get(inp.regime_id, RegimeOverride())

        # --- Stage 0: Identity (usually skipped) ---
        # No-op

        # --- Stage 1: Color ---
        if enable_mask & (1 << F_COLOR) and w_eff[F_COLOR] > 0:
            c = self.ops.apply_operator(c, 1, w_eff[F_COLOR])  # Tilt
            self.stats["ops_executed"] += 1

        # --- Stage 2: Local coherence ---
        if enable_mask & (1 << F_LOCAL) and w_eff[F_LOCAL] > 0:
            c = self.ops.apply_operator(c, 3, w_eff[F_LOCAL])  # IntraMix
            self.stats["ops_executed"] += 1

        # --- Stage 3: Transport ---
        if enable_mask & (1 << F_TRANSPORT) and w_eff[F_TRANSPORT] > 0:
            c = self.ops.apply_operator(c, 4, w_eff[F_TRANSPORT])   # DiffuseUp
            c = self.ops.apply_operator(c, 5, w_eff[F_TRANSPORT])   # DiffuseDown
            c = self.ops.apply_operator(c, 7, w_eff[F_TRANSPORT])   # LowRankMorph
            self.stats["ops_executed"] += 3

        # --- Stage 4: Topology/motion ---
        if enable_mask & (1 << F_TOPOLOGY) and w_eff[F_TOPOLOGY] > 0:
            c = self.ops.apply_operator(c, 6, w_eff[F_TOPOLOGY])    # BandShear
            c = self.ops.apply_operator(c, 10, w_eff[F_TOPOLOGY])   # RotateNodes
            c = self.ops.apply_operator(c, 17, w_eff[F_TOPOLOGY])   # QuasiRotate
            self.stats["ops_executed"] += 3

            # SwapBands only at degradation 0 and if explicitly allowed
            if inp.degradation_level == DEGRADE_NONE:
                # SwapBands disabled by default for safety
                pass

        # --- Stage 5: Catastrophe corridor ---
        if enable_mask & (1 << F_CATASTROPHE) and w_eff[F_CATASTROPHE] > 0:
            c = self.ops.apply_operator(c, 15, w_eff[F_CATASTROPHE])  # PressureRamp
            c = self.ops.apply_operator(c, 20, w_eff[F_CATASTROPHE])  # FoldPressure
            c = self.ops.apply_operator(c, 16, w_eff[F_CATASTROPHE])  # GateMask
            c = self.ops.apply_operator(c, 22, w_eff[F_CATASTROPHE])  # SwallowtailMask
            self.stats["ops_executed"] += 4

        # --- Stage 6: Event injections ---
        c = self.apply_events(c, inp, program)

        # --- Stage 7: Constraints / glue (pre-tail) ---
        if enable_mask & (1 << F_GLUE) and w_eff[F_GLUE] > 0:
            # GluePair
            c = self.ops.apply_operator(c, 12, w_eff[F_GLUE],
                                       {"band_a": regime.glue_pairs[0][0] if regime.glue_pairs else 2,
                                        "band_b": regime.glue_pairs[0][1] if regime.glue_pairs else 3})
            # SheafProject
            c = self.ops.apply_operator(c, 23, w_eff[F_GLUE],
                                       {"num_sections": regime.sheaf_sections})
            self.stats["ops_executed"] += 2

        # --- Stage 8: Tail constraints (ALWAYS) ---
        c = self.apply_tail(c, program, regime)

        # --- Stage 9: Safety-only intervention ---
        if inp.degradation_level >= DEGRADE_EMERGENCY:
            c = self.ops.apply_operator(c, 19, Q15_ONE)  # EmergencySimplify
            self.stats["simplify_triggered"] += 1

        # Check for numeric overflow and trigger emergency if needed
        max_coeff = max(abs(x) for x in c)
        if max_coeff > 32000:  # Near saturation
            c = self.ops.apply_operator(c, 19, Q15_ONE)  # EmergencySimplify
            self.stats["clamps_hit"] += 1

        return c

    def apply_events(self, c: List[int], inp: ExecutorInput,
                     program: InstrumentProgram) -> List[int]:
        """
        Apply event injections (Stage 6).

        Executed in lane order if corresponding bit in event_mask is set.
        """
        if inp.event_mask == 0:
            return c

        # E0/E1/E5/E6 → Impulse (ID 13)
        for lane in [E_IMPULSE_0, E_IMPULSE_1, E_IMPULSE_2, E_IMPULSE_3]:
            if inp.event_mask & (1 << lane):
                amp = inp.event_strength[lane]
                c = self.ops.apply_operator(c, 13, amp, {"amplitude": amp})
                self.stats["ops_executed"] += 1

        # E2 → NoiseBand (ID 14)
        if inp.event_mask & (1 << E_NOISE):
            amp = inp.event_strength[E_NOISE]
            c = self.ops.apply_operator(c, 14, amp, {"amplitude": amp})
            self.stats["ops_executed"] += 1

        # E3 → GlobalPulse (ID 8)
        if inp.event_mask & (1 << E_PULSE):
            amp = inp.event_strength[E_PULSE]
            c = self.ops.apply_operator(c, 8, amp, {"amplitude": amp})
            self.stats["ops_executed"] += 1

        # E4 → GateMask short window (ID 16)
        if inp.event_mask & (1 << E_GATE):
            c = self.ops.apply_operator(c, 16, inp.event_strength[E_GATE])
            self.stats["ops_executed"] += 1

        # E7 → CuspInjection (ID 21)
        if inp.event_mask & (1 << E_CUSP):
            amp = inp.event_strength[E_CUSP]
            c = self.ops.apply_operator(c, 21, amp, {"amplitude": amp})
            self.stats["ops_executed"] += 1

        return c

    def apply_tail(self, c: List[int], program: InstrumentProgram,
                   regime: RegimeOverride) -> List[int]:
        """
        Apply tail constraints (Stage 8) - ALWAYS executed.

        NormalizeScale + FineClamp
        """
        tail = program.tail

        # NormalizeScale (ID 11)
        c = self.ops.apply_operator(c, 11, Q15_ONE,
                                   {"target_curve": regime.normalize_curve or tail.normalize_curve})

        # FineClamp (ID 2)
        cap = regime.fineclamp_cap_ratio if regime else tail.fineclamp_cap_ratio
        c = self.ops.apply_operator(c, 2, Q15_ONE, {"cap_ratio": cap})

        self.stats["ops_executed"] += 2

        return c

    # =========================================================================
    # §5.6 OUTPUT STATE (POST-CHAIN SLEW)
    # =========================================================================

    def apply_slew(self, state: VoiceCoeffState, c_new: List[int],
                   program: InstrumentProgram) -> List[int]:
        """
        Apply post-chain slew (Stage 10) - ALWAYS executed.

        Numerical stability stage, not musical timing.
        """
        max_delta = program.tail.slew_max_delta
        c_slewed = self.ops.slew_coeffs(state.c_prev, c_new, max_delta)
        state.c_prev = list(c_slewed)
        return c_slewed

    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================

    def step_voice(self, vf: VoiceFrame, program_id: int = 0) -> List[int]:
        """
        Execute one block for one voice.

        Returns: Updated coefficient vector c_q15[N]
        """
        # Get program
        program = self.programs.get(program_id, self.programs[0])

        # Decode input
        inp = self.decode_input(vf, program_id)

        # Get voice state
        state = self.get_voice_state(inp.voice_id)

        # Skip check (§8)
        if self._should_skip(inp, program):
            self.stats["voices_skipped"] += 1
            return state.c_q15

        # §5.2 Ensure morph buffers
        self.ensure_morph_buffers(state, inp)

        # §5.3 Compute effective family enable
        enable_mask, w_eff = self.compute_family_enable(inp, program)

        # §5.5 Apply operator chain to primary buffer
        c_new = self.apply_chain(list(state.c_q15), inp, program, enable_mask, w_eff)

        # Apply to secondary buffer if in morph
        if state.has_secondary and state.cB_q15 is not None:
            cB_new = self.apply_chain(list(state.cB_q15), inp, program, enable_mask, w_eff)
            state.cB_q15 = self.apply_slew(
                VoiceCoeffState(c_prev=state.cB_q15 or [0]*self.coeff_len),
                cB_new, program
            )

        # §5.6 Apply slew
        c_slewed = self.apply_slew(state, c_new, program)

        # Update state
        state.c_q15 = c_slewed

        return c_slewed

    def _should_skip(self, inp: ExecutorInput, program: InstrumentProgram) -> bool:
        """Check if voice should be skipped this block."""
        if inp.activity_q15 < program.activity_min_q15:
            if inp.gain_q15 < program.gain_min_q15:
                return True
        return False

    def step_block(self, frames: List[VoiceFrame],
                   program_id: int = 0) -> List[Tuple[int, List[int]]]:
        """
        Execute one block for all active voices.

        Returns: List of (voice_id, c_q15[N]) tuples
        """
        results = []

        for vf in frames:
            c = self.step_voice(vf, program_id)
            results.append((vf.voice_id, c))

        return results

    def reset_stats(self):
        """Reset debug counters."""
        for key in self.stats:
            self.stats[key] = 0


# =============================================================================
# LEGACY COMPATIBILITY LAYER
# =============================================================================

class SymbolEngine:
    """
    Legacy SymbolEngine interface wrapping the new executor.

    Maintains backwards compatibility with existing code.
    """

    def __init__(self, n_voices: int = 4, coeff_len: int = COEFF_LEN):
        self.n_voices = n_voices
        self.coeff_len = coeff_len

        # New executor
        self.executor = SymbolExecutor(n_voices, coeff_len)

        # Legacy state
        self.voice_gains = [0x4000] * n_voices
        self.voice_pans = [0] * n_voices
        self.voice_trees = [0] * n_voices
        self.voice_materials = [1] * n_voices
        self.active_voices: set = set()

        # Legacy operator engine reference
        self.ops = self.executor.ops

    @property
    def c_q15(self) -> List[List[int]]:
        """Legacy access to coefficient state."""
        result = []
        for i in range(self.n_voices):
            state = self.executor.get_voice_state(i)
            result.append(state.c_q15)
        return result

    def set_voice_program(self, voice_id: int, op_ids: List[int]):
        """Legacy: Set operator program (ignored in new executor)."""
        pass  # New executor uses compiled programs

    def inject_note(self, voice_id: int, note_coeffs: List[int]):
        """Inject note coefficients into a voice."""
        if voice_id < self.n_voices:
            state = self.executor.get_voice_state(voice_id)
            for i in range(min(len(note_coeffs), self.coeff_len)):
                state.c_q15[i] = sat16(state.c_q15[i] + note_coeffs[i])
            self.active_voices.add(voice_id)

    def clear_voice(self, voice_id: int):
        """Clear voice coefficients."""
        if voice_id < self.n_voices:
            state = self.executor.get_voice_state(voice_id)
            state.c_q15 = [0] * self.coeff_len
            state.c_prev = [0] * self.coeff_len
            self.active_voices.discard(voice_id)

    def step_block(self, params: ParamFrame) -> List[VoiceFrame]:
        """Process one block, return VoiceFrames for active voices."""
        frames: List[VoiceFrame] = []

        for vid in list(self.active_voices):
            # Build VoiceFrame for executor
            vf_in = VoiceFrame(
                voice_id=vid,
                tree_id=self.voice_trees[vid],
                tree_a=0.5,
                tree_b=0.5,
                tree_blend=0.0,
                gain_q15=self.voice_gains[vid],
                pan_q15=self.voice_pans[vid],
                material_id=self.voice_materials[vid],
                tilt_env_q15=params.tone_q15,
                density_env_q15=params.gate_q15,
                aux_flags=0,
                aux_param_q15=params.pressure_q15,
            )

            # Execute
            c_out = self.executor.step_voice(vf_in, program_id=0)

            # Build output VoiceFrame
            frames.append(VoiceFrame(
                tag=0xA0,
                voice_id=vid,
                tree_id=self.voice_trees[vid],
                flags=0,
                gain_q15=self.voice_gains[vid],
                pan_q15=self.voice_pans[vid],
                material_id=self.voice_materials[vid],
                role_id=0,
                aux_flags=0,
                aux_param_q15=0,
                coeff_q15=list(c_out)
            ))

        return frames

    def get_voice_energy(self, voice_id: int) -> int:
        """Get total energy of voice (for gating decisions)."""
        if voice_id >= self.n_voices:
            return 0
        state = self.executor.get_voice_state(voice_id)
        energy = sum(c * c for c in state.c_q15)
        return min(Q15_ONE, energy >> 15)


# =============================================================================
# ACCEPTANCE TESTS
# =============================================================================

def test_determinism():
    """Test 1: Repeated runs produce identical results."""
    print("Test 1: Determinism...")

    executor = SymbolExecutor(n_voices=4)

    # Create test VoiceFrame
    vf = VoiceFrame(
        voice_id=0,
        tree_a=0.3,
        tree_b=0.7,
        tree_blend=0.5,
        gain_q15=0x4000,
        tilt_env_q15=0x6000,
        density_env_q15=0x5000,
        material_id=1,
    )

    # Inject some initial coefficients
    state = executor.get_voice_state(0)
    state.c_q15[0] = 16384
    state.c_q15[8] = 8192
    state.c_q15[24] = 4096

    # Run 1
    executor.reset_stats()
    c1 = executor.step_voice(vf, program_id=0)

    # Reset state and run 2
    state = executor.get_voice_state(0)
    state.c_q15 = [0] * COEFF_LEN
    state.c_q15[0] = 16384
    state.c_q15[8] = 8192
    state.c_q15[24] = 4096
    state.c_prev = [0] * COEFF_LEN

    c2 = executor.step_voice(vf, program_id=0)

    # Compare
    match = all(c1[i] == c2[i] for i in range(COEFF_LEN))
    print(f"  Determinism: {'PASS' if match else 'FAIL'}")
    return match


def test_totality():
    """Test 2: No input produces undefined behavior."""
    print("Test 2: Totality...")

    executor = SymbolExecutor(n_voices=4)

    test_cases = [
        # Normal case
        VoiceFrame(voice_id=0),
        # Edge: all zeros
        VoiceFrame(voice_id=1, gain_q15=0, tilt_env_q15=0),
        # Edge: max values
        VoiceFrame(voice_id=2, gain_q15=Q15_ONE, tree_blend=1.0),
        # Edge: invalid material
        VoiceFrame(voice_id=3, material_id=999),
    ]

    all_pass = True
    for i, vf in enumerate(test_cases):
        try:
            c = executor.step_voice(vf, program_id=0)
            if len(c) != COEFF_LEN:
                print(f"  Case {i}: FAIL (wrong length)")
                all_pass = False
            else:
                print(f"  Case {i}: PASS")
        except Exception as e:
            print(f"  Case {i}: FAIL ({e})")
            all_pass = False

    return all_pass


def test_boundedness():
    """Test 3: Coefficients remain bounded after processing."""
    print("Test 3: Boundedness...")

    executor = SymbolExecutor(n_voices=4)

    # Inject extreme values
    state = executor.get_voice_state(0)
    state.c_q15 = [32000] * COEFF_LEN  # Near max

    vf = VoiceFrame(
        voice_id=0,
        gain_q15=Q15_ONE,
        tilt_env_q15=Q15_ONE,
        density_env_q15=Q15_ONE,
        aux_flags=0xFF,  # All events
        aux_param_q15=Q15_ONE,
    )

    c = executor.step_voice(vf, program_id=0)

    max_val = max(abs(x) for x in c)
    bounded = max_val <= 32767
    print(f"  Max coefficient: {max_val}")
    print(f"  Boundedness: {'PASS' if bounded else 'FAIL'}")

    return bounded


def test_degradation():
    """Test 4: Degradation levels drop correct families."""
    print("Test 4: Degradation...")

    executor = SymbolExecutor(n_voices=4)
    program = executor.programs[0]

    test_cases = [
        (DEGRADE_NONE, "Core families enabled"),
        (DEGRADE_LIGHT, "Transport capped"),
        (DEGRADE_HEAVY, "Transport/Topology disabled"),
        (DEGRADE_EMERGENCY, "Only tail stages"),
    ]

    all_pass = True
    for level, desc in test_cases:
        inp = ExecutorInput(
            degradation_level=level,
            w_family=[Q15_ONE] * NUM_FAMILIES,
            regime_id=0,  # Use regime 0 (catastrophe disabled by default)
        )

        enable_mask, w_eff = executor.compute_family_enable(inp, program)

        if level == DEGRADE_NONE:
            # Core families enabled (F1-F5), F7 disabled in regime 0
            core_families = (1 << F_COLOR) | (1 << F_TRANSPORT) | (1 << F_LOCAL) | (1 << F_TOPOLOGY) | (1 << F_GLUE)
            expected = (enable_mask & core_families) == core_families
        elif level == DEGRADE_LIGHT:
            expected = w_eff[F_TRANSPORT] <= int(0.3 * Q15_ONE)
        elif level == DEGRADE_HEAVY:
            expected = (enable_mask & (1 << F_TRANSPORT)) == 0
        else:  # EMERGENCY
            expected = enable_mask == 0

        status = "PASS" if expected else "FAIL"
        print(f"  Level {level} ({desc}): {status} (mask=0x{enable_mask:02X})")
        if not expected:
            all_pass = False

    return all_pass


def test_morph():
    """Test 5: Tree morph sequences don't corrupt state."""
    print("Test 5: Morph correctness...")

    executor = SymbolExecutor(n_voices=4)

    # Inject initial
    state = executor.get_voice_state(0)
    state.c_q15[0] = 16384

    # Sequence: pure A -> morph -> pure B -> back to A
    blends = [0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0.0]

    all_valid = True
    for blend in blends:
        vf = VoiceFrame(
            voice_id=0,
            tree_a=0.3,
            tree_b=0.8,
            tree_blend=blend,
        )

        c = executor.step_voice(vf, program_id=0)

        # Check state consistency
        state = executor.get_voice_state(0)
        if blend > 0 and blend < 1:
            if not state.has_secondary:
                print(f"  Blend {blend}: FAIL (no secondary during morph)")
                all_valid = False

        if len(c) != COEFF_LEN:
            print(f"  Blend {blend}: FAIL (wrong length)")
            all_valid = False

    print(f"  Morph sequence: {'PASS' if all_valid else 'FAIL'}")
    return all_valid


def run_acceptance_tests():
    """Run all acceptance tests."""
    print("=" * 60)
    print("SYMBOL ENGINE EXECUTOR ACCEPTANCE TESTS")
    print("=" * 60)

    results = [
        test_determinism(),
        test_totality(),
        test_boundedness(),
        test_degradation(),
        test_morph(),
    ]

    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)

    return all(results)


if __name__ == "__main__":
    run_acceptance_tests()
