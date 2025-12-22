"""
Safety Engine v1.0
==================

Comprehensive safety layer for the Symcrash audio pipeline.

The Safety Engine is not about preventing bad sounds — it is about guaranteeing
that whatever happens is still interpretable as an instrument.

Architecture
------------
The Tail is a fixed, mandatory end-of-block procedure that runs after all
scheduled operator work and before audio leaves the pipeline.

Two goals:
1. Prevent blow-up / runaway / invalid state in coefficient-space and audio-space
2. Make block boundaries inaudible by default (clicks must be intentional)

The Tail is split across engines:
- Symbol Tail (coefficient-space): projection, clamping, legality enforcement
- Safety Tail (audio-space): OLA, limiting, DC management

NORMATIVE: Both tails MUST exist and are independent; neither may assume the other.

Block Ordering (normative)
--------------------------
For each voice per block:
1. Operator Body (backbone ops + event ops at splice point)
2. Symbol Tail (coefficient-space)
3. Wavelet inverse lift / reconstruction
4. Material Engine
5. Safety Tail (audio-space)

I. Structural Safety (Coefficient-Space Invariants)
---------------------------------------------------
These act BEFORE audio exists:

1. FineClamp - Per-band energy caps (anti-ultrafine blowup)
2. NormalizeScale - Global normalization toward perceptual target
3. SheafProject - Topological consistency across scales
4. GluePair - Local coherence constraints

II. Behavioral Safety (State & Time Invariants)
-----------------------------------------------
These protect the instrument as a process:

5. Regime hysteresis & dwell guarantees
6. Catastrophe gating & corridor control
7. Recovery regimes (mandatory)

III. Execution Safety (Determinism & Scheduling)
------------------------------------------------
These protect time, order, and fairness:

8. Mandatory tail execution (non-skippable)
9. Deterministic degradation policy
10. State freeze semantics

IV. Numerical Safety (Fixed-Point Hygiene)
------------------------------------------
These protect math integrity:

11. Fixed-point saturation & rounding discipline (WSE-STD-001)
12. Deterministic PRNG draw order

V. Audio-Plane Safety (Last Line of Defense)
--------------------------------------------
These protect ears and speakers:

13. OLA / crossfade invisibility
14. Final amplitude limiting

VI. Meta-Safety (Human Factors)
-------------------------------
15. Bounded performance macros
16. Editor vs performance separation

Symbol Tail Invariants (coefficient-space)
------------------------------------------
Let leaves be L=32, each leaf has 8 coefficients.
Define per-leaf energy: E_ℓ = Σⱼ c²_{ℓ,j}
Define total energy: E = Σ_ℓ E_ℓ
Define fine band set: F ⊆ [0,31] (default: {28,29,30,31})

S1: Finite coefficients - all int16 after saturation
S2: Per-leaf cap - E_ℓ ≤ E_{ℓ,max}
S3: Fine-band cap - Σ_{ℓ∈F} E_ℓ ≤ E_{F,max}
S4: Global energy cap - E ≤ E_max
S5: Bounded spectral tilt - E_F/E_C ∈ [R_min, R_max]

Safety Tail Invariants (audio-space)
------------------------------------
A1: Peak bound - max_n |x[n]| ≤ P_max
A2: Slew bound - max_n |x[n] - x[n-1]| ≤ S_max
A3: DC bound - |μ| ≤ D_max where μ = (1/64)Σx[n]

Two Kinds of Clicks
-------------------
- Boundary click = bug (Safety OLA eats it)
- Interior click = effect (survives - Impulse operator etc.)

Degrade Priority Order (normative)
----------------------------------
1. Reduce LowRankMorph rank
2. Drop Diffuse ops effect
3. Zero injections (Impulse/NoiseBand/GlobalPulse)
4. Apply EmergencySimplify (zero fine leaves + re-normalize)

Current implementation:
- Metrics measurement (RMS, peak, HF, DC)
- Regulation (RG smoothed tracking)
- Q projection (gain/filter adjustment)
- De-click/OLA (block boundary smoothing)
- Limiter (peak protection)
- FineClamp (coefficient-space)
- NormalizeScale (coefficient-space)
- SlewLimit (audio-space)
- SAFE predicate verification
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import IntEnum
import math

from fixed import (
    sat16, mul_q15_sat, shift_round, slew_q15, slew_q15_asym,
    sqrt_q15, soft_clip_q15,
    Q15_ONE, Q15_SHIFT, Q15_HALF
)
from frames import ParamFrame, Metrics


# =============================================================================
# SYMBOL TAIL CONSTANTS (coefficient-space)
# =============================================================================

LEAF_COUNT = 32           # L = 32 leaves per tree
COEFF_PER_LEAF = 8        # 8 coefficients per leaf
COEFF_LEN = 256           # Total coefficient vector length

# Fine band set (highest frequency leaves, prone to ultrafine blowup)
FINE_BAND_SET = {28, 29, 30, 31}  # Default: last 4 leaves
COARSE_BAND_SET = set(range(0, 28))  # Everything else

# Energy caps (Q30 for squared values before normalization)
E_LEAF_MAX = 0x40000000       # Per-leaf energy cap (Q30)
E_FINE_MAX = 0x30000000       # Fine-band total energy cap (Q30)
E_GLOBAL_MAX = 0x7FFFFFFF     # Global energy cap (max s32)

# Spectral tilt bounds (fine/coarse ratio limits)
TILT_R_MIN = 0x0100           # Minimum fine/coarse ratio (avoid all-bass)
TILT_R_MAX = 0x4000           # Maximum fine/coarse ratio (avoid all-hiss)
TILT_EPSILON = 0x0100         # Small constant to avoid divide-by-zero

# Normalization targets (Q15)
NORMALIZE_TARGET_PINK = 0x4000   # Pink noise target (~-6dB)
NORMALIZE_TARGET_WHITE = 0x5000  # White noise target (~-4dB)
NORMALIZE_TARGET_BROWN = 0x3000  # Brown noise target (~-8dB)


# =============================================================================
# SAFETY TAIL CONSTANTS (audio-space)
# =============================================================================

FADE_LEN = 8          # Samples for fade in/out (power of 2)
OLA_OVERLAP = 8       # Samples of overlap-add per block edge

# Audio-space invariant thresholds
PEAK_MAX = 0x7F00         # A1: Peak bound (~0.996 of full scale)
SLEW_MAX = 0x2000         # A2: Maximum sample-to-sample change
DC_MAX = 0x0800           # A3: Maximum DC offset allowed

# Degrade levels
class DegradeLevel(IntEnum):
    """Degradation levels for CPU overload handling."""
    NONE = 0              # Full processing
    REDUCE_RANK = 1       # Reduce LowRankMorph rank
    DROP_DIFFUSE = 2      # Disable diffuse operators
    ZERO_INJECT = 3       # Zero injections
    EMERGENCY = 4         # Emergency simplify (zero fine leaves)

# Voice phases
VOICE_INACTIVE = 0
VOICE_STARTING = 1
VOICE_RUNNING = 2
VOICE_STOPPING = 3


# =============================================================================
# HANN WINDOW LUT (precomputed for FPGA compatibility)
# =============================================================================

def _build_hann_lut(n: int) -> List[int]:
    """Build Hann half-window LUT: 0.5 * (1 - cos(pi * i / n))"""
    lut = []
    for i in range(n + 1):
        # Raised cosine: 0 at i=0, 1 at i=n
        val = 0.5 * (1.0 - math.cos(math.pi * i / n))
        lut.append(int(round(val * Q15_ONE)))
    return lut

# Precomputed Hann window for FADE_LEN samples
_HANN_FADE_LUT = _build_hann_lut(FADE_LEN)
_HANN_OLA_LUT = _build_hann_lut(OLA_OVERLAP)


def hann_fade_in(n: int, fade_len: int = FADE_LEN) -> int:
    """Returns Q1.15 gain for sample n of fade-in (0..fade_len)."""
    if n <= 0:
        return 0
    if n >= fade_len:
        return Q15_ONE
    if fade_len == FADE_LEN:
        return _HANN_FADE_LUT[n]
    # Fallback for non-standard lengths
    val = 0.5 * (1.0 - math.cos(math.pi * n / fade_len))
    return int(round(val * Q15_ONE))


def hann_fade_out(n: int, fade_len: int = FADE_LEN) -> int:
    """Returns Q1.15 gain for sample n of fade-out (0..fade_len)."""
    return hann_fade_in(fade_len - n, fade_len)


# =============================================================================
# OLA STATE
# =============================================================================

@dataclass
class OLAState:
    """Per-voice OLA state for de-clicking."""
    prev_tail: List[int] = field(default_factory=list)  # Last OLA_OVERLAP samples
    voice_phase: int = VOICE_INACTIVE  # 0=inactive, 1=starting, 2=running, 3=stopping
    fade_pos: int = 0  # Position within fade (0..FADE_LEN)


@dataclass
class RGState:
    """Regulation state (smoothed metrics)."""
    E_hat: int = 0       # Smoothed RMS energy
    HF_hat: int = 0      # Smoothed HF energy
    peak_hat: int = 0    # Smoothed peak
    dc_hat: int = 0      # Smoothed DC offset


@dataclass
class LimiterState:
    """Limiter state."""
    gain: int = Q15_ONE  # Current gain (Q1.15)
    hold_count: int = 0  # Hold counter


@dataclass
class CoeffInvariantResult:
    """Result of checking coefficient-space invariants."""
    s1_finite: bool = True      # All coefficients finite (int16)
    s2_leaf_caps: bool = True   # Per-leaf energy caps satisfied
    s3_fine_cap: bool = True    # Fine-band cap satisfied
    s4_global_cap: bool = True  # Global energy cap satisfied
    s5_tilt_ok: bool = True     # Spectral tilt in bounds

    # Energy values for diagnostics
    total_energy: int = 0
    fine_energy: int = 0
    coarse_energy: int = 0
    max_leaf_energy: int = 0
    violating_leaf: int = -1

    @property
    def is_safe(self) -> bool:
        """True if all coefficient invariants hold."""
        return (self.s1_finite and self.s2_leaf_caps and
                self.s3_fine_cap and self.s4_global_cap and self.s5_tilt_ok)


@dataclass
class AudioInvariantResult:
    """Result of checking audio-space invariants."""
    a1_peak_ok: bool = True     # Peak within bounds
    a2_slew_ok: bool = True     # Slew rate within bounds
    a3_dc_ok: bool = True       # DC offset within bounds

    # Values for diagnostics
    peak: int = 0
    max_slew: int = 0
    dc_offset: int = 0

    @property
    def is_safe(self) -> bool:
        """True if all audio invariants hold."""
        return self.a1_peak_ok and self.a2_slew_ok and self.a3_dc_ok


# =============================================================================
# SYMBOL TAIL (coefficient-space safety)
# =============================================================================

class SymbolTail:
    """
    Coefficient-space safety tail.

    Runs at end of Symbol Engine step to enforce invariants S1-S5.

    Tail operator sequence (normative, fixed order):
    1. FineClamp - anti-ultrafine cap
    2. NormalizeScale - normalize to target distribution
    3. SheafProject - local-to-global consistency (placeholder)
    4. GluePair - glue constraints (placeholder)
    5. HardClamp - final saturate to int16
    6. EmergencySimplify - conditional, if still unsafe
    """

    def __init__(self,
                 normalize_curve: str = "pink",
                 fine_band_set: set = None,
                 e_leaf_max: int = E_LEAF_MAX,
                 e_fine_max: int = E_FINE_MAX,
                 e_global_max: int = E_GLOBAL_MAX):
        """
        Initialize Symbol Tail.

        Args:
            normalize_curve: "pink", "white", or "brown"
            fine_band_set: Set of leaf indices considered "fine" (default: {28-31})
            e_leaf_max: Per-leaf energy cap
            e_fine_max: Fine-band total energy cap
            e_global_max: Global energy cap
        """
        self.normalize_curve = normalize_curve
        self.fine_band_set = fine_band_set if fine_band_set else FINE_BAND_SET
        self.e_leaf_max = e_leaf_max
        self.e_fine_max = e_fine_max
        self.e_global_max = e_global_max

        # Normalization target based on curve
        self.normalize_target = {
            "pink": NORMALIZE_TARGET_PINK,
            "white": NORMALIZE_TARGET_WHITE,
            "brown": NORMALIZE_TARGET_BROWN,
        }.get(normalize_curve, NORMALIZE_TARGET_PINK)

    def compute_leaf_energy(self, coeffs: List[int], leaf: int) -> int:
        """Compute energy for a single leaf (sum of squared coefficients)."""
        start = leaf * COEFF_PER_LEAF
        end = start + COEFF_PER_LEAF
        energy = 0
        for i in range(start, min(end, len(coeffs))):
            energy += coeffs[i] * coeffs[i]
        return energy

    def check_invariants(self, coeffs: List[int]) -> CoeffInvariantResult:
        """
        Check all coefficient-space invariants S1-S5.

        Returns CoeffInvariantResult with detailed status.
        """
        result = CoeffInvariantResult()

        if len(coeffs) < COEFF_LEN:
            # Pad with zeros if short
            coeffs = list(coeffs) + [0] * (COEFF_LEN - len(coeffs))

        # S1: Finite coefficients (all representable as int16)
        for c in coeffs:
            if c > 32767 or c < -32768:
                result.s1_finite = False
                break

        # Compute per-leaf energies
        leaf_energies = []
        for leaf in range(LEAF_COUNT):
            e = self.compute_leaf_energy(coeffs, leaf)
            leaf_energies.append(e)

        # S2: Per-leaf caps
        result.max_leaf_energy = 0
        for leaf, e in enumerate(leaf_energies):
            if e > result.max_leaf_energy:
                result.max_leaf_energy = e
                result.violating_leaf = leaf
            if e > self.e_leaf_max:
                result.s2_leaf_caps = False

        # S3: Fine-band cap
        result.fine_energy = sum(leaf_energies[leaf] for leaf in self.fine_band_set
                                  if leaf < len(leaf_energies))
        if result.fine_energy > self.e_fine_max:
            result.s3_fine_cap = False

        # S4: Global energy cap
        result.total_energy = sum(leaf_energies)
        if result.total_energy > self.e_global_max:
            result.s4_global_cap = False

        # S5: Bounded spectral tilt
        result.coarse_energy = sum(leaf_energies[leaf] for leaf in COARSE_BAND_SET
                                    if leaf < len(leaf_energies))
        if result.coarse_energy + TILT_EPSILON > 0:
            tilt_ratio = ((result.fine_energy + TILT_EPSILON) << 15) // \
                         (result.coarse_energy + TILT_EPSILON)
            if tilt_ratio < TILT_R_MIN or tilt_ratio > TILT_R_MAX:
                result.s5_tilt_ok = False

        return result

    def fine_clamp(self, coeffs: List[int]) -> List[int]:
        """
        FineClamp: Cap per-leaf and fine-band energy.

        Protects against ultrafine coefficient blowup.
        """
        out = list(coeffs)
        if len(out) < COEFF_LEN:
            out.extend([0] * (COEFF_LEN - len(out)))

        # Check and clamp each leaf
        for leaf in range(LEAF_COUNT):
            e = self.compute_leaf_energy(out, leaf)
            if e > self.e_leaf_max and e > 0:
                # Scale down this leaf
                scale = int(math.sqrt(self.e_leaf_max / e) * Q15_ONE)
                start = leaf * COEFF_PER_LEAF
                for i in range(COEFF_PER_LEAF):
                    out[start + i] = mul_q15_sat(out[start + i], scale)

        # Check and clamp fine bands collectively
        fine_energy = sum(self.compute_leaf_energy(out, leaf) for leaf in self.fine_band_set)
        if fine_energy > self.e_fine_max and fine_energy > 0:
            scale = int(math.sqrt(self.e_fine_max / fine_energy) * Q15_ONE)
            for leaf in self.fine_band_set:
                start = leaf * COEFF_PER_LEAF
                for i in range(COEFF_PER_LEAF):
                    if start + i < len(out):
                        out[start + i] = mul_q15_sat(out[start + i], scale)

        return out

    def normalize_scale(self, coeffs: List[int]) -> List[int]:
        """
        NormalizeScale: Normalize coefficients toward target distribution.

        Rescales to match perceptual target curve while preserving relative structure.
        """
        out = list(coeffs)
        if len(out) < COEFF_LEN:
            out.extend([0] * (COEFF_LEN - len(out)))

        # Compute current RMS energy
        total_energy = sum(c * c for c in out)
        if total_energy == 0:
            return out

        # Compute current RMS (approximate)
        current_rms = int(math.sqrt(total_energy / COEFF_LEN))

        # Scale to target
        if current_rms > 0:
            scale = (self.normalize_target << 15) // current_rms
            scale = min(Q15_ONE * 2, max(Q15_ONE // 4, scale))  # Limit scale range
            for i in range(len(out)):
                out[i] = mul_q15_sat(out[i], scale)

        return out

    def hard_clamp(self, coeffs: List[int]) -> List[int]:
        """
        HardClamp: Final saturation to int16 range.

        Ensures all coefficients are representable.
        """
        return [sat16(c) for c in coeffs]

    def emergency_simplify(self, coeffs: List[int], k: int = 4) -> List[int]:
        """
        EmergencySimplify: Zero top-k fine leaves and re-normalize.

        Used when other measures fail to achieve safety.
        """
        out = list(coeffs)
        if len(out) < COEFF_LEN:
            out.extend([0] * (COEFF_LEN - len(out)))

        # Zero the finest k leaves
        fine_leaves = sorted(self.fine_band_set, reverse=True)[:k]
        for leaf in fine_leaves:
            start = leaf * COEFF_PER_LEAF
            for i in range(COEFF_PER_LEAF):
                if start + i < len(out):
                    out[start + i] = 0

        # Re-normalize
        out = self.normalize_scale(out)
        return out

    def process(self, coeffs: List[int], allow_emergency: bool = True) -> Tuple[List[int], CoeffInvariantResult]:
        """
        Run full Symbol Tail processing.

        Sequence: FineClamp -> NormalizeScale -> HardClamp -> FinalFineClamp -> (EmergencySimplify if needed)

        Returns: (processed coefficients, invariant check result)
        """
        # 1. FineClamp (first pass - catch extreme inputs)
        c = self.fine_clamp(coeffs)

        # 2. NormalizeScale
        c = self.normalize_scale(c)

        # 3. HardClamp
        c = self.hard_clamp(c)

        # 4. FineClamp again (normalization may have pushed energy back up)
        c = self.fine_clamp(c)
        c = self.hard_clamp(c)

        # Check invariants
        result = self.check_invariants(c)

        # 5. EmergencySimplify if still unsafe (but not for tilt-only issues)
        if not result.is_safe and allow_emergency:
            # Only apply emergency if we have actual energy/cap violations
            if not (result.s1_finite and result.s2_leaf_caps and
                    result.s3_fine_cap and result.s4_global_cap):
                c = self.emergency_simplify(c)
                c = self.hard_clamp(c)
                result = self.check_invariants(c)

        return c, result


# =============================================================================
# SAFE PREDICATE
# =============================================================================

def check_coeff_safe(coeffs: List[int], symbol_tail: SymbolTail = None) -> CoeffInvariantResult:
    """
    Check if coefficients satisfy all invariants S1-S5.

    The SAFE predicate for coefficient-space.
    """
    if symbol_tail is None:
        symbol_tail = SymbolTail()
    return symbol_tail.check_invariants(coeffs)


def check_audio_safe(audio: List[int], prev_sample: int = 0) -> AudioInvariantResult:
    """
    Check if audio block satisfies all invariants A1-A3.

    The SAFE predicate for audio-space.

    Args:
        audio: Audio samples for this block
        prev_sample: Last sample from previous block (for slew check)
    """
    result = AudioInvariantResult()

    if not audio:
        return result

    # A1: Peak bound
    result.peak = max(abs(s) for s in audio)
    result.a1_peak_ok = result.peak <= PEAK_MAX

    # A2: Slew bound (max sample-to-sample change)
    result.max_slew = 0
    prev = prev_sample
    for s in audio:
        slew = abs(s - prev)
        if slew > result.max_slew:
            result.max_slew = slew
        prev = s
    result.a2_slew_ok = result.max_slew <= SLEW_MAX

    # A3: DC bound
    result.dc_offset = sum(audio) // len(audio) if audio else 0
    result.a3_dc_ok = abs(result.dc_offset) <= DC_MAX

    return result


def is_block_safe(coeffs: List[int], audio: List[int],
                  prev_sample: int = 0,
                  symbol_tail: SymbolTail = None) -> bool:
    """
    Check if a complete block is SAFE.

    NORMATIVE: System MUST NOT emit audio unless SAFE is satisfied.

    Args:
        coeffs: Coefficient vector (256 values)
        audio: Audio samples (64 values)
        prev_sample: Last sample from previous block
        symbol_tail: SymbolTail instance for coefficient checks

    Returns:
        True if both coefficient and audio invariants hold
    """
    coeff_result = check_coeff_safe(coeffs, symbol_tail)
    audio_result = check_audio_safe(audio, prev_sample)
    return coeff_result.is_safe and audio_result.is_safe


class SafetyEngine:
    """
    Safety processing: metrics, regulation, de-click/OLA, limiting, slew control.

    This is the audio-space safety layer (Safety Tail).

    Processing chain:
        Input -> Metrics -> RG -> Q -> De-click/OLA -> Slew limit -> Limiter -> DC removal -> Output

    Audio Invariants Enforced:
        A1: Peak bound - max_n |x[n]| ≤ P_max
        A2: Slew bound - max_n |x[n] - x[n-1]| ≤ S_max
        A3: DC bound - |μ| ≤ D_max

    For coefficient-space safety, use SymbolTail.
    """

    # Target levels by K_mode
    E_TARGETS = [0x4000, 0x5000, 0x6000, 0x7000]  # Q1.15 RMS targets
    HF_BUDGETS = [0x2000, 0x3000, 0x4000, 0x5000]  # HF energy budgets

    # Limiter thresholds
    LIMIT_THRESHOLD = 0x7000  # ~0.875
    LIMIT_CEILING = 0x7F00    # ~0.996

    # Attack/release (samples per block)
    ATTACK_RATE = 2048   # Fast attack
    RELEASE_RATE = 64    # Slow release

    def __init__(self, block_size: int = 64, normalize_curve: str = "pink"):
        """
        Initialize Safety Engine.

        Args:
            block_size: Samples per block (default 64)
            normalize_curve: Normalization curve for SymbolTail ("pink", "white", "brown")
        """
        self.B = block_size

        # Regulation state (smoothed)
        self.rg = RGState()

        # Limiter state
        self.lim_L = LimiterState()
        self.lim_R = LimiterState()

        # Q state (slewed gains)
        self.q_gain = Q15_ONE
        self.q_hf_cut = Q15_ONE

        # Smoothing window
        self.W = 16  # Blocks for RG averaging
        self.alpha = Q15_ONE // self.W

        # OLA state per voice
        self.ola_state: Dict[int, OLAState] = {}

        # Master OLA state (for mixed stereo output)
        self.master_ola_L: List[int] = []
        self.master_ola_R: List[int] = []

        # Slew state (A2 invariant)
        self.prev_sample_L: int = 0
        self.prev_sample_R: int = 0

        # Symbol Tail for coefficient-space safety
        self.symbol_tail = SymbolTail(normalize_curve=normalize_curve)

        # Current degrade level
        self.degrade_level: DegradeLevel = DegradeLevel.NONE

    # =========================================================================
    # OLA STATE MANAGEMENT
    # =========================================================================

    def get_ola_state(self, voice_id: int) -> OLAState:
        """Get or create OLA state for a voice."""
        if voice_id not in self.ola_state:
            self.ola_state[voice_id] = OLAState()
        return self.ola_state[voice_id]

    def voice_on(self, voice_id: int):
        """Signal voice start - triggers fade-in."""
        state = self.get_ola_state(voice_id)
        state.voice_phase = VOICE_STARTING
        state.fade_pos = 0
        state.prev_tail = []

    def voice_off(self, voice_id: int):
        """Signal voice stop - triggers fade-out."""
        state = self.get_ola_state(voice_id)
        if state.voice_phase == VOICE_RUNNING:
            state.voice_phase = VOICE_STOPPING
            state.fade_pos = 0

    def is_voice_active(self, voice_id: int) -> bool:
        """Check if voice is active (not inactive)."""
        if voice_id not in self.ola_state:
            return False
        return self.ola_state[voice_id].voice_phase != VOICE_INACTIVE

    # =========================================================================
    # VOICE START/STOP DE-CLICK
    # =========================================================================

    def apply_voice_start(self, x: List[int], voice_id: int) -> List[int]:
        """Apply fade-in for voice start."""
        state = self.get_ola_state(voice_id)

        if state.voice_phase != VOICE_STARTING:
            return x

        out = list(x)
        samples_to_fade = min(FADE_LEN - state.fade_pos, len(x))

        for i in range(samples_to_fade):
            gain = hann_fade_in(state.fade_pos + i, FADE_LEN)
            out[i] = mul_q15_sat(x[i], gain)

        state.fade_pos += len(x)
        if state.fade_pos >= FADE_LEN:
            state.voice_phase = VOICE_RUNNING
            state.fade_pos = 0

        return out

    def apply_voice_stop(self, x: List[int], voice_id: int) -> List[int]:
        """Apply fade-out for voice stop."""
        state = self.get_ola_state(voice_id)

        if state.voice_phase != VOICE_STOPPING:
            return x

        out = list(x)
        samples_to_fade = min(FADE_LEN - state.fade_pos, len(x))

        for i in range(samples_to_fade):
            gain = hann_fade_out(state.fade_pos + i, FADE_LEN)
            out[i] = mul_q15_sat(x[i], gain)

        # Zero remaining samples after fade completes
        if state.fade_pos + len(x) >= FADE_LEN:
            for i in range(samples_to_fade, len(x)):
                out[i] = 0

        state.fade_pos += len(x)
        if state.fade_pos >= FADE_LEN:
            state.voice_phase = VOICE_INACTIVE
            state.fade_pos = 0
            state.prev_tail = []

        return out

    # =========================================================================
    # BLOCK BOUNDARY OLA
    # =========================================================================

    def apply_ola(self, x: List[int], voice_id: int) -> List[int]:
        """Overlap-add for block boundary smoothing."""
        state = self.get_ola_state(voice_id)

        # Skip if voice is inactive or just starting
        if state.voice_phase == VOICE_INACTIVE:
            return x

        if not state.prev_tail or len(state.prev_tail) < OLA_OVERLAP:
            # First block or insufficient tail - just save tail
            state.prev_tail = list(x[-OLA_OVERLAP:])
            return x

        out = list(x)

        # Crossfade at block start: blend previous tail with current start
        for i in range(OLA_OVERLAP):
            fade_out_gain = hann_fade_out(i, OLA_OVERLAP)
            fade_in_gain = hann_fade_in(i, OLA_OVERLAP)

            prev_contrib = mul_q15_sat(state.prev_tail[i], fade_out_gain)
            curr_contrib = mul_q15_sat(x[i], fade_in_gain)
            out[i] = sat16(prev_contrib + curr_contrib)

        # Save tail for next block
        state.prev_tail = list(x[-OLA_OVERLAP:])

        return out

    def apply_declick(self, x: List[int], voice_id: int) -> List[int]:
        """Full de-click chain for a voice block."""
        x = self.apply_voice_start(x, voice_id)
        x = self.apply_voice_stop(x, voice_id)
        x = self.apply_ola(x, voice_id)
        return x

    # =========================================================================
    # MASTER OLA (for mixed stereo output)
    # =========================================================================

    def apply_master_ola(self, L: List[int], R: List[int]) -> Tuple[List[int], List[int]]:
        """Apply OLA to mixed stereo output (block boundary smoothing)."""
        if not self.master_ola_L or len(self.master_ola_L) < OLA_OVERLAP:
            # First block - just save tail
            self.master_ola_L = list(L[-OLA_OVERLAP:])
            self.master_ola_R = list(R[-OLA_OVERLAP:])
            return L, R

        outL = list(L)
        outR = list(R)

        # Crossfade at block start
        for i in range(OLA_OVERLAP):
            fade_out_gain = hann_fade_out(i, OLA_OVERLAP)
            fade_in_gain = hann_fade_in(i, OLA_OVERLAP)

            # Left channel
            prev_L = mul_q15_sat(self.master_ola_L[i], fade_out_gain)
            curr_L = mul_q15_sat(L[i], fade_in_gain)
            outL[i] = sat16(prev_L + curr_L)

            # Right channel
            prev_R = mul_q15_sat(self.master_ola_R[i], fade_out_gain)
            curr_R = mul_q15_sat(R[i], fade_in_gain)
            outR[i] = sat16(prev_R + curr_R)

        # Save tail for next block
        self.master_ola_L = list(L[-OLA_OVERLAP:])
        self.master_ola_R = list(R[-OLA_OVERLAP:])

        return outL, outR

    # =========================================================================
    # METRICS
    # =========================================================================

    def measure_metrics(self, L: List[int], R: List[int]) -> Metrics:
        """Measure block metrics."""
        # Peak
        peak = 0
        for i in range(self.B):
            peak = max(peak, abs(L[i]), abs(R[i]))

        # RMS energy (sum of squares, scaled)
        energy = 0
        for i in range(self.B):
            energy += (L[i] * L[i] + R[i] * R[i]) >> 16
        rms = sqrt_q15(min(Q15_ONE, energy // (2 * self.B)))

        # HF energy (simple: sum of sample differences)
        hf = 0
        for i in range(1, self.B):
            dL = L[i] - L[i-1]
            dR = R[i] - R[i-1]
            hf += (dL * dL + dR * dR) >> 18
        hf_energy = min(Q15_ONE, hf // self.B)

        # DC offset (average)
        dc = sum(L) + sum(R)
        dc_offset = sat16(dc // (2 * self.B))

        # Crest factor (peak / rms)
        if rms > 0:
            crest = (peak << 15) // rms
            crest = min(Q15_ONE, crest)
        else:
            crest = Q15_ONE

        return Metrics(
            rms_q15=rms,
            peak_q15=peak,
            hf_energy_q15=hf_energy,
            dc_offset_q15=dc_offset,
            crest_factor_q15=crest
        )

    def update_rg(self, m: Metrics, K_mode: int):
        """Update regulation state with smoothed metrics."""
        # Exponential smoothing: hat = alpha * new + (1-alpha) * hat
        alpha = self.alpha
        alpha_inv = Q15_ONE - alpha

        self.rg.E_hat = shift_round(
            alpha * m.rms_q15 + alpha_inv * self.rg.E_hat,
            Q15_SHIFT
        )
        self.rg.HF_hat = shift_round(
            alpha * m.hf_energy_q15 + alpha_inv * self.rg.HF_hat,
            Q15_SHIFT
        )
        self.rg.peak_hat = shift_round(
            alpha * m.peak_q15 + alpha_inv * self.rg.peak_hat,
            Q15_SHIFT
        )
        self.rg.dc_hat = shift_round(
            alpha * m.dc_offset_q15 + alpha_inv * self.rg.dc_hat,
            Q15_SHIFT
        )

    # =========================================================================
    # Q PROJECTION
    # =========================================================================

    def compute_q_targets(self, K_mode: int) -> Tuple[int, int]:
        """Compute Q gain and HF cut targets based on RG state."""
        E_target = self.E_TARGETS[min(K_mode, 3)]
        HF_budget = self.HF_BUDGETS[min(K_mode, 3)]

        # Gain target: scale to bring RMS toward target
        if self.rg.E_hat > 0:
            gain_target = (E_target << 15) // self.rg.E_hat
            gain_target = min(Q15_ONE, gain_target)
        else:
            gain_target = Q15_ONE

        # HF cut target: reduce if HF over budget
        if self.rg.HF_hat > HF_budget:
            hf_cut_target = (HF_budget << 15) // self.rg.HF_hat
            hf_cut_target = max(Q15_HALF, min(Q15_ONE, hf_cut_target))
        else:
            hf_cut_target = Q15_ONE

        return gain_target, hf_cut_target

    def apply_q(self, L: List[int], R: List[int], K_mode: int) -> Tuple[List[int], List[int]]:
        """Apply Q projection (gain + HF control)."""
        gain_target, hf_cut_target = self.compute_q_targets(K_mode)

        # Slew gains
        self.q_gain = slew_q15(self.q_gain, gain_target, 256)
        self.q_hf_cut = slew_q15(self.q_hf_cut, hf_cut_target, 128)

        outL = []
        outR = []

        # Simple HF reduction: blend with previous sample
        blend = Q15_ONE - self.q_hf_cut

        for i in range(self.B):
            # Apply gain
            yL = mul_q15_sat(L[i], self.q_gain)
            yR = mul_q15_sat(R[i], self.q_gain)

            # HF reduction (simple lowpass blend)
            if i > 0 and blend > 0:
                yL = sat16(mul_q15_sat(yL, self.q_hf_cut) +
                          mul_q15_sat(outL[i-1], blend))
                yR = sat16(mul_q15_sat(yR, self.q_hf_cut) +
                          mul_q15_sat(outR[i-1], blend))

            outL.append(yL)
            outR.append(yR)

        return outL, outR

    # =========================================================================
    # LIMITER
    # =========================================================================

    def apply_limiter(self, L: List[int], R: List[int]) -> Tuple[List[int], List[int]]:
        """Apply brick-wall limiter with attack/release."""
        outL = []
        outR = []

        for i in range(self.B):
            # Find peak
            peak = max(abs(L[i]), abs(R[i]))

            # Compute target gain
            if peak > self.LIMIT_THRESHOLD:
                # Gain to bring peak to ceiling
                target_gain = (self.LIMIT_CEILING << 15) // max(1, peak)
                target_gain = min(Q15_ONE, target_gain)
            else:
                target_gain = Q15_ONE

            # Asymmetric slew (fast attack, slow release)
            self.lim_L.gain = slew_q15_asym(
                self.lim_L.gain, target_gain,
                self.ATTACK_RATE, self.RELEASE_RATE
            )

            # Apply gain
            yL = mul_q15_sat(L[i], self.lim_L.gain)
            yR = mul_q15_sat(R[i], self.lim_L.gain)

            # Final hard clip (should rarely trigger)
            yL = sat16(yL)
            yR = sat16(yR)

            outL.append(yL)
            outR.append(yR)

        return outL, outR

    def remove_dc(self, L: List[int], R: List[int]) -> Tuple[List[int], List[int]]:
        """Remove DC offset."""
        if abs(self.rg.dc_hat) < 100:
            return L, R

        dc = self.rg.dc_hat
        outL = [sat16(x - dc) for x in L]
        outR = [sat16(x - dc) for x in R]

        return outL, outR

    # =========================================================================
    # SLEW LIMITER (A2 invariant enforcement)
    # =========================================================================

    def apply_slew_limit(self, L: List[int], R: List[int],
                         prev_L: int = 0, prev_R: int = 0,
                         max_slew: int = SLEW_MAX) -> Tuple[List[int], List[int], int, int]:
        """
        Apply slew rate limiting to prevent clicks and sudden jumps.

        Invariant A2: max_n |x[n] - x[n-1]| ≤ S_max

        Returns: (limited_L, limited_R, last_L, last_R)
        """
        outL = []
        outR = []
        pL = prev_L
        pR = prev_R

        for i in range(len(L)):
            # Left channel
            delta_L = L[i] - pL
            if delta_L > max_slew:
                sL = pL + max_slew
            elif delta_L < -max_slew:
                sL = pL - max_slew
            else:
                sL = L[i]
            outL.append(sL)
            pL = sL

            # Right channel
            delta_R = R[i] - pR
            if delta_R > max_slew:
                sR = pR + max_slew
            elif delta_R < -max_slew:
                sR = pR - max_slew
            else:
                sR = R[i]
            outR.append(sR)
            pR = sR

        return outL, outR, pL, pR

    # =========================================================================
    # SAFE PREDICATE CHECK
    # =========================================================================

    def check_audio_safety(self, L: List[int], R: List[int]) -> Tuple[AudioInvariantResult, AudioInvariantResult]:
        """
        Check audio invariants A1-A3 for both channels.

        Returns: (left_result, right_result)
        """
        result_L = check_audio_safe(L, self.prev_sample_L if hasattr(self, 'prev_sample_L') else 0)
        result_R = check_audio_safe(R, self.prev_sample_R if hasattr(self, 'prev_sample_R') else 0)
        return result_L, result_R

    # =========================================================================
    # MAIN PROCESSING CHAIN
    # =========================================================================

    def step_block(self, L: List[int], R: List[int],
                   params: ParamFrame,
                   apply_master_ola: bool = True,
                   apply_slew_limit: bool = True) -> Tuple[List[int], List[int]]:
        """
        Full safety processing chain (Safety Tail).

        Chain: Metrics -> RG -> Q -> Master OLA -> Slew Limit -> Limiter -> DC removal

        For per-voice de-click, call apply_declick() on each voice block
        BEFORE mixing to stereo.

        Enforces invariants:
        - A1: Peak bound (via limiter)
        - A2: Slew bound (via slew limiter)
        - A3: DC bound (via DC removal)
        """
        # 1) Measure metrics
        m = self.measure_metrics(L, R)

        # 2) Update RG (smoothed tracking)
        self.update_rg(m, params.K_mode)

        # 3) Apply Q (gain + HF control)
        Lq, Rq = self.apply_q(L, R, params.K_mode)

        # 4) Apply master OLA (block boundary smoothing on mixed output)
        if apply_master_ola:
            Lq, Rq = self.apply_master_ola(Lq, Rq)

        # 5) Apply slew limit (A2 invariant)
        if apply_slew_limit:
            Lq, Rq, self.prev_sample_L, self.prev_sample_R = self.apply_slew_limit(
                Lq, Rq, self.prev_sample_L, self.prev_sample_R
            )

        # 6) Apply limiter (A1 invariant)
        Lo, Ro = self.apply_limiter(Lq, Rq)

        # 7) Remove DC (A3 invariant)
        Lo, Ro = self.remove_dc(Lo, Ro)

        return Lo, Ro

    def step_block_with_voices(self, voice_blocks: List[Tuple[int, List[int], List[int]]],
                                params: ParamFrame) -> Tuple[List[int], List[int]]:
        """
        Process with per-voice de-clicking before mixing.

        voice_blocks: List of (voice_id, L, R) tuples

        This applies per-voice de-click, then mixes, then runs the rest of the chain.
        """
        # Apply per-voice de-click
        declicked_blocks = []
        for voice_id, vL, vR in voice_blocks:
            dL = self.apply_declick(vL, voice_id)
            dR = self.apply_declick(vR, voice_id)
            declicked_blocks.append((dL, dR))

        # Mix to stereo
        mix_L = [0] * self.B
        mix_R = [0] * self.B

        for dL, dR in declicked_blocks:
            for i in range(self.B):
                mix_L[i] = sat16(mix_L[i] + dL[i])
                mix_R[i] = sat16(mix_R[i] + dR[i])

        # Run through main chain (skip master OLA since we did per-voice)
        return self.step_block(mix_L, mix_R, params, apply_master_ola=False)

    def get_metrics(self) -> Metrics:
        """Get current smoothed metrics."""
        return Metrics(
            rms_q15=self.rg.E_hat,
            peak_q15=self.rg.peak_hat,
            hf_energy_q15=self.rg.HF_hat,
            dc_offset_q15=self.rg.dc_hat,
            crest_factor_q15=0
        )
