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
2. Make BLOCK BOUNDARIES inaudible by default (clicks must be intentional)

IMPORTANT ARCHITECTURAL INVARIANT:
- Safety Tail handles BLOCK BOUNDARY artifacts ONLY
- Safety NEVER handles musical transitions (note changes, pitch glides, coefficient morphs)
- All coefficient changes are slewed or micro-interpolated by the Wavelet Engine
- Clicks from musical transitions = bug in Wavelet/Symbol Engine, NOT Safety's job

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

from ..fixed import (
    sat16, mul_q15_sat, shift_round, slew_q15, slew_q15_asym,
    sqrt_q15, soft_clip_q15,
    Q15_ONE, Q15_SHIFT, Q15_HALF
)
from ..frames import ParamFrame, Metrics

# Import from primitives (single source of truth)
from ..primitives import NUM_LEAVES, Q15_NEG_ONE


# =============================================================================
# SYMBOL TAIL CONSTANTS (coefficient-space)
# =============================================================================

LEAF_COUNT = NUM_LEAVES   # L = 64 leaves per tree (D=6 master tree)
COEFF_PER_LEAF = 4        # 4 coefficients per leaf (256 / 64 = 4)
COEFF_LEN = LEAF_COUNT * COEFF_PER_LEAF  # Total coefficient vector length (256)

# Fine band set (highest frequency leaves, prone to ultrafine blowup)
FINE_BAND_SET = {60, 61, 62, 63}  # Default: last 4 leaves (updated for 64 leaves)
COARSE_BAND_SET = set(range(0, 60))  # Everything else

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

# Regularity operator constants
R_BRIDGE_LEN = 16         # Samples for micro-bridge (event-local)
R_CURVATURE_MAX = 0x1000  # C2: Maximum second difference (curvature)

# =============================================================================
# DEGENERATE REGION HANDLING (low-energy awareness)
# =============================================================================
#
# Peak-based logic breaks in near-silence. These thresholds define the
# energy floor below which we switch to energy-aware repair mode.
#
# Key insight: in low-energy regions, *relative* jumps can be huge even
# if absolute jumps are small. We need dual-detector: (E, J) not just J.

# Energy floor thresholds (Q15 RMS scale)
# -60 dBFS RMS ≈ 0.001 * 32768 ≈ 33
# -72 dBFS RMS ≈ 0.00025 * 32768 ≈ 8
E_FLOOR_RMS = 0x0080      # ~-48 dBFS - below this is "degenerate" region
E_FLOOR_HYSTERESIS = 0x0100  # Exit degenerate mode when E > 2*floor

# Jump thresholds for degenerate regions (tighter than normal)
J_DEGENERATE_THRESHOLD = 0x0020  # 32 LSB - much stricter in low energy

# Normalization guards
NORM_EPSILON_RMS = 0x0040   # Minimum RMS for normalization (prevents blowup)
NORM_MAX_GAIN = 0x4000      # Maximum normalization gain (Q15, ~2x)
NORM_SLEW_RATE = 0x0100     # Maximum gain change per block

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
# CONSTANT-POWER CROSSFADE
# =============================================================================

def constant_power_fade(t: float) -> Tuple[float, float]:
    """
    Compute constant-power (sin²) crossfade gains.

    Args:
        t: Normalized position [0.0, 1.0] through the crossfade

    Returns:
        (fade_out, fade_in) gains that sum to 1.0 in power
    """
    fade_in = math.sin(math.pi / 2 * t) ** 2
    fade_out = math.cos(math.pi / 2 * t) ** 2
    return fade_out, fade_in


def crossfade_append(
    dest: List[int],
    src: List[int],
    fade_samples: int = 128,
    constant_power: bool = True,
) -> None:
    """
    Append src to dest with a crossfade to avoid clicks at boundaries.

    This is a key safety primitive for click-free audio concatenation.
    Uses constant-power (sin²) fades by default to avoid amplitude dips.

    Handles degenerate (near-silence) regions by using energy-aware fading
    to avoid amplifying noise or creating artifacts when blending silence.

    Args:
        dest: Destination list (modified in place)
        src: Source samples to append
        fade_samples: Number of samples for crossfade region
        constant_power: Use sin² constant-power fade (True) or linear (False)
    """
    if not dest or not src or fade_samples <= 0:
        dest.extend(src)
        return

    fade_samples = min(fade_samples, len(dest), len(src))

    # Compute energy of dest tail and src head to detect degenerate regions
    dest_tail_start = len(dest) - fade_samples
    dest_energy = sum(abs(dest[dest_tail_start + i]) for i in range(fade_samples))
    src_energy = sum(abs(src[i]) for i in range(fade_samples))

    # Energy floor for degenerate detection (average |sample| < 64 is considered degenerate)
    energy_floor_total = fade_samples * 64

    dest_is_degenerate = dest_energy < energy_floor_total
    src_is_degenerate = src_energy < energy_floor_total

    # Crossfade the overlap region
    for i in range(fade_samples):
        if constant_power:
            t = i / (fade_samples - 1) if fade_samples > 1 else 1.0
            fade_out, fade_in = constant_power_fade(t)
        else:
            # Linear fade
            fade_in = i / fade_samples
            fade_out = 1.0 - fade_in

        idx = len(dest) - fade_samples + i
        dest_val = dest[idx]
        src_val = src[i]

        if dest_is_degenerate and not src_is_degenerate:
            # Dest is near-silence, src has energy: fade in src from zero baseline
            # This prevents weird artifacts from amplifying dest noise
            dest[idx] = int(src_val * fade_in)
        elif src_is_degenerate and not dest_is_degenerate:
            # Src is near-silence, dest has energy: fade out dest to zero
            dest[idx] = int(dest_val * fade_out)
        elif dest_is_degenerate and src_is_degenerate:
            # Both degenerate: just linear blend toward zero then src
            dest[idx] = int(dest_val * fade_out + src_val * fade_in)
        else:
            # Normal case: both have energy, use standard crossfade
            dest[idx] = int(dest_val * fade_out + src_val * fade_in)

    # Append the rest (after crossfade region)
    dest.extend(src[fade_samples:])


def crossfade_blocks(
    block_a: List[int],
    block_b: List[int],
    fade_samples: int = 64,
    constant_power: bool = True,
) -> List[int]:
    """
    Crossfade between two audio blocks, returning the blended result.

    Unlike crossfade_append, this returns a new block of length
    len(block_a) + len(block_b) - fade_samples.

    Handles degenerate (near-silence) regions by using energy-aware fading
    to avoid amplifying noise or creating artifacts when blending silence.

    Args:
        block_a: First audio block
        block_b: Second audio block
        fade_samples: Number of samples for crossfade overlap
        constant_power: Use sin² constant-power fade (True) or linear (False)

    Returns:
        New list with crossfaded audio
    """
    if not block_a:
        return list(block_b)
    if not block_b:
        return list(block_a)

    fade_samples = min(fade_samples, len(block_a), len(block_b))

    # Compute energy of overlap regions to detect degenerate cases
    a_tail_start = len(block_a) - fade_samples
    a_energy = sum(abs(block_a[a_tail_start + i]) for i in range(fade_samples))
    b_energy = sum(abs(block_b[i]) for i in range(fade_samples))

    # Energy floor for degenerate detection (average |sample| < 64 is considered degenerate)
    energy_floor_total = fade_samples * 64

    a_is_degenerate = a_energy < energy_floor_total
    b_is_degenerate = b_energy < energy_floor_total

    # Copy the non-overlapping part of block_a
    result = list(block_a[:-fade_samples])

    # Crossfade region
    for i in range(fade_samples):
        if constant_power:
            t = i / (fade_samples - 1) if fade_samples > 1 else 1.0
            fade_out, fade_in = constant_power_fade(t)
        else:
            fade_in = i / fade_samples
            fade_out = 1.0 - fade_in

        a_idx = len(block_a) - fade_samples + i
        a_val = block_a[a_idx]
        b_val = block_b[i]

        if a_is_degenerate and not b_is_degenerate:
            # A is near-silence, B has energy: fade in B from zero baseline
            sample = int(b_val * fade_in)
        elif b_is_degenerate and not a_is_degenerate:
            # B is near-silence, A has energy: fade out A to zero
            sample = int(a_val * fade_out)
        elif a_is_degenerate and b_is_degenerate:
            # Both degenerate: just linear blend
            sample = int(a_val * fade_out + b_val * fade_in)
        else:
            # Normal case: both have energy, use standard crossfade
            sample = int(a_val * fade_out + b_val * fade_in)

        result.append(sample)

    # Append the rest of block_b
    result.extend(block_b[fade_samples:])

    return result


# =============================================================================
# REGULARITY OPERATOR R (WSE-STD-070)
# =============================================================================
#
# The Regularity Operator projects audio onto a bounded-variation set.
# Clicks are impulses; impulses are unbounded derivatives.
#
# R consists of two components:
# 1. Event-local micro-bridge: C¹ cosine ramp at known change points
# 2. Slope projection fallback: symmetric slew limiter for any remaining spikes
#
# R is universal: it doesn't care what caused the change (pitch, tree, op-chain,
# injection). It only enforces that the output has bounded first and second
# differences.

def _build_cosine_bridge_lut(n: int) -> List[int]:
    """
    Build C¹ cosine bridge LUT for micro-bridge.

    The bridge function w(t) satisfies:
    - w(0) = 0, w(1) = 1 (amplitude continuity)
    - w'(0) = 0, w'(1) = 0 (derivative continuity at endpoints)

    Uses smoothstep: w(t) = 3t² - 2t³
    """
    lut = []
    for i in range(n + 1):
        t = i / n
        # Smoothstep: 3t² - 2t³ (C¹ continuous)
        w = 3 * t * t - 2 * t * t * t
        lut.append(int(round(w * Q15_ONE)))
    return lut


# Precomputed micro-bridge LUT
_BRIDGE_LUT = _build_cosine_bridge_lut(R_BRIDGE_LEN)


def micro_bridge(
    old_samples: List[int],
    new_samples: List[int],
    bridge_len: int = R_BRIDGE_LEN,
) -> List[int]:
    """
    Apply C¹-continuous micro-bridge between old and new generator states.

    This is the event-local regularity fix: when something changes at a known
    sample index, we blend old→new using a smoothstep that matches both value
    and derivative at the endpoints.

    Args:
        old_samples: Samples from the "old" state (at least bridge_len)
        new_samples: Samples from the "new" state (at least bridge_len)
        bridge_len: Number of samples for the transition

    Returns:
        Blended samples (length = bridge_len)
    """
    if len(old_samples) < bridge_len or len(new_samples) < bridge_len:
        # Not enough context, fall back to simple blend
        bridge_len = min(len(old_samples), len(new_samples), bridge_len)
        if bridge_len == 0:
            return []

    result = []
    for i in range(bridge_len):
        # w goes from 0 to Q15_ONE over bridge_len
        if bridge_len == R_BRIDGE_LEN:
            w = _BRIDGE_LUT[i]
        else:
            t = i / (bridge_len - 1) if bridge_len > 1 else 1.0
            w = int((3 * t * t - 2 * t * t * t) * Q15_ONE)

        inv_w = Q15_ONE - w

        # Blend: out = old * (1-w) + new * w
        out = (old_samples[i] * inv_w + new_samples[i] * w) >> 15
        result.append(sat16(out))

    return result


def compute_local_rms(samples: List[int], center: int, window: int = 16) -> int:
    """
    Compute local RMS energy around a sample index.

    Args:
        samples: Audio samples
        center: Center index for window
        window: Half-width of window

    Returns:
        RMS energy in Q15 scale
    """
    if not samples:
        return 0

    start = max(0, center - window)
    end = min(len(samples), center + window)
    n = end - start

    if n == 0:
        return 0

    # Sum of squares (avoiding overflow)
    energy = 0
    for i in range(start, end):
        s = samples[i]
        energy += (s * s) >> 16  # Scale down to prevent overflow

    # RMS = sqrt(energy / n), scaled to Q15
    avg_energy = energy // n
    return sqrt_q15(min(Q15_ONE, avg_energy))


def detect_discontinuities_energy_aware(
    samples: List[int],
    prev_sample: int = 0,
    slew_threshold: int = SLEW_MAX,
    curvature_threshold: int = R_CURVATURE_MAX,
    energy_floor: int = E_FLOOR_RMS,
    degenerate_slew: int = J_DEGENERATE_THRESHOLD,
) -> Tuple[List[int], bool]:
    """
    Energy-aware discontinuity detection with dual (E, J) detector.

    In normal energy regions: standard slew/curvature detection.
    In degenerate (low-energy) regions: stricter thresholds + energy gating.

    Args:
        samples: Audio samples to analyze
        prev_sample: Last sample from previous block
        slew_threshold: Normal slew threshold
        curvature_threshold: Normal curvature threshold
        energy_floor: RMS below which we're in degenerate mode
        degenerate_slew: Stricter slew threshold for degenerate regions

    Returns:
        (discontinuity_indices, is_degenerate_region)
    """
    if len(samples) < 3:
        return [], False

    # Compute block-level RMS
    block_rms = compute_local_rms(samples, len(samples) // 2, len(samples) // 2)
    is_degenerate = block_rms < energy_floor

    discontinuities = []
    prev = prev_sample
    prev_delta = 0

    for i, s in enumerate(samples):
        delta = s - prev
        abs_delta = abs(delta)
        curvature = abs(delta - prev_delta) if i > 0 else 0

        if is_degenerate:
            # Degenerate mode: only flag if jump is significant relative to local energy
            local_rms = compute_local_rms(samples, i, 8)
            if local_rms < energy_floor:
                # In true silence, only flag if absolute jump exceeds degenerate threshold
                if abs_delta > degenerate_slew:
                    discontinuities.append(i)
            else:
                # Transitioning out of silence - use normal thresholds
                if abs_delta > slew_threshold or curvature > curvature_threshold:
                    discontinuities.append(i)
        else:
            # Normal mode: standard detection
            if abs_delta > slew_threshold or curvature > curvature_threshold:
                discontinuities.append(i)

        prev_delta = delta
        prev = s

    return discontinuities, is_degenerate


def hermite_bridge(
    samples: List[int],
    k: int,
    bridge_len: int = 8,
) -> List[int]:
    """
    Apply C¹-continuous Hermite bridge at a known discontinuity point.

    The Hermite bridge guarantees both value AND slope continuity by
    interpolating between left and right context using cubic Hermite
    interpolation:

        H(t) = (2t³ - 3t² + 1)p₀ + (t³ - 2t² + t)m₀ + (-2t³ + 3t²)p₁ + (t³ - t²)m₁

    where:
        p₀, p₁ = endpoint values (left, right)
        m₀, m₁ = endpoint slopes (left, right)

    This is superior to raised-cosine for handling slope discontinuities
    because it explicitly matches both value and derivative.

    Args:
        samples: Audio samples (modified in place conceptually)
        k: Index of discontinuity
        bridge_len: Half-width of bridge (total width = 2*bridge_len)

    Returns:
        Corrected samples
    """
    if k < bridge_len or k >= len(samples) - bridge_len:
        return list(samples)

    out = list(samples)

    # Extract left and right context
    # p0 = value at left edge of bridge
    # p1 = value at right edge of bridge
    # m0 = slope at left edge (average of nearby slopes)
    # m1 = slope at right edge
    left_idx = k - bridge_len
    right_idx = k + bridge_len

    p0 = samples[left_idx]
    p1 = samples[right_idx]

    # Compute slopes from surrounding context (average of 2 samples)
    if left_idx >= 2:
        m0 = ((samples[left_idx] - samples[left_idx - 1]) +
              (samples[left_idx - 1] - samples[left_idx - 2])) // 2
    elif left_idx >= 1:
        m0 = samples[left_idx] - samples[left_idx - 1]
    else:
        m0 = 0

    if right_idx < len(samples) - 2:
        m1 = ((samples[right_idx + 1] - samples[right_idx]) +
              (samples[right_idx + 2] - samples[right_idx + 1])) // 2
    elif right_idx < len(samples) - 1:
        m1 = samples[right_idx + 1] - samples[right_idx]
    else:
        m1 = 0

    # Scale slopes for the interpolation interval
    # The Hermite basis expects tangents in the same units as the interval
    interval = 2 * bridge_len
    m0_scaled = m0 * interval // 4  # Scaled for smoother blend
    m1_scaled = m1 * interval // 4

    # Apply Hermite interpolation across the bridge
    for i in range(2 * bridge_len + 1):
        idx = left_idx + i
        if idx < 0 or idx >= len(out):
            continue

        t = i / (2 * bridge_len)  # 0 to 1

        # Hermite basis functions
        t2 = t * t
        t3 = t2 * t
        h00 = 2*t3 - 3*t2 + 1      # 1 at t=0, 0 at t=1, zero slopes
        h10 = t3 - 2*t2 + t        # tangent basis at t=0
        h01 = -2*t3 + 3*t2         # 0 at t=0, 1 at t=1, zero slopes
        h11 = t3 - t2              # tangent basis at t=1

        # Compute interpolated value
        interp = int(h00 * p0 + h10 * m0_scaled + h01 * p1 + h11 * m1_scaled)
        out[idx] = sat16(interp)

    return out


def apply_degenerate_hold(
    samples: List[int],
    prev_output: int,
    energy_floor: int = E_FLOOR_RMS,
) -> List[int]:
    """
    Energy-aware hold/fade for degenerate (near-silence) regions.

    When both the input signal is near-silence, instead of amplifying noise
    or letting tiny discontinuities through, we fade toward the previous
    output value or zero.

    This prevents the "normalizing noise" problem where tiny signals get
    blown up into audible artifacts.

    Args:
        samples: Audio samples
        prev_output: Previous block's final output sample (used as fade target)
        energy_floor: RMS threshold for degenerate mode

    Returns:
        Processed samples (held/faded if degenerate)
    """
    block_rms = compute_local_rms(samples, len(samples) // 2, len(samples) // 2)

    if block_rms >= energy_floor:
        # Normal energy - pass through
        return list(samples)

    # Degenerate region - fade toward previous output (for continuity) then zero
    out = []

    # Use prev_output as initial fade target, decay toward zero
    fade_target = prev_output

    for i, s in enumerate(samples):
        # Compute local energy to determine fade amount
        local_rms = compute_local_rms(samples, i, 8)

        if local_rms < energy_floor // 2:
            # Very low energy - strong fade toward target
            fade_amount = 0.8
        elif local_rms < energy_floor:
            # Low energy - moderate fade
            fade_amount = 0.5
        else:
            # Transitioning - light fade
            fade_amount = 0.2

        # Blend toward target
        blended = int(s * (1 - fade_amount) + fade_target * fade_amount)
        out.append(sat16(blended))

        # Decay fade target toward zero over the block
        fade_target = int(fade_target * 0.95)

    return out


def detect_discontinuities(
    samples: List[int],
    prev_sample: int = 0,
    slew_threshold: int = SLEW_MAX,
    curvature_threshold: int = R_CURVATURE_MAX,
) -> List[int]:
    """
    Detect sample indices where discontinuities occur.

    Discontinuities are detected by:
    1. First difference (slew) exceeding threshold
    2. Second difference (curvature) exceeding threshold

    Args:
        samples: Audio samples to analyze
        prev_sample: Last sample from previous block
        slew_threshold: Maximum allowed |x[n] - x[n-1]|
        curvature_threshold: Maximum allowed |(x[n]-x[n-1]) - (x[n-1]-x[n-2])|

    Returns:
        List of sample indices where discontinuities were detected
    """
    if len(samples) < 2:
        return []

    discontinuities = []
    prev = prev_sample
    prev_delta = 0

    for i, s in enumerate(samples):
        delta = s - prev
        curvature = abs(delta - prev_delta) if i > 0 else 0

        # Check both slew and curvature
        if abs(delta) > slew_threshold or curvature > curvature_threshold:
            discontinuities.append(i)

        prev_delta = delta
        prev = s

    return discontinuities


def slope_project(
    samples: List[int],
    prev_sample: int = 0,
    max_slew: int = SLEW_MAX,
    bidirectional: bool = True,
) -> Tuple[List[int], int]:
    """
    Project audio onto bounded-slope set (symmetric slew limiter).

    This is the fallback regularity operator: even if we don't know where
    the change happened, we clamp all sample-to-sample differences.

    GUARANTEE: Output always satisfies |x[n] - x[n-1]| <= max_slew.

    For symmetric (unbiased) limiting, we do a forward pass then optionally
    a backward pass to reduce directional bias, followed by a final
    enforcement pass.

    Args:
        samples: Audio samples
        prev_sample: Last sample from previous block
        max_slew: Maximum |x[n] - x[n-1]|
        bidirectional: Apply backward pass for symmetry

    Returns:
        (limited_samples, last_sample)
    """
    if not samples:
        return [], prev_sample

    # Forward pass - strict slew limiting
    out = []
    prev = prev_sample
    for s in samples:
        delta = s - prev
        if delta > max_slew:
            limited = prev + max_slew
        elif delta < -max_slew:
            limited = prev - max_slew
        else:
            limited = s
        out.append(limited)
        prev = limited

    if not bidirectional or len(out) < 2:
        return out, out[-1] if out else prev_sample

    # Backward pass (reduces bias from forward-only limiting)
    # Work backward to pull samples toward target while respecting slew
    backward = list(out)
    prev = backward[-1]
    for i in range(len(backward) - 2, -1, -1):
        target = samples[i]
        current = backward[i]
        next_val = backward[i + 1]

        # Can we move current toward target without breaking slew to next?
        delta_to_next = next_val - current
        if abs(delta_to_next) < max_slew:
            # We have headroom - try to move toward target
            headroom = max_slew - abs(delta_to_next)
            delta_to_target = target - current

            # Move by at most headroom, and at most half the distance to target
            move = min(headroom, abs(delta_to_target) // 2)
            if delta_to_target > 0:
                backward[i] = current + move
            elif delta_to_target < 0:
                backward[i] = current - move

    # Final enforcement pass - GUARANTEE slew bound
    # This handles any edge cases from backward pass
    out = []
    prev = prev_sample
    for s in backward:
        delta = s - prev
        if delta > max_slew:
            limited = prev + max_slew
        elif delta < -max_slew:
            limited = prev - max_slew
        else:
            limited = s
        out.append(limited)
        prev = limited

    return out, out[-1] if out else prev_sample


class RegularityOperator:
    """
    Universal Regularity Operator R.

    Enforces bounded variation (no impulses) through:
    1. Event-local micro-bridge at known change points
    2. Slope projection fallback for unknown/residual discontinuities

    Usage:
        R = RegularityOperator()

        # When you know where a change happened (e.g., note change at sample 32):
        audio = R.apply_bridge(audio, change_idx=32)

        # Or let it auto-detect discontinuities:
        audio = R.regularize(audio, prev_sample)

        # Always apply slope projection as final safety net:
        audio, last = R.slope_project(audio, prev_sample)
    """

    def __init__(
        self,
        bridge_len: int = R_BRIDGE_LEN,
        max_slew: int = SLEW_MAX,
        max_curvature: int = R_CURVATURE_MAX,
    ):
        self.bridge_len = bridge_len
        self.max_slew = max_slew
        self.max_curvature = max_curvature
        self.prev_sample = 0

    def apply_bridge_at(
        self,
        samples: List[int],
        change_idx: int,
        old_samples: List[int] = None,
    ) -> List[int]:
        """
        Apply micro-bridge at a known change point.

        If old_samples is provided, blends old→new at change_idx.
        Otherwise, applies local smoothing around the discontinuity.

        Args:
            samples: Current audio samples (modified in place conceptually)
            change_idx: Sample index where change occurred
            old_samples: Optional previous state samples

        Returns:
            Regularized samples
        """
        if change_idx < 0 or change_idx >= len(samples):
            return list(samples)

        out = list(samples)
        half_bridge = self.bridge_len // 2

        if old_samples is not None and len(old_samples) >= half_bridge:
            # Blend old → new using micro-bridge
            start = max(0, change_idx - half_bridge)
            end = min(len(samples), change_idx + half_bridge)

            for i in range(start, end):
                rel_pos = i - (change_idx - half_bridge)
                if rel_pos < 0 or rel_pos >= self.bridge_len:
                    continue

                t = rel_pos / (self.bridge_len - 1) if self.bridge_len > 1 else 1.0
                w = 3 * t * t - 2 * t * t * t  # smoothstep

                old_idx = i - change_idx + len(old_samples) - 1
                if 0 <= old_idx < len(old_samples):
                    old_val = old_samples[old_idx]
                    new_val = samples[i]
                    out[i] = int(old_val * (1 - w) + new_val * w)
        else:
            # No old samples - apply local smoothing
            # Use a short weighted average around the discontinuity
            start = max(0, change_idx - half_bridge)
            end = min(len(samples), change_idx + half_bridge)

            if end - start >= 3:
                # Simple 3-tap smoothing centered on change point
                for i in range(max(1, start), min(len(out) - 1, end)):
                    out[i] = (samples[i - 1] + samples[i] * 2 + samples[i + 1]) // 4

        return out

    def regularize(
        self,
        samples: List[int],
        prev_sample: int = None,
        change_indices: List[int] = None,
    ) -> List[int]:
        """
        Full regularization: micro-bridges + slope projection.

        Args:
            samples: Audio samples
            prev_sample: Last sample from previous block (or use internal state)
            change_indices: Optional list of known change points

        Returns:
            Regularized samples
        """
        if prev_sample is None:
            prev_sample = self.prev_sample

        out = list(samples)

        # 1. Apply micro-bridges at known change points
        if change_indices:
            for idx in change_indices:
                out = self.apply_bridge_at(out, idx)

        # 2. Detect remaining discontinuities
        remaining = detect_discontinuities(
            out, prev_sample, self.max_slew, self.max_curvature
        )

        # 3. Apply micro-bridges at detected discontinuities
        for idx in remaining:
            out = self.apply_bridge_at(out, idx)

        # 4. Final slope projection (always, as safety net)
        out, self.prev_sample = slope_project(
            out, prev_sample, self.max_slew, bidirectional=True
        )

        return out

    def reset(self):
        """Reset internal state."""
        self.prev_sample = 0


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
            if c > Q15_ONE or c < Q15_NEG_ONE:
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


def _tail_jit_active() -> bool:
    """True when step_block should use the numba kernel (shares the VM switch)."""
    try:
        from ..vm.hotpath_jit import jit_enabled
        from . import safety_tail_jit  # noqa: F401
    except ImportError:  # pragma: no cover - numba is a project dependency
        return False
    return jit_enabled()


def _tail_tables():
    """Module LUTs/tables as int64 arrays for the kernel (built once)."""
    import numpy as np
    from ..primitives import _SQRT_LUT
    return (np.array(SafetyEngine.E_TARGETS, dtype=np.int64),
            np.array(SafetyEngine.HF_BUDGETS, dtype=np.int64),
            np.array(_SQRT_LUT, dtype=np.int64),
            np.array(_HANN_OLA_LUT, dtype=np.int64))


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

        # Degenerate mode hysteresis state (per voice)
        self.degenerate_mode: Dict[int, bool] = {}
        self.prev_output: Dict[int, int] = {}  # For degenerate hold

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

    def apply_declick(self, x: List[int], voice_id: int, seam: bool = False,
                      apply_c1_patch: bool = True) -> List[int]:
        """
        Full de-click chain for a voice block.

        Args:
            x: Audio samples
            voice_id: Voice identifier
            seam: If True, apply OLA crossfade at block boundary.
                  Seam events include: voice start/stop, wet-valid transition,
                  tree/tuning change. For phase-continuous carrier (beta=0),
                  seam should always be False.
            apply_c1_patch: If True, apply C1 discontinuity correction.
                  Disable for phase-continuous carrier which has no interior
                  discontinuities.

        IMPORTANT: OLA is ONLY applied when seam=True. This prevents the
        block-rate artifacts (750 Hz at 48kHz/64 samples) that occur when
        Hann crossfade is applied to phase-continuous audio.
        """
        x = self.apply_voice_start(x, voice_id)
        x = self.apply_voice_stop(x, voice_id)

        # OLA only on explicit seam events, NOT every block
        if seam:
            x = self.apply_ola(x, voice_id)

        # C1 patch for interior discontinuity correction (wavelet synthesis)
        if apply_c1_patch:
            x = self.apply_c1_patch(x, voice_id)

        return x

    # =========================================================================
    # C¹ DISCONTINUITY PATCHER (interior click removal)
    # =========================================================================

    def detect_discontinuity_energy_aware(
        self,
        x: List[int],
        voice_id: int = 0,
        threshold: int = SLEW_MAX,
    ) -> Tuple[List[int], bool]:
        """
        Energy-aware discontinuity detection with hysteresis.

        Uses dual (E, J) detector: in low-energy (degenerate) regions,
        applies stricter thresholds and different repair strategies.

        Implements hysteresis to prevent flapping:
        - Enter degenerate mode when E < E_FLOOR_RMS
        - Exit degenerate mode when E > E_FLOOR_HYSTERESIS

        Args:
            x: Audio samples
            voice_id: Voice ID for hysteresis state tracking
            threshold: Normal slew threshold

        Returns:
            (discontinuity_indices, is_degenerate)
        """
        # Get/init hysteresis state
        was_degenerate = self.degenerate_mode.get(voice_id, False)

        # Use the module-level energy-aware detector
        discontinuities, is_degenerate_now = detect_discontinuities_energy_aware(
            x, 0, threshold, R_CURVATURE_MAX, E_FLOOR_RMS, J_DEGENERATE_THRESHOLD
        )

        # Apply hysteresis
        if was_degenerate:
            # Already in degenerate mode - need higher energy to exit
            block_rms = compute_local_rms(x, len(x) // 2, len(x) // 2)
            if block_rms > E_FLOOR_HYSTERESIS:
                is_degenerate = False
            else:
                is_degenerate = True
        else:
            is_degenerate = is_degenerate_now

        # Update state
        self.degenerate_mode[voice_id] = is_degenerate

        return discontinuities, is_degenerate

    def apply_c1_patch(
        self,
        x: List[int],
        voice_id: int = 0,
        patch_radius: int = 8,
    ) -> List[int]:
        """
        Apply C¹-continuous correction using energy-aware detection + Hermite bridge.

        This is the improved interior click repair that:
        1. Uses dual (E, J) detection with hysteresis for degenerate regions
        2. Applies Hermite bridge for true C¹ continuity (value + slope match)
        3. In degenerate regions, applies hold/fade instead of amplifying noise

        Args:
            x: Audio samples
            voice_id: Voice ID for state tracking
            patch_radius: Half-width of Hermite bridge (samples)

        Returns:
            Corrected audio samples
        """
        # Energy-aware detection with hysteresis
        discontinuities, is_degenerate = self.detect_discontinuity_energy_aware(
            x, voice_id
        )

        # Get previous output for degenerate hold
        prev_out = self.prev_output.get(voice_id, 0)

        if is_degenerate:
            # Degenerate region - apply hold/fade first
            out = apply_degenerate_hold(x, prev_out, E_FLOOR_RMS)
        else:
            out = list(x)

        if not discontinuities:
            # Update prev_output and return
            self.prev_output[voice_id] = out[-1] if out else 0
            return out

        # Apply Hermite bridge at each discontinuity
        for k in discontinuities:
            # Skip if too close to block edges for Hermite bridge
            if k < patch_radius or k >= len(out) - patch_radius:
                continue

            # Apply Hermite bridge (guarantees C¹ continuity)
            out = hermite_bridge(out, k, patch_radius)

        # Update prev_output for next block
        self.prev_output[voice_id] = out[-1] if out else 0

        return out

    def detect_discontinuity(self, x: List[int], threshold: int = SLEW_MAX) -> List[int]:
        """
        Detect interior discontinuities (clicks) in a block.

        Legacy method - uses simple slew/curvature detection.
        For energy-aware detection, use detect_discontinuity_energy_aware().

        Args:
            x: Audio samples
            threshold: Slew threshold for discontinuity detection

        Returns:
            List of sample indices where discontinuities were detected
        """
        return detect_discontinuities(x, 0, threshold, R_CURVATURE_MAX)

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

        # Track previous samples for HF filter (use previous block's last samples for i=0)
        prev_yL = self.prev_sample_L if hasattr(self, 'prev_sample_L') else 0
        prev_yR = self.prev_sample_R if hasattr(self, 'prev_sample_R') else 0

        for i in range(self.B):
            # Apply gain
            yL = mul_q15_sat(L[i], self.q_gain)
            yR = mul_q15_sat(R[i], self.q_gain)

            # HF reduction (simple lowpass blend) - now includes first sample
            if blend > 0:
                yL = sat16(mul_q15_sat(yL, self.q_hf_cut) +
                          mul_q15_sat(prev_yL, blend))
                yR = sat16(mul_q15_sat(yR, self.q_hf_cut) +
                          mul_q15_sat(prev_yR, blend))

            prev_yL = yL
            prev_yR = yR
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

            # Asymmetric slew (fast attack, slow release) - independent per channel
            self.lim_L.gain = slew_q15_asym(
                self.lim_L.gain, target_gain,
                self.ATTACK_RATE, self.RELEASE_RATE
            )
            self.lim_R.gain = slew_q15_asym(
                self.lim_R.gain, target_gain,
                self.ATTACK_RATE, self.RELEASE_RATE
            )

            # Apply gain - use per-channel limiter gain
            yL = mul_q15_sat(L[i], self.lim_L.gain)
            yR = mul_q15_sat(R[i], self.lim_R.gain)

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
                   apply_slew_limit: bool = True,
                   apply_dc_removal: bool = True) -> Tuple[List[int], List[int]]:
        """
        Full safety processing chain (Safety Tail).

        Chain: Metrics -> RG -> Q -> Master OLA -> Slew Limit -> Limiter -> DC removal

        IMPORTANT: Safety Tail handles BLOCK BOUNDARY artifacts only.
        Musical transitions (note changes, coefficient morphing) are handled
        by the Wavelet Engine's coefficient slewing, NOT here.

        For per-voice de-click, call apply_declick() on each voice block
        BEFORE mixing to stereo.

        Enforces invariants:
        - A1: Peak bound (via limiter)
        - A2: Slew bound (via slew limiter at block boundaries)
        - A3: DC bound (via DC removal)

        Args:
            L, R: Audio samples (stereo)
            params: ParamFrame with K_mode etc.
            apply_master_ola: Apply master OLA for block boundary smoothing
            apply_slew_limit: Apply final slew limiting
            apply_dc_removal: Apply DC offset removal. Disable for phase-continuous
                              carrier signals that are zero-mean over full cycles.
        """
        if _tail_jit_active() and len(L) == self.B and len(R) == self.B:
            return self._step_block_jit(L, R, params, apply_master_ola,
                                        apply_slew_limit, apply_dc_removal)

        # 1) Measure metrics
        m = self.measure_metrics(L, R)

        # 2) Update RG (smoothed tracking)
        self.update_rg(m, params.K_mode)

        # 3) Apply Q (gain + HF control)
        Lq, Rq = self.apply_q(L, R, params.K_mode)

        # 4) Apply master OLA (block boundary smoothing on mixed output)
        if apply_master_ola:
            Lq, Rq = self.apply_master_ola(Lq, Rq)

        # 5) Apply slew limit (A2 invariant - final safety net after all processing)
        if apply_slew_limit:
            Lq, Rq, self.prev_sample_L, self.prev_sample_R = self.apply_slew_limit(
                Lq, Rq, self.prev_sample_L, self.prev_sample_R
            )

        # 6) Apply limiter (A1 invariant)
        Lo, Ro = self.apply_limiter(Lq, Rq)

        # 7) Remove DC (A3 invariant)
        # Skip for phase-continuous signals (sine waves have zero-mean over full cycles)
        if apply_dc_removal:
            Lo, Ro = self.remove_dc(Lo, Ro)

        return Lo, Ro

    def _step_block_jit(self, L: List[int], R: List[int], params: ParamFrame,
                        apply_master_ola: bool, apply_slew_limit: bool,
                        apply_dc_removal: bool) -> Tuple[List[int], List[int]]:
        """step_block via the bit-exact numba kernel (safety_tail_jit.safety_tail_step).

        All state is read from and written back to the same instance fields the
        pure-Python methods use, so the two paths can be alternated block by block.
        """
        import numpy as np
        from . import safety_tail_jit as stj

        n = self.B
        primed = 1 if (self.master_ola_L and len(self.master_ola_L) >= OLA_OVERLAP) else 0
        st = np.array([
            self.rg.E_hat, self.rg.HF_hat, self.rg.peak_hat, self.rg.dc_hat,
            self.q_gain, self.q_hf_cut,
            self.prev_sample_L, self.prev_sample_R,
            self.lim_L.gain, self.lim_R.gain,
            primed, self.alpha,
        ], dtype=np.int64)
        if primed:
            tail_L = np.array(self.master_ola_L[-OLA_OVERLAP:], dtype=np.int64)
            tail_R = np.array(self.master_ola_R[-OLA_OVERLAP:], dtype=np.int64)
        else:
            tail_L = np.zeros(OLA_OVERLAP, dtype=np.int64)
            tail_R = np.zeros(OLA_OVERLAP, dtype=np.int64)
        flags = ((stj.FLAG_MASTER_OLA if apply_master_ola else 0)
                 | (stj.FLAG_SLEW_LIMIT if apply_slew_limit else 0)
                 | (stj.FLAG_DC_REMOVAL if apply_dc_removal else 0))
        outL = np.empty(n, dtype=np.int64)
        outR = np.empty(n, dtype=np.int64)

        stj.safety_tail_step(
            np.asarray(L, dtype=np.int64), np.asarray(R, dtype=np.int64), n, st, flags,
            int(params.K_mode), _E_TARGETS_ARR, _HF_BUDGETS_ARR, _SQRT_LUT_ARR, _HANN_OLA_ARR,
            tail_L, tail_R,
            self.LIMIT_THRESHOLD, self.LIMIT_CEILING, self.ATTACK_RATE, self.RELEASE_RATE,
            SLEW_MAX, OLA_OVERLAP, outL, outR,
        )

        self.rg.E_hat = int(st[stj.ST_E_HAT])
        self.rg.HF_hat = int(st[stj.ST_HF_HAT])
        self.rg.peak_hat = int(st[stj.ST_PEAK_HAT])
        self.rg.dc_hat = int(st[stj.ST_DC_HAT])
        self.q_gain = int(st[stj.ST_Q_GAIN])
        self.q_hf_cut = int(st[stj.ST_Q_HF_CUT])
        self.prev_sample_L = int(st[stj.ST_PREV_L])
        self.prev_sample_R = int(st[stj.ST_PREV_R])
        self.lim_L.gain = int(st[stj.ST_LIM_L])
        self.lim_R.gain = int(st[stj.ST_LIM_R])
        if apply_master_ola:
            self.master_ola_L = tail_L.tolist()
            self.master_ola_R = tail_R.tolist()
        return outL.tolist(), outR.tolist()

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


# Kernel tables (after SafetyEngine so its class constants are available).
_E_TARGETS_ARR, _HF_BUDGETS_ARR, _SQRT_LUT_ARR, _HANN_OLA_ARR = _tail_tables()
