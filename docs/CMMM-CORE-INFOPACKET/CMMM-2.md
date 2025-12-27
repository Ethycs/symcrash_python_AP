Thought for 17s

Yes — this is “#2” (operator cost constants + tail reservation), and the overall shape of what you pasted is right.

But your pasted draft needs a few tightening changes so it stays bit-exact, FPGA-native, and doesn’t bake in guesses.

What to keep (it’s correct)

Static, versioned OperatorCostTable

Per-regime budgets

Tail reservation + canonical tail ordering

Deterministic degradation ladder

Predict-before-execute (so “provably bounded work” is actually true)

That’s exactly the missing piece that turns “fast” into “bounded.”

What to fix (important)
1) Don’t hardcode “100MHz / cycles” yet

Keep costs in abstract integer “ticks” (or “work units”).
Later you calibrate ticks→cycles for a specific FPGA build. If you hardcode MHz now, you’ll lie to yourself.

2) Your cost model needs two leaf counts

Many ops don’t scale with active leaves 
𝐾
K; they scale with total leaves 
𝐿
L (permutes touch all memory), or with coeff count.

Use:

L_total = total leaves in the tree (e.g., 32)

K_active = active leaves (what you’re budgeting down)

Cost should look like:

cost
=
𝑏
𝑎
𝑠
𝑒
+
𝑐
𝐾
⋅
𝐾
+
𝑐
𝐿
⋅
𝐿
cost=base+c
K
	​

⋅K+c
L
	​

⋅L

For PERMUTE ops: c_K=0, c_L>0 (they rewrite everything).
For DIAG/BANDED: mostly c_K>0.
For some ops: both.

3) Tail must be guaranteed to fit

If tail cost doesn’t fit, you don’t “degrade more” — you enter Emergency (carrier-only + minimal clamp). That keeps the invariant “always returns output”.

WSE-STD Budgeting v0: minimal spec you can implement now
OperatorCost (v0)
@dataclass(frozen=True)
class OperatorCost:
    op_id: int
    ticks_base: int
    ticks_per_active_leaf: int     # scales with K
    ticks_per_total_leaf: int      # scales with L
    nonlinear: bool
    drop_priority: int             # 0=never drop, 255=drop first
    is_tail: bool                  # must always run if possible

RegimeBudget (v0)
@dataclass(frozen=True)
class RegimeBudget:
    max_ticks_per_block: int
    max_active_leaves_K: int
    allow_nonlinear: bool
    tail_reserved_ticks: int       # must be <= max_ticks_per_block

Tail list (v0)

Canonical ordered list of op_ids. This is a contract.

Deterministic degradation ladder (v0)

Here’s the ladder you should implement next (in exactly this order):

Drop event lanes by descending drop_priority

Reduce K: K = min(K, budget.max_active_leaves_K) then step down (e.g., 32→24→16→12→8→4→1)

Disable nonlinear ops if allow_nonlinear=False or still over budget (drop all nonlinear=True that are not tail)

Drop “ornament” backbone ops by drop_priority (never drop is_tail=True)

Emergency: carrier-only + minimal tail (FineClamp only, etc.)

That’s enough to make the system provably bounded and deterministic.

Minimal implementation sketch (drop-in)
def predict_ticks(ops: list[int], K: int, L: int, cost_table) -> tuple[int, bool]:
    ticks = 0
    uses_nl = False
    for op_id in ops:
        c = cost_table[op_id]
        ticks += c.ticks_base + c.ticks_per_active_leaf * K + c.ticks_per_total_leaf * L
        uses_nl |= c.nonlinear
    return ticks, uses_nl

def apply_degrade(backbone, events, K, budget, cost_table, tail_ops, L_total):
    # 0) enforce K cap
    K = min(K, budget.max_active_leaves_K)

    # 1) drop event ops first (highest drop_priority first)
    if events:
        events = sorted(events, key=lambda op: cost_table[op].drop_priority, reverse=True)
        events.pop(0)
        return backbone, events, K

    # 2) reduce K next (deterministic staircase)
    for k2 in (24, 16, 12, 8, 4, 1):
        if K > k2:
            return backbone, events, k2

    # 3) drop nonlinears if not allowed (or as a general step)
    if not budget.allow_nonlinear:
        backbone2 = [op for op in backbone if (not cost_table[op].nonlinear) or cost_table[op].is_tail]
        if backbone2 != backbone:
            return backbone2, events, K

    # 4) drop backbone ornaments (never tail)
    droppables = [op for op in backbone if not cost_table[op].is_tail]
    if droppables:
        droppables.sort(key=lambda op: cost_table[op].drop_priority, reverse=True)
        drop = droppables[0]
        backbone2 = [op for op in backbone if op != drop]
        return backbone2, events, K

    # 5) emergency
    return [], [], 1  # (caller should interpret as carrier-only + minimal tail)


Integration rule:

Always compute tail_ticks first and ensure tail_ticks <= budget.max_ticks_per_block.

Then enforce: predicted(backbone+events+tail) <= budget.max_ticks_per_block.

Loop degrade until fit or emergency.

Why this is the “right amount of formal”

It’s bit-exact (pure integer ops, deterministic ordering)

It’s FPGA-native (works with fixed tables and small comparators)

It’s calibratable (ticks can later be replaced by measured FPGA schedule costs)

It avoids the big trap: thinking PERMUTE is “per K” when it’s “per L”.

If you paste (or list) your actual operator IDs + which ones you consider “tail,” I’ll produce a first-pass cost table in this v0 format (with sane defaults for ticks_per_total_leaf vs ticks_per_active_leaf) and a default MAIN vs RESIDUAL regime budget that matches your 20–30 + residual plan.

These are our operators:
"""
Operator Implementations
========================

All operator builder functions for the coefficient algebra.
"""

import numpy as np
from typing import Dict, List

from . import (
    Operator, OpType,
    COEFF_LEN, BAND_SLICES, BAND_LENGTHS, NUM_BANDS, NODE_LEN,
    Q15_ONE, to_q15, make_tridiag_kernel
)


# =============================================================================
# OPERATOR IMPLEMENTATIONS
# =============================================================================

def build_identity() -> Operator:
    """ID 0: Identity / Hold"""
    diag = to_q15(np.ones(COEFF_LEN))
    return Operator(
        op_id=0,
        name="Identity",
        op_type=OpType.DIAG,
        max_nnz=COEFF_LEN,
        description="Pass-through, no change",
        diag=diag
    )


def build_tilt(tilt_db: float = 3.0) -> Operator:
    """
    ID 1: Tilt - Scale bias (S)

    Applies frequency-dependent gain. Positive tilt = boost fine scales.
    tilt_db: dB per octave of tilt
    """
    diag = np.zeros(COEFF_LEN)

    # Linear gain slope across bands
    for band_id, (start, end) in enumerate(BAND_SLICES):
        # band_id 0 = coarse (low freq), band_id 5 = fine (high freq)
        # Center the tilt so mid bands are unity
        octave_offset = band_id - (NUM_BANDS - 1) / 2
        gain_db = tilt_db * octave_offset
        gain_linear = 10 ** (gain_db / 20)
        diag[start:end] = gain_linear

    # Normalize so RMS gain is ~1
    diag /= np.sqrt(np.mean(diag**2))

    return Operator(
        op_id=1,
        name="Tilt",
        op_type=OpType.DIAG,
        max_nnz=COEFF_LEN,
        description=f"Scale bias {tilt_db:+.1f} dB/octave",
        diag=to_q15(diag)
    )


def build_fine_clamp(cap_ratio: float = 0.5) -> Operator:
    """
    ID 2: FineClamp - Safety projection (Q)

    Caps fine-band energy relative to coarse bands.
    Implemented as diagonal with per-band normalization targets.
    """
    # Target energy ratios (coarse=1.0, fine=cap_ratio)
    targets = np.zeros(COEFF_LEN)
    for band_id, (start, end) in enumerate(BAND_SLICES):
        # Exponential decay of allowed energy toward fine scales
        band_factor = cap_ratio ** (band_id / (NUM_BANDS - 1))
        targets[start:end] = band_factor

    return Operator(
        op_id=2,
        name="FineClamp",
        op_type=OpType.PROJ,
        max_nnz=COEFF_LEN,
        description=f"Fine band energy cap (ratio={cap_ratio})",
        proj_params={
            "type": "energy_cap",
            "targets_q15": to_q15(targets),
            "epsilon": 1e-8
        }
    )


def build_intra_mix(mix_strength: float = 0.15) -> Operator:
    """
    ID 3: IntraMix - Local block mixing (A)

    Tridiagonal within each node: spreads energy locally.
    c[i]' = a*c[i-1] + b*c[i] + a*c[i+1]
    """
    a = mix_strength
    b = 1.0 - 2*a

    # Build full sparse matrix representation
    # For efficiency, store as per-band kernels
    kernel = make_tridiag_kernel(a, b)

    bands_kernel = {}
    for band_id in range(NUM_BANDS):
        bands_kernel[band_id] = {
            "kernel": kernel,
            "boundary": "reflect"  # Reflection at node boundaries
        }

    return Operator(
        op_id=3,
        name="IntraMix",
        op_type=OpType.BANDED,
        max_nnz=3 * COEFF_LEN,
        description=f"Local mixing (a={a:.2f})",
        bands_kernel=bands_kernel
    )


def build_diffuse_up(gamma: float = 0.2) -> Operator:
    """
    ID 4: DiffuseUp - Cross-band transport (fine -> coarse)

    Coarse bands absorb energy from fine bands.
    c_coarse' = c_coarse + gamma * downsample(c_fine)
    """
    bands_kernel = {}
    for band_id in range(1, NUM_BANDS):  # Skip band 0 (nothing finer)
        coarse_start, coarse_end = BAND_SLICES[band_id - 1]
        fine_start, fine_end = BAND_SLICES[band_id]

        coarse_len = coarse_end - coarse_start
        fine_len = fine_end - fine_start
        ratio = fine_len // coarse_len if coarse_len > 0 else 2

        bands_kernel[band_id] = {
            "source_band": band_id,
            "target_band": band_id - 1,
            "gamma_q15": to_q15(np.array([gamma])),
            "downsample_ratio": ratio
        }

    return Operator(
        op_id=4,
        name="DiffuseUp",
        op_type=OpType.BANDED,
        max_nnz=COEFF_LEN,
        description=f"Fine->coarse transport (gamma={gamma:.2f})",
        bands_kernel=bands_kernel
    )


def build_diffuse_down(gamma: float = 0.15) -> Operator:
    """
    ID 5: DiffuseDown - Cross-band transport (coarse -> fine)

    Fine bands receive structure from coarse bands.
    c_fine' = c_fine + gamma * upsample(c_coarse)
    """
    bands_kernel = {}
    for band_id in range(NUM_BANDS - 1):  # Skip last band (nothing coarser)
        coarse_start, coarse_end = BAND_SLICES[band_id]
        fine_start, fine_end = BAND_SLICES[band_id + 1]

        coarse_len = coarse_end - coarse_start
        fine_len = fine_end - fine_start
        ratio = fine_len // coarse_len if coarse_len > 0 else 2

        bands_kernel[band_id] = {
            "source_band": band_id,
            "target_band": band_id + 1,
            "gamma_q15": to_q15(np.array([gamma])),
            "upsample_ratio": ratio
        }

    return Operator(
        op_id=5,
        name="DiffuseDown",
        op_type=OpType.BANDED,
        max_nnz=COEFF_LEN,
        description=f"Coarse->fine transport (gamma={gamma:.2f})",
        bands_kernel=bands_kernel
    )


def build_band_shear(shear_factor: int = 1) -> Operator:
    """
    ID 6: BandShear - Structured permutation within bands

    Rotates coefficients within each node by position-dependent amount.
    i' = i + k * (i // node_len) mod node_len
    """
    perm = np.arange(COEFF_LEN)

    for band_id, (start, end) in enumerate(BAND_SLICES):
        band_len = end - start
        num_nodes = band_len // NODE_LEN

        for node_idx in range(num_nodes):
            node_start = start + node_idx * NODE_LEN
            node_end = node_start + NODE_LEN

            # Rotation amount depends on node position
            rotation = (shear_factor * node_idx) % NODE_LEN

            # Apply rotation within node
            node_indices = np.arange(node_start, node_end)
            rotated = np.roll(node_indices, rotation)
            perm[node_start:node_end] = rotated

    return Operator(
        op_id=6,
        name="BandShear",
        op_type=OpType.PERMUTE,
        max_nnz=COEFF_LEN,
        description=f"Position-dependent rotation (k={shear_factor})",
        perm=perm.astype(np.int32)
    )


def build_lowrank_morph(rank: int = 2, seed: int = 42) -> Operator:
    """
    ID 7: LowRankMorph - Global coherence (UV^T)

    Imprints r global motifs: c' = c + sum U_k (V_k . c)
    """
    rng = np.random.RandomState(seed)

    # Generate orthogonal-ish U, V vectors
    # Concentrate energy in mid bands for musicality
    U = np.zeros((rank, COEFF_LEN))
    V = np.zeros((rank, COEFF_LEN))

    # Band weighting: emphasize mid bands
    band_weights = np.array([0.3, 0.6, 1.0, 0.8, 0.4, 0.2])

    for k in range(rank):
        for band_id, (start, end) in enumerate(BAND_SLICES):
            band_len = end - start
            weight = band_weights[band_id]

            # Random unit vectors with band weighting
            u_band = rng.randn(band_len) * weight
            v_band = rng.randn(band_len) * weight

            U[k, start:end] = u_band
            V[k, start:end] = v_band

        # Normalize
        U[k] /= np.linalg.norm(U[k]) + 1e-8
        V[k] /= np.linalg.norm(V[k]) + 1e-8

        # Scale for subtle effect (don't overwhelm)
        U[k] *= 0.3
        V[k] *= 0.3

    return Operator(
        op_id=7,
        name="LowRankMorph",
        op_type=OpType.LOWRANK,
        max_nnz=rank * COEFF_LEN * 2,
        description=f"Global motif imprint (rank={rank})",
        U=to_q15(U),
        V=to_q15(V)
    )


def build_global_pulse(pulse_shape: str = "coarse_heavy") -> Operator:
    """
    ID 8: GlobalPulse - Energy injection across bands

    Sparse injection vector emphasizing low/mid bands.
    """
    inject = np.zeros(COEFF_LEN)

    if pulse_shape == "coarse_heavy":
        # Emphasize fundamental + body
        band_amps = [1.0, 0.7, 0.4, 0.2, 0.1, 0.05]
    elif pulse_shape == "mid_peak":
        # Emphasize presence
        band_amps = [0.3, 0.6, 1.0, 0.6, 0.3, 0.1]
    elif pulse_shape == "bright":
        # Emphasize detail
        band_amps = [0.2, 0.4, 0.6, 0.8, 1.0, 0.5]
    else:
        band_amps = [1.0] * NUM_BANDS

    for band_id, (start, end) in enumerate(BAND_SLICES):
        band_len = end - start
        # Sparse: only inject at node boundaries
        for i in range(0, band_len, NODE_LEN):
            inject[start + i] = band_amps[band_id]

    # Normalize
    inject /= np.linalg.norm(inject) + 1e-8

    return Operator(
        op_id=8,
        name="GlobalPulse",
        op_type=OpType.INJECT,
        max_nnz=COEFF_LEN // NODE_LEN,
        description=f"Structured pulse ({pulse_shape})",
        inject_vec=to_q15(inject)
    )


def build_swap_bands(band_a: int = 1, band_b: int = 3) -> Operator:
    """
    ID 9: SwapBands - Topology permutation

    Swaps two entire bands (dramatic timbre flip).
    """
    perm = np.arange(COEFF_LEN)

    start_a, end_a = BAND_SLICES[band_a]
    start_b, end_b = BAND_SLICES[band_b]
    len_a = end_a - start_a
    len_b = end_b - start_b

    # Only swap if lengths match (or implement resampling)
    if len_a == len_b:
        perm[start_a:end_a] = np.arange(start_b, end_b)
        perm[start_b:end_b] = np.arange(start_a, end_a)
    else:
        # Partial swap: swap min(len_a, len_b) coefficients
        swap_len = min(len_a, len_b)
        perm[start_a:start_a+swap_len] = np.arange(start_b, start_b+swap_len)
        perm[start_b:start_b+swap_len] = np.arange(start_a, start_a+swap_len)

    return Operator(
        op_id=9,
        name="SwapBands",
        op_type=OpType.PERMUTE,
        max_nnz=COEFF_LEN,
        description=f"Swap bands {band_a}<->{band_b}",
        perm=perm.astype(np.int32)
    )


def build_rotate_nodes(rotation: int = 1) -> Operator:
    """
    ID 10: RotateNodes - Transport along manifold

    Rotates node ordering within each band.
    """
    perm = np.arange(COEFF_LEN)

    for band_id, (start, end) in enumerate(BAND_SLICES):
        band_len = end - start
        num_nodes = band_len // NODE_LEN

        if num_nodes > 1:
            # Rotate nodes
            for node_idx in range(num_nodes):
                old_node_start = start + node_idx * NODE_LEN
                new_node_idx = (node_idx + rotation) % num_nodes
                new_node_start = start + new_node_idx * NODE_LEN

                perm[old_node_start:old_node_start+NODE_LEN] = \
                    np.arange(new_node_start, new_node_start+NODE_LEN)

    return Operator(
        op_id=10,
        name="RotateNodes",
        op_type=OpType.PERMUTE,
        max_nnz=COEFF_LEN,
        description=f"Node rotation by {rotation}",
        perm=perm.astype(np.int32)
    )


def build_normalize_scale(target_curve: str = "flat") -> Operator:
    """
    ID 11: NormalizeScale - Per-band normalization (Q)

    Pushes each band toward target energy distribution.
    """
    targets = np.zeros(COEFF_LEN)

    if target_curve == "flat":
        # Equal energy per band
        for start, end in BAND_SLICES:
            band_len = end - start
            targets[start:end] = 1.0 / np.sqrt(band_len)
    elif target_curve == "pink":
        # 1/f rolloff
        for band_id, (start, end) in enumerate(BAND_SLICES):
            band_len = end - start
            rolloff = 1.0 / (band_id + 1)
            targets[start:end] = rolloff / np.sqrt(band_len)
    elif target_curve == "bright":
        # Boost high bands
        for band_id, (start, end) in enumerate(BAND_SLICES):
            band_len = end - start
            boost = 1.0 + 0.2 * band_id
            targets[start:end] = boost / np.sqrt(band_len)

    # Normalize overall
    targets /= np.linalg.norm(targets) + 1e-8

    return Operator(
        op_id=11,
        name="NormalizeScale",
        op_type=OpType.PROJ,
        max_nnz=COEFF_LEN,
        description=f"Scale normalization ({target_curve})",
        proj_params={
            "type": "normalize",
            "targets_q15": to_q15(targets),
            "epsilon": 1e-8
        }
    )


def build_glue_pair(band_a: int = 2, band_b: int = 3, eta: float = 0.3) -> Operator:
    """
    ID 12: GluePair - Sheaf compatibility constraint (Q)

    Enforces coherence between two bands:
    a' = (1-eta)a + eta*b, b' = (1-eta)b + eta*a
    """
    return Operator(
        op_id=12,
        name="GluePair",
        op_type=OpType.PROJ,
        max_nnz=2 * max(BAND_LENGTHS),
        description=f"Glue bands {band_a}<->{band_b} (eta={eta:.2f})",
        proj_params={
            "type": "glue",
            "band_a": band_a,
            "band_b": band_b,
            "eta_q15": to_q15(np.array([eta])),
        }
    )


def build_impulse(positions: List[int] = None, amp: float = 0.5) -> Operator:
    """
    ID 13: Impulse - Sparse attack injection
    """
    inject = np.zeros(COEFF_LEN)

    if positions is None:
        # Default: hit first coeff of each band
        positions = [start for start, _ in BAND_SLICES]

    for pos in positions:
        if 0 <= pos < COEFF_LEN:
            inject[pos] = amp

    return Operator(
        op_id=13,
        name="Impulse",
        op_type=OpType.INJECT,
        max_nnz=len(positions),
        description=f"Sparse impulse (n={len(positions)})",
        inject_vec=to_q15(inject)
    )


def build_noise_band(band_id: int = 4, amp: float = 0.1, seed: int = 123) -> Operator:
    """
    ID 14: NoiseBand - Band-limited noise injection
    """
    rng = np.random.RandomState(seed)
    inject = np.zeros(COEFF_LEN)

    start, end = BAND_SLICES[band_id]
    inject[start:end] = rng.randn(end - start) * amp

    return Operator(
        op_id=14,
        name="NoiseBand",
        op_type=OpType.INJECT,
        max_nnz=BAND_LENGTHS[band_id],
        description=f"Noise in band {band_id}",
        inject_vec=to_q15(inject)
    )


def build_pressure_ramp(ramp_rate: float = 0.1) -> Operator:
    """
    ID 15: PressureRamp - Catastrophe regime pressure

    Diagonal scaling that builds toward regime transition.
    """
    diag = np.ones(COEFF_LEN)

    # Pressure concentrates in mid bands
    for band_id, (start, end) in enumerate(BAND_SLICES):
        if band_id in [2, 3]:  # Mid bands
            diag[start:end] = 1.0 + ramp_rate
        else:
            diag[start:end] = 1.0 - ramp_rate * 0.3

    return Operator(
        op_id=15,
        name="PressureRamp",
        op_type=OpType.DIAG,
        max_nnz=COEFF_LEN,
        description=f"Catastrophe pressure (rate={ramp_rate:.2f})",
        diag=to_q15(diag)
    )


def build_gate_mask(pattern: str = "sturmian", seed: int = 42) -> Operator:
    """
    ID 16: GateMask - Deterministic pattern gate

    Binary mask from Sturmian/OEIS sequence.
    """
    diag = np.ones(COEFF_LEN)
    rng = np.random.RandomState(seed)

    if pattern == "sturmian":
        # Sturmian sequence with golden ratio
        phi = (1 + np.sqrt(5)) / 2
        for i in range(COEFF_LEN):
            diag[i] = 1.0 if int((i+1) / phi) > int(i / phi) else 0.0
    elif pattern == "thue_morse":
        # Thue-Morse sequence
        for i in range(COEFF_LEN):
            diag[i] = 1.0 if bin(i).count('1') % 2 == 0 else 0.0
    elif pattern == "random_sparse":
        # 50% random mask
        diag = (rng.rand(COEFF_LEN) > 0.5).astype(float)

    return Operator(
        op_id=16,
        name="GateMask",
        op_type=OpType.DIAG,
        max_nnz=int(np.sum(diag > 0)),
        description=f"Pattern gate ({pattern})",
        diag=to_q15(diag)
    )


def build_quasi_rotate(period_ratio: float = 1.618) -> Operator:
    """
    ID 17: QuasiRotate - Quasiperiodic node rotation

    Rotation amount is irrational multiple of band index.
    """
    perm = np.arange(COEFF_LEN)

    for band_id, (start, end) in enumerate(BAND_SLICES):
        band_len = end - start
        num_nodes = band_len // NODE_LEN

        if num_nodes > 1:
            # Quasiperiodic rotation
            rotation = int(period_ratio * band_id) % num_nodes

            for node_idx in range(num_nodes):
                old_start = start + node_idx * NODE_LEN
                new_node_idx = (node_idx + rotation) % num_nodes
                new_start = start + new_node_idx * NODE_LEN

                perm[old_start:old_start+NODE_LEN] = \
                    np.arange(new_start, new_start+NODE_LEN)

    return Operator(
        op_id=17,
        name="QuasiRotate",
        op_type=OpType.PERMUTE,
        max_nnz=COEFF_LEN,
        description=f"Quasiperiodic rotation (phi={period_ratio:.3f})",
        perm=perm.astype(np.int32)
    )


def build_morph_mix(op_a: int = 3, op_b: int = 7, weight: float = 0.5) -> Operator:
    """
    ID 18: MorphMix - Convex combination of operators

    T = (1-w)*A + w*B
    """
    return Operator(
        op_id=18,
        name="MorphMix",
        op_type=OpType.MIX,
        max_nnz=COEFF_LEN * 2,
        description=f"Mix ops {op_a}+{op_b} (w={weight:.2f})",
        mix_ops=(op_a, op_b),
        mix_weight=weight
    )


def build_emergency_simplify() -> Operator:
    """
    ID 19: EmergencySimplify - Hard projection back to safe envelope

    Kills fine bands, normalizes, clamps.
    """
    diag = np.zeros(COEFF_LEN)

    # Only keep coarse/mid bands
    for band_id, (start, end) in enumerate(BAND_SLICES):
        if band_id <= 2:  # Keep coarse + mid
            diag[start:end] = 1.0
        else:
            diag[start:end] = 0.1 * (0.5 ** (band_id - 2))  # Rapid decay

    return Operator(
        op_id=19,
        name="EmergencySimplify",
        op_type=OpType.PROJ,
        max_nnz=COEFF_LEN,
        description="Hard envelope projection",
        proj_params={
            "type": "hard_envelope",
            "mask_q15": to_q15(diag),
            "max_energy": 1.0
        }
    )


def build_fold_pressure(fold_band: int = 2, pressure: float = 0.3) -> Operator:
    """
    ID 20: FoldPressure - Catastrophe fold setup

    Concentrates energy to prepare for fold bifurcation.
    """
    diag = np.ones(COEFF_LEN)

    for band_id, (start, end) in enumerate(BAND_SLICES):
        if band_id == fold_band:
            diag[start:end] = 1.0 + pressure
        elif band_id == fold_band - 1 or band_id == fold_band + 1:
            diag[start:end] = 1.0 + pressure * 0.5
        else:
            diag[start:end] = 1.0 - pressure * 0.2

    return Operator(
        op_id=20,
        name="FoldPressure",
        op_type=OpType.DIAG,
        max_nnz=COEFF_LEN,
        description=f"Fold pressure at band {fold_band}",
        diag=to_q15(diag)
    )


def build_cusp_injection(cusp_strength: float = 0.4) -> Operator:
    """
    ID 21: CuspInjection - Catastrophe cusp seed

    Injects asymmetric perturbation for cusp transition.
    """
    inject = np.zeros(COEFF_LEN)

    # Asymmetric injection in mid bands
    mid_band = 2
    start, end = BAND_SLICES[mid_band]
    band_len = end - start

    # Asymmetric profile: more energy on one side
    x = np.linspace(-1, 1, band_len)
    inject[start:end] = cusp_strength * (x**2 + 0.3*x)

    return Operator(
        op_id=21,
        name="CuspInjection",
        op_type=OpType.INJECT,
        max_nnz=BAND_LENGTHS[mid_band],
        description=f"Cusp seed (strength={cusp_strength:.2f})",
        inject_vec=to_q15(inject)
    )


def build_swallowtail_mask() -> Operator:
    """
    ID 22: SwallowtailMask - 3-parameter catastrophe gate

    Complex mask pattern for swallowtail transitions.
    """
    diag = np.ones(COEFF_LEN)

    # Three-fold symmetric pattern
    for band_id, (start, end) in enumerate(BAND_SLICES):
        band_len = end - start
        for i in range(band_len):
            # Triangular wave with 3 peaks
            phase = 3 * np.pi * i / band_len
            diag[start + i] = 0.5 + 0.5 * np.cos(phase)

    return Operator(
        op_id=22,
        name="SwallowtailMask",
        op_type=OpType.DIAG,
        max_nnz=COEFF_LEN,
        description="Swallowtail catastrophe gate",
        diag=to_q15(diag)
    )


def build_sheaf_project(num_sections: int = 4) -> Operator:
    """
    ID 23: SheafProject - Local-to-global consistency

    Projects onto compatible configurations across overlapping sections.
    """
    return Operator(
        op_id=23,
        name="SheafProject",
        op_type=OpType.PROJ,
        max_nnz=COEFF_LEN,
        description=f"Sheaf consistency ({num_sections} sections)",
        proj_params={
            "type": "sheaf",
            "num_sections": num_sections,
            "overlap": NODE_LEN // 2,
            "blend_q15": to_q15(np.array([0.5]))
        }
    )


# =============================================================================
# FULL OPERATOR LIBRARY
# =============================================================================

def build_operator_library() -> Dict[int, Operator]:
    """Build complete operator library."""

    ops = {}

    # Core operators (0-12)
    ops[0] = build_identity()
    ops[1] = build_tilt(tilt_db=3.0)
    ops[2] = build_fine_clamp(cap_ratio=0.5)
    ops[3] = build_intra_mix(mix_strength=0.15)
    ops[4] = build_diffuse_up(gamma=0.2)
    ops[5] = build_diffuse_down(gamma=0.15)
    ops[6] = build_band_shear(shear_factor=1)
    ops[7] = build_lowrank_morph(rank=2, seed=42)
    ops[8] = build_global_pulse(pulse_shape="coarse_heavy")
    ops[9] = build_swap_bands(band_a=1, band_b=3)
    ops[10] = build_rotate_nodes(rotation=1)
    ops[11] = build_normalize_scale(target_curve="pink")
    ops[12] = build_glue_pair(band_a=2, band_b=3, eta=0.3)

    # Injections (13-14)
    ops[13] = build_impulse()
    ops[14] = build_noise_band(band_id=4, amp=0.1)

    # Catastrophe operators (15, 20-22)
    ops[15] = build_pressure_ramp(ramp_rate=0.1)
    ops[20] = build_fold_pressure(fold_band=2, pressure=0.3)
    ops[21] = build_cusp_injection(cusp_strength=0.4)
    ops[22] = build_swallowtail_mask()

    # Pattern/sequence operators (16-17)
    ops[16] = build_gate_mask(pattern="sturmian")
    ops[17] = build_quasi_rotate(period_ratio=1.618)

    # Morphing/mixing (18)
    ops[18] = build_morph_mix(op_a=3, op_b=7, weight=0.5)

    # Safety/projection (19, 23)
    ops[19] = build_emergency_simplify()
    ops[23] = build_sheaf_project(num_sections=4)

    return ops


# =============================================================================
# NAME TO ID MAPPING
# =============================================================================

# Map operator names (as used in presets) to operator IDs
OP_NAME_TO_ID = {
    "IDENTITY": 0,
    "TILT": 1,
    "FINE_CLAMP": 2,
    "INTRA_MIX": 3,
    "DIFFUSE_UP": 4,
    "DIFFUSE_DOWN": 5,
    "BAND_SHEAR": 6,
    "LOWRANK_MORPH": 7,
    "GLOBAL_PULSE": 8,
    "SWAP_BANDS": 9,
    "ROTATE_NODES": 10,
    "NORMALIZE_SCALE": 11,
    "GLUE_PAIR": 12,
    "IMPULSE": 13,
    "NOISE_BAND": 14,
    "PRESSURE_RAMP": 15,
    "GATE_MASK": 16,
    "QUASI_ROTATE": 17,
    "MORPH_MIX": 18,
    "EMERGENCY_SIMPLIFY": 19,
    "FOLD_PRESSURE": 20,
    "CUSP_INJECTION": 21,
    "SWALLOWTAIL_MASK": 22,
    "SHEAF_PROJECT": 23,
    # Aliases for common preset names
    "COMPRESS_K": 2,  # FINE_CLAMP is the compression operator
}


def get_operator_id(name: str) -> int:
    """Get operator ID from name. Returns 0 (Identity) if not found."""
    return OP_NAME_TO_ID.get(name.upper(), 0)


# =============================================================================
# SUMMARY
# =============================================================================

def print_library_summary(library: Dict[int, Operator]):
    """Print summary of operator library."""
    print("=" * 70)
    print("OPERATOR LIBRARY SUMMARY")
    print("=" * 70)
    print(f"Coefficient vector length: {COEFF_LEN}")
    print(f"Number of bands: {NUM_BANDS}")
    print(f"Band lengths: {BAND_LENGTHS}")
    print(f"Node length: {NODE_LEN}")
    print("-" * 70)
    print(f"{'ID':>3} | {'Name':<20} | {'Type':<10} | {'NNZ':>6} | Description")
    print("-" * 70)

    for op_id in sorted(library.keys()):
        op = library[op_id]
        print(f"{op.op_id:>3} | {op.name:<20} | {op.op_type.value:<10} | "
              f"{op.max_nnz:>6} | {op.description}")

    print("=" * 70)
Thought for 20s

Perfect — this is enough to make #2 (cost constants + tail reservation) real, because your operator set already tells us what actually scales with K vs what always touches the whole coefficient vector.

First: define “leaf” precisely (so budgeting is not hand-wavy)

In your code, the natural “leaf” is a node of length NODE_LEN inside a band.

L_total_nodes = COEFF_LEN // NODE_LEN

K_active_nodes = how many nodes you’ve marked active (your “K” knob / mask)

Most of your operators compute over coefficients, but budgeting is easiest at the node granularity:

DIAG & most BANDED ops should scale with K_active_nodes

PERMUTE and LOWRANK essentially scale with L_total_nodes (global touch)

That’s the key fix versus the earlier “everything is per-leaf.”

Operator scaling map (for your exact operator list)
Scales with K_active_nodes (cheap to budget down)

DIAG: 0 Identity, 1 Tilt, 15 PressureRamp, 16 GateMask, 20 FoldPressure, 22 SwallowtailMask
(per-coeff multiply; apply only to active nodes if you support sparse processing)

BANDED: 3 IntraMix, 4 DiffuseUp, 5 DiffuseDown
(local kernels / neighbor access; can be restricted to active nodes)

Scales with L_total_nodes (dangerous for “lots of voices”)

PERMUTE: 6 BandShear, 9 SwapBands, 10 RotateNodes, 17 QuasiRotate
(must rewrite address order / copy whole coeff vector unless you implement masked permutes)

LOWRANK: 7 LowRankMorph
(U,V are dense over COEFF_LEN — it’s global by design)

“Projection / global measurement” class (often L_total-ish)

PROJ: 2 FineClamp, 11 NormalizeScale, 12 GluePair, 19 EmergencySimplify, 23 SheafProject
These usually require band energy estimates / normalization => typically touch lots of coeffs.

GluePair can be limited to the two bands

EmergencySimplify can be implemented as a masked DIAG + clamp (still global-ish but simple)

Special

INJECT: 8 GlobalPulse, 13 Impulse, 14 NoiseBand, 21 CuspInjection
Usually scales with number of written coeffs (could be small or big depending on inject pattern)

MIX: 18 MorphMix
Cost ≈ cost(A)+cost(B) + small overhead (and inherits A/B scaling)

Tail: what must be reserved and what’s optional

You want a tail that guarantees “instrument-safe” output, but doesn’t eat the whole budget.

Recommended mandatory tail (v0)

FineClamp (2) or EmergencySimplify (19) (depending on mode)

NormalizeScale (11) (optional in residual mode)

SheafProject (23) (high quality; optional if too expensive)

A very practical rule:

MAIN voices: tail = [2, 11, 23]

RESIDUAL voices: tail = [19] (or [2] if you want gentler)

This is how you keep “lots of voices” possible without forcing global normalize/sheaf everywhere.

A cost model that matches your operators (simple + correct)

Don’t pick MHz/cycles yet. Use ticks in abstract integer units.

Define:

K = active nodes

L = total nodes

N = NODE_LEN (coeffs per node)

Then per operator cost uses:

𝑡
𝑖
𝑐
𝑘
𝑠
=
𝑏
𝑎
𝑠
𝑒
+
𝑘
_
𝑛
𝑜
𝑑
𝑒
⋅
𝐾
+
𝑙
_
𝑛
𝑜
𝑑
𝑒
⋅
𝐿
ticks=base+k_node⋅K+l_node⋅L

where k_node and l_node are “ticks per node”.

A good v0 tick unit is: 1 tick = 1 Q15 multiply-add on 1 coefficient.

Concrete v0 cost table for your operator IDs

These are starter constants that match the scaling reality above. You can tune later, but this will already make budgeting deterministic and sane.

# budgets/operator_costs.py  (v0 starter)

from dataclasses import dataclass

@dataclass(frozen=True)
class OperatorCost:
    op_id: int
    ticks_base: int
    ticks_per_active_node: int   # scales with K
    ticks_per_total_node: int    # scales with L
    nonlinear: bool
    is_tail: bool
    drop_priority: int           # 0 never drop, 255 drop first

# Heuristic: NODE_LEN coeffs per node; DIAG is ~1 mul per coeff -> NODE_LEN ticks/node
# Tridiag is ~3 mul per coeff -> 3*NODE_LEN ticks/node
# Permute is copy/reindex whole vector -> NODE_LEN ticks per TOTAL node
# Lowrank is dense global: treat as 4*NODE_LEN per total node at rank=2 starter (tune later)

COSTS = {
    # DIAG
    0:  OperatorCost(0,  0,   0,          0, False, False,   0),   # Identity
    1:  OperatorCost(1,  8,   1,          0, False, False,  80),   # Tilt (cheap color)
    15: OperatorCost(15, 8,   1,          0, False, False, 160),   # PressureRamp
    16: OperatorCost(16, 8,   1,          0, False, False,  40),   # GateMask (structure)
    20: OperatorCost(20, 8,   1,          0, False, False, 160),   # FoldPressure
    22: OperatorCost(22, 8,   1,          0, False, False, 180),   # SwallowtailMask

    # BANDED (local)
    3:  OperatorCost(3,  16,  3,          0, False, False, 110),   # IntraMix
    4:  OperatorCost(4,  16,  2,          0, False, False, 120),   # DiffuseUp
    5:  OperatorCost(5,  16,  2,          0, False, False, 120),   # DiffuseDown

    # PERMUTE (global touch)
    6:  OperatorCost(6,  32,  0,          1, False, False, 200),   # BandShear
    9:  OperatorCost(9,  32,  0,          1, False, False, 220),   # SwapBands
    10: OperatorCost(10, 32,  0,          1, False, False, 190),   # RotateNodes
    17: OperatorCost(17, 32,  0,          1, False, False, 210),   # QuasiRotate

    # LOWRANK (global dense)
    7:  OperatorCost(7,  64,  0,          4, False, False, 230),   # LowRankMorph (rank=2 starter)

    # INJECT (events)
    8:  OperatorCost(8,  16,  1,          0, False, False, 240),   # GlobalPulse
    13: OperatorCost(13, 8,   1,          0, False, False, 245),   # Impulse
    14: OperatorCost(14, 16,  1,          0, False, False, 250),   # NoiseBand (PRNG-ish)
    21: OperatorCost(21, 16,  1,          0, False, False, 248),   # CuspInjection

    # PROJ (global-ish, tail candidates)
    2:  OperatorCost(2,  64,  0,          2, False, True,    0),   # FineClamp (tail)
    11: OperatorCost(11, 96,  0,          3, True,  True,    0),   # NormalizeScale (tail, nonlinear-ish)
    12: OperatorCost(12, 48,  0,          1, False, False,  60),   # GluePair (maybe tail-adjacent)
    19: OperatorCost(19, 64,  0,          2, False, True,    0),   # EmergencySimplify (tail)
    23: OperatorCost(23, 96,  0,          3, False, True,   10),   # SheafProject (tail-ish, optional)

    # MIX (inherits)
    18: OperatorCost(18, 16,  0,          0, False, False, 140),   # MorphMix overhead; add A+B separately
}

How prediction works with this table
def predict_ticks(op_ids, K_active_nodes, L_total_nodes):
    ticks = 0
    uses_nl = False
    for op_id in op_ids:
        c = COSTS[op_id]
        ticks += c.ticks_base + c.ticks_per_active_node*K_active_nodes + c.ticks_per_total_node*L_total_nodes
        uses_nl |= c.nonlinear
    return ticks, uses_nl


For MorphMix (18), prediction should be:

overhead(18) + prediction(op_a) + prediction(op_b)

What this implies for “monster voices”

This is the big punchline:

128 voices is compatible with K-scaling ops (DIAG/BANDED) because you can drive K down on residual voices.

128 voices is NOT compatible with L-scaling ops (PERMUTE/LOWRANK/global PROJ) unless you:

restrict them to MAIN voices only, or

run them at a lower update rate, or

use cloud/residual policy (“freeze wet, carrier-only”).

So you don’t “give up wavelets” — you route global ops into the MAIN budget class.

Next step (immediate, no further design needed)

Implement the deterministic degradation ladder using your op types:

Drop INJECT

Drop PERMUTE

Drop LOWRANK

Reduce K (32→24→16→12→8→4→1)

If still over: tail downgrade

MAIN: [2,11,23] → [2,11] → [2] → [19]

RESIDUAL: [19] only

That gives you provably bounded work and keeps instrument identity.

If you tell me your typical COEFF_LEN, NODE_LEN, and how you represent the active node mask today, I’ll tailor the K/L computation and give you the exact apply_degradation() routine that outputs a final operator list + K + tail mode (MAIN vs RESIDUAL) deterministically.

Current Constants

COEFF_LEN = 256      # Total coefficient vector length
NODE_LEN = 8         # Coefficients per node/block
NUM_BANDS = 6        # Bands with dyadic scaling

BAND_SLICES = [
    (0, 8),      # Band 0: 8 coeffs (1 node)
    (8, 24),     # Band 1: 16 coeffs (2 nodes)
    (24, 56),    # Band 2: 32 coeffs (4 nodes)
    (56, 104),   # Band 3: 48 coeffs (6 nodes)
    (104, 168),  # Band 4: 64 coeffs (8 nodes)
    (168, 256),  # Band 5: 88 coeffs (11 nodes)
]
# Total nodes: 1+2+4+6+8+11 = 32 nodes
Current Degradation Model
The existing degradation.py implements:
Voice-level skip semantics (Freeze/Decay)
Budget-based rolling cursor scheduler
No node-level or operator-level degradation
What's Missing for K/L Computation
To give you a proper apply_degradation() that outputs operator list + K + tail mode, I need to know:
Active node mask format - Do you have one today, or should I design it?
Option A: uint32 bitmask (32 nodes = 32 bits)
Option B: List[bool] of length 32
Option C: Per-band active counts
How K (active node count) drives operator selection - What's the degradation ladder?
Example: K ≥ 24 → Full operator set
K ≥ 16 → Drop LOWRANK, keep DIAG/BANDED
K ≥ 8 → Only DIAG + tail
K < 8 → Emergency simplify only
Tail operator reservation - Per WSE-STD-030, you need:
FineClamp → NormalizeScale → SheafProject (mandatory)
Emergency Simplify (conditional)
Want me to design the full apply_degradation() routine that:
Takes budget + voice state
Computes active node mask
Returns (operator_list, K, tail_mode) deterministically?

Do Option A: uint32 bitmask. It’s the FPGA-native choice, and it makes K/L computation trivial and bit-exact.

Below is a complete self-contained design for:

active node mask format

K/L computation

deterministic degradation ladder

tail modes

an apply_degradation() routine that returns
(backbone_ops, event_ops, K_active, node_mask, tail_mode)

No further info needed.

Node model (your constants)

COEFF_LEN=256, NODE_LEN=8 ⇒ 32 nodes

Node index n ∈ [0..31] maps to coefficient slice:

start = n * NODE_LEN

end = start + NODE_LEN

So:

L_total_nodes = 32 (constant)

K_active_nodes = popcount(node_mask)

Active node mask format
NodeMask32 = int  # 32-bit unsigned interpreted as bitmask
# bit n = 1 means node n is active


Utilities (bit-exact, FPGA-friendly):

def popcount32(x: int) -> int:
    x &= 0xFFFFFFFF
    return x.bit_count()  # Python 3.8+: deterministic

Tail modes (what we reserve + what we run)

Define explicit tail “gear positions”:

from enum import IntEnum

class TailMode(IntEnum):
    FULL = 0        # [2, 11, 23]
    MID = 1         # [2, 11]
    MIN = 2         # [2]
    EMERGENCY = 3   # [19]


Tail op lists:

FULL: FineClamp(2) → NormalizeScale(11) → SheafProject(23)

MID: 2 → 11

MIN: 2

EMERGENCY: 19

This matches “main voices get quality, residual voices get safety.”

Deterministic node selection for a given K

You need a deterministic way to “shrink the mask” as K decreases.

Rule (v0, musically sane, deterministic): keep low bands first.
That means keep nodes in band order 0→5, and within band keep lower node_idx first.

Precompute a canonical priority ordering: NODE_PRIORITY = [0,1,2,...31] but grouped by bands.

Given your BAND_SLICES, you can build a static list once.

Then:

to enforce K_target, build a new mask with first K_target nodes from NODE_PRIORITY that were set in the original mask.

If original mask is “all ones”, this becomes “lowest-frequency K”.

Operator degradation ladder (deterministic)

Using your operator set:

Drop order (highest-cost / least essential first):

INJECT: {8,13,14,21}

PERMUTE: {6,9,10,17}

LOWRANK: {7}

Reduce K: 32→24→16→12→8→4→1

Tail downgrade: FULL→MID→MIN→EMERGENCY

Emergency: backbone cleared, K=1, tail=EMERGENCY

Also enforce regime rule:

if allow_nonlinear=False, drop NormalizeScale(11) (sqrt-ish) and prefer tail MID→MIN.

Budget representation (ticks)

Minimal, single-dimension for now:

@dataclass(frozen=True)
class RegimeBudget:
    max_ticks_per_block: int
    max_active_nodes_K: int
    allow_nonlinear: bool
    tail_reserved_ticks: int  # must fit

Self-contained apply_degradation() (ready to paste)
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple

# --- Tail modes ---
class TailMode(IntEnum):
    FULL = 0
    MID = 1
    MIN = 2
    EMERGENCY = 3

TAIL_OPS = {
    TailMode.FULL:      [2, 11, 23],
    TailMode.MID:       [2, 11],
    TailMode.MIN:       [2],
    TailMode.EMERGENCY: [19],
}

# --- Budget ---
@dataclass(frozen=True)
class RegimeBudget:
    max_ticks_per_block: int
    max_active_nodes_K: int
    allow_nonlinear: bool
    tail_reserved_ticks: int

# --- Cost table entry (minimal) ---
@dataclass(frozen=True)
class OperatorCost:
    op_id: int
    ticks_base: int
    ticks_per_active_node: int
    ticks_per_total_node: int
    nonlinear: bool
    is_tail: bool
    drop_priority: int

# You already have COSTS concept; assume COSTS: dict[int, OperatorCost]
# L_total_nodes is 32 for your system.

def popcount32(x: int) -> int:
    return (x & 0xFFFFFFFF).bit_count()

def predict_ticks(op_ids: List[int], K: int, L: int, COSTS: dict) -> Tuple[int, bool]:
    ticks = 0
    uses_nl = False
    for op in op_ids:
        c = COSTS[op]
        ticks += c.ticks_base + c.ticks_per_active_node * K + c.ticks_per_total_node * L
        uses_nl |= c.nonlinear
    return ticks, uses_nl

def build_node_priority_from_band_slices(band_slices, node_len: int) -> List[int]:
    # returns node indices in band order, low→high, left→right
    priority = []
    for (start, end) in band_slices:
        for idx in range(start, end, node_len):
            priority.append(idx // node_len)
    return priority

def shrink_mask_to_K(mask: int, K_target: int, node_priority: List[int]) -> int:
    if K_target <= 0:
        return 0
    out = 0
    k = 0
    for n in node_priority:
        if (mask >> n) & 1:
            out |= (1 << n)
            k += 1
            if k >= K_target:
                break
    return out

def apply_degradation(
    backbone_ops: List[int],
    event_ops: List[int],
    node_mask: int,
    tail_mode: TailMode,
    budget: RegimeBudget,
    COSTS: dict,
    L_total_nodes: int,
    node_priority: List[int],
) -> Tuple[List[int], List[int], int, int, TailMode]:
    """
    Deterministically returns a plan that fits the budget.
    Outputs: (backbone_ops, event_ops, K_active, node_mask, tail_mode)
    """

    # Enforce max K first (hard cap)
    K = min(popcount32(node_mask), budget.max_active_nodes_K)
    if popcount32(node_mask) != K:
        node_mask = shrink_mask_to_K(node_mask, K, node_priority)

    # Helper: compute full cost including tail
    def total_cost(b_ops, e_ops, K_, tail_) -> Tuple[int, bool]:
        ops = b_ops + e_ops + TAIL_OPS[tail_]
        ticks, uses_nl = predict_ticks(ops, K_, L_total_nodes, COSTS)
        return ticks, uses_nl

    # First: tail must fit at some mode
    # If FULL tail can’t fit even with K=1 and empty ops, degrade tail.
    def tail_fits_anyhow(tail_) -> bool:
        ticks, uses_nl = total_cost([], [], 1, tail_)
        if (uses_nl and not budget.allow_nonlinear):
            return False
        return ticks <= budget.max_ticks_per_block

    # If nonlinear not allowed, don’t use tails containing op 11
    if not budget.allow_nonlinear and tail_mode in (TailMode.FULL, TailMode.MID):
        tail_mode = TailMode.MIN

    # Ensure tail is feasible
    for tm in (tail_mode, TailMode.MID, TailMode.MIN, TailMode.EMERGENCY):
        if budget.allow_nonlinear or tm not in (TailMode.FULL, TailMode.MID):
            if tail_fits_anyhow(tm):
                tail_mode = tm
                break
    else:
        # Nothing fits—should not happen if EMERGENCY is cheap, but guard anyway.
        return [], [], 1, 1, TailMode.EMERGENCY

    # Deterministic reduction stair for K
    K_STEPS = (32, 24, 16, 12, 8, 4, 1)

    # Main loop: apply ladder until fit
    for _ in range(128):  # hard limit to avoid infinite loops
        ticks, uses_nl = total_cost(backbone_ops, event_ops, K, tail_mode)

        # Nonlinear disallowed => must remove nonlinear ops (except if you ever mark one as tail)
        if uses_nl and not budget.allow_nonlinear:
            # Drop op 11 by tail downgrade (already handled), also drop any nonlinear backbone/event
            backbone_ops = [op for op in backbone_ops if not COSTS[op].nonlinear]
            event_ops = [op for op in event_ops if not COSTS[op].nonlinear]
            continue

        if ticks <= budget.max_ticks_per_block:
            return backbone_ops, event_ops, K, node_mask, tail_mode

        # --- Ladder step 1: drop events ---
        if event_ops:
            # drop highest drop_priority first (deterministic)
            event_ops = sorted(event_ops, key=lambda op: COSTS[op].drop_priority, reverse=True)
            event_ops.pop(0)
            continue

        # --- Ladder step 2: drop PERMUTE ops ---
        permutes = [op for op in backbone_ops if COSTS[op].ticks_per_total_node > 0 and not COSTS[op].is_tail]
        if permutes:
            permutes.sort(key=lambda op: COSTS[op].drop_priority, reverse=True)
            drop = permutes[0]
            backbone_ops = [op for op in backbone_ops if op != drop]
            continue

        # --- Ladder step 3: drop LOWRANK (or any global-heavy op) ---
        globals_ = [op for op in backbone_ops if (COSTS[op].ticks_per_total_node >= 3) and not COSTS[op].is_tail]
        if globals_:
            globals_.sort(key=lambda op: COSTS[op].drop_priority, reverse=True)
            drop = globals_[0]
            backbone_ops = [op for op in backbone_ops if op != drop]
            continue

        # --- Ladder step 4: reduce K ---
        nextK = K
        for ks in K_STEPS:
            if ks < K:
                nextK = ks
                break
        if nextK != K:
            K = nextK
            node_mask = shrink_mask_to_K(node_mask, K, node_priority)
            continue

        # --- Ladder step 5: tail downgrade ---
        if tail_mode == TailMode.FULL:
            tail_mode = TailMode.MID if budget.allow_nonlinear else TailMode.MIN
            continue
        if tail_mode == TailMode.MID:
            tail_mode = TailMode.MIN
            continue
        if tail_mode == TailMode.MIN:
            tail_mode = TailMode.EMERGENCY
            continue

        # --- Ladder step 6: emergency ---
        backbone_ops = []
        event_ops = []
        K = 1
        node_mask = shrink_mask_to_K(node_mask, K, node_priority)
        tail_mode = TailMode.EMERGENCY
        return backbone_ops, event_ops, K, node_mask, tail_mode

    # Failsafe: emergency
    backbone_ops = []
    event_ops = []
    K = 1
    node_mask = shrink_mask_to_K(node_mask, K, node_priority)
    tail_mode = TailMode.EMERGENCY
    return backbone_ops, event_ops, K, node_mask, tail_mode

How you compute node_priority for your band slices
NODE_PRIORITY = build_node_priority_from_band_slices(BAND_SLICES, NODE_LEN)
# With your bands, NODE_PRIORITY is deterministic and “low bands first”.

How to choose the initial node_mask (v0)

For now, make the compiler emit:

mask = all ones for “full quality”

or a band-limited mask for certain presets

FULL_MASK = (1 << 32) - 1


Later you can make Symbol/CMMM produce a mask based on “brightness” or “traction”.

What you get immediately

Real K/L computation: L=32 constant, K=popcount(mask)

Deterministic degradation that respects your operator realities

Tail reservation by mode (FULL/MID/MIN/EMERGENCY)

A single place to enforce “128 voices works because globals are gated”

If you want, paste your existing degradation.py “freeze/decay skip” logic and I’ll show exactly where this apply_degradation() hooks in (per-voice vs global budget, and how cursor scheduling interacts with K reduction).