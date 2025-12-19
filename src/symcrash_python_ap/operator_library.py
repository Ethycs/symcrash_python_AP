"""
Operator Library v0: Bit-Accurate Matrix Implementations
=========================================================

Minimal algebra acting on wavelet packet coefficient space.
All matrices are q15 fixed-point (int16), accumulations in int32.

State space: c ∈ V = R^N with N = 256 coefficients
Partitioned into 6 bands (scales) with block structure.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================

COEFF_LEN = 256  # Total coefficient vector length
Q15_SCALE = 32768  # 2^15 for q15 fixed-point
Q15_ONE = 32767  # Max positive q15 value

# Band structure: 6 bands with dyadic scaling
# Band 0 (coarsest): 8 coeffs   - fundamental structure
# Band 1: 16 coeffs             - low-mid body
# Band 2: 32 coeffs             - mid presence
# Band 3: 48 coeffs             - high-mid detail
# Band 4: 64 coeffs             - fine texture
# Band 5 (finest): 88 coeffs    - ultrafine / transients

BAND_SLICES = [
    (0, 8),      # Band 0: coarse
    (8, 24),     # Band 1: low-mid
    (24, 56),    # Band 2: mid
    (56, 104),   # Band 3: high-mid
    (104, 168),  # Band 4: fine
    (168, 256),  # Band 5: ultrafine
]

BAND_LENGTHS = [e - s for s, e in BAND_SLICES]  # [8, 16, 32, 48, 64, 88]
NUM_BANDS = len(BAND_SLICES)

# Node structure within bands (for block operations)
NODE_LEN = 8  # Coefficients per node/block


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class OpType(Enum):
    DIAG = "DIAG"           # Diagonal scaling
    BANDED = "BANDED"       # Banded/sparse coupling
    LOWRANK = "LOWRANK"     # UV^T low-rank
    PERMUTE = "PERMUTE"     # Coefficient permutation
    PROJ = "PROJ"           # Projection/constraint
    INJECT = "INJECT"       # Content injection
    MIX = "MIX"             # Convex combination


@dataclass
class Operator:
    """Single operator specification."""
    op_id: int
    name: str
    op_type: OpType
    max_nnz: int
    description: str
    # Matrix data (depends on type)
    diag: Optional[np.ndarray] = None          # For DIAG
    bands_kernel: Optional[Dict] = None         # For BANDED
    U: Optional[np.ndarray] = None              # For LOWRANK
    V: Optional[np.ndarray] = None              # For LOWRANK
    perm: Optional[np.ndarray] = None           # For PERMUTE
    proj_params: Optional[Dict] = None          # For PROJ
    inject_vec: Optional[np.ndarray] = None     # For INJECT
    mix_ops: Optional[Tuple[int, int]] = None   # For MIX
    mix_weight: float = 0.5                     # For MIX


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def to_q15(x: np.ndarray) -> np.ndarray:
    """Convert float array to q15 int16."""
    return np.clip(np.round(x * Q15_SCALE), -Q15_SCALE, Q15_ONE).astype(np.int16)


def from_q15(x: np.ndarray) -> np.ndarray:
    """Convert q15 int16 to float."""
    return x.astype(np.float64) / Q15_SCALE


def q15_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two q15 arrays with proper scaling."""
    result = (a.astype(np.int32) * b.astype(np.int32)) >> 15
    return np.clip(result, -Q15_SCALE, Q15_ONE).astype(np.int16)


def make_tridiag_kernel(a: float, b: float) -> np.ndarray:
    """Create tridiagonal kernel [a, b, a] in q15."""
    return to_q15(np.array([a, b, a]))


def band_indices(band_id: int) -> Tuple[int, int]:
    """Get start, end indices for a band."""
    return BAND_SLICES[band_id]


# =============================================================================
# OPERATOR IMPLEMENTATIONS AS MATRICES
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
    ID 4: DiffuseUp - Cross-band transport (fine → coarse)
    
    Coarse bands absorb energy from fine bands.
    c_coarse' = c_coarse + gamma * downsample(c_fine)
    """
    # Build as banded matrix with cross-band connections
    # Each coarse coeff receives from 2 fine coeffs (2:1 ratio)
    
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
        description=f"Fine→coarse transport (γ={gamma:.2f})",
        bands_kernel=bands_kernel
    )


def build_diffuse_down(gamma: float = 0.15) -> Operator:
    """
    ID 5: DiffuseDown - Cross-band transport (coarse → fine)
    
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
        description=f"Coarse→fine transport (γ={gamma:.2f})",
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
    
    Imprints r global motifs: c' = c + Σ U_k (V_k · c)
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
        description=f"Swap bands {band_a}↔{band_b}",
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
    a' = (1-η)a + ηb, b' = (1-η)b + ηa
    """
    return Operator(
        op_id=12,
        name="GluePair",
        op_type=OpType.PROJ,
        max_nnz=2 * max(BAND_LENGTHS),
        description=f"Glue bands {band_a}↔{band_b} (η={eta:.2f})",
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
        description=f"Quasiperiodic rotation (φ={period_ratio:.3f})",
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
# OPERATOR APPLICATION (BIT-ACCURATE)
# =============================================================================

def apply_diag(c: np.ndarray, op: Operator) -> np.ndarray:
    """Apply diagonal operator."""
    return q15_mul(c, op.diag)


def apply_permute(c: np.ndarray, op: Operator) -> np.ndarray:
    """Apply permutation operator."""
    return c[op.perm]


def apply_lowrank(c: np.ndarray, op: Operator) -> np.ndarray:
    """Apply low-rank operator: c + Σ U_k (V_k · c)."""
    result = c.copy().astype(np.int32)
    
    rank = op.U.shape[0]
    for k in range(rank):
        # V_k · c (dot product)
        dot = np.sum(op.V[k].astype(np.int32) * c.astype(np.int32)) >> 15
        dot = np.clip(dot, -Q15_SCALE, Q15_ONE)
        
        # U_k * dot
        contribution = (op.U[k].astype(np.int32) * dot) >> 15
        result += contribution
    
    return np.clip(result, -Q15_SCALE, Q15_ONE).astype(np.int16)


def apply_banded_intra(c: np.ndarray, op: Operator) -> np.ndarray:
    """Apply banded intra-band mixing."""
    result = np.zeros_like(c, dtype=np.int32)
    
    for band_id, (start, end) in enumerate(BAND_SLICES):
        kernel_info = op.bands_kernel.get(band_id)
        if kernel_info is None:
            result[start:end] = c[start:end]
            continue
        
        kernel = kernel_info["kernel"]  # [a, b, a] in q15
        
        for i in range(start, end):
            # Reflection boundary handling
            i_left = max(start, i - 1)
            i_right = min(end - 1, i + 1)
            
            # Tridiagonal: a*c[i-1] + b*c[i] + a*c[i+1]
            val = (kernel[0].astype(np.int32) * c[i_left] +
                   kernel[1].astype(np.int32) * c[i] +
                   kernel[2].astype(np.int32) * c[i_right]) >> 15
            result[i] = val
    
    return np.clip(result, -Q15_SCALE, Q15_ONE).astype(np.int16)


def apply_inject(c: np.ndarray, op: Operator, amplitude: float = 1.0) -> np.ndarray:
    """Apply injection operator."""
    amp_q15 = to_q15(np.array([amplitude]))[0]
    inject_scaled = q15_mul(op.inject_vec, np.full_like(op.inject_vec, amp_q15))
    
    result = c.astype(np.int32) + inject_scaled.astype(np.int32)
    return np.clip(result, -Q15_SCALE, Q15_ONE).astype(np.int16)


def apply_proj_energy_cap(c: np.ndarray, op: Operator) -> np.ndarray:
    """Apply energy capping projection."""
    targets = from_q15(op.proj_params["targets_q15"])
    epsilon = op.proj_params["epsilon"]
    
    result = c.copy().astype(np.float64) / Q15_SCALE
    
    for band_id, (start, end) in enumerate(BAND_SLICES):
        band = result[start:end]
        energy = np.sum(band**2)
        target_energy = np.sum(targets[start:end]**2)
        
        if energy > target_energy + epsilon:
            scale = np.sqrt(target_energy / (energy + epsilon))
            result[start:end] *= scale
    
    return to_q15(result)


def apply_operator(c: np.ndarray, op: Operator, **kwargs) -> np.ndarray:
    """Apply any operator to coefficient vector."""
    
    if op.op_type == OpType.DIAG:
        return apply_diag(c, op)
    
    elif op.op_type == OpType.PERMUTE:
        return apply_permute(c, op)
    
    elif op.op_type == OpType.LOWRANK:
        return apply_lowrank(c, op)
    
    elif op.op_type == OpType.BANDED:
        # Check if it's intra-band mixing
        if op.bands_kernel and "kernel" in list(op.bands_kernel.values())[0]:
            return apply_banded_intra(c, op)
        else:
            # Cross-band diffuse (simplified: just return c for now)
            return c
    
    elif op.op_type == OpType.INJECT:
        amp = kwargs.get("amplitude", 1.0)
        return apply_inject(c, op, amplitude=amp)
    
    elif op.op_type == OpType.PROJ:
        proj_type = op.proj_params.get("type", "energy_cap")
        if proj_type == "energy_cap":
            return apply_proj_energy_cap(c, op)
        elif proj_type == "normalize":
            return apply_proj_energy_cap(c, op)  # Same logic
        else:
            return c
    
    elif op.op_type == OpType.MIX:
        # MIX requires library reference (handle in orchestrator)
        return c
    
    return c


# =============================================================================
# OPERATOR WORD COMPOSITION
# =============================================================================

def compose_word(c: np.ndarray, op_ids: List[int], library: Dict[int, Operator]) -> np.ndarray:
    """Apply sequence of operators (operator word)."""
    result = c.copy()
    
    for op_id in op_ids:
        if op_id in library:
            result = apply_operator(result, library[op_id])
    
    return result


# =============================================================================
# EXPORT MATRICES AS DENSE (FOR INSPECTION)
# =============================================================================

def export_as_dense(op: Operator) -> Optional[np.ndarray]:
    """Export operator as dense matrix (for debugging/visualization)."""
    
    if op.op_type == OpType.DIAG:
        return np.diag(from_q15(op.diag))
    
    elif op.op_type == OpType.PERMUTE:
        P = np.zeros((COEFF_LEN, COEFF_LEN))
        for i, j in enumerate(op.perm):
            P[i, j] = 1.0
        return P
    
    elif op.op_type == OpType.LOWRANK:
        U = from_q15(op.U)
        V = from_q15(op.V)
        M = np.eye(COEFF_LEN)
        for k in range(U.shape[0]):
            M += np.outer(U[k], V[k])
        return M
    
    elif op.op_type == OpType.BANDED and op.bands_kernel:
        # Build tridiagonal
        M = np.zeros((COEFF_LEN, COEFF_LEN))
        
        for band_id, (start, end) in enumerate(BAND_SLICES):
            kernel_info = op.bands_kernel.get(band_id)
            if kernel_info and "kernel" in kernel_info:
                kernel = from_q15(kernel_info["kernel"])
                for i in range(start, end):
                    if i > start:
                        M[i, i-1] = kernel[0]
                    M[i, i] = kernel[1]
                    if i < end - 1:
                        M[i, i+1] = kernel[2]
        return M
    
    return None


# =============================================================================
# SUMMARY / DOCUMENTATION
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


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Build library
    library = build_operator_library()
    
    # Print summary
    print_library_summary(library)
    
    # Test application
    print("\n" + "=" * 70)
    print("TEST: Operator application")
    print("=" * 70)
    
    # Random initial state
    np.random.seed(0)
    c0 = to_q15(np.random.randn(COEFF_LEN) * 0.3)
    
    print(f"Initial state: norm = {np.linalg.norm(from_q15(c0)):.4f}")
    
    # Apply some operators
    test_sequence = [1, 3, 7, 11]  # Tilt → IntraMix → LowRank → Normalize
    
    c = c0.copy()
    for op_id in test_sequence:
        c = apply_operator(c, library[op_id])
        print(f"After op {op_id:2d} ({library[op_id].name}): "
              f"norm = {np.linalg.norm(from_q15(c)):.4f}")
    
    # Test operator word
    print("\nOperator word composition:")
    c_word = compose_word(c0, test_sequence, library)
    print(f"Word result: norm = {np.linalg.norm(from_q15(c_word)):.4f}")
    
    # Export a dense matrix for inspection
    print("\nExporting LowRankMorph as dense matrix...")
    M = export_as_dense(library[7])
    if M is not None:
        print(f"Matrix shape: {M.shape}")
        print(f"Matrix rank (approx): {np.linalg.matrix_rank(M - np.eye(COEFF_LEN), tol=0.01)}")
        print(f"Frobenius norm: {np.linalg.norm(M - np.eye(COEFF_LEN), 'fro'):.4f}")
