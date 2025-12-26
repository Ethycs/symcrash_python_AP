"""
Operator Library
================

Minimal algebra acting on wavelet packet coefficient space.
All matrices are q15 fixed-point (int16), accumulations in int32.

State space: c in V = R^N with N = 256 coefficients
Partitioned into 6 bands (scales) with block structure.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
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
# IMPORTS FROM SUBMODULES
# =============================================================================

from .operators import (
    build_identity,
    build_tilt,
    build_fine_clamp,
    build_intra_mix,
    build_diffuse_up,
    build_diffuse_down,
    build_band_shear,
    build_lowrank_morph,
    build_global_pulse,
    build_swap_bands,
    build_rotate_nodes,
    build_normalize_scale,
    build_glue_pair,
    build_impulse,
    build_noise_band,
    build_pressure_ramp,
    build_gate_mask,
    build_quasi_rotate,
    build_morph_mix,
    build_emergency_simplify,
    build_fold_pressure,
    build_cusp_injection,
    build_swallowtail_mask,
    build_sheaf_project,
    build_operator_library,
    print_library_summary,
    OP_NAME_TO_ID,
    get_operator_id,
)

from .apply import (
    apply_diag,
    apply_permute,
    apply_lowrank,
    apply_banded_intra,
    apply_inject,
    apply_proj_energy_cap,
    apply_operator,
    compose_word,
)

from .export import (
    export_as_dense,
    sparse_triplets,
    visualize_matrix,
    export_all_matrices,
)

__all__ = [
    # Constants
    'COEFF_LEN', 'Q15_SCALE', 'Q15_ONE',
    'BAND_SLICES', 'BAND_LENGTHS', 'NUM_BANDS', 'NODE_LEN',
    # Types
    'OpType', 'Operator',
    # Helpers
    'to_q15', 'from_q15', 'q15_mul', 'make_tridiag_kernel', 'band_indices',
    # Operator builders
    'build_identity', 'build_tilt', 'build_fine_clamp', 'build_intra_mix',
    'build_diffuse_up', 'build_diffuse_down', 'build_band_shear',
    'build_lowrank_morph', 'build_global_pulse', 'build_swap_bands',
    'build_rotate_nodes', 'build_normalize_scale', 'build_glue_pair',
    'build_impulse', 'build_noise_band', 'build_pressure_ramp',
    'build_gate_mask', 'build_quasi_rotate', 'build_morph_mix',
    'build_emergency_simplify', 'build_fold_pressure', 'build_cusp_injection',
    'build_swallowtail_mask', 'build_sheaf_project',
    'build_operator_library', 'print_library_summary',
    'OP_NAME_TO_ID', 'get_operator_id',
    # Application
    'apply_diag', 'apply_permute', 'apply_lowrank', 'apply_banded_intra',
    'apply_inject', 'apply_proj_energy_cap', 'apply_operator', 'compose_word',
    # Export
    'export_as_dense', 'sparse_triplets', 'visualize_matrix', 'export_all_matrices',
]
