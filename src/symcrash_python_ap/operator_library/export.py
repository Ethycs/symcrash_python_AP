"""
Matrix Export Functions
=======================

Export operators as dense matrices, sparse triplets, and visualizations.
"""

import numpy as np
from typing import Dict, Optional
import json

from . import (
    Operator, OpType,
    COEFF_LEN, BAND_SLICES, BAND_LENGTHS, NODE_LEN, NUM_BANDS,
    Q15_SCALE, from_q15, to_q15
)
from .operators import build_operator_library


# =============================================================================
# EXPORT AS DENSE MATRIX
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
# SPARSE TRIPLETS
# =============================================================================

def sparse_triplets(M: np.ndarray, threshold: float = 1e-10) -> dict:
    """Convert dense matrix to sparse triplets."""
    rows, cols = np.where(np.abs(M) > threshold)
    vals = M[rows, cols]
    return {
        "rows": rows.tolist(),
        "cols": cols.tolist(),
        "vals": vals.tolist(),
        "nnz": len(vals),
        "shape": list(M.shape)
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_matrix(M: np.ndarray, name: str, max_display: int = 20):
    """Print matrix structure visualization."""
    print(f"\n{'='*60}")
    print(f"MATRIX: {name}")
    print(f"{'='*60}")
    print(f"Shape: {M.shape}")
    print(f"Nonzeros: {np.sum(np.abs(M) > 1e-10)}")
    print(f"Frobenius norm: {np.linalg.norm(M, 'fro'):.6f}")
    print(f"Max abs value: {np.max(np.abs(M)):.6f}")
    print(f"Rank: {np.linalg.matrix_rank(M, tol=1e-6)}")

    # Show structure heatmap (reduced)
    if M.shape[0] > max_display:
        step = M.shape[0] // max_display
        M_reduced = M[::step, ::step]
        print(f"\nStructure (downsampled {step}x):")
    else:
        M_reduced = M

    # Text-based heatmap
    for i in range(M_reduced.shape[0]):
        row_str = ""
        for j in range(M_reduced.shape[1]):
            val = M_reduced[i, j]
            if abs(val) < 1e-6:
                row_str += "."
            elif val > 0.5:
                row_str += "#"
            elif val > 0.1:
                row_str += "+"
            elif val > 0.01:
                row_str += ":"
            elif val > 0:
                row_str += "."
            elif val < -0.5:
                row_str += "X"
            elif val < -0.1:
                row_str += "-"
            else:
                row_str += "o"
        print(f"  {row_str}")


# =============================================================================
# FULL EXPORT
# =============================================================================

def export_all_matrices() -> Dict:
    """Export all operators as explicit matrices."""
    library = build_operator_library()

    print("="*70)
    print("EXPLICIT MATRIX EXPORT")
    print("="*70)
    print(f"\nCoefficient space: V = R^{COEFF_LEN}")
    print(f"Band structure: {BAND_LENGTHS}")
    print(f"Node length: {NODE_LEN}")

    all_matrices = {}

    # Process each operator
    for op_id in sorted(library.keys()):
        op = library[op_id]

        print(f"\n{'-'*60}")
        print(f"Op {op_id}: {op.name} ({op.op_type.value})")
        print(f"Description: {op.description}")
        print(f"Max NNZ: {op.max_nnz}")

        # Export based on type
        if op.op_type == OpType.DIAG:
            diag_float = from_q15(op.diag)

            print(f"\nDiagonal values by band:")
            for band_id, (start, end) in enumerate(BAND_SLICES):
                band_vals = diag_float[start:end]
                print(f"  Band {band_id}: mean={np.mean(band_vals):.4f}, "
                      f"std={np.std(band_vals):.4f}, "
                      f"range=[{np.min(band_vals):.4f}, {np.max(band_vals):.4f}]")

            all_matrices[op.name] = {
                "type": "DIAG",
                "diagonal_q15": op.diag.tolist(),
                "diagonal_float": diag_float.tolist()
            }

        elif op.op_type == OpType.PERMUTE:
            perm = op.perm

            # Check if identity
            is_identity = np.all(perm == np.arange(COEFF_LEN))
            cycles = []
            visited = set()
            for i in range(COEFF_LEN):
                if i not in visited and perm[i] != i:
                    cycle = [i]
                    j = perm[i]
                    while j != i:
                        cycle.append(j)
                        visited.add(j)
                        j = perm[j]
                    visited.add(i)
                    if len(cycle) > 1:
                        cycles.append(cycle[:min(5, len(cycle))])

            print(f"\nPermutation structure:")
            print(f"  Is identity: {is_identity}")
            print(f"  Number of cycles: {len(cycles)}")
            if cycles:
                print(f"  Sample cycles: {cycles[:5]}")

            all_matrices[op.name] = {
                "type": "PERMUTE",
                "permutation": perm.tolist(),
                "inverse": np.argsort(perm).tolist()
            }

        elif op.op_type == OpType.LOWRANK:
            U = from_q15(op.U)
            V = from_q15(op.V)
            rank = U.shape[0]

            # Full matrix: I + U V^T
            M = np.eye(COEFF_LEN)
            for k in range(rank):
                M += np.outer(U[k], V[k])

            print(f"\nLow-rank structure (rank={rank}):")
            for k in range(rank):
                print(f"  U_{k} norm: {np.linalg.norm(U[k]):.4f}")
                print(f"  V_{k} norm: {np.linalg.norm(V[k]):.4f}")
                print(f"  U_{k}.V_{k}: {np.dot(U[k], V[k]):.4f}")

            print(f"\n  Matrix departure from I:")
            print(f"    Frobenius: {np.linalg.norm(M - np.eye(COEFF_LEN), 'fro'):.4f}")
            print(f"    Spectral: {np.linalg.norm(M - np.eye(COEFF_LEN), 2):.4f}")

            all_matrices[op.name] = {
                "type": "LOWRANK",
                "rank": rank,
                "U_q15": op.U.tolist(),
                "V_q15": op.V.tolist(),
                "U_float": U.tolist(),
                "V_float": V.tolist()
            }

        elif op.op_type == OpType.BANDED:
            if op.bands_kernel and "kernel" in list(op.bands_kernel.values())[0]:
                # Intra-band tridiagonal
                M = np.zeros((COEFF_LEN, COEFF_LEN))

                print(f"\nTridiagonal kernels by band:")
                for band_id, (start, end) in enumerate(BAND_SLICES):
                    kernel_info = op.bands_kernel.get(band_id)
                    if kernel_info and "kernel" in kernel_info:
                        kernel = from_q15(kernel_info["kernel"])
                        print(f"  Band {band_id}: [{kernel[0]:.4f}, {kernel[1]:.4f}, {kernel[2]:.4f}]")

                        for i in range(start, end):
                            if i > start:
                                M[i, i-1] = kernel[0]
                            M[i, i] = kernel[1]
                            if i < end - 1:
                                M[i, i+1] = kernel[2]

                all_matrices[op.name] = {
                    "type": "BANDED_TRIDIAG",
                    "kernels": {str(k): from_q15(v["kernel"]).tolist()
                               for k, v in op.bands_kernel.items() if "kernel" in v},
                    "triplets": sparse_triplets(M)
                }
            else:
                # Cross-band (diffuse)
                print(f"\nCross-band coupling:")
                for band_id, info in op.bands_kernel.items():
                    gamma = from_q15(info["gamma_q15"])[0]
                    src = info.get("source_band", band_id)
                    tgt = info.get("target_band", band_id)
                    ratio = info.get("downsample_ratio", info.get("upsample_ratio", 2))
                    print(f"  Band {src} -> Band {tgt}: gamma={gamma:.4f}, ratio={ratio}")

                all_matrices[op.name] = {
                    "type": "BANDED_CROSSBAND",
                    "couplings": {str(k): {
                        "source": v.get("source_band", k),
                        "target": v.get("target_band", k),
                        "gamma": from_q15(v["gamma_q15"])[0],
                        "ratio": v.get("downsample_ratio", v.get("upsample_ratio", 2))
                    } for k, v in op.bands_kernel.items()}
                }

        elif op.op_type == OpType.INJECT:
            inject = from_q15(op.inject_vec)
            nonzero_idx = np.where(np.abs(inject) > 1e-10)[0]

            print(f"\nInjection vector:")
            print(f"  Nonzero entries: {len(nonzero_idx)}")
            print(f"  Norm: {np.linalg.norm(inject):.4f}")
            print(f"  Energy by band:")
            for band_id, (start, end) in enumerate(BAND_SLICES):
                band_energy = np.sum(inject[start:end]**2)
                if band_energy > 1e-10:
                    print(f"    Band {band_id}: {band_energy:.4f}")

            all_matrices[op.name] = {
                "type": "INJECT",
                "vector_q15": op.inject_vec.tolist(),
                "vector_float": inject.tolist(),
                "nonzero_indices": nonzero_idx.tolist()
            }

        elif op.op_type == OpType.PROJ:
            print(f"\nProjection parameters:")
            for k, v in op.proj_params.items():
                if isinstance(v, np.ndarray):
                    print(f"  {k}: array shape {v.shape}")
                else:
                    print(f"  {k}: {v}")

            all_matrices[op.name] = {
                "type": "PROJ",
                "params": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in op.proj_params.items()}
            }

        elif op.op_type == OpType.MIX:
            print(f"\nMix operator:")
            print(f"  Op A: {op.mix_ops[0]}")
            print(f"  Op B: {op.mix_ops[1]}")
            print(f"  Weight: {op.mix_weight}")

            all_matrices[op.name] = {
                "type": "MIX",
                "op_a": op.mix_ops[0],
                "op_b": op.mix_ops[1],
                "weight": op.mix_weight
            }

    return all_matrices


def export_dense_matrices_npz(output_path: str = "operator_matrices.npz"):
    """Export dense matrices to NPZ file for numerical work."""
    library = build_operator_library()

    matrices = {}
    for op_id in sorted(library.keys()):
        op = library[op_id]
        M = export_as_dense(op)
        if M is not None:
            matrices[f"op{op_id:02d}_{op.name}"] = M.astype(np.float32)

    np.savez_compressed(output_path, **matrices)
    print(f"\nExported {len(matrices)} dense matrices to {output_path}")

    return matrices


def show_matrix_visualizations():
    """Show text-based visualizations of key matrices."""
    library = build_operator_library()

    # Key operators to visualize
    key_ops = [1, 3, 6, 7, 10]  # Tilt, IntraMix, BandShear, LowRankMorph, RotateNodes

    for op_id in key_ops:
        op = library[op_id]
        M = export_as_dense(op)
        if M is not None:
            visualize_matrix(M, f"Op {op_id}: {op.name}")
