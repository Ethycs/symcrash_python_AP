"""
Cathedral of Stress - Leaf Atom Matrices
========================================

Complex leaf transform matrices for rich timbral variation.
Each matrix is 8x8 and applied to the 8 coefficients in a leaf node.
"""

import numpy as np
from typing import List

NODE_SIZE = 8


# =============================================================================
# MATRIX BUILDERS
# =============================================================================

def build_identity() -> np.ndarray:
    """M0: Identity matrix - no transform."""
    return np.eye(NODE_SIZE)


def build_dct8() -> np.ndarray:
    """
    M1: DCT-II basis matrix.
    
    Projects coefficients onto cosine basis for harmonic analysis.
    Good for pitched/tonal content.
    """
    M = np.zeros((NODE_SIZE, NODE_SIZE))
    for k in range(NODE_SIZE):
        for n in range(NODE_SIZE):
            if k == 0:
                M[k, n] = 1.0 / np.sqrt(NODE_SIZE)
            else:
                M[k, n] = np.sqrt(2.0 / NODE_SIZE) * np.cos(
                    np.pi * k * (2*n + 1) / (2 * NODE_SIZE)
                )
    return M


def build_chirp() -> np.ndarray:
    """
    M2: Chirp-ish matrix.
    
    Frequency-sweeping basis that creates time-varying content.
    Good for "tearing" and evolving textures.
    """
    M = np.zeros((NODE_SIZE, NODE_SIZE), dtype=complex)
    for k in range(NODE_SIZE):
        for n in range(NODE_SIZE):
            # Chirp: frequency increases with both k and n
            phase = np.pi * k * n * (k + n) / (NODE_SIZE * NODE_SIZE)
            M[k, n] = np.exp(1j * phase)
    
    # Return real part (could also use magnitude)
    return np.real(M) / np.linalg.norm(np.real(M), axis=1, keepdims=True)


def build_spike() -> np.ndarray:
    """
    M3: Spike/asymmetric matrix.
    
    Creates sharp transients and asymmetric waveforms.
    Good for percussive content and attack emphasis.
    """
    M = np.zeros((NODE_SIZE, NODE_SIZE))
    
    for k in range(NODE_SIZE):
        # Each row is a shifted, asymmetric impulse response
        center = k
        for n in range(NODE_SIZE):
            dist = (n - center) % NODE_SIZE
            if dist == 0:
                M[k, n] = 1.0
            elif dist == 1:
                M[k, n] = 0.7
            elif dist == 2:
                M[k, n] = 0.3
            elif dist == NODE_SIZE - 1:
                M[k, n] = -0.5  # Asymmetry
            elif dist == NODE_SIZE - 2:
                M[k, n] = -0.2
    
    # Normalize rows
    for k in range(NODE_SIZE):
        norm = np.linalg.norm(M[k])
        if norm > 0:
            M[k] /= norm
    
    return M


# =============================================================================
# MATRIX REGISTRY
# =============================================================================

ATOM_MATRICES = {
    0: build_identity(),
    1: build_dct8(),
    2: build_chirp(),
    3: build_spike(),
}


def get_atom_matrix(atom_id: int) -> np.ndarray:
    """Get atom matrix by ID."""
    return ATOM_MATRICES.get(atom_id, ATOM_MATRICES[0])


def get_blended_matrix(atom_id: int, mix: float) -> np.ndarray:
    """
    Get atom matrix blended with identity.
    
    mix=0: pure identity
    mix=1: pure atom matrix
    """
    I = ATOM_MATRICES[0]
    M = ATOM_MATRICES.get(atom_id, I)
    return (1 - mix) * I + mix * M


# =============================================================================
# COEFFICIENT TRANSFORMS
# =============================================================================

def apply_atom_transform(coeffs: List[int], atom_id: int, mix: float = 1.0) -> List[int]:
    """
    Apply atom transform to 8 coefficients.
    
    coeffs: List of 8 Q15 integers
    atom_id: Which atom matrix to use
    mix: Blend with identity (0=identity, 1=full transform)
    
    Returns: Transformed coefficients (8 Q15 integers)
    """
    if mix < 0.001:
        return list(coeffs)
    
    M = get_blended_matrix(atom_id, mix)
    
    # Convert to float, transform, convert back
    c = np.array(coeffs, dtype=np.float64)
    c_transformed = M @ c
    
    # Saturate to Q15 range
    result = []
    for v in c_transformed:
        v_int = int(round(v))
        v_int = max(-32768, min(32767, v_int))
        result.append(v_int)
    
    return result


def apply_inverse_atom_transform(coeffs: List[int], atom_id: int, mix: float = 1.0) -> List[int]:
    """
    Apply inverse atom transform (for synthesis).
    """
    if mix < 0.001:
        return list(coeffs)
    
    M = get_blended_matrix(atom_id, mix)
    
    # For orthogonal matrices, inverse = transpose
    # For non-orthogonal, use pseudoinverse
    try:
        M_inv = np.linalg.pinv(M)
    except:
        M_inv = M.T  # Fallback to transpose
    
    c = np.array(coeffs, dtype=np.float64)
    c_transformed = M_inv @ c
    
    result = []
    for v in c_transformed:
        v_int = int(round(v))
        v_int = max(-32768, min(32767, v_int))
        result.append(v_int)
    
    return result


# =============================================================================
# COMPLEX LEAF MASK
# =============================================================================

def get_complex_leaf_mask(tree_id: int) -> List[bool]:
    """
    Get mask indicating which leaves use complex atom transforms.
    
    Different trees have different complex leaf patterns.
    """
    mask = [False] * 32
    
    if tree_id == 0:  # Balanced
        # Every 4th leaf is complex
        for i in range(0, 32, 4):
            mask[i] = True
    
    elif tree_id == 1:  # Harmonic
        # Odd harmonics are complex (for richer overtones)
        for i in [1, 3, 5, 7, 9, 11, 13, 15]:
            mask[i] = True
    
    elif tree_id == 2:  # Texture
        # Upper half is complex
        for i in range(16, 32):
            mask[i] = True
    
    elif tree_id == 3:  # Impact
        # First 8 and last 8 are complex (attack/release)
        for i in list(range(8)) + list(range(24, 32)):
            mask[i] = True
    
    elif tree_id == 4:  # Field
        # Sparse complex leaves
        for i in [0, 7, 15, 23, 31]:
            mask[i] = True
    
    return mask


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ATOM MATRICES")
    print("=" * 60)
    
    for name, atom_id in [("Identity", 0), ("DCT8", 1), ("Chirp", 2), ("Spike", 3)]:
        M = get_atom_matrix(atom_id)
        print(f"\n{name} (M{atom_id}):")
        print(f"  Shape: {M.shape}")
        print(f"  Orthogonal: {np.allclose(M @ M.T, np.eye(8), atol=0.1)}")
        print(f"  First row: {M[0].round(2)}")
    
    print("\n" + "=" * 60)
    print("TRANSFORM TEST")
    print("=" * 60)
    
    test_coeffs = [8000, 4000, 2000, 1000, -1000, -2000, -4000, -8000]
    
    for name, atom_id in [("DCT8", 1), ("Chirp", 2), ("Spike", 3)]:
        transformed = apply_atom_transform(test_coeffs, atom_id, mix=1.0)
        print(f"\n{name} transform:")
        print(f"  Input:  {test_coeffs}")
        print(f"  Output: {transformed}")
    
    print("\n" + "=" * 60)
    print("COMPLEX LEAF MASKS")
    print("=" * 60)
    
    tree_names = ["Balanced", "Harmonic", "Texture", "Impact", "Field"]
    for tid, name in enumerate(tree_names):
        mask = get_complex_leaf_mask(tid)
        complex_leaves = [i for i, m in enumerate(mask) if m]
        print(f"  Tree {tid} ({name}): {complex_leaves}")
