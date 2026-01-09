"""
Polyphase-Interleaved IDWPT (Streaming-Compatible)
==================================================

This module provides the **production** "streaming" IDWPT used by the CCS path.

Why this exists
---------------
The balanced 64-point WPT/IDWPT (depth=6) is a **block transform**. If you
reconstruct each 64-sample block independently from coefficients that are being
updated over time, boundary samples become phase-dependent and you get audible
block artifacts (clicks / combing).

CCS fixes this by:
  1) representing coefficient state as **K polyphase time slots** per leaf
  2) running **K block IDWPTs** per audio block (one per slot)
  3) **interleaving** the results so sample n uses the reconstruction whose
     slot matches n % K.

This is a constrained polyphase synthesis network that behaves "streaming"
from the VM's point of view, without requiring a full sample-by-sample lifting
pipeline with delay elements.

Layout conventions
------------------
- Leaves: L = 64
- Audio block: B = 64 samples
- Polyphase slots: K (VM default K=4)

Coefficient state can be provided either as:
- coeff_2d: shape [64, K]  (leaf-major)
- coeff_flat: length 64*K, laid out as [leaf0_k0..kK-1, leaf1_k0.., ...]

API note
--------
Older experimental "true streaming" code (sample-by-sample lifting) has been
removed because it was incomplete and produced incorrect audio. If you need a
true single-sample-step synthesis filterbank, it should live in its own module
with its own acceptance tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .fast_lifting import idwpt_jit

__all__ = [
    "IDWPTPolyphaseState",
    "idwpt_polyphase_block",
    "idwpt_polyphase_from_flat",
    # Back-compat names (map to polyphase block operation)
    "IDWPTStreamingState",
    "idwpt_streaming_block",
]


@dataclass
class IDWPTPolyphaseState:
    """Optional holder for configuration/state.

    The current production polyphase interleave path is stateless, but keeping a
    state object allows future additions (e.g., seam correction/crossfade) while
    maintaining call sites.
    """

    K: int = 4
    headroom_bits: int = 1


# -----------------------------------------------------------------------------
# Core polyphase interleave
# -----------------------------------------------------------------------------

def idwpt_polyphase_block(
    coeff_2d: np.ndarray,
    *,
    headroom_bits: int = 1,
) -> np.ndarray:
    """Run K IDWPTs and combine into one 64-sample audio block.

    Interleaved Polyphase Reconstruction:
    -------------------------------------
    Each slot k represents a 1-sample phase-shifted version of the signal.
    Reconstruction interleaves outputs: sample n uses slot (n % K).

        out[k::K] = idwpt(slot_k)[0::K]

    This works for any K (4 or 8) with K_STRIDE=1 basis frames.

    Args:
        coeff_2d: np.ndarray shape [64, K], dtype int16 (or int32). leaf-major.
        headroom_bits: passed to idwpt_jit

    Returns:
        np.ndarray[64] int16 audio block
    """
    if coeff_2d.ndim != 2 or coeff_2d.shape[0] != 64:
        raise ValueError(f"coeff_2d must have shape [64, K], got {coeff_2d.shape}")

    K = int(coeff_2d.shape[1])
    out = np.zeros(64, dtype=np.int16)

    # Interleaved reconstruction for any K
    for k in range(K):
        coeffs_k = coeff_2d[:, k].astype(np.int16).tolist()
        audio_k = np.array(idwpt_jit(coeffs_k, headroom_bits=headroom_bits), dtype=np.int16)
        out[k::K] = audio_k[0::K]

    return out


def idwpt_polyphase_from_flat(
    coeff_flat: np.ndarray,
    *,
    K: int = 4,
    headroom_bits: int = 1,
) -> np.ndarray:
    """Convenience wrapper for flat [64*K] coefficient state."""
    coeff_flat = np.asarray(coeff_flat)
    if coeff_flat.size < 64 * K:
        raise ValueError(f"coeff_flat must have at least {64*K} elements, got {coeff_flat.size}")
    coeff_2d = coeff_flat[: 64 * K].reshape(64, K)
    return idwpt_polyphase_block(coeff_2d, headroom_bits=headroom_bits)


# -----------------------------------------------------------------------------
# Back-compat aliases
# -----------------------------------------------------------------------------

IDWPTStreamingState = IDWPTPolyphaseState


def idwpt_streaming_block(
    state: Optional[IDWPTPolyphaseState],
    coeff_state_flat: np.ndarray,
    headroom_bits: int = 1,
) -> np.ndarray:
    """Back-compat wrapper.

    If `state` is provided, its K/headroom_bits are used unless explicitly
    overridden via arguments.
    """
    if state is not None:
        K = int(getattr(state, "K", 4))
        hb = int(getattr(state, "headroom_bits", headroom_bits))
    else:
        K = 4
        hb = headroom_bits
    return idwpt_polyphase_from_flat(coeff_state_flat, K=K, headroom_bits=hb)
