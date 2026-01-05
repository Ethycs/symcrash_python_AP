"""
Fast CDF 5/3 Lifting for Real-Time Audio
=========================================

Optimized lifting transform for the audio render path.
Uses simplified arithmetic (no bit-exact RSHIFT_RNE) for speed.

For bit-exact Verilog-compatible code, use wavelet_packet_trees.py.

Performance comparison (64-sample blocks):
- Standard (bit-exact): ~1200 µs/block
- Fast path (this): ~650 µs/block
- Fused path: ~350 µs/block

The difference is in rounding mode:
- Standard: RSHIFT_RNE (round-to-nearest, ties-to-even) - 15+ branches per call
- Fast: (x + bias) >> n - single operation

For audio, the rounding difference is inaudible (max 1 LSB).
"""

from typing import List, Tuple, Optional
import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

__all__ = [
    'FastCDF53', 'fwpt_balanced', 'idwpt_balanced', 'effect_mode_fast',
    'FusedWPT', 'effect_mode_jit', 'NUMBA_AVAILABLE',
    # Separated FWPT/IDWPT (CCS architecture)
    'fwpt_jit', 'idwpt_jit',
]


def _sat16(x: int) -> int:
    """Saturate to int16 range."""
    if x > 32767:
        return 32767
    if x < -32768:
        return -32768
    return x


class FastCDF53:
    """
    Fast CDF 5/3 lifting with simplified rounding.

    For real-time audio only. Use CDF53Lifting for bit-exact paths.
    """

    @staticmethod
    def forward_step(x: List[int]) -> Tuple[List[int], List[int]]:
        """
        Forward CDF 5/3 lifting step (fast path).

        x[0..2n-1] -> s[0..n-1] (approx), d[0..n-1] (detail)
        """
        n = len(x) // 2
        if n == 0:
            return x, []

        s = [0] * n
        d = [0] * n

        # Predict: d[i] = x[2i+1] - (x[2i] + x[2i+2]) / 2
        for i in range(n):
            left = x[2 * i]
            right = x[2 * i + 2] if 2 * i + 2 < len(x) else x[2 * i]
            avg = (left + right + 1) >> 1  # Round-to-nearest (fast)
            d[i] = _sat16(x[2 * i + 1] - avg)

        # Update: s[i] = x[2i] + (d[i-1] + d[i]) / 4
        for i in range(n):
            d_left = d[i - 1] if i > 0 else d[0]
            d_right = d[i]
            update = (d_left + d_right + 2) >> 2  # Round-to-nearest (fast)
            s[i] = _sat16(x[2 * i] + update)

        return s, d

    @staticmethod
    def inverse_step(s: List[int], d: List[int]) -> List[int]:
        """
        Inverse CDF 5/3 lifting step (fast path).

        s[0..n-1], d[0..m-1] -> x[0..n+m-1]
        """
        n_s = len(s)
        n_d = len(d)

        if n_s == 0:
            return []
        if n_d == 0:
            return list(s)

        n = max(n_s, n_d)

        # Pad if needed
        if n_s < n:
            s = list(s) + [s[-1]] * (n - n_s)
        else:
            s = list(s)
        if n_d < n:
            d = list(d) + [0] * (n - n_d)
        else:
            d = list(d)

        x = [0] * (2 * n)

        # Undo update: x[2i] = s[i] - (d[i-1] + d[i]) / 4
        for i in range(n):
            d_left = d[i - 1] if i > 0 else d[0]
            d_right = d[i]
            update = (d_left + d_right + 2) >> 2
            x[2 * i] = _sat16(s[i] - update)

        # Undo predict: x[2i+1] = d[i] + (x[2i] + x[2i+2]) / 2
        for i in range(n):
            left = x[2 * i]
            right = x[2 * i + 2] if 2 * i + 2 < 2 * n else x[2 * i]
            avg = (left + right + 1) >> 1
            x[2 * i + 1] = _sat16(d[i] + avg)

        return x


# Pre-instantiate for convenience
_lifting = FastCDF53()


def fwpt_balanced(x: List[int], depth: int = 6) -> List[List[int]]:
    """
    Fast Forward Wavelet Packet Transform (balanced tree).

    Args:
        x: Input samples (64 for standard block)
        depth: Tree depth (6 for 64 leaves)

    Returns:
        List of 2^depth leaf coefficient lists
    """
    queue = [list(x)]

    for _ in range(depth):
        next_queue = []
        for node in queue:
            if len(node) >= 2:
                s, d = _lifting.forward_step(node)
                next_queue.append(s)
                next_queue.append(d)
            else:
                next_queue.append(node)
        queue = next_queue

    return queue


def idwpt_balanced(leaves: List[List[int]], depth: int = 6) -> List[int]:
    """
    Fast Inverse Wavelet Packet Transform (balanced tree).

    Args:
        leaves: List of 2^depth leaf coefficient lists
        depth: Tree depth (6 for 64 leaves)

    Returns:
        Reconstructed samples (64 for standard block)
    """
    queue = [list(l) for l in leaves]

    for _ in range(depth):
        next_queue = []
        for i in range(0, len(queue), 2):
            s = queue[i] if i < len(queue) else [0]
            d = queue[i + 1] if i + 1 < len(queue) else []
            combined = _lifting.inverse_step(s, d)
            next_queue.append(combined)
        queue = next_queue

    return queue[0] if queue else [0] * 64


def effect_mode_fast(
    carrier: List[int],
    mask_q15: List[int] = None,
    headroom_bits: int = 3
) -> List[int]:
    """
    Fast effect mode: carrier -> FWPT -> mask -> IDWPT.

    This is the fast path for real-time audio. Uses simplified
    rounding for speed while maintaining audio-quality accuracy.

    Args:
        carrier: 64-sample carrier audio (Q15)
        mask_q15: Optional 64-element mask (Q15), one per leaf
        headroom_bits: Headroom for wavelet transform (default 3)

    Returns:
        64-sample wet audio (Q15)
    """
    # Scale down input for headroom
    scaled = [s >> headroom_bits for s in carrier]

    # FWPT
    leaves = fwpt_balanced(scaled, depth=6)

    # Apply mask if provided
    if mask_q15 is not None:
        for i, leaf in enumerate(leaves):
            if i < len(mask_q15):
                mask_val = mask_q15[i]
                # Scale each coefficient by mask (Q15 multiply)
                leaves[i] = [_sat16((c * mask_val) >> 15) for c in leaf]

    # IDWPT
    result = idwpt_balanced(leaves, depth=6)

    # Scale back up
    output = []
    for s in result[:64]:
        scaled_up = s << headroom_bits
        output.append(_sat16(scaled_up))

    return output


# =============================================================================
# FUSED FORWARD + INVERSE (PURE PYTHON, PRE-ALLOCATED)
# =============================================================================

class FusedWPT:
    """
    Fused Forward + Inverse Wavelet Packet Transform.

    Uses pure Python lists with pre-allocation for maximum speed.
    NumPy is slower for 64-element arrays due to overhead.

    For a 64-sample block with 6 levels:
    - Level 0: 1 node × 64 samples
    - Level 1: 2 nodes × 32 samples
    - Level 2: 4 nodes × 16 samples
    - Level 3: 8 nodes × 8 samples
    - Level 4: 16 nodes × 4 samples
    - Level 5: 32 nodes × 2 samples
    - Level 6: 64 nodes × 1 sample (leaves)
    """

    def __init__(self, block_size: int = 64, depth: int = 6):
        self.block_size = block_size
        self.depth = depth

        # Pre-allocate flat Python lists for each level
        self.fwd_levels = [[0] * block_size for _ in range(depth + 1)]
        self.inv_levels = [[0] * block_size for _ in range(depth + 1)]

        # Temp buffers for lifting
        self.temp_s = [0] * block_size
        self.temp_d = [0] * block_size

    def effect_mode(
        self,
        carrier: List[int],
        mask_q15: Optional[List[int]] = None,
        headroom_bits: int = 3
    ) -> List[int]:
        """
        Fused effect mode: carrier -> FWPT -> mask -> IDWPT.

        All operations use pre-allocated buffers.
        """
        block_size = self.block_size
        depth = self.depth
        fwd = self.fwd_levels
        inv = self.inv_levels
        temp_s = self.temp_s
        temp_d = self.temp_d

        # Scale down input and copy to level 0
        level0 = fwd[0]
        for i in range(block_size):
            level0[i] = carrier[i] >> headroom_bits

        # ===== FORWARD TRANSFORM =====
        for level in range(depth):
            src = fwd[level]
            dst = fwd[level + 1]
            num_nodes = 1 << level
            node_size = block_size >> level

            for node in range(num_nodes):
                start = node * node_size
                half = node_size >> 1
                if half == 0:
                    dst[node] = src[start]
                    continue

                # Predict: d[i] = odd[i] - (even[i] + even[i+1]) / 2
                for i in range(half):
                    idx = start + 2 * i
                    even_left = src[idx]
                    even_right = src[idx + 2] if idx + 2 < start + node_size else even_left
                    odd = src[idx + 1]
                    avg = (even_left + even_right + 1) >> 1
                    temp_d[i] = odd - avg

                # Update: s[i] = even[i] + (d[i-1] + d[i]) / 4
                for i in range(half):
                    d_left = temp_d[i - 1] if i > 0 else temp_d[0]
                    d_right = temp_d[i]
                    update = (d_left + d_right + 2) >> 2
                    temp_s[i] = src[start + 2 * i] + update

                # Write to next level
                out_half = half
                approx_start = node * 2 * out_half
                detail_start = approx_start + out_half
                for i in range(half):
                    v = temp_s[i]
                    dst[approx_start + i] = -32768 if v < -32768 else (32767 if v > 32767 else v)
                    v = temp_d[i]
                    dst[detail_start + i] = -32768 if v < -32768 else (32767 if v > 32767 else v)

        # ===== APPLY MASK AT LEAVES =====
        leaves = fwd[depth]
        inv_leaves = inv[depth]

        if mask_q15 is not None:
            for i in range(block_size):
                mask_val = mask_q15[i] if i < len(mask_q15) else 0
                v = (leaves[i] * mask_val) >> 15
                inv_leaves[i] = -32768 if v < -32768 else (32767 if v > 32767 else v)
        else:
            for i in range(block_size):
                inv_leaves[i] = leaves[i]

        # ===== INVERSE TRANSFORM =====
        for level in range(depth - 1, -1, -1):
            src = inv[level + 1]
            dst = inv[level]
            num_nodes = 1 << level
            node_size = block_size >> level

            for node in range(num_nodes):
                out_start = node * node_size
                half = node_size >> 1
                if half == 0:
                    dst[out_start] = src[node]
                    continue

                in_half = half
                approx_start = node * 2 * in_half
                detail_start = approx_start + in_half

                # Undo update
                for i in range(half):
                    s_val = src[approx_start + i]
                    d_left = src[detail_start + i - 1] if i > 0 else src[detail_start]
                    d_right = src[detail_start + i]
                    update = (d_left + d_right + 2) >> 2
                    temp_s[i] = s_val - update

                # Undo predict
                for i in range(half):
                    even_left = temp_s[i]
                    even_right = temp_s[i + 1] if i + 1 < half else temp_s[i]
                    avg = (even_left + even_right + 1) >> 1
                    temp_d[i] = src[detail_start + i] + avg

                # Interleave to output
                for i in range(half):
                    v = temp_s[i]
                    dst[out_start + 2 * i] = -32768 if v < -32768 else (32767 if v > 32767 else v)
                    v = temp_d[i]
                    dst[out_start + 2 * i + 1] = -32768 if v < -32768 else (32767 if v > 32767 else v)

        # ===== SCALE BACK UP =====
        result = inv[0]
        output = [0] * block_size
        for i in range(block_size):
            v = result[i] << headroom_bits
            output[i] = -32768 if v < -32768 else (32767 if v > 32767 else v)

        return output


# Global instance
_fused_wpt: Optional[FusedWPT] = None


def _get_fused_wpt() -> FusedWPT:
    """Get or create the global FusedWPT instance."""
    global _fused_wpt
    if _fused_wpt is None:
        _fused_wpt = FusedWPT(64, 6)
    return _fused_wpt


def effect_mode_fused(
    carrier: List[int],
    mask_q15: Optional[List[int]] = None,
    headroom_bits: int = 3
) -> List[int]:
    """
    Fused effect mode using pre-allocated buffers.

    Drop-in replacement for effect_mode_fast with better performance.
    """
    return _get_fused_wpt().effect_mode(carrier, mask_q15, headroom_bits)


# =============================================================================
# NUMBA JIT VERSION
# =============================================================================

@njit(cache=True, fastmath=True)
def _sat16_jit(x: int) -> int:
    """Saturate to int16 range (JIT version)."""
    if x > 32767:
        return 32767
    if x < -32768:
        return -32768
    return x


@njit(cache=True, fastmath=True)
def _effect_mode_jit_core(
    carrier: np.ndarray,
    mask_q15: np.ndarray,
    use_mask: bool,
    headroom_bits: int,
    output: np.ndarray,
    fwd_levels: np.ndarray,
    inv_levels: np.ndarray,
    temp_s: np.ndarray,
    temp_d: np.ndarray,
) -> None:
    """
    JIT-compiled core of effect_mode.

    All arrays must be pre-allocated int32 for numerical headroom.
    Output is written in-place to `output`.
    """
    block_size = 64
    depth = 6

    # Scale down input and copy to level 0
    for i in range(block_size):
        fwd_levels[0, i] = carrier[i] >> headroom_bits

    # ===== FORWARD TRANSFORM =====
    for level in range(depth):
        num_nodes = 1 << level
        node_size = block_size >> level

        for node in range(num_nodes):
            start = node * node_size
            half = node_size >> 1
            if half == 0:
                fwd_levels[level + 1, node] = fwd_levels[level, start]
                continue

            # Predict: d[i] = odd[i] - (even[i] + even[i+1]) / 2
            for i in range(half):
                idx = start + 2 * i
                even_left = fwd_levels[level, idx]
                if idx + 2 < start + node_size:
                    even_right = fwd_levels[level, idx + 2]
                else:
                    even_right = even_left
                odd = fwd_levels[level, idx + 1]
                avg = (even_left + even_right + 1) >> 1
                temp_d[i] = odd - avg

            # Update: s[i] = even[i] + (d[i-1] + d[i]) / 4
            for i in range(half):
                if i > 0:
                    d_left = temp_d[i - 1]
                else:
                    d_left = temp_d[0]
                d_right = temp_d[i]
                update = (d_left + d_right + 2) >> 2
                temp_s[i] = fwd_levels[level, start + 2 * i] + update

            # Write to next level with saturation
            approx_start = node * 2 * half
            detail_start = approx_start + half
            for i in range(half):
                v = temp_s[i]
                if v < -32768:
                    v = -32768
                elif v > 32767:
                    v = 32767
                fwd_levels[level + 1, approx_start + i] = v

                v = temp_d[i]
                if v < -32768:
                    v = -32768
                elif v > 32767:
                    v = 32767
                fwd_levels[level + 1, detail_start + i] = v

    # ===== APPLY MASK AT LEAVES =====
    if use_mask:
        for i in range(block_size):
            # Add rounding (+16384) to avoid systematic negative bias from truncation
            v = (fwd_levels[depth, i] * mask_q15[i] + 16384) >> 15
            if v < -32768:
                v = -32768
            elif v > 32767:
                v = 32767
            inv_levels[depth, i] = v
    else:
        for i in range(block_size):
            inv_levels[depth, i] = fwd_levels[depth, i]

    # ===== INVERSE TRANSFORM =====
    for level in range(depth - 1, -1, -1):
        num_nodes = 1 << level
        node_size = block_size >> level

        for node in range(num_nodes):
            out_start = node * node_size
            half = node_size >> 1
            if half == 0:
                inv_levels[level, out_start] = inv_levels[level + 1, node]
                continue

            approx_start = node * 2 * half
            detail_start = approx_start + half

            # Undo update (MUST match forward rounding for perfect reconstruction)
            # Forward used: update = (d_left + d_right + 2) >> 2
            # Inverse must use SAME rounding to avoid cumulative error
            for i in range(half):
                s_val = inv_levels[level + 1, approx_start + i]
                if i > 0:
                    d_left = inv_levels[level + 1, detail_start + i - 1]
                else:
                    d_left = inv_levels[level + 1, detail_start]
                d_right = inv_levels[level + 1, detail_start + i]
                update = (d_left + d_right + 2) >> 2  # +2 matches forward transform
                temp_s[i] = s_val - update

            # Undo predict (MUST match forward rounding for perfect reconstruction)
            # Forward used: avg = (even_left + even_right + 1) >> 1
            # Inverse must use SAME rounding to avoid cumulative error
            for i in range(half):
                even_left = temp_s[i]
                if i + 1 < half:
                    even_right = temp_s[i + 1]
                else:
                    even_right = temp_s[i]
                avg = (even_left + even_right + 1) >> 1  # +1 matches forward transform
                temp_d[i] = inv_levels[level + 1, detail_start + i] + avg

            # Interleave to output with saturation
            for i in range(half):
                v = temp_s[i]
                if v < -32768:
                    v = -32768
                elif v > 32767:
                    v = 32767
                inv_levels[level, out_start + 2 * i] = v

                v = temp_d[i]
                if v < -32768:
                    v = -32768
                elif v > 32767:
                    v = 32767
                inv_levels[level, out_start + 2 * i + 1] = v

    # ===== SCALE BACK UP =====
    for i in range(block_size):
        v = inv_levels[0, i] << headroom_bits
        if v < -32768:
            v = -32768
        elif v > 32767:
            v = 32767
        output[i] = v


class JITWaveletProcessor:
    """
    JIT-accelerated wavelet processor with pre-allocated buffers.

    Uses Numba JIT for ~10-20x speedup over pure Python.
    """

    def __init__(self, block_size: int = 64, depth: int = 6):
        self.block_size = block_size
        self.depth = depth

        # Pre-allocate numpy arrays (int32 for headroom during computation)
        self.fwd_levels = np.zeros((depth + 1, block_size), dtype=np.int32)
        self.inv_levels = np.zeros((depth + 1, block_size), dtype=np.int32)
        self.temp_s = np.zeros(block_size, dtype=np.int32)
        self.temp_d = np.zeros(block_size, dtype=np.int32)
        self.output = np.zeros(block_size, dtype=np.int32)
        self._carrier_buf = np.zeros(block_size, dtype=np.int32)
        self._mask_buf = np.zeros(block_size, dtype=np.int32)

    def effect_mode(
        self,
        carrier: List[int],
        mask_q15: Optional[List[int]] = None,
        headroom_bits: int = 3
    ) -> List[int]:
        """
        JIT-accelerated effect mode: carrier -> FWPT -> mask -> IDWPT.
        """
        # Convert input to numpy
        for i in range(min(len(carrier), self.block_size)):
            self._carrier_buf[i] = carrier[i]

        use_mask = mask_q15 is not None
        if use_mask:
            for i in range(min(len(mask_q15), self.block_size)):
                self._mask_buf[i] = mask_q15[i]

        # Call JIT core
        _effect_mode_jit_core(
            self._carrier_buf,
            self._mask_buf,
            use_mask,
            headroom_bits,
            self.output,
            self.fwd_levels,
            self.inv_levels,
            self.temp_s,
            self.temp_d,
        )

        # Convert back to list
        return [int(self.output[i]) for i in range(self.block_size)]


# Global JIT instance
_jit_processor: Optional[JITWaveletProcessor] = None


def _get_jit_processor() -> JITWaveletProcessor:
    """Get or create the global JIT processor instance."""
    global _jit_processor
    if _jit_processor is None:
        _jit_processor = JITWaveletProcessor(64, 6)
    return _jit_processor


def effect_mode_jit(
    carrier: List[int],
    mask_q15: Optional[List[int]] = None,
    headroom_bits: int = 1
) -> List[int]:
    """
    JIT-accelerated effect mode using Numba.

    Drop-in replacement for effect_mode_fused with ~10-20x better performance.
    Falls back to pure Python if Numba is not available.

    Note: headroom_bits=1 gives ~32 LSB error vs headroom_bits=3 giving ~94 LSB.
    Since carrier is already scaled down by DC_HEADROOM (~0.9456), we can use
    less headroom for better reconstruction accuracy.
    """
    if NUMBA_AVAILABLE:
        return _get_jit_processor().effect_mode(carrier, mask_q15, headroom_bits)
    else:
        return effect_mode_fused(carrier, mask_q15, headroom_bits)


# =============================================================================
# SEPARATED FWPT / IDWPT (CCS Architecture)
# =============================================================================
# These functions separate the forward and inverse transforms for the CCS path.
# FWPT is now only used for analysis (not critical path).
# IDWPT is used by CoeffOsc to convert coefficients directly to audio.

@njit(cache=True, fastmath=True)
def _fwpt_jit_core(
    input_buf: np.ndarray,
    headroom_bits: int,
    output_coeffs: np.ndarray,
    fwd_levels: np.ndarray,
    temp_s: np.ndarray,
    temp_d: np.ndarray,
) -> None:
    """
    JIT-compiled forward wavelet packet transform only.

    Args:
        input_buf: int32 input samples (64 samples)
        headroom_bits: bits to shift down for headroom
        output_coeffs: int32 output coefficients (64 leaves)
        fwd_levels: working buffer for levels (7 x 64)
        temp_s: temp buffer for approx coeffs
        temp_d: temp buffer for detail coeffs
    """
    block_size = 64
    depth = 6

    # Scale down input and copy to level 0
    for i in range(block_size):
        fwd_levels[0, i] = input_buf[i] >> headroom_bits

    # Forward transform
    for level in range(depth):
        num_nodes = 1 << level
        node_size = block_size >> level

        for node in range(num_nodes):
            start = node * node_size
            half = node_size >> 1
            if half == 0:
                fwd_levels[level + 1, node] = fwd_levels[level, start]
                continue

            # Predict: d[i] = odd[i] - (even[i] + even[i+1]) / 2
            for i in range(half):
                idx = start + 2 * i
                even_left = fwd_levels[level, idx]
                if idx + 2 < start + node_size:
                    even_right = fwd_levels[level, idx + 2]
                else:
                    even_right = even_left
                odd = fwd_levels[level, idx + 1]
                avg = (even_left + even_right + 1) >> 1
                temp_d[i] = odd - avg

            # Update: s[i] = even[i] + (d[i-1] + d[i]) / 4
            for i in range(half):
                if i > 0:
                    d_left = temp_d[i - 1]
                else:
                    d_left = temp_d[0]
                d_right = temp_d[i]
                update = (d_left + d_right + 2) >> 2
                temp_s[i] = fwd_levels[level, start + 2 * i] + update

            # Write to next level with saturation
            approx_start = node * 2 * half
            detail_start = approx_start + half
            for i in range(half):
                v = temp_s[i]
                if v < -32768:
                    v = -32768
                elif v > 32767:
                    v = 32767
                fwd_levels[level + 1, approx_start + i] = v

                v = temp_d[i]
                if v < -32768:
                    v = -32768
                elif v > 32767:
                    v = 32767
                fwd_levels[level + 1, detail_start + i] = v

    # Copy leaf coefficients to output
    for i in range(block_size):
        output_coeffs[i] = fwd_levels[depth, i]


@njit(cache=True, fastmath=True)
def _idwpt_jit_core(
    input_coeffs: np.ndarray,
    headroom_bits: int,
    output_buf: np.ndarray,
    inv_levels: np.ndarray,
    temp_s: np.ndarray,
    temp_d: np.ndarray,
) -> None:
    """
    JIT-compiled inverse wavelet packet transform only.

    Args:
        input_coeffs: int32 leaf coefficients (64 values)
        headroom_bits: bits to shift up after transform
        output_buf: int32 output samples (64 samples)
        inv_levels: working buffer for levels (7 x 64)
        temp_s: temp buffer for approx coeffs
        temp_d: temp buffer for detail coeffs
    """
    block_size = 64
    depth = 6

    # Copy input coefficients to leaf level
    for i in range(block_size):
        inv_levels[depth, i] = input_coeffs[i]

    # Inverse transform
    for level in range(depth - 1, -1, -1):
        num_nodes = 1 << level
        node_size = block_size >> level

        for node in range(num_nodes):
            out_start = node * node_size
            half = node_size >> 1
            if half == 0:
                inv_levels[level, out_start] = inv_levels[level + 1, node]
                continue

            approx_start = node * 2 * half
            detail_start = approx_start + half

            # Undo update (MUST match forward rounding for perfect reconstruction)
            for i in range(half):
                s_val = inv_levels[level + 1, approx_start + i]
                if i > 0:
                    d_left = inv_levels[level + 1, detail_start + i - 1]
                else:
                    d_left = inv_levels[level + 1, detail_start]
                d_right = inv_levels[level + 1, detail_start + i]
                update = (d_left + d_right + 2) >> 2
                temp_s[i] = s_val - update

            # Undo predict (MUST match forward rounding for perfect reconstruction)
            for i in range(half):
                even_left = temp_s[i]
                if i + 1 < half:
                    even_right = temp_s[i + 1]
                else:
                    even_right = temp_s[i]
                avg = (even_left + even_right + 1) >> 1
                temp_d[i] = inv_levels[level + 1, detail_start + i] + avg

            # Interleave to output with saturation
            for i in range(half):
                v = temp_s[i]
                if v < -32768:
                    v = -32768
                elif v > 32767:
                    v = 32767
                inv_levels[level, out_start + 2 * i] = v

                v = temp_d[i]
                if v < -32768:
                    v = -32768
                elif v > 32767:
                    v = 32767
                inv_levels[level, out_start + 2 * i + 1] = v

    # Scale back up
    for i in range(block_size):
        v = inv_levels[0, i] << headroom_bits
        if v < -32768:
            v = -32768
        elif v > 32767:
            v = 32767
        output_buf[i] = v


class SeparatedWaveletProcessor:
    """
    Separated FWPT/IDWPT processor for CCS architecture.

    Unlike the fused effect_mode, this allows:
    - FWPT for analysis only (not in critical path)
    - IDWPT from CoeffOsc coefficients (direct synthesis)
    """

    def __init__(self, block_size: int = 64, depth: int = 6):
        self.block_size = block_size
        self.depth = depth

        # Pre-allocate numpy arrays
        self.fwd_levels = np.zeros((depth + 1, block_size), dtype=np.int32)
        self.inv_levels = np.zeros((depth + 1, block_size), dtype=np.int32)
        self.temp_s = np.zeros(block_size, dtype=np.int32)
        self.temp_d = np.zeros(block_size, dtype=np.int32)
        self.output = np.zeros(block_size, dtype=np.int32)
        self._input_buf = np.zeros(block_size, dtype=np.int32)
        self._coeff_buf = np.zeros(block_size, dtype=np.int32)

    def fwpt(self, samples: List[int], headroom_bits: int = 1) -> List[int]:
        """
        Forward wavelet packet transform only.

        Args:
            samples: 64 int16 audio samples
            headroom_bits: bits to shift down for numerical headroom

        Returns:
            64 leaf coefficients (int16 range)
        """
        for i in range(min(len(samples), self.block_size)):
            self._input_buf[i] = samples[i]

        _fwpt_jit_core(
            self._input_buf,
            headroom_bits,
            self._coeff_buf,
            self.fwd_levels,
            self.temp_s,
            self.temp_d,
        )

        return [int(self._coeff_buf[i]) for i in range(self.block_size)]

    def idwpt(self, coeffs: List[int], headroom_bits: int = 1) -> List[int]:
        """
        Inverse wavelet packet transform only.

        Args:
            coeffs: 64 leaf coefficients (int16 range)
            headroom_bits: bits to shift up after transform

        Returns:
            64 audio samples (int16 range)
        """
        for i in range(min(len(coeffs), self.block_size)):
            self._coeff_buf[i] = coeffs[i]

        _idwpt_jit_core(
            self._coeff_buf,
            headroom_bits,
            self.output,
            self.inv_levels,
            self.temp_s,
            self.temp_d,
        )

        return [int(self.output[i]) for i in range(self.block_size)]


# Global separated processor instance
_separated_processor: Optional[SeparatedWaveletProcessor] = None


def _get_separated_processor() -> SeparatedWaveletProcessor:
    """Get or create the global separated processor instance."""
    global _separated_processor
    if _separated_processor is None:
        _separated_processor = SeparatedWaveletProcessor(64, 6)
    return _separated_processor


def fwpt_jit(samples: List[int], headroom_bits: int = 1) -> List[int]:
    """
    JIT-accelerated forward wavelet packet transform.

    This is the analysis-only path, used for coefficient extraction.
    Not in the critical audio rendering path for CCS architecture.

    Args:
        samples: 64 int16 audio samples
        headroom_bits: bits to shift down for numerical headroom

    Returns:
        64 leaf coefficients (int16 range)
    """
    if NUMBA_AVAILABLE:
        return _get_separated_processor().fwpt(samples, headroom_bits)
    else:
        # Fallback to fwpt_balanced
        leaves = fwpt_balanced(samples, depth=6)
        return [leaf[0] if leaf else 0 for leaf in leaves]


def idwpt_jit(coeffs: List[int], headroom_bits: int = 1) -> List[int]:
    """
    JIT-accelerated inverse wavelet packet transform.

    This is the synthesis path used by CCS architecture.
    Converts CoeffOsc coefficients directly to audio samples.

    Args:
        coeffs: 64 leaf coefficients (int16 range)
        headroom_bits: bits to shift up after transform

    Returns:
        64 audio samples (int16 range)
    """
    if NUMBA_AVAILABLE:
        return _get_separated_processor().idwpt(coeffs, headroom_bits)
    else:
        # Fallback to idwpt_balanced
        leaves = [[c] for c in coeffs]
        return idwpt_balanced(leaves, depth=6)


def idwpt_polyphase_streaming(
    coeff_state: np.ndarray,
    headroom_bits: int = 1
) -> np.ndarray:
    """
    Streaming polyphase IDWPT per WSE-STD-120 section 4.1.

    Treats coeff_state as persistent time-series data:
        coeff_state[leaf][k] where k represents time samples

    Structure options:
    - [64 leaves][4 time_samples] = 256 coefficients
    - [32 nodes][8 time_samples] = 256 coefficients

    This function processes the coefficient time-series to generate
    audio continuously without block-boundary artifacts.

    Args:
        coeff_state: np.ndarray[256] int16, viewed as [64, 4] or [32, 8]
        headroom_bits: bits to shift up after transform

    Returns:
        np.ndarray[64] int16 audio samples
    """
    # Interpret as [64 leaves][4 time_samples]
    coeff_2d = coeff_state[:256].reshape(64, 4)

    # Process each of the 4 time slots. Each slot produces a full 64-sample
    # reconstruction for that phase position; we then take every 4th sample
    # from that reconstruction (indices t, t+4, t+8, ...) and interleave.
    output = np.zeros(64, dtype=np.int16)

    for t in range(4):
        coeffs_t = coeff_2d[:, t].tolist()  # [64] coefficients at slot t
        audio_full = idwpt_jit(coeffs_t, headroom_bits=headroom_bits)
        # Take the polyphase-aligned samples from this reconstruction
        for out_idx in range(t, 64, 4):
            output[out_idx] = audio_full[out_idx]

    return output
