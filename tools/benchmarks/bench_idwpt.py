#!/usr/bin/env python3
"""Benchmark IDWPT effect_mode performance."""

import time
import numpy as np
from symcrash_python_ap.wavelet.wavelet_packet_trees import PacketTreeSet, PacketAnalyzer, PacketSynthesizer

pts = PacketTreeSet()

# Create test carrier
carrier = [int(16384 * np.sin(2 * np.pi * 440 * i / 48000)) for i in range(64)]
mask = [32767] * 64  # Unity mask

# Warmup
for _ in range(10):
    pts.effect_mode(0, carrier, mask)

# Benchmark effect_mode
N = 1000
start = time.perf_counter()
for _ in range(N):
    pts.effect_mode(0, carrier, mask)
elapsed = time.perf_counter() - start
us_per_block = elapsed * 1e6 / N
print(f"effect_mode (Python): {us_per_block:.1f} us/block")

# Benchmark individual components
analyzer = pts.analyzers[0]
synth = pts.synthesizers[0]

# FWPT only
start = time.perf_counter()
for _ in range(N):
    nodes = analyzer.analyze(carrier)
elapsed = time.perf_counter() - start
print(f"  FWPT (analyze): {elapsed * 1e6 / N:.1f} us/block")

# Get sample nodes for synthesis test
nodes = analyzer.analyze(carrier)

# IDWPT only (synthesize_full)
start = time.perf_counter()
for _ in range(N):
    synth.synthesize_full(nodes)
elapsed = time.perf_counter() - start
print(f"  IDWPT (synthesize_full): {elapsed * 1e6 / N:.1f} us/block")

# Mask application
from symcrash_python_ap.fixed import mul_q15_sat, Q15_ONE
LEAF_COUNT = 64
NODE_SIZE = 4

start = time.perf_counter()
for _ in range(N):
    masked_nodes = []
    for leaf_idx in range(LEAF_COUNT):
        node = nodes[leaf_idx] if leaf_idx < len(nodes) else [0]
        masked_node = []
        mask_val = mask[leaf_idx] if leaf_idx < len(mask) else Q15_ONE
        for coeff in node:
            scaled = mul_q15_sat(coeff, mask_val)
            masked_node.append(scaled)
        masked_nodes.append(masked_node)
elapsed = time.perf_counter() - start
print(f"  Mask application: {elapsed * 1e6 / N:.1f} us/block")

print()
print(f"Realtime budget: 1333 us/block")
total = us_per_block
print(f"Headroom: {1333 - total:.1f} us ({(1333 - total) / 1333 * 100:.0f}%)")

# =============================================================================
# Now test NumPy-vectorized CDF 5/3 lifting
# =============================================================================

print("\n" + "="*60)
print("NumPy-vectorized CDF 5/3 lifting")
print("="*60)

def sat16(x):
    """Saturate to int16 range."""
    return np.clip(x, -32768, 32767).astype(np.int16)

def shift_round(x, n):
    """Round-to-nearest right shift."""
    bias = (1 << (n - 1)) if n > 0 else 0
    return (x + bias) >> n

def cdf53_forward_np(x):
    """
    NumPy-vectorized CDF 5/3 forward step.
    x[0..2n-1] -> s[0..n-1] (approx), d[0..n-1] (detail)
    """
    x = np.asarray(x, dtype=np.int32)
    n = len(x) // 2
    if n == 0:
        return x, np.array([], dtype=np.int32)

    # Step 1: Predict (compute detail)
    # d[i] = x[2i+1] - (x[2i] + x[2i+2]) / 2
    evens = x[::2]
    odds = x[1::2]

    # Symmetric extension for boundary
    evens_ext = np.concatenate([evens, [evens[-1]]])

    avg = shift_round(evens_ext[:-1] + evens_ext[1:], 1)
    d = sat16(odds - avg)

    # Step 2: Update (compute approximation)
    # s[i] = x[2i] + (d[i-1] + d[i]) / 4
    d_ext = np.concatenate([[d[0]], d])
    update = shift_round(d_ext[:-1] + d_ext[1:], 2)
    s = sat16(evens + update)

    return s, d

def cdf53_inverse_np(s, d):
    """
    NumPy-vectorized CDF 5/3 inverse step.
    s[0..n-1], d[0..n-1] -> x[0..2n-1]
    """
    s = np.asarray(s, dtype=np.int32)
    d = np.asarray(d, dtype=np.int32)

    n = len(s)
    if n == 0:
        return np.array([], dtype=np.int16)
    if len(d) == 0:
        return s.astype(np.int16)

    # Pad if needed
    if len(d) < n:
        d = np.concatenate([d, np.zeros(n - len(d), dtype=np.int32)])

    x = np.zeros(2 * n, dtype=np.int32)

    # Step 1: Undo update (recover even samples)
    # x[2i] = s[i] - (d[i-1] + d[i]) / 4
    d_ext = np.concatenate([[d[0]], d])
    update = shift_round(d_ext[:-1] + d_ext[1:], 2)
    x[::2] = sat16(s - update)

    # Step 2: Undo predict (recover odd samples)
    # x[2i+1] = d[i] + (x[2i] + x[2i+2]) / 2
    evens = x[::2]
    evens_ext = np.concatenate([evens, [evens[-1]]])
    avg = shift_round(evens_ext[:-1] + evens_ext[1:], 1)
    x[1::2] = sat16(d[:n] + avg)

    return x.astype(np.int16)

# Test correctness
test_signal = np.array(carrier, dtype=np.int32)
s_np, d_np = cdf53_forward_np(test_signal)
reconstructed = cdf53_inverse_np(s_np, d_np)
error = np.max(np.abs(test_signal[:len(reconstructed)] - reconstructed))
print(f"Roundtrip error: {error}")

# Benchmark single-level lifting
start = time.perf_counter()
for _ in range(N):
    s, d = cdf53_forward_np(test_signal)
    _ = cdf53_inverse_np(s, d)
elapsed = time.perf_counter() - start
print(f"Single-level forward+inverse (NumPy): {elapsed * 1e6 / N:.1f} us")

# Benchmark full tree (6 levels)
def fwpt_np_balanced(x):
    """NumPy FWPT for balanced tree."""
    x = np.asarray(x, dtype=np.int32)
    leaves = []
    queue = [x]

    for depth in range(6):
        next_queue = []
        for node in queue:
            if len(node) >= 2:
                s, d = cdf53_forward_np(node)
                next_queue.extend([s, d])
            else:
                next_queue.append(node)
        queue = next_queue

    return queue

def idwpt_np_balanced(leaves):
    """NumPy IDWPT for balanced tree."""
    queue = list(leaves)

    for depth in range(5, -1, -1):
        next_queue = []
        for i in range(0, len(queue), 2):
            s = queue[i] if i < len(queue) else np.array([0], dtype=np.int16)
            d = queue[i+1] if i+1 < len(queue) else np.array([], dtype=np.int16)
            combined = cdf53_inverse_np(s, d)
            next_queue.append(combined)
        queue = next_queue

    return queue[0] if queue else np.zeros(64, dtype=np.int16)

# Warmup
carrier_np = np.array(carrier, dtype=np.int32)
for _ in range(10):
    leaves = fwpt_np_balanced(carrier_np)
    _ = idwpt_np_balanced(leaves)

# Benchmark
start = time.perf_counter()
for _ in range(N):
    leaves = fwpt_np_balanced(carrier_np)
elapsed = time.perf_counter() - start
print(f"FWPT balanced tree (NumPy): {elapsed * 1e6 / N:.1f} us/block")

start = time.perf_counter()
for _ in range(N):
    result = idwpt_np_balanced(leaves)
elapsed = time.perf_counter() - start
print(f"IDWPT balanced tree (NumPy): {elapsed * 1e6 / N:.1f} us/block")

# Combined
start = time.perf_counter()
for _ in range(N):
    leaves = fwpt_np_balanced(carrier_np)
    result = idwpt_np_balanced(leaves)
elapsed = time.perf_counter() - start
total_np = elapsed * 1e6 / N
print(f"Full FWPT+IDWPT (NumPy): {total_np:.1f} us/block")

print()
speedup = us_per_block / total_np
print(f"Speedup: {speedup:.1f}x")
print(f"Headroom with NumPy: {1333 - total_np:.1f} us ({(1333 - total_np) / 1333 * 100:.0f}%)")

# =============================================================================
# Try in-place NumPy version with pre-allocated buffers
# =============================================================================
print("\n" + "="*60)
print("In-place NumPy with pre-allocated buffers")
print("="*60)

class FastIDWPT:
    """
    Fast IDWPT using pre-allocated buffers and in-place operations.

    For a balanced tree with 6 levels and 64 samples:
    Level 0: 1 node of 64 samples
    Level 1: 2 nodes of 32 samples
    Level 2: 4 nodes of 16 samples
    Level 3: 8 nodes of 8 samples
    Level 4: 16 nodes of 4 samples
    Level 5: 32 nodes of 2 samples
    Level 6: 64 nodes of 1 sample
    """

    def __init__(self, block_size=64):
        self.block_size = block_size
        self.levels = 6

        # Pre-allocate all level buffers
        # Level i has 2^i nodes of size block_size / 2^i
        self.fwd_buffers = []
        self.inv_buffers = []
        for level in range(self.levels + 1):
            num_nodes = 2 ** level
            node_size = block_size // num_nodes
            # Each level is a flat array of all nodes
            self.fwd_buffers.append(np.zeros(block_size, dtype=np.int32))
            self.inv_buffers.append(np.zeros(block_size, dtype=np.int32))

        # Temp buffers for lifting
        self.temp_d = np.zeros(block_size, dtype=np.int32)
        self.temp_s = np.zeros(block_size, dtype=np.int32)

    def forward(self, x):
        """Forward transform in-place."""
        # Copy input to level 0
        self.fwd_buffers[0][:] = x

        for level in range(self.levels):
            src = self.fwd_buffers[level]
            dst = self.fwd_buffers[level + 1]
            num_nodes = 2 ** level
            node_size = self.block_size // num_nodes

            for node in range(num_nodes):
                start = node * node_size
                half = node_size // 2
                if half == 0:
                    continue

                # Forward lifting on this node
                evens = src[start::2][:half]
                odds = src[start + 1::2][:half]

                # Predict: d = odd - (even_left + even_right) / 2
                # Symmetric extension at boundary
                for i in range(half):
                    left = evens[i]
                    right = evens[i + 1] if i + 1 < half else evens[i]
                    avg = (left + right + 1) >> 1
                    self.temp_d[i] = odds[i] - avg

                # Update: s = even + (d_left + d_right) / 4
                for i in range(half):
                    d_left = self.temp_d[i - 1] if i > 0 else self.temp_d[0]
                    d_right = self.temp_d[i]
                    update = (d_left + d_right + 2) >> 2
                    self.temp_s[i] = evens[i] + update

                # Write to next level: approx to left child, detail to right child
                out_node_size = node_size // 2
                approx_start = node * 2 * out_node_size
                detail_start = approx_start + out_node_size
                dst[approx_start:approx_start + half] = np.clip(self.temp_s[:half], -32768, 32767)
                dst[detail_start:detail_start + half] = np.clip(self.temp_d[:half], -32768, 32767)

        return self.fwd_buffers[self.levels]

    def inverse(self, leaves):
        """Inverse transform in-place."""
        # Copy leaves to deepest level
        self.inv_buffers[self.levels][:] = leaves

        for level in range(self.levels - 1, -1, -1):
            src = self.inv_buffers[level + 1]
            dst = self.inv_buffers[level]
            num_nodes = 2 ** level
            node_size = self.block_size // num_nodes

            for node in range(num_nodes):
                out_start = node * node_size
                half = node_size // 2
                if half == 0:
                    continue

                # Source: approx from left child, detail from right child
                in_node_size = node_size // 2
                approx_start = node * 2 * in_node_size
                detail_start = approx_start + in_node_size
                s = src[approx_start:approx_start + half]
                d = src[detail_start:detail_start + half]

                # Undo update: even = s - (d_left + d_right) / 4
                for i in range(half):
                    d_left = d[i - 1] if i > 0 else d[0]
                    d_right = d[i]
                    update = (d_left + d_right + 2) >> 2
                    self.temp_s[i] = s[i] - update

                # Undo predict: odd = d + (even_left + even_right) / 2
                for i in range(half):
                    left = self.temp_s[i]
                    right = self.temp_s[i + 1] if i + 1 < half else self.temp_s[i]
                    avg = (left + right + 1) >> 1
                    self.temp_d[i] = d[i] + avg

                # Interleave to output
                for i in range(half):
                    dst[out_start + 2 * i] = np.clip(self.temp_s[i], -32768, 32767)
                    dst[out_start + 2 * i + 1] = np.clip(self.temp_d[i], -32768, 32767)

        return self.inv_buffers[0]

# Create and test
fast_idwpt = FastIDWPT(64)

# Warmup
for _ in range(10):
    leaves = fast_idwpt.forward(carrier_np)
    result = fast_idwpt.inverse(leaves)

# Benchmark
start = time.perf_counter()
for _ in range(N):
    leaves = fast_idwpt.forward(carrier_np)
elapsed = time.perf_counter() - start
print(f"FWPT in-place: {elapsed * 1e6 / N:.1f} us/block")

start = time.perf_counter()
for _ in range(N):
    result = fast_idwpt.inverse(leaves)
elapsed = time.perf_counter() - start
print(f"IDWPT in-place: {elapsed * 1e6 / N:.1f} us/block")

start = time.perf_counter()
for _ in range(N):
    leaves = fast_idwpt.forward(carrier_np)
    result = fast_idwpt.inverse(leaves)
elapsed = time.perf_counter() - start
total_inplace = elapsed * 1e6 / N
print(f"Full FWPT+IDWPT in-place: {total_inplace:.1f} us/block")

print()
speedup = us_per_block / total_inplace
print(f"Speedup vs Python effect_mode: {speedup:.1f}x")
print(f"Headroom with in-place: {1333 - total_inplace:.1f} us ({(1333 - total_inplace) / 1333 * 100:.0f}%)")
