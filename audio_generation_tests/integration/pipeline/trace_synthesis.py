"""Trace exactly what the synthesizer does."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "symcrash_python_ap"))

import numpy as np
from symcrash_python_ap.wavelet.wavelet_packet_trees import PacketTreeSet, LEAF_COUNT, NODE_SIZE, BLOCK_SIZE
from symcrash_python_ap.wavelet.leaf_generator import LeafGenerator
from symcrash_python_ap.wavelet.carrier_lift import Q15_ONE

# Replicate synthesis with debug output
sample_rate = 48000.0
nyquist = sample_rate / 2
band_width = nyquist / LEAF_COUNT

leaf_gen = LeafGenerator()
f0_hz = 261.63
harmonic_amps = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]
leaf_content = leaf_gen.generate_from_fundamental(f0_hz, harmonic_amps)

print(f"Fundamental: {f0_hz} Hz")
print(f"Leaf bandwidth: {band_width} Hz")
print(f"Non-zero leaf content:")
for i, c in enumerate(leaf_content.coeffs_q15):
    if c != 0:
        print(f"  coeffs_q15[{i}] = {c}")

# Build nodes as in working test
nodes = []
for leaf in range(LEAF_COUNT):
    node_coeffs = []
    for c in range(NODE_SIZE):
        if leaf < len(leaf_content.coeffs_q15) // NODE_SIZE:
            idx = leaf * NODE_SIZE + c
            if idx < len(leaf_content.coeffs_q15):
                node_coeffs.append(leaf_content.coeffs_q15[idx])
            else:
                node_coeffs.append(0)
        else:
            node_coeffs.append(0)
    nodes.append(node_coeffs)

print(f"\nNon-zero nodes:")
for i, node in enumerate(nodes):
    if any(c != 0 for c in node):
        print(f"  node[{i}] = {node}")

# Trace synthesis for non-zero nodes
print(f"\nSynthesis trace (what frequencies get generated):")
for leaf_idx in range(LEAF_COUNT):
    node = nodes[leaf_idx]
    node_energy = sum(abs(c) for c in node)
    if node_energy < 10:
        continue

    freq_low_hz = leaf_idx * band_width
    freq_high_hz = (leaf_idx + 1) * band_width
    freq_span = freq_high_hz - freq_low_hz

    print(f"\n  Node {leaf_idx}: freq range {freq_low_hz:.1f} - {freq_high_hz:.1f} Hz")

    for coeff_idx in range(NODE_SIZE):
        coeff_val = node[coeff_idx]
        if abs(coeff_val) < 10:
            continue

        sub_pos = (coeff_idx + 0.5) / NODE_SIZE
        sub_freq_hz = freq_low_hz + sub_pos * freq_span

        if sub_freq_hz < 20:
            print(f"    coeff[{coeff_idx}] = {coeff_val} -> {sub_freq_hz:.1f} Hz (SKIPPED < 20 Hz)")
        else:
            omega = 2 * np.pi * sub_freq_hz / sample_rate
            print(f"    coeff[{coeff_idx}] = {coeff_val} -> {sub_freq_hz:.1f} Hz, omega = {omega:.6f}")
