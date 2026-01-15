#!/usr/bin/env python3
"""
Bootstrap Comparison: FM6 IDWPT Bypass vs FWPT Path
====================================================

Compares two FM6 audio paths to verify round-trip fidelity:
- Path A: Direct audio output (bypass FWPT/IDWPT)
- Path B: audio -> FWPT -> IDWPT -> output (no effects, just transform)

If FWPT->IDWPT is a valid adapter, both paths should produce equivalent audio.
"""

import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from symcrash_python_ap.vm.fm6_state import FM6State, Q15_ONE
from symcrash_python_ap.vm.fm6_engine import fm6_generate_block_full, configure_jump_brass
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit


def compare_paths(num_blocks: int = 100):
    """Compare Path A (bypass) vs Path B (FWPT->IDWPT) for FM6 synthesis."""

    print("FM6 Path Comparison: Bypass vs FWPT->IDWPT")
    print("=" * 60)

    # Initialize FM6 state
    state = FM6State()
    configure_jump_brass(state)

    # Override with reduced detune (same as current preset)
    state.operators[0].ratio_q16 = 65536   # 1.0x
    state.operators[1].ratio_q16 = 65726   # +5 cents
    state.operators[2].ratio_q16 = 65346   # -5 cents
    state.operators[3].ratio_q16 = 65916   # +10 cents
    state.operators[4].ratio_q16 = 65536   # 1.0x
    state.operators[5].ratio_q16 = 65536   # 1.0x

    # Set envelope to sustain
    for op in state.operators:
        op.env_value_q15 = Q15_ONE

    # Phase increment for C4 (262 Hz at 48 kHz)
    base_freq = 262.0
    sample_rate = 48000
    phase_inc = int((base_freq / sample_rate) * (2**32))

    print(f"Base frequency: {base_freq} Hz")
    print(f"Phase increment: {phase_inc}")
    print(f"Generating {num_blocks} blocks ({num_blocks * 64 / sample_rate:.2f}s)")
    print()

    # Collect audio from both paths
    path_a_audio = []  # Direct (bypass)
    path_b_audio = []  # FWPT -> IDWPT

    # Per-block metrics
    max_errors = []
    correlations = []

    for block in range(num_blocks):
        # Generate FM6 audio (shared source)
        audio = fm6_generate_block_full(state, phase_inc)

        # PATH A: Direct audio (bypass FWPT/IDWPT)
        path_a = audio.copy()

        # PATH B: FWPT -> IDWPT round-trip
        coeffs = fwpt_jit(audio.tolist(), headroom_bits=1)
        path_b = np.array(idwpt_jit(coeffs, headroom_bits=1), dtype=np.int16)

        path_a_audio.append(path_a)
        path_b_audio.append(path_b)

        # Per-block comparison
        diff = path_a.astype(np.float64) - path_b.astype(np.float64)
        max_error = np.max(np.abs(diff))
        max_errors.append(max_error)

        if np.std(path_a) > 0 and np.std(path_b) > 0:
            corr = np.corrcoef(path_a.astype(np.float64), path_b.astype(np.float64))[0, 1]
        else:
            corr = 1.0  # Both zero = perfect match
        correlations.append(corr)

    # Concatenate for full comparison
    full_a = np.concatenate(path_a_audio)
    full_b = np.concatenate(path_b_audio)

    # Global metrics
    full_diff = full_a.astype(np.float64) - full_b.astype(np.float64)
    global_max_error = np.max(np.abs(full_diff))
    global_corr = np.corrcoef(full_a.astype(np.float64), full_b.astype(np.float64))[0, 1]

    print("Per-Block Metrics")
    print("-" * 60)
    print(f"Max error (LSB):  min={min(max_errors):.0f}, max={max(max_errors):.0f}, avg={np.mean(max_errors):.1f}")
    print(f"Correlation:      min={min(correlations):.6f}, max={max(correlations):.6f}, avg={np.mean(correlations):.6f}")
    print()

    print("Global Metrics")
    print("-" * 60)
    print(f"Total samples:     {len(full_a)}")
    print(f"Global max error:  {global_max_error:.0f} LSB")
    print(f"Global correlation: {global_corr:.6f}")
    print()

    # Check for significant differences
    print("Validation")
    print("-" * 60)

    if global_max_error <= 2:
        print(f"[PASS] Max error {global_max_error:.0f} <= 2 LSB (quantization noise)")
    else:
        print(f"[WARN] Max error {global_max_error:.0f} > 2 LSB")

    if global_corr > 0.9999:
        print(f"[PASS] Correlation {global_corr:.6f} > 0.9999 (nearly identical)")
    elif global_corr > 0.999:
        print(f"[OK]   Correlation {global_corr:.6f} > 0.999 (very close)")
    else:
        print(f"[WARN] Correlation {global_corr:.6f} < 0.999 (noticeable difference)")

    # Spectral comparison
    print()
    print("Spectral Comparison (first 4096 samples)")
    print("-" * 60)

    fft_a = np.abs(np.fft.rfft(full_a[:4096]))
    fft_b = np.abs(np.fft.rfft(full_b[:4096]))
    freqs = np.fft.rfftfreq(4096, 1/sample_rate)

    # Compare top 5 peaks
    top_a = np.argsort(fft_a)[-5:][::-1]
    top_b = np.argsort(fft_b)[-5:][::-1]

    print("Path A top frequencies:")
    for idx in top_a:
        print(f"  {freqs[idx]:7.1f} Hz: {fft_a[idx]:10.0f}")

    print("\nPath B top frequencies:")
    for idx in top_b:
        print(f"  {freqs[idx]:7.1f} Hz: {fft_b[idx]:10.0f}")

    spectral_corr = np.corrcoef(fft_a, fft_b)[0, 1]
    print(f"\nSpectral correlation: {spectral_corr:.6f}")

    # Sample-by-sample comparison at key points
    print()
    print("Sample Snapshots")
    print("-" * 60)

    snapshots = [(0, "Start"), (1000, "Early"), (3000, "Mid")]
    for offset, name in snapshots:
        print(f"\n{name} (samples {offset}-{offset+7}):")
        print(f"  {'Path A':>10} {'Path B':>10} {'Diff':>8}")
        for i in range(8):
            idx = offset + i
            a_val = full_a[idx]
            b_val = full_b[idx]
            diff = a_val - b_val
            print(f"  {a_val:10d} {b_val:10d} {diff:8d}")

    return global_max_error <= 2 and global_corr > 0.999


if __name__ == "__main__":
    success = compare_paths()
    print()
    print("=" * 60)
    if success:
        print("RESULT: FWPT->IDWPT is a valid adapter (paths are equivalent)")
    else:
        print("RESULT: Paths differ significantly - investigate FWPT/IDWPT")
    sys.exit(0 if success else 1)
