#!/usr/bin/env python3
"""
Benchmark CCS oscillator modes (Mode A, B, Sample, FM6).

This benchmarks the coefficient-native synthesis paths introduced with CCS.
For legacy VM component timing, see bench_vm_components.py.

Tests:
- Mode A (LUT-based): Cheapest, pre-computed lookup tables
- Mode B (Harmonic Bank): Most expensive, H×K×leaves iterations
- Mode Sample: Pre-computed coefficients from sample
- Mode FM6: 6-operator FM synthesis

Reports per-mode timing vs real-time budget (1.33ms @ 48kHz, 64 samples/block).
"""

import sys
sys.path.insert(0, str(__file__).rsplit('tools', 1)[0] + 'src')

import time
import numpy as np
from typing import Dict, List

# Core imports
from symcrash_python_ap.vm.state import (
    VMState, OSC_MODE_A, OSC_MODE_B, OSC_MODE_SAMPLE, OSC_MODE_FM6
)
from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.frames import VoiceFrame

BLOCK_SIZE = 64
SAMPLE_RATE = 48000
Q15_ONE = 32767
NUM_BLOCKS = 1000
NUM_WARMUP = 50

# Real-time budget: 64 samples @ 48kHz = 1.333ms
RT_BUDGET_MS = (BLOCK_SIZE / SAMPLE_RATE) * 1000


def create_vm_state(osc_mode: int) -> VMState:
    """Create a VMState configured for the specified oscillator mode."""
    state = VMState()
    state.osc_mode = osc_mode

    # VMState uses lazy initialization for mode-specific states
    # Just trigger initialization by accessing the property
    if osc_mode == OSC_MODE_B:
        # Access to trigger lazy init
        _ = state.mode_b_state
    elif osc_mode == OSC_MODE_FM6:
        # Configure a typical brass algorithm (Algorithm 22)
        state.fm6_state.set_algorithm(22)
    elif osc_mode == OSC_MODE_SAMPLE:
        # Create a dummy sample buffer (64 blocks worth)
        dummy_audio = (np.sin(np.linspace(0, 20*np.pi, BLOCK_SIZE * 64)) * 32767).astype(np.int16)
        state.sample_state.set_buffer(
            dummy_audio,
            loop_enabled=True,
            precompute_coeffs=True
        )
        state.sample_state.reset()

    return state


def create_voice_frame(note: int, osc_mode: int) -> VoiceFrame:
    """Create a VoiceFrame for the specified note and mode."""
    f0_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))

    frame = VoiceFrame(
        voice_id=1,
        note=note,
        f0_hz=f0_hz,
        gain_q15=Q15_ONE,
        is_active=True,
        coeff_osc_mode=osc_mode,
    )

    # Add FM6 parameters if needed
    if osc_mode == OSC_MODE_FM6:
        frame.fm_algorithm_id = 22
        frame.fm_op_ratios = [65536, 65536, 65536, 65536, 65536, 131072]
        frame.fm_op_levels = [32767, 24576, 24576, 20480, 20480, 16384]
        frame.fm_feedback_depth = 4

    return frame


def benchmark_mode(mode_name: str, osc_mode: int, num_blocks: int) -> Dict:
    """
    Benchmark a specific oscillator mode.

    Returns timing statistics.
    """
    note = 60  # Middle C
    state = create_vm_state(osc_mode)
    frame = create_voice_frame(note, osc_mode)

    # Warmup - let state stabilize
    for _ in range(NUM_WARMUP):
        state, _ = vm_step_block(state, frame)

    # Timed run
    times = []
    for _ in range(num_blocks):
        start = time.perf_counter()
        state, _ = vm_step_block(state, frame)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    times = np.array(times)
    return {
        'mode': mode_name,
        'osc_mode': osc_mode,
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99),
        'rt_ratio': np.mean(times) / RT_BUDGET_MS,
    }


def print_results(results: List[Dict]):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 80)
    print("OSCILLATOR MODE BENCHMARK RESULTS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Block size: {BLOCK_SIZE} samples")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Real-time budget: {RT_BUDGET_MS:.3f} ms/block")
    print(f"  Blocks tested: {NUM_BLOCKS}")
    print(f"  Warmup blocks: {NUM_WARMUP}")

    print("\n" + "-" * 80)
    print(f"{'Mode':<20} {'Mean (ms)':<12} {'Std':<10} {'P95':<10} {'RT %':<10} {'Status':<10}")
    print("-" * 80)

    for r in results:
        rt_pct = r['rt_ratio'] * 100
        status = "OK" if rt_pct < 100 else "OVER"
        status_color = "" if rt_pct < 100 else "!"

        print(f"{r['mode']:<20} {r['mean_ms']:<12.4f} {r['std_ms']:<10.4f} {r['p95_ms']:<10.4f} {rt_pct:<10.1f} {status_color}{status}")

    print("-" * 80)

    # Summary
    print("\n" + "=" * 80)
    print("COST ANALYSIS (relative to cheapest mode)")
    print("=" * 80)

    min_time = min(r['mean_ms'] for r in results)

    # Sort by cost
    sorted_results = sorted(results, key=lambda r: r['mean_ms'])

    print(f"\n{'Rank':<6} {'Mode':<20} {'Time (ms)':<12} {'Relative':<10} {'RT Budget %':<12}")
    print("-" * 60)

    for i, r in enumerate(sorted_results, 1):
        relative = r['mean_ms'] / min_time
        rt_pct = r['rt_ratio'] * 100
        print(f"{i:<6} {r['mode']:<20} {r['mean_ms']:<12.4f} {relative:<10.2f}x {rt_pct:<12.1f}%")

    print("\n" + "=" * 80)

    # Budget analysis
    total_budget_pct = sum(r['rt_ratio'] * 100 for r in results) / len(results)
    print(f"\nAverage RT budget usage: {total_budget_pct:.1f}%")

    fastest = sorted_results[0]
    slowest = sorted_results[-1]
    print(f"Fastest: {fastest['mode']} ({fastest['mean_ms']:.4f} ms)")
    print(f"Slowest: {slowest['mode']} ({slowest['mean_ms']:.4f} ms)")
    print(f"Ratio: {slowest['mean_ms']/fastest['mean_ms']:.1f}x")


def benchmark_stages(num_blocks: int = 500) -> Dict[str, Dict]:
    """
    Benchmark individual pipeline stages to isolate costs.

    Returns timing for:
    - Coefficient generation (per mode)
    - IDWPT reconstruction (shared)
    - CCS operators (shared)
    """
    from symcrash_python_ap.vm.decode import decode_frame
    from symcrash_python_ap.vm.tables import DEFAULT_TABLES
    from symcrash_python_ap.vm.lanes import lane_ccs_render
    from symcrash_python_ap.vm.coeff_lut_streaming import ccs_generate_coeffs_polyphase
    from symcrash_python_ap.vm.harmonic_bank import mode_b_generate_coeffs, get_basis_for_note
    from symcrash_python_ap.vm.sample_source import sample_generate_coeffs
    from symcrash_python_ap.vm.fm6_engine import fm6_generate_coeffs

    results = {}

    # Create states for each mode
    note = 60
    f0_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))

    # --- Mode A: LUT coefficient generation ---
    print("\n  Benchmarking Mode A coeff gen...", end=" ", flush=True)
    state_a = create_vm_state(OSC_MODE_A)
    carrier_512 = np.zeros(512, dtype=np.int16)
    times = []
    for _ in range(num_blocks):
        start = time.perf_counter()
        ccs_generate_coeffs_polyphase(state_a.ccs_state, carrier_512)
        times.append((time.perf_counter() - start) * 1000)
    results['mode_a_coeff'] = np.mean(times)
    print(f"{results['mode_a_coeff']:.4f} ms")

    # --- Mode B: Harmonic bank coefficient generation ---
    print("  Benchmarking Mode B coeff gen...", end=" ", flush=True)
    state_b = create_vm_state(OSC_MODE_B)
    state_b.mode_b_state.reset(note, int((f0_hz / SAMPLE_RATE) * (2**32)))
    basis = get_basis_for_note(note)
    times = []
    for _ in range(num_blocks):
        start = time.perf_counter()
        mode_b_generate_coeffs(state_b.mode_b_state, basis, carrier_512)
        state_b.mode_b_state.advance_block()
        times.append((time.perf_counter() - start) * 1000)
    results['mode_b_coeff'] = np.mean(times)
    print(f"{results['mode_b_coeff']:.4f} ms")

    # --- Mode Sample: Sample coefficient generation ---
    print("  Benchmarking Mode Sample coeff gen...", end=" ", flush=True)
    state_s = create_vm_state(OSC_MODE_SAMPLE)
    times = []
    for _ in range(num_blocks):
        start = time.perf_counter()
        sample_generate_coeffs(state_s.sample_state, carrier_512)
        times.append((time.perf_counter() - start) * 1000)
    results['mode_sample_coeff'] = np.mean(times)
    print(f"{results['mode_sample_coeff']:.4f} ms")

    # --- Mode FM6: FM6 coefficient generation ---
    print("  Benchmarking Mode FM6 coeff gen...", end=" ", flush=True)
    state_fm6 = create_vm_state(OSC_MODE_FM6)
    base_phase_inc = int((f0_hz / SAMPLE_RATE) * (2**32))
    carrier_512_32 = np.zeros(512, dtype=np.int32)
    times = []
    for _ in range(num_blocks):
        start = time.perf_counter()
        fm6_generate_coeffs(state_fm6.fm6_state, carrier_512_32, base_phase_inc)
        times.append((time.perf_counter() - start) * 1000)
    results['mode_fm6_coeff'] = np.mean(times)
    print(f"{results['mode_fm6_coeff']:.4f} ms")

    # --- IDWPT: Shared reconstruction cost ---
    print("  Benchmarking IDWPT reconstruction...", end=" ", flush=True)
    # Use Mode A state with wet enabled
    frame = create_voice_frame(note, OSC_MODE_A)
    state_idwpt = create_vm_state(OSC_MODE_A)
    # Run a few blocks to enable wet path
    for _ in range(10):
        state_idwpt, _ = vm_step_block(state_idwpt, frame)

    cfg = decode_frame(frame, DEFAULT_TABLES, 0, Q15_ONE)
    times = []
    for _ in range(num_blocks):
        # Generate fresh coefficients
        ccs_generate_coeffs_polyphase(state_idwpt.ccs_state, carrier_512)
        state_idwpt.coeff_state[:] = carrier_512
        start = time.perf_counter()
        _ = lane_ccs_render(state_idwpt, cfg)
        times.append((time.perf_counter() - start) * 1000)
    results['idwpt'] = np.mean(times)
    print(f"{results['idwpt']:.4f} ms")

    return results


def print_stage_results(stage_results: Dict[str, float]):
    """Print per-stage timing breakdown."""
    print("\n" + "=" * 80)
    print("PER-STAGE TIMING BREAKDOWN")
    print("=" * 80)

    idwpt_cost = stage_results.get('idwpt', 0)

    print(f"\n{'Stage':<25} {'Time (ms)':<12} {'% of Budget':<12}")
    print("-" * 50)

    # Coefficient generation per mode
    coeff_stages = [
        ('Mode A Coeff Gen', 'mode_a_coeff'),
        ('Mode B Coeff Gen', 'mode_b_coeff'),
        ('Mode Sample Coeff Gen', 'mode_sample_coeff'),
        ('Mode FM6 Coeff Gen', 'mode_fm6_coeff'),
    ]

    for name, key in coeff_stages:
        if key in stage_results:
            t = stage_results[key]
            pct = (t / RT_BUDGET_MS) * 100
            print(f"{name:<25} {t:<12.4f} {pct:<12.1f}%")

    print("-" * 50)
    print(f"{'IDWPT Reconstruction':<25} {idwpt_cost:<12.4f} {(idwpt_cost/RT_BUDGET_MS)*100:<12.1f}%")
    print("-" * 50)

    # Show totals
    print(f"\n{'Mode':<25} {'Coeff + IDWPT':<15} {'% Budget':<12}")
    print("-" * 50)
    for name, key in coeff_stages:
        if key in stage_results:
            total = stage_results[key] + idwpt_cost
            pct = (total / RT_BUDGET_MS) * 100
            mode_name = name.replace(' Coeff Gen', '')
            print(f"{mode_name:<25} {total:<15.4f} {pct:<12.1f}%")


def main():
    print("Symcrash Python AP - Oscillator Mode Benchmark")
    print("-" * 50)

    # Define modes to test
    modes = [
        ("Mode A (LUT)", OSC_MODE_A),
        ("Mode B (Harmonic)", OSC_MODE_B),
        ("Mode Sample", OSC_MODE_SAMPLE),
        ("Mode FM6", OSC_MODE_FM6),
    ]

    results = []

    for mode_name, osc_mode in modes:
        print(f"\nBenchmarking {mode_name}...", end=" ", flush=True)
        try:
            result = benchmark_mode(mode_name, osc_mode, NUM_BLOCKS)
            results.append(result)
            print(f"{result['mean_ms']:.4f} ms ({result['rt_ratio']*100:.1f}% RT)")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                'mode': mode_name,
                'osc_mode': osc_mode,
                'mean_ms': float('nan'),
                'std_ms': float('nan'),
                'min_ms': float('nan'),
                'max_ms': float('nan'),
                'p95_ms': float('nan'),
                'p99_ms': float('nan'),
                'rt_ratio': float('nan'),
            })

    print_results(results)

    # Per-stage breakdown
    print("\n" + "=" * 80)
    print("STAGE-BY-STAGE ANALYSIS")
    print("=" * 80)
    try:
        stage_results = benchmark_stages(500)
        print_stage_results(stage_results)
    except Exception as e:
        print(f"Stage benchmark failed: {e}")

    # Theoretical op count comparison
    print("\n" + "=" * 80)
    print("THEORETICAL OP COUNTS (from analysis)")
    print("=" * 80)
    print("""
    | Mode           | Coeff Gen | CCS Ops | IDWPT  | Total   | Notes                    |
    |----------------|-----------|---------|--------|---------|--------------------------|
    | Mode Sample    | ~832      | ~2,000  | ~4,032 | ~6,866  | Pre-computed coeffs      |
    | Mode A (LUT)   | ~1,600    | ~2,000  | ~4,032 | ~7,632  | Lookup table             |
    | Mode FM6       | ~2,808    | ~2,000  | ~4,032 | ~8,840  | FWPT gateway             |
    | Mode B (H=16)  | ~40,960   | ~2,000  | ~4,032 | ~47,000 | K*H*leaves iterations    |
    """)


if __name__ == "__main__":
    main()
