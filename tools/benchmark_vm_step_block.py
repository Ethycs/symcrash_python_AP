#!/usr/bin/env python3
"""
VM step_block Realtime Benchmark (Packet E3)
============================================

Measures sustained single-voice block render time through vm_step_block for
both execution paths:

  - reference: pure-Python conformance implementation
  - jit:       numba hot-path kernels (vm/hotpath_jit.py, bit-exact)

The realtime budget is BLOCK_SIZE / SAMPLE_RATE = 64 / 48000 = 1333 us/block.

Also cross-checks bit-exactness: every scenario is rendered on both paths
from identical initial state and the outputs must match sample-for-sample.

Usage:
    pixi run python tools/benchmark_vm_step_block.py
    pixi run python tools/benchmark_vm_step_block.py --blocks 2000
    pixi run python tools/benchmark_vm_step_block.py --skip-reference
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from symcrash_python_ap.vm import vm_step_block, VMState  # noqa: E402
from symcrash_python_ap.vm import hotpath_jit as hot  # noqa: E402
from symcrash_python_ap.frames import VoiceFrame  # noqa: E402

BUDGET_US = 64 / 48000 * 1e6  # 1333.3 us per 64-sample block @ 48 kHz


# =============================================================================
# SCENARIOS (sustained single-voice workloads)
# =============================================================================

def _scenario_mode_a():
    return VoiceFrame(
        voice_id=1, note=60, cents=0, f0_hz=261.63,
        gain_q15=24576, beta_target_q15=16384,
        partial_amps_q15=[32767, 16384, 10923, 8192, 6554, 5461, 4681, 4096],
        is_active=True,
    )


def _scenario_mode_b_partials_ops():
    return VoiceFrame(
        voice_id=1, note=69, cents=0, f0_hz=440.0,
        coeff_osc_mode=1,
        gain_q15=24576, beta_target_q15=16384,
        partial_amps_q15=[32767, 16384, 10923, 8192, 6554, 5461,
                          4681, 4096, 3641, 3277, 2979, 2731],
        ops_enable=['tilt', 'intramix'],
        families={'tilt': 0.5, 'intramix': 0.8},
        is_active=True,
    )


def _scenario_mode_b_saw():
    return VoiceFrame(
        voice_id=1, note=48, cents=0, f0_hz=130.81,
        coeff_osc_mode=1, waveform_type=3, H_request=32,
        gain_q15=24576, beta_target_q15=16384,
        is_active=True,
    )


def _scenario_mode_b_pwm():
    return VoiceFrame(
        voice_id=1, note=52, cents=0, f0_hz=164.81,
        coeff_osc_mode=1, waveform_type=2,
        pwm_enabled=True, pwm_width_q15=12000, H_request=32,
        gain_q15=24576, beta_target_q15=16384,
        is_active=True,
    )


def _scenario_fm6():
    return VoiceFrame(
        voice_id=1, note=57, cents=0, f0_hz=220.0,
        coeff_osc_mode=3,
        fm_algorithm_id=5, fm_feedback_depth=5,
        fm_op_ratios=[65536, 131072, 65536, 98304, 65536, 131072],
        fm_op_levels=[32767, 16384, 24576, 8192, 12288, 16384],
        gain_q15=24576, beta_target_q15=16384,
        is_active=True,
    )


SCENARIOS = [
    ("Mode A (LUT sine + partials)", _scenario_mode_a, 60),
    ("Mode B (12 partials + 2 ops)", _scenario_mode_b_partials_ops, 69),
    ("Mode B (saw, matrix LUT)", _scenario_mode_b_saw, 48),
    ("Mode B (PWM, harmonic bank)", _scenario_mode_b_pwm, 52),
    ("Mode FM6 (alg 5 + feedback)", _scenario_fm6, 57),
]


# =============================================================================
# MEASUREMENT
# =============================================================================

def bench_scenario(frame, note, use_jit, num_blocks, warmup):
    """Render `num_blocks` sustained blocks, return per-block times in us."""
    hot.set_jit_enabled(use_jit)
    state = VMState.create(1)
    state.reset_for_note_on(note=note, cents=0)

    for _ in range(warmup):
        state, _ = vm_step_block(state, frame)

    times = np.empty(num_blocks, dtype=np.float64)
    for i in range(num_blocks):
        t0 = time.perf_counter()
        state, _ = vm_step_block(state, frame)
        times[i] = time.perf_counter() - t0
    return times * 1e6


def verify_bit_exact(frame, note, num_blocks=48):
    """Render both paths from identical state; require identical output."""
    outputs = {}
    for use_jit in (False, True):
        hot.set_jit_enabled(use_jit)
        state = VMState.create(1)
        state.reset_for_note_on(note=note, cents=0)
        blocks = []
        for _ in range(num_blocks):
            state, out = vm_step_block(state, frame)
            blocks.append(out.copy())
        outputs[use_jit] = (blocks, state.coeff_state.copy())

    ref_blocks, ref_cs = outputs[False]
    jit_blocks, jit_cs = outputs[True]
    for i, (a, b) in enumerate(zip(ref_blocks, jit_blocks)):
        if not np.array_equal(a, b):
            return False, f"block {i} audio diverged"
    if not np.array_equal(ref_cs, jit_cs):
        return False, "final coeff_state diverged"
    return True, "bit-exact"


def fmt_row(label, times):
    mean = times.mean()
    p95 = float(np.percentile(times, 95))
    pct = mean / BUDGET_US * 100
    status = "OK " if mean < BUDGET_US else "OVER"
    return (f"  {label:11s} mean={mean:8.1f} us  p95={p95:8.1f} us  "
            f"({pct:6.1f}% of budget) [{status}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=1000,
                        help="measured blocks per scenario (default 1000)")
    parser.add_argument("--warmup", type=int, default=100,
                        help="warm-up blocks before measuring (default 100)")
    parser.add_argument("--skip-reference", action="store_true",
                        help="skip the (slow) pure-Python reference timing")
    parser.add_argument("--skip-verify", action="store_true",
                        help="skip the bit-exactness cross-check")
    args = parser.parse_args()

    print("=" * 76)
    print("vm_step_block sustained single-voice benchmark")
    print(f"Budget: {BUDGET_US:.1f} us/block (64 samples @ 48 kHz)")
    print(f"Blocks: {args.blocks} measured after {args.warmup} warm-up")
    print(f"Numba available: {hot.NUMBA_AVAILABLE}")
    print("=" * 76)

    original_flag = hot.jit_enabled()
    any_over = False
    try:
        for name, factory, note in SCENARIOS:
            frame = factory()
            print(f"\n{name}")

            if not args.skip_verify:
                ok, msg = verify_bit_exact(factory(), note)
                print(f"  bit-exact vs reference: {msg}")
                if not ok:
                    any_over = True

            jit_times = bench_scenario(frame, note, True, args.blocks, args.warmup)
            print(fmt_row("jit:", jit_times))
            if jit_times.mean() >= BUDGET_US:
                any_over = True

            if not args.skip_reference:
                # Reference is ~20-40x slower; use fewer blocks to keep
                # total runtime sane while still being statistically useful.
                ref_blocks = max(100, args.blocks // 5)
                ref_times = bench_scenario(
                    frame, note, False, ref_blocks, min(args.warmup, 20)
                )
                print(fmt_row("reference:", ref_times))
                speedup = ref_times.mean() / jit_times.mean()
                print(f"  speedup: {speedup:.1f}x")
    finally:
        hot.set_jit_enabled(original_flag)

    print("\n" + "=" * 76)
    if any_over:
        print("RESULT: FAIL (over budget or bit-exactness mismatch)")
        return 1
    print("RESULT: PASS (all scenarios bit-exact and under budget on jit path)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
