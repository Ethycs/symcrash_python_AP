#!/usr/bin/env python3
"""
ETA Ramp Jitter Test
====================

Tests for floating-point jitter in the wet path when ETA (dry/wet blend) changes.

This test:
1. Renders multiple blocks with varying ETA values
2. Tracks key numerical values at each step
3. Identifies any non-deterministic behavior (jitter)
4. Helps locate remaining floating-point sources

Usage:
    python -m pytest pipeline_tests/debug/test_eta_jitter.py -v -s
    python pipeline_tests/debug/test_eta_jitter.py  # Direct run for detailed output
"""

import sys
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.fixed import midi_to_freq_mhz, midi_to_phase_inc_fixed


@dataclass
class BlockTrace:
    """Trace of numerical values for a single block."""
    block_idx: int
    eta_q15: int

    # Coefficient generation
    coeffs_hash: str = ""
    coeffs_nonzero: int = 0
    coeffs_sum: int = 0
    coeffs_max: int = 0

    # Carrier output
    carrier_hash: str = ""
    carrier_rms: int = 0

    # Wet output
    wet_hash: str = ""
    wet_rms: int = 0

    # Final mixed output
    output_hash: str = ""
    output_rms: int = 0

    # Phase state
    phase_inc: int = 0
    phase: int = 0


@dataclass
class JitterReport:
    """Report of jitter analysis."""
    note: int
    cents: int
    num_blocks: int
    eta_values: List[int]
    traces: List[BlockTrace] = field(default_factory=list)

    # Jitter metrics
    coeffs_unique_hashes: int = 0
    carrier_unique_hashes: int = 0
    wet_unique_hashes: int = 0

    def analyze(self):
        """Analyze traces for jitter."""
        # Group by eta value
        eta_groups: Dict[int, List[BlockTrace]] = {}
        for trace in self.traces:
            if trace.eta_q15 not in eta_groups:
                eta_groups[trace.eta_q15] = []
            eta_groups[trace.eta_q15].append(trace)

        # For same eta, all hashes should match
        jitter_found = False
        for eta, traces in eta_groups.items():
            if len(traces) < 2:
                continue

            coeffs_hashes = set(t.coeffs_hash for t in traces)
            carrier_hashes = set(t.carrier_hash for t in traces)
            wet_hashes = set(t.wet_hash for t in traces if t.wet_hash)

            if len(coeffs_hashes) > 1:
                print(f"[JITTER] eta={eta}: coeffs vary across {len(coeffs_hashes)} unique values")
                jitter_found = True
            if len(carrier_hashes) > 1:
                print(f"[JITTER] eta={eta}: carrier varies across {len(carrier_hashes)} unique values")
                jitter_found = True
            if len(wet_hashes) > 1:
                print(f"[JITTER] eta={eta}: wet varies across {len(wet_hashes)} unique values")
                jitter_found = True

        if not jitter_found:
            print("[OK] No jitter detected - same eta produces identical outputs")

        return not jitter_found


def hash_samples(samples: List[int]) -> str:
    """Hash a list of samples."""
    # Convert Q15 samples to bytes (2 bytes per sample, little-endian)
    data = b''.join(int(s).to_bytes(2, byteorder='little', signed=True) for s in samples)
    return hashlib.sha256(data).hexdigest()[:16]


def compute_rms(samples: List[int]) -> int:
    """Compute RMS of samples (returns Q15)."""
    if not samples:
        return 0
    sum_sq = sum(s * s for s in samples)
    mean_sq = sum_sq // len(samples)
    # Integer sqrt approximation
    rms = int(mean_sq ** 0.5)
    return min(32767, rms)


def run_eta_ramp_test(
    note: int = 69,
    cents: int = 0,
    num_blocks: int = 100,
    eta_pattern: str = "ramp"
) -> JitterReport:
    """
    Run ETA ramp test and collect traces.

    Args:
        note: MIDI note
        cents: Fine tuning
        num_blocks: Number of blocks to render
        eta_pattern: "ramp" (0->max), "sweep" (up/down), "random", "fixed_high"
    """
    # Create renderer
    stage = UnifiedOutputStage(block_size=64)
    voice_id = 0

    # Initialize voice
    stage.voice_on(voice_id, note)

    # Generate eta values
    if eta_pattern == "ramp":
        eta_values = [min(32767, (i * 32767) // num_blocks) for i in range(num_blocks)]
    elif eta_pattern == "sweep":
        half = num_blocks // 2
        eta_values = (
            [min(32767, (i * 32767) // half) for i in range(half)] +
            [max(0, 32767 - ((i - half) * 32767) // half) for i in range(half, num_blocks)]
        )
    elif eta_pattern == "fixed_high":
        eta_values = [24576] * num_blocks  # ~75% wet
    elif eta_pattern == "fixed_low":
        eta_values = [8192] * num_blocks  # ~25% wet
    else:
        eta_values = [16384] * num_blocks  # 50%

    report = JitterReport(
        note=note,
        cents=cents,
        num_blocks=num_blocks,
        eta_values=eta_values
    )

    # Render blocks and collect traces
    for block_idx in range(num_blocks):
        eta = eta_values[block_idx]

        frame = VoiceFrame(
            voice_id=voice_id,
            note=note,
            cents=cents,
            gain_q15=24576,
            pan_q15=0,
            beta_target_q15=eta,
            is_active=True,
            partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],  # Pure sine
        )

        trace = BlockTrace(block_idx=block_idx, eta_q15=eta)

        # Capture coefficient generation
        if eta > 0:
            from symcrash_python_ap.wavelet.harmonic_injector import (
                create_harmonic_coeffs_fixed, WaveType
            )
            coeffs = create_harmonic_coeffs_fixed(
                note=note,
                cents=cents,
                amplitude=16384,
                wave_type=WaveType.SINE,
                partials_n=8,
            )
            trace.coeffs_hash = hash_samples(coeffs)
            trace.coeffs_nonzero = sum(1 for c in coeffs if c != 0)
            trace.coeffs_sum = sum(coeffs)
            trace.coeffs_max = max(abs(c) for c in coeffs)

        # Render block
        left, right = stage.render_block(frame)

        # Capture output
        trace.output_hash = hash_samples(list(left))
        trace.output_rms = compute_rms(list(left))

        # Capture VM state if available
        if voice_id in stage.vm_states:
            vm_state = stage.vm_states[voice_id]
            trace.phase_inc = vm_state.phase_inc
            trace.phase = vm_state.phase

        report.traces.append(trace)

    return report


def print_trace_summary(report: JitterReport):
    """Print detailed trace summary."""
    print(f"\n{'='*70}")
    print(f"ETA Jitter Test: note={report.note}, cents={report.cents}")
    print(f"{'='*70}")

    # Print sample of traces
    print("\nSample traces (every 10th block):")
    print(f"{'Block':>6} {'ETA':>6} {'Coeffs Hash':>18} {'Output Hash':>18} {'RMS':>6}")
    print("-" * 60)

    for i, trace in enumerate(report.traces):
        if i % 10 == 0 or i == len(report.traces) - 1:
            print(f"{trace.block_idx:>6} {trace.eta_q15:>6} "
                  f"{trace.coeffs_hash:>18} {trace.output_hash:>18} {trace.output_rms:>6}")

    # Analyze for jitter
    print("\n" + "-" * 60)
    print("Jitter Analysis:")
    report.analyze()

    # Check for coefficient stability
    print("\nCoefficient stability check:")
    coeffs_by_eta: Dict[int, List[str]] = {}
    for trace in report.traces:
        if trace.coeffs_hash:
            if trace.eta_q15 not in coeffs_by_eta:
                coeffs_by_eta[trace.eta_q15] = []
            coeffs_by_eta[trace.eta_q15].append(trace.coeffs_hash)

    all_stable = True
    for eta, hashes in coeffs_by_eta.items():
        unique = set(hashes)
        if len(unique) > 1:
            print(f"  [UNSTABLE] eta={eta}: {len(unique)} different coefficient hashes")
            all_stable = False

    if all_stable:
        print("  [OK] All coefficient hashes stable for same eta")

    # Check phase increment stability
    print("\nPhase increment check:")
    phase_incs = set(t.phase_inc for t in report.traces if t.phase_inc > 0)
    if len(phase_incs) == 1:
        print(f"  [OK] Constant phase_inc: {list(phase_incs)[0]}")
    else:
        print(f"  [UNSTABLE] Phase increment varies: {phase_incs}")


def test_eta_ramp_no_jitter():
    """Test that ETA ramp produces no jitter."""
    report = run_eta_ramp_test(note=69, cents=0, num_blocks=50, eta_pattern="ramp")
    assert report.analyze(), "Jitter detected in ETA ramp"


def test_fixed_eta_deterministic():
    """Test that fixed high ETA is deterministic."""
    report = run_eta_ramp_test(note=69, cents=0, num_blocks=20, eta_pattern="fixed_high")

    # All traces with same eta should have same coeffs hash
    coeffs_hashes = set(t.coeffs_hash for t in report.traces if t.coeffs_hash)
    assert len(coeffs_hashes) == 1, f"Coefficient generation not deterministic: {len(coeffs_hashes)} variants"


def test_detune_cents_deterministic():
    """Test that detune cents produce deterministic output."""
    report1 = run_eta_ramp_test(note=69, cents=50, num_blocks=10, eta_pattern="fixed_high")
    report2 = run_eta_ramp_test(note=69, cents=50, num_blocks=10, eta_pattern="fixed_high")

    # Should produce identical traces
    for t1, t2 in zip(report1.traces, report2.traces):
        assert t1.coeffs_hash == t2.coeffs_hash, f"Coeffs differ at block {t1.block_idx}"


def trace_wet_path_floats(note: int, cents: int, eta: int):
    """
    Trace through the wet path looking for floating-point operations.

    This manually steps through each stage and checks for float usage.
    """
    print("\n1. Coefficient generation (create_harmonic_coeffs_fixed):")
    from symcrash_python_ap.wavelet.harmonic_injector import (
        create_harmonic_coeffs_fixed, WaveType, LEAF_EDGES_HZ
    )
    from symcrash_python_ap.fixed import midi_to_freq_mhz

    # Check frequency lookup
    f0_mhz = midi_to_freq_mhz(note, cents)
    print(f"   f0_mhz = {f0_mhz} (integer: {type(f0_mhz).__name__})")

    # Check leaf edges - these are precomputed floats!
    print(f"   LEAF_EDGES_HZ type: {type(LEAF_EDGES_HZ[0]).__name__}")
    if isinstance(LEAF_EDGES_HZ[0], float):
        print("   WARNING: LEAF_EDGES_HZ contains floats - potential jitter source!")

    # Generate coefficients and trace
    coeffs = create_harmonic_coeffs_fixed(note=note, cents=cents, amplitude=16384, wave_type=WaveType.SINE)
    print(f"   coeffs generated: {sum(1 for c in coeffs if c != 0)} non-zero")

    print("\n2. Tuning gains:")
    try:
        from symcrash_python_ap.tuning import GlobalTuning
        tuning = GlobalTuning()
        gains = tuning.get_current_gains()
        print(f"   tuning gains type: {type(gains[0]).__name__}")
    except Exception as e:
        print(f"   Could not check tuning: {e}")

    print("\n3. IDWPT (fast_lifting):")
    from symcrash_python_ap.wavelet.fast_lifting import effect_mode_jit
    import numpy as np

    # Check if numpy arrays maintain int types
    carrier = np.zeros(64, dtype=np.int16)
    coeff_state = np.array(coeffs, dtype=np.int16)

    print(f"   carrier dtype: {carrier.dtype}")
    print(f"   coeff_state dtype: {coeff_state.dtype}")

    # Call effect_mode_jit
    try:
        wet = effect_mode_jit(carrier, coeff_state)
        print(f"   wet output dtype: {wet.dtype}")
        if wet.dtype != np.int16:
            print("   WARNING: IDWPT output is not int16!")
    except Exception as e:
        print(f"   ERROR calling effect_mode_jit: {e}")

    print("\n4. Crossfade (lane_mix_gain):")
    print("   (skipping import check)")

    print("\n5. Checking unified_renderer.render_block for float operations...")
    # Look at the actual source
    import inspect
    from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
    source = inspect.getsource(UnifiedOutputStage.render_block)

    float_indicators = [
        ('/', 'Division (may produce float)'),
        ('0.', 'Float literal'),
        ('float(', 'Float conversion'),
        ('.0)', 'Float literal ending'),
        ('math.', 'Math module (often float)'),
        ('np.float', 'Numpy float'),
    ]

    print("\n   Scanning render_block source for float indicators:")
    lines = source.split('\n')
    for i, line in enumerate(lines):
        for indicator, desc in float_indicators:
            if indicator in line and '#' not in line.split(indicator)[0]:
                # Skip comments
                print(f"   Line ~{i}: {desc}")
                print(f"      {line.strip()[:80]}")


def diagnose_combing():
    """
    Diagnostic function to find combing source.

    Run this directly to get detailed output.
    """
    print("\n" + "="*70)
    print("COMBING DIAGNOSTIC")
    print("="*70)

    note = 69
    cents = 0
    eta = 24576  # 75% wet

    # Create two independent renderers
    stage1 = UnifiedOutputStage(block_size=64)
    stage2 = UnifiedOutputStage(block_size=64)

    stage1.voice_on(0, note)
    stage2.voice_on(0, note)

    frame = VoiceFrame(
        voice_id=0,
        note=note,
        cents=cents,
        gain_q15=24576,
        pan_q15=0,
        beta_target_q15=eta,
        is_active=True,
        partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    )

    print(f"\nRendering 5 blocks with identical params (note={note}, eta={eta})")
    print("-" * 60)

    results1 = []
    results2 = []

    for i in range(5):
        left1, right1 = stage1.render_block(frame)
        left2, right2 = stage2.render_block(frame)

        h1 = hash_samples(list(left1))
        h2 = hash_samples(list(left2))

        results1.append(h1)
        results2.append(h2)

        match = "MATCH" if h1 == h2 else "DIFF"
        print(f"Block {i}: stage1={h1}  stage2={h2}  {match}")

    print("\nIntra-stage consistency (same stage, consecutive blocks):")
    print(f"  Stage 1 unique hashes: {len(set(results1))}")
    print(f"  Stage 2 unique hashes: {len(set(results2))}")

    print("\nCross-stage consistency (same block, different stage):")
    matches = sum(1 for h1, h2 in zip(results1, results2) if h1 == h2)
    print(f"  Matching: {matches}/{len(results1)}")

    # Now check internal state
    print("\n" + "-" * 60)
    print("Internal state comparison:")

    if 0 in stage1.vm_states and 0 in stage2.vm_states:
        s1 = stage1.vm_states[0]
        s2 = stage2.vm_states[0]
        print(f"  phase_inc: {s1.phase_inc} vs {s2.phase_inc} {'MATCH' if s1.phase_inc == s2.phase_inc else 'DIFF'}")
        print(f"  phase: {s1.phase} vs {s2.phase} {'MATCH' if s1.phase == s2.phase else 'DIFF'}")
        print(f"  last_note: {s1.last_note} vs {s2.last_note}")
        print(f"  last_cents: {s1.last_cents} vs {s2.last_cents}")

    # Check coefficient generation directly
    print("\n" + "-" * 60)
    print("Direct coefficient generation test:")

    from symcrash_python_ap.wavelet.harmonic_injector import create_harmonic_coeffs_fixed, WaveType

    for i in range(3):
        coeffs = create_harmonic_coeffs_fixed(
            note=note,
            cents=cents,
            amplitude=16384,
            wave_type=WaveType.SINE,
            partials_n=8,
        )
        h = hash_samples(coeffs)
        print(f"  Run {i}: hash={h}, nonzero={sum(1 for c in coeffs if c != 0)}, sum={sum(coeffs)}")

    # Check for floating point in the wet path
    print("\n" + "-" * 60)
    print("Tracing wet path for floating-point operations...")
    trace_wet_path_floats(note, cents, eta)


def analyze_carrier_wet_correlation():
    """
    Analyze the correlation between carrier and wet signals.

    Combing can occur when:
    1. Carrier and wet are phase-correlated but slightly different
    2. Equal-power crossfade adds them, creating constructive/destructive interference
    """
    import numpy as np
    print("\n" + "="*70)
    print("CARRIER vs WET CORRELATION ANALYSIS")
    print("="*70)

    note = 69
    cents = 0

    stage = UnifiedOutputStage(block_size=64)
    stage.voice_on(0, note)

    # Render a few blocks with different eta values
    eta_values = [0, 8192, 16384, 24576, 32767]

    for eta in eta_values:
        frame = VoiceFrame(
            voice_id=0,
            note=note,
            cents=cents,
            gain_q15=32767,  # Full gain
            pan_q15=0,
            beta_target_q15=eta,
            is_active=True,
            partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
        )

        # Get carrier directly
        carrier = stage.carrier.render_block(0, 64)
        carrier_np = np.array(carrier, dtype=np.float32)

        # Get wet if eta > 0
        if eta > 0:
            from symcrash_python_ap.wavelet.wavelet_packet_trees import PacketTreeSet
            if not hasattr(stage, '_packet_tree_set'):
                stage._packet_tree_set = PacketTreeSet()
            wet = stage._packet_tree_set.effect_mode(
                tree_id=0,
                carrier_audio=carrier,
                mask_q15=None
            )
            wet_np = np.array(wet, dtype=np.float32)

            # Compute correlation
            carrier_norm = carrier_np / (np.linalg.norm(carrier_np) + 1e-10)
            wet_norm = wet_np / (np.linalg.norm(wet_np) + 1e-10)
            correlation = np.dot(carrier_norm, wet_norm)

            # Compute error
            error = wet_np - carrier_np
            error_rms = np.sqrt(np.mean(error**2))
            carrier_rms = np.sqrt(np.mean(carrier_np**2))

            print(f"\neta={eta:5d} ({100*eta/32767:.0f}% wet):")
            print(f"  Carrier RMS: {carrier_rms:.1f}")
            print(f"  Wet RMS:     {np.sqrt(np.mean(wet_np**2)):.1f}")
            print(f"  Error RMS:   {error_rms:.1f} ({100*error_rms/carrier_rms:.1f}% of carrier)")
            print(f"  Correlation: {correlation:.4f}")

            # Check for sign flips (indicates phase inversion)
            sign_matches = np.sum(np.sign(carrier_np) == np.sign(wet_np))
            print(f"  Sign matches: {sign_matches}/64 ({100*sign_matches/64:.0f}%)")
        else:
            print(f"\neta={eta:5d} (0% wet): Pure carrier")


def analyze_block_continuity():
    """
    Analyze block-to-block continuity looking for discontinuities.

    Combing can also occur from:
    1. Block boundary discontinuities
    2. Phase resets
    3. Gain jumps
    """
    import numpy as np
    print("\n" + "="*70)
    print("BLOCK CONTINUITY ANALYSIS")
    print("="*70)

    note = 69
    cents = 0
    eta = 24576  # 75% wet

    stage = UnifiedOutputStage(block_size=64)
    stage.voice_on(0, note)

    frame = VoiceFrame(
        voice_id=0,
        note=note,
        cents=cents,
        gain_q15=24576,
        pan_q15=0,
        beta_target_q15=eta,
        is_active=True,
        partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    )

    # Render 10 consecutive blocks
    all_samples = []
    boundary_diffs = []

    prev_last = None
    for i in range(10):
        left, right = stage.render_block(frame)
        samples = list(left)
        all_samples.extend(samples)

        if prev_last is not None:
            # Check discontinuity at block boundary
            diff = abs(samples[0] - prev_last)
            boundary_diffs.append(diff)

        prev_last = samples[-1]

    all_samples_np = np.array(all_samples, dtype=np.float32)

    print(f"\nRendered 10 blocks ({len(all_samples)} samples)")
    print(f"Overall RMS: {np.sqrt(np.mean(all_samples_np**2)):.1f}")

    print(f"\nBlock boundary discontinuities (sample[0] - prev_sample[-1]):")
    for i, diff in enumerate(boundary_diffs):
        pct = 100 * diff / 32767
        marker = " << LARGE" if diff > 1000 else ""
        print(f"  Block {i+1}: diff={diff:5d} ({pct:.1f}%){marker}")

    # Compute spectrum to look for artifacts
    from scipy.fft import rfft, rfftfreq
    freqs = rfftfreq(len(all_samples), 1/48000)
    spectrum = np.abs(rfft(all_samples_np))

    # Find peaks
    peak_idx = np.argsort(spectrum)[-5:][::-1]
    print(f"\nTop 5 frequency components:")
    for idx in peak_idx:
        print(f"  {freqs[idx]:.1f} Hz: {spectrum[idx]:.0f}")

    # Check for unexpected harmonics (combing artifacts)
    fundamental_idx = np.argmax(spectrum[1:]) + 1  # Skip DC
    fundamental_freq = freqs[fundamental_idx]
    print(f"\nFundamental frequency: {fundamental_freq:.1f} Hz (expected ~440 Hz)")

    # Look for comb filter pattern (regular spectral notches)
    # This would indicate block-rate modulation
    expected_a4 = 440.0
    if abs(fundamental_freq - expected_a4) > 10:
        print(f"  WARNING: Fundamental {fundamental_freq:.1f} Hz differs from expected {expected_a4:.1f} Hz")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ETA Jitter Test")
    parser.add_argument("--diagnose", action="store_true", help="Run diagnostic mode")
    parser.add_argument("--correlation", action="store_true", help="Run correlation analysis")
    parser.add_argument("--continuity", action="store_true", help="Run block continuity analysis")
    parser.add_argument("--pattern", default="ramp", choices=["ramp", "sweep", "fixed_high", "fixed_low"])
    parser.add_argument("--blocks", type=int, default=50, help="Number of blocks")
    parser.add_argument("--note", type=int, default=69, help="MIDI note")
    args = parser.parse_args()

    if args.diagnose:
        diagnose_combing()
    elif args.correlation:
        analyze_carrier_wet_correlation()
    elif args.continuity:
        analyze_block_continuity()
    else:
        report = run_eta_ramp_test(
            note=args.note,
            cents=0,
            num_blocks=args.blocks,
            eta_pattern=args.pattern
        )
        print_trace_summary(report)
