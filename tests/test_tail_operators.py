"""
CCS Conformance Test: Tail Operator Evaluation
==============================================

Validates the CCS-specific soft energy limiter.

Tests:
1. Normal energy passes through unchanged
2. Excessive energy is limited
3. Signal shape is preserved during limiting
4. No clipping artifacts introduced
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.vm.decode import DecodedPlan, BackboneEntry
from symcrash_python_ap.vm.lanes import lane_tail, _tail_ccs_soft_limit, _CCS_MAX_RMS_Q15
from symcrash_python_ap.vm.tables import COEFF_LEN


def compute_rms(coeffs: np.ndarray) -> float:
    """Compute RMS of coefficients."""
    return np.sqrt(np.mean(coeffs.astype(np.float64) ** 2))


def create_ccs_config():
    """Create a minimal CCS config for testing."""
    cfg = DecodedPlan(
        tree_a=0, tree_b=1, tree_blend_q15=0,
        eta_prev_q15=0, eta_curr_q15=0,
        gain_prev_q15=32767, gain_curr_q15=32767,
        mask_u32=0xFFFFFFFF, tail_ops=[],
        idwpt_enable=True, use_coeff_osc=True,
        backbone_instrs=[],
    )
    return cfg


def test_normal_energy_passthrough():
    """Test that normal energy coefficients pass through unchanged."""
    print("\n=== Normal Energy Passthrough Test ===\n")

    state = VMState.create()

    # Fill with moderate energy coefficients (below limit)
    state.coeff_state[:COEFF_LEN] = np.random.randint(-15000, 15000, COEFF_LEN, dtype=np.int16)
    original = state.coeff_state[:COEFF_LEN].copy()

    rms_before = compute_rms(original)
    print(f"RMS before: {rms_before:.1f}")
    print(f"Limit threshold: {_CCS_MAX_RMS_Q15}")

    # Apply soft limit
    _tail_ccs_soft_limit(state)

    rms_after = compute_rms(state.coeff_state[:COEFF_LEN])
    print(f"RMS after: {rms_after:.1f}")

    # Check that coefficients are unchanged if below threshold
    if rms_before < _CCS_MAX_RMS_Q15:
        diff = np.abs(state.coeff_state[:COEFF_LEN].astype(np.int32) - original.astype(np.int32))
        max_diff = np.max(diff)
        print(f"Max coefficient change: {max_diff}")
        assert max_diff == 0, f"Coefficients changed when below threshold: {max_diff}"
        print("[PASS] Normal energy passes through unchanged")
    else:
        print("[SKIP] Random coefficients exceeded threshold")

    return True


def test_excessive_energy_limited():
    """Test that excessive energy is limited."""
    print("\n=== Excessive Energy Limiting Test ===\n")

    state = VMState.create()

    # Fill with very high energy coefficients
    state.coeff_state[:COEFF_LEN] = np.full(COEFF_LEN, 30000, dtype=np.int16)

    rms_before = compute_rms(state.coeff_state[:COEFF_LEN])
    print(f"RMS before: {rms_before:.1f}")
    print(f"Limit threshold: {_CCS_MAX_RMS_Q15}")

    # Apply soft limit
    _tail_ccs_soft_limit(state)

    rms_after = compute_rms(state.coeff_state[:COEFF_LEN])
    print(f"RMS after: {rms_after:.1f}")

    # RMS should be reduced to approximately the threshold
    assert rms_after <= _CCS_MAX_RMS_Q15 * 1.1, f"RMS not sufficiently reduced: {rms_after}"

    print("[PASS] Excessive energy is limited")
    return True


def test_signal_shape_preserved():
    """Test that signal shape is preserved during limiting."""
    print("\n=== Signal Shape Preservation Test ===\n")

    state = VMState.create()

    # Create a pattern with varying amplitudes
    pattern = np.sin(np.linspace(0, 4 * np.pi, COEFF_LEN)) * 30000
    state.coeff_state[:COEFF_LEN] = pattern.astype(np.int16)
    original = state.coeff_state[:COEFF_LEN].copy().astype(np.float64)

    # Apply soft limit
    _tail_ccs_soft_limit(state)
    limited = state.coeff_state[:COEFF_LEN].astype(np.float64)

    # Check correlation (shape preservation)
    # If shape is preserved, correlation should be very high
    correlation = np.corrcoef(original, limited)[0, 1]
    print(f"Correlation: {correlation:.6f}")

    assert correlation > 0.999, f"Shape not preserved: correlation = {correlation}"

    # Check that relative amplitudes are preserved
    # (ratio of any two coefficients should be the same)
    if original[0] != 0 and limited[0] != 0:
        ratio_orig = original[100] / original[0] if original[0] != 0 else 0
        ratio_lim = limited[100] / limited[0] if limited[0] != 0 else 0
        ratio_diff = abs(ratio_orig - ratio_lim)
        print(f"Ratio preservation: {ratio_orig:.4f} vs {ratio_lim:.4f} (diff: {ratio_diff:.6f})")

    print("[PASS] Signal shape is preserved")
    return True


def test_no_clipping_artifacts():
    """Test that limiting doesn't introduce clipping artifacts."""
    print("\n=== No Clipping Artifacts Test ===\n")

    state = VMState.create()

    # Create extreme values
    state.coeff_state[:COEFF_LEN] = np.full(COEFF_LEN, 32767, dtype=np.int16)

    # Apply soft limit
    _tail_ccs_soft_limit(state)

    # Check that no hard clipping occurred (all values well within range)
    limited = state.coeff_state[:COEFF_LEN]
    max_val = np.max(np.abs(limited))

    print(f"Max absolute value after limiting: {max_val}")

    # Should be significantly below the hard clip limit
    assert max_val < 32767, f"Values at hard clip: {max_val}"
    assert max_val < 30000, f"Values too close to clip: {max_val}"

    print("[PASS] No clipping artifacts")
    return True


def test_lane_tail_ccs_mode():
    """Test that lane_tail uses CCS soft limit in CCS mode."""
    print("\n=== lane_tail CCS Mode Test ===\n")

    state = VMState.create()
    cfg = create_ccs_config()

    # Fill with high energy
    state.coeff_state[:COEFF_LEN] = np.full(COEFF_LEN, 28000, dtype=np.int16)
    rms_before = compute_rms(state.coeff_state[:COEFF_LEN])

    print(f"RMS before lane_tail: {rms_before:.1f}")

    # Apply lane_tail
    lane_tail(state, cfg)

    rms_after = compute_rms(state.coeff_state[:COEFF_LEN])
    print(f"RMS after lane_tail: {rms_after:.1f}")

    # Should have been limited
    assert rms_after < rms_before, "lane_tail did not reduce energy"
    assert rms_after <= _CCS_MAX_RMS_Q15 * 1.1, f"RMS not limited: {rms_after}"

    print("[PASS] lane_tail uses CCS soft limit in CCS mode")
    return True


def test_repeated_limiting():
    """Test that repeated limiting doesn't cause drift."""
    print("\n=== Repeated Limiting Test ===\n")

    state = VMState.create()

    # Start with excessive energy
    state.coeff_state[:COEFF_LEN] = np.full(COEFF_LEN, 25000, dtype=np.int16)

    rms_values = []
    for i in range(10):
        _tail_ccs_soft_limit(state)
        rms = compute_rms(state.coeff_state[:COEFF_LEN])
        rms_values.append(rms)

    print(f"RMS over 10 iterations: {[f'{r:.1f}' for r in rms_values]}")

    # After first limiting, values should stabilize
    rms_final = rms_values[-1]
    rms_second = rms_values[1]

    # No further reduction after first pass (if already at limit)
    drift = abs(rms_final - rms_second)
    print(f"Drift from iteration 2 to 10: {drift:.1f}")

    assert drift < 100, f"Excessive drift: {drift}"

    print("[PASS] No drift from repeated limiting")
    return True


def main():
    print("=" * 60)
    print("CCS Conformance: Tail Operator Evaluation")
    print("=" * 60)

    results = []

    results.append(("Normal Energy Passthrough", test_normal_energy_passthrough()))
    results.append(("Excessive Energy Limited", test_excessive_energy_limited()))
    results.append(("Signal Shape Preserved", test_signal_shape_preserved()))
    results.append(("No Clipping Artifacts", test_no_clipping_artifacts()))
    results.append(("lane_tail CCS Mode", test_lane_tail_ccs_mode()))
    results.append(("Repeated Limiting", test_repeated_limiting()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        all_passed = all_passed and passed

    print()
    if all_passed:
        print("[PASS] All tail operator tests passed!")
        return 0
    else:
        print("[FAIL] Some tail operator tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
