"""Exercise pitch interpolation and cache turnover against the existing RT gate."""
import numpy as np
import pytest

from symcrash_python_ap.vm import harmonic_bank, hotpath_jit
from symcrash_python_ap.vm.basis_resource import prepare_basis_resources
from tools.benchmark_vm_step_block import (
    BUDGET_US, bench_scenario, fmt_row, verify_bit_exact,
    _scenario_mode_b_detuned, _scenario_mode_b_pitch_sweep,
)


@pytest.mark.parametrize('factory,expect_misses', [
    (_scenario_mode_b_detuned, False), (_scenario_mode_b_pitch_sweep, True),
], ids=['sustained-detune', 'continuous-pitch-cache-turnover'])
def test_mode_b_pitch_workload_realtime_budget(factory, expect_misses):
    if not hotpath_jit.NUMBA_AVAILABLE:
        pytest.skip('realtime acceleration unavailable')
    prepare_basis_resources()
    original = hotpath_jit.jit_enabled()
    try:
        ok, reason = verify_bit_exact(factory(), 48, num_blocks=12)
        assert ok, reason
        harmonic_bank.get_basis_for_pitch.cache_clear()
        before = harmonic_bank.get_basis_for_pitch.cache_info()
        times = bench_scenario(factory(), 48, True, num_blocks=750, warmup=100)
        after = harmonic_bank.get_basis_for_pitch.cache_info()
        misses = after.misses - before.misses
        assert misses == (850 if expect_misses else 1)
        details = fmt_row('jit', times)
        print('\n' + details)
        # Same mean/p95 gates as the existing StreamingVoice smoke test.
        # Miss counts, p99 and maximum remain visible; this is not a hard
        # worst-case device deadline or an FPGA cycle qualification.
        assert times.mean() < BUDGET_US, details
        assert np.percentile(times, 95) < 2 * BUDGET_US, details
    finally:
        hotpath_jit.set_jit_enabled(original)


def test_timing_report_exposes_misses_even_when_mean_passes():
    report = fmt_row('example', np.array([100., 200., 1500.]))
    assert '1/3 observed deadline misses' in report
    assert 'p99=' in report and 'max=' in report
