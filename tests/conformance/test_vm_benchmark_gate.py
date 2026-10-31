"""A mean-time pass cannot be reported as E3 sustained qualification."""
import numpy as np
import pytest

from tools import benchmark_vm_step_block as benchmark


@pytest.mark.parametrize('times,exact,skip,expected', [
    ([100., 200., 1500.], True, False, 1),
    ([100., 200., 300.], False, False, 1),
    ([100., 200., 300.], True, True, 2),
    ([100., np.nan, 300.], True, False, 1),
    ([100., 200., 300.], True, False, 0),
])
def test_cli_requires_zero_misses_and_reference_replay(monkeypatch, capsys, times, exact, skip, expected):
    monkeypatch.setattr(benchmark, 'SCENARIOS', [('fixture', lambda: None, 60)])
    monkeypatch.setattr(benchmark, 'bench_scenario', lambda *args: np.array(times))
    verified = []

    def verify(frame, note, count):
        verified.append(count)
        return exact, 'bit-exact' if exact else 'fixture mismatch'

    monkeypatch.setattr(benchmark, 'verify_bit_exact', verify)
    args = ['--skip-reference', '--blocks', '3', '--verify-blocks', '320']
    if skip:
        args.append('--skip-verify')
    assert benchmark.main(args) == expected
    output = capsys.readouterr().out
    assert ('RESULT: PASS' in output) == (expected == 0)
    assert verified == ([] if skip else [320])
