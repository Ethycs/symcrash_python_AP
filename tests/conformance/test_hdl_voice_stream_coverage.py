"""Incomplete or overdue integration runs must not become passing evidence."""
import pytest

from tools.check_hdl_voice_stream import validate_summary


SUMMARY = ('PASS voice_stream cases=20 headroom=2 stalls=0 latency=20 '
    'frames=127 samples=8128 held=59 updated=10 silent=58 '
    'K2=185..185 K4=348..348 K8=798..798 windows=31 max_window=1044 '
    'resets=4 state_reads=2560 state_writes=192 conflicts=1 prefetch_conflicts=1')


def validate(summary):
    return validate_summary(summary, 20, 2, 20, False, 192)


def test_exact_window_deadline_is_allowed():
    counts, service = validate(SUMMARY.replace('max_window=1044', 'max_window=40000'))
    assert counts['samples'] == 8128
    assert service['K8'] == (798, 798)


@pytest.mark.parametrize('before,after', [
    ('max_window=1044', 'max_window=40001'),
    ('max_window=1044', 'max_window=0'),
    ('samples=8128', 'samples=8127'),
    ('frames=127', 'frames=125'),
    ('resets=4', 'resets=3'),
    ('state_writes=192', 'state_writes=191'),
    ('state_reads=2560', 'state_reads=2559'),
    ('prefetch_conflicts=1', 'prefetch_conflicts=0'),
    ('stalls=0', 'stalls=1'),
    ('K8=798..798 ', ''),
])
def test_incomplete_or_incompatible_coverage_is_rejected(before, after):
    with pytest.raises(RuntimeError):
        validate(SUMMARY.replace(before, after))
