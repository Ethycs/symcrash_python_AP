"""Address overlap is a separate gate from the sum of resource byte counts."""
import importlib.util
from itertools import combinations
from pathlib import Path

import pytest

from symcrash_python_ap.budgets.hardware_memory_map import (
    MemorySpan, check_memory_map, documented_memory_maps, memory_map_report,
)


def test_adjacent_allocations_and_exact_device_end_are_legal():
    assert check_memory_map([MemorySpan('a', 0, 11), MemorySpan('b', 11, 5)], 16) == ()
    assert check_memory_map([], 16) == ()


def test_small_address_space_exhaustively_matches_byte_set_oracle():
    # Independently enumerate occupied bytes, including nested/identical spans,
    # wholly out-of-device spans, reversed order and one-byte intersections.
    intervals = [(start, size) for start in range(9) for size in range(1, 10)]
    for (start_a, size_a), (start_b, size_b) in combinations(intervals, 2):
        sets = [set(range(start_a, start_a + size_a)), set(range(start_b, start_b + size_b))]
        spans = [MemorySpan('a', start_a, size_a), MemorySpan('b', start_b, size_b)]
        expected = {}
        for name, addresses in zip(('a', 'b'), sets):
            outside = addresses - set(range(8))
            if outside:
                expected[('capacity', frozenset((name,)))] = outside
        if sets[0] & sets[1]:
            expected[('overlap', frozenset(('a', 'b')))] = sets[0] & sets[1]
        for order in (spans, list(reversed(spans))):
            actual = check_memory_map(order, 8)
            assert {(v.kind, frozenset(v.names)): set(range(v.start, v.end)) for v in actual} == expected


def test_nested_spans_report_every_conflict_not_only_adjacent_sorted_pairs():
    spans = [MemorySpan('outer', 0, 100), MemorySpan('early', 10, 5), MemorySpan('late', 80, 5)]
    failures = check_memory_map(spans, 100)
    assert {frozenset(v.names) for v in failures} == {frozenset(('outer', 'early')), frozenset(('outer', 'late'))}


@pytest.mark.parametrize('value', [-1, True, 1.25, '16', None])
def test_invalid_address_size_or_capacity_is_not_coerced(value):
    with pytest.raises(ValueError):
        MemorySpan('bad', value, 1)
    with pytest.raises(ValueError):
        MemorySpan('bad', 0, value)
    with pytest.raises(ValueError):
        check_memory_map([], value)


def test_invalid_records_and_duplicate_names_fail():
    for name in ('', ' ', None):
        with pytest.raises(ValueError):
            MemorySpan(name, 0, 1)
    with pytest.raises(ValueError):
        MemorySpan('empty', 0, 0)
    with pytest.raises(ValueError):
        check_memory_map([], 0)
    with pytest.raises(ValueError, match='Expected'):
        check_memory_map([('untyped', 0, 1)], 16)
    with pytest.raises(ValueError, match='Duplicate'):
        check_memory_map([MemorySpan('same', 0, 1), MemorySpan('same', 2, 1)], 16)


def test_address_arithmetic_does_not_wrap_at_bus_width():
    violation, = check_memory_map([MemorySpan('wrap', 0xfffffff0, 32)], 1 << 32)
    assert (violation.start, violation.end, violation.bytes) == (1 << 32, (1 << 32) + 16, 16)


def test_small_total_still_fails_overlapping_addresses():
    spans = [MemorySpan('a', 4, 4), MemorySpan('b', 6, 4)]
    assert sum(span.bytes for span in spans) < 16
    violation, = check_memory_map(spans, 16)
    assert (violation.kind, violation.start, violation.end) == ('overlap', 6, 8)


def test_normative_maps_retain_and_report_their_arithmetic_conflicts():
    profiles = documented_memory_maps()
    assert len(profiles) == 3
    for profile in profiles[:2]:
        violation, = check_memory_map(profile.spans, profile.capacity)
        assert (violation.kind, violation.names, violation.start, violation.end) == (
            'capacity', ('Reserved',), 0x1f800, 0x20000)
        # Existing useful regions are not shrunk to make the result pass.
        assert sum(s.bytes for s in profile.spans if 'Voice' in s.name or 'Prefetch' in s.name) == 32768
        assert sum(s.bytes for s in profile.spans if 'Coeff' in s.name) == 32768
    failures = check_memory_map(profiles[2].spans, profiles[2].capacity)
    assert {(f.kind, f.names, f.start, f.end) for f in failures} == {
        ('capacity', ('Sample ROM',), 0x8000000, 0x8040000),
        ('overlap', ('Sample ROM', 'Reserved'), 0x7f00000, 0x8000000),
    }


def test_inventory_stays_bound_to_the_documented_tables():
    # Match each complete table row, not just numbers found elsewhere in prose.
    # This makes a future normative map edit require updating its executable gate.
    root = Path(__file__).resolve().parents[2] / 'symcrash_docs/05 - Standards'
    docs = [root / '01 - Architecture/WSE-STD-BRAM-RAM.md',
            root / '03 - Voice/WSE-STD-VOICE-TIERS.md']
    sections = [(docs[0], '### 6.1 Static Partitioning', '### 6.2 Dynamic Pool Allocation'),
                (docs[1], '### 5.3 BRAM Cache Strategy', '## 6. Overhead Optimization Strategies'),
                (docs[1], '### 5.2 Memory Map', '### 5.3 BRAM Cache Strategy')]
    for profile, (path, start, end) in zip(documented_memory_maps(), sections):
        text = path.read_text(encoding='utf-8')
        lines = text.split(start, 1)[1].split(end, 1)[0].splitlines()
        rows = []
        for line in lines:
            parts = [p.strip() for p in line.split('│')]
            if len(parts) >= 5 and parts[2].startswith('0x'):
                amount, unit = parts[3].split()
                rows.append((parts[1], int(parts[2].replace('_', ''), 16),
                             int(amount) * {'B': 1, 'KB': 1024, 'MB': 1048576}[unit]))
        assert [(span.name, span.start, span.bytes) for span in profile.spans] == rows


def _cli():
    path = Path(__file__).resolve().parents[2] / 'tools/check_hardware_resources.py'
    spec = importlib.util.spec_from_file_location('check_memory_cli', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_map_failure_cannot_be_hidden_by_fitting_oscillators(monkeypatch, capsys):
    import json
    cli = _cli()
    monkeypatch.setattr(cli, 'oscillator_allocations', lambda: ())
    assert cli.main(['--json']) == 1
    report = json.loads(capsys.readouterr().out)
    assert report['storage_passed'] and not report['passed']
    assert report['memory_maps'] == json.loads(json.dumps(memory_map_report()))


def test_cli_pass_requires_both_address_and_storage_checks(monkeypatch, capsys):
    import json
    cli = _cli()
    monkeypatch.setattr(cli, 'oscillator_allocations', lambda: ())
    monkeypatch.setattr(cli, 'memory_map_report', lambda: [dict(address_map_passed=True)])
    assert cli.main(['--json']) == 0
    assert json.loads(capsys.readouterr().out)['passed']
