"""Actual routing and command witnesses for broader live-budget qualification."""
import json

import pytest

from tools import benchmark_live_workloads as benchmark


@pytest.mark.parametrize('program', range(11))
def test_each_authored_material_executes_in_actual_live_output(program):
    result = benchmark.verify_workload(f'material-{program}', voices=2, blocks=24)
    assert result['material_id'] == program
    assert result['minimum_rendered_voices'] == 2


@pytest.mark.parametrize('workload', ['controls', 'note-turnover'])
@pytest.mark.parametrize('voices', [1, 4])
def test_live_control_and_note_workloads_have_exact_audio_and_effective_commands(workload, voices):
    result = benchmark.verify_workload(workload, voices, blocks=24)
    assert result['minimum_rendered_voices'] == voices
    if workload == 'controls':
        assert result['distinct_rendered_pitches'] > voices
    else:
        assert result['note_turnovers'] == 2
        assert result['started_note_frames'] >= voices + 2


def test_failed_workload_cannot_be_measured_or_reported_as_passing(tmp_path, monkeypatch):
    def invalid(*args, **kwargs):
        raise AssertionError('wrong actual Material dispatch')

    def forbidden(*args, **kwargs):
        pytest.fail('Measurement must not run after failed qualification')

    monkeypatch.setattr(benchmark, 'verify_workload', invalid)
    monkeypatch.setattr(benchmark, 'measure_workload', forbidden)
    output = tmp_path / 'evidence'
    assert benchmark.main(['--workloads', 'material-1', '--voices', '1',
                           '--output', str(output)]) == 1
    report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
    assert report['completed'] and not report['passed']
    assert report['results'] == [dict(workload='material-1', voices=1,
        status='qualification_failed', error='AssertionError: wrong actual Material dispatch',
        no_observed_deadline_misses=False)]
