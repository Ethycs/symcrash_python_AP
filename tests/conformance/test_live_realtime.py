"""The live timing report must include output DSP and expose missed deadlines."""
from pathlib import Path

import numpy as np
import pytest

from tools.benchmark_live_renderer import make_renderer, measure, timing_summary, verify_audio
from symcrash_python_ap.instrument_ir import load_instrument
from symcrash_python_ap.library.preset_compiler import compiled_preset_from_ir
from symcrash_python_ap.vm import hotpath_jit as hot


def test_mean_pass_does_not_hide_live_deadline_misses():
    report = timing_summary([100, 200, 1500])
    assert report['mean_within_budget'] is True
    assert report['no_observed_deadline_misses'] is False
    assert report['deadline_misses'] == 1
    assert report['p99_us'] > report['p95_us']
    assert report['max_us'] == 1500


@pytest.mark.parametrize('bad', [[], [-1], [np.inf], [np.nan], [[1, 2]]])
def test_invalid_timings_cannot_pass(bad):
    with pytest.raises(ValueError):
        timing_summary(bad)


def test_live_measurement_runs_cockpit_voices_material_and_safety(monkeypatch):
    if not hot.NUMBA_AVAILABLE:
        pytest.skip('Numba unavailable')
    preset_path = Path(__file__).resolve().parents[2] / 'presets/instruments/acid_bass_v0.yaml'
    preset = compiled_preset_from_ir(load_instrument(str(preset_path)))
    import tools.benchmark_live_renderer as benchmark
    real_factory = benchmark.make_renderer
    counts = {'control': 0, 'voices': 0, 'material': 0, 'safety': 0}

    def factory(*args):
        renderer = real_factory(*args)
        for owner, method, label in ((renderer.cockpit, 'tick', 'control'),
                                     (renderer.engine, 'step_block', 'voices'),
                                     (renderer.stage.material, 'step_block', 'material'),
                                     (renderer.stage.safety, 'step_block', 'safety')):
            original = getattr(owner, method)

            def count(*args, _original=original, _label=label, **kwargs):
                counts[_label] += 1
                return _original(*args, **kwargs)

            monkeypatch.setattr(owner, method, count)
        return renderer

    monkeypatch.setattr(benchmark, 'make_renderer', factory)
    report, times, audio = measure(preset, voices=2, blocks=12, warmup=4)
    assert counts == dict.fromkeys(counts, 17)  # First + warmup + measured.
    assert report['minimum_rendered_voices'] == 2 and report['filter_enabled']
    assert times.shape == (12,) and audio.shape == (12, 2, 64)
    assert np.any(audio)


def test_complete_live_audio_matches_reference_during_note_changes():
    if not hot.NUMBA_AVAILABLE:
        pytest.skip('Numba unavailable')
    preset_path = Path(__file__).resolve().parents[2] / 'presets/instruments/acid_bass_v0.yaml'
    preset = compiled_preset_from_ir(load_instrument(str(preset_path)))
    assert len(verify_audio(preset, voices=2, blocks=24)) == 64


@pytest.mark.parametrize('voices', [0, 17])
def test_workload_rejects_unsupported_voice_counts(voices):
    with pytest.raises(ValueError):
        make_renderer(None, voices)
