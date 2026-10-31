"""Public Material integration checks for the production SVF acceleration."""
import numpy as np
import pytest

from symcrash_python_ap.material.material_engine import MaterialEngine
ReferenceMaterial = MaterialEngine
from symcrash_python_ap.frames import ParamFrame
from symcrash_python_ap.vm import hotpath_jit as hot
from symcrash_python_ap.material import svf_jit

@pytest.fixture(autouse=True)
def execution_path():
    old = hot.jit_enabled()
    if not hot.NUMBA_AVAILABLE:
        pytest.skip('Numba unavailable')
    hot.set_jit_enabled(True)
    try:
        yield
    finally:
        hot.set_jit_enabled(old)


def test_public_material_alternates_paths_and_retains_state_and_slew():
    reference, actual = ReferenceMaterial(), MaterialEngine()
    rng = np.random.default_rng(6921)
    for engine in (reference, actual):
        engine.svf_L.ic1eq = -(1 << 31)
        engine.svf_R.ic2eq = (1 << 31) - 1
    for block in range(80):
        left, right = rng.integers(-32768, 32768, (2, 64)).tolist()
        params = ParamFrame(drive_q15=block * 321, pressure_q15=32767 - block * 103,
                            tone_q15=12000, bias_q15=-12345)
        kwargs = dict(filter_cutoff_q15=block * 607, filter_reso_q15=block * 1013,
                      filter_mode=block % 4)
        hot.set_jit_enabled(False)
        expected = reference.step_block(left, right, params, 9, **kwargs)
        hot.set_jit_enabled(block % 3 != 0)
        result = actual.step_block(left, right, params, 9, **kwargs)
        assert result == expected
        assert vars(actual.svf_L) == vars(reference.svf_L)
        assert vars(actual.svf_R) == vars(reference.svf_R)
        assert actual.params == reference.params


@pytest.mark.parametrize('mode', [0, 1, 2, 3, np.int64(2), True])
def test_material_constructor_prepares_every_accepted_mode_before_audio(monkeypatch, mode):
    engine = MaterialEngine()
    signatures = list(svf_jit.channel.signatures)
    assert len(signatures) == 1

    def forbidden(*args, **kwargs):
        raise AssertionError('SVF compiled during audio processing')

    monkeypatch.setattr(svf_jit.channel, 'compile', forbidden)
    result = engine.step_block([1] * 64, [-1] * 64, ParamFrame(), 9,
                               filter_cutoff_q15=12000, filter_mode=mode)
    assert len(result[0]) == 64 and svf_jit.channel.signatures == signatures


def test_public_material_preserves_unbounded_reference_fallback():
    reference, actual = ReferenceMaterial(), MaterialEngine()
    left = [1 << 100] * 64
    right = [-(1 << 100)] * 64
    for engine in (reference, actual):
        engine.svf_L.ic1eq = 1 << 80
    hot.set_jit_enabled(False)
    expected = reference.step_block(left, right, ParamFrame(), 9)
    hot.set_jit_enabled(True)
    assert actual.step_block(left, right, ParamFrame(), 9) == expected
    assert vars(actual.svf_L) == vars(reference.svf_L)
    assert vars(actual.svf_R) == vars(reference.svf_R)


@pytest.mark.parametrize('mode', [1.0, np.float64(2.0), '1'])
def test_invalid_mode_preserves_reference_exception_and_state(mode):
    reference, actual = ReferenceMaterial(), MaterialEngine()
    for engine in (reference, actual):
        engine.svf_L.ic1eq = 12345678
        engine.svf_R.ic2eq = -12345678
    args = ([123] * 64, [-456] * 64, ParamFrame(drive_q15=1234), 9)
    hot.set_jit_enabled(False)
    try:
        reference.step_block(*args, filter_mode=mode)
    except (TypeError, ValueError) as error:
        hot.set_jit_enabled(True)
        with pytest.raises(type(error)):
            actual.step_block(*args, filter_mode=mode)
    else:
        pytest.fail('Reference unexpectedly accepted invalid mode')
    assert vars(actual.svf_L) == vars(reference.svf_L)
    assert vars(actual.svf_R) == vars(reference.svf_R)
    assert actual.params == reference.params
