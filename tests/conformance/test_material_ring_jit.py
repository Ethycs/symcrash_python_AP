"""Exact reverb recurrence, bounded ring windows, fallback and startup checks."""
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
import subprocess
import sys

import numpy as np
import pytest

from symcrash_python_ap.material.material_engine import MaterialEngine
from symcrash_python_ap.material import material_ring_jit as jit
from symcrash_python_ap.vm import hotpath_jit as hot


class ReferenceMaterial(MaterialEngine):
    pass


@pytest.fixture(autouse=True)
def compiled():
    old = hot.jit_enabled()
    if not hot.NUMBA_AVAILABLE:
        pytest.skip('Numba unavailable')
    hot.set_jit_enabled(True)
    jit.prepare()
    yield
    hot.set_jit_enabled(old)


def initialize(engine, short=False):
    rng = np.random.default_rng(7701)
    for state in (engine.reverb_L, engine.reverb_R):
        if short:
            state.COMB_DELAYS, state.AP_DELAYS = (1, 2, 3, 5), (1, 7)
        for name, delay in zip(jit._NAMES, (*state.COMB_DELAYS, *state.AP_DELAYS)):
            setattr(state, name, rng.integers(-32768, 32768, delay + 3).tolist())
        state.comb_idx[:] = [d - 1 for d in state.COMB_DELAYS]
        state.ap_idx[:] = [d - 1 for d in state.AP_DELAYS]


@pytest.mark.parametrize('count', [1, 7, 64, 257])
@pytest.mark.parametrize('short', [False, True])
@pytest.mark.parametrize('decay,mix', [(-32768, -32768), (0, 0), (32767, 32767), (15000, 16384)])
def test_reverb_matches_original_audio_and_every_state_cell(count, short, decay, mix):
    engine, reference = MaterialEngine(count), ReferenceMaterial(count)
    initialize(engine, short)
    reference.reverb_L, reference.reverb_R = deepcopy(engine.reverb_L), deepcopy(engine.reverb_R)
    engine.params.drive_q15 = reference.params.drive_q15 = decay
    engine.params.pressure_q15 = reference.params.pressure_q15 = mix
    identities = [id(getattr(s, name)) for s in (engine.reverb_L, engine.reverb_R)
                  for name in (*jit._NAMES, 'comb_idx', 'ap_idx')]
    rng = np.random.default_rng(7702)
    signatures = list(jit.reverb_channel.signatures)
    for _ in range(8):
        channels = rng.integers(-32768, 32768, (2, count)).tolist()
        before = deepcopy(channels)
        assert engine._reverb(*channels) == reference._reverb(*channels)
        assert channels == before
        for a, b in ((engine.reverb_L, reference.reverb_L), (engine.reverb_R, reference.reverb_R)):
            assert asdict(a) == asdict(b)
    assert jit.reverb_channel.signatures == signatures
    assert identities == [id(getattr(s, name)) for s in (engine.reverb_L, engine.reverb_R)
                          for name in (*jit._NAMES, 'comb_idx', 'ap_idx')]


def test_reverb_matches_independent_fraction_oracle():
    engine = MaterialEngine(64)
    initialize(engine, short=True)
    expected = deepcopy(engine.reverb_L)
    engine.params.drive_q15, engine.params.pressure_q15 = 27111, -12345
    clip = lambda x: max(-32768, min(32767, x))
    mul = lambda a, b: clip(round(Fraction(a * b, 32768)))
    source = [-32768, 32767, -3, -1, 1, 3, 16384, -16384] * 8
    result = []
    for x in source:
        delayed = []
        for k, name in enumerate(jit._NAMES[:4]):
            buffer = getattr(expected, name)
            i = expected.comb_idx[k]
            y = buffer[i]
            delayed.append(y)
            buffer[i] = clip(x // 4 + mul(y, 27111 * 28000 // 32768))
            expected.comb_idx[k] = (i + 1) % expected.COMB_DELAYS[k]
        value = sum(delayed) // 4
        for k, name in enumerate(jit._NAMES[4:]):
            buffer = getattr(expected, name)
            i = expected.ap_idx[k]
            y = clip(buffer[i] - mul(16384, clip(value - buffer[i])))
            buffer[i] = value
            expected.ap_idx[k] = (i + 1) % expected.AP_DELAYS[k]
            value = y
        result.append(clip(mul(x, 32767 + 12345) + mul(clip(value * 2), -12345)))
    assert engine._reverb(source, source)[0] == result
    assert asdict(engine.reverb_L) == asdict(expected)


@pytest.mark.parametrize('damage', ['stereo', 'buffer', 'index', 'input', 'wide', 'numpy'])
def test_unsupported_state_is_rejected_without_mutation(damage):
    engine = MaterialEngine()
    channels = [[1] * 64, [2] * 64]
    if damage == 'stereo':
        engine.reverb_R = engine.reverb_L
    elif damage == 'buffer':
        engine.reverb_L.comb2 = engine.reverb_L.comb1
    elif damage == 'index':
        engine.reverb_L.comb_idx[0] = -1
    elif damage == 'input':
        channels[0] = engine.reverb_L.comb1
    elif damage == 'wide':
        engine.reverb_R.comb1[0] = 1 << 80
    else:
        engine.reverb_R.comb1[0] = np.int16(4)
    before = deepcopy((asdict(engine.reverb_L), asdict(engine.reverb_R), channels))
    assert jit.try_reverb(engine, *channels) is None
    assert before == (asdict(engine.reverb_L), asdict(engine.reverb_R), channels)


def test_custom_methods_and_reference_switch_preserve_original_dispatch(monkeypatch):
    engine = MaterialEngine()
    calls = []
    def sample(x, state, decay, damping):
        calls.append(x)
        return 0
    monkeypatch.setattr(engine, '_reverb_sample', sample)
    engine._reverb([1] * 64, [2] * 64)
    assert len(calls) == 128
    hot.set_jit_enabled(False)
    assert jit.try_reverb(engine, [1] * 64, [2] * 64) is None


def test_fresh_construction_prepares_reverb_before_audio():
    script = '''
from symcrash_python_ap.material.material_engine import MaterialEngine
from symcrash_python_ap.material.material_ring_jit import reverb_channel
engine = MaterialEngine()
assert reverb_channel.signatures
reverb_channel.disable_compile()
engine._reverb([32767] * 64, [-32768] * 64)
'''
    subprocess.run([sys.executable, '-c', script], check=True, timeout=60)
