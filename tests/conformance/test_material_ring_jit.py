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
from symcrash_python_ap.material import material_ring_jit as jit
engine = MaterialEngine()
for kernel in (jit.reverb_channel, jit.delay_channel, jit.ks_channel,
               jit.chorus_delays, jit.chorus_channel):
    assert kernel.signatures
    kernel.disable_compile()
engine._reverb([32767] * 64, [-32768] * 64)
engine._delay_line([32767] * 64, [-32768] * 64)
engine._chorus([32767] * 64, [-32768] * 64)
engine._karplus_strong([32767] * 64, [-32768] * 64)
'''
    subprocess.run([sys.executable, '-c', script], check=True, timeout=60)


PROGRAMS = {6: ('delay', '_delay_line'), 7: ('chorus', '_chorus'), 10: ('ks', '_karplus_strong')}


@pytest.mark.parametrize('program', [6, 7, 10])
@pytest.mark.parametrize('count,capacity', [(1,1), (7,3), (64,11), (64,960), (257,96)])
@pytest.mark.parametrize('params', [(-32768,)*4, (32767,)*4, (16384,21000,12345,-11317)])
def test_delay_programs_preserve_every_cell_across_wraps(program, count, capacity, params):
    engine, reference = MaterialEngine(count), ReferenceMaterial(count)
    prefix, method = PROGRAMS[program]
    rng = np.random.default_rng(7900 + program)
    identities = []
    for side in ('L', 'R'):
        state = getattr(engine, f'{prefix}_{side}')
        state.max_samples = capacity
        state.buffer = rng.integers(-32768, 32768, capacity + 3).tolist()
        state.write_idx = capacity - 1
        if program == 7:
            state.lfo_phase = 0xffffffff if side == 'L' else 0x80000000
        if program == 10:
            state.delay_samples = 1 if side == 'L' else capacity
            state.prev_sample = -32768 if side == 'L' else 32767
        identities.append(id(state.buffer))
        setattr(reference, f'{prefix}_{side}', deepcopy(state))
    for obj in (engine, reference):
        for name, value in zip(('drive_q15','tone_q15','pressure_q15','bias_q15'), params):
            setattr(obj.params, name, value)
    for _ in range(5):
        channels = rng.integers(-32768,32768,(2,count)).tolist()
        before = deepcopy(channels)
        assert getattr(engine,method)(*channels) == getattr(reference,method)(*channels)
        assert channels == before
        for index, side in enumerate(('L','R')):
            state = getattr(engine,f'{prefix}_{side}')
            assert asdict(state) == asdict(getattr(reference,f'{prefix}_{side}'))
            assert id(state.buffer) == identities[index]


@pytest.mark.parametrize('program', [6, 7, 10])
def test_default_delay_capacities_and_live_parameter_changes(program):
    engine, reference = MaterialEngine(), ReferenceMaterial()
    prefix, method = PROGRAMS[program]
    rng = np.random.default_rng(7903)
    for step in range(8):
        values = rng.integers(-32768,32768,4).tolist()
        for obj in (engine, reference):
            for name,value in zip(('drive_q15','tone_q15','pressure_q15','bias_q15'),values):
                setattr(obj.params,name,value)
        channels = rng.integers(-32768,32768,(2,64)).tolist()
        assert getattr(engine,method)(*channels) == getattr(reference,method)(*channels)
        for side in ('L','R'):
            assert asdict(getattr(engine,f'{prefix}_{side}')) == asdict(getattr(reference,f'{prefix}_{side}'))


@pytest.mark.parametrize('program', [6, 7, 10])
def test_delay_recurrences_match_independent_fraction_oracle(program):
    from symcrash_python_ap.wavelet.sine_core import sine_q15
    engine = MaterialEngine(17)
    prefix, method = PROGRAMS[program]
    for side in ('L','R'):
        state=getattr(engine,f'{prefix}_{side}')
        state.max_samples=5; state.buffer=[-32768,32767,-3,1,16384]; state.write_idx=4
        if program==7:state.lfo_phase=0xfffffff0
        if program==10:state.delay_samples=3;state.prev_sample=-32768
    engine.params.drive_q15=1000;engine.params.tone_q15=-12345
    engine.params.pressure_q15=16384;engine.params.bias_q15=0
    expected=deepcopy(getattr(engine,f'{prefix}_L'))
    clip=lambda x:max(-32768,min(32767,x))
    mul=lambda a,b:clip(round(Fraction(a*b,32768)))
    soft=lambda x:x if abs(x)<=24576 else (1 if x>=0 else -1)*min(32767,24576+(abs(x)-24576)//2)
    source=[-32768,32767,-3,-1,1,3,16384,-16384]*2+[1]
    outputs=[]
    for x in source:
        if program==6:delay=max(1,min(1000*48000//32768,4))
        elif program==10:delay=3
        else:
            inc=round(Fraction((32767+49*1000)*(1<<32),10*32767*48000))
            expected.lfo_phase=(expected.lfo_phase+inc)%(1<<32)
            depth=(-12345*480)//32768
            delay=max(1,min(720+round(Fraction(sine_q15(expected.lfo_phase)*depth,32768)),4))
        delayed=expected.buffer[(expected.write_idx-delay)%5]
        if program==6:
            write=clip(x+mul(delayed,-12345))
            value=soft(clip(mul(x,32767-16384)+mul(delayed,16384)))
        elif program==7:
            write=x;value=clip(mul(x,32767-16384)+mul(delayed,16384))
        else:
            average=(expected.prev_sample+delayed)//2
            filtered=delayed+mul(average-delayed,-12345)
            write=clip(mul(x,1000)+mul(filtered,29491+16384*3276//32768))
            expected.prev_sample=delayed;value=delayed
        expected.buffer[expected.write_idx]=write
        expected.write_idx=(expected.write_idx+1)%5
        outputs.append(value)
    assert getattr(engine,method)(source,source)[0] == outputs
    assert asdict(getattr(engine,f'{prefix}_L')) == asdict(expected)


@pytest.mark.parametrize('program', [6, 7, 10])
def test_delay_program_alias_and_nonstandard_input_fallbacks(program):
    engine=MaterialEngine();prefix,_=PROGRAMS[program]
    left,right=getattr(engine,f'{prefix}_L'),getattr(engine,f'{prefix}_R')
    before=deepcopy((asdict(left),asdict(right)))
    assert jit.try_delay_program(engine,np.ones(64,dtype=np.int16),[1]*64,program) is None
    assert before==(asdict(left),asdict(right))
    right.buffer=left.buffer
    before=deepcopy((asdict(left),asdict(right)))
    assert jit.try_delay_program(engine,[1]*64,[2]*64,program) is None
    assert before==(asdict(left),asdict(right))


def test_chorus_boolean_writes_keep_reference_scalar_types():
    engine=MaterialEngine()
    assert jit.try_delay_program(engine,[True]*64,[False]*64,7) is None
    engine._chorus([True]*64,[False]*64)
    assert all(type(x) is bool for x in engine.chorus_L.buffer[:64])
