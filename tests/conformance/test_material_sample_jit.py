"""Independent arithmetic, state and fallback checks for Tape/Rank acceleration."""
from dataclasses import fields, is_dataclass
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import pytest

from symcrash_python_ap.frames import ParamFrame
from symcrash_python_ap.fixed_rom import TANH_Q15
from symcrash_python_ap.material.material_engine import MaterialEngine
from symcrash_python_ap.material import material_sample_jit as accelerator
from symcrash_python_ap.vm import hotpath_jit as hot


class ReferenceMaterial(MaterialEngine):
    """The unchanged sample methods: subclasses bypass optional block kernels."""


@pytest.fixture(autouse=True)
def jit_path():
    before = hot.jit_enabled()
    if not hot.NUMBA_AVAILABLE:
        pytest.skip('Numba required for acceleration checks')
    hot.set_jit_enabled(True)
    yield
    hot.set_jit_enabled(before)


def clip(x):
    return max(-32768, min(32767, x))


def multiply(a, b):
    return clip(round(Fraction(int(a) * int(b), 32768)))


def tape_oracle(samples, drive, bias, tone, m, pole):
    values = []
    for sample in samples:
        driven = clip(multiply(sample, drive) + bias)
        target = TANH_Q15[(driven + 32768) // 256]
        damp = max(0, 32767 - multiply(drive // 4, multiply(m, m)))
        delta = multiply(multiply(target - m, min(32767, drive + 16384)), damp)
        m = clip(m + delta // 2)
        pole = clip(multiply(32767 - tone, pole) + multiply(tone, m))
        values.append(pole)
    return values, m, pole


def structural(value):
    if is_dataclass(value):
        return {f.name: structural(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (list, tuple)):
        return [structural(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, '__dict__'):
        return {k: structural(v) for k, v in vars(value).items() if k != '_strategies'}
    return value


@pytest.mark.parametrize('drive', [-32768, 0, 16383, 16384, 32767])
@pytest.mark.parametrize('tone', [-32768, 0, 32767])
@pytest.mark.parametrize('bias', [-32768, 32767])
def test_tape_matches_fraction_oracle_and_both_persistent_states(drive, tone, bias):
    engine = MaterialEngine()
    engine.params.drive_q15, engine.params.tone_q15, engine.params.bias_q15 = drive, tone, bias
    engine.tape_L.m, engine.tape_R.m = -32768, 32767
    engine.lp_zL, engine.lp_zR = 32767, -32768
    expected = [(-32768, 32767), (32767, -32768)]
    rng = np.random.default_rng(7411)
    for _ in range(8):
        channels = rng.integers(-32768, 32768, (2, 64)).tolist()
        outputs = []
        for ch, samples in enumerate(channels):
            values, m, pole = tape_oracle(samples, drive, bias, tone, *expected[ch])
            outputs.append(values)
            expected[ch] = m, pole
        assert engine._tape_bus(*channels) == tuple(outputs)
        assert (engine.tape_L.m, engine.lp_zL) == expected[0]
        assert (engine.tape_R.m, engine.lp_zR) == expected[1]


def test_tape_visits_full_signed_input_domain_without_a_different_curve():
    accelerator.prepare()
    samples = np.arange(-32768, 32768, dtype=np.int16)
    wanted, m, pole = tape_oracle(samples, 32767, 0, 32767, -32768, 32767)
    output, actual_m, actual_pole = accelerator.tape_channel(samples, 32767, 0, 32767, -32768, 32767)
    np.testing.assert_array_equal(output, wanted)
    assert (actual_m, actual_pole) == (m, pole)
    assert accelerator._TANH_ROM.dtype == np.int16
    assert accelerator._TANH_ROM.nbytes == 512 and not accelerator._TANH_ROM.flags.writeable
    assert accelerator._TANH_ROM.tolist() == list(TANH_Q15)


@pytest.mark.parametrize('start', [0, 1, 8, 16])
def test_rank_keeps_all_history_and_buffer_identity_through_window_changes(start):
    engine = MaterialEngine()
    histories = [[i * 1701 - 15000 for i in range(17)],
                 [16000 - i * 1801 for i in range(17)]]
    buffers = [engine.rank_L.buffer, engine.rank_R.buffer]
    for state, history in zip((engine.rank_L, engine.rank_R), histories):
        state.idx = start
        state.buffer[:] = history[-start:] + history[:-start] if start else history
    rng = np.random.default_rng(7412)
    for drive in [0, 32767, 16383, 16384, -32768, 20000, 1000] * 4:
        engine.params.drive_q15 = drive
        window = 17 if drive < 16384 else 9
        channels = rng.integers(-32768, 32768, (2, 64)).tolist()
        wanted = []
        for ch, samples in enumerate(channels):
            values = []
            for sample in samples:
                histories[ch] = (histories[ch] + [sample])[-17:]
                retained = sorted(histories[ch][-window:])[2:-2]
                filtered = round(Fraction(sum(retained), len(retained)))
                values.append(clip(multiply(filtered, drive) + multiply(sample, 32767 - drive)))
            wanted.append(values)
        assert engine._rank_grit(*channels) == tuple(wanted)
        for ch, state in enumerate((engine.rank_L, engine.rank_R)):
            assert state.buffer is buffers[ch]
            assert state.buffer[state.idx:] + state.buffer[:state.idx] == histories[ch]


def test_program_switching_reference_jit_state_and_proportional_slew():
    expected, actual = ReferenceMaterial(seed=74), MaterialEngine(seed=74)
    rng = np.random.default_rng(7413)
    for block in range(96):
        program = [1, 2, 0, 2, 9, 1, 3, 1][block % 8]
        channels = rng.integers(-32768, 32768, (2, 64)).tolist()
        params = ParamFrame(drive_q15=32767 if block % 3 else -32768,
                            tone_q15=(block * 977) % 65536 - 32768,
                            bias_q15=32767 if block % 2 else -32768)
        hot.set_jit_enabled(False)
        reference = expected.step_block(*channels, params, program)
        hot.set_jit_enabled(block % 5 != 0)
        assert actual.step_block(*channels, params, program) == reference
        assert structural(actual) == structural(expected)


@pytest.mark.parametrize('program', [1, 2])
@pytest.mark.parametrize('kind', ['wide-input', 'wide-state', 'short-input', 'alias-state', 'alias-buffer', 'custom-history'])
def test_outside_domain_retains_reference_output_exception_and_partial_state(program, kind):
    reference, actual = ReferenceMaterial(), MaterialEngine()
    channels = [[(i * 997) % 65536 - 32768 for i in range(64)], [19] * 64]
    if kind == 'wide-input':
        channels[0][3] = 1 << 90
    elif kind == 'short-input':
        channels[1] = channels[1][:3]
    for engine in (reference, actual):
        if kind == 'wide-state':
            engine.tape_L.m = 1 << 80
            engine.rank_L.buffer[0] = 1 << 80
        elif kind == 'alias-state':
            engine.tape_R = engine.tape_L
            engine.rank_R = engine.rank_L
        elif kind == 'alias-buffer':
            engine.rank_R.buffer = engine.rank_L.buffer
        elif kind == 'custom-history':
            engine.rank_L.buffer[:] = [7] * 23
            engine.rank_R.buffer[:] = [11] * 23
    args = (*channels, ParamFrame(drive_q15=19000), program)
    hot.set_jit_enabled(False)
    try:
        wanted = reference.step_block(*args)
    except Exception as error:
        hot.set_jit_enabled(True)
        with pytest.raises(type(error)):
            actual.step_block(*args)
    else:
        hot.set_jit_enabled(True)
        assert actual.step_block(*args) == wanted
    assert structural(actual) == structural(reference)


@pytest.mark.parametrize('method,program', [('_tape_sample', 1), ('_tone_filter', 1), ('_rank_sample', 2)])
def test_sample_method_overrides_keep_the_reference_path(method, program):
    reference, actual = ReferenceMaterial(), MaterialEngine()
    for engine in (reference, actual):
        setattr(engine, method, lambda *args: -1234)
    args = ([100] * 64, [-100] * 64, ParamFrame(), program)
    assert actual.step_block(*args) == reference.step_block(*args)
    # Ignore the custom callable while comparing DSP state.
    delattr(reference, method)
    delattr(actual, method)
    assert structural(actual) == structural(reference)


@pytest.mark.parametrize('container', ['list', 'tuple', 'bool-list', 'bool-tuple'])
def test_warmed_signatures_handle_python_inputs_without_compiling_in_audio(monkeypatch, container):
    engine = MaterialEngine()
    reference = ReferenceMaterial()
    kernels = (accelerator.tape_channel, accelerator.rank_channel)
    signatures = [list(k.signatures) for k in kernels]
    assert all(len(s) == 1 for s in signatures)
    calls = [0, 0]

    def forbidden(*args, **kwargs):
        raise AssertionError('Compiled during audio processing')

    for kernel in kernels:
        monkeypatch.setattr(kernel, 'compile', forbidden)

    def tape(*args):
        calls[0] += 1
        return kernels[0](*args)

    def rank(*args):
        calls[1] += 1
        return kernels[1](*args)

    monkeypatch.setattr(accelerator, 'tape_channel', tape)
    monkeypatch.setattr(accelerator, 'rank_channel', rank)
    channel = list(range(0, 128, 2))
    if container.startswith('bool'):
        channel = [i % 3 == 0 for i in channel]
    if container.endswith('tuple'):
        channel = tuple(channel)
    for program in (1, 2):
        result = engine.step_block(channel, channel[::-1], ParamFrame(), program)
        assert all(len(ch) == 64 for ch in result)
        assert result == reference.step_block(channel, channel[::-1], ParamFrame(), program)
    assert calls == [2, 2]  # Both stereo kernels actually ran; no silent fallback.
    assert [list(k.signatures) for k in kernels] == signatures


@pytest.mark.parametrize('program', [1, 2])
@pytest.mark.parametrize('dtype', [np.int16, np.int64, np.uint16, np.bool_])
def test_numpy_inputs_keep_reference_scalar_arithmetic(monkeypatch, program, dtype):
    reference, actual = ReferenceMaterial(), MaterialEngine()

    def forbidden(*args, **kwargs):
        raise AssertionError('NumPy inputs must retain reference scalar arithmetic')

    monkeypatch.setattr(accelerator, 'tape_channel', forbidden)
    monkeypatch.setattr(accelerator, 'rank_channel', forbidden)
    channel = np.arange(128).astype(dtype)[::2]
    channel.flags.writeable = False
    args = (channel, channel[::-1], ParamFrame(), program)
    # Narrow reference scalars can overflow. Preserve that existing behavior.
    with np.errstate(over='ignore'):
        expected = reference.step_block(*args)
        observed = actual.step_block(*args)
    assert observed == expected
    assert structural(actual) == structural(reference)


def test_unprepared_adapter_falls_back_without_compiling_or_mutating(monkeypatch):
    engine = MaterialEngine()
    before = structural(engine)
    monkeypatch.setattr(accelerator, '_prepared', False)
    assert accelerator.try_tape(engine, [1]*64, [-1]*64) is None
    assert accelerator.try_rank(engine, [1]*64, [-1]*64) is None
    assert structural(engine) == before


@pytest.mark.parametrize('program', [1, 2])
def test_custom_input_container_retains_per_sample_access_order(program):
    class Observed(list):
        def __getitem__(self, key):
            accesses.append(key)
            return super().__getitem__(key)

    reference, actual = ReferenceMaterial(), MaterialEngine()
    results, traces = [], []
    for engine in (reference, actual):
        accesses = []
        results.append(engine.step_block(Observed([123]*64), Observed([-456]*64), ParamFrame(), program))
        traces.append(accesses)
    assert results[0] == results[1] and traces[0] == traces[1]
    assert structural(actual) == structural(reference)


def test_fresh_process_warms_before_audio_and_uses_no_runtime_transcendentals(tmp_path):
    import os
    import subprocess

    cache = tmp_path/'jit-cache'
    cache.mkdir()
    code = '''import math, runpy, sys
namespace=runpy.run_path(sys.argv[1])
hot=namespace['hot']; hot.set_jit_enabled(True)
engine=namespace['MaterialEngine']()
accelerator=namespace['accelerator']
kernels=(accelerator.tape_channel,accelerator.rank_channel)
signatures=[list(kernel.signatures) for kernel in kernels]
assert all(len(s)==1 for s in signatures)
def forbidden(*args,**kwargs):
    raise AssertionError('Compilation/transcendental in audio processing')
for kernel in kernels:
    kernel.compile=forbidden
for name in ('sin','cos','tan','log','log2','exp'):
    setattr(math,name,forbidden)
for program in (1,2,1,2):
    result=engine.step_block([32767,-32768]*32,[-32768,32767]*32,namespace['ParamFrame'](),program)
    assert all(len(ch)==64 for ch in result)
assert [list(kernel.signatures) for kernel in kernels]==signatures
print('Cold process prepared both signatures before audio')
'''
    result = subprocess.run([sys.executable,'-c',code,str(Path(__file__).resolve())],
        env=dict(os.environ,NUMBA_CACHE_DIR=str(cache)),capture_output=True,text=True,timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'prepared both signatures before audio' in result.stdout
