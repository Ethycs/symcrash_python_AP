"""Independent arithmetic/state checks for WSE-STD-MATERIAL v0 corrections."""
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction

import numpy as np
import pytest

from symcrash_python_ap.frames import MaterialParams, ParamFrame
from symcrash_python_ap.material.material_engine import MaterialEngine, CatastropheState, RankState
from symcrash_python_ap.material.building_blocks import catastrophe_cusp_sample


def clip(x):
    return min(32767, max(-32768, x))


def multiply(a, b):
    return clip(round(Fraction(int(a) * int(b), 32768)))


def soft_clip(x):
    if abs(x) <= 24576:
        return x
    return (1 if x >= 0 else -1) * min(32767, 24576 + (abs(x) - 24576) // 2)


def cusp_reference(sample, state, pressure, bias):
    square = multiply(state, state)
    cube = multiply(square, state)
    force = clip(-4 * cube - 2 * multiply(pressure, state) - bias + sample // 4)
    state = clip(state + multiply(16384, force))
    return state, soft_clip(state)


@pytest.mark.parametrize('rate', [0, 1, 256, 16384, 32767])
@pytest.mark.parametrize('initial', [-32768, 0, 32767])
def test_material_slew_matches_fraction_oracle_for_every_parameter_and_retarget(rate, initial):
    names = tuple(asdict(MaterialParams()))
    current = MaterialParams(**{name: initial for name in names})
    reference = asdict(current)
    targets = [ParamFrame(drive_q15=32000, tone_q15=0, bias_q15=-16001,
                          pressure_q15=12001, cut_q15=511), ParamFrame()]
    for target in targets:
        original = deepcopy(target)
        for _ in range(20):
            reference = {name: clip(value + round(Fraction((getattr(target, name) - value) * rate, 32768)))
                         for name, value in reference.items()}
            current.slew_from(target, rate)
            assert asdict(current) == reference
        assert target == original


@pytest.mark.parametrize('delta,expected', [(64, 0), (192, 2), (320, 2), (-64, 0), (-192, -2), (-320, -2)])
def test_slew_rounds_the_delta_ties_to_even(delta, expected):
    current = MaterialParams(bias_q15=101)
    current.slew_from(ParamFrame(bias_q15=101 + delta))
    assert current.bias_q15 == 101 + expected


@pytest.mark.parametrize('rate', [-1, 32768, 0.5, True])
def test_invalid_material_slew_rate_fails_before_any_parameter_changes(rate):
    current = MaterialParams()
    before = deepcopy(current)
    with pytest.raises(ValueError, match='slew rate'):
        current.slew_from(ParamFrame(drive_q15=0), rate)
    assert current == before


def test_proportional_slew_retains_specified_integer_deadband():
    current = MaterialParams(drive_q15=0)
    for _ in range(2000):
        current.slew_from(ParamFrame(drive_q15=32767))
    assert current.drive_q15 == 32703  # Difference 64 produces a tie rounded to zero.


@pytest.mark.parametrize('initial', [-32768, -20000, 0, 16384, 32767])
@pytest.mark.parametrize('pressure,bias', [(-32768, 32767), (-8000, 6000), (0, 0), (32767, -32768)])
def test_cusp_matches_independent_recurrence_in_both_entry_points(initial, pressure, bias):
    engine = MaterialEngine()
    engine.params.pressure_q15, engine.params.bias_q15 = pressure, bias
    a, b = CatastropheState(s=initial), CatastropheState(s=initial)
    reference = initial
    # Includes repeated signed extrema, silence, and values at quarter-sample ties.
    samples = ([32767, -32768, 0, 1, -1, 3, -3] * 5 +
               [int(x) for x in np.random.default_rng(439).integers(-32768, 32768, 50)])
    for sample in samples:
        reference, expected = cusp_reference(sample, reference, pressure, bias)
        assert catastrophe_cusp_sample(np.int16(sample), a, pressure, bias) == expected
        assert engine._cat_cusp_sample(sample, b) == expected
        assert a.s == b.s == reference


def test_cusp_cubic_coefficient_and_half_rate_have_literal_expected_result():
    state = CatastropheState(s=16384)
    # s=.5, s^3=.125, force=-4*.125=-.5; ds=.5*(-.5)=-.25.
    assert catastrophe_cusp_sample(0, state) == 8192
    assert state.s == 8192


def test_cusp_force_cancellation_happens_before_saturation():
    state = CatastropheState(s=30000)
    expected_state, expected = cusp_reference(32767, 30000, -32768, -32768)
    assert catastrophe_cusp_sample(32767, state, -32768, -32768) == expected
    assert state.s == expected_state
    assert expected_state == 30188  # Prematurely clipping the -4*s^3 term rails instead.


def test_fold_and_cusp_switches_preserve_independent_stereo_state():
    engine = MaterialEngine()
    params = ParamFrame(pressure_q15=-12000, bias_q15=1500)
    engine.step_block([12000] * 64, [-18000] * 64, params, 4)
    held_fold = deepcopy(engine.fold)
    assert engine.fold.left.s != engine.fold.right.s
    assert engine.cusp.left.s == engine.cusp.right.s == 0
    engine.step_block([-3000] * 64, [24000] * 64, params, 5)
    assert engine.fold == held_fold
    held_cusp = deepcopy(engine.cusp)
    assert engine.cusp.left.s != engine.cusp.right.s
    reference = deepcopy(engine)
    reference.cusp.left.s = reference.cusp.right.s = -12345  # Cannot bleed into Fold.
    assert engine.step_block([4000] * 64, [0] * 64, params, 4) == reference.step_block([4000] * 64, [0] * 64, params, 4)
    assert engine.cusp == held_cusp
    assert engine.cat_L is engine.fold.left and engine.cat_R is engine.fold.right
    assert engine.cusp_L is engine.cusp.left and engine.cusp_R is engine.cusp.right


@pytest.mark.parametrize('windows', [[17] * 150, [9] * 150,
                                    [17] * 31 + [9] * 45 + [17] * 39 + [9] * 35,
                                    [9, 17] * 75])
def test_rank_keeps_chronological_history_across_window_changes(windows):
    state, engine = RankState(), MaterialEngine()
    history = [0] * 17
    samples = np.random.default_rng(78).integers(-32768, 32768, len(windows)).tolist()
    for sample, window in zip(samples, windows):
        history = (history + [sample])[-17:]
        retained = sorted(history[-window:])[2:-2]
        expected = round(Fraction(sum(retained), len(retained)))
        assert engine._rank_sample(sample, state, window) == expected
        assert 0 <= state.idx < 17
        assert state.buffer[state.idx:] + state.buffer[:state.idx] == history


def test_negative_rank_mean_uses_nearest_even_instead_of_floor():
    engine, state = MaterialEngine(), RankState()
    values = [-100, -100, -1, 0, 0, 0, 0, 100, 100]
    for sample in values:
        result = engine._rank_sample(sample, state, 9)
    assert result == 0  # -1/5, rather than floor(-1/5)=-1.


def test_rank_output_with_dynamic_drive_matches_independent_stream_and_slew():
    engine = MaterialEngine()
    histories = [[0] * 17, [0] * 17]
    drive = 8192
    for tick in range(220):
        target = 32767 if tick < 120 else 2000
        drive += round(Fraction((target - drive) * 256, 32768))
        window = 17 if drive < 16384 else 9
        channels = [[(tick * 79 + i * 2101) % 60001 - 30000 for i in range(64)],
                    [(tick * 31 + i * 1957) % 50001 - 25000 for i in range(64)]]
        expected = []
        for ch, samples in enumerate(channels):
            output = []
            for sample in samples:
                histories[ch] = (histories[ch] + [sample])[-17:]
                kept = sorted(histories[ch][-window:])[2:-2]
                filtered = round(Fraction(sum(kept), len(kept)))
                output.append(clip(multiply(filtered, drive) + multiply(sample, 32767 - drive)))
            expected.append(output)
        actual = engine.step_block(*channels, ParamFrame(drive_q15=target), 2)
        assert actual == tuple(expected)
        assert engine.params.drive_q15 == drive
