"""Independent RMS arithmetic and observable Safety control behavior."""
from dataclasses import asdict
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
import math

import numpy as np
import pytest

from symcrash_python_ap.frames import ParamFrame
from symcrash_python_ap.safety import safety_engine as safety
from symcrash_python_ap.safety import safety_tail_jit as jit
from symcrash_python_ap.safety.rms import rms_from_energy
from symcrash_python_ap.vm import hotpath_jit as hot


def reference_rms(energy, count):
    if count == 0:
        return 0
    with localcontext() as ctx:
        ctx.prec = 70
        value = (Decimal(int(energy)) / Decimal(int(count))).sqrt()
        return min(32767, int(value.to_integral_value(rounding=ROUND_HALF_EVEN)))


@pytest.fixture(params=[False, True], ids=['reference', 'jit'])
def execution_path(request):
    if request.param and not hot.NUMBA_AVAILABLE:
        pytest.skip('numba unavailable')
    previous = hot.jit_enabled()
    hot.set_jit_enabled(request.param)
    yield
    hot.set_jit_enabled(previous)


@pytest.mark.parametrize('amplitude', [-32768, -32767, -2048, -128, -1, 0, 1, 32,
                                      64, 127, 128, 129, 256, 257, 1024, 2048,
                                      4096, 16384, 32767])
@pytest.mark.filterwarnings('error')
def test_constant_signal_rms_is_its_amplitude_with_saturating_magnitude(amplitude):
    samples = np.full(64, amplitude, dtype=np.int16)
    expected = min(32767, abs(amplitude))
    assert safety.compute_local_rms(samples, 32, 32) == expected
    # The retained sqrt_lut argument is deliberately empty: RMS must not use it.
    assert jit._local_rms(samples, 64, 32, 32, np.zeros(0, dtype=np.int64)) == expected
    metrics = safety.SafetyEngine().measure_metrics(samples, samples)
    assert metrics.rms_q15 == metrics.peak_q15 == expected
    assert metrics.dc_offset_q15 == amplitude
    assert metrics.hf_energy_q15 == 0


@pytest.mark.parametrize('root', [0, 1, 2, 3, 31, 127, 128, 4095, 12345, 32766])
def test_rms_half_integer_boundaries_use_nearest_even_before_saturation(root):
    midpoint_energy = (2 * root + 1) ** 2
    for energy in (midpoint_energy - 1, midpoint_energy, midpoint_energy + 1):
        expected = reference_rms(energy, 4)
        assert rms_from_energy(energy, 4) == expected
        assert jit._rms_from_energy(energy, 4) == expected
    assert rms_from_energy(midpoint_energy, 4) == root + (root & 1)


def test_every_int16_single_sample_has_exact_rms():
    for value in range(-32768, 32768):
        expected = min(32767, abs(value))
        assert rms_from_energy(value * value, 1) == expected
        assert jit._rms_from_energy(value * value, 1) == expected


@pytest.mark.parametrize('count', [1, 3, 7, 64, 129, 4096])
def test_rms_random_energies_match_high_precision_oracle(count):
    rng = np.random.default_rng(6016 + count)
    for _ in range(60):
        samples = rng.integers(-32768, 32768, count, dtype=np.int16)
        energy = sum(int(x) ** 2 for x in samples)
        expected = reference_rms(energy, count)
        assert rms_from_energy(energy, count) == expected
        assert jit._rms_from_energy(energy, count) == expected


@pytest.mark.parametrize('energy,count', [(-1, 1), (1, 0), (0, -1)])
def test_invalid_energy_is_rejected(energy, count):
    for fn in (rms_from_energy, jit._rms_from_energy):
        with pytest.raises(ValueError):
            fn(energy, count)
    assert rms_from_energy(0, 0) == jit._rms_from_energy(0, 0) == 0


@pytest.mark.parametrize('center,window', [(-10, 2), (-1, 4), (0, 3), (2, 1),
                                          (3, 0), (6, 2), (10, 2)])
def test_local_rms_clips_the_window_and_handles_empty_intersections(center, window):
    samples = [1, -3, 7, -11, 19, -37, 101]
    selected = [value for i, value in enumerate(samples) if center - window <= i < center + window]
    expected = reference_rms(sum(x * x for x in selected), len(selected))
    assert safety.compute_local_rms(samples, center, window) == expected
    assert jit._local_rms(np.array(samples, dtype=np.int16), len(samples), center,
                          window, np.zeros(0, dtype=np.int64)) == expected


def test_empty_arrays_and_negative_windows_have_defined_behavior():
    assert safety.compute_local_rms(np.zeros(0, dtype=np.int16), 0) == 0
    with pytest.raises(ValueError, match='nonnegative'):
        safety.compute_local_rms([1], 0, -1)
    with pytest.raises(ValueError, match='nonnegative'):
        jit._local_rms(np.array([1]), 1, 0, -1, np.zeros(0, dtype=np.int64))


@pytest.mark.parametrize('left,right', [(1000, 0), (0, -1000), (3, 4),
                                      (32767, -32768), (128, -128)])
def test_stereo_metrics_average_channel_energies_once(left, right):
    engine = safety.SafetyEngine()
    actual = engine.measure_metrics([left] * 64, [right] * 64)
    assert actual.rms_q15 == reference_rms(left * left + right * right, 2)


@pytest.mark.filterwarnings('error')
def test_metric_arithmetic_widens_numpy_extrema():
    left = np.array([-32768, 32767] * 32, dtype=np.int16)
    right = np.array([32767, -32768] * 32, dtype=np.int16)
    engine = safety.SafetyEngine()
    assert asdict(engine.measure_metrics(left, right)) == asdict(
        engine.measure_metrics(left.tolist(), right.tolist()))


@pytest.mark.parametrize('amplitude,degenerate', [(127, True), (128, False),
                                                (256, False), (2048, False)])
def test_energy_detection_uses_the_authored_threshold(amplitude, degenerate):
    samples = [amplitude] * 64
    indices, actual = safety.detect_discontinuities_energy_aware(samples)
    assert actual is degenerate
    if not degenerate:
        assert indices == []
        assert safety.apply_degenerate_hold(samples, 0) == samples


def test_odd_block_detection_includes_the_final_sample():
    samples = [0, 0, 1000]
    assert not safety.detect_discontinuities_energy_aware(samples)[1]
    assert safety.apply_degenerate_hold(samples, 0) == samples
    indices = np.empty(3, dtype=np.int64)
    _, is_degenerate = jit._detect_energy_aware(np.array(samples), 3, 0, 8192,
        4096, 128, 32, np.zeros(0, dtype=np.int64), indices)
    assert not is_degenerate


def test_degenerate_hysteresis_uses_real_rms(execution_path):
    engine = safety.SafetyEngine()
    engine.apply_c1_patch([32] * 64, 9)
    assert engine.degenerate_mode[9]
    engine.apply_c1_patch([256] * 64, 9)
    assert engine.degenerate_mode[9]
    engine.apply_c1_patch([257] * 64, 9)
    assert not engine.degenerate_mode[9]
    engine.apply_c1_patch([128] * 64, 9)
    assert not engine.degenerate_mode[9]
    engine.apply_c1_patch([127] * 64, 9)
    assert engine.degenerate_mode[9]


def test_regulation_tracks_true_rms_and_attenuates_over_target_signal(execution_path):
    engine = safety.SafetyEngine()
    samples = [24000 if i % 32 < 16 else -24000 for i in range(64)]
    outputs = []
    for _ in range(80):
        outputs.append(engine.step_block(samples, samples, ParamFrame(K_mode=0),
            apply_master_ola=False, apply_slew_limit=False, apply_dc_removal=False)[0])
    assert 23000 < engine.rg.E_hat < 24000
    assert 21000 < engine.q_gain < 24000
    assert max(abs(x) for x in outputs[-1]) < max(abs(x) for x in outputs[0])
    assert any(outputs[-1])


def test_reference_jit_and_alternating_execution_preserve_audio_and_all_state():
    previous = hot.jit_enabled()
    engines = [safety.SafetyEngine() for _ in range(3)]
    rng = np.random.default_rng(160601)
    try:
        for block in range(80):
            amplitude = (64, 127, 128, 256, 257, 2048, 16000, 32767)[block % 8]
            left = rng.integers(-amplitude, amplitude + 1, 64).tolist()
            right = rng.integers(-amplitude, amplitude + 1, 64).tolist()
            outputs = []
            for index, engine in enumerate(engines):
                hot.set_jit_enabled(index == 1 or (index == 2 and block % 2 == 0))
                rendered_l = engine.apply_c1_patch(left, 7)
                rendered_r = engine.apply_c1_patch(right, 9)
                outputs.append(engine.step_block(rendered_l, rendered_r,
                    ParamFrame(K_mode=block % 4), apply_master_ola=block % 3 != 0,
                    apply_slew_limit=block % 5 != 0, apply_dc_removal=block % 7 != 0))
            assert outputs[0] == outputs[1] == outputs[2], block
            for engine in engines[1:]:
                for name in ('rg', 'lim_L', 'lim_R'):
                    assert asdict(getattr(engine, name)) == asdict(getattr(engines[0], name))
                for name in ('q_gain', 'q_hf_cut', 'prev_sample_L', 'prev_sample_R',
                             'master_ola_L', 'master_ola_R', 'degenerate_mode', 'prev_output'):
                    assert getattr(engine, name) == getattr(engines[0], name), name
    finally:
        hot.set_jit_enabled(previous)


def test_rms_does_not_call_floating_sqrt_or_legacy_lookup(monkeypatch):
    jit._rms_from_energy(12345, 2)  # Compile before replacing math functions.
    def forbidden(*args, **kwargs):
        pytest.fail('RMS must use widened integer energy and integer root')
    monkeypatch.setattr(math, 'sqrt', forbidden)
    monkeypatch.setattr(np, 'sqrt', forbidden)
    monkeypatch.setattr(safety, 'sqrt_q15', forbidden)
    for value in (1, 127, 128, 2048, 32767):
        assert safety.compute_local_rms([value] * 64, 32, 32) == value
        assert safety.SafetyEngine().measure_metrics([value] * 64, [value] * 64).rms_q15 == value
        assert jit._rms_from_energy(value * value * 64, 64) == value
