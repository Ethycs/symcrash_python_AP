"""CCS tail profile: independent scalar arithmetic, routing, and replay."""
from dataclasses import replace
from fractions import Fraction
from math import isqrt
import math

import numpy as np
import pytest

from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.primitives import TailMode
from symcrash_python_ap.tables.coefficient_tail import (
    CoefficientTailPolicy, DEFAULT_TAIL_POLICY, compile_tail_policy,
)
from symcrash_python_ap.vm import executor, lanes
from symcrash_python_ap.vm.coefficient_tail import limit_energy, scale_rne
from symcrash_python_ap.vm.decode import DecodedPlan
from symcrash_python_ap.vm.state import VMState


def reference_tail(state, cfg):
    """Scalar specification, deliberately independent of runtime helpers.

    Fractions provide signed ties-to-even rounding. The exceptional energy
    overshoot uses a descending gain search, not the production binary search.
    This also serves as the oracle when qualifying changed audio goldens.
    """
    p = cfg.tail_policy or DEFAULT_TAIL_POLICY
    slots = len(state.coeff_state) // 64
    c = [int(v) for v in state.coeff_state]
    traction = cfg.traction_q15

    def power(indices):
        return sum(c[i] ** 2 for i in indices)

    def cap(indices, budget):
        current = power(indices)
        if current <= budget:
            return
        gain = isqrt(budget * 2**30 // current)
        while True:
            scaled = [round(Fraction(c[i] * gain, 2**15)) for i in indices]
            if sum(v * v for v in scaled) <= budget:
                break
            gain -= 1
        for i, value in zip(indices, scaled):
            c[i] = value

    def fine():
        ratio = (p.fineclamp_cap_ratio_q15 if traction == 32767 else
                 32768 - round(Fraction((32768 - p.fineclamp_cap_ratio_q15) * traction, 32768)))
        cap(range(60 * slots, len(c)), min(p.fine_energy_max,
            power(range(60 * slots)) * ratio // 32768))

    def hard():
        for leaf in range(64):
            cap(range(leaf * slots, (leaf + 1) * slots), p.leaf_energy_max)
        fine()
        cap(range(len(c)), p.global_energy_max)
        fine()

    hard()
    if int(cfg.tail_mode) in (0, 1):
        weights = {'flat': [1] * 6, 'pink': [3600, 900, 400, 225, 144, 100],
                   'bright': [25, 36, 49, 64, 81, 100]}[p.normalize_curve]
        ends = [0, 2, 6, 14, 26, 42, 64]
        for band, weight in enumerate(weights):
            indices = range(ends[band] * slots, ends[band + 1] * slots)
            original = [c[i] for i in indices]
            cap(indices, p.global_energy_max * weight // sum(weights))
            if traction != 32767:
                for i, before in zip(indices, original):
                    c[i] = before + round(Fraction((c[i] - before) * traction, 32768))
    if int(cfg.tail_mode) == 0:
        for left, right in p.sheaf_pairs:
            for k in range(slots):
                a, b = left * slots + k, right * slots + k
                avg = round(Fraction(c[a] + c[b], 2))
                for i in (a, b):
                    c[i] = (avg if traction == 32767 else
                            c[i] + round(Fraction((avg - c[i]) * traction, 32768)))
    if int(cfg.tail_mode) == 3:
        c[60 * slots:] = [0] * (4 * slots)
    if p.slew_max_delta is not None:
        c = [max(int(old) - p.slew_max_delta, min(int(old) + p.slew_max_delta, v))
             for v, old in zip(c, state.tail_previous_coeffs)]
    hard()
    if int(cfg.tail_mode) == 3:
        c[60 * slots:] = [0] * (4 * slots)
    state.coeff_state[:] = c
    state.tail_previous_coeffs[:] = c


def run_tail(values, *, mode=TailMode.FULL, policy=None, traction=32767, mask=0):
    s = VMState()
    s.coeff_state = np.array(values, dtype=np.int16)
    s.tail_previous_coeffs = np.zeros_like(s.coeff_state)
    cfg = DecodedPlan(tail_mode=mode, tail_policy=policy, traction_q15=traction,
                      mask_u32=mask)
    lanes.lane_tail(s, cfg)
    return s, cfg


@pytest.mark.parametrize('slots', [4, 8])
@pytest.mark.parametrize('mode', list(TailMode))
@pytest.mark.parametrize('curve', ['flat', 'pink', 'bright'])
@pytest.mark.parametrize('traction', [0, 16384, 32767])
def test_tail_matches_independent_scalar_spec(slots, mode, curve, traction):
    source = np.random.default_rng(103).integers(-32768, 32768, 64 * slots, dtype=np.int16)
    policy = CoefficientTailPolicy(normalize_curve=curve, sheaf_pairs=((0, 2), (12, 61)))
    actual, cfg = run_tail(source, mode=mode, policy=policy, traction=traction)
    expected = VMState()
    expected.coeff_state = source.copy()
    expected.tail_previous_coeffs = np.zeros_like(source)
    reference_tail(expected, cfg)
    np.testing.assert_array_equal(actual.coeff_state, expected.coeff_state)
    # Hard safety is independent of the operator mask, including the empty mask.
    v = actual.coeff_state.astype(np.int64).reshape(64, slots)
    assert np.all(np.sum(v * v, axis=1) <= policy.leaf_energy_max)
    assert np.sum(v * v) <= policy.global_energy_max
    assert np.sum(v[60:] * v[60:]) <= policy.fine_energy_max


def test_projection_rounding_and_energy_near_integer_boundary():
    # Naively rounding sqrt(budget/E) can overshoot the integer energy cap.
    for budget in range(29):
        values = np.arange(-3, 4, dtype=np.int16)
        limit_energy(values, budget)
        assert sum(int(v) ** 2 for v in values) <= budget
        again = values.copy()
        limit_energy(again, budget)
        np.testing.assert_array_equal(again, values)
    for gain in [0, 1, 8192, 16384, 32767, 32768]:
        source = np.array([-32768, -32767, -3, -1, 0, 1, 3, 32767], dtype=np.int16)
        expected = [round(Fraction(int(v) * gain, 32768)) for v in source]
        np.testing.assert_array_equal(scale_rne(source, gain), expected)


@pytest.mark.parametrize('mode', list(TailMode))
def test_explicit_stricter_caps_survive_slew_normalization_and_sheaf(mode):
    policy = CoefficientTailPolicy(leaf_energy_max=0x1000000,
        fine_energy_max=0x800000, global_energy_max=0x2000000,
        fineclamp_cap_ratio_q15=100, slew_max_delta=10,
        sheaf_pairs=((0, 1), (59, 60)))
    state = VMState()
    state.coeff_state[:] = np.random.default_rng(21).integers(-32768, 32768, 256)
    state.tail_previous_coeffs.fill(20000)  # History violates the new caps.
    expected = VMState.restore(state.snapshot())
    cfg = DecodedPlan(tail_mode=mode, tail_policy=policy)
    lanes.lane_tail(state, cfg)
    reference_tail(expected, cfg)
    np.testing.assert_array_equal(state.coeff_state, expected.coeff_state)
    wide = state.coeff_state.astype(np.int64).reshape(64, 4)
    assert np.max(np.sum(wide**2, axis=1)) <= policy.leaf_energy_max
    assert np.sum(wide**2) <= policy.global_energy_max
    assert np.sum(wide[60:]**2) <= policy.fine_energy_max
    assert np.sum(wide[60:]**2) <= int(np.sum(wide[:60]**2)) * 100 // 32768


def test_legal_low_energy_and_silence_are_bit_exact_identity():
    source = np.zeros(256, dtype=np.int16)
    source[:224] = np.arange(-112, 112)
    for mode in TailMode:
        actual, _ = run_tail(source, mode=mode)
        np.testing.assert_array_equal(actual.coeff_state, source)
        silent, _ = run_tail(np.zeros(256, dtype=np.int16), mode=mode)
        assert not np.any(silent.coeff_state)


def test_sheaf_pairs_preserve_temporal_slots_and_have_no_implicit_boundaries():
    source = np.zeros(256, dtype=np.int16)
    source[:8] = [10, 100, 1000, -1000, 20, 202, 2002, -2002]
    full, _ = run_tail(source, policy=CoefficientTailPolicy(sheaf_pairs=((0, 1),)))
    np.testing.assert_array_equal(full.coeff_state[:8], [15, 151, 1501, -1501] * 2)
    for mode in [TailMode.FULL, TailMode.MID, TailMode.MIN]:
        unpaired, _ = run_tail(source, mode=mode)
        np.testing.assert_array_equal(unpaired.coeff_state, source)


def test_curve_cap_and_traction_have_distinct_effects_without_energy_creation():
    source = np.full(256, 32767, dtype=np.int16)
    results = []
    for curve in ['flat', 'pink', 'bright']:
        state, _ = run_tail(source, policy=CoefficientTailPolicy(normalize_curve=curve))
        results.append(state.coeff_state.copy())
    assert all(not np.array_equal(results[a], results[b]) for a, b in [(0, 1), (1, 2), (0, 2)])
    dry, _ = run_tail(source, traction=0)
    assert not np.array_equal(dry.coeff_state, results[1])
    for cap in [0, 8192, 16384]:
        state, _ = run_tail(source, mode=TailMode.MIN,
                            policy=CoefficientTailPolicy(fineclamp_cap_ratio_q15=cap))
        v = state.coeff_state.astype(np.int64)
        assert np.sum(v[240:]**2) <= (int(np.sum(v[:240]**2)) * cap) // 32768


def test_runtime_uses_no_float_sqrt_or_random_draw(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('float/random operation in tail')
    monkeypatch.setattr(math, 'sqrt', forbidden)
    monkeypatch.setattr(np, 'sqrt', forbidden)
    monkeypatch.setattr(np.random, 'random', forbidden)
    for mode in TailMode:
        state, _ = run_tail(np.full(256, -32768, dtype=np.int16), mode=mode)
        assert state.prng_state == 0x12345678


def test_slew_is_per_voice_replayable_and_resets_on_note_on():
    state = VMState()
    cfg = DecodedPlan(tail_policy=CoefficientTailPolicy(slew_max_delta=32))
    for sign in [1, 1, -1, -1]:
        previous = state.tail_previous_coeffs.copy()
        state.coeff_state.fill(0)
        state.coeff_state[0] = sign * 1000
        lanes.lane_tail(state, cfg)
        assert abs(int(state.coeff_state[0]) - int(previous[0])) <= 32
    restored = VMState.restore(state.snapshot())
    for s in [state, restored]:
        s.coeff_state[0] = 5000
        lanes.lane_tail(s, cfg)
    np.testing.assert_array_equal(state.coeff_state, restored.coeff_state)
    state.reset_for_note_on()
    assert not np.any(state.tail_previous_coeffs)
    assert np.any(restored.tail_previous_coeffs)
    older = restored.snapshot()
    del older['tail_previous_coeffs']
    np.testing.assert_array_equal(VMState.restore(older).tail_previous_coeffs, restored.coeff_state)


def test_tightened_safety_and_emergency_override_incompatible_slew():
    s = VMState()
    s.coeff_state.fill(1000)
    s.tail_previous_coeffs.fill(1000)
    cfg = DecodedPlan(tail_mode=TailMode.EMERGENCY,
                      tail_policy=CoefficientTailPolicy(slew_max_delta=0, global_energy_max=0))
    lanes.lane_tail(s, cfg)
    assert not np.any(s.coeff_state)
    assert not np.any(s.tail_previous_coeffs)


def test_mode_latch_is_used_on_same_block_and_persists(monkeypatch):
    observed = []
    actual_tail = executor.lane_tail

    def record(state, cfg):
        observed.append((cfg.tail_mode, cfg.traction_q15, cfg.tail_ops))
        actual_tail(state, cfg)
    monkeypatch.setattr(executor, 'lane_tail', record)
    state = VMState()
    for frame in [VoiceFrame(cmmm_tail_mode=3, cmmm_traction_q15=9000), VoiceFrame()]:
        executor.vm_step_block(state, frame)
    assert observed == [(TailMode.EMERGENCY, 9000, [2, 19])] * 2
    legacy = VoiceFrame()
    legacy.tail_mode = 1
    executor.vm_step_block(state, legacy)
    assert observed[-1] == (TailMode.MID, 9000, [2, 11])
    assert state.executor_latch.tail_mode == 3


def test_emergency_zeroes_only_finest_four_leaves():
    source = np.full(256, 100, dtype=np.int16)
    s, _ = run_tail(source, mode=TailMode.EMERGENCY)
    np.testing.assert_array_equal(s.coeff_state[:240], source[:240])
    assert not np.any(s.coeff_state[240:])


def test_regime_policy_reaches_rendered_frame_and_changes_audio():
    from symcrash_python_ap.voice import VoiceEngine, Regime, RegimeSpec
    e = VoiceEngine()
    e.allocate_voice()
    e.note_on(0)
    from copy import deepcopy
    e.voices[0].program = deepcopy(e.voices[0].program)
    e.voices[0].program.regime_specs[Regime.R1_ATTACK] = RegimeSpec(
        regime=Regime.R1_ATTACK, normalize_curve='bright', fineclamp_cap_ratio=0.25,
        slew_max_delta=123)
    abi = e.step_voice(0)
    frame = e._abi_to_voice_frame(abi, 0)
    assert frame.tail_policy == CoefficientTailPolicy('bright', 8192, 123)
    assert compile_tail_policy('bright', 0.25, 123) is frame.tail_policy
    limited, unbounded = VMState(), VMState()
    audio_limited, audio_unbounded = [], []
    for _ in range(8):
        audio_limited.extend(executor.vm_step_block(limited, frame)[1])
        audio_unbounded.extend(executor.vm_step_block(unbounded, replace(frame, tail_policy=None))[1])
    assert np.any(audio_limited)
    assert not np.array_equal(audio_limited, audio_unbounded)


def test_omitted_regime_tail_fields_inherit_program_and_explicit_zero_wins():
    from copy import deepcopy
    from symcrash_python_ap.voice import VoiceEngine, Regime, RegimeSpec
    e = VoiceEngine()
    e.allocate_voice()
    e.note_on(0)
    program = deepcopy(e.voices[0].program)
    e.voices[0].program = program
    program.normalize_curve = 'flat'
    program.fineclamp_cap_ratio = 0.75
    program.slew_max_delta = 100
    program.regime_specs[Regime.R1_ATTACK] = RegimeSpec(regime=Regime.R1_ATTACK)
    frame = e._abi_to_voice_frame(e.step_voice(0), 0)
    assert frame.tail_policy == CoefficientTailPolicy('flat', 24576, 100)
    program.regime_specs[Regime.R1_ATTACK].fineclamp_cap_ratio = 0.0
    program.regime_specs[Regime.R1_ATTACK].slew_max_delta = 0
    frame = e._abi_to_voice_frame(e.step_voice(0), 0)
    assert frame.tail_policy == CoefficientTailPolicy('flat', 0, 0)


@pytest.mark.parametrize('fields', [
    {'normalize_curve': 'unknown'}, {'fineclamp_cap_ratio_q15': -1},
    {'fineclamp_cap_ratio_q15': 32768}, {'fineclamp_cap_ratio_q15': 0.5},
    {'slew_max_delta': -1}, {'slew_max_delta': 65536}, {'global_energy_max': -1},
    {'leaf_energy_max': 2**63}, {'sheaf_pairs': [(0, 1)]},
    {'sheaf_pairs': ((0, 1), (1, 2))}, {'sheaf_pairs': ((0, 64),)},
])
def test_invalid_runtime_policies_rejected(fields):
    with pytest.raises(ValueError):
        CoefficientTailPolicy(**fields)


@pytest.mark.parametrize('ratio', [-0.1, 1.1, float('nan'), float('inf')])
def test_invalid_authoring_cap_rejected(ratio):
    with pytest.raises(ValueError):
        compile_tail_policy('pink', ratio, 512)


@pytest.mark.parametrize('mode, traction', [(4, 32767), (-1, 32767), (1, -1), (1, 32768)])
def test_invalid_tail_event_does_not_mutate_latch(mode, traction):
    latch = VMState().executor_latch
    with pytest.raises(ValueError):
        latch.apply_tailmode_event(mode, traction)
    assert (latch.tail_mode, latch.traction_q15) == (0, 32767)
