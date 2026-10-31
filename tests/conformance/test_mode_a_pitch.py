"""Accepted Mode A pitch must select matching tables and phase progression."""
from dataclasses import replace
from fractions import Fraction
import json

import numpy as np
import pytest

from symcrash_python_ap.fixed import midi_to_phase_inc_fixed
from symcrash_python_ap.pitch_rom import MIDI_CENT_PHASE_Q32
from symcrash_python_ap.vm import VMState, vm_step_block, coeff_lut_streaming as lut
from symcrash_python_ap.vm import mode_a_resource as resource
from symcrash_python_ap.vm.scheduling import ExecutionPolicy
from symcrash_python_ap.voice.voice_state_codec import vmstate_to_bytes, vmstate_from_bytes
from tests.conformance.test_carrier_spectrum import frame
from tests.conformance.test_mode_a_resource import reference_table, prohibit_preparation
from tests.conformance.test_pitch_conversion import oracle
from tests.conformance.test_vm_snapshot_replay import execution_path


@pytest.fixture(scope='module', autouse=True)
def atlas():
    return resource.prepare_mode_a_resources()


def command(**kw):
    f = frame(coeff_osc_mode=0, gain_q15=24000, **kw)
    f.backbone_ops = []
    return f


@pytest.mark.parametrize('wave,note,cents', [(0,69,1200), (0,69,-199),
    (1,96,37), (2,108,-99), (3,127,-1), (4,80,99)])
def test_detuned_frames_match_independent_q32_generation(wave, note, cents):
    expected = reference_table(wave, note, cents)
    actual = lut.get_lut_for_note(wave, note, cents)
    np.testing.assert_array_equal(actual.data, expected.data)
    assert actual.name == expected.name
    assert actual is lut.get_lut_for_pitch(wave, oracle(note, cents))


def test_every_cent_has_a_prepared_table_for_each_wave(atlas, monkeypatch):
    prohibit_preparation(monkeypatch)
    assert resource.PITCH_COUNT == 12701
    for wave in range(5):
        for cent, phase in enumerate(MIDI_CENT_PHASE_Q32):
            table = lut.get_lut_for_note(wave, 0, cent)
            assert table is lut.get_lut_for_pitch(wave, phase)
            assert resource.phase_address(wave, phase)[2] == wave*12701+cent
            assert table.data.shape == (64, 4, 64)
            assert not table.data.flags.writeable
    assert len(lut._NOTE_LUT_CACHE) == 128


@pytest.mark.parametrize('increment', [0, -1, 2**32, True, np.bool_(True),
                                      39370533, 39370534.0, None])
def test_unprepared_q32_values_are_rejected_without_retuning(increment):
    with pytest.raises(ValueError):
        lut.get_lut_for_pitch(0, increment)


@pytest.mark.parametrize('note,cents,canonical', [(69,1200,81), (69,-1200,57),
    (69,100,70), (60,2400,84), (60,-2400,36), (0,-999,0), (127,999,127)])
@pytest.mark.parametrize('wave', range(5))
def test_equivalent_pitches_have_identical_audio(note, cents, canonical, wave, execution_path):
    a, b = VMState.create(0), VMState.create(0)
    f = command(note=note, cents=cents, coeff_table_id=wave)
    canonical_frame = replace(f, note=canonical, cents=0)
    for _ in range(12):
        np.testing.assert_array_equal(vm_step_block(a, f)[1], vm_step_block(b, canonical_frame)[1])
        assert a.ccs_state.phase == b.ccs_state.phase
        assert a.ccs_state.phase_inc == b.ccs_state.phase_inc == oracle(note,cents)


def test_accepted_pitch_advances_immediately_without_hidden_glide(execution_path):
    state = VMState.create(0)
    phase = 0
    for i, cents in enumerate([0, 100, -100, 37, 1200, -1200, 1, -1, 0]):
        inc = oracle(69, cents)
        f = command(note=69, cents=cents, legato_mode=True)
        vm_step_block(state, f)
        phase = (phase+64*inc) % 2**32
        assert state.ccs_state.phase == phase
        assert state.ccs_state.phase_inc == state.ccs_state.target_phase_inc == inc
        assert state.ccs_state.phase_slew_rate == 0


def test_snapshot_with_old_slew_is_corrected_only_when_serviced(execution_path):
    state = VMState.create(0)
    vm_step_block(state, command())
    state.ccs_state.phase_inc = 39370000
    state.ccs_state.target_phase_inc = oracle(69, 100)
    state.ccs_state.phase_slew_rate = 100
    states = [VMState.restore(json.loads(json.dumps(state.snapshot()))),
              vmstate_from_bytes(vmstate_to_bytes(state))]
    for restored in states:
        before = restored.snapshot()
        vm_step_block(restored, command(cents=100), policy=ExecutionPolicy(serviced=False))
        assert restored.snapshot() == before
        vm_step_block(restored, command(cents=100))
        assert restored.ccs_state.phase_inc == oracle(69, 100)
        assert restored.ccs_state.phase_slew_rate == 0


@pytest.mark.parametrize('mode', [lut.INTERP_LINEAR, lut.INTERP_CUBIC])
@pytest.mark.parametrize('fraction', [0, 1, 2**24, 2**25, 3*2**24, 2**26-1])
def test_interpolation_uses_signed_even_ties_and_wide_products(mode, fraction):
    rng = np.random.default_rng(3701)
    data = rng.integers(-32768, 32768, (64,4,64), dtype=np.int16)
    # Include half-way cases where even parity belongs to the complete value.
    data[0,:4,0], data[0,:4,1] = [0,1,-2,-1], [1,2,-1,0]
    table = lut.StreamingCoeffLUT(0, 'rounding oracle', data)
    for p in (0,63):
        output = np.empty((64,4), dtype=np.int16)
        table.get_subband_samples((p<<26)+fraction, output, interp_mode=mode)
        expected = np.empty_like(output)
        for leaf in range(64):
            for k in range(4):
                a,b,c,d = [int(data[leaf,k,(p+j)%64]) for j in (-1,0,1,2)]
                t = Fraction(fraction, 2**26)
                if mode == lut.INTERP_LINEAR:
                    value = round(b*(1-t)+c*t)
                else:
                    value = round((-a+3*b-3*c+d)*t)
                    value = round((value+2*a-5*b+4*c-d)*t)
                    value = round((value-a+c)*t)
                    value = round(Fraction(value+2*b, 2))
                expected[leaf,k] = max(-32768,min(32767,value))
        np.testing.assert_array_equal(output, expected)


def steady_audio(note, cents, wave=0):
    state = VMState.create(0)
    f = command(note=note, cents=cents, coeff_table_id=wave)
    return np.concatenate([vm_step_block(state,f)[1].copy() for _ in range(758)])[512:].astype(np.float64)


@pytest.mark.parametrize('note,cents', [(69,1200),(69,-1200),(69,2400),
                                      (69,-199),(69,99),(108,99)])
def test_actual_sine_pitch_and_block_continuity(note,cents):
    from tests.test_ccs_frequency import measure_fundamental_fft
    samples = steady_audio(note,cents)
    expected = oracle(note,cents)*48000/2**32
    measured = measure_fundamental_fft(samples)
    assert abs(measured/expected-1) < .001, (measured,expected)
    differences = np.abs(np.diff(samples))
    boundary = np.arange(len(differences)) % 64 == 63
    assert differences[boundary].mean() / differences[~boundary].mean() < 3.0


@pytest.mark.parametrize('cents', [-1200,0,1200])
def test_actual_sine_spurs_are_below_recommended_threshold(cents):
    samples = steady_audio(69,cents)
    frequency = oracle(69,cents)*48000/2**32
    phase = 2*np.pi*frequency*np.arange(len(samples))/48000
    basis = np.column_stack([np.sin(phase), np.cos(phase), np.ones(len(samples))])
    fitted = basis @ np.linalg.lstsq(basis, samples, rcond=None)[0]
    window = np.hanning(len(samples))
    residual_peak = np.abs(np.fft.rfft((samples-fitted)*window))[1:].max()
    fundamental_peak = np.abs(np.fft.rfft(fitted*window))[1:].max()
    assert 20*np.log10(residual_peak/fundamental_peak) < -60


def test_high_square_excludes_folded_seventh_harmonic():
    samples = steady_audio(108,0,2)
    frequency = oracle(108,0)*48000/2**32
    # At this pitch the physical 0.9-Nyquist cap is H=5; H=7 would fold.
    phase = 2*np.pi*frequency*np.arange(len(samples))/48000
    columns = [np.ones(len(samples))]
    for h in (1,2,3,4,5,7):
        columns += [np.sin(h*phase), np.cos(h*phase)]
    weights = np.linalg.lstsq(np.column_stack(columns), samples, rcond=None)[0]
    levels = np.hypot(weights[1::2],weights[2::2])
    assert levels[2]/levels[0] == pytest.approx(1/3, rel=.03)
    assert levels[4]/levels[0] == pytest.approx(1/5, rel=.03)
    assert max(levels[1],levels[3],levels[5])/levels[0] < .002
