"""Fixed anchor arithmetic, streaming execution and measured interpolation quality."""
from fractions import Fraction
from functools import lru_cache
from unittest.mock import patch

import numpy as np
import pytest

from symcrash_python_ap.fixed import midi_to_phase_inc_fixed
from symcrash_python_ap.pitch_rom import MIDI_CENT_PHASE_Q32
from symcrash_python_ap.vm import VMState, vm_step_block, harmonic_bank as bank, hotpath_jit
from symcrash_python_ap.vm.basis_anchors import (
    ANCHORS, InterpolatedBasis, weights_for_frequency, frequency_stencil, interpolate_arrays,
)
from symcrash_python_ap.vm.basis_resource import prepare_basis_resources
from tests.conformance.test_carrier_spectrum import frame


def test_complete_pitch_domain_keeps_weights_bounded_and_unity_exact(monkeypatch):
    from symcrash_python_ap.vm import basis_anchors
    atlas = prepare_basis_resources()
    original_divide=basis_anchors._divide_even
    def bounded_divide(numerator,denominator):
        assert -(1<<63) <= numerator < (1<<63)
        assert 0 < abs(denominator) < (1<<32)
        return original_divide(numerator,denominator)
    monkeypatch.setattr(basis_anchors,'_divide_even',bounded_divide)
    original_jit = hotpath_jit.jit_enabled()
    try:
        for cent,pitch in enumerate(MIDI_CENT_PHASE_Q32):
            count=min(32,9*2**31//(10*pitch))
            hotpath_jit.set_jit_enabled(False)
            basis=atlas.basis(cent//100,pitch,count)
            hotpath_jit.set_jit_enabled(True)
            accelerated=atlas.basis(cent//100,pitch,count)
            if cent % 100 == 0:
                assert not isinstance(basis,InterpolatedBasis)
            else:
                np.testing.assert_array_equal(accelerated.source_offsets,basis.source_offsets)
                np.testing.assert_array_equal(accelerated.interpolation_weights,basis.interpolation_weights)
                assert np.all(basis.interpolation_weights.astype(np.int64).sum(axis=1)==2**30)
                assert np.max(np.abs(basis.interpolation_weights.astype(np.int64))) < 2**31
                assert basis.source_offsets.nbytes+basis.interpolation_weights.nbytes <= 1536
    finally:
        hotpath_jit.set_jit_enabled(original_jit)


def test_oversized_basis_recipes_fail_before_creating_output(tmp_path):
    from symcrash_python_ap.vm.basis_resource_builder import build_basis_resource
    from symcrash_python_ap.vm.basis_resource import PreparedBasisAtlas
    # 128 full 32-harmonic anchors plus their frequency directory exceed
    # the 4 MiB allocation. The canonical bandlimited recipe is smaller.
    keys = tuple(range(1,129))
    target = tmp_path/'too_large.bin'
    with pytest.raises(ValueError,match='budget'):
        build_basis_resource(target,pitch_increments=keys)
    with pytest.raises(ValueError,match='budget'):
        PreparedBasisAtlas(target,expected_keys=keys)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('cent', [1,37,99,100,6150,6950,12599,12650,12699])
def test_weight_quantization_tracks_rational_cubic_polynomials(cent):
    atlas=prepare_basis_resources()
    target = MIDI_CENT_PHASE_Q32[cent]
    stencil=frequency_stencil(target,atlas._grid_frequencies)
    nodes=tuple(atlas._grid_frequencies[i] for i in stencil)
    weights = weights_for_frequency(target,nodes)
    for power in range(4):
        normalized = [Fraction(p-nodes[1],nodes[2]-nodes[1])**power for p in nodes]
        actual = sum(Fraction(w,2**30)*y for w,y in zip(weights,normalized))
        expected = Fraction(target-nodes[1],nodes[2]-nodes[1])**power
        assert abs(actual-expected) < Fraction(1,2**25)


@pytest.mark.parametrize('cent', [1,50,6950,12650])
def test_signed_interpolation_matches_exact_fraction_and_saturates_once(cent):
    rng = np.random.default_rng(40+cent)
    data = [rng.integers(-32768,32768,(32,64,4),dtype=np.int16) for _ in range(4)]
    atlas=prepare_basis_resources()
    target=MIDI_CENT_PHASE_Q32[cent]
    nodes=tuple(atlas._grid_frequencies[i] for i in frequency_stencil(target,atlas._grid_frequencies))
    weights = weights_for_frequency(target,nodes)
    expected = [max(-32768,min(32767,round(Fraction(sum(int(x)*w for x,w in zip(values,weights)),2**30))))
                for values in zip(*(x.reshape(-1) for x in data))]
    actual = interpolate_arrays(data,weights)
    np.testing.assert_array_equal(actual.reshape(-1),expected)
    assert not actual.flags.writeable


@pytest.mark.parametrize('cent', [1,37,50,99,6350,6950,9650,10899,12699])
def test_jit_streams_anchors_without_materializing_a_detuned_bank(cent, monkeypatch):
    if not hotpath_jit.NUMBA_AVAILABLE:
        pytest.skip('numba unavailable')
    state = VMState.create()
    command = frame(note=cent//100,cents=cent%100,H_request=32,legato_mode=True)
    prior = hotpath_jit.jit_enabled()
    try:
        hotpath_jit.set_jit_enabled(False)
        expected = vm_step_block(state,command)[1].copy()
        state = VMState.create()
        def forbidden(self):
            pytest.fail('Audio JIT materialized a full interpolated basis')
        monkeypatch.setattr(InterpolatedBasis,'Ucos_h',property(forbidden))
        monkeypatch.setattr(InterpolatedBasis,'Usin_h',property(forbidden))
        hotpath_jit.set_jit_enabled(True)
        np.testing.assert_array_equal(vm_step_block(state,command)[1],expected)
    finally:
        hotpath_jit.set_jit_enabled(prior)


@pytest.mark.parametrize('note,cents', [(48,37),(69,25),(69,50),(69,99)])
def test_midband_interpolated_sine_pitch_spurs_and_block_continuity(note,cents):
    from tests.test_ccs_frequency import measure_fundamental_fft
    state = VMState.create()
    command = frame(note=note,cents=cents,waveform_type=0,H_request=1,gain_q15=32767,legato_mode=True)
    command.backbone_ops=[]
    samples=np.concatenate([vm_step_block(state,command)[1].copy() for _ in range(758)])[512:].astype(np.float64)
    frequency=midi_to_phase_inc_fixed(note,cents)*48000/2**32
    assert abs(measure_fundamental_fft(samples)/frequency-1) < .001
    phase=2*np.pi*frequency*np.arange(len(samples))/48000
    fitting=np.column_stack([np.sin(phase),np.cos(phase),np.ones(len(samples))])
    fitted=fitting@np.linalg.lstsq(fitting,samples,rcond=None)[0]
    window=np.hanning(len(samples))
    residual=np.abs(np.fft.rfft((samples-fitted)*window))[1:].max()
    fundamental=np.abs(np.fft.rfft(fitted*window))[1:].max()
    assert 20*np.log10(residual/fundamental) < -60
    differences=np.abs(np.diff(samples))
    boundary=np.arange(len(differences))%64==63
    assert differences[boundary].mean()/differences[~boundary].mean() < 3


@lru_cache(maxsize=32)
def exact_basis(note,pitch,count):
    return bank.generate_basis_frames(note,count,phase_inc=pitch)


@pytest.mark.parametrize('note,cents', [(48,37),(60,50),(69,50),(84,50),(96,50),(108,50)])
@pytest.mark.parametrize('wave', [1,3,5], ids=['square','saw','triangle'])
def test_interpolation_preserves_rich_spectra_against_exact_q32_basis(note,cents,wave):
    """Whole-bank semitone interpolation loses up to 96% of upper partials.

    The frequency-selected atlas must stay within 5% for partials above
    -46 dBc and 1.5% relative waveform RMS on this low-to-high detuned set.
    """
    command=frame(note=note,cents=cents,waveform_type=wave,H_request=32,
                  gain_q15=32767,legato_mode=True)
    command.backbone_ops=[]
    def render():
        state=VMState.create()
        return np.concatenate([vm_step_block(state,command)[1].copy()
                               for _ in range(758)])[512:].astype(np.float64)
    actual=render()
    with patch.object(bank,'get_basis_for_pitch',exact_basis):
        expected=render()
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < .015
    increment=midi_to_phase_inc_fixed(note,cents)
    count=min(32,9*2**31//(10*increment))
    phase=2*np.pi*(increment/2**32)*np.arange(len(actual))
    columns=[np.ones(len(actual))]
    for h in range(1,count+1):
        columns.extend((np.sin(h*phase),np.cos(h*phase)))
    coefficients=np.linalg.lstsq(np.column_stack(columns),np.column_stack((actual,expected)),rcond=None)[0]
    levels=np.hypot(coefficients[1::2],coefficients[2::2])
    audible=levels[:,1] > .005*levels[0,1]
    assert np.max(np.abs(levels[audible,0]/levels[audible,1]-1)) < .05
