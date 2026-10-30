"""Equal-tempered integer pitch: independent arithmetic and real VM control."""
import ast
from dataclasses import replace
from decimal import Decimal, localcontext, ROUND_HALF_EVEN
from fractions import Fraction
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from symcrash_python_ap import primitives as p, fixed
from symcrash_python_ap.pitch_rom import MIDI_CENT_PHASE_Q32, MIDI_CENT_FREQ_MHZ
from symcrash_python_ap.vm import VMState, vm_step_block, harmonic_bank as bank
from symcrash_python_ap.vm.decode import decode_frame
from symcrash_python_ap.vm.basis_resource import prepare_basis_resources
from symcrash_python_ap.voice.voice_state_codec import vmstate_to_bytes, vmstate_from_bytes
from tests.conformance.test_carrier_spectrum import frame
from tests.conformance.test_vm_snapshot_replay import execution_path


def oracle(note, cents=0, scale=None):
    # Independently expressed in octave/semitone units, not production indexing.
    note = max(0, min(127, int(note)))
    semitone = max(Decimal(0), min(Decimal(127), Decimal(note) + Decimal(int(cents))/100))
    with localcontext() as ctx:
        ctx.prec = 90
        ctx.rounding = ROUND_HALF_EVEN
        frequency = Decimal(440) * ((semitone - 69) * Decimal(2).ln() / 12).exp()
        return round(frequency * 2**32 / 48000 if scale is None else frequency * scale)


def test_every_cent_is_single_quantization_of_the_equal_tempered_frequency():
    # Independent logarithm/exponential formulation at a different precision
    # from the generator's decimal power recipe. No runtime pitch table oracle.
    with localcontext() as ctx:
        ctx.prec = 90
        ctx.rounding = ROUND_HALF_EVEN
        ln2 = Decimal(2).ln()
        for cent in range(12701):
            frequency = Decimal(440) * ((Decimal(cent)/100 - 69) * ln2 / 12).exp()
            phase = p.midi_to_phase_inc_fixed(0, cent)
            mhz = p.midi_to_freq_mhz(0, cent)
            assert phase == round(frequency * 2**32 / 48000), cent
            assert mhz == round(frequency * 1000), cent
            assert abs(Decimal(phase) * 48000 / 2**32 / frequency - 1) < Decimal('0.0000007')
            assert abs(Decimal(mhz)/1000/frequency - 1) < Decimal('0.000062')
    assert type(MIDI_CENT_PHASE_Q32) is type(MIDI_CENT_FREQ_MHZ) is tuple
    assert len(MIDI_CENT_PHASE_Q32) == len(MIDI_CENT_FREQ_MHZ) == 12701
    assert all(a < b for a,b in zip(MIDI_CENT_PHASE_Q32, MIDI_CENT_PHASE_Q32[1:]))
    assert all(a < b for a,b in zip(MIDI_CENT_FREQ_MHZ, MIDI_CENT_FREQ_MHZ[1:]))


@pytest.mark.parametrize('note', [0, 1, 2, 12, 60, 69, 115, 126, 127])
def test_equivalent_spellings_and_semitone_boundaries_are_monotonic(note):
    before = [0, 0]
    for cents in range(-1300, 1301):
        values = [p.midi_to_phase_inc_fixed(note,cents), p.midi_to_freq_mhz(note,cents)]
        assert all(a <= b for a,b in zip(before,values)), (note,cents)
        for n in (0, 60, 127):
            offset = (note - n)*100 + cents
            assert p.midi_to_phase_inc_fixed(n,offset) == values[0]
            assert p.midi_to_freq_mhz(n,offset) == values[1]
        before = values


@pytest.mark.parametrize('note,cents', [(69,-101),(69,-199),(2,-101),(0,-1),(127,1),
                                     (0,-10**100),(127,10**100),(-12,37),(200,-37)])
def test_regressions_and_endpoint_saturation(note, cents):
    assert p.midi_to_phase_inc_fixed(note,cents) == oracle(note,cents)
    assert p.midi_to_freq_mhz(note,cents) == oracle(note,cents,Decimal(1000))
    assert p.midi_to_freq_hz(note,cents) == p.midi_to_freq_mhz(note,cents)/1000


@pytest.mark.parametrize('scalar', [np.int16,np.int32,np.int64,np.uint16,np.uint64])
@pytest.mark.filterwarnings('error')
def test_numpy_integer_inputs_widen_before_arithmetic(scalar):
    assert p.midi_to_phase_inc_fixed(scalar(127),scalar(32767)) == p.midi_to_phase_inc_fixed(127)
    assert p.midi_to_freq_mhz(scalar(127),scalar(32767)) == p.midi_to_freq_mhz(127)


@pytest.mark.parametrize('bad', [True,False,np.bool_(True),1.5,'60',None,float('nan')])
@pytest.mark.parametrize('field', ['note','cents'])
def test_invalid_native_pitch_fails_before_musical_state_changes(bad, field):
    state = VMState.create(0)
    before = state.snapshot()
    f = frame(**{field:bad})
    for fn in (p.midi_to_phase_inc_fixed,p.midi_to_freq_mhz,p.midi_to_freq_hz):
        with pytest.raises(ValueError,match='integers'):
            fn(f.note,f.cents)
    with pytest.raises(ValueError,match='integers'):
        vm_step_block(state,f)
    assert state.snapshot() == before


def test_nominal_compatibility_loaders_are_detached_and_display_units_agree():
    a,b=p._build_pitch_table(),p._build_freq_table_mhz()
    a[69]=b[69]=0
    assert p.midi_to_phase_inc_fixed(69) == 39370534
    assert p.midi_to_freq_mhz(69) == 440000
    for note in range(128):
        assert p._PITCH_TABLE[note] == oracle(note)
        assert p._FREQ_TABLE_MHZ[note] == oracle(note,scale=Decimal(1000))
    for cents in range(-200,201):
        phase=p.midi_to_phase_inc_fixed(69,cents)
        freq=p.midi_to_freq_mhz(69,cents)
        assert abs(Fraction(phase*48000,2**32)-Fraction(freq,1000)) <= Fraction(1,2000)+Fraction(24000,2**32)


def test_generator_matches_committed_rom():
    root=Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable,str(root/'tools/generate_pitch_rom.py'),'--check'],check=True,capture_output=True)


def test_cold_import_and_pitch_use_frozen_integers_only():
    for fn in (p._pitch_cent_index,p._build_pitch_table,p._build_freq_table_mhz,
               p.midi_to_phase_inc_fixed,p.midi_to_freq_mhz):
        tree=ast.parse(inspect.getsource(fn))
        assert not any(isinstance(node,(ast.Pow,ast.Div)) for node in ast.walk(tree))
        assert not any(isinstance(node,ast.Constant) and type(node.value) is float for node in ast.walk(tree))
    code='''
import math, sys, numpy, numpy.random, numba
assert 'symcrash_python_ap' not in sys.modules
def forbidden(*a,**k): raise AssertionError('runtime transcendental')
math.pow=math.exp=math.exp2=math.log=math.log2=forbidden
from symcrash_python_ap import primitives as p
assert p.midi_to_phase_inc_fixed(69,-101)==p.midi_to_phase_inc_fixed(67,99)
assert p.midi_to_freq_mhz(69,-101)==p.midi_to_freq_mhz(67,99)
assert p.midi_to_freq_mhz(69)==440000
'''
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('note,cents',[(48,-101),(60,-199),(69,-101),(69,-99),(69,99),(108,99)])
def test_rendered_mode_b_frequency_follows_detuned_equal_temperament(note,cents):
    from tests.test_ccs_frequency import measure_fundamental_fft
    from symcrash_python_ap.vm import hotpath_jit
    old=hotpath_jit.jit_enabled()
    hotpath_jit.set_jit_enabled(True)
    try:
        state=VMState.create(0)
        f=frame(note=note,cents=cents,waveform_type=0,H_request=1)
        f.backbone_ops=[]
        audio=np.concatenate([vm_step_block(state,f)[1].copy() for _ in range(758)])[512:]
        frequency=measure_fundamental_fft(audio.astype(np.float64))
        expected=440*2**((note-69+cents/100)/12)
        assert abs(frequency/expected-1)<0.001, (note,cents,frequency,expected)
    finally:
        hotpath_jit.set_jit_enabled(old)


COMMANDS=[(69,c) for c in (-201,-200,-199,-150,-101,-100,-99,-1,0,1,99,100,101)]
COMMANDS += [(2,-101),(0,-99),(127,99),(60,2400),(60,-2400)]


@pytest.mark.parametrize('mode',[0,1,3])
def test_actual_vm_pitch_feed_audio_and_checkpoint_replay(mode,execution_path,monkeypatch):
    prepare_basis_resources()
    frames=[frame(coeff_osc_mode=mode,note=n,cents=c,legato_mode=True,is_starting=i==0)
            for i,(n,c) in enumerate(COMMANDS)]
    independent=VMState.create(0)
    with monkeypatch.context() as patch:
        patch.setattr(fixed,'midi_to_phase_inc_fixed',lambda n,c=0:oracle(n,c))
        expected=[vm_step_block(independent,f)[1].copy() for f in frames]
    state=VMState.create(0)
    saved=None
    for i,f in enumerate(frames):
        assert decode_frame(f).coeff_osc_params.phase_inc == oracle(f.note,f.cents)
        bank.clear_basis_atlas()
        np.testing.assert_array_equal(vm_step_block(state,f)[1],expected[i])
        if i==8:
            saved=[VMState.restore(json.loads(json.dumps(state.snapshot()))),
                   vmstate_from_bytes(vmstate_to_bytes(state))]
    for restored in saved:
        for f,audio in zip(frames[9:],expected[9:]):
            bank.clear_basis_atlas()
            np.testing.assert_array_equal(vm_step_block(restored,f)[1],audio)
    assert any(np.any(a) for a in expected)


@pytest.mark.parametrize('components',[(-99,-50,-52),(-200,100,-1),(0,-2400,200),(32767,32767,32767)])
def test_wire_pitch_fields_sum_before_clamping_and_reach_mode_b(components,execution_path):
    from tests.conformance.test_scheduled_vm import packet
    cents,detune,bend=components
    wire=packet(note_u8=69,cents_i16=cents,detune_cents_i16=detune,
                pitchbend_cents_i16=bend,carrier_model_u8=1)
    f=wire.to_voice_frame()
    inc=oracle(69,cents+detune+bend)
    assert decode_frame(f).coeff_osc_params.phase_inc == inc
    state=VMState.create(0)
    vm_step_block(state,f)
    assert state.mode_b_state.phase_inc == inc
    assert f.f0_hz == p.midi_to_freq_hz(69,cents+detune+bend)
